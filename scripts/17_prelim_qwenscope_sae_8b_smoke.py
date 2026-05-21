# Qwen 公式 Qwen-Scope SAE (residual stream SAE) の 8B 版 smoke test。
# - scripts/16_prelim_qwenscope_sae_smoke.py の 8B 対応版。1.7B 版とは別ファイルで残す。
# - MODEL_ID = Qwen/Qwen3-8B-Base, SAE_ID = Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50, LAYER_IDX = 24.
# - 8B は重いので peak memory を下げるため、次の順序で処理する:
#     1. SAE checkpoint は hf_hub_download で path だけ確保（まだ load しない）。
#     2. Qwen3-8B-Base model を load → clean / corrupt forward → residual を CPU へコピー。
#     3. model を del して gc.collect() + MPS/CUDA cache empty。
#     4. その後で SAE checkpoint を torch.load し、feature / diff / reconstruction を計算。
# - 巨大な tensor は保存しない。CSV / JSON / PNG のみ出力。
# - model / SAE checkpoint は Hugging Face cache 上に置く。snapshot_download は使わない。
#
# hidden_states インデックス対応:
#   hidden_states[0]     = embedding output
#   hidden_states[j + 1] = block j の output (= residual stream after block j)
#
#   LAYER_IDX = 24 → residual_source = hidden_states[25] = block 24 の output。
#   これは Qwen-Scope の residual-stream SAE に入力する表現である。
#   mwhanna/qwen3-4b-transcoders の MLP input とは違うので注意。
#
# 環境: llm2026-dev

from __future__ import annotations

import csv
import gc
import json
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import resolve_outputs_dir

# ── User-configurable settings (8B 版) ─────────────────────────────────────────
MODEL_ID  = "Qwen/Qwen3-8B-Base"
SAE_ID    = "Qwen/SAE-Res-Qwen3-8B-Base-W64K-L0_50"
LAYER_IDX = 24
TOP_K_SAE = 50  # L0_50 (TopK with k=50)

CLEAN_PROMPT   = "The capital of Japan is"
CORRUPT_PROMPT = "The capital of France is"
CLEAN_ANSWER   = " Tokyo"
CORRUPT_ANSWER = " Paris"

TOP_K_FEATURES = 20
MAX_FEATURES_FOR_HEATMAP = 60
TOP_K_DIFF = 20

OUT_PREFIX_PRELIM  = f"prelim_qwenscope_sae_qwen3_8b_layer{LAYER_IDX}"
OUT_PREFIX_NB      = f"nb03_qwenscope_sae_qwen3_8b_layer{LAYER_IDX}"

outputs_dir = resolve_outputs_dir()

# ── Device ─────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"model_id  : {MODEL_ID}")
print(f"sae_id    : {SAE_ID}")
print(f"layer_idx : {LAYER_IDX}")
print(f"top_k_sae : {TOP_K_SAE}")
print(f"device    : {device}")
print(f"dtype     : {dtype}")

# ── [1] Prepare SAE checkpoint path only (do NOT load yet) ─────────────────────
print("\n[1] Fetching SAE checkpoint path via hf_hub_download (load deferred)")
sae_filename = f"layer{LAYER_IDX}.sae.pt"
try:
    sae_path = hf_hub_download(repo_id=SAE_ID, filename=sae_filename)
except Exception as e:
    print(f"  [error] failed to fetch {sae_filename}: {e}")
    print("  hint: pip install -U huggingface_hub hf_xet")
    raise
sae_path_p = Path(sae_path)
sae_size = sae_path_p.stat().st_size
print(f"  sae file: {sae_path}")
print(f"  sae size: {sae_size:,} bytes ({sae_size / 1e6:.2f} MB)")

# ── [2] Tokenizer & token tables ───────────────────────────────────────────────
print(f"\n[2] Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def piece_repr(piece: str) -> str:
    return piece.replace(" ", "·")


clean_inputs   = tokenizer(CLEAN_PROMPT,   return_tensors="pt")
corrupt_inputs = tokenizer(CORRUPT_PROMPT, return_tensors="pt")
clean_seq_len   = clean_inputs["input_ids"].shape[1]
corrupt_seq_len = corrupt_inputs["input_ids"].shape[1]
clean_last_pos   = clean_seq_len   - 1
corrupt_last_pos = corrupt_seq_len - 1

prompt_tokens: dict[str, list[dict]] = {"clean": [], "corrupt": []}
for prompt_type, ids_tensor in [
    ("clean",   clean_inputs["input_ids"][0]),
    ("corrupt", corrupt_inputs["input_ids"][0]),
]:
    for i, tid in enumerate(ids_tensor.cpu().tolist()):
        raw_token = str(tokenizer.convert_ids_to_tokens([tid])[0])
        piece = tokenizer.decode([tid])
        prompt_tokens[prompt_type].append({
            "position":   i,
            "token_id":   tid,
            "raw_token":  raw_token,
            "piece":      piece,
            "piece_repr": piece_repr(piece),
        })
    print(f"  {prompt_type}: {[r['piece'] for r in prompt_tokens[prompt_type]]}")

# ── [3] Load model and run forward (then free immediately) ─────────────────────
print(f"\n[3] Loading model: {MODEL_ID}  (this is the heavy step on 8B)")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype)
model.to(device).eval()  # type: ignore[union-attr]
K = len(model.model.layers)
hidden_size = model.config.hidden_size
print(f"  num_decoder_layers = {K}")
print(f"  hidden_size        = {hidden_size}")
assert 0 <= LAYER_IDX < K, f"LAYER_IDX {LAYER_IDX} out of range [0,{K})"

# Move inputs to device after model is on device
clean_inputs   = {k: v.to(device) for k, v in clean_inputs.items()}
corrupt_inputs = {k: v.to(device) for k, v in corrupt_inputs.items()}

print(f"\n[4] Forward (output_hidden_states=True) and pick hidden_states[{LAYER_IDX + 1}]")
residual_by_prompt: dict[str, torch.Tensor] = {}
top_pieces_by_prompt: dict[str, list[str]] = {}
for prompt_type, inputs in [("clean", clean_inputs), ("corrupt", corrupt_inputs)]:
    with torch.no_grad():
        out = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )
    residual = out.hidden_states[LAYER_IDX + 1][0].detach().float().cpu()
    residual_by_prompt[prompt_type] = residual  # [seq_len, d_model]

    last_pos = inputs["input_ids"].shape[1] - 1
    logits_last = out.logits[0, last_pos, :].float().cpu()
    top_vals, top_ids = torch.topk(logits_last.softmax(dim=-1), k=5)
    top_pieces_by_prompt[prompt_type] = [tokenizer.decode([t]) for t in top_ids.tolist()]
    print(f"  {prompt_type:7s} residual shape: {tuple(residual.shape)}   "
          f"top5 next-token: {top_pieces_by_prompt[prompt_type]}")
    del out, logits_last, top_vals, top_ids

clean_top1 = top_pieces_by_prompt["clean"][0]
corrupt_top1 = top_pieces_by_prompt["corrupt"][0]
print(f"  sanity: clean   top1 = {clean_top1!r}  (expect contains ' Tokyo')")
print(f"  sanity: corrupt top1 = {corrupt_top1!r}  (expect contains ' Paris')")

print("\n[4b] Freeing 8B model from device memory")
del model
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ── [5] Load SAE checkpoint NOW (after model is freed) ─────────────────────────
print("\n[5] Loading SAE checkpoint (CPU) — now that 8B model is freed")
try:
    checkpoint = torch.load(sae_path, map_location="cpu", weights_only=True)
except Exception as e:
    print(f"  [info] weights_only=True failed ({e}); retrying with weights_only=False")
    checkpoint = torch.load(sae_path, map_location="cpu", weights_only=False)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(
    checkpoint["state_dict"], dict
):
    state_dict = checkpoint["state_dict"]
elif isinstance(checkpoint, dict):
    state_dict = checkpoint
else:
    raise RuntimeError(f"unexpected SAE checkpoint type: {type(checkpoint)}")

keys_info = []
for k, v in state_dict.items():
    if isinstance(v, torch.Tensor):
        keys_info.append({"key": k, "shape": list(v.shape),
                          "dtype": str(v.dtype), "numel": int(v.numel())})
    else:
        keys_info.append({"key": k, "shape": None,
                          "dtype": type(v).__name__, "numel": None})
keys_info.sort(key=lambda r: r["key"])
for r in keys_info:
    print(f"    {r['key']:32s}  shape={r['shape']!s:24s}  dtype={r['dtype']}")

keys_json = outputs_dir / f"{OUT_PREFIX_PRELIM}_keys.json"
with keys_json.open("w", encoding="utf-8") as f:
    json.dump({"sae_path": str(sae_path), "keys": keys_info}, f, indent=2)
print(f"  saved: {keys_json}")


def pick(pool, candidates):
    for c in candidates:
        if c in pool:
            return c
    return None


key_pool = {k for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
enc_w_key = pick(key_pool, ["W_enc", "encoder.weight", "encoder.W", "enc.weight"])
dec_w_key = pick(key_pool, ["W_dec", "decoder.weight", "decoder.W", "dec.weight"])
b_enc_key = pick(key_pool, ["b_enc", "encoder.bias", "enc.bias"])
b_dec_key = pick(key_pool, ["b_dec", "decoder.bias", "dec.bias", "b_pre"])

if enc_w_key is None:
    print("  [error] no encoder weight in SAE checkpoint; saved keys for inspection.")
    raise SystemExit(0)

W_enc = state_dict[enc_w_key].to(torch.float32)
W_dec = state_dict[dec_w_key].to(torch.float32) if dec_w_key else None
b_enc = state_dict[b_enc_key].to(torch.float32) if b_enc_key else None
b_dec = state_dict[b_dec_key].to(torch.float32) if b_dec_key else None
del checkpoint, state_dict
gc.collect()

print(f"  enc_w_key = {enc_w_key}   shape = {tuple(W_enc.shape)}")
print(f"  dec_w_key = {dec_w_key}   shape = {tuple(W_dec.shape) if W_dec is not None else None}")
print(f"  b_enc_key = {b_enc_key}   shape = {tuple(b_enc.shape) if b_enc is not None else None}")
print(f"  b_dec_key = {b_dec_key}   shape = {tuple(b_dec.shape) if b_dec is not None else None}")

# Infer d_model / d_sae from W_enc shape
if W_enc.shape[0] == hidden_size:
    d_model, d_sae = W_enc.shape  # [d_model, d_sae]
    enc_orientation = "in_x_features"
elif W_enc.shape[1] == hidden_size:
    d_sae, d_model = W_enc.shape  # [d_sae, d_model]
    enc_orientation = "features_x_in"
else:
    raise RuntimeError(
        f"neither dim of W_enc ({W_enc.shape}) matches hidden_size={hidden_size}"
    )

dec_orientation = None
if W_dec is not None:
    if W_dec.shape == (d_sae, d_model):
        dec_orientation = "features_x_out"
    elif W_dec.shape == (d_model, d_sae):
        dec_orientation = "out_x_features"
    else:
        print(f"  [warning] W_dec shape {tuple(W_dec.shape)} unrecognized; reconstruction skipped.")
        W_dec = None

print(f"  d_model   = {d_model}")
print(f"  d_sae     = {d_sae}")
print(f"  encoder orientation = {enc_orientation}")
print(f"  decoder orientation = {dec_orientation}")

# ── [6] SAE encode (TopK) ──────────────────────────────────────────────────────
print(f"\n[6] SAE encode with TopK (k={TOP_K_SAE})")


def encode_topk(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if enc_orientation == "in_x_features":
        pre = X @ W_enc
    else:
        pre = X @ W_enc.T
    if b_enc is not None:
        pre = pre + b_enc
    k_eff = min(TOP_K_SAE, pre.shape[-1])
    vals, idx = torch.topk(pre, k=k_eff, dim=-1)
    feats = torch.zeros_like(pre)
    feats.scatter_(dim=-1, index=idx, src=vals)
    return pre, feats


features_by_prompt: dict[str, torch.Tensor] = {}
for prompt_type in ("clean", "corrupt"):
    _, feats = encode_topk(residual_by_prompt[prompt_type])
    features_by_prompt[prompt_type] = feats
    active_count = float((feats != 0).float().sum(dim=-1).mean().item())
    active_frac  = float((feats != 0).float().mean().item())
    print(f"  {prompt_type:7s} feats shape: {tuple(feats.shape)}   "
          f"active_count/pos≈{active_count:.1f}  active_frac={active_frac:.6f}")

# ── [7] Top-k features per position → CSV ──────────────────────────────────────
print(f"\n[7] Top-{TOP_K_FEATURES} active features per position")
top_feature_rows: list[dict] = []
for prompt_type in ("clean", "corrupt"):
    feats = features_by_prompt[prompt_type]
    for i, tok in enumerate(prompt_tokens[prompt_type]):
        vec = feats[i]
        k_take = min(TOP_K_FEATURES, vec.numel())
        top_vals, top_ids = torch.topk(vec, k=k_take)
        for rank, (fid, val) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), start=1):
            if val == 0.0:
                continue
            top_feature_rows.append({
                "prompt_type": prompt_type,
                "position":    i,
                "token_id":    tok["token_id"],
                "raw_token":   tok["raw_token"],
                "piece":       tok["piece"],
                "piece_repr":  tok["piece_repr"],
                "feature_id":  int(fid),
                "rank":        rank,
                "activation":  float(val),
            })

top_features_csv = outputs_dir / f"{OUT_PREFIX_PRELIM}_top_features.csv"
with top_features_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "prompt_type", "position", "token_id", "raw_token",
            "piece", "piece_repr", "feature_id", "rank", "activation",
        ],
    )
    writer.writeheader()
    writer.writerows(top_feature_rows)
print(f"  saved: {top_features_csv}")

print("  top1 examples:")
for prompt_type in ("clean", "corrupt"):
    rows = [r for r in top_feature_rows
            if r["prompt_type"] == prompt_type and r["rank"] == 1]
    for r in rows:
        print(f"    [{prompt_type:7s}] pos={r['position']} {r['piece']!r:14s}  "
              f"top1 f{r['feature_id']:6d}  act={r['activation']:+.4f}")

# ── [8] Token × feature heatmap ────────────────────────────────────────────────
print(f"\n[8] Token × feature heatmap (max {MAX_FEATURES_FOR_HEATMAP} features)")
selected_set = set(int(r["feature_id"]) for r in top_feature_rows)
selected_feature_ids = sorted(selected_set)
if len(selected_feature_ids) > MAX_FEATURES_FOR_HEATMAP:
    feature_max_act: dict[int, float] = {}
    for prompt_type in ("clean", "corrupt"):
        feats = features_by_prompt[prompt_type]
        for fid in selected_feature_ids:
            v = float(feats[:, fid].max().item())
            if v > feature_max_act.get(fid, float("-inf")):
                feature_max_act[fid] = v
    selected_feature_ids = sorted(
        sorted(feature_max_act.keys(), key=lambda x: feature_max_act[x], reverse=True)[
            :MAX_FEATURES_FOR_HEATMAP
        ]
    )
print(f"  num selected features: {len(selected_feature_ids)}")

row_labels: list[str] = []
matrix_rows: list[list[float]] = []
for prompt_type in ("clean", "corrupt"):
    feats = features_by_prompt[prompt_type]
    for i, tok in enumerate(prompt_tokens[prompt_type]):
        row_labels.append(f"{prompt_type[0]}:{i}:{tok['piece_repr']}")
        matrix_rows.append([float(feats[i, fid].item()) for fid in selected_feature_ids])

matrix = np.array(matrix_rows, dtype=np.float32)
matrix_csv = outputs_dir / f"{OUT_PREFIX_PRELIM}_feature_matrix.csv"
with matrix_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["token_label"] + [f"f{fid}" for fid in selected_feature_ids])
    for label, row in zip(row_labels, matrix_rows):
        writer.writerow([label] + row)
print(f"  saved: {matrix_csv}")

fig_w = max(8.0, 0.20 * len(selected_feature_ids) + 4.0)
fig_h = max(4.0, 0.35 * len(row_labels) + 1.5)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
im = ax.imshow(matrix, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(selected_feature_ids)))
ax.set_xticklabels([str(fid) for fid in selected_feature_ids], rotation=90, fontsize=7)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=8, family="monospace")
ax.set_xlabel("feature_id")
ax.set_ylabel("prompt:position:token (· = leading space)")
ax.set_title(
    f"{MODEL_ID}  layer{LAYER_IDX} residual-stream SAE features  "
    f"(repo: {SAE_ID}, TopK k={TOP_K_SAE})"
)
plt.colorbar(im, ax=ax, label="activation")
fig.tight_layout()
heatmap_png = outputs_dir / f"{OUT_PREFIX_NB}_feature_heatmap.png"
fig.savefig(heatmap_png, dpi=140)
plt.close(fig)
print(f"  saved: {heatmap_png}")

# ── [9] Differential analysis ──────────────────────────────────────────────────
print("\n[9] Differential analysis (clean − corrupt)")
clean_feats = features_by_prompt["clean"]
corrupt_feats = features_by_prompt["corrupt"]

comparisons = [
    {
        "name":        "pos3_japan_minus_france",
        "label":       "pos=3  Japan − France",
        "clean_pos":   3,
        "corrupt_pos": 3,
    },
    {
        "name":        "last_clean_minus_corrupt",
        "label":       f"last  clean(pos={clean_last_pos}) − corrupt(pos={corrupt_last_pos})",
        "clean_pos":   clean_last_pos,
        "corrupt_pos": corrupt_last_pos,
    },
]

diff_rows: list[dict] = []
diffs_by_comparison: dict[str, torch.Tensor] = {}
top_feature_ids_for_diff_heatmap: list[int] = []

for cmp in comparisons:
    c_vec = clean_feats[cmp["clean_pos"]]
    k_vec = corrupt_feats[cmp["corrupt_pos"]]
    diff = c_vec - k_vec
    diffs_by_comparison[cmp["name"]] = diff
    top_pos_vals, top_pos_ids = torch.topk(diff, k=TOP_K_DIFF)
    top_neg_vals, top_neg_ids = torch.topk(-diff, k=TOP_K_DIFF)
    selected_ids = set(top_pos_ids.tolist()) | set(top_neg_ids.tolist())
    for fid in sorted(selected_ids):
        ca = float(c_vec[fid].item())
        ka = float(k_vec[fid].item())
        d  = ca - ka
        direction = "clean_gt_corrupt" if d > 0 else "corrupt_gt_clean"
        diff_rows.append({
            "comparison":         cmp["name"],
            "feature_id":         int(fid),
            "clean_activation":   ca,
            "corrupt_activation": ka,
            "diff":               d,
            "abs_diff":           abs(d),
            "direction":          direction,
        })
    top_feature_ids_for_diff_heatmap.extend(sorted(selected_ids))
    print(f"  [{cmp['name']}]")
    print("    top clean > corrupt:")
    for fid, val in zip(top_pos_ids.tolist()[:5], top_pos_vals.tolist()[:5]):
        ca = float(c_vec[fid].item()); ka = float(k_vec[fid].item())
        print(f"      f{fid:6d}  diff=+{val:.4f}  (clean={ca:+.3f}  corrupt={ka:+.3f})")
    print("    top corrupt > clean:")
    for fid, val in zip(top_neg_ids.tolist()[:5], top_neg_vals.tolist()[:5]):
        ca = float(c_vec[fid].item()); ka = float(k_vec[fid].item())
        print(f"      f{fid:6d}  diff=-{val:.4f}  (clean={ca:+.3f}  corrupt={ka:+.3f})")

diff_rows.sort(key=lambda r: (r["comparison"], -r["abs_diff"]))
feature_diffs_csv = outputs_dir / f"{OUT_PREFIX_PRELIM}_feature_diffs.csv"
with feature_diffs_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "comparison", "feature_id", "clean_activation", "corrupt_activation",
            "diff", "abs_diff", "direction",
        ],
    )
    writer.writeheader()
    writer.writerows(diff_rows)
print(f"  saved: {feature_diffs_csv}")

# Diff bar plot (4 subplots)
fig, axes = plt.subplots(4, 1, figsize=(10, 11))


def _signed_topk(vec, k, descending):
    if descending:
        vals, ids = torch.topk(vec, k=k)
        return ids.tolist(), vals.tolist()
    else:
        vals, ids = torch.topk(-vec, k=k)
        return ids.tolist(), [-v for v in vals.tolist()]


bar_panels = [
    ("pos3_japan_minus_france", True,
     "Japan > France  (clean pos=3 − corrupt pos=3, top positive)", "tab:red"),
    ("pos3_japan_minus_france", False,
     "France > Japan  (corrupt pos=3 − clean pos=3, top positive)", "tab:blue"),
    ("last_clean_minus_corrupt", True,
     f"clean last > corrupt last  "
     f"(clean pos={clean_last_pos} − corrupt pos={corrupt_last_pos}, top positive)", "tab:red"),
    ("last_clean_minus_corrupt", False,
     f"corrupt last > clean last  "
     f"(corrupt pos={corrupt_last_pos} − clean pos={clean_last_pos}, top positive)", "tab:blue"),
]
for ax, (cmp_name, descending, title, color) in zip(axes, bar_panels):
    diff_vec = diffs_by_comparison[cmp_name]
    fids, vals = _signed_topk(diff_vec, TOP_K_DIFF, descending)
    plot_vals = [abs(v) for v in vals]
    ax.bar(range(len(fids)), plot_vals, color=color)
    ax.set_xticks(range(len(fids)))
    ax.set_xticklabels([str(f) for f in fids], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|Δactivation|")
    ax.set_title(title, fontsize=10)
fig.suptitle(
    f"{MODEL_ID} layer{LAYER_IDX} Qwen-Scope SAE differential features  (repo: {SAE_ID})",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
diffs_bar_png = outputs_dir / f"{OUT_PREFIX_NB}_feature_diffs_bar.png"
fig.savefig(diffs_bar_png, dpi=140)
plt.close(fig)
print(f"  saved: {diffs_bar_png}")

# Diff heatmap (diverging)
heat_fids = sorted(set(top_feature_ids_for_diff_heatmap))


def _max_abs(fid: int) -> float:
    return max(
        abs(float(diffs_by_comparison[c["name"]][fid].item())) for c in comparisons
    )


heat_fids.sort(key=_max_abs, reverse=True)
diff_matrix = np.array(
    [[float(diffs_by_comparison[c["name"]][fid].item()) for fid in heat_fids]
     for c in comparisons],
    dtype=np.float32,
)
vmax = float(np.max(np.abs(diff_matrix))) if diff_matrix.size else 1.0
fig_w = max(8.0, 0.18 * len(heat_fids) + 3.0)
fig, ax = plt.subplots(figsize=(fig_w, 3.2))
im = ax.imshow(diff_matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(heat_fids)))
ax.set_xticklabels([str(fid) for fid in heat_fids], rotation=90, fontsize=7)
ax.set_yticks(range(len(comparisons)))
ax.set_yticklabels([c["label"] for c in comparisons], fontsize=9)
ax.set_xlabel("feature_id")
ax.set_title(
    f"{MODEL_ID} layer{LAYER_IDX} Qwen-Scope SAE differential features  "
    f"(red: clean>corrupt, blue: corrupt>clean)"
)
plt.colorbar(im, ax=ax, label="Δactivation (clean − corrupt)")
fig.tight_layout()
diffs_heatmap_png = outputs_dir / f"{OUT_PREFIX_NB}_feature_diffs_heatmap.png"
fig.savefig(diffs_heatmap_png, dpi=140)
plt.close(fig)
print(f"  saved: {diffs_heatmap_png}")

# ── [10] Reconstruction check ──────────────────────────────────────────────────
recon_metrics_rows: list[dict] = []
reconstruction_check_available = False
rmse_by_prompt: dict[str, float] = {}
mean_cos_by_prompt: dict[str, float] = {}
if W_dec is not None:
    print("\n[10] Reconstruction check (target = residual stream input)")
    reconstruction_check_available = True
    try:
        for prompt_type in ("clean", "corrupt"):
            feats = features_by_prompt[prompt_type]
            if dec_orientation == "features_x_out":
                recon = feats @ W_dec
            else:
                recon = feats @ W_dec.T
            if b_dec is not None:
                recon = recon + b_dec
            target = residual_by_prompt[prompt_type]
            diff = recon - target
            mse  = float((diff ** 2).mean().item())
            rmse = float(mse ** 0.5)
            rmse_by_prompt[prompt_type] = rmse
            cos_vals: list[float] = []
            for i in range(target.shape[0]):
                r_vec = recon[i]; t_vec = target[i]
                denom = (r_vec.norm() * t_vec.norm()).clamp_min(1e-12)
                cos = float((r_vec @ t_vec / denom).item())
                cos_vals.append(cos)
                norm_ratio = float((r_vec.norm() / t_vec.norm().clamp_min(1e-12)).item())
                recon_metrics_rows.append({
                    "prompt_type":    prompt_type,
                    "position":       i,
                    "token_id":       prompt_tokens[prompt_type][i]["token_id"],
                    "piece":          prompt_tokens[prompt_type][i]["piece"],
                    "mse_seq":        mse,
                    "rmse_seq":       rmse,
                    "cosine_pos":     cos,
                    "norm_ratio_pos": norm_ratio,
                })
            mean_cos_by_prompt[prompt_type] = float(np.mean(cos_vals))
            print(f"  {prompt_type:7s}  rmse={rmse:.4f}  mean_cos={mean_cos_by_prompt[prompt_type]:.4f}")
    except Exception as e:
        print(f"  [warning] reconstruction failed: {e}")
        traceback.print_exc()
        reconstruction_check_available = False
else:
    print("\n[10] Reconstruction check skipped (no decoder weights).")

if recon_metrics_rows:
    recon_csv = outputs_dir / f"{OUT_PREFIX_PRELIM}_reconstruction_metrics.csv"
    with recon_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt_type", "position", "token_id", "piece",
                "mse_seq", "rmse_seq", "cosine_pos", "norm_ratio_pos",
            ],
        )
        writer.writeheader()
        writer.writerows(recon_metrics_rows)
    print(f"  saved: {recon_csv}")
else:
    recon_csv = None

# ── [11] Summary JSON ──────────────────────────────────────────────────────────
print("\n[11] Saving summary JSON")
summary = {
    "model_id":             MODEL_ID,
    "sae_id":               SAE_ID,
    "layer_idx":            LAYER_IDX,
    "top_k_sae":            TOP_K_SAE,
    "residual_source":      f"outputs.hidden_states[{LAYER_IDX + 1}]  (= block {LAYER_IDX} output)",
    "device":               device,
    "dtype":                str(dtype),
    "clean_prompt":         CLEAN_PROMPT,
    "corrupt_prompt":       CORRUPT_PROMPT,
    "clean_answer":         CLEAN_ANSWER,
    "corrupt_answer":       CORRUPT_ANSWER,
    "clean_seq_len":        clean_seq_len,
    "corrupt_seq_len":      corrupt_seq_len,
    "clean_top1_piece":     clean_top1,
    "corrupt_top1_piece":   corrupt_top1,
    "sae_checkpoint_path":  str(sae_path),
    "sae_checkpoint_size_bytes": sae_size,
    "sae_keys":             [r["key"] for r in keys_info],
    "inferred_encoder_key": enc_w_key,
    "inferred_decoder_key": dec_w_key,
    "inferred_b_enc_key":   b_enc_key,
    "inferred_b_dec_key":   b_dec_key,
    "W_enc_shape":          list(W_enc.shape),
    "W_dec_shape":          list(W_dec.shape) if W_dec is not None else None,
    "inferred_d_model":     int(d_model),
    "inferred_d_sae":       int(d_sae),
    "encoder_orientation":  enc_orientation,
    "decoder_orientation":  dec_orientation,
    "top_k_features_per_position": TOP_K_FEATURES,
    "heatmap_num_features": len(selected_feature_ids),
    "diff_top_k_per_direction": TOP_K_DIFF,
    "diff_comparisons":     [c["name"] for c in comparisons],
    "diff_heatmap_num_features": len(heat_fids),
    "reconstruction_check_available": reconstruction_check_available,
    "reconstruction_rmse_clean":     rmse_by_prompt.get("clean"),
    "reconstruction_rmse_corrupt":   rmse_by_prompt.get("corrupt"),
    "reconstruction_mean_cos_clean":   mean_cos_by_prompt.get("clean"),
    "reconstruction_mean_cos_corrupt": mean_cos_by_prompt.get("corrupt"),
    "output_files": [
        str(keys_json),
        str(top_features_csv),
        str(matrix_csv),
        str(heatmap_png),
        str(feature_diffs_csv),
        str(diffs_bar_png),
        str(diffs_heatmap_png),
        *([str(recon_csv)] if recon_csv else []),
    ],
}
summary_json = outputs_dir / f"{OUT_PREFIX_PRELIM}_summary.json"
with summary_json.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"  saved: {summary_json}")

print("\nDone.")
