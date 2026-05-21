# mwhanna/qwen3-4b-transcoders を用いて Qwen3-4B の全 MLP layer (0..35) を sweep し、
# clean / corrupt prompt の transcoder feature activation 差分が layer 方向にどう変化するかを調べる。
#
# - 14_prelim_qwen3_4b_transcoder_smoke.py の単一 / 3 layer 詳細版に対し、
#   このスクリプトは aggregate 専用。per-layer の heatmap / bar plot は作らない。
# - 巨大な repo 全体は download しない。layer_{idx}.safetensors を 1 layer ずつ取得し、
#   使い終わったら CPU 上の tensor を del + gc.collect() する。
#   各 layer safetensors は約 1.68GB あるため、36 layer 全部取ると cache は約 60GB になる。
# - Qwen3-4B model は 1 回だけ load し、全 36 layer の MLP に hook を付けて
#   clean / corrupt の forward を 1 回ずつ実行する。model はそのあと解放する。
#
# hidden_states インデックス対応:
#   hidden_states[0]     = embed_tokens output
#   hidden_states[j + 1] = block j の output  (j = 0..K-1)
#
#   layer_idx = j の transcoder は block j の MLP input / output を見る:
#     layer 23: hidden_states[23] → block 23 → hidden_states[24] の MLP 部分
#     layer 24: hidden_states[24] → block 24 → hidden_states[25] の MLP 部分 (note02 k=24→25 の本命)
#     layer 25: hidden_states[25] → block 25 → hidden_states[26] の MLP 部分
#
# 注意:
#   MLP input は residual stream そのものではなく、
#   Qwen3DecoderLayer 内で attention 後の residual stream に post_attention_layernorm を
#   かけたもの。transcoder feature の解釈ではこの違いに注意。
#
# 出力:
#   outputs/prelim_qwen3_4b_transcoder_layer_sweep_summary.csv
#   outputs/prelim_qwen3_4b_transcoder_layer_sweep_summary.json
#   outputs/nb03_qwen3_4b_transcoder_layer_sweep_diff_strength.png
#   outputs/nb03_qwen3_4b_transcoder_layer_sweep_active_fraction.png
#   outputs/nb03_qwen3_4b_transcoder_layer_sweep_reconstruction.png
#   outputs/nb03_qwen3_4b_transcoder_layer_sweep_summary_panel.png
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
import yaml

# 文字化け対策: macOS 組み込みの Hiragino Sans を優先
plt.rcParams["font.family"] = ["Hiragino Sans", "Apple SD Gothic Neo", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_config, resolve_outputs_dir

# ── Config ─────────────────────────────────────────────────────────────────────
cfg = load_config()
model_id = cfg["model_id"]
attn_impl = cfg["attn_implementation"]

TRANSCODER_REPO = "mwhanna/qwen3-4b-transcoders"
NUM_LAYERS_EXPECTED = 36
LAYER_IDXS = list(range(NUM_LAYERS_EXPECTED))
TOP_K_DIFF = 20
LAYER_REFERENCE = 24  # note02 で着目した layer

# Heatmap (per-layer combined sum+diff) 関連 — script 14 と同じ規約
TOP_K_FEATURES = 20                    # per-position top-k features (top_features.csv 用)
MAX_FEATURES_FOR_HEATMAP_POOL = 300    # feature_matrix.csv に保存する pool 列数
MAX_FEATURES_FOR_HEATMAP       = 60    # 表示する features 数

CLEAN_PROMPT   = "The capital of Japan is"
CORRUPT_PROMPT = "The capital of France is"
CLEAN_ANSWER   = " Tokyo"
CORRUPT_ANSWER = " Paris"

D_MODEL_EXPECTED = 2560

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

print(f"model_id        : {model_id}")
print(f"transcoder_repo : {TRANSCODER_REPO}")
print(f"layer_idxs      : 0..{NUM_LAYERS_EXPECTED - 1}  ({len(LAYER_IDXS)} layers)")
print(f"device          : {device}")
print(f"dtype           : {dtype}")
assert model_id == "Qwen/Qwen3-4B", f"unexpected model_id: {model_id}"

# ── [1] Download shared YAML configs ───────────────────────────────────────────
print("\n[1] Downloading YAML configs via hf_hub_download")
try:
    config_yaml_path = hf_hub_download(repo_id=TRANSCODER_REPO, filename="config.yaml")
    print(f"  config.yaml      : {config_yaml_path}")
except Exception as e:
    print(f"  [warning] config.yaml fetch failed: {e}")
    config_yaml_path = None

try:
    wandb_yaml_path = hf_hub_download(repo_id=TRANSCODER_REPO, filename="wandb-config.yaml")
    print(f"  wandb-config.yaml: {wandb_yaml_path}")
except Exception as e:
    print(f"  [warning] wandb-config.yaml fetch failed: {e}")
    wandb_yaml_path = None


def _load_yaml(path):
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"  [warning] failed to parse {path}: {e}")
        return None


config_yaml = _load_yaml(config_yaml_path)
wandb_yaml  = _load_yaml(wandb_yaml_path)

act_fn_name = None
config_d_feature_hint = None
feature_input_hook = None
feature_output_hook = None
for src in [config_yaml, wandb_yaml]:
    if not isinstance(src, dict):
        continue
    flat = {}
    for k, v in src.items():
        if isinstance(v, dict) and set(v.keys()) <= {"value", "desc"} and "value" in v:
            flat[k] = v["value"]
        else:
            flat[k] = v
    if act_fn_name is None and "act_fn" in flat:
        act_fn_name = flat["act_fn"]
    if config_d_feature_hint is None:
        for key in ("d_feature", "d_sae", "dictionary_size", "n_features"):
            if key in flat:
                config_d_feature_hint = flat[key]
                break
    if feature_input_hook is None and "feature_input_hook" in flat:
        feature_input_hook = flat["feature_input_hook"]
    if feature_output_hook is None and "feature_output_hook" in flat:
        feature_output_hook = flat["feature_output_hook"]

activation_fn_used = (act_fn_name or "relu").lower()
print(f"  act_fn (from yaml)            : {act_fn_name}")
print(f"  d_feature hint (from yaml)    : {config_d_feature_hint}")
print(f"  feature_input_hook  (yaml)    : {feature_input_hook}")
print(f"  feature_output_hook (yaml)    : {feature_output_hook}")

# ── [2] Tokenizer & token tables ───────────────────────────────────────────────
print("\n[2] Tokenizer (raw, no chat template)")
tokenizer = AutoTokenizer.from_pretrained(model_id)

clean_inputs   = tokenizer(CLEAN_PROMPT,   return_tensors="pt").to(device)
corrupt_inputs = tokenizer(CORRUPT_PROMPT, return_tensors="pt").to(device)
clean_seq_len   = clean_inputs["input_ids"].shape[1]
corrupt_seq_len = corrupt_inputs["input_ids"].shape[1]
clean_last_pos   = clean_seq_len   - 1
corrupt_last_pos = corrupt_seq_len - 1

assert clean_seq_len == 5, f"unexpected clean_seq_len: {clean_seq_len}"
assert corrupt_seq_len == 5, f"unexpected corrupt_seq_len: {corrupt_seq_len}"


def piece_repr(piece: str) -> str:
    return piece.replace(" ", "·")


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

# ── [3] Load model ─────────────────────────────────────────────────────────────
print("\n[3] Loading Qwen3-4B model (once)")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=dtype,
    attn_implementation=attn_impl,
)
model.to(device).eval()  # type: ignore[union-attr]
K = len(model.model.layers)
hidden_size = model.config.hidden_size
print(f"  num_decoder_layers = {K}")
print(f"  hidden_size        = {hidden_size}")
assert K == NUM_LAYERS_EXPECTED, f"expected {NUM_LAYERS_EXPECTED} layers, got {K}"
assert hidden_size == D_MODEL_EXPECTED, (
    f"hidden_size {hidden_size} != expected d_model {D_MODEL_EXPECTED}"
)

# ── [4] Single forward with hooks on every MLP ─────────────────────────────────
print(f"\n[4] Forward with hooks on all {K} MLP layers")
mlp_in_by_layer:  dict[int, dict[str, torch.Tensor]] = {li: {} for li in LAYER_IDXS}
mlp_out_by_layer: dict[int, dict[str, torch.Tensor]] = {li: {} for li in LAYER_IDXS}
current_prompt_type = {"name": None}


def _make_hooks(layer_idx: int):
    def pre_hook(module, inputs):
        mlp_in_by_layer[layer_idx][current_prompt_type["name"]] = inputs[0].detach().cpu()
    def post_hook(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        mlp_out_by_layer[layer_idx][current_prompt_type["name"]] = out.detach().cpu()
    return pre_hook, post_hook


handles = []
top5_by_prompt: dict[str, list[str]] = {}
try:
    for li in LAYER_IDXS:
        pre, post = _make_hooks(li)
        target_mlp = model.model.layers[li].mlp
        handles.append(target_mlp.register_forward_pre_hook(pre))
        handles.append(target_mlp.register_forward_hook(post))

    for prompt_type, inputs in [("clean", clean_inputs), ("corrupt", corrupt_inputs)]:
        current_prompt_type["name"] = prompt_type
        with torch.no_grad():
            out = model(
                **inputs,
                output_hidden_states=False,
                output_attentions=False,
                use_cache=False,
            )
        last_pos = inputs["input_ids"].shape[1] - 1
        logits_last = out.logits[0, last_pos, :].float().cpu()
        top_vals, top_ids = torch.topk(logits_last.softmax(dim=-1), k=5)
        top_pieces = [tokenizer.decode([tid]) for tid in top_ids.tolist()]
        top5_by_prompt[prompt_type] = top_pieces
        print(f"  {prompt_type:7s} top5 next-token: {top_pieces}")
finally:
    for h in handles:
        h.remove()

clean_top1   = top5_by_prompt["clean"][0]   if top5_by_prompt["clean"]   else None
corrupt_top1 = top5_by_prompt["corrupt"][0] if top5_by_prompt["corrupt"] else None
print(f"  sanity: clean   top1 = {clean_top1!r}  (expect ' Tokyo')")
print(f"  sanity: corrupt top1 = {corrupt_top1!r}  (expect ' Paris')")
assert clean_top1 == CLEAN_ANSWER, f"clean top1 != {CLEAN_ANSWER!r}"
assert corrupt_top1 == CORRUPT_ANSWER, f"corrupt top1 != {CORRUPT_ANSWER!r}"

# Free model now: only mlp_in / mlp_out on CPU are needed.
print("\n[4b] Freeing Qwen3-4B model from device memory")
del out, logits_last, top_vals, top_ids, model
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ── Helpers ────────────────────────────────────────────────────────────────────
def pick_key(pool, candidates):
    for c in candidates:
        if c in pool:
            return c
    return None


def infer_orientations(W_enc: torch.Tensor, W_dec_or_none, d_model: int):
    if W_enc.ndim != 2:
        raise RuntimeError(f"unexpected encoder weight ndim: {W_enc.shape}")
    if W_enc.shape[0] == d_model:
        dm, df = W_enc.shape
        enc_orientation = "in_x_features"
    elif W_enc.shape[1] == d_model:
        df, dm = W_enc.shape
        enc_orientation = "features_x_in"
    else:
        raise RuntimeError(
            f"neither dim of W_enc ({W_enc.shape}) matches d_model={d_model}"
        )

    dec_orientation = None
    if W_dec_or_none is not None:
        if W_dec_or_none.shape == (df, dm):
            dec_orientation = "features_x_out"
        elif W_dec_or_none.shape == (dm, df):
            dec_orientation = "out_x_features"
        else:
            W_dec_or_none = None
    return dm, df, enc_orientation, dec_orientation, W_dec_or_none


def encode_features(
    X: torch.Tensor,
    W_enc: torch.Tensor,
    b_enc,
    enc_orientation: str,
    act_fn: str,
) -> torch.Tensor:
    if enc_orientation == "in_x_features":
        pre = X @ W_enc
    else:
        pre = X @ W_enc.T
    if b_enc is not None:
        pre = pre + b_enc
    if act_fn == "relu":
        return torch.relu(pre)
    if act_fn in ("gelu", "gelu_approx"):
        return torch.nn.functional.gelu(pre)
    if act_fn == "identity":
        return pre
    return torch.relu(pre)


def compute_diff_stats(diff: torch.Tensor, top_k: int) -> dict:
    abs_diff = diff.abs()
    max_abs  = float(abs_diff.max().item())
    l2       = float(diff.norm(p=2).item())
    mean_abs = float(abs_diff.mean().item())

    top_abs_vals, _ = torch.topk(abs_diff, k=top_k)
    top_abs_sum  = float(top_abs_vals.sum().item())
    top_abs_mean = float(top_abs_vals.mean().item())

    top_pos_vals, top_pos_ids = torch.topk(diff, k=1)
    top_neg_vals, top_neg_ids = torch.topk(-diff, k=1)
    return {
        "max_abs_diff":          max_abs,
        "l2_diff":               l2,
        "mean_abs_diff":         mean_abs,
        "top_k_abs_diff_sum":    top_abs_sum,
        "top_k_abs_diff_mean":   top_abs_mean,
        "top_clean_gt_corrupt_feature": int(top_pos_ids[0].item()),
        "top_clean_gt_corrupt_diff":    float(top_pos_vals[0].item()),
        "top_corrupt_gt_clean_feature": int(top_neg_ids[0].item()),
        "top_corrupt_gt_clean_diff":   -float(top_neg_vals[0].item()),
    }


# ── [5] Per-layer sweep ────────────────────────────────────────────────────────
print(f"\n[5] Per-layer sweep over {len(LAYER_IDXS)} layers")
print("    progress format: layer XX  active=clean/corrupt  pos3_max  last_max  rmse  cos")

aggregate_rows: list[dict] = []
position_metric_rows: list[dict] = []  # per (layer, position) detailed metrics (script 14 と同形式)

for layer_idx in LAYER_IDXS:
    # Download safetensors for this layer (snapshot_download NOT used)
    try:
        layer_path = hf_hub_download(
            repo_id=TRANSCODER_REPO,
            filename=f"layer_{layer_idx}.safetensors",
        )
    except Exception as e:
        print(f"  [error] layer_{layer_idx}.safetensors download failed: {e}")
        print("  hint: pip install -U huggingface_hub hf_xet safetensors pyyaml")
        raise

    tensors = load_safetensors(str(layer_path), device="cpu")
    key_pool = set(tensors.keys())
    enc_w_key = pick_key(key_pool, ["W_enc", "encoder.weight", "encoder.W", "enc.weight"])
    dec_w_key = pick_key(key_pool, ["W_dec", "decoder.weight", "decoder.W", "dec.weight"])
    b_enc_key = pick_key(key_pool, ["b_enc", "encoder.bias", "enc.bias"])
    b_dec_key = pick_key(key_pool, ["b_dec", "decoder.bias", "dec.bias", "b_pre"])

    if enc_w_key is None:
        print(f"  [error] no encoder weight key in layer {layer_idx}; skipping.")
        del tensors
        gc.collect()
        continue

    W_enc = tensors[enc_w_key].to(torch.float32)
    b_enc = tensors[b_enc_key].to(torch.float32) if b_enc_key else None
    W_dec = tensors[dec_w_key].to(torch.float32) if dec_w_key else None
    b_dec = tensors[b_dec_key].to(torch.float32) if b_dec_key else None
    del tensors
    gc.collect()

    d_model, d_feature, enc_orientation, dec_orientation, W_dec = infer_orientations(
        W_enc, W_dec, D_MODEL_EXPECTED
    )

    # Encode features for clean / corrupt
    features_by_prompt: dict[str, torch.Tensor] = {}
    for prompt_type in ("clean", "corrupt"):
        X = mlp_in_by_layer[layer_idx][prompt_type][0].to(torch.float32)
        feats = encode_features(X, W_enc, b_enc, enc_orientation, activation_fn_used)
        features_by_prompt[prompt_type] = feats

    clean_feats   = features_by_prompt["clean"]
    corrupt_feats = features_by_prompt["corrupt"]

    active_mask_clean   = (clean_feats   > 0).float()
    active_mask_corrupt = (corrupt_feats > 0).float()
    active_frac_clean   = float(active_mask_clean.mean().item())
    active_frac_corrupt = float(active_mask_corrupt.mean().item())
    active_count_clean_mean   = float(active_mask_clean.sum(dim=-1).mean().item())
    active_count_corrupt_mean = float(active_mask_corrupt.sum(dim=-1).mean().item())

    # [PM] Per-(layer, position) summary metrics (script 14 と同形式)
    n_pos = min(clean_feats.shape[0], corrupt_feats.shape[0])
    for p in range(n_pos):
        clean_vec   = clean_feats[p]
        corrupt_vec = corrupt_feats[p]
        delta    = clean_vec - corrupt_vec
        sigma    = clean_vec + corrupt_vec
        elem_max = torch.maximum(clean_vec, corrupt_vec)

        clean_norm   = clean_vec.norm(p=2).clamp_min(1e-12)
        corrupt_norm = corrupt_vec.norm(p=2).clamp_min(1e-12)
        cos_clean_corrupt = float(
            (torch.dot(clean_vec, corrupt_vec) / (clean_norm * corrupt_norm)).item()
        )

        clean_active   = clean_vec   > 0
        corrupt_active = corrupt_vec > 0
        active_inter   = int((clean_active & corrupt_active).sum().item())
        active_uni     = int((clean_active | corrupt_active).sum().item())
        jaccard_active = float(active_inter) / float(active_uni) if active_uni > 0 else 1.0

        tanimoto_num = float(torch.minimum(clean_vec, corrupt_vec).sum().item())
        tanimoto_den = float(torch.maximum(clean_vec, corrupt_vec).sum().item())
        tanimoto = tanimoto_num / tanimoto_den if tanimoto_den > 1e-12 else 1.0

        position_metric_rows.append({
            "layer_idx":           layer_idx,
            "position":            p,
            "clean_token":         prompt_tokens["clean"][p]["piece"],
            "corrupt_token":       prompt_tokens["corrupt"][p]["piece"],
            "max_delta_pos":       float(delta.max().item()),
            "max_delta_neg":       float((-delta).max().item()),
            "max_abs_delta":       float(delta.abs().max().item()),
            "max_sum":             float(sigma.max().item()),
            "max_single":          float(elem_max.max().item()),
            "l1_abs_delta":        float(delta.abs().sum().item()),
            "l2_delta":            float(delta.norm(p=2).item()),
            "l2_sum":              float(sigma.norm(p=2).item()),
            "l2_max_single":       float(elem_max.norm(p=2).item()),
            "l2_clean":            float(clean_vec.norm(p=2).item()),
            "l2_corrupt":          float(corrupt_vec.norm(p=2).item()),
            "cos_clean_corrupt":   cos_clean_corrupt,
            "active_clean":        int(clean_active.sum().item()),
            "active_corrupt":      int(corrupt_active.sum().item()),
            "active_intersection": active_inter,
            "active_union":        active_uni,
            "jaccard_active":      jaccard_active,
            "tanimoto":            tanimoto,
        })

    # [HM] Per-layer combined sum + diff heatmap (script 14 と同形式)
    max_over_10 = torch.maximum(
        clean_feats.max(dim=0).values,
        corrupt_feats.max(dim=0).values,
    )
    pool_k = min(MAX_FEATURES_FOR_HEATMAP_POOL, int(max_over_10.numel()))
    _, top_indices = torch.topk(max_over_10, k=pool_k)
    selected_feature_ids = top_indices.tolist()

    matrix_rows: list[list[float]] = []
    for prompt_type in ("clean", "corrupt"):
        feats_p = features_by_prompt[prompt_type]
        for i in range(feats_p.shape[0]):
            matrix_rows.append([float(feats_p[i, fid].item()) for fid in selected_feature_ids])
    matrix = np.array(matrix_rows, dtype=np.float32)

    matrix_csv = outputs_dir / f"prelim_qwen3_4b_transcoder_layer{layer_idx}_feature_matrix.csv"
    with matrix_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_label"] + [f"f{fid}" for fid in selected_feature_ids])
        row_idx = 0
        for prompt_type in ("clean", "corrupt"):
            feats_p = features_by_prompt[prompt_type]
            for i in range(feats_p.shape[0]):
                lbl = f"{prompt_type[0]}:{i}:{prompt_tokens[prompt_type][i]['piece_repr']}"
                writer.writerow([lbl] + matrix_rows[row_idx])
                row_idx += 1

    display_n   = min(MAX_FEATURES_FOR_HEATMAP, matrix.shape[1])
    display_ids = selected_feature_ids[:display_n]
    mat_disp    = matrix[:, :display_n]
    clean_disp   = mat_disp[: clean_seq_len]
    corrupt_disp = mat_disp[clean_seq_len:]
    sum_mat  = clean_disp + corrupt_disp
    diff_mat = clean_disp - corrupt_disp

    combined_row_labels: list[str] = []
    nposLabel = min(clean_seq_len, corrupt_seq_len)
    for p in range(nposLabel):
        clean_tok = prompt_tokens["clean"][p]["piece_repr"]
        corrupt_tok = prompt_tokens["corrupt"][p]["piece_repr"]
        combined_row_labels.append(
            f"pos {p}: {clean_tok}" if clean_tok == corrupt_tok
            else f"pos {p}: {clean_tok} / {corrupt_tok}"
        )

    fig_w = max(8.0, 0.20 * display_n + 4.0)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 8.5))
    ax = axes[0]
    im0 = ax.imshow(sum_mat, aspect="auto", cmap="viridis", vmin=0)
    ax.set_xticks(range(display_n))
    ax.set_xticklabels([str(f) for f in display_ids], rotation=90, fontsize=7)
    ax.set_yticks(range(len(combined_row_labels)))
    ax.set_yticklabels(combined_row_labels, fontsize=9)
    ax.set_title("sum = clean + corrupt", fontsize=10)
    plt.colorbar(im0, ax=ax, label="clean + corrupt")
    ax = axes[1]
    vmax_diff = float(np.abs(diff_mat).max())
    if vmax_diff == 0.0:
        vmax_diff = 1.0
    im1 = ax.imshow(diff_mat, aspect="auto", cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff)
    ax.set_xticks(range(display_n))
    ax.set_xticklabels([str(f) for f in display_ids], rotation=90, fontsize=7)
    ax.set_yticks(range(len(combined_row_labels)))
    ax.set_yticklabels(combined_row_labels, fontsize=9)
    ax.set_xlabel("feature_id  (sort: max-over-10-cells desc)")
    ax.set_title("diff = clean − corrupt  (red: clean>corrupt, blue: corrupt>clean)", fontsize=10)
    plt.colorbar(im1, ax=ax, label="Δactivation")
    fig.suptitle(
        f"Qwen3-4B layer{layer_idx} transcoder — sum & diff heatmaps  "
        f"(top {display_n} features by max-over-10-cells)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    heatmap_png = outputs_dir / f"nb03_qwen3_4b_transcoder_layer{layer_idx}_feature_heatmap.png"
    fig.savefig(heatmap_png, dpi=140)
    plt.close(fig)

    # Differential analysis
    pos3_diff = clean_feats[3] - corrupt_feats[3]
    last_diff = clean_feats[clean_last_pos] - corrupt_feats[corrupt_last_pos]
    pos3_stats = compute_diff_stats(pos3_diff, TOP_K_DIFF)
    last_stats = compute_diff_stats(last_diff, TOP_K_DIFF)

    # Reconstruction check
    rmse_by_prompt: dict[str, float] = {}
    mean_cos_by_prompt: dict[str, float] = {}
    if W_dec is not None:
        try:
            for prompt_type in ("clean", "corrupt"):
                feats = features_by_prompt[prompt_type]
                if dec_orientation == "features_x_out":
                    recon = feats @ W_dec
                else:
                    recon = feats @ W_dec.T
                if b_dec is not None:
                    recon = recon + b_dec
                target = mlp_out_by_layer[layer_idx][prompt_type][0].to(torch.float32)
                diff = recon - target
                rmse_by_prompt[prompt_type] = float((diff ** 2).mean().sqrt().item())
                cos_vals = []
                for i in range(target.shape[0]):
                    r_vec = recon[i]; t_vec = target[i]
                    denom = (r_vec.norm() * t_vec.norm()).clamp_min(1e-12)
                    cos_vals.append(float((r_vec @ t_vec / denom).item()))
                mean_cos_by_prompt[prompt_type] = float(np.mean(cos_vals))
        except Exception as e:
            print(f"  [warning] layer {layer_idx} reconstruction failed: {e}")
            traceback.print_exc()

    row = {
        "layer_idx":                              layer_idx,
        "input_hidden_index":                     layer_idx,
        "output_hidden_index":                    layer_idx + 1,
        "d_model":                                int(d_model),
        "d_feature":                              int(d_feature),
        "active_fraction_clean_mean":             active_frac_clean,
        "active_fraction_corrupt_mean":           active_frac_corrupt,
        "active_count_clean_mean":                active_count_clean_mean,
        "active_count_corrupt_mean":              active_count_corrupt_mean,
        "pos3_max_abs_diff":                      pos3_stats["max_abs_diff"],
        "pos3_l2_diff":                           pos3_stats["l2_diff"],
        "pos3_mean_abs_diff":                     pos3_stats["mean_abs_diff"],
        "pos3_top20_abs_diff_sum":                pos3_stats["top_k_abs_diff_sum"],
        "pos3_top20_abs_diff_mean":               pos3_stats["top_k_abs_diff_mean"],
        "pos3_top_clean_gt_corrupt_feature":      pos3_stats["top_clean_gt_corrupt_feature"],
        "pos3_top_clean_gt_corrupt_diff":         pos3_stats["top_clean_gt_corrupt_diff"],
        "pos3_top_corrupt_gt_clean_feature":      pos3_stats["top_corrupt_gt_clean_feature"],
        "pos3_top_corrupt_gt_clean_diff":         pos3_stats["top_corrupt_gt_clean_diff"],
        "last_max_abs_diff":                      last_stats["max_abs_diff"],
        "last_l2_diff":                           last_stats["l2_diff"],
        "last_mean_abs_diff":                     last_stats["mean_abs_diff"],
        "last_top20_abs_diff_sum":                last_stats["top_k_abs_diff_sum"],
        "last_top20_abs_diff_mean":               last_stats["top_k_abs_diff_mean"],
        "last_top_clean_gt_corrupt_feature":      last_stats["top_clean_gt_corrupt_feature"],
        "last_top_clean_gt_corrupt_diff":         last_stats["top_clean_gt_corrupt_diff"],
        "last_top_corrupt_gt_clean_feature":      last_stats["top_corrupt_gt_clean_feature"],
        "last_top_corrupt_gt_clean_diff":         last_stats["top_corrupt_gt_clean_diff"],
        "reconstruction_rmse_clean":              rmse_by_prompt.get("clean"),
        "reconstruction_rmse_corrupt":            rmse_by_prompt.get("corrupt"),
        "reconstruction_mean_cos_clean":          mean_cos_by_prompt.get("clean"),
        "reconstruction_mean_cos_corrupt":        mean_cos_by_prompt.get("corrupt"),
    }
    aggregate_rows.append(row)

    rmse_c   = rmse_by_prompt.get("clean")
    cos_c    = mean_cos_by_prompt.get("clean")
    rmse_s   = f"{rmse_c:.3f}" if rmse_c is not None else "  -  "
    cos_s    = f"{cos_c:.3f}"  if cos_c  is not None else "  -  "
    print(
        f"  layer {layer_idx:2d}  "
        f"active={active_frac_clean:.4f}/{active_frac_corrupt:.4f}  "
        f"pos3_max={pos3_stats['max_abs_diff']:7.3f}  "
        f"last_max={last_stats['max_abs_diff']:7.3f}  "
        f"rmse(c)={rmse_s}  cos(c)={cos_s}"
    )

    # Free per-layer tensors before next iteration
    del W_enc, b_enc, W_dec, b_dec
    del features_by_prompt, clean_feats, corrupt_feats
    del active_mask_clean, active_mask_corrupt
    del pos3_diff, last_diff
    del mlp_in_by_layer[layer_idx], mlp_out_by_layer[layer_idx]
    gc.collect()


# ── [6] Aggregate CSV ──────────────────────────────────────────────────────────
print("\n[6] Saving aggregate CSV / JSON")
agg_csv = outputs_dir / "prelim_qwen3_4b_transcoder_layer_sweep_summary.csv"
agg_fields = [
    "layer_idx", "input_hidden_index", "output_hidden_index",
    "d_model", "d_feature",
    "active_fraction_clean_mean", "active_fraction_corrupt_mean",
    "active_count_clean_mean", "active_count_corrupt_mean",
    "pos3_max_abs_diff", "pos3_l2_diff", "pos3_mean_abs_diff",
    "pos3_top20_abs_diff_sum", "pos3_top20_abs_diff_mean",
    "pos3_top_clean_gt_corrupt_feature", "pos3_top_clean_gt_corrupt_diff",
    "pos3_top_corrupt_gt_clean_feature", "pos3_top_corrupt_gt_clean_diff",
    "last_max_abs_diff", "last_l2_diff", "last_mean_abs_diff",
    "last_top20_abs_diff_sum", "last_top20_abs_diff_mean",
    "last_top_clean_gt_corrupt_feature", "last_top_clean_gt_corrupt_diff",
    "last_top_corrupt_gt_clean_feature", "last_top_corrupt_gt_clean_diff",
    "reconstruction_rmse_clean", "reconstruction_rmse_corrupt",
    "reconstruction_mean_cos_clean", "reconstruction_mean_cos_corrupt",
]
with agg_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=agg_fields)
    writer.writeheader()
    writer.writerows(aggregate_rows)
print(f"  saved: {agg_csv}  ({len(aggregate_rows)} rows)")

agg_json = outputs_dir / "prelim_qwen3_4b_transcoder_layer_sweep_summary.json"
with agg_json.open("w", encoding="utf-8") as f:
    json.dump({
        "model_id":            model_id,
        "transcoder_repo":     TRANSCODER_REPO,
        "layer_idxs":          LAYER_IDXS,
        "num_layers":          len(aggregate_rows),
        "clean_prompt":        CLEAN_PROMPT,
        "corrupt_prompt":      CORRUPT_PROMPT,
        "clean_top1_piece":    clean_top1,
        "corrupt_top1_piece":  corrupt_top1,
        "device":              device,
        "dtype":               str(dtype),
        "activation_function": activation_fn_used,
        "feature_input_hook_from_yaml":  feature_input_hook,
        "feature_output_hook_from_yaml": feature_output_hook,
        "top_k_diff":          TOP_K_DIFF,
        "note_on_indexing": (
            "hidden_states[0] = embed_tokens output; hidden_states[j+1] = block j output. "
            "layer_idx = j means the transcoder targets block j's MLP "
            "(input = post_attention_layernorm(residual_after_attn), output = MLP output before residual add)."
        ),
        "note_on_feature_ids": (
            "feature_id is per-layer; do NOT compare ids across layers. "
            "Across-layer comparison uses diff magnitudes, active fractions, and reconstruction quality."
        ),
        "rows": aggregate_rows,
    }, f, indent=2, ensure_ascii=False)
print(f"  saved: {agg_json}")

# ── [7] Plots ──────────────────────────────────────────────────────────────────
print("\n[7] Generating plots")

xs = [r["layer_idx"] for r in aggregate_rows]


def _add_reference_line(ax, label="layer 24 (note02 reference)"):
    ax.axvline(LAYER_REFERENCE, color="gray", linestyle="--", alpha=0.6, linewidth=1.0)
    ax.text(
        LAYER_REFERENCE, ax.get_ylim()[1], label,
        rotation=90, va="top", ha="right", fontsize=7, color="gray",
    )


# A. Per-(layer, position) metrics CSV
pm_csv = outputs_dir / "prelim_qwen3_4b_transcoder_layer_sweep_position_metrics.csv"
pm_fields = [
    "layer_idx", "position", "clean_token", "corrupt_token",
    "max_delta_pos", "max_delta_neg", "max_abs_delta",
    "max_sum", "max_single",
    "l1_abs_delta", "l2_delta",
    "l2_sum", "l2_max_single",
    "l2_clean", "l2_corrupt",
    "cos_clean_corrupt",
    "active_clean", "active_corrupt", "active_intersection", "active_union",
    "jaccard_active", "tanimoto",
]
with pm_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=pm_fields)
    writer.writeheader()
    writer.writerows(position_metric_rows)
print(f"  saved: {pm_csv}  ({len(position_metric_rows)} rows)")

# A'. 4 line plots from per-(layer, position) metrics — script 14 と同じ 4 種
xs_layers = sorted(set(r["layer_idx"] for r in position_metric_rows))
n_positions = max(r["position"] for r in position_metric_rows) + 1

position_colors = {
    p: "tab:gray" if p < 3 else ("tab:red" if p == 3 else "tab:blue")
    for p in range(n_positions)
}
position_linestyles = {p: ":" if p < 3 else "-" for p in range(n_positions)}
position_alphas     = {p: 0.5 if p < 3 else 1.0 for p in range(n_positions)}
position_markers    = {
    p: "o" if p == 3 else ("s" if p == 4 else "x")
    for p in range(n_positions)
}

def _series_for(metric: str, position: int) -> list[float]:
    by_layer = {r["layer_idx"]: r[metric] for r in position_metric_rows
                if r["position"] == position}
    return [by_layer[ell] for ell in xs_layers]

def _position_legend_label(p: int) -> str:
    clean_tok = next(r["clean_token"] for r in position_metric_rows if r["position"] == p)
    corrupt_tok = next(r["corrupt_token"] for r in position_metric_rows if r["position"] == p)
    if clean_tok == corrupt_tok:
        return f"pos {p}: {clean_tok}"
    return f"pos {p}: {clean_tok} / {corrupt_tok}"

def render_layer_metric(metric_key: str, title: str, ylabel: str, out_filename: str) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    for p in range(n_positions):
        ys = _series_for(metric_key, p)
        ax.plot(
            xs_layers, ys,
            color=position_colors[p],
            linestyle=position_linestyles[p],
            alpha=position_alphas[p],
            marker=position_markers[p],
            markersize=5,
            linewidth=1.4,
            label=_position_legend_label(p),
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("layer_idx")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(xs_layers[::2])
    ax.grid(True, alpha=0.3)
    _add_reference_line(ax)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_path = outputs_dir / out_filename
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved: {out_path}")

render_layer_metric(
    "max_abs_delta",
    "max ⱼ |cleanⱼ − corruptⱼ|  (outlier-driven discrimination)",
    "max |Δactivation|",
    "nb03_qwen3_4b_transcoder_layer_sweep_max_abs_delta.png",
)
render_layer_metric(
    "l2_delta",
    "‖clean − corrupt‖₂  (total L2 discrimination)",
    "L2 of Δactivation",
    "nb03_qwen3_4b_transcoder_layer_sweep_l2_delta.png",
)
render_layer_metric(
    "tanimoto",
    "Tanimoto(clean, corrupt)  (連続 Jaccard、非負 vector 用、閾値不要)",
    "∑ⱼ min(cleanⱼ, corruptⱼ) / ∑ⱼ max(cleanⱼ, corruptⱼ)",
    "nb03_qwen3_4b_transcoder_layer_sweep_tanimoto.png",
)
render_layer_metric(
    "max_single",
    "max ⱼ max(cleanⱼ, corruptⱼ)  (strongest single activation)",
    "max single activation",
    "nb03_qwen3_4b_transcoder_layer_sweep_max_single.png",
)

# Variables used in subsequent existing plots
y_active_clean   = [r["active_fraction_clean_mean"]   for r in aggregate_rows]
y_active_corrupt = [r["active_fraction_corrupt_mean"] for r in aggregate_rows]

# B. active fraction plot
y_active_clean   = [r["active_fraction_clean_mean"]   for r in aggregate_rows]
y_active_corrupt = [r["active_fraction_corrupt_mean"] for r in aggregate_rows]

fig, ax = plt.subplots(figsize=(10.5, 4.5))
ax.plot(xs, y_active_clean,   marker="o", color="tab:green",  label="clean")
ax.plot(xs, y_active_corrupt, marker="s", color="tab:orange", label="corrupt")
ax.set_xticks(xs[::2])
ax.set_xlabel("layer_idx")
ax.set_ylabel("mean active fraction  (= P(feature > 0))")
ax.set_title("Qwen3-4B transcoder mean active feature fraction across layers")
ax.grid(True, alpha=0.3)
_add_reference_line(ax)
ax.legend(loc="best")
fig.tight_layout()
active_frac_png = outputs_dir / "nb03_qwen3_4b_transcoder_layer_sweep_active_fraction.png"
fig.savefig(active_frac_png, dpi=140)
plt.close(fig)
print(f"  saved: {active_frac_png}")

# C. reconstruction plot (RMSE on top, mean cosine on bottom)
y_rmse_clean   = [r["reconstruction_rmse_clean"]      for r in aggregate_rows]
y_rmse_corrupt = [r["reconstruction_rmse_corrupt"]    for r in aggregate_rows]
y_cos_clean    = [r["reconstruction_mean_cos_clean"]  for r in aggregate_rows]
y_cos_corrupt  = [r["reconstruction_mean_cos_corrupt"] for r in aggregate_rows]

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
ax_top.plot(xs, y_rmse_clean,   marker="o", color="tab:green",  label="clean")
ax_top.plot(xs, y_rmse_corrupt, marker="s", color="tab:orange", label="corrupt")
ax_top.set_ylabel("reconstruction RMSE")
ax_top.set_title("Qwen3-4B transcoder reconstruction (target = MLP output)")
ax_top.grid(True, alpha=0.3)
_add_reference_line(ax_top)
ax_top.legend(loc="best")

ax_bot.plot(xs, y_cos_clean,   marker="o", color="tab:green",  label="clean")
ax_bot.plot(xs, y_cos_corrupt, marker="s", color="tab:orange", label="corrupt")
ax_bot.set_ylabel("reconstruction mean cosine")
ax_bot.set_xlabel("layer_idx")
ax_bot.grid(True, alpha=0.3)
ax_bot.set_xticks(xs[::2])
_add_reference_line(ax_bot)
ax_bot.legend(loc="best")
fig.tight_layout()
reconstruction_png = outputs_dir / "nb03_qwen3_4b_transcoder_layer_sweep_reconstruction.png"
fig.savefig(reconstruction_png, dpi=140)
plt.close(fig)
print(f"  saved: {reconstruction_png}")

print("\nDone.")
