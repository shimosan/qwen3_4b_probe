# mwhanna/qwen3-4b-transcoders の layer 23, 24, 25 を取得し、
# Qwen3-4B の各 layer の MLP input に対して transcoder feature activation を計算する smoke test。
# - 巨大な repo 全体は download しない。layer_{23,24,25}.safetensors と config.yaml / wandb-config.yaml のみ取得する。
# - transcoder weights は CPU 上で扱い、1 layer ずつ処理して終わったら del + gc.collect() する。
# - Qwen3-4B model は 1 回だけ load し、全対象 layer の mlp に同時に hook を付けて forward は 1 回で済ませる。
# - 各 layer ごとに top-k features / heatmap / bar plot / 差分解析を出力し、
#   最後に aggregate summary CSV/JSON と diff strength plot を作る。
# 出力:
#   outputs/prelim_qwen3_4b_transcoder_layer{23,24,25}_*.{json,csv}
#   outputs/nb03_qwen3_4b_transcoder_layer{23,24,25}_*.png
#   outputs/prelim_qwen3_4b_transcoder_layers23_24_25_summary.{csv,json}
#   outputs/nb03_qwen3_4b_transcoder_layers23_24_25_diff_strength.png
#
# hidden_states インデックス対応:
#   hidden_states[i] = block i-1 の output (i = 1..K)
#   ただし hidden_states[0] は embed_tokens output。
#   したがって:
#     hidden_states[24] = block 23 output
#     hidden_states[25] = block 24 output
#     hidden_states[26] = block 25 output
#
#   layer 23 transcoder: hidden_states[23] → block 23 → hidden_states[24] の MLP 部分を見る
#   layer 24 transcoder: hidden_states[24] → block 24 → hidden_states[25] の MLP 部分を見る (note02 k=24→25 の本命)
#   layer 25 transcoder: hidden_states[25] → block 25 → hidden_states[26] の MLP 部分を見る
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
LAYER_IDXS = [23, 24, 25]
TOP_K_FEATURES = 20
MAX_FEATURES_FOR_HEATMAP = 60          # heatmap で表示する列数
MAX_FEATURES_FOR_HEATMAP_POOL = 300    # CSV に保存する pool 列数（max-over-10 で top-N）
TOP_K_DIFF = 20

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
print(f"layer_idxs      : {LAYER_IDXS}")
print(f"device          : {device}")
print(f"dtype           : {dtype}")
assert model_id == "Qwen/Qwen3-4B", f"unexpected model_id: {model_id}"

# ── [1] Download shared YAML configs (no snapshot_download) ────────────────────
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
print("\n[2] Tokenizer & token tables (raw, no chat template)")
tokenizer = AutoTokenizer.from_pretrained(model_id)

clean_inputs   = tokenizer(CLEAN_PROMPT,   return_tensors="pt").to(device)
corrupt_inputs = tokenizer(CORRUPT_PROMPT, return_tensors="pt").to(device)
clean_seq_len   = clean_inputs["input_ids"].shape[1]
corrupt_seq_len = corrupt_inputs["input_ids"].shape[1]
clean_last_pos   = clean_seq_len   - 1
corrupt_last_pos = corrupt_seq_len - 1


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
for li in LAYER_IDXS:
    assert 0 <= li < K, f"LAYER_IDX {li} out of range [0,{K})"
assert hidden_size == D_MODEL_EXPECTED, (
    f"hidden_size {hidden_size} != expected d_model {D_MODEL_EXPECTED}"
)

# ── [4] Single forward with hooks on all target MLPs ───────────────────────────
print(f"\n[4] Single forward with hooks on model.model.layers[{LAYER_IDXS}].mlp")
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

clean_top1 = top5_by_prompt["clean"][0] if top5_by_prompt["clean"] else None
corrupt_top1 = top5_by_prompt["corrupt"][0] if top5_by_prompt["corrupt"] else None
print(f"  sanity: clean top1   = {clean_top1!r}   (expect ' Tokyo')")
print(f"  sanity: corrupt top1 = {corrupt_top1!r}   (expect ' Paris')")

for li in LAYER_IDXS:
    s_in  = tuple(mlp_in_by_layer[li]["clean"].shape)
    s_out = tuple(mlp_out_by_layer[li]["clean"].shape)
    print(f"  layer{li:2d}  MLP in/out shape (clean): {s_in} / {s_out}")

# Free model now: we only need stored mlp_in / mlp_out from here on.
del out, logits_last, top_vals, top_ids
del model
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ── Helper functions used inside the per-layer loop ────────────────────────────
def pick_key(pool, candidates):
    for c in candidates:
        if c in pool:
            return c
    return None


def infer_orientations(W_enc: torch.Tensor, W_dec_or_none, d_model: int):
    """Return (d_model, d_feature, enc_orientation, dec_orientation, W_dec_normalized)."""
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
            print(
                f"  [warning] W_dec shape {tuple(W_dec_or_none.shape)} does not match; "
                f"reconstruction will be skipped."
            )
            W_dec_or_none = None
    return dm, df, enc_orientation, dec_orientation, W_dec_or_none


def encode_features(
    X: torch.Tensor,
    W_enc: torch.Tensor,
    b_enc,
    enc_orientation: str,
    act_fn: str,
) -> torch.Tensor:
    """X: [seq_len, d_model] (float32, cpu). Returns features [seq_len, d_feature]."""
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
    print(f"  [warning] unknown act_fn {act_fn!r}, falling back to ReLU")
    return torch.relu(pre)


# ── [5] Per-layer loop ─────────────────────────────────────────────────────────
print("\n[5] Per-layer processing loop")

aggregate_rows: list[dict] = []
max_abs_diff_by_layer: dict[int, dict[str, float]] = {}
position_metric_rows: list[dict] = []  # per (layer, position) metrics, see [PM] section below

for layer_idx in LAYER_IDXS:
    print(f"\n  ── layer {layer_idx} ──")

    # 5-1. Download safetensors for this layer only
    try:
        layer_path = hf_hub_download(
            repo_id=TRANSCODER_REPO,
            filename=f"layer_{layer_idx}.safetensors",
        )
    except Exception as e:
        print(f"  [error] layer_{layer_idx}.safetensors download failed: {e}")
        print("  hint: pip install -U huggingface_hub hf_xet safetensors pyyaml")
        raise

    layer_path_p = Path(layer_path)
    layer_size = layer_path_p.stat().st_size
    print(f"    file: {layer_path}")
    print(f"    size: {layer_size:,} bytes ({layer_size / 1e6:.2f} MB)")

    # 5-2. Load safetensors and inspect
    tensors = load_safetensors(str(layer_path), device="cpu")
    keys_info = []
    for k, t in tensors.items():
        keys_info.append({
            "key":   k,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "numel": int(t.numel()),
        })
    keys_info.sort(key=lambda r: r["key"])
    for r in keys_info:
        print(f"    {r['key']:32s}  shape={r['shape']!s:24s}  dtype={r['dtype']}")

    keys_json = outputs_dir / f"prelim_qwen3_4b_transcoder_layer{layer_idx}_keys.json"
    with keys_json.open("w", encoding="utf-8") as f:
        json.dump({"layer_path": str(layer_path), "keys": keys_info}, f, indent=2)
    print(f"    saved: {keys_json}")

    # 5-3. Pick enc / dec keys
    key_pool = set(tensors.keys())
    enc_w_key = pick_key(key_pool, ["W_enc", "encoder.weight", "encoder.W", "enc.weight"])
    dec_w_key = pick_key(key_pool, ["W_dec", "decoder.weight", "decoder.W", "dec.weight"])
    b_enc_key = pick_key(key_pool, ["b_enc", "encoder.bias", "enc.bias"])
    b_dec_key = pick_key(key_pool, ["b_dec", "decoder.bias", "dec.bias", "b_pre"])

    if enc_w_key is None:
        print(f"    [error] no encoder weight key in layer {layer_idx}; skipping.")
        del tensors
        gc.collect()
        continue

    W_enc = tensors[enc_w_key].to(torch.float32)
    b_enc = tensors[b_enc_key].to(torch.float32) if b_enc_key else None
    W_dec = tensors[dec_w_key].to(torch.float32) if dec_w_key else None
    b_dec = tensors[b_dec_key].to(torch.float32) if b_dec_key else None

    # Free original tensors dict (we already cast what we need)
    del tensors
    gc.collect()

    d_model, d_feature, enc_orientation, dec_orientation, W_dec = infer_orientations(
        W_enc, W_dec, D_MODEL_EXPECTED
    )
    print(f"    d_model   = {d_model}")
    print(f"    d_feature = {d_feature}")
    print(f"    encoder orientation = {enc_orientation}")
    print(f"    decoder orientation = {dec_orientation}")

    # 5-4. Compute features for clean / corrupt from stored mlp_in
    features_by_prompt: dict[str, torch.Tensor] = {}
    for prompt_type in ("clean", "corrupt"):
        X = mlp_in_by_layer[layer_idx][prompt_type][0].to(torch.float32)
        feats = encode_features(X, W_enc, b_enc, enc_orientation, activation_fn_used)
        features_by_prompt[prompt_type] = feats
    active_frac_clean   = float((features_by_prompt["clean"] > 0).float().mean().item())
    active_frac_corrupt = float((features_by_prompt["corrupt"] > 0).float().mean().item())
    print(f"    feature shape: {tuple(features_by_prompt['clean'].shape)}   "
          f"active_frac clean/corrupt = {active_frac_clean:.4f} / {active_frac_corrupt:.4f}")

    # [PM] Per-(layer, position) summary metrics
    # 各 position p について clean / corrupt の feature ベクトル c, k を:
    #   discrimination: delta = c - k          (max-based & L2/L1)
    #   joint strength: sigma = c + k, m = max(c, k)
    #   direction:      cos(c, k)
    #   density:        active counts
    # を計算し、position_metric_rows に append (after the loop で CSV と plot に集約)
    clean_t   = features_by_prompt["clean"].to(torch.float32)
    corrupt_t = features_by_prompt["corrupt"].to(torch.float32)
    n_pos = min(clean_t.shape[0], corrupt_t.shape[0])
    for p in range(n_pos):
        clean_vec   = clean_t[p]
        corrupt_vec = corrupt_t[p]
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

        # Tanimoto coefficient (連続 Jaccard、非負 vector 用)
        # T = sum_j min(c_j, k_j) / sum_j max(c_j, k_j)
        # threshold 不要、magnitude-aware、ReLU の noise floor に robust
        tanimoto_num = float(torch.minimum(clean_vec, corrupt_vec).sum().item())
        tanimoto_den = float(torch.maximum(clean_vec, corrupt_vec).sum().item())
        tanimoto     = tanimoto_num / tanimoto_den if tanimoto_den > 1e-12 else 1.0

        position_metric_rows.append({
            "layer_idx":           layer_idx,
            "position":            p,
            "clean_token":         prompt_tokens["clean"][p]["piece"],
            "corrupt_token":       prompt_tokens["corrupt"][p]["piece"],
            # discrimination max-based
            "max_delta_pos":       float(delta.max().item()),
            "max_delta_neg":       float((-delta).max().item()),
            "max_abs_delta":       float(delta.abs().max().item()),
            # joint strength max-based
            "max_sum":             float(sigma.max().item()),
            "max_single":          float(elem_max.max().item()),
            # discrimination L1 / L2
            "l1_abs_delta":        float(delta.abs().sum().item()),
            "l2_delta":            float(delta.norm(p=2).item()),
            # joint L2
            "l2_sum":              float(sigma.norm(p=2).item()),
            "l2_max_single":       float(elem_max.norm(p=2).item()),
            # per-prompt L2
            "l2_clean":            float(clean_vec.norm(p=2).item()),
            "l2_corrupt":          float(corrupt_vec.norm(p=2).item()),
            # direction
            "cos_clean_corrupt":   cos_clean_corrupt,
            # density (counts), Jaccard, Tanimoto
            "active_clean":        int(clean_active.sum().item()),
            "active_corrupt":      int(corrupt_active.sum().item()),
            "active_intersection": active_inter,
            "active_union":        active_uni,
            "jaccard_active":      jaccard_active,
            "tanimoto":            tanimoto,
        })

    # 5-5. Top-k features per position → CSV
    top_feature_rows: list[dict] = []
    for prompt_type in ("clean", "corrupt"):
        feats = features_by_prompt[prompt_type]
        for i, tok in enumerate(prompt_tokens[prompt_type]):
            vec = feats[i]
            k_take = min(TOP_K_FEATURES, vec.numel())
            top_vals, top_ids = torch.topk(vec, k=k_take)
            for rank, (fid, val) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), start=1):
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

    top_features_csv = outputs_dir / f"prelim_qwen3_4b_transcoder_layer{layer_idx}_top_features.csv"
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
    print(f"    saved: {top_features_csv}")

    # 5-6. Token × feature heatmap (per layer)
    # Selection: 全 d_feature の中で max-over-10-cells (5 clean + 5 corrupt position) 上位
    # CSV  には pool size (= MAX_FEATURES_FOR_HEATMAP_POOL, e.g. 300) を保存
    # 図は display 数 (= MAX_FEATURES_FOR_HEATMAP, e.g. 60) を sum+diff の combined で表示
    clean_feats_t   = features_by_prompt["clean"]    # torch [seq_len, d_feature]
    corrupt_feats_t = features_by_prompt["corrupt"]  # torch [seq_len, d_feature]
    max_over_10 = torch.maximum(
        clean_feats_t.max(dim=0).values,
        corrupt_feats_t.max(dim=0).values,
    )  # [d_feature]
    pool_k = min(MAX_FEATURES_FOR_HEATMAP_POOL, int(max_over_10.numel()))
    _, top_indices = torch.topk(max_over_10, k=pool_k)
    selected_feature_ids = top_indices.tolist()  # 既に max-over-10 降順

    row_labels: list[str] = []
    matrix_rows: list[list[float]] = []
    for prompt_type in ("clean", "corrupt"):
        feats = features_by_prompt[prompt_type]
        for i, tok in enumerate(prompt_tokens[prompt_type]):
            row_labels.append(f"{prompt_type[0]}:{i}:{tok['piece_repr']}")
            matrix_rows.append([float(feats[i, fid].item()) for fid in selected_feature_ids])

    matrix = np.array(matrix_rows, dtype=np.float32)  # [10, pool_k]

    matrix_csv = outputs_dir / f"prelim_qwen3_4b_transcoder_layer{layer_idx}_feature_matrix.csv"
    with matrix_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["token_label"] + [f"f{fid}" for fid in selected_feature_ids])
        for label, row in zip(row_labels, matrix_rows):
            writer.writerow([label] + row)
    print(f"    saved: {matrix_csv}  (pool={pool_k} features)")

    # Combined sum + diff heatmap (display top MAX_FEATURES_FOR_HEATMAP)
    display_n   = min(MAX_FEATURES_FOR_HEATMAP, matrix.shape[1])
    display_ids = selected_feature_ids[:display_n]
    mat_disp    = matrix[:, :display_n]                 # [10, display_n]
    clean_disp   = mat_disp[: clean_seq_len]            # [5, display_n]
    corrupt_disp = mat_disp[clean_seq_len:]             # [5, display_n]
    sum_mat  = clean_disp + corrupt_disp                # [5, display_n]
    diff_mat = clean_disp - corrupt_disp                # [5, display_n]

    # Row labels: "pos N: ctok" or "pos N: ctok / ktok"
    combined_row_labels: list[str] = []
    n_pos = min(clean_seq_len, corrupt_seq_len)
    for p in range(n_pos):
        c = prompt_tokens["clean"][p]["piece_repr"]
        k = prompt_tokens["corrupt"][p]["piece_repr"]
        combined_row_labels.append(f"pos {p}: {c}" if c == k else f"pos {p}: {c} / {k}")

    fig_w = max(8.0, 0.20 * display_n + 4.0)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 8.5))

    # upper: sum (viridis)
    ax = axes[0]
    im0 = ax.imshow(sum_mat, aspect="auto", cmap="viridis", vmin=0)
    ax.set_xticks(range(display_n))
    ax.set_xticklabels([str(f) for f in display_ids], rotation=90, fontsize=7)
    ax.set_yticks(range(len(combined_row_labels)))
    ax.set_yticklabels(combined_row_labels, fontsize=9)
    ax.set_title("sum = clean + corrupt", fontsize=10)
    plt.colorbar(im0, ax=ax, label="clean + corrupt")

    # lower: diff (RdBu_r diverging)
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
    print(f"    saved: {heatmap_png}  (display={display_n} features)")

    # 5-8. Reconstruction check (mlp_out target)
    recon_metrics_rows: list[dict] = []
    reconstruction_check_available = False
    rmse_by_prompt: dict[str, float] = {}
    mean_cos_by_prompt: dict[str, float] = {}
    if W_dec is not None:
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
                target = mlp_out_by_layer[layer_idx][prompt_type][0].to(torch.float32)
                diff = recon - target
                mse  = float((diff ** 2).mean().item())
                rmse = float(mse ** 0.5)
                rmse_by_prompt[prompt_type] = rmse
                cos_vals = []
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
                print(f"    recon {prompt_type:7s}  rmse={rmse:.4f}  "
                      f"mean_cos={mean_cos_by_prompt[prompt_type]:.4f}")
        except Exception as e:
            print(f"    [warning] reconstruction failed: {e}")
            traceback.print_exc()
            reconstruction_check_available = False

    if recon_metrics_rows:
        recon_csv = outputs_dir / f"prelim_qwen3_4b_transcoder_layer{layer_idx}_reconstruction_metrics.csv"
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
        print(f"    saved: {recon_csv}")
    else:
        recon_csv = None

    # 5-9. Differential analysis (clean − corrupt)
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
    diff_summary: dict[str, dict] = {}

    for cmp in comparisons:
        clean_vec   = clean_feats[cmp["clean_pos"]]
        corrupt_vec = corrupt_feats[cmp["corrupt_pos"]]
        diff = clean_vec - corrupt_vec
        diffs_by_comparison[cmp["name"]] = diff

        top_pos_vals, top_pos_ids = torch.topk(diff, k=TOP_K_DIFF)
        top_neg_vals, top_neg_ids = torch.topk(-diff, k=TOP_K_DIFF)
        selected_ids = set(top_pos_ids.tolist()) | set(top_neg_ids.tolist())

        for fid in sorted(selected_ids):
            ca = float(clean_vec[fid].item())
            ka = float(corrupt_vec[fid].item())
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

        max_abs_diff = float(diff.abs().max().item())
        top_pos_fid  = int(top_pos_ids[0].item())
        top_pos_val  = float(top_pos_vals[0].item())
        top_neg_fid  = int(top_neg_ids[0].item())
        top_neg_val  = float(top_neg_vals[0].item())
        diff_summary[cmp["name"]] = {
            "max_abs_diff":          max_abs_diff,
            "top_clean_gt_corrupt":  {"feature_id": top_pos_fid, "diff":  top_pos_val},
            "top_corrupt_gt_clean":  {"feature_id": top_neg_fid, "diff": -top_neg_val},
        }
        print(f"    [{cmp['name']}]  max|Δ|={max_abs_diff:.4f}  "
              f"top+ f{top_pos_fid}=+{top_pos_val:.3f}  "
              f"top- f{top_neg_fid}=-{top_neg_val:.3f}")

    max_abs_diff_by_layer[layer_idx] = {
        "pos3":  diff_summary["pos3_japan_minus_france"]["max_abs_diff"],
        "last":  diff_summary["last_clean_minus_corrupt"]["max_abs_diff"],
    }

    diff_rows.sort(key=lambda r: (r["comparison"], -r["abs_diff"]))
    feature_diffs_csv = outputs_dir / f"prelim_qwen3_4b_transcoder_layer{layer_idx}_feature_diffs.csv"
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
    print(f"    saved: {feature_diffs_csv}")

    # 5-10. Per-layer summary JSON
    summary = {
        "model_id":            model_id,
        "transcoder_repo":     TRANSCODER_REPO,
        "layer_idx":           layer_idx,
        "input_hidden_index":  layer_idx,
        "output_hidden_index": layer_idx + 1,
        "device":              device,
        "dtype":               str(dtype),
        "clean_prompt":        CLEAN_PROMPT,
        "corrupt_prompt":      CORRUPT_PROMPT,
        "clean_answer":        CLEAN_ANSWER,
        "corrupt_answer":      CORRUPT_ANSWER,
        "clean_seq_len":       clean_seq_len,
        "corrupt_seq_len":     corrupt_seq_len,
        "clean_top1_piece":    clean_top1,
        "corrupt_top1_piece":  corrupt_top1,
        "weight_file_path":    str(layer_path),
        "weight_file_size_bytes": layer_size,
        "safetensors_keys":    [r["key"] for r in keys_info],
        "inferred_encoder_key": enc_w_key,
        "inferred_decoder_key": dec_w_key,
        "inferred_b_enc_key":   b_enc_key,
        "inferred_b_dec_key":   b_dec_key,
        "inferred_d_model":     int(d_model),
        "inferred_d_feature":   int(d_feature),
        "config_d_feature_hint": config_d_feature_hint,
        "encoder_orientation":  enc_orientation,
        "decoder_orientation":  dec_orientation,
        "activation_function":  activation_fn_used,
        "act_fn_from_yaml":     act_fn_name,
        "feature_input_hook_from_yaml":  feature_input_hook,
        "feature_output_hook_from_yaml": feature_output_hook,
        "active_fraction_clean_mean":   active_frac_clean,
        "active_fraction_corrupt_mean": active_frac_corrupt,
        "top_k_features_per_position":  TOP_K_FEATURES,
        "heatmap_num_features":         len(selected_feature_ids),
        "reconstruction_check_available": reconstruction_check_available,
        "reconstruction_rmse_clean":    rmse_by_prompt.get("clean"),
        "reconstruction_rmse_corrupt":  rmse_by_prompt.get("corrupt"),
        "reconstruction_mean_cos_clean":   mean_cos_by_prompt.get("clean"),
        "reconstruction_mean_cos_corrupt": mean_cos_by_prompt.get("corrupt"),
        "diff_top_k_per_direction":     TOP_K_DIFF,
        "diff_comparisons":             [c["name"] for c in comparisons],
        "diff_summary":                 diff_summary,
        "output_files": [
            str(keys_json),
            str(top_features_csv),
            str(matrix_csv),
            str(heatmap_png),
            *([str(recon_csv)] if recon_csv else []),
            str(feature_diffs_csv),
        ],
    }
    summary_json = outputs_dir / f"prelim_qwen3_4b_transcoder_layer{layer_idx}_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"    saved: {summary_json}")

    # 5-11. Aggregate row
    pos3_sum = diff_summary["pos3_japan_minus_france"]
    last_sum = diff_summary["last_clean_minus_corrupt"]
    aggregate_rows.append({
        "layer_idx":                              layer_idx,
        "input_hidden_index":                     layer_idx,
        "output_hidden_index":                    layer_idx + 1,
        "clean_top1_piece":                       clean_top1,
        "corrupt_top1_piece":                     corrupt_top1,
        "d_feature":                              int(d_feature),
        "active_fraction_clean_mean":             active_frac_clean,
        "active_fraction_corrupt_mean":           active_frac_corrupt,
        "pos3_max_abs_diff":                      pos3_sum["max_abs_diff"],
        "pos3_top_clean_gt_corrupt_feature":      pos3_sum["top_clean_gt_corrupt"]["feature_id"],
        "pos3_top_clean_gt_corrupt_diff":         pos3_sum["top_clean_gt_corrupt"]["diff"],
        "pos3_top_corrupt_gt_clean_feature":      pos3_sum["top_corrupt_gt_clean"]["feature_id"],
        "pos3_top_corrupt_gt_clean_diff":         pos3_sum["top_corrupt_gt_clean"]["diff"],
        "last_max_abs_diff":                      last_sum["max_abs_diff"],
        "last_top_clean_gt_corrupt_feature":      last_sum["top_clean_gt_corrupt"]["feature_id"],
        "last_top_clean_gt_corrupt_diff":         last_sum["top_clean_gt_corrupt"]["diff"],
        "last_top_corrupt_gt_clean_feature":      last_sum["top_corrupt_gt_clean"]["feature_id"],
        "last_top_corrupt_gt_clean_diff":         last_sum["top_corrupt_gt_clean"]["diff"],
        "reconstruction_rmse_clean":              rmse_by_prompt.get("clean"),
        "reconstruction_rmse_corrupt":            rmse_by_prompt.get("corrupt"),
        "reconstruction_mean_cos_clean":          mean_cos_by_prompt.get("clean"),
        "reconstruction_mean_cos_corrupt":        mean_cos_by_prompt.get("corrupt"),
    })

    # 5-12. Free per-layer tensors before next iteration
    del W_enc, b_enc, W_dec, b_dec
    del features_by_prompt, clean_feats, corrupt_feats
    del diffs_by_comparison
    del mlp_in_by_layer[layer_idx], mlp_out_by_layer[layer_idx]
    gc.collect()


# ── [6] Aggregate summary (CSV + JSON) ─────────────────────────────────────────
print("\n[6] Aggregate summary across layers")
agg_csv = outputs_dir / "prelim_qwen3_4b_transcoder_layers23_24_25_summary.csv"
agg_fields = [
    "layer_idx", "input_hidden_index", "output_hidden_index",
    "clean_top1_piece", "corrupt_top1_piece",
    "d_feature",
    "active_fraction_clean_mean", "active_fraction_corrupt_mean",
    "pos3_max_abs_diff",
    "pos3_top_clean_gt_corrupt_feature", "pos3_top_clean_gt_corrupt_diff",
    "pos3_top_corrupt_gt_clean_feature", "pos3_top_corrupt_gt_clean_diff",
    "last_max_abs_diff",
    "last_top_clean_gt_corrupt_feature", "last_top_clean_gt_corrupt_diff",
    "last_top_corrupt_gt_clean_feature", "last_top_corrupt_gt_clean_diff",
    "reconstruction_rmse_clean", "reconstruction_rmse_corrupt",
    "reconstruction_mean_cos_clean", "reconstruction_mean_cos_corrupt",
]
with agg_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=agg_fields)
    writer.writeheader()
    writer.writerows(aggregate_rows)
print(f"  saved: {agg_csv}")

agg_json = outputs_dir / "prelim_qwen3_4b_transcoder_layers23_24_25_summary.json"
with agg_json.open("w", encoding="utf-8") as f:
    json.dump({
        "model_id":          model_id,
        "transcoder_repo":   TRANSCODER_REPO,
        "layer_idxs":        LAYER_IDXS,
        "clean_prompt":      CLEAN_PROMPT,
        "corrupt_prompt":    CORRUPT_PROMPT,
        "clean_top1_piece":  clean_top1,
        "corrupt_top1_piece": corrupt_top1,
        "device":            device,
        "dtype":             str(dtype),
        "activation_function": activation_fn_used,
        "note": (
            "feature_id is per-layer; do NOT compare ids across layers. "
            "Across-layer comparison uses diff magnitudes / active fractions."
        ),
        "rows": aggregate_rows,
    }, f, indent=2, ensure_ascii=False)
print(f"  saved: {agg_json}")

# ── [7] Per-(layer, position) metrics: CSV + multi-panel plot ─────────────────
print("\n[7] Per-(layer, position) metrics CSV + 9-panel plot")

# 7-1. Save CSV
pm_csv = outputs_dir / "prelim_qwen3_4b_transcoder_layers23_24_25_position_metrics.csv"
pm_fields = [
    "layer_idx", "position", "clean_token", "corrupt_token",
    "max_delta_pos", "max_delta_neg", "max_abs_delta",
    "max_sum", "max_single",
    "l1_abs_delta", "l2_delta",
    "l2_sum", "l2_max_single",
    "l2_clean", "l2_corrupt",
    "cos_clean_corrupt",
    "active_clean", "active_corrupt", "active_intersection", "active_union",
    "jaccard_active",
    "tanimoto",
]
with pm_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=pm_fields)
    writer.writeheader()
    writer.writerows(position_metric_rows)
print(f"  saved: {pm_csv}  ({len(position_metric_rows)} rows)")

# 7-2. 4 separate metric plots (それぞれ単独 figure)
xs_layers = sorted(set(r["layer_idx"] for r in position_metric_rows))
n_positions = max(r["position"] for r in position_metric_rows) + 1

# Position 配色: 0..2 はグレー (causal mask で同一)、3 = 赤、4 = 青
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
    """layer 順に並んだ values を返す。"""
    by_layer = {r["layer_idx"]: r[metric] for r in position_metric_rows
                if r["position"] == position}
    return [by_layer[ell] for ell in xs_layers]

def _position_legend_label(p: int) -> str:
    clean_tok = next(r["clean_token"] for r in position_metric_rows if r["position"] == p)
    corrupt_tok = next(r["corrupt_token"] for r in position_metric_rows if r["position"] == p)
    if clean_tok == corrupt_tok:
        return f"pos {p}: {clean_tok}"
    return f"pos {p}: {clean_tok} / {corrupt_tok}"

def render_pos34_metric(metric_key: str, title: str, ylabel: str, out_filename: str) -> None:
    """pos=3, 4 のみの 2 系列 line plot。pos 0..2 は causal mask で trivial なので省略。"""
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for p in [3, 4]:
        ys = _series_for(metric_key, p)
        ax.plot(
            xs_layers, ys,
            color=position_colors[p],
            marker=position_markers[p],
            markersize=7,
            linewidth=1.6,
            label=_position_legend_label(p),
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("layer_idx")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(xs_layers)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_path = outputs_dir / out_filename
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved: {out_path}")


def render_log_all_positions(metric_key: str, title: str, ylabel: str, out_filename: str) -> None:
    """全 5 position、縦軸 log scale の line plot。"""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for p in range(n_positions):
        ys = _series_for(metric_key, p)
        ax.plot(
            xs_layers, ys,
            color=position_colors[p],
            linestyle=position_linestyles[p],
            alpha=position_alphas[p],
            marker=position_markers[p],
            markersize=6,
            linewidth=1.4,
            label=_position_legend_label(p),
        )
    ax.set_yscale("log")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("layer_idx")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(xs_layers)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_path = outputs_dir / out_filename
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved: {out_path}")


# 4 discrimination/strength metrics — pos=3, 4 のみ表示
render_pos34_metric(
    "max_abs_delta",
    "max ⱼ |cleanⱼ − corruptⱼ|  (outlier-driven discrimination, pos=3, 4)",
    "max |Δactivation|",
    "nb03_qwen3_4b_transcoder_layers23_24_25_max_abs_delta.png",
)
render_pos34_metric(
    "l2_delta",
    "‖clean − corrupt‖₂  (total L2 discrimination, pos=3, 4)",
    "L2 of Δactivation",
    "nb03_qwen3_4b_transcoder_layers23_24_25_l2_delta.png",
)
render_pos34_metric(
    "tanimoto",
    "Tanimoto(clean, corrupt)  (連続 Jaccard、非負 vector 用、pos=3, 4)",
    "∑ⱼ min(cleanⱼ, corruptⱼ) / ∑ⱼ max(cleanⱼ, corruptⱼ)",
    "nb03_qwen3_4b_transcoder_layers23_24_25_tanimoto.png",
)
render_pos34_metric(
    "jaccard_active",
    "Jaccard(active sets, threshold > 0)  (binary set 一致度、pos=3, 4)",
    "|active∩| / |active∪|",
    "nb03_qwen3_4b_transcoder_layers23_24_25_jaccard.png",
)
render_pos34_metric(
    "max_single",
    "max ⱼ max(cleanⱼ, corruptⱼ)  (strongest single activation, pos=3, 4)",
    "max single activation",
    "nb03_qwen3_4b_transcoder_layers23_24_25_max_single.png",
)

# max_single log y、全 5 position 比較 — pos 0..2 と pos=3, 4 の magnitude 関係を見る
render_log_all_positions(
    "max_single",
    "max ⱼ max(cleanⱼ, corruptⱼ)  (全 5 position、縦軸 log)",
    "max single activation (log)",
    "nb03_qwen3_4b_transcoder_layers23_24_25_max_single_log.png",
)

# Reconstruction quality plot: 上段 RMSE log、下段 mean cosine linear
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
xs_agg = [r["layer_idx"] for r in aggregate_rows]
rmse_c = [r["reconstruction_rmse_clean"]      for r in aggregate_rows]
rmse_k = [r["reconstruction_rmse_corrupt"]    for r in aggregate_rows]
cos_c  = [r["reconstruction_mean_cos_clean"]  for r in aggregate_rows]
cos_k  = [r["reconstruction_mean_cos_corrupt"] for r in aggregate_rows]

ax_top.plot(xs_agg, rmse_c, marker="o", color="tab:green",  label="clean")
ax_top.plot(xs_agg, rmse_k, marker="s", color="tab:orange", label="corrupt")
ax_top.set_yscale("log")
ax_top.set_ylabel("reconstruction RMSE (log)")
ax_top.set_title("Reconstruction quality (RMSE log y + mean cosine linear)", fontsize=12)
ax_top.grid(True, alpha=0.3, which="both")
ax_top.legend(loc="best")

ax_bot.plot(xs_agg, cos_c, marker="o", color="tab:green",  label="clean")
ax_bot.plot(xs_agg, cos_k, marker="s", color="tab:orange", label="corrupt")
ax_bot.set_ylabel("reconstruction mean cosine (linear)")
ax_bot.set_xlabel("layer_idx")
ax_bot.set_xticks(xs_agg)
ax_bot.grid(True, alpha=0.3)
ax_bot.legend(loc="best")

fig.tight_layout()
recon_png = outputs_dir / "nb03_qwen3_4b_transcoder_layers23_24_25_reconstruction.png"
fig.savefig(recon_png, dpi=140)
plt.close(fig)
print(f"  saved: {recon_png}")

print("\nDone.")
