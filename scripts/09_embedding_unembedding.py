from __future__ import annotations

import csv
import json
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_config, resolve_outputs_dir

# ── Config ────────────────────────────────────────────────────────────────────
cfg = load_config()
model_id = cfg["model_id"]
prompt = cfg["default_prompt"]
attn_impl = cfg["attn_implementation"]
BACKGROUND_SIZE = 300

outputs_dir = resolve_outputs_dir()

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"model_id : {model_id}")
print(f"device   : {device}")
print(f"dtype    : {dtype}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(model_id)
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt").to(device)
seq_len = inputs["input_ids"].shape[1]
print(f"input length: {seq_len} tokens")

# ── Model ─────────────────────────────────────────────────────────────────────
print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=dtype,
    attn_implementation=attn_impl,
)
model.to(device).eval()  # type: ignore[union-attr]

# ── [1] W_E and W_U ──────────────────────────────────────────────────────────
print("\n[1] Embedding and unembedding matrices")
W_E = model.model.embed_tokens.weight   # [vocab_size, hidden_size]
W_U = model.lm_head.weight              # [vocab_size, hidden_size]

print(f"  W_E : shape={tuple(W_E.shape)}, dtype={W_E.dtype}, device={W_E.device}")
print(f"  W_U : shape={tuple(W_U.shape)}, dtype={W_U.dtype}, device={W_U.device}")
print(f"  tie_word_embeddings: {model.config.tie_word_embeddings}")

same_data_ptr = W_E.data_ptr() == W_U.data_ptr()
print(f"  data_ptr equal: {same_data_ptr}")

# Diff calculation — avoid creating a full [vocab_size, hidden_size] diff tensor.
# If data_ptr is equal, they are the same tensor: diff is zero by definition.
# Otherwise compute chunk-wise to limit peak memory.
CHUNK = 4096
if same_data_ptr:
    max_abs_diff, mean_abs_diff, are_close = 0.0, 0.0, True
else:
    _max_d, _sum_d, _n = 0.0, 0.0, 0
    with torch.no_grad():
        for _start in range(0, W_E.shape[0], CHUNK):
            _e = W_E[_start:_start + CHUNK].float()
            _u = W_U[_start:_start + CHUNK].float()
            _d = (_e - _u).abs()
            _max_d = max(_max_d, _d.max().item())
            _sum_d += _d.sum().item()
            _n += _d.numel()
    max_abs_diff = _max_d
    mean_abs_diff = _sum_d / _n if _n > 0 else 0.0
    are_close = max_abs_diff <= 1e-5

print(f"  max_abs_diff  : {max_abs_diff:.4e}")
print(f"  mean_abs_diff : {mean_abs_diff:.4e}")
print(f"  torch.allclose: {are_close}")

# ── [2] Final RMSNorm gain ────────────────────────────────────────────────────
# g = model.model.norm.weight is the learned element-wise gain of the final RMSNorm.
# effective_unembedding[i] = W_U[i] * g defines the direction in the pre-norm
# hidden space that the model effectively reads for token i.  It absorbs the
# RMSNorm learned gain only — the RMS scalar in the denominator depends on each
# hidden state and is omitted here.  This is a visualization aid, NOT W_U itself.
g = model.model.norm.weight     # [hidden_size]
print(f"\n  final norm gain: shape={tuple(g.shape)}, dtype={g.dtype}")

vocab_size = model.config.vocab_size

# ── [3] Token subset ──────────────────────────────────────────────────────────
print("\n[2] Building token subset")

# Prompt tokens
prompt_ids: set[int] = set(inputs["input_ids"][0].tolist())
print(f"  prompt tokens        : {len(prompt_ids)}")

# Special tokens
special_ids: set[int] = set(tokenizer.all_special_ids)
print(f"  special tokens       : {len(special_ids)}")

# Logit lens top-k tokens (from 08 output, if available)
logit_lens_ids: set[int] = set()
topk_csv_path = outputs_dir / "prelim_logit_lens_topk.csv"
if topk_csv_path.exists():
    with topk_csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = int(row["token_id"])
            if 0 <= tid < vocab_size:
                logit_lens_ids.add(tid)
    print(f"  logit_lens_topk      : {len(logit_lens_ids)}")
else:
    print(f"  logit_lens_topk      : not found, skipped")

# Manual tokens (encode each string; one string may map to multiple tokens)
manual_strings = [
    "言語", "モデル", "京都", "大学", "情報", "学科", "AI", "人工知能",
    "token", "embedding", "softmax", "attention", "Transformer",
    "\n", "\n\n", "。", "、",
]
manual_ids: set[int] = set()
for s in manual_strings:
    for tid in tokenizer.encode(s, add_special_tokens=False):
        if 0 <= tid < vocab_size:
            manual_ids.add(tid)
print(f"  manual tokens        : {len(manual_ids)}")

# Background tokens (random, seed=0)
rng = random.Random(0)
background_ids: set[int] = set(rng.sample(range(vocab_size), BACKGROUND_SIZE))
print(f"  background tokens    : {len(background_ids)}")

# Combined (sorted for deterministic ordering)
subset_ids: list[int] = sorted(
    prompt_ids | special_ids | logit_lens_ids | manual_ids | background_ids
)
N = len(subset_ids)
print(f"  total subset         : {N} tokens")

# ── Subset-only weight copy ───────────────────────────────────────────────────
# Copy only the subset rows to CPU float32 — avoids a full [vocab_size, hidden_size]
# allocation for the metadata and coordinate computation that follows.
subset_tensor = torch.tensor(subset_ids, device=W_E.device, dtype=torch.long)
W_E_subset = W_E.index_select(0, subset_tensor).detach().cpu().float()  # [N, hidden_size]
W_U_subset = W_U.index_select(0, subset_tensor).detach().cpu().float()  # [N, hidden_size]
g_cpu = g.detach().cpu().float()                                         # [hidden_size]

# ── [4] Per-token metadata ────────────────────────────────────────────────────
print("\n[3] Computing per-token metadata")


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = a.norm().item(), b.norm().item()
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float((a @ b).item() / (na * nb))


token_rows: list[dict] = []
for i, token_id in enumerate(subset_ids):
    e = W_E_subset[i]       # input embedding vector
    u = W_U_subset[i]       # unembedding vector
    eu = u * g_cpu          # effective unembedding (absorbs RMSNorm gain; see [2])

    raw_token = str(tokenizer.convert_ids_to_tokens([token_id])[0])
    piece = tokenizer.decode([token_id])

    is_prompt = token_id in prompt_ids
    is_special = token_id in special_ids
    is_logit_lens = token_id in logit_lens_ids
    is_manual = token_id in manual_ids
    is_background = token_id in background_ids

    parts = []
    if is_prompt:      parts.append("prompt")
    if is_special:     parts.append("special")
    if is_logit_lens:  parts.append("logit_lens_topk")
    if is_manual:      parts.append("manual")
    if is_background:  parts.append("background")
    sources = ";".join(parts) if parts else "unknown"

    token_rows.append({
        "token_id": token_id,
        "raw_token": raw_token,
        "piece": piece,
        "sources": sources,
        "is_prompt_token": is_prompt,
        "is_special_token": is_special,
        "is_logit_lens_topk_token": is_logit_lens,
        "is_manual_token": is_manual,
        "is_background_token": is_background,
        "input_norm": e.norm().item(),
        "unembedding_norm": u.norm().item(),
        "effective_unembedding_norm": eu.norm().item(),
        "input_unembedding_cosine": cosine_sim(e, u),
        "input_effective_unembedding_cosine": cosine_sim(e, eu),
        "unembedding_effective_unembedding_cosine": cosine_sim(u, eu),
    })

print(f"  computed metadata for {len(token_rows)} tokens")

# Lookup dicts for coord loop
source_lookup = {r["token_id"]: r["sources"] for r in token_rows}
raw_lookup    = {r["token_id"]: r["raw_token"] for r in token_rows}
piece_lookup  = {r["token_id"]: r["piece"] for r in token_rows}

# ── [5] 2D coordinates ────────────────────────────────────────────────────────
# PCA and t-SNE are fit on the combined [3*N, hidden_size] matrix that stacks
# input_embedding, unembedding, and effective_unembedding for all subset tokens.
# Fitting once on the combined data places all three representation types in the
# same 2D space, enabling direct visual comparison per token.
print("\n[4] Computing 2D coordinates")

try:
    from sklearn.decomposition import PCA as SklearnPCA
    from sklearn.manifold import TSNE
    sklearn_available = True
    print("  sklearn available: PCA + t-SNE")
except ImportError:
    sklearn_available = False
    print("  sklearn not available: PCA via SVD, t-SNE skipped")

coord_rows: list[dict] = []
methods_run: list[str] = []
methods_skipped: list[str] = []

# Build [3*N, hidden_size] combined matrix
eu_subset = W_U_subset * g_cpu.unsqueeze(0)  # [N, hidden_size]

X_input   = W_E_subset.numpy().astype(np.float32)   # [N, hidden_size]
X_unembed = W_U_subset.numpy().astype(np.float32)   # [N, hidden_size]
X_eff     = eu_subset.numpy().astype(np.float32)    # [N, hidden_size]

norms_input   = W_E_subset.norm(dim=1).tolist()
norms_unembed = W_U_subset.norm(dim=1).tolist()
norms_eff     = eu_subset.norm(dim=1).tolist()

X_all = np.concatenate([X_input, X_unembed, X_eff], axis=0)  # [3*N, hidden_size]

rep_names_all = (
    ["input_embedding"] * N
    + ["unembedding"] * N
    + ["effective_unembedding"] * N
)
norms_all = norms_input + norms_unembed + norms_eff
tids_all  = subset_ids * 3   # token_id repeated for each representation type

# ── PCA (single fit on X_all) ─────────────────────────────────────────────────
print(f"  PCA on [{3 * N}, {X_all.shape[1]}] combined matrix ...", end="", flush=True)
if sklearn_available:
    pca = SklearnPCA(n_components=2)
    X_pca = pca.fit_transform(X_all)
else:
    X_c = X_all - X_all.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    X_pca = X_c @ Vt[:2].T
methods_run.append("pca")
print(" done")

for idx in range(3 * N):
    tid = tids_all[idx]
    coord_rows.append({
        "method": "pca",
        "representation_type": rep_names_all[idx],
        "token_id": tid,
        "raw_token": raw_lookup[tid],
        "piece": piece_lookup[tid],
        "sources": source_lookup[tid],
        "x": float(X_pca[idx, 0]),
        "y": float(X_pca[idx, 1]),
        "vector_norm": norms_all[idx],
    })

# ── t-SNE (single fit on X_all, sklearn only) ─────────────────────────────────
if sklearn_available:
    perplexity = min(30, max(5, (3 * N - 1) // 3), 3 * N - 1)
    print(
        f"  t-SNE on [{3 * N}, {X_all.shape[1]}] combined matrix "
        f"(perplexity={perplexity}) ...",
        end="", flush=True,
    )
    tsne = TSNE(
        n_components=2,
        random_state=0,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    X_tsne = tsne.fit_transform(X_all)
    methods_run.append("tsne")
    print(" done")

    for idx in range(3 * N):
        tid = tids_all[idx]
        coord_rows.append({
            "method": "tsne",
            "representation_type": rep_names_all[idx],
            "token_id": tid,
            "raw_token": raw_lookup[tid],
            "piece": piece_lookup[tid],
            "sources": source_lookup[tid],
            "x": float(X_tsne[idx, 0]),
            "y": float(X_tsne[idx, 1]),
            "vector_norm": norms_all[idx],
        })
else:
    methods_skipped.append("tsne: sklearn not available")

print(f"  total coord rows: {len(coord_rows)}")

# ── [6] Save outputs ──────────────────────────────────────────────────────────
print("\n[5] Saving output files")

tokens_csv = outputs_dir / "prelim_embedding_unembedding_tokens.csv"
token_fields = [
    "token_id", "raw_token", "piece", "sources",
    "is_prompt_token", "is_special_token", "is_logit_lens_topk_token",
    "is_manual_token", "is_background_token",
    "input_norm", "unembedding_norm", "effective_unembedding_norm",
    "input_unembedding_cosine", "input_effective_unembedding_cosine",
    "unembedding_effective_unembedding_cosine",
]
with tokens_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=token_fields)
    writer.writeheader()
    writer.writerows(token_rows)
print(f"  saved: {tokens_csv}")

coords_csv = outputs_dir / "prelim_embedding_unembedding_coords.csv"
coord_fields = [
    "method", "representation_type", "token_id", "raw_token", "piece",
    "sources", "x", "y", "vector_norm",
]
with coords_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=coord_fields)
    writer.writeheader()
    writer.writerows(coord_rows)
print(f"  saved: {coords_csv}")

summary_json = outputs_dir / "prelim_embedding_unembedding_summary.json"
summary = {
    "model_id": model_id,
    "attn_implementation": attn_impl,
    "device": device,
    "dtype": str(dtype),
    "hidden_size": model.config.hidden_size,
    "vocab_size": vocab_size,
    "num_decoder_layers": len(model.model.layers),
    "num_attention_heads": model.config.num_attention_heads,
    "num_key_value_heads": model.config.num_key_value_heads,
    "rms_norm_eps": model.config.rms_norm_eps,
    "tie_word_embeddings": model.config.tie_word_embeddings,
    "W_E_shape": list(W_E.shape),
    "W_U_shape": list(W_U.shape),
    "W_E_dtype": str(W_E.dtype),
    "W_U_dtype": str(W_U.dtype),
    "W_E_device": str(W_E.device),
    "W_U_device": str(W_U.device),
    "data_ptr_equal": same_data_ptr,
    "max_abs_diff": max_abs_diff,
    "mean_abs_diff": mean_abs_diff,
    "torch_allclose": are_close,
    "prompt": prompt,
    "input_length_tokens": seq_len,
    "num_subset_tokens": N,
    "num_coordinate_rows": len(coord_rows),
    "random_background_size": BACKGROUND_SIZE,
    "manual_strings": manual_strings,
    "coordinate_methods_run": methods_run,
    "coordinate_methods_skipped": methods_skipped,
    "output_files": [str(tokens_csv), str(coords_csv), str(summary_json)],
}
with summary_json.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"  saved: {summary_json}")

print("\nDone.")
