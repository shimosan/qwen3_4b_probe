# Qwen3-4B の attention を notebook 03 の準備として総合的に調べる実験 script。
# clean = "The capital of Japan is" / corrupt = "The capital of France is"、
# metric = logit(" Tokyo") - logit(" Paris")。
#
# 実装する解析:
#   A1 model architecture summary + Qwen3 source snippets
#   A2 prompt token table
#   A3 clean/corrupt forward + baseline top-k
#   A4 self-attention matrix を long CSV にすべて保存
#   A5 head score (attn_to_country, focus, entropy, row L1/JS, ranked)
#   A6 attention heatmap grid PNG (last-query row / full matrix; clean / corrupt / diff)
#   A7 scalar score heatmap (layer × head)
#   A8 residual stream update metric (attn / mlp / before / after; hook + logit lens)
#   A9 component-level activation patching (attn_output, mlp_output at pos=4)
#
# 出力はすべて outputs/ に置く。compact tensor は保存しない (CSV / JSON / PNG のみ)。
# 環境: llm2026

from __future__ import annotations

import csv
import inspect
import json
import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colors as mcolors
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3

from common import load_config, resolve_outputs_dir

plt.rcParams["font.family"] = ["Hiragino Sans", "Apple SD Gothic Neo", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ── Config ─────────────────────────────────────────────────────────────────────
cfg = load_config()
model_id = cfg["model_id"]
attn_impl = cfg["attn_implementation"]

CLEAN_PROMPT   = "The capital of Japan is"
CORRUPT_PROMPT = "The capital of France is"
CLEAN_ANSWER   = " Tokyo"
CORRUPT_ANSWER = " Paris"

COUNTRY_POS = 3      # " Japan" / " France"
QUERY_POS   = 4      # " is"
TOP_K       = 10

outputs_dir = resolve_outputs_dir()

# ── Device / dtype ─────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16
elif torch.backends.mps.is_available():
    device, dtype = "mps", torch.float16
else:
    device, dtype = "cpu", torch.float32

print(f"model_id : {model_id}")
print(f"device   : {device}")
print(f"dtype    : {dtype}")

# ── Tokenizer ──────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(model_id)

clean_inputs   = tokenizer(CLEAN_PROMPT,   return_tensors="pt").to(device)
corrupt_inputs = tokenizer(CORRUPT_PROMPT, return_tensors="pt").to(device)
clean_seq_len   = int(clean_inputs["input_ids"].shape[1])
corrupt_seq_len = int(corrupt_inputs["input_ids"].shape[1])
if clean_seq_len != corrupt_seq_len:
    raise RuntimeError(f"clean/corrupt seq_len mismatch: {clean_seq_len} vs {corrupt_seq_len}")
SEQ_LEN = clean_seq_len
print(f"\nclean prompt   : {CLEAN_PROMPT!r}  ({SEQ_LEN} tokens)")
print(f"corrupt prompt : {CORRUPT_PROMPT!r} ({SEQ_LEN} tokens)")

clean_ans_ids   = tokenizer.encode(CLEAN_ANSWER,   add_special_tokens=False)
corrupt_ans_ids = tokenizer.encode(CORRUPT_ANSWER, add_special_tokens=False)
if len(clean_ans_ids) != 1 or len(corrupt_ans_ids) != 1:
    print("[warning] answer string is not single token")
    print(f"  CLEAN_ANSWER={CLEAN_ANSWER!r} -> {clean_ans_ids}")
    print(f"  CORRUPT_ANSWER={CORRUPT_ANSWER!r} -> {corrupt_ans_ids}")
clean_ans_id   = clean_ans_ids[0]
corrupt_ans_id = corrupt_ans_ids[0]
clean_ans_piece   = tokenizer.decode([clean_ans_id])
corrupt_ans_piece = tokenizer.decode([corrupt_ans_id])
print(f"clean_answer   : {CLEAN_ANSWER!r}  -> id={clean_ans_id}  piece={clean_ans_piece!r}")
print(f"corrupt_answer : {CORRUPT_ANSWER!r}  -> id={corrupt_ans_id}  piece={corrupt_ans_piece!r}")


def _piece_repr(piece: str) -> str:
    # CSV で空白や改行が壊れないように、repr を返す。
    return repr(piece)


# ── Token table (A2) ───────────────────────────────────────────────────────────
print("\n[A2] Building prompt token table")
clean_ids_list   = clean_inputs["input_ids"][0].cpu().tolist()
corrupt_ids_list = corrupt_inputs["input_ids"][0].cpu().tolist()
clean_pieces   = [tokenizer.decode([t]) for t in clean_ids_list]
corrupt_pieces = [tokenizer.decode([t]) for t in corrupt_ids_list]
clean_raw   = [str(tokenizer.convert_ids_to_tokens([t])[0]) for t in clean_ids_list]
corrupt_raw = [str(tokenizer.convert_ids_to_tokens([t])[0]) for t in corrupt_ids_list]
print(f"  clean   tokens: {clean_pieces}")
print(f"  corrupt tokens: {corrupt_pieces}")

prompt_token_rows: list[dict] = []
for run_type, ids, raws, pieces in [
    ("clean",   clean_ids_list,   clean_raw,   clean_pieces),
    ("corrupt", corrupt_ids_list, corrupt_raw, corrupt_pieces),
]:
    for i, tid in enumerate(ids):
        prompt_token_rows.append({
            "run_type":    run_type,
            "position":    i,
            "token_id":    tid,
            "raw_token":   raws[i],
            "piece":       pieces[i],
            "piece_repr":  _piece_repr(pieces[i]),
        })

prompt_tokens_csv = outputs_dir / "prelim_attention_prompt_tokens.csv"
with prompt_tokens_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["run_type","position","token_id","raw_token","piece","piece_repr"])
    w.writeheader(); w.writerows(prompt_token_rows)
print(f"  saved: {prompt_tokens_csv}")

# country / query position token (clean basis)
country_piece_clean   = clean_pieces[COUNTRY_POS]
country_piece_corrupt = corrupt_pieces[COUNTRY_POS]
query_piece           = clean_pieces[QUERY_POS]
print(f"  pos={COUNTRY_POS}: clean={country_piece_clean!r}  corrupt={country_piece_corrupt!r}")
print(f"  pos={QUERY_POS}: {query_piece!r}  (両 prompt で同じはず)")

# ── Model (A1) ─────────────────────────────────────────────────────────────────
print("\n[A1] Loading model")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=dtype,
    attn_implementation=attn_impl,
)
model.to(device).eval()  # type: ignore[union-attr]

cfg_model = model.config
K = cfg_model.num_hidden_layers
hidden_size = cfg_model.hidden_size
num_heads = cfg_model.num_attention_heads
num_kv_heads = cfg_model.num_key_value_heads
head_dim = cfg_model.head_dim
intermediate_size = cfg_model.intermediate_size
vocab_size = cfg_model.vocab_size
num_kv_groups = num_heads // num_kv_heads

layer0 = model.model.layers[0]
arch_summary = {
    "model_id": model_id,
    "device": device,
    "dtype": str(dtype),
    "attn_implementation": attn_impl,
    "num_hidden_layers": K,
    "hidden_size": hidden_size,
    "vocab_size": vocab_size,
    "intermediate_size": intermediate_size,
    "num_attention_heads": num_heads,
    "num_key_value_heads": num_kv_heads,
    "num_key_value_groups": num_kv_groups,
    "head_dim": head_dim,
    "rms_norm_eps": cfg_model.rms_norm_eps,
    "tie_word_embeddings": bool(cfg_model.tie_word_embeddings),
    "decoder_layer_class": type(layer0).__name__,
    "attention_module_class": type(layer0.self_attn).__name__,
    "mlp_module_class": type(layer0.mlp).__name__,
    "GQA_note": (
        f"Qwen3-4B は GQA: Q heads={num_heads}, K/V heads={num_kv_heads}, "
        f"num_kv_groups={num_kv_groups}, head_dim={head_dim}"
    ),
}
arch_json = outputs_dir / "prelim_attention_architecture_summary.json"
with arch_json.open("w", encoding="utf-8") as f:
    json.dump(arch_summary, f, indent=2, ensure_ascii=False)
print(f"  saved: {arch_json}")
print(f"  K={K}, hidden={hidden_size}, Q heads={num_heads}, K/V heads={num_kv_heads}, head_dim={head_dim}")

snippets_path = outputs_dir / "prelim_attention_source_snippets.txt"
with snippets_path.open("w", encoding="utf-8") as f:
    f.write(f"# transformers source snippets used by script 19\n")
    f.write(f"# file: {modeling_qwen3.__file__}\n\n")
    for name in ["Qwen3DecoderLayer", "Qwen3Attention", "Qwen3MLP"]:
        cls = getattr(modeling_qwen3, name)
        f.write(f"=== {name}.forward ===\n")
        f.write(inspect.getsource(cls.forward))
        f.write("\n\n")
print(f"  saved: {snippets_path}")


# ── A3 clean/corrupt forward + baseline top-k ──────────────────────────────────
print("\n[A3] Clean / corrupt forward")
with torch.no_grad():
    clean_out = model(
        **clean_inputs, output_hidden_states=True, output_attentions=True, use_cache=False
    )
    corrupt_out = model(
        **corrupt_inputs, output_hidden_states=True, output_attentions=True, use_cache=False
    )

clean_hidden    = clean_out.hidden_states     # tuple length K+1, each [1, T, H]
corrupt_hidden  = corrupt_out.hidden_states
clean_attns     = clean_out.attentions        # tuple length K, each [1, num_heads, T, T]
corrupt_attns   = corrupt_out.attentions

clean_logits   = clean_out.logits[0, QUERY_POS, :].float().cpu()
corrupt_logits = corrupt_out.logits[0, QUERY_POS, :].float().cpu()
clean_probs   = torch.softmax(clean_logits,   dim=-1)
corrupt_probs = torch.softmax(corrupt_logits, dim=-1)

clean_metric   = (clean_logits[clean_ans_id]   - clean_logits[corrupt_ans_id]).item()
corrupt_metric = (corrupt_logits[clean_ans_id] - corrupt_logits[corrupt_ans_id]).item()
metric_range   = clean_metric - corrupt_metric
print(f"  clean   metric = {clean_metric:+.4f}, top1 = {tokenizer.decode([int(clean_probs.argmax())])!r}")
print(f"  corrupt metric = {corrupt_metric:+.4f}, top1 = {tokenizer.decode([int(corrupt_probs.argmax())])!r}")
print(f"  metric_range (clean - corrupt) = {metric_range:+.4f}")

baseline_rows: list[dict] = []
for run_type, logits_v, probs_v in [
    ("clean",   clean_logits,   clean_probs),
    ("corrupt", corrupt_logits, corrupt_probs),
]:
    top_vals, top_ids = torch.topk(probs_v, k=TOP_K)
    for rank, (tid, p) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), start=1):
        piece = tokenizer.decode([tid])
        baseline_rows.append({
            "run_type":   run_type,
            "rank":       rank,
            "token_id":   tid,
            "raw_token":  str(tokenizer.convert_ids_to_tokens([tid])[0]),
            "piece":      piece,
            "piece_repr": _piece_repr(piece),
            "logit":      logits_v[tid].item(),
            "prob":       p,
        })
baseline_csv = outputs_dir / "prelim_attention_baseline_topk.csv"
with baseline_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(baseline_rows[0].keys()))
    w.writeheader(); w.writerows(baseline_rows)
print(f"  saved: {baseline_csv}")

baseline_summary = {
    "clean_metric": clean_metric,
    "corrupt_metric": corrupt_metric,
    "metric_range": metric_range,
    "clean_top1_piece":   tokenizer.decode([int(clean_probs.argmax())]),
    "corrupt_top1_piece": tokenizer.decode([int(corrupt_probs.argmax())]),
    "clean_top1_token_id":   int(clean_probs.argmax()),
    "corrupt_top1_token_id": int(corrupt_probs.argmax()),
    "clean_prob_clean_ans":   clean_probs[clean_ans_id].item(),
    "clean_prob_corrupt_ans": clean_probs[corrupt_ans_id].item(),
    "corrupt_prob_clean_ans":   corrupt_probs[clean_ans_id].item(),
    "corrupt_prob_corrupt_ans": corrupt_probs[corrupt_ans_id].item(),
    "clean_answer_token_id":   clean_ans_id,
    "corrupt_answer_token_id": corrupt_ans_id,
    "clean_answer_piece":   clean_ans_piece,
    "corrupt_answer_piece": corrupt_ans_piece,
    "country_pos": COUNTRY_POS,
    "query_pos":   QUERY_POS,
    "seq_len":     SEQ_LEN,
}
baseline_summary_path = outputs_dir / "prelim_attention_baseline_summary.json"
with baseline_summary_path.open("w", encoding="utf-8") as f:
    json.dump(baseline_summary, f, indent=2, ensure_ascii=False)
print(f"  saved: {baseline_summary_path}")

# stack attentions into [run, layer, head, T, T] float32 cpu numpy
def stack_attns(attns_tuple: Any) -> np.ndarray:
    # tuple len K of [1, H, T, T]
    arrs = [a[0].float().cpu().numpy() for a in attns_tuple]
    return np.stack(arrs, axis=0)  # [K, H, T, T]

clean_A   = stack_attns(clean_attns)
corrupt_A = stack_attns(corrupt_attns)
assert clean_A.shape == (K, num_heads, SEQ_LEN, SEQ_LEN), f"unexpected attn shape {clean_A.shape}"

# ── A4 attention matrix long CSV ───────────────────────────────────────────────
print("\n[A4] Writing self-attention matrix long CSV (all layers × heads × q × k)")
attn_long_csv = outputs_dir / "prelim_attention_self_attention_matrix_long.csv"
fields_long = [
    "run_type","layer_idx","head_idx","query_pos","key_pos",
    "query_token_id","key_token_id",
    "query_piece","key_piece","query_piece_repr","key_piece_repr",
    "attn_weight",
]
n_written = 0
with attn_long_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(fields_long)
    for run_type, A, ids, pieces in [
        ("clean",   clean_A,   clean_ids_list,   clean_pieces),
        ("corrupt", corrupt_A, corrupt_ids_list, corrupt_pieces),
    ]:
        for L in range(K):
            for H in range(num_heads):
                for q in range(SEQ_LEN):
                    for k in range(SEQ_LEN):
                        w.writerow([
                            run_type, L, H, q, k,
                            ids[q], ids[k],
                            pieces[q], pieces[k],
                            _piece_repr(pieces[q]), _piece_repr(pieces[k]),
                            f"{float(A[L,H,q,k]):.6e}",
                        ])
                        n_written += 1
print(f"  saved: {attn_long_csv}  ({n_written} rows)")

# ── A5 head scoring ────────────────────────────────────────────────────────────
print("\n[A5] Computing per-head scores")

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, 0.0, 1.0); q = np.clip(q, 0.0, 1.0)
    sp = p.sum(); sq = q.sum()
    if sp <= 0 or sq <= 0:
        return float("nan")
    p = p / sp; q = q / sq
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * (np.log(a[mask] + eps) - np.log(b[mask] + eps))))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

def row_entropy(row: np.ndarray, eps: float = 1e-12) -> float:
    r = np.clip(row, eps, 1.0)
    return float(-np.sum(row * np.log(r)))

# per (run, layer, head) row at query_pos
per_head_rows: list[dict] = []
# also keep numpy arrays for ranking comparison
scores_attn_to_country = {"clean": np.zeros((K, num_heads)), "corrupt": np.zeros((K, num_heads))}
scores_focus           = {"clean": np.zeros((K, num_heads)), "corrupt": np.zeros((K, num_heads))}

# entropy normalization: causal mask -> visible keys at query_pos = query_pos+1
visible_count = QUERY_POS + 1
log_visible = math.log(visible_count) if visible_count > 1 else 1.0

for run_type, A, ids, pieces in [
    ("clean",   clean_A,   clean_ids_list,   clean_pieces),
    ("corrupt", corrupt_A, corrupt_ids_list, corrupt_pieces),
]:
    for L in range(K):
        for H in range(num_heads):
            row = A[L, H, QUERY_POS, :].copy()  # [T]
            attn_country = float(row[COUNTRY_POS])
            # rank of country among keys (only causally-visible keys 0..QUERY_POS)
            visible = row[: QUERY_POS + 1]
            # rank: 1 = largest. ties -> count of strictly greater
            country_rank = int(np.sum(visible > attn_country)) + 1
            # margin against next-best key (excluding country itself)
            others = np.delete(visible, COUNTRY_POS)
            country_margin = attn_country - float(others.max()) if others.size > 0 else float("nan")
            self_w = float(row[QUERY_POS])
            first_w = float(row[0])
            max_key_pos = int(np.argmax(visible))
            max_key_attn = float(visible[max_key_pos])
            ent = row_entropy(visible)
            ent_norm = ent / log_visible if log_visible > 0 else float("nan")
            focus = attn_country * (1.0 - ent_norm)
            scores_attn_to_country[run_type][L, H] = attn_country
            scores_focus[run_type][L, H]           = focus
            per_head_rows.append({
                "run_type": run_type,
                "layer_idx": L,
                "head_idx": H,
                "query_pos": QUERY_POS,
                "query_piece_repr": _piece_repr(pieces[QUERY_POS]),
                "country_pos": COUNTRY_POS,
                "country_piece_repr": _piece_repr(pieces[COUNTRY_POS]),
                "attn_to_country": attn_country,
                "country_rank": country_rank,
                "country_margin": country_margin,
                "self_attn_weight": self_w,
                "first_token_weight": first_w,
                "max_key_pos": max_key_pos,
                "max_key_piece_repr": _piece_repr(pieces[max_key_pos]),
                "max_key_attn": max_key_attn,
                "row_entropy": ent,
                "row_entropy_norm": ent_norm,
                "focus_score": focus,
            })

head_scores_csv = outputs_dir / "prelim_attention_head_scores.csv"
with head_scores_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(per_head_rows[0].keys()))
    w.writeheader(); w.writerows(per_head_rows)
print(f"  saved: {head_scores_csv}  ({len(per_head_rows)} rows)")

# clean/corrupt compare per (layer, head)
mean_attn = 0.5 * (scores_attn_to_country["clean"] + scores_attn_to_country["corrupt"])
abs_diff_attn = np.abs(scores_attn_to_country["clean"] - scores_attn_to_country["corrupt"])
mean_focus = 0.5 * (scores_focus["clean"] + scores_focus["corrupt"])
row_l1 = np.zeros((K, num_heads))
row_js = np.zeros((K, num_heads))
clean_country_rank   = np.zeros((K, num_heads), dtype=int)
corrupt_country_rank = np.zeros((K, num_heads), dtype=int)
# fill rank from per_head_rows
for r in per_head_rows:
    L = r["layer_idx"]; H = r["head_idx"]
    if r["run_type"] == "clean":
        clean_country_rank[L, H] = r["country_rank"]
    else:
        corrupt_country_rank[L, H] = r["country_rank"]
for L in range(K):
    for H in range(num_heads):
        rc = clean_A[L, H, QUERY_POS, :]
        ro = corrupt_A[L, H, QUERY_POS, :]
        row_l1[L, H] = float(np.abs(rc - ro).sum())
        row_js[L, H] = js_divergence(rc, ro)

ranked_rows: list[dict] = []
def add_ranking(rank_type: str, mat: np.ndarray, descending: bool = True, top_n: int = 50) -> None:
    flat = mat.flatten()
    order = np.argsort(-flat) if descending else np.argsort(flat)
    for rank, idx in enumerate(order[:top_n], start=1):
        L = int(idx // num_heads); H = int(idx % num_heads)
        ranked_rows.append({
            "rank_type": rank_type,
            "rank": rank,
            "layer_idx": L,
            "head_idx": H,
            "clean_attn_to_country":   float(scores_attn_to_country["clean"][L, H]),
            "corrupt_attn_to_country": float(scores_attn_to_country["corrupt"][L, H]),
            "mean_attn_to_country":    float(mean_attn[L, H]),
            "abs_diff_attn_to_country": float(abs_diff_attn[L, H]),
            "clean_focus_score":   float(scores_focus["clean"][L, H]),
            "corrupt_focus_score": float(scores_focus["corrupt"][L, H]),
            "mean_focus_score":    float(mean_focus[L, H]),
            "row_l1_clean_corrupt": float(row_l1[L, H]),
            "row_js_clean_corrupt": float(row_js[L, H]),
            "clean_country_rank":   int(clean_country_rank[L, H]),
            "corrupt_country_rank": int(corrupt_country_rank[L, H]),
            "score_value":          float(mat[L, H]),
        })

add_ranking("country_pointer_by_mean_attn", mean_attn)
add_ranking("country_pointer_by_focus",     mean_focus)
add_ranking("context_sensitive_by_l1",      row_l1)
add_ranking("context_sensitive_by_js",      row_js)
add_ranking("country_difference_by_abs_diff", abs_diff_attn)

ranked_csv = outputs_dir / "prelim_attention_head_scores_ranked.csv"
with ranked_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(ranked_rows[0].keys()))
    w.writeheader(); w.writerows(ranked_rows)
print(f"  saved: {ranked_csv}  ({len(ranked_rows)} rows)")

# ── A6 attention heatmap grid PNG ──────────────────────────────────────────────
print("\n[A6] Writing attention heatmap grid PNGs")

# Each cell will be drawn as a tile in one big image (much faster than many subplots).
# last-query row: tile shape (1, T). Compose into image (K, H*T) with thin gap columns.
# full matrix:    tile shape (T, T). Compose into image (K*T, H*T) with gap rows/cols.
GAP = 1

def make_last_query_grid_image(A_array: np.ndarray) -> np.ndarray:
    """A_array shape [K, H, T, T]. Return image shape [K, H*(T+GAP)-GAP]."""
    rows = []
    for L in range(A_array.shape[0]):
        row_blocks = []
        for H_idx in range(A_array.shape[1]):
            tile = A_array[L, H_idx, QUERY_POS, :].reshape(1, -1)  # [1, T]
            row_blocks.append(tile)
            if H_idx < A_array.shape[1] - 1:
                row_blocks.append(np.full((1, GAP), np.nan))
        rows.append(np.concatenate(row_blocks, axis=1))  # [1, total_cols]
    return np.concatenate(rows, axis=0)  # [K, total_cols]


def make_full_matrix_grid_image(A_array: np.ndarray) -> np.ndarray:
    """A_array shape [K, H, T, T]. Return image shape [K*(T+GAP)-GAP, H*(T+GAP)-GAP]."""
    layer_blocks = []
    for L in range(A_array.shape[0]):
        head_blocks = []
        for H_idx in range(A_array.shape[1]):
            tile = A_array[L, H_idx]  # [T, T]
            head_blocks.append(tile)
            if H_idx < A_array.shape[1] - 1:
                head_blocks.append(np.full((tile.shape[0], GAP), np.nan))
        layer_blocks.append(np.concatenate(head_blocks, axis=1))
        if L < A_array.shape[0] - 1:
            gap_row = np.full((GAP, layer_blocks[-1].shape[1]), np.nan)
            layer_blocks.append(gap_row)
    return np.concatenate(layer_blocks, axis=0)


def draw_grid_image(
    img: np.ndarray,
    title: str,
    out_path,
    *,
    cell_h: int,
    cell_w: int,
    K_: int,
    H_: int,
    diverging: bool,
    vlim: tuple[float, float] | None = None,
) -> None:
    """Render the assembled image with layer rows / head columns ticks."""
    cmap = plt.get_cmap("RdBu_r") if diverging else plt.get_cmap("viridis")
    cmap.set_bad(color="white")
    if diverging:
        if vlim is None:
            vmax = float(np.nanmax(np.abs(img))); vmin = -vmax
        else:
            vmin, vmax = vlim
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        if vlim is None:
            vmin, vmax = 0.0, float(np.nanmax(img))
        else:
            vmin, vmax = vlim
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # figsize: width ~ H_*cell_w*0.06+3, height ~ K_*cell_h*0.06+3, with caps
    fig_w = min(0.04 * H_ * cell_w + 4.0, 24.0)
    fig_h = min(0.05 * K_ * cell_h + 4.0, 28.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(np.ma.masked_invalid(img), aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")
    ax.set_title(title)
    # x ticks at head centers, y ticks at layer centers
    head_centers = [H_idx * (cell_w + GAP) + cell_w / 2 - 0.5 for H_idx in range(H_)]
    layer_centers = [L * (cell_h + GAP) + cell_h / 2 - 0.5 for L in range(K_)]
    # show every 4th head, every 2nd layer
    ax.set_xticks(head_centers[::4])
    ax.set_xticklabels([str(i) for i in range(0, H_, 4)], fontsize=8)
    ax.set_yticks(layer_centers[::2])
    ax.set_yticklabels([str(i) for i in range(0, K_, 2)], fontsize=8)
    ax.set_xlabel(f"head index (0..{H_ - 1})")
    ax.set_ylabel(f"layer index (0..{K_ - 1})")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  saved: {out_path}")

# (a) last-query row grids
diff_A = clean_A - corrupt_A
for label, A_array, diverging in [
    ("clean",                clean_A,   False),
    ("corrupt",              corrupt_A, False),
    ("clean_minus_corrupt",  diff_A,    True),
]:
    img = make_last_query_grid_image(A_array)
    out = outputs_dir / f"nb03_attention_grid_last_query_{label}.png"
    draw_grid_image(
        img,
        title=f"Qwen3-4B attention last-query row (q=pos{QUERY_POS}='{query_piece}'): {label}",
        out_path=out,
        cell_h=1, cell_w=SEQ_LEN,
        K_=K, H_=num_heads,
        diverging=diverging,
        vlim=(0.0, 1.0) if not diverging else None,
    )

# (b) full 5x5 matrix grids
for label, A_array, diverging in [
    ("clean",                clean_A,   False),
    ("corrupt",              corrupt_A, False),
    ("clean_minus_corrupt",  diff_A,    True),
]:
    img = make_full_matrix_grid_image(A_array)
    out = outputs_dir / f"nb03_attention_grid_full_matrix_{label}.png"
    draw_grid_image(
        img,
        title=f"Qwen3-4B attention full {SEQ_LEN}x{SEQ_LEN} self-attention: {label}",
        out_path=out,
        cell_h=SEQ_LEN, cell_w=SEQ_LEN,
        K_=K, H_=num_heads,
        diverging=diverging,
        vlim=(0.0, 1.0) if not diverging else None,
    )

# ── A7 scalar score heatmap ────────────────────────────────────────────────────
print("\n[A7] Writing scalar score heatmaps (layer × head)")
def scalar_heatmap(mat: np.ndarray, title: str, out_path, diverging: bool = False) -> None:
    cmap = plt.get_cmap("RdBu_r") if diverging else plt.get_cmap("viridis")
    if diverging:
        vmax = float(np.nanmax(np.abs(mat))); vmin = -vmax
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=float(np.nanmin(mat)), vmax=float(np.nanmax(mat)))
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(f"head index (0..{num_heads - 1})")
    ax.set_ylabel(f"layer index (0..{K - 1})")
    ax.set_xticks(range(0, num_heads, 4))
    ax.set_yticks(range(0, K, 2))
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  saved: {out_path}")

scalar_heatmap(mean_attn,
               f"mean attn_to_country (q={QUERY_POS}, key={COUNTRY_POS})",
               outputs_dir / "nb03_attention_score_mean_attn_to_country.png")
scalar_heatmap(mean_focus,
               f"mean focus_score = attn_to_country * (1 - entropy_norm)",
               outputs_dir / "nb03_attention_score_mean_focus.png")
scalar_heatmap(row_l1,
               f"row L1 |A_clean[{QUERY_POS},:] - A_corrupt[{QUERY_POS},:]|",
               outputs_dir / "nb03_attention_score_row_l1_clean_corrupt.png")
scalar_heatmap(row_js,
               f"row JS divergence between clean / corrupt A[{QUERY_POS},:]",
               outputs_dir / "nb03_attention_score_row_js_clean_corrupt.png")

# ── A11 selected-head single heatmaps (large, with token labels) ──────────────
# 5 個の代表的 head を 3-panel (clean / corrupt / clean-corrupt) で大きく描く。
# 各 cell に値を annotate する。横軸 = key token, 縦軸 = query token。
print("\n[A11] Writing single-head attention heatmaps for selected heads")

SELECTED_HEADS: list[tuple[int, int, str]] = [
    # original 5 (selected by mixed criteria: stable pointer, causal, switcher 等)
    (8,  29, "stable country pointer"),
    (24, 26, "country pointer + context shift; L24 attn patching center"),
    (17, 17, "context-sensitive champion (largest row L1)"),
    (14, 11, "country attention flips clean(0.19) -> corrupt(0.65)"),
    (12, 18, "context-sensitive, opposite direction: clean(0.56) -> corrupt(0.10)"),
    # top-10 by row_l1_clean_corrupt (Fig.9) — 上の 3 個 (17,17)(14,11)(12,18) は重複
    (17, 25, "top-L1 rank 2: shift between 'The' (sink) と self attention"),
    (26, 17, "top-L1 rank 3: attn pattern shift, country は弱"),
    (26, 20, "top-L1 rank 6: clean で国名を見る、corrupt では sink へ"),
    (16, 25, "top-L1 rank 7: clean で国名を見る、corrupt では sink へ"),
    (16, 27, "top-L1 rank 8: 対角型 head、country attn ほぼ 0 だが self-attn が大きく動く"),
    (31, 1,  "top-L1 rank 9: corrupt で国名を強く見る (back-half switcher)"),
    (34, 20, "top-L1 rank 10: corrupt で国名を強く見る、より深い layer"),
    # key 位置別 (国名以外) のチャンピオン
    (6,  15, "' capital' (pos=1) pointer rank 1: clean 0.97 / corrupt 0.95"),
    (6,  27, "' capital' (pos=1) pointer rank 2: clean 0.88 / corrupt 0.87"),
    (1,  26, "' of' (pos=2) pointer rank 1: clean 0.95 / corrupt 0.94"),
    (13, 12, "'The' (pos=0, sink) head: clean 1.00 / corrupt 1.00"),
    (14, 30, "' is' (pos=4, self) head: clean 0.94 / corrupt 0.98"),
]
# 順序通りで重複なし (set でユニーク化)
seen: set[tuple[int, int]] = set()
SELECTED_HEADS = [(L, H, n) for (L, H, n) in SELECTED_HEADS if not ((L, H) in seen or seen.add((L, H)))]


def tick_label(pos: int, piece: str) -> str:
    p = piece.replace("\n", "\\n")
    if len(p) > 12:
        p = p[:12] + "…"
    return f"{pos}: {p}"


def draw_single_head_3panel(
    A_clean_mat: np.ndarray, A_corrupt_mat: np.ndarray,
    clean_labels_y: list[str], clean_labels_x: list[str],
    corrupt_labels_y: list[str], corrupt_labels_x: list[str],
    layer_idx: int, head_idx: int, note: str, out_path,
) -> None:
    diff = A_clean_mat - A_corrupt_mat
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    titles = [
        f"clean  A[L{layer_idx},H{head_idx}]",
        f"corrupt  A[L{layer_idx},H{head_idx}]",
        f"clean - corrupt  A[L{layer_idx},H{head_idx}]",
    ]
    panels = [
        (A_clean_mat,   False, mcolors.Normalize(vmin=0.0, vmax=1.0)),
        (A_corrupt_mat, False, mcolors.Normalize(vmin=0.0, vmax=1.0)),
        (diff,          True,  mcolors.TwoSlopeNorm(
            vmin=-max(0.05, float(np.max(np.abs(diff)))),
            vcenter=0.0,
            vmax= max(0.05, float(np.max(np.abs(diff)))))),
    ]
    label_pairs = [
        (clean_labels_y,   clean_labels_x),
        (corrupt_labels_y, corrupt_labels_x),
        (clean_labels_y,   clean_labels_x),  # diff uses clean labels for query axis;
        # key axis differs at pos=3 between clean/corrupt — note this in caption.
    ]
    for ax, (mat, diverging, norm), (ylabels, xlabels), title in zip(
        axes, panels, label_pairs, titles
    ):
        cmap = plt.get_cmap("RdBu_r") if diverging else plt.get_cmap("viridis")
        im = ax.imshow(mat, aspect="equal", cmap=cmap, norm=norm,
                       interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("key token")
        ax.set_ylabel("query token")
        ax.set_xticks(range(len(xlabels)))
        ax.set_yticks(range(len(ylabels)))
        ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(ylabels, fontsize=9)
        # cell annotations
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if diverging:
                    color = "black" if abs(val) < 0.5 else "white"
                else:
                    color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Qwen3-4B attention head  L{layer_idx} / H{head_idx}   —  {note}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  saved: {out_path}")


clean_x_labels = [tick_label(i, p) for i, p in enumerate(clean_pieces)]
corrupt_x_labels = [tick_label(i, p) for i, p in enumerate(corrupt_pieces)]

for L_sel, H_sel, note in SELECTED_HEADS:
    A_c = clean_A[L_sel, H_sel]
    A_o = corrupt_A[L_sel, H_sel]
    out = outputs_dir / f"nb03_attention_head_L{L_sel:02d}_H{H_sel:02d}.png"
    draw_single_head_3panel(
        A_c, A_o,
        clean_labels_y=clean_x_labels, clean_labels_x=clean_x_labels,
        corrupt_labels_y=corrupt_x_labels, corrupt_labels_x=corrupt_x_labels,
        layer_idx=L_sel, head_idx=H_sel, note=note, out_path=out,
    )

# overview montage: top-10 row_l1 head の clean / corrupt / diff を 1 枚にまとめる
# (前半 layer の head と後半 layer の head を視覚的に比較するため)
print("\n[A11b] Writing overview montage of top-10 row_l1 heads")
top10_rows = sorted(
    [r for r in ranked_rows if r["rank_type"] == "context_sensitive_by_l1"],
    key=lambda x: x["rank"],
)[:10]
top10_LH = [(r["layer_idx"], r["head_idx"]) for r in top10_rows]


def draw_top10_montage(panel_kind: str, out_path) -> None:
    """panel_kind in {'clean', 'corrupt', 'diff'}."""
    n = len(top10_LH)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4.4 * nrows))
    axes = np.atleast_2d(axes)
    for idx, (L_sel, H_sel) in enumerate(top10_LH):
        r = idx // ncols; c = idx % ncols
        ax = axes[r, c]
        Ac = clean_A[L_sel, H_sel]
        Ao = corrupt_A[L_sel, H_sel]
        if panel_kind == "clean":
            mat = Ac; cmap = "viridis"
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            xlabels = clean_x_labels; ylabels = clean_x_labels
        elif panel_kind == "corrupt":
            mat = Ao; cmap = "viridis"
            norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
            xlabels = corrupt_x_labels; ylabels = corrupt_x_labels
        else:
            mat = Ac - Ao
            vmax = max(0.1, float(np.max(np.abs(mat))))
            cmap = "RdBu_r"
            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
            xlabels = clean_x_labels; ylabels = clean_x_labels
        im = ax.imshow(mat, aspect="equal", cmap=cmap, norm=norm, interpolation="nearest")
        # half_color tag
        half_tag = "FRONT" if L_sel < 20 else "BACK"
        ax.set_title(f"[{half_tag}] L{L_sel} H{H_sel}", fontsize=11)
        ax.set_xticks(range(len(xlabels))); ax.set_yticks(range(len(ylabels)))
        ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=7)
        ax.set_yticklabels(ylabels, fontsize=7)
        # value annotation
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if panel_kind == "diff":
                    color = "black" if abs(v) < 0.4 else "white"
                else:
                    color = "white" if v < 0.5 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # hide remaining axes
    for k in range(len(top10_LH), nrows * ncols):
        r = k // ncols; c = k % ncols
        axes[r, c].set_visible(False)
    title = {
        "clean":   "Top-10 row_l1 heads: clean attention A[q, k]",
        "corrupt": "Top-10 row_l1 heads: corrupt attention A[q, k]",
        "diff":    "Top-10 row_l1 heads: clean - corrupt  A[q, k]",
    }[panel_kind]
    fig.suptitle(title + "  (FRONT = layer<20, BACK = layer>=20)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  saved: {out_path}")


for kind in ["clean", "corrupt", "diff"]:
    draw_top10_montage(kind, outputs_dir / f"nb03_attention_top10_row_l1_{kind}.png")

# ── A12 key-position pointer analysis (all 5 positions) ────────────────────────
# 国名以外も含めて、pos=4 row が各 key position をどれだけ見るかを
# layer × head × key_pos の grid で記録し、key 位置別 scalar heatmap も出す。
print("\n[A12] Per-key-position head scoring (pos=0..4)")

# attn_to_pos[run, L, H, key] for query=pos4
per_pos_rows: list[dict] = []
attn_to_pos: dict[str, np.ndarray] = {
    "clean":   np.zeros((K, num_heads, SEQ_LEN)),
    "corrupt": np.zeros((K, num_heads, SEQ_LEN)),
}
for run_type, A_array, pieces in [
    ("clean",   clean_A,   clean_pieces),
    ("corrupt", corrupt_A, corrupt_pieces),
]:
    for L in range(K):
        for H in range(num_heads):
            row = A_array[L, H, QUERY_POS, :]
            for kp in range(SEQ_LEN):
                attn_to_pos[run_type][L, H, kp] = float(row[kp])
                per_pos_rows.append({
                    "run_type": run_type,
                    "layer_idx": L,
                    "head_idx": H,
                    "query_pos": QUERY_POS,
                    "key_pos": kp,
                    "key_piece_repr": _piece_repr(pieces[kp]),
                    "attn_weight": float(row[kp]),
                })

per_pos_csv = outputs_dir / "prelim_attention_head_attn_by_keypos.csv"
with per_pos_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(per_pos_rows[0].keys()))
    w.writeheader(); w.writerows(per_pos_rows)
print(f"  saved: {per_pos_csv}  ({len(per_pos_rows)} rows)")

# top heads per key_pos (mean over clean/corrupt), saved to ranked CSV
per_pos_rank_rows: list[dict] = []
for kp in range(SEQ_LEN):
    mat_c = attn_to_pos["clean"][:, :, kp]
    mat_o = attn_to_pos["corrupt"][:, :, kp]
    mean = 0.5 * (mat_c + mat_o)
    diff = mat_c - mat_o
    abs_diff = np.abs(diff)
    flat = mean.flatten()
    order = np.argsort(-flat)
    for rank, idx in enumerate(order[:15], start=1):
        L = int(idx // num_heads); H = int(idx % num_heads)
        per_pos_rank_rows.append({
            "key_pos": kp,
            "key_piece_repr": _piece_repr(clean_pieces[kp]),
            "rank_by": "mean_attn",
            "rank": rank,
            "layer_idx": L,
            "head_idx": H,
            "clean_attn": float(mat_c[L, H]),
            "corrupt_attn": float(mat_o[L, H]),
            "mean_attn": float(mean[L, H]),
            "abs_diff_attn": float(abs_diff[L, H]),
        })

per_pos_rank_csv = outputs_dir / "prelim_attention_head_attn_by_keypos_ranked.csv"
with per_pos_rank_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(per_pos_rank_rows[0].keys()))
    w.writeheader(); w.writerows(per_pos_rank_rows)
print(f"  saved: {per_pos_rank_csv}  ({len(per_pos_rank_rows)} rows)")

# scalar heatmap per key_pos
for kp in range(SEQ_LEN):
    mat_c = attn_to_pos["clean"][:, :, kp]
    mat_o = attn_to_pos["corrupt"][:, :, kp]
    mean = 0.5 * (mat_c + mat_o)
    piece = clean_pieces[kp]  # clean/corrupt は pos=3 のみ違うが、それ以外は同じ
    if kp == COUNTRY_POS:
        piece_label = f"{clean_pieces[kp]!r}/{corrupt_pieces[kp]!r}"
    else:
        piece_label = f"{piece!r}"
    scalar_heatmap(
        mean,
        title=f"mean attn from q=pos{QUERY_POS}(' is') to key=pos{kp} ({piece_label})",
        out_path=outputs_dir / f"nb03_attention_score_mean_attn_to_pos{kp}.png",
    )

# Optional overview: top-3 heads per key_pos の clean attention 5x5 を 5x3 grid に並べる
print("  drawing per-position top-3 head overview")
fig, axes = plt.subplots(5, 3, figsize=(13, 18))
for kp in range(SEQ_LEN):
    mat_mean = 0.5 * (attn_to_pos["clean"][:, :, kp] + attn_to_pos["corrupt"][:, :, kp])
    flat = mat_mean.flatten()
    order = np.argsort(-flat)
    for j in range(3):
        idx = int(order[j])
        L = idx // num_heads; H = idx % num_heads
        Ac = clean_A[L, H]
        ax = axes[kp, j]
        im = ax.imshow(Ac, aspect="equal", cmap="viridis",
                       norm=mcolors.Normalize(0.0, 1.0), interpolation="nearest")
        kp_piece = (clean_pieces[kp] if kp != COUNTRY_POS
                    else f"{clean_pieces[kp]}/{corrupt_pieces[kp]}")
        ax.set_title(
            f"key=pos{kp} ({kp_piece!r})  rank {j + 1}\n"
            f"L{L} H{H}  clean A[4,{kp}]={Ac[QUERY_POS, kp]:.2f}",
            fontsize=9,
        )
        ax.set_xticks(range(SEQ_LEN)); ax.set_yticks(range(SEQ_LEN))
        ax.set_xticklabels(clean_x_labels, rotation=35, ha="right", fontsize=7)
        ax.set_yticklabels(clean_x_labels, fontsize=7)
        for ii in range(SEQ_LEN):
            for jj in range(SEQ_LEN):
                val = Ac[ii, jj]
                color = "white" if val < 0.5 else "black"
                ax.text(jj, ii, f"{val:.2f}", ha="center", va="center",
                        color=color, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle(
    "Top-3 heads attending most strongly to each key position (clean run shown)\n"
    f"q=pos{QUERY_POS} (' is') → key=pos0..4",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = outputs_dir / "nb03_attention_per_keypos_top3_overview.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"  saved: {out}")

# ── A8 residual stream update metric ───────────────────────────────────────────
print("\n[A8] Computing residual stream update metrics (attn / mlp per layer)")

# unembedding using final RMSNorm + lm_head, dtype-safe
norm_module = model.model.norm
lm_head = model.lm_head

def metric_from_residual(h_pos: torch.Tensor) -> tuple[float, float, float]:
    """h_pos: [hidden] tensor on device, dtype = model dtype.
    Returns (metric, logit_clean, logit_corrupt)."""
    with torch.no_grad():
        h2 = norm_module(h_pos.unsqueeze(0).unsqueeze(0))  # [1,1,H]
        logits = lm_head(h2).squeeze(0).squeeze(0).float().cpu()
    m = (logits[clean_ans_id] - logits[corrupt_ans_id]).item()
    return m, logits[clean_ans_id].item(), logits[corrupt_ans_id].item()


def run_with_component_capture(inputs: dict) -> dict[str, list[torch.Tensor]]:
    """Run forward and capture per-layer (input, attn_output, mlp_output) at all positions.
    Returns dict with keys 'h_before_attn', 'attn_update', 'mlp_update' each
    a list of length K of [T, H] tensors on cpu float32 (full sequence)."""
    h_before_attn: list[torch.Tensor | None] = [None] * K
    attn_update:   list[torch.Tensor | None] = [None] * K
    mlp_update:    list[torch.Tensor | None] = [None] * K
    handles = []

    def pre_hook_factory(L):
        def hook(module, args, kwargs):
            x = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            h_before_attn[L] = x[0].detach().to(torch.float32).cpu()
            return None
        return hook

    def attn_hook_factory(L):
        def hook(module, args, output):
            out_tensor = output[0] if isinstance(output, (tuple, list)) else output
            attn_update[L] = out_tensor[0].detach().to(torch.float32).cpu()
            return None
        return hook

    def mlp_hook_factory(L):
        def hook(module, args, output):
            out_tensor = output[0] if isinstance(output, (tuple, list)) else output
            mlp_update[L] = out_tensor[0].detach().to(torch.float32).cpu()
            return None
        return hook

    try:
        for L in range(K):
            layer = model.model.layers[L]
            handles.append(layer.register_forward_pre_hook(pre_hook_factory(L), with_kwargs=True))
            handles.append(layer.self_attn.register_forward_hook(attn_hook_factory(L)))
            handles.append(layer.mlp.register_forward_hook(mlp_hook_factory(L)))
        with torch.no_grad():
            _ = model(**inputs, output_hidden_states=False, output_attentions=False, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    # Sanity: ensure all captured
    for L in range(K):
        if h_before_attn[L] is None or attn_update[L] is None or mlp_update[L] is None:
            raise RuntimeError(f"hook missed at layer {L}")
    return {
        "h_before_attn": [t for t in h_before_attn],   # type: ignore
        "attn_update":   [t for t in attn_update],     # type: ignore
        "mlp_update":    [t for t in mlp_update],      # type: ignore
    }


print("  capturing components (clean)...")
clean_caps   = run_with_component_capture(clean_inputs)
print("  capturing components (corrupt)...")
corrupt_caps = run_with_component_capture(corrupt_inputs)

# Sanity check: h_after_mlp at layer L  ≈  hidden_states[L+1]
# Note: in transformers 5.x, hidden_states tuple has K+1 entries and the LAST
# one (index K) is the *post-RMSNorm* tensor, not the raw layer output.  For
# L=0..K-2 we compare against the raw layer output (hidden_states[L+1]); for
# L=K-1 we compare norm(h_after_mlp) against hidden_states[K] instead.
sanity_rows: list[dict] = []
warn_thresh = 5.0   # generous threshold for fp16 on MPS over many additions
for run_type, caps, hs in [
    ("clean",   clean_caps,   clean_hidden),
    ("corrupt", corrupt_caps, corrupt_hidden),
]:
    for L in range(K):
        h_in   = caps["h_before_attn"][L]            # [T, H] f32 cpu
        h_post = h_in + caps["attn_update"][L] + caps["mlp_update"][L]
        if L < K - 1:
            hs_ref = hs[L + 1][0].to(torch.float32).cpu()
            ref_kind = "hidden_states[L+1]_raw"
        else:
            # compare post-norm versions for the last layer
            h_post_normed = norm_module(h_post.to(device, dtype=dtype).unsqueeze(0))[0]
            h_post = h_post_normed.to(torch.float32).cpu()
            hs_ref = hs[K][0].to(torch.float32).cpu()
            ref_kind = "hidden_states[K]_post_norm"
        diff = float((h_post - hs_ref).abs().max().item())
        sanity_rows.append({
            "run_type":   run_type,
            "layer_idx":  L,
            "reference":  ref_kind,
            "max_abs_diff_h_after_mlp_vs_reference": diff,
        })
        if diff > warn_thresh:
            print(f"  [warn] {run_type} L={L}: large sanity diff {diff:.3e}  (ref={ref_kind})")

sanity_csv = outputs_dir / "prelim_attention_residual_update_sanity.csv"
with sanity_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(sanity_rows[0].keys()))
    w.writeheader(); w.writerows(sanity_rows)
print(f"  saved: {sanity_csv}")

# Now compute residual update metrics at pos=QUERY_POS
print("  computing residual update metrics at query position...")
res_rows: list[dict] = []
for run_type, caps, pieces in [
    ("clean",   clean_caps,   clean_pieces),
    ("corrupt", corrupt_caps, corrupt_pieces),
]:
    for L in range(K):
        h_in    = caps["h_before_attn"][L][QUERY_POS].to(device, dtype=dtype)
        a_upd   = caps["attn_update"][L][QUERY_POS].to(device, dtype=dtype)
        m_upd   = caps["mlp_update"][L][QUERY_POS].to(device, dtype=dtype)
        h_attn  = h_in + a_upd
        h_mlp   = h_attn + m_upd
        m_before, _, _   = metric_from_residual(h_in)
        m_after_attn, _, _ = metric_from_residual(h_attn)
        m_after_mlp, _, _  = metric_from_residual(h_mlp)
        norm_before     = float(h_in.float().norm().item())
        norm_after_attn = float(h_attn.float().norm().item())
        norm_after_mlp  = float(h_mlp.float().norm().item())
        a_norm = float(a_upd.float().norm().item())
        m_norm = float(m_upd.float().norm().item())
        res_rows.append({
            "run_type": run_type,
            "layer_idx": L,
            "pos": QUERY_POS,
            "piece_repr": _piece_repr(pieces[QUERY_POS]),
            "metric_before_attn": m_before,
            "metric_after_attn":  m_after_attn,
            "metric_after_mlp":   m_after_mlp,
            "delta_metric_attn":  m_after_attn - m_before,
            "delta_metric_mlp":   m_after_mlp  - m_after_attn,
            "residual_norm_before":     norm_before,
            "residual_norm_after_attn": norm_after_attn,
            "residual_norm_after_mlp":  norm_after_mlp,
            "attn_update_norm": a_norm,
            "mlp_update_norm":  m_norm,
            "attn_update_relative_norm": a_norm / norm_before if norm_before > 0 else float("nan"),
            "mlp_update_relative_norm":  m_norm / norm_after_attn if norm_after_attn > 0 else float("nan"),
        })

res_csv = outputs_dir / "prelim_attention_residual_update_metrics.csv"
with res_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(res_rows[0].keys()))
    w.writeheader(); w.writerows(res_rows)
print(f"  saved: {res_csv}  ({len(res_rows)} rows)")

# Build figures from res_rows
def res_array(run_type: str, key: str) -> np.ndarray:
    arr = np.zeros(K)
    for r in res_rows:
        if r["run_type"] == run_type:
            arr[r["layer_idx"]] = r[key]
    return arr

for run_type in ["clean", "corrupt"]:
    # metric vs layer
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(K)
    ax.plot(xs, res_array(run_type, "metric_before_attn"), "o-", label="before attn", linewidth=1.5)
    ax.plot(xs, res_array(run_type, "metric_after_attn"),  "s-", label="after attn",  linewidth=1.5)
    ax.plot(xs, res_array(run_type, "metric_after_mlp"),   "^-", label="after mlp",   linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("layer index")
    ax.set_ylabel(f"logit({CLEAN_ANSWER!r}) - logit({CORRUPT_ANSWER!r})  at pos={QUERY_POS}")
    ax.set_title(f"Residual stream metric vs layer  [{run_type}]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = outputs_dir / f"nb03_attention_residual_metric_{run_type}.png"
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f"  saved: {out}")

    # delta metric: attn vs mlp
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.4
    ax.bar(xs - w/2, res_array(run_type, "delta_metric_attn"), width=w, label="Δ attn", color="C0")
    ax.bar(xs + w/2, res_array(run_type, "delta_metric_mlp"),  width=w, label="Δ mlp",  color="C2")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_xlabel("layer index")
    ax.set_ylabel(f"Δ metric")
    ax.set_title(f"Δ metric per layer (attn vs mlp)  [{run_type}]")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = outputs_dir / f"nb03_attention_delta_metric_attn_vs_mlp_{run_type}.png"
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f"  saved: {out}")

    # update norms
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, res_array(run_type, "attn_update_norm"), "o-", label="||attn update||", linewidth=1.5)
    ax.plot(xs, res_array(run_type, "mlp_update_norm"),  "s-", label="||mlp update||",  linewidth=1.5)
    ax.plot(xs, res_array(run_type, "residual_norm_before"), "--", label="||residual before||",
            color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("layer index")
    ax.set_ylabel("L2 norm")
    ax.set_title(f"Update norms vs residual norm per layer  [{run_type}]  pos={QUERY_POS}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = outputs_dir / f"nb03_attention_update_norms_{run_type}.png"
    fig.savefig(out, dpi=160); plt.close(fig)
    print(f"  saved: {out}")

# Free large arrays not needed below
del clean_A, corrupt_A, diff_A
import gc; gc.collect()

# ── A9 component-level activation patching ─────────────────────────────────────
print("\n[A9] Component-level activation patching (attn_output, mlp_output at pos=4)")

# Capture clean component outputs (on-device, model dtype) at pos=QUERY_POS for each layer.
clean_attn_out_pos: dict[int, torch.Tensor] = {}
clean_mlp_out_pos:  dict[int, torch.Tensor] = {}

def capture_pos_hooks() -> list:
    hs = []
    def attn_h(L):
        def hook(module, args, output):
            out_tensor = output[0] if isinstance(output, (tuple, list)) else output
            clean_attn_out_pos[L] = out_tensor[0, QUERY_POS, :].detach().clone()
            return None
        return hook
    def mlp_h(L):
        def hook(module, args, output):
            out_tensor = output[0] if isinstance(output, (tuple, list)) else output
            clean_mlp_out_pos[L] = out_tensor[0, QUERY_POS, :].detach().clone()
            return None
        return hook
    for L in range(K):
        hs.append(model.model.layers[L].self_attn.register_forward_hook(attn_h(L)))
        hs.append(model.model.layers[L].mlp.register_forward_hook(mlp_h(L)))
    return hs

print("  capturing clean component outputs at pos=4...")
_hs = capture_pos_hooks()
try:
    with torch.no_grad():
        _ = model(**clean_inputs, output_hidden_states=False, output_attentions=False, use_cache=False)
finally:
    for h in _hs:
        h.remove()
assert len(clean_attn_out_pos) == K and len(clean_mlp_out_pos) == K


def patch_run(component: str, layer_idx: int, patch_vec: torch.Tensor):
    """Patch component output at (layer_idx, pos=QUERY_POS) with patch_vec on corrupt run."""
    target = (model.model.layers[layer_idx].self_attn if component == "attn_output"
              else model.model.layers[layer_idx].mlp)

    def hook(module, args, output):
        if isinstance(output, tuple):
            out_tensor = output[0].clone()
            out_tensor[0, QUERY_POS, :] = patch_vec.to(out_tensor.dtype)
            return (out_tensor,) + tuple(output[1:])
        elif isinstance(output, list):
            out_list = list(output)
            t = out_list[0].clone()
            t[0, QUERY_POS, :] = patch_vec.to(t.dtype)
            out_list[0] = t
            return out_list
        else:
            t = output.clone()
            t[0, QUERY_POS, :] = patch_vec.to(t.dtype)
            return t

    handle = target.register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(**corrupt_inputs, output_hidden_states=False,
                        output_attentions=False, use_cache=False)
    finally:
        handle.remove()
    return out


patch_by_layer_rows: list[dict] = []
patch_topk_rows: list[dict] = []

for component, patch_pool in [
    ("attn_output", clean_attn_out_pos),
    ("mlp_output",  clean_mlp_out_pos),
]:
    for L in range(K):
        patch_vec = patch_pool[L].to(device)
        patched_out = patch_run(component, L, patch_vec)
        logits_pos = patched_out.logits[0, QUERY_POS, :].float().cpu()
        probs_pos  = torch.softmax(logits_pos, dim=-1)
        patched_metric = (logits_pos[clean_ans_id] - logits_pos[corrupt_ans_id]).item()
        recovery = ((patched_metric - corrupt_metric) / metric_range
                    if abs(metric_range) > 1e-9 else float("nan"))
        top1 = int(probs_pos.argmax())
        top1_piece = tokenizer.decode([top1])
        patch_by_layer_rows.append({
            "component": component,
            "layer_idx": L,
            "patch_site": f"{component}_L{L:02d}",
            "clean_metric": clean_metric,
            "corrupt_metric": corrupt_metric,
            "patched_metric": patched_metric,
            "recovery": recovery,
            "clean_prob":             clean_probs[clean_ans_id].item(),
            "corrupt_prob":           corrupt_probs[clean_ans_id].item(),
            "patched_clean_prob":     probs_pos[clean_ans_id].item(),
            "patched_corrupt_prob":   probs_pos[corrupt_ans_id].item(),
            "patched_top1_token_id":  top1,
            "patched_top1_piece":     top1_piece,
            "patched_top1_piece_repr": _piece_repr(top1_piece),
        })
        top_vals, top_ids = torch.topk(probs_pos, k=TOP_K)
        for rank, (tid, p) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), start=1):
            piece = tokenizer.decode([tid])
            patch_topk_rows.append({
                "component": component,
                "layer_idx": L,
                "patch_site": f"{component}_L{L:02d}",
                "rank": rank,
                "token_id": tid,
                "raw_token": str(tokenizer.convert_ids_to_tokens([tid])[0]),
                "piece": piece,
                "piece_repr": _piece_repr(piece),
                "logit": logits_pos[tid].item(),
                "prob": p,
            })
        if L < 2 or L >= K - 2 or L in (12, 24):
            print(f"  [{component}] L={L:2d}  patched_metric={patched_metric:+.3f}"
                  f"  recovery={recovery:+.3f}  top1={top1_piece!r}")

patch_by_layer_csv = outputs_dir / "prelim_attention_component_patching_by_layer.csv"
with patch_by_layer_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(patch_by_layer_rows[0].keys()))
    w.writeheader(); w.writerows(patch_by_layer_rows)
print(f"  saved: {patch_by_layer_csv}")

patch_topk_csv = outputs_dir / "prelim_attention_component_patching_topk.csv"
with patch_topk_csv.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(patch_topk_rows[0].keys()))
    w.writeheader(); w.writerows(patch_topk_rows)
print(f"  saved: {patch_topk_csv}")

patch_summary = {
    "clean_metric": clean_metric,
    "corrupt_metric": corrupt_metric,
    "metric_range": metric_range,
    "components": ["attn_output", "mlp_output"],
    "num_layers": K,
    "patch_position": QUERY_POS,
    "metric_definition": "logit(clean_ans) - logit(corrupt_ans) at pos=QUERY_POS",
    "recovery_definition": "(patched_metric - corrupt_metric) / (clean_metric - corrupt_metric)",
}
patch_summary_path = outputs_dir / "prelim_attention_component_patching_summary.json"
with patch_summary_path.open("w", encoding="utf-8") as f:
    json.dump(patch_summary, f, indent=2, ensure_ascii=False)
print(f"  saved: {patch_summary_path}")

# Patching plots
def patch_array(component: str, key: str) -> np.ndarray:
    arr = np.zeros(K)
    for r in patch_by_layer_rows:
        if r["component"] == component:
            arr[r["layer_idx"]] = r[key]
    return arr

xs = np.arange(K)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xs, patch_array("attn_output", "recovery"), "o-", label="patch attn_output", linewidth=1.5)
ax.plot(xs, patch_array("mlp_output",  "recovery"), "s-", label="patch mlp_output",  linewidth=1.5)
ax.axhline(0.0, color="black", linewidth=0.5)
ax.axhline(1.0, color="green", linewidth=0.5, linestyle="--", alpha=0.6)
ax.set_xlabel("layer index"); ax.set_ylabel("recovery")
ax.set_title(f"Component activation patching at pos={QUERY_POS}: recovery vs layer")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
out = outputs_dir / "nb03_attention_component_patching_recovery.png"
fig.savefig(out, dpi=160); plt.close(fig); print(f"  saved: {out}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(xs, patch_array("attn_output", "patched_metric"), "o-", label="patch attn_output", linewidth=1.5)
ax.plot(xs, patch_array("mlp_output",  "patched_metric"), "s-", label="patch mlp_output",  linewidth=1.5)
ax.axhline(clean_metric,   color="C0", linestyle="--", linewidth=0.8, alpha=0.6, label="clean")
ax.axhline(corrupt_metric, color="C3", linestyle="--", linewidth=0.8, alpha=0.6, label="corrupt")
ax.set_xlabel("layer index"); ax.set_ylabel("patched metric")
ax.set_title(f"Component activation patching at pos={QUERY_POS}: patched metric vs layer")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
out = outputs_dir / "nb03_attention_component_patching_metric.png"
fig.savefig(out, dpi=160); plt.close(fig); print(f"  saved: {out}")

# top1 piece-as-categorical-color heatmap
# 2 row × 36 col のセル。色は patched top-1 トークンのカテゴリ:
#   green = " Tokyo" (= clean answer; answer 入れ替え成功)
#   red   = " Paris" (= corrupt answer; 入れ替え失敗、corrupt のまま)
#   gray  = それ以外 (中間状態)
# セル中央には patched top-1 piece をテキスト表示 (色のバックアップ)。
def categorize_top1(piece: str) -> str:
    if piece == CLEAN_ANSWER:
        return "tokyo"
    if piece == CORRUPT_ANSWER:
        return "paris"
    return "other"

CAT_COLORS = {"tokyo": "#2ca02c", "paris": "#d62728", "other": "#bdbdbd"}
CAT_LABEL  = {"tokyo": f"top-1 = {CLEAN_ANSWER!r} (answer flipped to clean)",
              "paris": f"top-1 = {CORRUPT_ANSWER!r} (still corrupt)",
              "other": "top-1 = other token"}

# index map: row 0 = attn_output, row 1 = mlp_output
row_index = {"attn_output": 0, "mlp_output": 1}
color_grid = np.empty((2, K), dtype=object)
piece_grid = np.empty((2, K), dtype=object)
recovery_grid = np.zeros((2, K))
for r in patch_by_layer_rows:
    ri = row_index[r["component"]]
    ci = r["layer_idx"]
    cat = categorize_top1(r["patched_top1_piece"])
    color_grid[ri, ci] = CAT_COLORS[cat]
    piece_grid[ri, ci] = r["patched_top1_piece"]
    recovery_grid[ri, ci] = r["recovery"]

fig, ax = plt.subplots(figsize=(18, 3.6))
# draw colored rectangles per cell
import matplotlib.patches as mpatches
for ri in range(2):
    for ci in range(K):
        ax.add_patch(plt.Rectangle((ci - 0.5, ri - 0.5), 1.0, 1.0,
                                   facecolor=color_grid[ri, ci],
                                   edgecolor="white", linewidth=0.6))
        piece = piece_grid[ri, ci]
        label = piece.strip() if piece.strip() else repr(piece)
        if len(label) > 6:
            label = label[:6]
        # recovery value, small font, contrasting color
        rec = recovery_grid[ri, ci]
        # text color: black on light, white on dark
        face = color_grid[ri, ci]
        is_dark = face == CAT_COLORS["tokyo"] or face == CAT_COLORS["paris"]
        txt_color = "white" if is_dark else "black"
        ax.text(ci, ri - 0.15, label, ha="center", va="center",
                fontsize=8, color=txt_color, fontweight="bold")
        ax.text(ci, ri + 0.22, f"r={rec:+.2f}", ha="center", va="center",
                fontsize=6, color=txt_color)

ax.set_xlim(-0.5, K - 0.5); ax.set_ylim(1.5, -0.5)  # invert y so attn on top
ax.set_xticks(range(0, K, 2))
ax.set_yticks([0, 1]); ax.set_yticklabels(["attn_output patching", "mlp_output patching"])
ax.set_xlabel("layer index")
ax.set_title(
    f"Component activation patching at pos={QUERY_POS}: patched top-1 token & recovery\n"
    f"(cell color = top-1 category, 'r=...' = recovery)"
)
# legend
legend_handles = [
    mpatches.Patch(color=CAT_COLORS["tokyo"], label=CAT_LABEL["tokyo"]),
    mpatches.Patch(color=CAT_COLORS["paris"], label=CAT_LABEL["paris"]),
    mpatches.Patch(color=CAT_COLORS["other"], label=CAT_LABEL["other"]),
]
ax.legend(handles=legend_handles, loc="upper center",
          bbox_to_anchor=(0.5, -0.25), ncol=3, fontsize=9)
fig.tight_layout()
out = outputs_dir / "nb03_attention_component_patching_top1.png"
fig.savefig(out, dpi=160, bbox_inches="tight"); plt.close(fig); print(f"  saved: {out}")

# ── Final summary print ───────────────────────────────────────────────────────
print("\n========== SUMMARY ==========")
print(f"clean baseline top1   : {baseline_summary['clean_top1_piece']!r}  metric={clean_metric:+.4f}")
print(f"corrupt baseline top1 : {baseline_summary['corrupt_top1_piece']!r}  metric={corrupt_metric:+.4f}")
print(f"clean answer matches  Tokyo : {baseline_summary['clean_top1_piece'] == CLEAN_ANSWER}")
print(f"corrupt answer matches Paris: {baseline_summary['corrupt_top1_piece'] == CORRUPT_ANSWER}")
print(f"attn long CSV rows    : {n_written}  (expect 2*K*H*T*T = {2*K*num_heads*SEQ_LEN*SEQ_LEN})")
print(f"head scores rows      : {len(per_head_rows)}")
print(f"residual update rows  : {len(res_rows)}")
print(f"component patching rows: {len(patch_by_layer_rows)}")
key_pngs = [
    "nb03_attention_grid_last_query_clean.png",
    "nb03_attention_grid_last_query_corrupt.png",
    "nb03_attention_grid_last_query_clean_minus_corrupt.png",
    "nb03_attention_grid_full_matrix_clean.png",
    "nb03_attention_grid_full_matrix_corrupt.png",
    "nb03_attention_grid_full_matrix_clean_minus_corrupt.png",
    "nb03_attention_score_mean_attn_to_country.png",
    "nb03_attention_score_mean_focus.png",
    "nb03_attention_score_row_l1_clean_corrupt.png",
    "nb03_attention_score_row_js_clean_corrupt.png",
    "nb03_attention_residual_metric_clean.png",
    "nb03_attention_residual_metric_corrupt.png",
    "nb03_attention_delta_metric_attn_vs_mlp_clean.png",
    "nb03_attention_delta_metric_attn_vs_mlp_corrupt.png",
    "nb03_attention_update_norms_clean.png",
    "nb03_attention_update_norms_corrupt.png",
    "nb03_attention_component_patching_recovery.png",
    "nb03_attention_component_patching_metric.png",
    "nb03_attention_component_patching_top1.png",
] + [f"nb03_attention_head_L{L:02d}_H{H:02d}.png" for L, H, _ in SELECTED_HEADS] + [
    "nb03_attention_top10_row_l1_clean.png",
    "nb03_attention_top10_row_l1_corrupt.png",
    "nb03_attention_top10_row_l1_diff.png",
] + [f"nb03_attention_score_mean_attn_to_pos{kp}.png" for kp in range(SEQ_LEN)] + [
    "nb03_attention_per_keypos_top3_overview.png",
]
missing = [p for p in key_pngs if not (outputs_dir / p).exists()]
print(f"key PNGs present      : {len(key_pngs) - len(missing)} / {len(key_pngs)}")
if missing:
    print(f"  missing: {missing}")

print("\nDone.")
