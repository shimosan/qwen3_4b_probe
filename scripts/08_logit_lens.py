from __future__ import annotations

import csv
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_config, resolve_outputs_dir

# ── Config ────────────────────────────────────────────────────────────────────
cfg = load_config()
model_id = cfg["model_id"]
prompt = cfg["default_prompt"]
attn_impl = cfg["attn_implementation"]
TOP_K = 20

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

# ── [1] Forward pass ──────────────────────────────────────────────────────────
print("\n[1] Forward pass (output_hidden_states=True)")
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
    )

hs = outputs.hidden_states
K = len(model.model.layers)
if len(hs) != K + 1:
    raise RuntimeError(f"Expected len(hidden_states)={K + 1}, got {len(hs)}")
print(f"  K={K}, len(hidden_states)={len(hs)} ✓")

# ── Selected position ─────────────────────────────────────────────────────────
pos = seq_len - 1
pos_token_id = inputs["input_ids"][0, pos].item()
pos_raw_token = tokenizer.convert_ids_to_tokens([pos_token_id])[0]
pos_piece = tokenizer.decode([pos_token_id])
print(f"  selected pos={pos}: token_id={pos_token_id}, raw={pos_raw_token!r}, piece={pos_piece!r}")

# ── [2] Sanity check: logit_lens(k=K) vs outputs.logits ──────────────────────
# Prerequisite from 07: hidden_states[-1] = hidden_states[K] is already post-norm.
# So lm_head(hs[K]) should match outputs.logits exactly.
print("\n[2] Sanity check: lm_head(hs[K]) vs outputs.logits")
with torch.no_grad():
    logits_K_pos = model.lm_head(hs[K][:, pos, :].to(device)).float()   # [1, vocab]
    diff_pos = (logits_K_pos - outputs.logits[:, pos, :].float()).abs().max().item()
    print(f"  max abs diff (selected pos={pos}): {diff_pos:.4e}")

    logits_K_full = model.lm_head(hs[K].to(device)).float()              # [1, seq_len, vocab]
    diff_full = (logits_K_full - outputs.logits.float()).abs().max().item()
    print(f"  max abs diff (full sequence):      {diff_full:.4e}")
    del logits_K_full

# Final top-1 at selected position (used as reference across all layers)
probs_K = torch.softmax(logits_K_pos[0], dim=-1)
final_top1_token_id = int(probs_K.argmax().item())
final_top1_prob = probs_K[final_top1_token_id].item()
final_top1_logit = logits_K_pos[0, final_top1_token_id].item()
final_top1_raw_token = tokenizer.convert_ids_to_tokens([final_top1_token_id])[0]
final_top1_piece = tokenizer.decode([final_top1_token_id])
print(f"  final top1: id={final_top1_token_id}, piece={final_top1_piece!r}, prob={final_top1_prob:.4f}")

# ── [3] Logit lens per layer ──────────────────────────────────────────────────
# Formula:
#   k = 0,...,K-1 : readout = model.model.norm(hs[k])   (norm not yet applied)
#   k = K         : readout = hs[K]                      (already post-norm)
print("\n[3] Logit lens per layer (k=0..K)")

topk_rows: list[dict] = []
metric_rows: list[dict] = []

for k in range(K + 1):
    with torch.no_grad():
        if k < K:
            # RMSNorm is position-independent; apply only to selected slice.
            readout_pos = model.model.norm(hs[k][:, pos:pos + 1, :])[:, 0, :]
            norm_applied = True
        else:
            # hs[K] is already post-norm (confirmed in 07).
            readout_pos = hs[K][:, pos, :]
            norm_applied = False

        logits_pos = model.lm_head(readout_pos.to(device)).float()  # [1, vocab]

    logits_vec = logits_pos[0]          # [vocab]
    probs_vec = torch.softmax(logits_vec, dim=-1)

    # Top-k tokens by probability
    top_vals, top_ids = torch.topk(probs_vec, k=TOP_K)
    for rank, (tid, prob) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), start=1):
        topk_rows.append({
            "layer_index": k,
            "rank": rank,
            "token_id": tid,
            "raw_token": tokenizer.convert_ids_to_tokens([tid])[0],
            "piece": tokenizer.decode([tid]),
            "logit": logits_vec[tid].item(),
            "prob": prob,
        })

    # Layer top-1
    top1_id = top_ids[0].item()
    top1_prob = top_vals[0].item()
    top1_logit = logits_vec[top1_id].item()
    top1_raw_token = tokenizer.convert_ids_to_tokens([top1_id])[0]
    top1_piece = tokenizer.decode([top1_id])

    # Entropy (nats)
    entropy = -(probs_vec * torch.log(probs_vec + 1e-9)).sum().item()

    # Rank and stats of the final top-1 token in this layer's distribution
    final_logit_here = logits_vec[final_top1_token_id].item()
    final_prob_here = probs_vec[final_top1_token_id].item()
    final_rank_here = int((logits_vec > final_logit_here).sum().item()) + 1

    metric_rows.append({
        "layer_index": k,
        "norm_applied": norm_applied,
        "top1_token_id": top1_id,
        "top1_raw_token": top1_raw_token,
        "top1_piece": top1_piece,
        "top1_logit": top1_logit,
        "top1_prob": top1_prob,
        "entropy": entropy,
        "final_top1_token_id": final_top1_token_id,
        "final_top1_rank_in_this_layer": final_rank_here,
        "final_top1_logit_in_this_layer": final_logit_here,
        "final_top1_prob_in_this_layer": final_prob_here,
    })

    show = k < 3 or k >= K - 2
    if show:
        print(f"  k={k:2d}: top1={top1_piece!r:12s} prob={top1_prob:.4f}  "
              f"final_top1_rank={final_rank_here}")
    elif k == 3:
        print(f"  ... (k=3..{K - 3} omitted) ...")

# ── [4] Save output files ─────────────────────────────────────────────────────
print("\n[4] Saving output files")

topk_csv = outputs_dir / "prelim_logit_lens_topk.csv"
topk_fields = ["layer_index", "rank", "token_id", "raw_token", "piece", "logit", "prob"]
with topk_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=topk_fields)
    writer.writeheader()
    writer.writerows(topk_rows)
print(f"  saved: {topk_csv}")

metrics_csv = outputs_dir / "prelim_logit_lens_layer_metrics.csv"
metric_fields = [
    "layer_index", "norm_applied",
    "top1_token_id", "top1_raw_token", "top1_piece", "top1_logit", "top1_prob",
    "entropy",
    "final_top1_token_id", "final_top1_rank_in_this_layer",
    "final_top1_logit_in_this_layer", "final_top1_prob_in_this_layer",
]
with metrics_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=metric_fields)
    writer.writeheader()
    writer.writerows(metric_rows)
print(f"  saved: {metrics_csv}")

summary_json = outputs_dir / "prelim_logit_lens_summary.json"
summary = {
    "model_id": model_id,
    "attn_implementation": attn_impl,
    "device": device,
    "dtype": str(dtype),
    "num_decoder_layers": K,
    "num_hidden_states": len(hs),
    "hidden_size": model.config.hidden_size,
    "vocab_size": model.config.vocab_size,
    "tie_word_embeddings": model.config.tie_word_embeddings,
    "rms_norm_eps": model.config.rms_norm_eps,
    "num_attention_heads": model.config.num_attention_heads,
    "num_key_value_heads": model.config.num_key_value_heads,
    "input_length_tokens": seq_len,
    "prompt": prompt,
    "selected_position": pos,
    "selected_position_token_id": pos_token_id,
    "selected_position_raw_token": pos_raw_token,
    "selected_position_piece": pos_piece,
    "top_k": TOP_K,
    "final_selected_pos_max_abs_diff": diff_pos,
    "final_full_sequence_max_abs_diff": diff_full,
    "final_top1_token_id": final_top1_token_id,
    "final_top1_raw_token": final_top1_raw_token,
    "final_top1_piece": final_top1_piece,
    "final_top1_prob": final_top1_prob,
    "output_files": [str(topk_csv), str(metrics_csv), str(summary_json)],
}
with summary_json.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"  saved: {summary_json}")

print(f"\nfinal top1: {final_top1_piece!r} (prob={final_top1_prob:.4f})")
print("\nDone.")
