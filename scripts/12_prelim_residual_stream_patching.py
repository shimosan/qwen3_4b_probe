from __future__ import annotations

import csv
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_config, resolve_outputs_dir

# ── Config ─────────────────────────────────────────────────────────────────────
cfg = load_config()
model_id = cfg["model_id"]
attn_impl = cfg["attn_implementation"]

CLEAN_PROMPT   = "The capital of Japan is"
CORRUPT_PROMPT = "The capital of France is"
CLEAN_ANSWER   = " Tokyo"
CORRUPT_ANSWER = " Paris"
TOP_K = 10

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

print(f"model_id : {model_id}")
print(f"device   : {device}")
print(f"dtype    : {dtype}")

# ── Tokenizer ──────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(model_id)

clean_inputs   = tokenizer(CLEAN_PROMPT,   return_tensors="pt").to(device)
corrupt_inputs = tokenizer(CORRUPT_PROMPT, return_tensors="pt").to(device)
clean_seq_len   = clean_inputs["input_ids"].shape[1]
corrupt_seq_len = corrupt_inputs["input_ids"].shape[1]
clean_pos   = clean_seq_len   - 1
corrupt_pos = corrupt_seq_len - 1

print(f"\nclean prompt   : {CLEAN_PROMPT!r}  ({clean_seq_len} tokens, last pos={clean_pos})")
print(f"corrupt prompt : {CORRUPT_PROMPT!r}  ({corrupt_seq_len} tokens, last pos={corrupt_pos})")

# Answer token IDs (use first subword token of each answer string)
clean_ans_ids   = tokenizer.encode(CLEAN_ANSWER,   add_special_tokens=False)
corrupt_ans_ids = tokenizer.encode(CORRUPT_ANSWER, add_special_tokens=False)
clean_ans_id   = clean_ans_ids[0]
corrupt_ans_id = corrupt_ans_ids[0]
if len(clean_ans_ids) != 1 or len(corrupt_ans_ids) != 1:
    print("[warning] answer string is not a single token")
    print(f"  CLEAN_ANSWER={CLEAN_ANSWER!r} -> {clean_ans_ids}")
    print(f"  CORRUPT_ANSWER={CORRUPT_ANSWER!r} -> {corrupt_ans_ids}")
clean_ans_piece   = tokenizer.decode([clean_ans_id])
corrupt_ans_piece = tokenizer.decode([corrupt_ans_id])
print(f"clean_answer   : {CLEAN_ANSWER!r}  -> token_id={clean_ans_id}  piece={clean_ans_piece!r}")
print(f"corrupt_answer : {CORRUPT_ANSWER!r}  -> token_id={corrupt_ans_id}  piece={corrupt_ans_piece!r}")

# ── [0] Prompt token table ─────────────────────────────────────────────────────
print("\n[0] Building prompt token table")
prompt_token_rows: list[dict] = []
for prompt_type, ids_tensor in [
    ("clean",   clean_inputs["input_ids"][0]),
    ("corrupt", corrupt_inputs["input_ids"][0]),
]:
    for i, tid in enumerate(ids_tensor.cpu().tolist()):
        prompt_token_rows.append({
            "prompt_type": prompt_type,
            "position":    i,
            "token_id":    tid,
            "raw_token":   str(tokenizer.convert_ids_to_tokens([tid])[0]),
            "piece":       tokenizer.decode([tid]),
        })
    print(f"  {prompt_type}: {[tokenizer.decode([t]) for t in ids_tensor.cpu().tolist()]}")

# ── Model ──────────────────────────────────────────────────────────────────────
print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=dtype,
    attn_implementation=attn_impl,
)
model.to(device).eval()  # type: ignore[union-attr]
K = len(model.model.layers)
num_params = sum(p.numel() for p in model.parameters())
print(f"  parameters         : {num_params:,}")
print(f"  num_decoder_layers : K = {K}")
print(f"  hidden_size        : {model.config.hidden_size}")

# ── [1] Clean forward ──────────────────────────────────────────────────────────
# Saves all hidden states:
#   clean_hs[0]       = embed_tokens output          (pre layer 0)
#   clean_hs[j]       = layers[j-1] output           (j = 1..K-1)
#   clean_hs[K]       = model.model.norm output      (post final norm)
print("\n[1] Clean forward  (output_hidden_states=True)")
with torch.no_grad():
    clean_out = model(
        **clean_inputs,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
    )
clean_hs = clean_out.hidden_states          # tuple of K+1 tensors, each [1, clean_seq_len, hidden]
clean_logits_pos = clean_out.logits[0, clean_pos, :].float().cpu()
clean_probs = torch.softmax(clean_logits_pos, dim=-1)

clean_metric       = (clean_logits_pos[clean_ans_id] - clean_logits_pos[corrupt_ans_id]).item()
clean_prob_clean   = clean_probs[clean_ans_id].item()
clean_prob_corrupt = clean_probs[corrupt_ans_id].item()

print(f"  metric = logit({clean_ans_piece!r}) - logit({corrupt_ans_piece!r}) = {clean_metric:+.4f}")
print(f"  P({clean_ans_piece!r}) = {clean_prob_clean:.4f}  "
      f"P({corrupt_ans_piece!r}) = {clean_prob_corrupt:.4f}")

baseline_topk_rows: list[dict] = []
top_vals, top_ids = torch.topk(clean_probs, k=TOP_K)
for rank, (tid, prob) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), start=1):
    baseline_topk_rows.append({
        "run_type":  "clean_baseline",
        "rank":      rank,
        "token_id":  tid,
        "raw_token": str(tokenizer.convert_ids_to_tokens([tid])[0]),
        "piece":     tokenizer.decode([tid]),
        "logit":     clean_logits_pos[tid].item(),
        "prob":      prob,
    })

# ── [2] Corrupt forward ────────────────────────────────────────────────────────
print("\n[2] Corrupt forward  (baseline)")
with torch.no_grad():
    corrupt_out = model(
        **corrupt_inputs,
        output_hidden_states=False,
        output_attentions=False,
        use_cache=False,
    )
corrupt_logits_pos = corrupt_out.logits[0, corrupt_pos, :].float().cpu()
corrupt_probs = torch.softmax(corrupt_logits_pos, dim=-1)

corrupt_metric       = (corrupt_logits_pos[clean_ans_id] - corrupt_logits_pos[corrupt_ans_id]).item()
corrupt_prob_clean   = corrupt_probs[clean_ans_id].item()
corrupt_prob_corrupt = corrupt_probs[corrupt_ans_id].item()

metric_range = clean_metric - corrupt_metric   # denominator for recovery

print(f"  metric = logit({clean_ans_piece!r}) - logit({corrupt_ans_piece!r}) = {corrupt_metric:+.4f}")
print(f"  P({clean_ans_piece!r}) = {corrupt_prob_clean:.4f}  "
      f"P({corrupt_ans_piece!r}) = {corrupt_prob_corrupt:.4f}")
print(f"  metric_range (clean - corrupt) = {metric_range:+.4f}")

top_vals, top_ids = torch.topk(corrupt_probs, k=TOP_K)
for rank, (tid, prob) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), start=1):
    baseline_topk_rows.append({
        "run_type":  "corrupt_baseline",
        "rank":      rank,
        "token_id":  tid,
        "raw_token": str(tokenizer.convert_ids_to_tokens([tid])[0]),
        "piece":     tokenizer.decode([tid]),
        "logit":     corrupt_logits_pos[tid].item(),
        "prob":      prob,
    })

# ── Hook factories ─────────────────────────────────────────────────────────────

def make_embed_hook(patch_vec: torch.Tensor, pos: int):
    """Patch embed_tokens tensor output at position pos."""
    def hook(module, inp, out):
        out = out.clone()
        out[0, pos, :] = patch_vec
        return out
    return hook


def make_layer_hook(patch_vec: torch.Tensor, pos: int):
    """Patch decoder layer tensor output (residual stream) at position pos.

    Transformers 5.x: Qwen3DecoderLayer.forward returns a plain tensor [batch, seq, hidden],
    not a tuple.  The hook therefore mirrors make_embed_hook / make_norm_hook.
    """
    def hook(module, inp, out):
        out = out.clone()
        out[0, pos, :] = patch_vec
        return out
    return hook


def make_norm_hook(patch_vec: torch.Tensor, pos: int):
    """Patch final RMSNorm tensor output at position pos."""
    def hook(module, inp, out):
        out = out.clone()
        out[0, pos, :] = patch_vec
        return out
    return hook


# ── [3] Patching loop ──────────────────────────────────────────────────────────
# For each k = 0..K:
#   k = 0     : patch embed_tokens output  with clean_hs[0][:, clean_pos, :]
#   k = 1..K-1: patch layers[k-1] output  with clean_hs[k][:, clean_pos, :]
#   k = K     : patch norm output          with clean_hs[K][:, clean_pos, :]
#
# The corrupt run is executed for each k; only position corrupt_pos is patched.
# recovery = (patched_metric - corrupt_metric) / (clean_metric - corrupt_metric)
print(f"\n[3] Patching loop  k=0..{K}")

by_layer_rows: list[dict] = []
topk_rows: list[dict] = []

for k in range(K + 1):
    patch_vec = clean_hs[k][0, clean_pos, :].detach().to(device)

    if k == 0:
        patch_site = "embed_tokens"
        handle = model.model.embed_tokens.register_forward_hook(
            make_embed_hook(patch_vec, corrupt_pos)
        )
    elif k < K:
        patch_site = f"layer_{k - 1:02d}"
        handle = model.model.layers[k - 1].register_forward_hook(
            make_layer_hook(patch_vec, corrupt_pos)
        )
    else:
        patch_site = "norm"
        handle = model.model.norm.register_forward_hook(
            make_norm_hook(patch_vec, corrupt_pos)
        )

    try:
        with torch.no_grad():
            patched_out = model(
                **corrupt_inputs,
                output_hidden_states=False,
                output_attentions=False,
                use_cache=False,
            )
    finally:
        handle.remove()

    patched_logits_pos = patched_out.logits[0, corrupt_pos, :].float().cpu()
    patched_probs = torch.softmax(patched_logits_pos, dim=-1)

    patched_metric        = (patched_logits_pos[clean_ans_id] - patched_logits_pos[corrupt_ans_id]).item()
    recovery              = ((patched_metric - corrupt_metric) / metric_range
                             if abs(metric_range) > 1e-9 else float("nan"))
    patched_prob_clean    = patched_probs[clean_ans_id].item()
    patched_prob_corrupt  = patched_probs[corrupt_ans_id].item()
    patched_top1_id       = int(patched_probs.argmax().item())
    patched_top1_piece    = tokenizer.decode([patched_top1_id])

    by_layer_rows.append({
        "layer_k":               k,
        "patch_site":            patch_site,
        "clean_metric":          clean_metric,
        "corrupt_metric":        corrupt_metric,
        "patched_metric":        patched_metric,
        "recovery":              recovery,
        "clean_prob":            clean_prob_clean,
        "corrupt_prob":          corrupt_prob_clean,
        "patched_clean_prob":    patched_prob_clean,
        "patched_corrupt_prob":  patched_prob_corrupt,
        "patched_top1_token_id": patched_top1_id,
        "patched_top1_piece":    patched_top1_piece,
    })

    # Top-K tokens for this patch site
    top_vals, top_ids = torch.topk(patched_probs, k=TOP_K)
    for rank, (tid, prob) in enumerate(zip(top_ids.tolist(), top_vals.tolist()), start=1):
        topk_rows.append({
            "layer_k":    k,
            "patch_site": patch_site,
            "rank":       rank,
            "token_id":   tid,
            "raw_token":  str(tokenizer.convert_ids_to_tokens([tid])[0]),
            "piece":      tokenizer.decode([tid]),
            "logit":      patched_logits_pos[tid].item(),
            "prob":       prob,
        })

    show = k < 3 or k >= K - 2
    if show:
        print(f"  k={k:2d} [{patch_site:12s}]  metric={patched_metric:+.4f}"
              f"  recovery={recovery:+.3f}  top1={patched_top1_piece!r}")
    elif k == 3:
        print(f"  ... (k=3..{K - 3} omitted) ...")

print("  Patching done.")

# ── [4] Save output files ──────────────────────────────────────────────────────
print("\n[4] Saving output files")

prompt_tokens_csv = outputs_dir / "prelim_residual_patching_prompt_tokens.csv"
with prompt_tokens_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["prompt_type", "position", "token_id", "raw_token", "piece"]
    )
    writer.writeheader()
    writer.writerows(prompt_token_rows)
print(f"  saved: {prompt_tokens_csv}")

by_layer_csv = outputs_dir / "prelim_residual_patching_by_layer.csv"
by_layer_fields = [
    "layer_k", "patch_site",
    "clean_metric", "corrupt_metric", "patched_metric", "recovery",
    "clean_prob", "corrupt_prob", "patched_clean_prob", "patched_corrupt_prob",
    "patched_top1_token_id", "patched_top1_piece",
]
with by_layer_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=by_layer_fields)
    writer.writeheader()
    writer.writerows(by_layer_rows)
print(f"  saved: {by_layer_csv}")

topk_csv = outputs_dir / "prelim_residual_patching_topk.csv"
topk_fields = ["layer_k", "patch_site", "rank", "token_id", "raw_token", "piece", "logit", "prob"]
with topk_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=topk_fields)
    writer.writeheader()
    writer.writerows(topk_rows)
print(f"  saved: {topk_csv}")

baseline_topk_csv = outputs_dir / "prelim_residual_patching_baseline_topk.csv"
baseline_topk_fields = ["run_type", "rank", "token_id", "raw_token", "piece", "logit", "prob"]
with baseline_topk_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=baseline_topk_fields)
    writer.writeheader()
    writer.writerows(baseline_topk_rows)
print(f"  saved: {baseline_topk_csv}")

summary_json = outputs_dir / "prelim_residual_patching_summary.json"
summary = {
    "model_id": model_id,
    "attn_implementation": attn_impl,
    "device": device,
    "dtype": str(dtype),
    "num_parameters": num_params,
    "num_decoder_layers": K,
    "hidden_size": model.config.hidden_size,
    "vocab_size": model.config.vocab_size,
    "clean_prompt": CLEAN_PROMPT,
    "corrupt_prompt": CORRUPT_PROMPT,
    "clean_answer": CLEAN_ANSWER,
    "corrupt_answer": CORRUPT_ANSWER,
    "clean_ans_token_id": clean_ans_id,
    "corrupt_ans_token_id": corrupt_ans_id,
    "clean_ans_piece": clean_ans_piece,
    "corrupt_ans_piece": corrupt_ans_piece,
    "clean_seq_len": clean_seq_len,
    "corrupt_seq_len": corrupt_seq_len,
    "clean_pos": clean_pos,
    "corrupt_pos": corrupt_pos,
    "metric_definition": "logit(clean_answer_token) - logit(corrupt_answer_token)",
    "recovery_definition": "(patched_metric - corrupt_metric) / (clean_metric - corrupt_metric)",
    "patch_site_mapping": {
        "k=0":     "embed_tokens output    = clean_hs[0]",
        "k=1..K-1": "layers[k-1] output   = clean_hs[k]",
        "k=K":     "model.model.norm output = clean_hs[K]  (post final RMSNorm)",
    },
    "column_notes": {
        "clean_prob":          f"P({CLEAN_ANSWER!r} first token | clean run)",
        "corrupt_prob":        f"P({CLEAN_ANSWER!r} first token | corrupt run)",
        "patched_clean_prob":  f"P({CLEAN_ANSWER!r} first token | patched run at each k)",
        "patched_corrupt_prob": f"P({CORRUPT_ANSWER!r} first token | patched run at each k)",
    },
    "clean_metric": clean_metric,
    "corrupt_metric": corrupt_metric,
    "metric_range": metric_range,
    "clean_prob_clean_ans": clean_prob_clean,
    "clean_prob_corrupt_ans": clean_prob_corrupt,
    "corrupt_prob_clean_ans": corrupt_prob_clean,
    "corrupt_prob_corrupt_ans": corrupt_prob_corrupt,
    "top_k": TOP_K,
    "num_patch_sites": K + 1,
    "baseline_topk_file_note": "Top-k next-token distributions for clean and corrupt baseline runs.",
    "output_files": [
        str(prompt_tokens_csv),
        str(by_layer_csv),
        str(topk_csv),
        str(baseline_topk_csv),
        str(summary_json),
    ],
}
with summary_json.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"  saved: {summary_json}")

print("\nDone.")
