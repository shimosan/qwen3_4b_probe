# Qwen3-4B の hidden states の構造を hook を使って検証する。
# embed_tokens の出力と hidden_states[0] の一致、各 decoder layer の出力と hidden_states[j+1] の一致、
# lm_head(hidden_states[-1]) と outputs.logits の一致を数値で確認する。
# 出力: outputs/prelim_hidden_state_mapping_diffs.csv, prelim_hidden_state_mapping_summary.json
# 環境: llm2026-dev

from __future__ import annotations

import csv
import inspect
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_config, resolve_outputs_dir

cfg = load_config()
model_id = cfg["model_id"]
prompt = cfg["default_prompt"]
attn_impl = cfg["attn_implementation"]

outputs_dir = resolve_outputs_dir()

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

print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=dtype,
    attn_implementation=attn_impl,
)
model.to(device).eval()  # type: ignore[union-attr]
num_params = sum(p.numel() for p in model.parameters())
print(f"  parameters: {num_params:,}")

# ── 1. Source file inspection ─────────────────────────────────────────────────
print("\n[1] Source file inspection")

import transformers.models.qwen3.modeling_qwen3 as qwen3_mod
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
)

source_file = inspect.getfile(qwen3_mod)
print(f"  {source_file}")

paths_txt = outputs_dir / "prelim_qwen3_source_paths.txt"
paths_txt.write_text(
    f"transformers.models.qwen3.modeling_qwen3:\n  {source_file}\n",
    encoding="utf-8",
)
print(f"  saved: {paths_txt}")

snippets_txt = outputs_dir / "prelim_qwen3_source_snippets.txt"
with snippets_txt.open("w", encoding="utf-8") as f:
    for cls_name, cls in [
        ("Qwen3ForCausalLM", Qwen3ForCausalLM),
        ("Qwen3Model", Qwen3Model),
        ("Qwen3DecoderLayer", Qwen3DecoderLayer),
    ]:
        sep = "=" * 72
        f.write(f"{sep}\n# {cls_name}.forward\n{sep}\n")
        try:
            f.write(inspect.getsource(cls.forward))
        except Exception as e:
            f.write(f"# could not extract source: {e}\n")
        f.write("\n\n")
print(f"  saved: {snippets_txt}")

# ── 2. Register hooks and run single forward pass ────────────────────────────
print("\n[2] Registering hooks and running forward (output_hidden_states=True)")

K = len(model.model.layers)
hook_outputs: dict[int, torch.Tensor] = {}

def make_hook(j: int):
    def hook(module, inp, out):
        raw = out[0] if isinstance(out, (tuple, list)) else out
        hook_outputs[j] = raw.detach()
    return hook

handles = [
    layer.register_forward_hook(make_hook(j))
    for j, layer in enumerate(model.model.layers)
]

with torch.no_grad():
    outputs_hook = model(
        **inputs,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
    )

for h in handles:
    h.remove()

hs_hook = outputs_hook.hidden_states  # tuple of length K+1
print(f"  num decoder layers K   = {K}")
print(f"  len(hidden_states)     = {len(hs_hook)}  (expected K+1 = {K + 1})")
print(f"  hidden_states[0].shape = {tuple(hs_hook[0].shape)}")
print(f"  hidden_states[-1].shape= {tuple(hs_hook[-1].shape)}")
print(f"  captured hook outputs for {len(hook_outputs)} layers, hooks removed")

diffs: list[dict] = []

# ── 3. embed_tokens(input_ids) vs hidden_states[0] ───────────────────────────
print("\n[3] embed_tokens(input_ids) vs hidden_states[0]")
with torch.no_grad():
    embed_out = model.model.embed_tokens(inputs["input_ids"])
d = (embed_out - hs_hook[0].to(embed_out.device)).abs().max().item()
print(f"  max abs diff = {d:.4e}")
diffs.append({"label": "embed_tokens_vs_hs[0]", "max_abs_diff": d})

# ── 4. Hook outputs vs hidden_states[j+1] ────────────────────────────────────
print("\n[4] Decoder layer hook outputs vs hidden_states[j+1]")

for j in range(K):
    hook_out = hook_outputs[j]
    if j < K - 1:
        ref = hs_hook[j + 1].to(hook_out.device)
        d = (hook_out - ref).abs().max().item()
        label = f"layer_{j:02d}_hook_vs_hs[{j + 1}]"
    else:
        # Last layer: hook captures pre-norm output; compare norm(hook) vs hs[-1].
        with torch.no_grad():
            normed = model.model.norm(hook_out)
        ref = hs_hook[j + 1].to(normed.device)
        d = (normed - ref).abs().max().item()
        label = f"layer_{j:02d}_norm(hook)_vs_hs[{j + 1}]"
    diffs.append({"label": label, "max_abs_diff": d})

    show = j < 3 or j >= K - 3
    if show:
        print(f"  {label}: {d:.4e}")
    elif j == 3:
        print(f"  ... (layers 3–{K - 4} omitted) ...")

# ── 5. lm_head(hidden_states[-1]) vs outputs_hook.logits ─────────────────────
print("\n[5] lm_head(hidden_states[-1]) vs outputs_hook.logits")
with torch.no_grad():
    lm_out = model.lm_head(hs_hook[-1].to(device))
d = (lm_out - outputs_hook.logits).abs().max().item()
print(f"  max abs diff = {d:.4e}")
diffs.append({"label": "lm_head(hs[-1])_vs_logits", "max_abs_diff": d})

# ── 6. Save diffs CSV ─────────────────────────────────────────────────────────
diffs_csv = outputs_dir / "prelim_hidden_state_mapping_diffs.csv"
with diffs_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["label", "max_abs_diff"])
    writer.writeheader()
    writer.writerows(diffs)
print(f"\nsaved: {diffs_csv}")

# ── 7. Save summary JSON ──────────────────────────────────────────────────────
summary = {
    "model_id": model_id,
    "attn_implementation": attn_impl,
    "device": device,
    "dtype": str(dtype),
    "num_parameters": num_params,
    "num_hidden_states": len(hs_hook),
    "num_decoder_layers": K,
    "input_length_tokens": seq_len,
    "hidden_size": model.config.hidden_size,
    "vocab_size": model.config.vocab_size,
    "tie_word_embeddings": model.config.tie_word_embeddings,
    "rms_norm_eps": model.config.rms_norm_eps,
    "num_attention_heads": model.config.num_attention_heads,
    "num_key_value_heads": model.config.num_key_value_heads,
    "source_file": source_file,
    "max_diffs": {row["label"]: row["max_abs_diff"] for row in diffs},
}

summary_json = outputs_dir / "prelim_hidden_state_mapping_summary.json"
with summary_json.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"saved: {summary_json}")

print("\nDone.")
