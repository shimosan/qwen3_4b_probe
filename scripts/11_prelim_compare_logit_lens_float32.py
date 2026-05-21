# HF モデルと TransformerLens をどちらも float32/CPU でロードし、logit lens の数値を比較する。
# dtype 差を排除した条件で自前実装と TransformerLens の一致を確認する（max_abs_diff < 1e-3 が目標）。
# 出力: outputs/prelim_compare_transformerlens_float32_layer_diffs.csv, _summary.json
# 環境: llm2026-dev（transformer-lens が必要）

from __future__ import annotations

import csv
import gc
import importlib.metadata as ilm
import json
import sys
import traceback as tb

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_config, resolve_outputs_dir

# ── Config ─────────────────────────────────────────────────────────────────────
cfg = load_config()
model_id = cfg["model_id"]
prompt = cfg["default_prompt"]
attn_impl = cfg["attn_implementation"]

outputs_dir = resolve_outputs_dir()

hf_device = "cpu"
tl_device = "cpu"

# ── Environment ────────────────────────────────────────────────────────────────
env_info: dict = {
    "python_version": sys.version.split()[0],
    "torch_version": torch.__version__,
    "hf_device": hf_device,
    "tl_device": tl_device,
}
try:
    env_info["transformers_version"] = ilm.version("transformers")
except Exception:
    env_info["transformers_version"] = "unknown"
try:
    env_info["transformer_lens_version"] = ilm.version("transformer-lens")
except Exception:
    env_info["transformer_lens_version"] = "not installed"

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print(f"model_id : {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(text, return_tensors="pt")
seq_len = inputs["input_ids"].shape[1]
pos = seq_len - 1

pos_token_id = int(inputs["input_ids"][0, pos].item())
pos_raw_token = str(tokenizer.convert_ids_to_tokens([pos_token_id])[0])
pos_piece = tokenizer.decode([pos_token_id])

print(f"input length : {seq_len} tokens")
print(f"selected pos : {pos}  token_id={pos_token_id}  piece={pos_piece!r}")
print(f"hf_device: {hf_device}  tl_device: {tl_device}")

# ── Summary skeleton ───────────────────────────────────────────────────────────
summary: dict = {
    "model_id": model_id,
    "prompt": prompt,
    "attn_implementation": attn_impl,
    "input_length_tokens": seq_len,
    "selected_position": pos,
    "selected_position_token_id": pos_token_id,
    "selected_position_raw_token": pos_raw_token,
    "selected_position_piece": pos_piece,
    "environment": env_info,
    "hf_float32": {"attempted": False, "success": False},
    "transformer_lens_float32": {"attempted": False, "success": False},
    "comparison": {"attempted": False, "comparison_available": False},
    "notes": [
        "HF and TransformerLens are both intended to be float32 in this test.",
        "HF model and TL model are not held simultaneously.",
        (
            "If comparison still differs, possible causes include implementation "
            "convention differences, RMSNorm details, cache point mismatch, "
            "or TransformerLens conversion details."
        ),
    ],
}

# Phase A results (K_hf set here, used in Phase B config check)
hf_logits_by_layer: list[torch.Tensor] = []
K_hf: int = 0
hf_hidden_size: int = 0
hf_vocab_size: int = 0

# ── Phase A: HF Qwen3-4B float32 own_logit_lens ───────────────────────────────
print("\n[Phase A] HF Qwen3-4B float32 own_logit_lens")
hf_s = summary["hf_float32"]
hf_s["attempted"] = True

try:
    print("  Loading HF model (dtype=float32, device=cpu) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        attn_implementation=attn_impl,
    )
    model.to(hf_device).eval()  # type: ignore[union-attr]

    hf_dtype_str = str(next(model.parameters()).dtype)
    K_hf = len(model.model.layers)
    hf_hidden_size = model.config.hidden_size
    hf_vocab_size = model.config.vocab_size

    print(f"  K={K_hf}  hidden_size={hf_hidden_size}  vocab_size={hf_vocab_size}")
    print(f"  model dtype: {hf_dtype_str}")

    inputs_hf = {k: v.to(hf_device) for k, v in inputs.items()}

    print("  Forward pass (output_hidden_states=True, output_attentions=False, use_cache=False) ...")
    with torch.no_grad():
        outputs = model(
            **inputs_hf,
            output_hidden_states=True,
            output_attentions=False,
            use_cache=False,
        )

    hs = outputs.hidden_states
    hf_logits_dtype_str = str(outputs.logits.dtype)

    if len(hs) != K_hf + 1:
        raise RuntimeError(f"Expected {K_hf + 1} hidden states, got {len(hs)}")
    print(f"  len(hidden_states)={len(hs)} ✓")

    # own logit lens: k=0..K
    print(f"  Computing own_logit_lens (k=0..{K_hf}) ...")
    for k in range(K_hf + 1):
        with torch.no_grad():
            if k < K_hf:
                readout = model.model.norm(hs[k][:, pos:pos + 1, :])[:, 0, :]
            else:
                readout = hs[K_hf][:, pos, :]
            logits_k = model.lm_head(readout).float()
        hf_logits_by_layer.append(logits_k[0].cpu())
    print(f"  computed {len(hf_logits_by_layer)} logit vectors")

    # sanity: selected position
    diff_sel = (hf_logits_by_layer[K_hf] - outputs.logits[:, pos, :].float().cpu()[0]).abs()
    sel_max = diff_sel.max().item()
    sel_mean = diff_sel.mean().item()
    print(f"  own[K] vs outputs.logits[pos]: max={sel_max:.4e}  mean={sel_mean:.4e}")

    # sanity: full sequence
    with torch.no_grad():
        lm_full = model.lm_head(hs[-1].to(hf_device)).float()
    diff_full = (lm_full.cpu() - outputs.logits.float().cpu()).abs()
    full_max = diff_full.max().item()
    full_mean = diff_full.mean().item()
    del lm_full
    print(f"  lm_head(hs[-1]) vs outputs.logits (full seq): max={full_max:.4e}  mean={full_mean:.4e}")

    hf_s.update({
        "success": True,
        "model_dtype": hf_dtype_str,
        "logits_dtype": hf_logits_dtype_str,
        "num_hidden_states": len(hs),
        "num_decoder_layers": K_hf,
        "hidden_size": hf_hidden_size,
        "vocab_size": hf_vocab_size,
        "own_final_selected_pos_max_abs_diff": sel_max,
        "own_final_selected_pos_mean_abs_diff": sel_mean,
        "own_final_full_sequence_max_abs_diff": full_max,
        "own_final_full_sequence_mean_abs_diff": full_mean,
    })

    del outputs, hs, model
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  HF model freed.")

except MemoryError as exc:
    hf_s["failure_reason"] = f"MemoryError: {exc}"
    print(f"  FAILED (OOM): {hf_s['failure_reason']}")
except Exception as exc:
    hf_s["failure_reason"] = f"{type(exc).__name__}: {exc}"
    print(f"  FAILED: {hf_s['failure_reason']}")
    if not isinstance(exc, RuntimeError):
        print(tb.format_exc())

# Phase A 失敗時は早期終了
if not hf_s["success"]:
    summary["notes"].append("Phase A failed. Phase B was not attempted.")
    _early_json = outputs_dir / "prelim_compare_transformerlens_float32_summary.json"
    with _early_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nsaved: {_early_json}")
    print("Done (Phase A failed, exiting early).")
    sys.exit(0)

# ── Phase B: TransformerLens Qwen3-4B float32 ─────────────────────────────────
print("\n[Phase B] TransformerLens Qwen3-4B float32")
tl_s = summary["transformer_lens_float32"]
tl_s["attempted"] = True

tl_logits_by_layer: list[torch.Tensor] = []

try:
    import transformer_lens as _tl_pkg  # noqa: F401
    from transformer_lens import HookedTransformer

    tl_version = ilm.version("transformer-lens")
    tl_s["version"] = tl_version
    env_info["transformer_lens_version"] = tl_version
    print(f"  transformer_lens {tl_version}")

    print(
        "  Loading HookedTransformer"
        " (fold_ln=False, center_writing_weights=False, center_unembed=False, float32) ..."
    )
    tl_model = HookedTransformer.from_pretrained(
        model_id,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        default_prepend_bos=False,
        dtype=torch.float32,  # type: ignore[arg-type]
        trust_remote_code=True,
    )
    tl_model.to(tl_device).eval()

    K_tl = tl_model.cfg.n_layers
    d_model_tl = tl_model.cfg.d_model
    d_vocab_tl = tl_model.cfg.d_vocab
    tl_dtype_str = str(next(tl_model.parameters()).dtype)
    print(f"  n_layers={K_tl}  d_model={d_model_tl}  d_vocab={d_vocab_tl}")
    print(f"  TL dtype: {tl_dtype_str}")

    n_layers_match = K_tl == K_hf
    d_model_match = d_model_tl == hf_hidden_size
    d_vocab_match = d_vocab_tl == hf_vocab_size

    tokens_cpu = inputs["input_ids"].cpu()
    tl_s.update({
        "model_dtype": tl_dtype_str,
        "n_layers": K_tl,
        "d_model": d_model_tl,
        "d_vocab": d_vocab_tl,
        "n_layers_match": n_layers_match,
        "d_model_match": d_model_match,
        "d_vocab_match": d_vocab_match,
        "token_check": {
            "tokens_shape": list(tokens_cpu.shape),
            "first_token_id": int(tokens_cpu[0, 0].item()),
            "last_token_id": int(tokens_cpu[0, -1].item()),
            "selected_position_token_id": pos_token_id,
            "input_ids_equal": True,
        },
    })
    print(
        f"  config match: n_layers={n_layers_match}"
        f"  d_model={d_model_match}  d_vocab={d_vocab_match}"
    )

    if not n_layers_match or not d_vocab_match:
        raise RuntimeError(
            f"Config mismatch: n_layers match={n_layers_match}, d_vocab match={d_vocab_match}"
        )

    print(f"  run_with_cache ({seq_len} tokens) ...")
    with torch.no_grad():
        tl_logits_out, tl_cache = tl_model.run_with_cache(
            tokens_cpu,
            prepend_bos=False,
            remove_batch_dim=False,
        )
    tl_s["logits_dtype"] = str(tl_logits_out.dtype)  # type: ignore[union-attr]
    del tl_logits_out

    # TL logit lens: k=0..K
    # k=0: hook_embed -> ln_final -> unembed   (own hs[0] = embed output)
    # k=j (1<=j<=K): resid_post,j-1 -> ln_final -> unembed  (own hs[j] = post-layer j-1 residual)
    # k=K: resid_post,K-1 -> ln_final -> unembed  (own hs[K] = already post-norm, equivalent)
    print(f"  Computing TL logit lens (k=0..{K_tl}) ...")
    tl_errors: list[str] = []
    for k in range(K_tl + 1):
        with torch.no_grad():
            try:
                if k == 0:
                    resid_3d = tl_cache["hook_embed"][:, pos:pos + 1, :]
                else:
                    resid_3d = tl_cache["resid_post", k - 1][:, pos:pos + 1, :]
            except KeyError as ke:
                tl_errors.append(f"k={k}: KeyError {ke}")
                break
            normed_3d = tl_model.ln_final(resid_3d)
            logits_3d = tl_model.unembed(normed_3d)
            tl_logits_by_layer.append(logits_3d[0, 0].float().cpu())

    n_tl = len(tl_logits_by_layer)
    print(f"  computed {n_tl} logit vectors (expected {K_tl + 1})")
    if tl_errors:
        print(f"  errors: {tl_errors}")
        tl_s["layer_errors"] = tl_errors

    tl_s.update({
        "success": n_tl == K_tl + 1,
        "num_tl_logit_lens_computed": n_tl,
    })
    if n_tl != K_tl + 1:
        tl_s["failure_reason"] = f"Only {n_tl}/{K_tl + 1} layers computed"

    del tl_model, tl_cache
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  TL model freed.")

except MemoryError as exc:
    tl_s["failure_reason"] = f"MemoryError: {exc} (OOM during TL load or forward)"
    print(f"  FAILED (OOM): {tl_s['failure_reason']}")
except Exception as exc:
    tl_s["failure_reason"] = f"{type(exc).__name__}: {exc}"
    print(f"  FAILED: {tl_s['failure_reason']}")
    if not isinstance(exc, (RuntimeError, NotImplementedError, ValueError)):
        print(tb.format_exc())

# ── Comparison ─────────────────────────────────────────────────────────────────
print("\n[Comparison] HF float32 own_logit_lens vs TL float32 logit_lens")
comp = summary["comparison"]

if not tl_s.get("success") or len(tl_logits_by_layer) != K_hf + 1:
    reason = tl_s.get(
        "failure_reason",
        f"TL computed {len(tl_logits_by_layer)}/{K_hf + 1} layers",
    )
    comp["failure_reason"] = reason
    comp["comparison_available"] = False
    print(f"  SKIP: {reason}")
else:
    comp["attempted"] = True
    comp["comparison_available"] = True

    rows: list[dict] = []
    max_diffs: list[float] = []
    mean_diffs: list[float] = []

    for k in range(K_hf + 1):
        own_k = hf_logits_by_layer[k]
        tl_k = tl_logits_by_layer[k]
        diff = (own_k - tl_k).abs()
        max_d = diff.max().item()
        mean_d = diff.mean().item()
        max_diffs.append(max_d)
        mean_diffs.append(mean_d)
        own_top1 = int(own_k.argmax().item())
        tl_top1 = int(tl_k.argmax().item())
        rows.append({
            "layer_index": k,
            "max_abs_diff_to_own": max_d,
            "mean_abs_diff_to_own": mean_d,
            "own_top1_token_id": own_top1,
            "tl_top1_token_id": tl_top1,
            "own_top1_piece": tokenizer.decode([own_top1]),
            "tl_top1_piece": tokenizer.decode([tl_top1]),
            "top1_match": own_top1 == tl_top1,
            "own_top1_logit": float(own_k[own_top1].item()),
            "tl_top1_logit": float(tl_k[tl_top1].item()),
        })

    max_all = max(max_diffs)
    mean_all = sum(mean_diffs) / len(mean_diffs)
    max_diff_layer = int(max_diffs.index(max_all))
    num_top1_matches = sum(1 for r in rows if r["top1_match"])

    csv_path = outputs_dir / "prelim_compare_transformerlens_float32_layer_diffs.csv"
    csv_fields = [
        "layer_index", "max_abs_diff_to_own", "mean_abs_diff_to_own",
        "own_top1_token_id", "tl_top1_token_id",
        "own_top1_piece", "tl_top1_piece", "top1_match",
        "own_top1_logit", "tl_top1_logit",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved: {csv_path}")

    comp.update({
        "max_abs_diff_to_own": max_all,
        "mean_abs_diff_to_own": mean_all,
        "max_diff_layer_index": max_diff_layer,
        "num_layers_compared": K_hf + 1,
        "num_top1_matches": num_top1_matches,
        "success_tol_1e_3": max_all <= 1e-3,
        "success_tol_1e_2": max_all <= 1e-2,
        "success_tol_1e_1": max_all <= 1e-1,
        "layer_diff_csv": str(csv_path),
    })

    print(f"  max_abs_diff    = {max_all:.4e}")
    print(f"  mean_abs_diff   = {mean_all:.4e}")
    print(f"  max_diff at layer {max_diff_layer}")
    print(f"  top1_matches    = {num_top1_matches}/{K_hf + 1}")
    print(f"  success (1e-3)  = {comp['success_tol_1e_3']}")
    print(f"  success (1e-2)  = {comp['success_tol_1e_2']}")
    print(f"  success (1e-1)  = {comp['success_tol_1e_1']}")

# ── Save summary JSON ──────────────────────────────────────────────────────────
summary_json = outputs_dir / "prelim_compare_transformerlens_float32_summary.json"
with summary_json.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\nsaved: {summary_json}")
print("Done.")
