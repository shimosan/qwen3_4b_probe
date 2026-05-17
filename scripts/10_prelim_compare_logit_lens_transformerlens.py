# 自前の logit lens 実装と TransformerLens の logit lens を比較し、実装の正確性を検証する。
# HF モデル（float16/MPS）と TransformerLens（float32/CPU）の dtype 差による誤差も確認する。
# tuned-lens は Qwen3 未対応のため不動作。[3] に記録コメントのみ残している。
# 出力: outputs/prelim_compare_logit_lens_transformerlens_summary.json
# 環境: llm2026-dev（transformer-lens が必要）

from __future__ import annotations

import csv
import gc
import importlib.metadata as ilm
import json
import traceback as tb

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_config, resolve_outputs_dir

# ── Config ────────────────────────────────────────────────────────────────────
cfg = load_config()
model_id = cfg["model_id"]
prompt = cfg["default_prompt"]
attn_impl = cfg["attn_implementation"]

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
pos = seq_len - 1

pos_token_id = int(inputs["input_ids"][0, pos].item())
pos_raw_token = str(tokenizer.convert_ids_to_tokens([pos_token_id])[0])
pos_piece = tokenizer.decode([pos_token_id])

print(f"input length : {seq_len} tokens")
print(f"selected pos : {pos}  token_id={pos_token_id}  piece={pos_piece!r}")

# ── HF Model ──────────────────────────────────────────────────────────────────
print("\nLoading HF model...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=dtype,
    attn_implementation=attn_impl,
)
model.to(device).eval()  # type: ignore[union-attr]
num_params = sum(p.numel() for p in model.parameters())
hf_model_dtype_str = str(next(model.parameters()).dtype)
print(f"  parameters: {num_params:,}")
print(f"  HF model dtype: {hf_model_dtype_str}")

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
hf_logits_dtype_str = str(outputs.logits.dtype)
K = len(model.model.layers)
if len(hs) != K + 1:
    raise RuntimeError(f"Expected {K + 1} hidden states, got {len(hs)}")
print(f"  K={K}, len(hidden_states)={len(hs)} ✓")

# ── [2] own logit lens ────────────────────────────────────────────────────────
def compute_own_logit_lens_selected_position(
    model: AutoModelForCausalLM,
    outputs,
    pos: int,
) -> list[torch.Tensor]:
    """Return float32 CPU logit tensor at `pos` for each k=0..K.

    k < K:  readout = model.model.norm(hs[k][:, pos:pos+1, :])[:, 0, :]
    k = K:  readout = hs[K][:, pos, :]   (already post-norm)
    Each tensor has shape [vocab_size].
    """
    hs_local = outputs.hidden_states
    K_local = model.config.num_hidden_layers  # type: ignore[union-attr]
    if len(hs_local) != K_local + 1:
        raise RuntimeError(f"Expected {K_local + 1} hidden states, got {len(hs_local)}")
    result: list[torch.Tensor] = []
    for k in range(K_local + 1):
        with torch.no_grad():
            if k < K_local:
                readout = model.model.norm(hs_local[k][:, pos:pos + 1, :])[:, 0, :]  # type: ignore[union-attr]
            else:
                readout = hs_local[K_local][:, pos, :]
            logits_k = model.lm_head(readout.to(device)).float()  # type: ignore[union-attr]
        result.append(logits_k[0].cpu())  # [vocab], float32
    return result


print("\n[2] own logit lens")
own_logits_by_layer = compute_own_logit_lens_selected_position(model, outputs, pos)  # type: ignore[arg-type]
print(f"  computed {len(own_logits_by_layer)} logit vectors (k=0..{K})")

# sanity: k=K selected position
diff_k_pos = (
    own_logits_by_layer[K] - outputs.logits[:, pos, :].float().cpu()[0]
).abs().max().item()
print(f"  own[K] vs outputs.logits[pos]: max_abs_diff = {diff_k_pos:.4e}")

# sanity: full sequence
with torch.no_grad():
    lm_full = model.lm_head(hs[-1].to(device)).float()
diff_full = (lm_full.cpu() - outputs.logits.float().cpu()).abs().max().item()
del lm_full
print(f"  lm_head(hs[-1]) vs outputs.logits (full seq): max_abs_diff = {diff_full:.4e}")

own_summary: dict = {
    "success": True,
    "num_layers": K,
    "num_hidden_states": len(hs),
    "hf_model_dtype": hf_model_dtype_str,
    "hf_logits_dtype": hf_logits_dtype_str,
    "final_selected_pos_max_abs_diff": diff_k_pos,
    "final_full_sequence_max_abs_diff": diff_full,
    "reference_description": (
        "k<K: model.model.norm(hs[k][:,pos:pos+1,:])[:,0,:] -> model.lm_head; "
        "k=K: hs[K][:,pos,:] (already post-norm) -> model.lm_head"
    ),
}

# ── [3] tuned-lens（不動作記録） ──────────────────────────────────────────────
# tuned-lens 0.2.0 を試みたが Qwen3 は未対応（NotImplementedError）。
# tuned_lens.model_surgery.get_final_norm の対応モデルは
# OPT / GPTNeoX / Bloom / GPT2 / GPTNeo / GPTJ / Llama のみ。
# Qwen3 に対して LogitLens.from_model() が NotImplementedError を送出する。
# 再現不要のため実行コードは削除し、記録のみ残す。

# ── [4] TransformerLens ───────────────────────────────────────────────────────
# TransformerLens 3.2.1 は Qwen/Qwen3-4B を OFFICIAL_MODEL_NAMES に含む。
# fold_ln=False / center_writing_weights=False / center_unembed=False で
# 重みの変換を最小限にし、own_logit_lens との数値比較を可能にする。
# TL logit lens: resid_post[k] -> ln_final -> unembed  (k=0..K-1)
#                hook_embed     -> ln_final -> unembed  (k=0, embed only)
print("\n[4] TransformerLens")
transformer_lens_summary: dict = {
    "import_ok": False,
    "attempted": False,
    "success": False,
    "comparison_available": False,
}
comparison_own_vs_tl: dict = {
    "attempted": False,
    "comparison_available": False,
    "success": False,
}

try:
    import transformer_lens as tl_pkg
    from transformer_lens import HookedTransformer
    from transformer_lens.supported_models import OFFICIAL_MODEL_NAMES

    transformer_lens_summary["import_ok"] = True
    transformer_lens_summary["version"] = ilm.version("transformer-lens")
    transformer_lens_summary["source_path"] = tl_pkg.__file__
    print(f"  transformer_lens: import OK  version={transformer_lens_summary['version']}")

    if model_id not in OFFICIAL_MODEL_NAMES:
        raise NotImplementedError(
            f"{model_id} is not in TransformerLens OFFICIAL_MODEL_NAMES"
        )
    print(f"  {model_id} is in OFFICIAL_MODEL_NAMES ✓")

    transformer_lens_summary["attempted"] = True

    # fold_ln=False : RMSNorm 重みを attention/MLP 重みに畳み込まない
    # center_writing_weights=False : 重みの列平均センタリングをしない
    # center_unembed=False : unembed 行平均センタリングをしない
    # これにより HF の重みと数値的に近い状態を保つ
    print(
        "  Loading HookedTransformer"
        " (fold_ln=False, center_writing_weights=False, center_unembed=False) ..."
    )
    # from_pretrained の **kwargs として trust_remote_code を直接渡す
    # (from_pretrained_kwargs は **kwargs 形式のため dict 渡しは不可)
    tl_model = HookedTransformer.from_pretrained(
        model_id,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        default_prepend_bos=False,
        dtype=torch.float32,  # type: ignore[arg-type]
        trust_remote_code=True,
    )
    tl_model.eval()
    K_tl = tl_model.cfg.n_layers
    d_model_tl = tl_model.cfg.d_model
    d_vocab_tl = tl_model.cfg.d_vocab
    tl_model_dtype_str = str(next(tl_model.parameters()).dtype)
    print(f"  HookedTransformer: n_layers={K_tl}, d_model={d_model_tl}, d_vocab={d_vocab_tl}")
    print(f"  TL model dtype: {tl_model_dtype_str}")

    # config 一致確認
    hf_n_layers = K
    hf_hidden_size = model.config.hidden_size
    hf_vocab_size = model.config.vocab_size
    n_layers_match = K_tl == hf_n_layers
    d_model_match = d_model_tl == hf_hidden_size
    d_vocab_match = d_vocab_tl == hf_vocab_size
    transformer_lens_summary["config_check"] = {
        "tl_n_layers": K_tl,
        "tl_d_model": d_model_tl,
        "tl_d_vocab": d_vocab_tl,
        "hf_num_layers": hf_n_layers,
        "hf_hidden_size": hf_hidden_size,
        "hf_vocab_size": hf_vocab_size,
        "n_layers_match": n_layers_match,
        "d_model_match": d_model_match,
        "d_vocab_match": d_vocab_match,
    }
    print(
        f"  config match: n_layers={n_layers_match}"
        f"  d_model={d_model_match}"
        f"  d_vocab={d_vocab_match}"
    )

    if not n_layers_match:
        comparison_own_vs_tl["failure_reason"] = (
            f"n_layers mismatch: TL={K_tl}, HF={hf_n_layers}"
        )
        comparison_own_vs_tl["comparison_available"] = False
        print(f"  SKIP comparison: {comparison_own_vs_tl['failure_reason']}")
        raise RuntimeError(comparison_own_vs_tl["failure_reason"])
    if not d_vocab_match:
        comparison_own_vs_tl["failure_reason"] = (
            f"d_vocab mismatch: TL={d_vocab_tl}, HF={hf_vocab_size}"
        )
        comparison_own_vs_tl["comparison_available"] = False
        print(f"  SKIP comparison: {comparison_own_vs_tl['failure_reason']}")
        raise RuntimeError(comparison_own_vs_tl["failure_reason"])

    # chat template で構築済みの token 列をそのまま渡す（BOS 等を二重追加しない）
    tokens_cpu = inputs["input_ids"].cpu()
    transformer_lens_summary["token_check"] = {
        "tokens_shape": list(tokens_cpu.shape),
        "first_token_id": int(tokens_cpu[0, 0].item()),
        "last_token_id": int(tokens_cpu[0, -1].item()),
        "selected_position_token_id": pos_token_id,
        "input_ids_equal": True,
    }
    print(f"  run_with_cache on {tokens_cpu.shape[1]} tokens ...")
    with torch.no_grad():
        tl_logits_out, tl_cache = tl_model.run_with_cache(
            tokens_cpu,
            prepend_bos=False,
            remove_batch_dim=False,
        )
    tl_logits_dtype_str = str(tl_logits_out.dtype)  # type: ignore[union-attr]
    transformer_lens_summary["dtype_info"] = {
        "hf_model_dtype": hf_model_dtype_str,
        "tl_model_dtype": tl_model_dtype_str,
        "hf_logits_dtype": hf_logits_dtype_str,
        "tl_logits_dtype": tl_logits_dtype_str,
    }
    print(f"  HF logits dtype: {hf_logits_dtype_str}  TL logits dtype: {tl_logits_dtype_str}")

    # selected position での TL logit lens を各層 (k=0..K) について計算する。
    # 対応関係:
    #   own hs[0]   = embed tokens   ↔  TL cache['hook_embed']
    #   own hs[k]   (k=1..K-1)       ↔  TL cache['resid_post', k-1]
    #   own hs[K]   (post-norm)       ↔  TL ln_final(cache['resid_post', K-1])
    # TL logit lens は resid / embed を ln_final にかけてから unembed する。
    # own の k=K は hs[K] が既に post-norm なので TL と convention が一致する
    # (TL も ln_final を通した後の値と等価なため)。
    tl_logits_by_layer: list[torch.Tensor] = []
    tl_layer_errors: list[str] = []

    for k in range(K_tl + 1):
        with torch.no_grad():
            try:
                if k == 0:
                    resid_3d = tl_cache["hook_embed"][:, pos:pos + 1, :]
                else:
                    resid_3d = tl_cache["resid_post", k - 1][:, pos:pos + 1, :]
            except KeyError as ke:
                tl_layer_errors.append(f"k={k}: KeyError {ke}")
                break

            normed_3d = tl_model.ln_final(resid_3d)   # [1, 1, d_model]
            logits_3d = tl_model.unembed(normed_3d)    # [1, 1, d_vocab]
            tl_logits_by_layer.append(logits_3d[0, 0].float().cpu())  # [d_vocab]

    n_computed = len(tl_logits_by_layer)
    print(f"  computed TL logit lens for {n_computed} layers (expected {K + 1})")
    if tl_layer_errors:
        print(f"  layer errors: {tl_layer_errors}")

    transformer_lens_summary["success"] = True
    transformer_lens_summary["tl_n_layers"] = K_tl
    transformer_lens_summary["tl_d_model"] = d_model_tl
    transformer_lens_summary["num_tl_logit_lens_computed"] = n_computed
    transformer_lens_summary["comparison_available"] = n_computed == K + 1
    if tl_layer_errors:
        transformer_lens_summary["layer_errors"] = tl_layer_errors

    if transformer_lens_summary["comparison_available"]:
        comparison_own_vs_tl["attempted"] = True
        comparison_own_vs_tl["comparison_available"] = True

        layer_diff_rows: list[dict] = []
        max_diffs: list[float] = []
        mean_diffs: list[float] = []
        for k in range(K + 1):
            own_k = own_logits_by_layer[k]    # [vocab], float32, CPU
            tl_k = tl_logits_by_layer[k]      # [vocab], float32, CPU
            diff = (own_k - tl_k).abs()
            max_d = diff.max().item()
            mean_d = diff.mean().item()
            max_diffs.append(max_d)
            mean_diffs.append(mean_d)
            own_top1 = int(own_k.argmax().item())
            tl_top1 = int(tl_k.argmax().item())
            layer_diff_rows.append({
                "layer_index": k,
                "max_abs_diff_to_own": max_d,
                "mean_abs_diff_to_own": mean_d,
                "own_top1_token_id": own_top1,
                "tl_top1_token_id": tl_top1,
                "own_top1_piece": tokenizer.decode([own_top1]),
                "tl_top1_piece": tokenizer.decode([tl_top1]),
                "top1_match": own_top1 == tl_top1,
            })

        max_all = max(max_diffs)
        mean_all = sum(mean_diffs) / len(mean_diffs)
        max_diff_layer = int(max_diffs.index(max_all))
        num_top1_matches = sum(1 for r in layer_diff_rows if r["top1_match"])

        # per-layer CSV を保存
        layer_diff_csv = (
            outputs_dir / "prelim_compare_existing_logit_lens_transformerlens_layer_diffs.csv"
        )
        csv_fields = [
            "layer_index", "max_abs_diff_to_own", "mean_abs_diff_to_own",
            "own_top1_token_id", "tl_top1_token_id",
            "own_top1_piece", "tl_top1_piece", "top1_match",
        ]
        with layer_diff_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(layer_diff_rows)
        print(f"  saved: {layer_diff_csv}")

        comparison_own_vs_tl["layers_compared"] = K + 1
        comparison_own_vs_tl["num_layers_compared"] = K + 1
        comparison_own_vs_tl["max_abs_diff_to_own"] = max_all
        comparison_own_vs_tl["mean_abs_diff_to_own"] = mean_all
        comparison_own_vs_tl["mean_diff_across_layers"] = mean_all
        comparison_own_vs_tl["max_diff_layer_index"] = max_diff_layer
        comparison_own_vs_tl["num_top1_matches"] = num_top1_matches
        comparison_own_vs_tl["layer_diff_csv"] = str(layer_diff_csv)
        comparison_own_vs_tl["tolerance_used"] = 1e-3
        comparison_own_vs_tl["success"] = max_all <= 1e-3
        print(f"  comparison: max_abs_diff={max_all:.4e}  mean_abs_diff={mean_all:.4e}")
        print(f"  max_diff at layer {max_diff_layer}  top1_matches={num_top1_matches}/{K + 1}")
        print(
            f"  own == transformer_lens: {comparison_own_vs_tl['success']}"
            f" (tol={comparison_own_vs_tl['tolerance_used']})"
        )
    else:
        reason = (
            f"TL logit lens computed for {n_computed} layers, expected {K + 1}. "
            f"Errors: {tl_layer_errors}"
        )
        comparison_own_vs_tl["failure_reason"] = reason
        print(f"  comparison not available: {reason}")

except MemoryError as exc:
    fail_msg = f"MemoryError: {exc} (OOM during TL model load or run_with_cache)"
    transformer_lens_summary.setdefault("failure_reason", fail_msg)
    comparison_own_vs_tl.setdefault("failure_reason", fail_msg)
    print(f"  FAILED (OOM): {fail_msg}")
except Exception as exc:
    fail_msg = f"{type(exc).__name__}: {exc}"
    transformer_lens_summary.setdefault("failure_reason", fail_msg)
    comparison_own_vs_tl.setdefault("failure_reason", fail_msg)
    print(f"  FAILED: {fail_msg}")
    if not isinstance(exc, (NotImplementedError, RuntimeError, ValueError)):
        print(tb.format_exc())

# ── [5] TransformerLens dtype-matched 比較 (float16) ─────────────────────────
# HF model は float16 (MPS) で動作している。
# TL を同じ float16 でロードして比較し、dtype 差が誤差の主因かを確認する。
# float32 版 TL model は前セクションで使用済みのため、不要なら gc で解放してから試みる。
print("\n[5] TransformerLens dtype-matched comparison (float16)")
comparison_dtype_matched: dict = {
    "attempted": False,
    "success": False,
    "dtype_strategy": f"TL float16 to match HF {hf_model_dtype_str}",
}

try:
    # 前セクションの float32 TL model を解放してからロード
    for _name in ("tl_model", "tl_cache", "tl_logits_out", "tl_logits_by_layer"):
        if _name in globals():
            del globals()[_name]
    gc.collect()

    from transformer_lens import HookedTransformer  # already imported, no-op

    print("  Loading HookedTransformer with dtype=torch.float16 ...")
    tl_model_f16 = HookedTransformer.from_pretrained(
        model_id,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        default_prepend_bos=False,
        dtype=torch.float16,  # type: ignore[arg-type]
        trust_remote_code=True,
    )
    tl_model_f16.eval()
    tl_f16_dtype_str = str(next(tl_model_f16.parameters()).dtype)
    print(f"  TL float16 model dtype: {tl_f16_dtype_str}")

    comparison_dtype_matched["attempted"] = True
    comparison_dtype_matched["tl_model_dtype"] = tl_f16_dtype_str

    tokens_cpu = inputs["input_ids"].cpu()
    with torch.no_grad():
        _, tl_cache_f16 = tl_model_f16.run_with_cache(
            tokens_cpu,
            prepend_bos=False,
            remove_batch_dim=False,
        )

    tl_logits_f16_by_layer: list[torch.Tensor] = []
    for k in range(K + 1):
        with torch.no_grad():
            try:
                if k == 0:
                    resid_3d = tl_cache_f16["hook_embed"][:, pos:pos + 1, :]
                else:
                    resid_3d = tl_cache_f16["resid_post", k - 1][:, pos:pos + 1, :]
            except KeyError:
                break
            normed_3d = tl_model_f16.ln_final(resid_3d)
            logits_3d = tl_model_f16.unembed(normed_3d)
            tl_logits_f16_by_layer.append(logits_3d[0, 0].float().cpu())

    del tl_model_f16, tl_cache_f16
    gc.collect()

    n_f16 = len(tl_logits_f16_by_layer)
    print(f"  computed {n_f16} logit vectors")

    if n_f16 == K + 1:
        max_diffs_f16 = []
        mean_diffs_f16 = []
        for k in range(K + 1):
            diff = (own_logits_by_layer[k] - tl_logits_f16_by_layer[k]).abs()
            max_diffs_f16.append(diff.max().item())
            mean_diffs_f16.append(diff.mean().item())

        max_f16 = max(max_diffs_f16)
        mean_f16 = sum(mean_diffs_f16) / len(mean_diffs_f16)
        comparison_dtype_matched["max_abs_diff_to_own"] = max_f16
        comparison_dtype_matched["mean_abs_diff_to_own"] = mean_f16
        comparison_dtype_matched["num_layers_compared"] = K + 1
        comparison_dtype_matched["tolerance_used"] = 1e-3
        comparison_dtype_matched["success"] = max_f16 <= 1e-3
        print(f"  max_abs_diff={max_f16:.4e}  mean_abs_diff={mean_f16:.4e}")
        print(f"  own == TL(float16): {comparison_dtype_matched['success']} (tol=1e-3)")
    else:
        comparison_dtype_matched["failure_reason"] = (
            f"Only {n_f16}/{K + 1} layers computed (cache key error)"
        )
        print(f"  SKIP: {comparison_dtype_matched['failure_reason']}")

except MemoryError as exc:
    comparison_dtype_matched["failure_reason"] = f"MemoryError: {exc} (OOM)"
    print(f"  FAILED (OOM): {comparison_dtype_matched['failure_reason']}")
except Exception as exc:
    comparison_dtype_matched["failure_reason"] = f"{type(exc).__name__}: {exc}"
    print(f"  FAILED: {comparison_dtype_matched['failure_reason']}")
    if not isinstance(exc, (RuntimeError, ValueError, NotImplementedError)):
        print(tb.format_exc())

# ── [6] Save summary JSON ──────────────────────────────────────────────────────
notes = [
    "Primary goal: verify own_logit_lens == transformer_lens logit_lens.",
    (
        f"transformer_lens {transformer_lens_summary.get('version', '?')}: "
        "Qwen/Qwen3-4B is listed in OFFICIAL_MODEL_NAMES as of 3.2.1."
    ),
    (
        "transformer_lens loading options: fold_ln=False, center_writing_weights=False, "
        "center_unembed=False to minimize weight transformation relative to HF checkpoint."
    ),
    "All logit comparisons are done in float32 on CPU regardless of model loading dtype.",
    (
        "HF float32 comparison (HF model loaded in float32 for direct weight-level comparison) "
        "is NOT run automatically due to memory/time cost. "
        "If further investigation is needed, see 11_prelim_compare_logit_lens_float32.py."
    ),
    (
        "tuned-lens 0.2.0 was attempted but does not support Qwen3 (NotImplementedError). "
        "Execution code removed; see [3] comment for details."
    ),
]

summary = {
    "model_id": model_id,
    "prompt": prompt,
    "attn_implementation": attn_impl,
    "device": device,
    "dtype": str(dtype),
    "input_length_tokens": seq_len,
    "selected_position": pos,
    "selected_position_token_id": pos_token_id,
    "selected_position_raw_token": pos_raw_token,
    "selected_position_piece": pos_piece,
    "own_logit_lens": own_summary,
    "transformer_lens": transformer_lens_summary,
    "comparison_own_vs_transformer_lens": comparison_own_vs_tl,
    "comparison_own_vs_transformer_lens_dtype_matched": comparison_dtype_matched,
    "notes": notes,
}

summary_json = outputs_dir / "prelim_compare_logit_lens_transformerlens_summary.json"
with summary_json.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\nsaved: {summary_json}")
print("\nDone.")
