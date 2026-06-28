# Experiment 11: 自前 logit lens vs TransformerLens — 両側 float32/CPU で完全一致確認

Script: [`scripts/11_prelim_compare_logit_lens_float32.py`](../scripts/11_prelim_compare_logit_lens_float32.py)
最終更新: 2026-05-12
ステータス: ✅ **`max_abs_diff = 6.7e-5`**（tolerance `1e-3` を 15 倍下回って合格）。top1 も 37/37 一致。[docs/10](10_compare_logit_lens_transformerlens.md) で残った fp16 量子化由来の差が、両側 fp32 化で解消されたことを確認。

---

## 1. 目的

[docs/10](10_compare_logit_lens_transformerlens.md) で確認した「自前 logit lens と TransformerLens は top1 完全一致だが、logit 値が fp16 量子化で 0.06 ズレる」問題を、**両側を float32 / CPU で動かして**解消できるかを試す。

実用上は top1 match で十分（[docs/10](10_compare_logit_lens_transformerlens.md) § 6-2）だが、

- 「実装そのものは完全一致するか？（fp16 を取り除けば数値も合うか？）」
- 「合わないなら convention の違いなど別原因がある」

を切り分けるための **数値精度ベンチマーク**。

---

## 2. 背景: なぜ float32/CPU か

[docs/10](10_compare_logit_lens_transformerlens.md) では:

- HF model: float16 / **MPS**
- TransformerLens model: float32 / **CPU**

の組み合わせで比較し、`max_abs_diff = 0.06` だった。これは:

1. **dtype の違い** (fp16 vs fp32) による量子化誤差
2. **device の違い** (MPS vs CPU) による matmul accumulator の違い

の 2 要因が混ざる。本実験では両方の要因を排除するため、**両方を float32 / CPU で動かす**。

メモリ的には:
- Qwen3-4B float32 = 16 GB（HF 単体で）
- TL float32 model + cache = さらに ≈ 20 GB

合計 36 GB なので、両モデルを **同時には載せず**、Phase A で HF だけロードして logit lens 計算 → `del` で解放 → Phase B で TL だけロードして同様、という流れにします。

---

## 3. 実験設定

| 項目 | 値 |
|---|---|
| 対象モデル | `Qwen/Qwen3-4B` |
| HF dtype / device | **float32 / CPU** |
| TL dtype / device | **float32 / CPU** |
| TL バージョン | 3.2.1 |
| TL 構成オプション | `fold_ln=False, center_writing_weights=False, center_unembed=False, default_prepend_bos=False, trust_remote_code=True` |
| プロンプト | デフォルト ([qwen3_4b_probe.json](../scripts/qwen3_4b_probe.json) の `default_prompt`、35 token) |
| 選択 position | 34（最後の `\n\n`）|
| tolerance | 3 段階 `1e-1, 1e-2, 1e-3` |

両 model を同時に保持しないため、Phase A 終了後に `del model; gc.collect(); torch.mps.empty_cache()` で完全解放してから Phase B に進む。

---

## 4. 方法

### Phase A: HF Qwen3-4B (float32 / CPU) own_logit_lens

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    attn_implementation="eager",
)
model.to("cpu").eval()

# 通常の forward
outputs = model(**inputs, output_hidden_states=True, use_cache=False)
hs = outputs.hidden_states

# 各層 own logit lens
for k in range(K + 1):
    if k < K:
        readout = model.model.norm(hs[k][:, pos:pos+1, :])[:, 0, :]
    else:
        readout = hs[K][:, pos, :]
    logits_k = model.lm_head(readout).float()
    hf_logits_by_layer.append(logits_k[0].cpu())

# Phase A 終了後、メモリ完全解放
del outputs, hs, model
gc.collect()
```

### Phase B: TransformerLens Qwen3-4B (float32 / CPU)

```python
tl_model = HookedTransformer.from_pretrained(
    model_id,
    fold_ln=False, center_writing_weights=False, center_unembed=False,
    default_prepend_bos=False,
    dtype=torch.float32,
    trust_remote_code=True,
)
tl_model.to("cpu").eval()

# 同じトークン列で run_with_cache
_, tl_cache = tl_model.run_with_cache(tokens_cpu, prepend_bos=False, remove_batch_dim=False)

# 各層 TL logit lens
for k in range(K + 1):
    if k == 0:
        resid_3d = tl_cache["hook_embed"][:, pos:pos+1, :]
    else:
        resid_3d = tl_cache["resid_post", k-1][:, pos:pos+1, :]
    normed_3d = tl_model.ln_final(resid_3d)
    logits_3d = tl_model.unembed(normed_3d)
    tl_logits_by_layer.append(logits_3d[0, 0].float().cpu())

del tl_model, tl_cache
gc.collect()
```

### Comparison

```python
for k in range(K + 1):
    diff = (hf_logits_by_layer[k] - tl_logits_by_layer[k]).abs()
    max_d, mean_d = diff.max(), diff.mean()
```

---

## 5. 結果

### 5-1. HF float32 own sanity check

`hf_float32` セクションの sanity check:

| 比較 | max abs diff | mean abs diff |
|---|---|---|
| `own[K]` vs `outputs.logits[pos]` (selected pos の再計算) | **5.4e-5** | 2.8e-6 |
| `lm_head(hs[-1])` vs `outputs.logits` (full sequence) | **0.0** | 0.0 |

→ full sequence は 0 完全一致。selected position だけスライスして `lm_head` を呼び直すと、cuBLAS / OpenBLAS の accumulation order の影響で `5.4e-5` 程度のズレが出る。これは float32 の精度限界内。

### 5-2. HF own vs TL — [outputs/prelim_compare_transformerlens_float32_layer_diffs.csv](../outputs/prelim_compare_transformerlens_float32_layer_diffs.csv)

37 layer 全比較:

| 量 | 値 |
|---|---|
| `max_abs_diff_to_own` (全層通しの最大) | **6.70e-5** |
| `mean_abs_diff_to_own` (全層平均) | 6.66e-6 |
| `num_top1_matches` | **37 / 37** ✓ |
| `max_diff_layer_index` | 16 |
| `success (tol=1e-3)` | **true** ✓ |
| `success (tol=1e-2)` | true |
| `success (tol=1e-1)` | true |

各層の差分の代表値（抜粋）:

| layer | max abs diff | own top1 logit | tl top1 logit | top1 match |
|---:|---:|---:|---:|---|
| 0 | 1.53e-5 | 147.55905 | 147.55905 | ✓ |
| 9 | 3.43e-5 | 13.65263 | 13.65262 | ✓ |
| 14 | 6.28e-5 | 12.35993 | 12.35994 | ✓ |
| 16 | **6.70e-5** | 13.05099 | 13.05098 | ✓ |
| 29 | 4.52e-5 | 14.86523 | 14.86521 | ✓ |
| 34 | 4.05e-5 | 14.82733 | 14.82731 | ✓ |
| 36 | 6.60e-5 | 26.91186 | 26.91185 | ✓ |

→ **全層で logit 値が 7 e-5 以下、top1 が完全一致**。

### 5-3. fp16 → fp32 で誤差が約 900x 改善

[docs/10](10_compare_logit_lens_transformerlens.md) の fp16 比較と並べてみると:

| 設定 | HF dtype | TL dtype | max_abs_diff | mean_abs_diff | top1 match |
|---|---|---|---:|---:|---:|
| docs/10 | fp16/MPS | fp32/CPU | 6.02e-2 | 5.24e-3 | 37/37 ✓ |
| docs/10 dtype-matched | fp16/MPS | fp16/CPU | 9.38e-2 | 6.70e-3 | 37/37 ✓ |
| **docs/11 (本実験)** | **fp32/CPU** | **fp32/CPU** | **6.70e-5** | 6.66e-6 | 37/37 ✓ |

→ fp16 → fp32 で **約 900 倍誤差が下がる**。fp16 の精度限界 (\~ 1e-3) に対し fp32 は \~ 1e-7 なので、機械精度のスケーリング通り。

### 5-4. 観察

1. **両実装は数値的に完全一致**（fp32 機械精度内）。これで「自前 logit lens は TransformerLens と等価な計算をしている」ことが**強い意味で確認**された。
2. **fp16 環境での 0.06 のズレは、純粋に量子化誤差**だった。implementation convention の違い（RMSNorm の式、unembed のセンタリング等）は無い。
3. **top1 は dtype に関わらず常に一致**（fp16 でも fp32 でも 37/37 match）。これは「top1 と次点の logit 差が量子化誤差（±0.06 程度）より大きい層では順位が保たれる」ことを意味し、本実験では全 37 層で top1 が一致した。一般には差が量子化誤差に埋もれる層で順位が入れ替わりうる点に注意。実用上、デモは fp16 で十分。
4. **max_diff_layer が 16**（fp16 比較では 36 だった）: fp32 では特定の中間層がたまたま最大になるが、絶対値は機械精度のオーダーなので意味のある "ピーク" ではない。

---

## 6. 解釈

### 6-1. 「実装の正当性」検証として完全成功

- [docs/10](10_compare_logit_lens_transformerlens.md): top1 一致だが logit 値は fp16 量子化で 0.06 ズレる → 「実装が正しいか fp16 のせいか不明」だった
- [docs/11](11_compare_logit_lens_float32.md)（本実験）: fp32 化で logit 値も 7e-5 まで一致 → **実装そのものは完全一致**

これで「自前 logit lens は TransformerLens と理論上同じ計算をしている」ことが確定的に言える。

### 6-2. 実用上の含意

- **デモ・nb02 では fp16 / MPS で十分**: top1 match が保証されているので、可視化や ranking 表示は fp16 の量子化誤差に影響されない
- **数値解析で完全一致が必要な場面では fp32 / CPU**: 例えば patching の差分 ([docs/12](12_residual_stream_patching.md)) で `Δ logit < 0.1` のような小さな量を扱うなら fp32 推奨
- **TransformerLens は今後使わない判断**: fp32 で完全一致することは確認できたので「自前実装で OK」と確定。TL の重い再パック処理を毎回回す意味はない

### 6-3. RMSNorm 周りに convention 差なし

「TransformerLens の `ln_final` の式と HF の `model.model.norm` の式に微妙な差があるのでは？」という疑念が [docs/10](10_compare_logit_lens_transformerlens.md) では残っていたが、fp32 で 6.7e-5 まで一致するので **実装は数式レベルで同一**。「分母の RMS の computation order」や「`fold_ln=False` 時の重みのコピー方法」も問題なし。

---

## 7. 出力ファイル

- [outputs/prelim_compare_transformerlens_float32_summary.json](../outputs/prelim_compare_transformerlens_float32_summary.json) — 全結果サマリ（環境情報 / Phase A / Phase B / comparison）
- [outputs/prelim_compare_transformerlens_float32_layer_diffs.csv](../outputs/prelim_compare_transformerlens_float32_layer_diffs.csv) — 37 行 × 10 列。`max_abs_diff_to_own`, `mean_abs_diff_to_own`, `own_top1_logit`, `tl_top1_logit`, `top1_match` 等

---

## 8. 注意事項

- **メモリ**: fp32 で Qwen3-4B 単体が 16 GB、TL でさらに加算。両 model を同時に保持しないよう Phase A / Phase B の二段運用は必須。M4 Max 64 GB なら順次なら問題なし。CPU のメモリが少ない環境 (16 GB) では実行不可。
- **CPU での forward 時間**: float32 / 35 token / Qwen3-4B で **数十秒〜数分**（M4 Max でも MPS の何倍も遅い）。fp32 で精度を取る代償。デモには向かない。
- **`own[K]` vs `outputs.logits[pos]` の selected pos 再計算で 5.4e-5 出る理由**: `lm_head` は内部で BLAS の matmul を呼ぶが、`hs[K][:, pos, :]` (shape `[1, d]`) と `hs[K]` (shape `[1, T, d]`) では呼び出されるカーネルや accumulation order が違う。fp32 でも floating point summation の非結合性で 1e-5 程度の差は出る。
- **early exit**: Phase A 失敗時は Phase B をスキップして JSON だけ書いて終了。OOM への対処。
- **本実験で確認できないこと**: GPU との比較。MPS と CUDA で同じ fp32 でも結果が完全一致するかは未確認（CUDA は cudnn を使うため、reduction order が異なる可能性）。
