# Experiment 10: 自前 logit lens vs TransformerLens — 実装の正当性検証（dtype 混在）

Script: [`scripts/10_prelim_compare_logit_lens_transformerlens.py`](../scripts/10_prelim_compare_logit_lens_transformerlens.py)
最終更新: 2026-05-12
ステータス: ⚠️ 部分成功。**top1 は全 37 層で一致**するが、logit 値の `max_abs_diff` は 0.06 で許容 1e-3 を超過（fp16 量子化が原因と推定 → [docs/11](11_compare_logit_lens_float32.md) で全 fp32 化して再検証）。

---

## 1. 目的

[docs/08_logit_lens.md](08_logit_lens.md) で書いた**自前の logit lens 実装**が正しいことを、サードパーティ実装の **TransformerLens** ([Nanda et al.](https://github.com/TransformerLensOrg/TransformerLens), v3.2.1) と数値比較で確認する。

「自前で書いた `model.model.norm(hs[k])` → `model.lm_head` という流れが、TransformerLens の `hook_resid_post[k-1]` → `ln_final` → `unembed` という流れと**同じ値**を出すか」を、各 layer $k = 0, \dots, 36$ について確認する。

ついでに tuned-lens も試したが、Qwen3 未対応で動作せず（後述）。

---

## 2. 背景: 比較する 2 つの実装

### 2-1. 自前実装（HF Transformers 直叩き）

```python
# k < K
readout = model.model.norm(hs[k][:, pos:pos+1, :])[:, 0, :]    # final RMSNorm
logits  = model.lm_head(readout).float()                        # unembedding

# k = K
readout = hs[K][:, pos, :]                                      # 既に post-norm
logits  = model.lm_head(readout).float()
```

詳細は [docs/08_logit_lens.md](08_logit_lens.md) § 4。

### 2-2. TransformerLens 実装

[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) は mechanistic interpretability のための薄いラッパで、HF モデルの重みを**自前の HookedTransformer** に再パックする。再パックの過程で次のような **オプショナル変換**を適用するが、本実験では数値比較のため**すべて OFF**:

| オプション | 役割 | 本実験での設定 |
|---|---|---|
| `fold_ln` | RMSNorm 重みを attention/MLP 重みに事前畳み込む | **False**（HF と同じ構造を保つ）|
| `center_writing_weights` | 各 layer の writing 重みを列平均センタリング | **False** |
| `center_unembed` | unembed 行列を行平均センタリング | **False** |
| `default_prepend_bos` | 自動 BOS 追加 | **False**（chat template 既にあり）|

TransformerLens 流の logit lens は:

```python
resid = tl_cache["hook_embed"]                  # k=0 の場合
resid = tl_cache["resid_post", k-1]              # k>=1 の場合
normed = tl_model.ln_final(resid)                # final RMSNorm
logits = tl_model.unembed(normed)                # = lm_head 相当
```

**自前との対応関係**:

| 自前 (HF) | TransformerLens |
|---|---|
| `hs[0]` (embedding 出力) | `cache["hook_embed"]` |
| `hs[k]` ($k = 1, \dots, K-1$, pre-norm) | `cache["resid_post", k-1]` |
| `model.model.norm(hs[k])` | `tl_model.ln_final(resid)` |
| `model.lm_head` | `tl_model.unembed` |
| `hs[K]` (post-norm) | `ln_final(cache["resid_post", K-1])` |

`hs[K]` が post-norm なのは [docs/07](07_hidden_state_mapping.md) で確認した HF の仕様。TransformerLens 側では `resid_post[K-1]` が pre-norm なので、両者を「`ln_final` 通過後」で揃えて比較する。

### 2-3. tuned-lens について

[tuned-lens (Belrose et al. 2023)](https://github.com/AlignmentResearch/tuned-lens) は untuned logit lens の改良版で、各層に **学習された affine 変換**を挟む。スクリプトで動作確認したところ、**Qwen3 は未対応で `NotImplementedError`** が出ます (`tuned_lens.model_surgery.get_final_norm` の対応モデルは OPT / GPTNeoX / Bloom / GPT2 / GPTNeo / GPTJ / Llama のみ)。実行コードは削除し、§ 3 のコメントだけ残してあります。

---

## 3. 実験設定

| 項目 | 値 |
|---|---|
| 対象モデル | `Qwen/Qwen3-4B` |
| プロンプト | デフォルト ([qwen3_4b_probe.json](../scripts/qwen3_4b_probe.json) の `default_prompt`、35 token) |
| 選択 position | 34（最後の `\n\n`）|
| HF 側 | mps / float16 / `attn_implementation=eager` |
| TL 側 | **CPU / float32**（デフォルト）と **CPU / float16**（dtype-matched 版） |
| TL バージョン | 3.2.1 (`OFFICIAL_MODEL_NAMES` に Qwen/Qwen3-4B が含まれる) |
| 全比較は `float32 CPU` に cast してから | |
| 許容 tolerance | `max_abs_diff <= 1e-3` |

---

## 4. 方法

### 4-1. 両実装を並走実行

1. HF model を float16 / MPS でロード、forward して `outputs.hidden_states` 取得（37 本）
2. TransformerLens model を float32 / CPU でロード、`run_with_cache` で全 cache 取得
3. 自前経路と TL 経路で **各 $k = 0, \dots, 36$ について position 34 の logit vector** を計算
4. 全 logits を float32 / CPU に cast して差分を取る

### 4-2. Config 整合性チェック

```python
config_check = {
    "tl_n_layers":  36 == 36,    # ✓
    "tl_d_model":   2560 == 2560,
    "tl_d_vocab":   151936 == 151936,
}
```

→ shape は完全一致（後述）。

### 4-3. Token id 一致確認

chat template で構築した `input_ids` をそのまま TL に渡す (`prepend_bos=False`)。`last_token_id`, `first_token_id`, `selected_position_token_id` 全部一致を summary に記録。

### 4-4. Per-layer comparison

```python
for k in range(K + 1):
    own_k = own_logits_by_layer[k]    # float32 CPU
    tl_k  = tl_logits_by_layer[k]     # float32 CPU
    diff  = (own_k - tl_k).abs()
    max_d, mean_d = diff.max(), diff.mean()
    own_top1 = own_k.argmax()
    tl_top1  = tl_k.argmax()
```

### 4-5. Dtype-matched 比較（fp16 TL）

TL も float16 でロードし直して同じ比較を実施。HF と TL の dtype を揃えれば、量子化誤差は打ち消し合うのか / それとも独立して累積するのかを見る。

---

## 5. 結果

### 5-1. Sanity check — [outputs/prelim_compare_logit_lens_transformerlens_summary.json](../outputs/prelim_compare_logit_lens_transformerlens_summary.json)

```text
own_logit_lens:
  final_selected_pos_max_abs_diff   : 0.0078125  (= 1/128, fp16 floor、selected position の再計算)
  final_full_sequence_max_abs_diff  : 0.0       (full sequence cache の再利用)

transformer_lens:
  version          : 3.2.1
  n_layers / d_model / d_vocab : 36 / 2560 / 151936  (all match HF)
  tokens shape     : [1, 35]
  input_ids_equal  : true
  num_tl_logit_lens_computed : 37  (k=0..36, 期待通り)
```

### 5-2. 自前 vs TL（float32）— [outputs/prelim_compare_existing_logit_lens_transformerlens_layer_diffs.csv](../outputs/prelim_compare_existing_logit_lens_transformerlens_layer_diffs.csv)

37 layer 全比較:

| 量 | 値 |
|---|---|
| `max_abs_diff_to_own` (全層通しの最大) | **0.0602** |
| `mean_abs_diff_to_own` (全層平均) | 0.0052 |
| `num_top1_matches` | **37 / 37** ✓ |
| `max_diff_layer_index` | 36（最終 post-norm 層） |
| `success` (≤ 1e-3) | **false**（tolerance 不達） |

各層の max diff（抜粋）:

| layer $k$ | max abs diff | mean abs diff | own top1 | TL top1 | top1 match |
|---:|---:|---:|---|---|---|
| 0 | 0.0591 | 0.0031 | `\n\n` | `\n\n` | ✓ |
| 5 | 0.0146 | 0.0022 | `ôm` | `ôm` | ✓ |
| 10 | 0.0226 | 0.0036 | `性和` | `性和` | ✓ |
| 20 | 0.0280 | 0.0046 | `抱歉` | `抱歉` | ✓ |
| 25 | 0.0328 | 0.0052 | `当` | `当` | ✓ |
| 29 | 0.0489 | 0.0061 | `当然` | `当然` | ✓ |
| 33 | 0.0379 | 0.0083 | `当然` | `当然` | ✓ |
| 34 | 0.0461 | 0.0072 | `言` | `言` | ✓ |
| 35 | 0.0382 | 0.0064 | `言` | `言` | ✓ |
| **36** | **0.0602** | 0.0075 | `言` | `言` | ✓ |

→ **logit の絶対値の差は 0.01–0.06 程度**。tolerance 1e-3 は超えるが、**top1 が全層で一致するので予測の意味では同じ結果**を出している。

### 5-3. 自前 (fp16) vs TL (fp16) — dtype-matched

| 量 | 値 |
|---|---|
| `max_abs_diff_to_own` | **0.0938** |
| `mean_abs_diff_to_own` | 0.0067 |
| `success` (≤ 1e-3) | false |

→ **TL を fp16 にしても差は減らない、むしろ増える**（0.060 → 0.094）。これは「fp16 同士でも HF MPS と TL CPU では量子化が独立に効くため、誤差が打ち消し合うのではなく **累積する**」ことを示唆。

### 5-4. 観察まとめ

1. **Top1 prediction は全層で一致**: 「`当然` が layer 29 で出る」「`言` が layer 34 で top1 になる」という意味的な結論は両実装で完全に一致。
2. **logit の絶対値は 0.01–0.06 のズレ**: 主因は **fp16 量子化**（HF が float16/MPS、TL が float32/CPU で計算）と推察。
3. **dtype を揃えても解消しない**: 両者を fp16 にすると逆に悪化（0.06 → 0.09）。MPS と CPU の matmul accumulator が違うため、fp16 誤差が独立に乗っかる。
4. **完全一致を取るには両方 fp32 が必要** → [docs/11_compare_logit_lens_float32.md](11_compare_logit_lens_float32.md) で実施。

---

## 6. 解釈

### 6-1. 自前実装の検証としては成功

「TransformerLens と top1 が全層一致、logit も 0.06 程度の量子化誤差以内で一致」というのは、**自前 logit lens の実装が正しい**ことの強い証拠。RMSNorm の適用タイミング（中間層 norm / 最終層 skip）の扱いが正しいことが確認できる。

### 6-2. 「success: false」の意味

スクリプトでは `tolerance = 1e-3` を success/fail の閾値にしているが、これは tight すぎる設定。**fp16 で動かしている時点で `1e-3` は原理的に達成不能**（fp16 の精度 floor は ≈ 2e-3〜1e-2 程度）。意味的な検証（top1 match）は通っているので、「実装上の bug ではなく数値精度上の限界」と読むのが正しい。

### 6-3. 最終層 (k=36) で max diff がピークの理由

`max_diff_layer_index = 36` は post-norm された hidden state を `lm_head` に通すケース。中間層は `model.model.norm(...)` を **自前と TL でそれぞれ独立に**通すため、norm 計算の量子化誤差が独立に乗る。一方 layer 36 では、自前は post-norm された `hs[K]` を使い、TL は `ln_final(resid_post[K-1])` を使う。両者は理論的に同じ値だが、norm 計算経路の違い（HF の `model.model.norm` vs TL の `tl_model.ln_final`）で float16 量子化の累積タイミングが微妙に違う。最後の層は logit の絶対値自体が大きい（`言` の logit は 26.875）ので、相対誤差が小さくても絶対誤差は大きくなる。

---

## 7. 応用への示唆

- **自前 logit lens は採用 OK**: top1 が TL と完全一致するので、[docs/08](08_logit_lens.md) の方法はそのまま nb02 に使える。
- **TransformerLens をデモには使わない判断材料**: 同等の結果が出るが、HF model のロード時間 + TL の再パック時間が積み上がるため、デモには重い。「自前で `model.model.norm` + `model.lm_head` を呼ぶだけ」の薄い実装で十分。
- **fp16 環境での意味的検証は top1 match で**: 数値完全一致は要求しない方が現実的。「top1 が一致 + entropy / rank の傾向が一致」を success criteria に。
- **fp32 完全一致は [docs/11](11_compare_logit_lens_float32.md) で**: もしどうしても `max_abs_diff < 1e-3` が必要なら、HF も TL も両方 float32 / CPU で動かす必要がある。

---

## 8. 出力ファイル

- [outputs/prelim_compare_logit_lens_transformerlens_summary.json](../outputs/prelim_compare_logit_lens_transformerlens_summary.json) — 全結果のサマリ JSON
- [outputs/prelim_compare_existing_logit_lens_transformerlens_layer_diffs.csv](../outputs/prelim_compare_existing_logit_lens_transformerlens_layer_diffs.csv) — 37 行 × 8 列。各層の `max_abs_diff_to_own`, `mean_abs_diff_to_own`, `own_top1` / `tl_top1` / `top1_match`
- [outputs/prelim_existing_lens_env_summary.txt](../outputs/prelim_existing_lens_env_summary.txt) — 環境情報

---

## 9. 注意事項

- **TL 3.2.1 が Qwen3-4B を支援**: それ以前のバージョンでは `OFFICIAL_MODEL_NAMES` に含まれず動かない可能性。pip install で最新を入れること。
- **`fold_ln=True` で動かすと数値が変わる**: RMSNorm 重みが attention/MLP に畳み込まれ、`ln_final` の挙動も変わる。本実験では数値比較が目的なので **False** 必須。実利用で interpretability の analysis を simplify したい場合は True にすることもある。
- **`prepend_bos=False` 必須**: chat template で既に `<|im_start|>` が先頭にあるので、TL に BOS を追加させると seq_len が 36 になり HF と異なる。
- **tuned-lens は Qwen3 未対応**: 2026-05 時点。実行コードは scripts から削除済みで、コメントのみ残っている。
- **MPS と CPU の混在**: HF は MPS、TL は CPU 上で動く。比較は両方 `.cpu().float()` してから。
- **メモリ消費**: TL の `run_with_cache` は全 layer の resid_post を保持するので **約 2 GB / 35 tokens × 36 layers × 2560 d_model × 4 bytes**。fp16 / MPS の HF model と並存させると ≈ 20 GB ピークになる。M4 Max 64 GB なら問題ないが、メモリ少ない環境では `del` + `gc.collect()` で float32 model を解放してから float16 を読み直す（スクリプトでそうしている）。
