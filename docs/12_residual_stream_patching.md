# Experiment 12: Residual stream activation patching — `The capital of Japan is` で「Tokyo は何層目で決まるか」

Script: [`scripts/12_residual_stream_patching.py`](../scripts/12_residual_stream_patching.py)
最終更新: 2026-05-15
ステータス: ✅ 37 patch site の recovery 曲線を取得。**layer 24 出力で `Paris → Tokyo` の top1 swap、recovery が 0.029 → 0.631 にジャンプ**を確認。

---

## 1. 目的

「**`The capital of Japan is` の答え `Tokyo` は、何層目の residual stream に書き込まれているのか**」を、activation patching で同定する。

具体的には:

- **clean prompt**: `"The capital of Japan is"` → 期待答え `" Tokyo"`
- **corrupt prompt**: `"The capital of France is"` → 期待答え `" Paris"`
- 各 layer の出力 (residual stream) を corrupt run の中で **clean の値に置き換え**たとき、top1 が `Paris` から `Tokyo` に切り替わる layer を特定する

これが mechanistic interpretability の典型手法 [Activation Patching / Causal Tracing (Meng et al. 2022; Heimersheim & Nanda 2024)](https://arxiv.org/abs/2202.05262) で、講義デモ ([notebooks/02_residual_stream_logit_lens_patching.ipynb](../notebooks/02_residual_stream_logit_lens_patching.ipynb)) の中核を成す。

---

## 2. 背景

### 2-1. Activation patching とは

Transformer の **residual stream** は、各 layer の出力として累積されていく中間表現:

$$
h^{(0)} = W_E x, \quad h^{(j+1)} = h^{(j)} + f_j(h^{(j)}) \quad (j = 0, \dots, K-1)
$$

ある「事実」（例: 「日本の首都」）は、計算の途中で複数の layer に分散して書き込まれる。**どの layer のどの token position に書き込まれているか**を特定するのが activation patching の動機。

手続き:

1. **Clean run**: `"The capital of Japan is"` で forward して全 layer の hidden state $h_{\text{clean}}^{(k)}$ を保存
2. **Corrupt run**: `"The capital of France is"` で forward して baseline 答え (`Paris`) を得る
3. **Patched run** (各 layer $k$, position $t$ について 1 つずつ):
   - corrupt prompt で forward
   - **layer $k$ の出力で、position $t$ の値を $h_{\text{clean}}^{(k)}[t]$ で上書き**（hook 経由）
   - patch 後の出力 logits を測定

「layer $k$ で patch すると `Tokyo` が予測されるようになる」ならば、**layer $k$ の時点で `Japan → Tokyo` を生成する情報が既に書き込まれていた**ことを意味する。

### 2-2. Metric と Recovery

「Tokyo と Paris のどちらが優位か」を測るスカラ指標:

$$
\text{metric} \;=\; \mathrm{logit}_{\text{Tokyo}} - \mathrm{logit}_{\text{Paris}} \quad (\text{at the last token position})
$$

- clean run では大きい正値（`Tokyo` が圧倒的優位）
- corrupt run では大きい負値（`Paris` が優位）

**Recovery（回復率）**を以下で定義:

$$
\text{recovery}(k) \;=\; \frac{\text{patched\_metric}(k) - \text{corrupt\_metric}}{\text{clean\_metric} - \text{corrupt\_metric}} \;\in\; [0, 1]
$$

- $\text{recovery} = 0$: patch しても効果なし（corrupt と同じ）→ 「layer $k$ に Tokyo 情報がまだない」
- $\text{recovery} = 1$: patch で完全に clean と同じになる → 「layer $k$ に Tokyo 情報が完全に揃っている」
- 0–1 の中間値: 部分的な復元

「**recovery が急に上昇する layer**」が、その情報が初めて書き込まれた layer の候補。

### 2-3. なぜ単一位置だけ patch するか

両 prompt とも `"The capital of [Japan|France] is"` の 5 token で構成され、**position 0, 1, 2 (`The`, `capital`, `of`) と position 4 (`is`) が共通**、**position 3 だけ違う**。

本実験では position 4 (last position) のみを patch する。理由:

1. 最後の next-token 予測に直接効く位置
2. position 3 の情報は attention 経由で position 4 に流れ込むので、最終的に position 4 を見ればよい
3. 全 (layer × position) のグリッドは notebook 02 で行う（[outputs/nb02_activation_patching_grid_recovery.png](../outputs/nb02_activation_patching_grid_recovery.png)）— 本実験はその予備調査

---

## 3. 実験設定

| 項目 | 値 |
|---|---|
| 対象モデル | `Qwen/Qwen3-4B` (K=36) |
| device / dtype | mps / float16 |
| `attn_implementation` | `eager` |
| Clean prompt | `"The capital of Japan is"` (5 token) |
| Corrupt prompt | `"The capital of France is"` (5 token) |
| Clean answer | `" Tokyo"` → token_id 26194 |
| Corrupt answer | `" Paris"` → token_id 12095 |
| Patch position | 最後の token (`is`, position 4)（**両 prompt で同じ位置**）|
| Patch site | $k = 0, \dots, K = 36$（37 site）|
| `use_cache` | False |

両答えとも単一トークンで encode される（`tokenizer.encode(" Tokyo")` → `[26194]`、`tokenizer.encode(" Paris")` → `[12095]`）ので、単純な logit 比較が成立する。

### Patch site の対応

| $k$ | patch site | 何を上書きするか |
|---|---|---|
| 0 | `embed_tokens` 出力 | $h^{(0)} = W_E x$ |
| $1 \leq k \leq K - 1$ | `layers[k-1]` 出力 | $h^{(k)}$ |
| $K = 36$ | `model.model.norm` 出力 | post-final-norm |

これは [docs/07](07_hidden_state_mapping.md) で確認した「hook 出力 = `hidden_states[k]`」関係に依拠（layer 35 だけは norm 後を比較）。

---

## 4. 方法

### 4-1. Clean run で全 hidden state を保存

```python
with torch.no_grad():
    clean_out = model(
        **clean_inputs,
        output_hidden_states=True,
        use_cache=False,
    )
clean_hs = clean_out.hidden_states     # (K+1) tensors, each [1, 5, 2560]

# clean_pos = 4 における各層の hidden state
clean_logits_pos = clean_out.logits[0, 4, :]
clean_metric = clean_logits_pos[Tokyo_id] - clean_logits_pos[Paris_id]
# → +11.70 (Tokyo 圧勝)
```

### 4-2. Corrupt baseline

```python
corrupt_out = model(**corrupt_inputs, use_cache=False)
corrupt_logits_pos = corrupt_out.logits[0, 4, :]
corrupt_metric = corrupt_logits_pos[Tokyo_id] - corrupt_logits_pos[Paris_id]
# → -11.98 (Paris 圧勝)

metric_range = clean_metric - corrupt_metric  # = 23.68
```

### 4-3. Patch hook の登録と実行

```python
def make_layer_hook(patch_vec, pos):
    def hook(module, inp, out):
        out = out.clone()
        out[0, pos, :] = patch_vec
        return out
    return hook

for k in range(K + 1):
    # k 番目の patch site に対応する clean 値を取り出す
    patch_vec = clean_hs[k][0, 4, :]   # position 4 だけ

    # 適切な module に hook を仕掛ける
    if k == 0:
        handle = model.model.embed_tokens.register_forward_hook(
            make_embed_hook(patch_vec, pos=4)
        )
    elif k < K:
        handle = model.model.layers[k-1].register_forward_hook(
            make_layer_hook(patch_vec, pos=4)
        )
    else:
        handle = model.model.norm.register_forward_hook(
            make_norm_hook(patch_vec, pos=4)
        )

    try:
        with torch.no_grad():
            patched_out = model(**corrupt_inputs, use_cache=False)
    finally:
        handle.remove()

    patched_logits_pos = patched_out.logits[0, 4, :]
    patched_metric = patched_logits_pos[Tokyo_id] - patched_logits_pos[Paris_id]
    recovery = (patched_metric - corrupt_metric) / metric_range
```

37 回の forward を回す（メモリ削減のため `output_hidden_states=False`、毎回 hook を `try/finally` で外す）。

---

## 5. 結果

### 5-1. Baseline — [outputs/prelim_residual_patching_summary.json](../outputs/prelim_residual_patching_summary.json)

| run | metric (Tokyo − Paris) | P(Tokyo) | P(Paris) | top1 |
|---|---:|---:|---:|---|
| clean (Japan) | **+11.70** | 0.8947 | 7.5e-6 | `Tokyo` ✓ |
| corrupt (France) | **−11.98** | 4.0e-6 | 0.6346 | `Paris` ✓ |

metric_range = 23.68（patching で完全 recovery したらこれだけ動かせる）。

### 5-2. Recovery 曲線 — [outputs/prelim_residual_patching_by_layer.csv](../outputs/prelim_residual_patching_by_layer.csv)

37 patch site の recovery（一部抜粋）:

| $k$ | patch_site | patched_metric | recovery | top1 | P(Tokyo) | P(Paris) |
|---:|---|---:|---:|---|---:|---:|
| 0 | `embed_tokens` | −11.98 | **0.000** | `Paris` | 4.0e-6 | 0.635 |
| 5 | layer_04 | −11.93 | 0.002 | `Paris` | 4.4e-6 | 0.668 |
| 10 | layer_09 | −13.27 | −0.054 | `Paris` | 1.7e-6 | 0.963 |
| 15 | layer_14 | −12.66 | −0.028 | `Paris` | 2.9e-6 | 0.908 |
| 20 | layer_19 | −12.28 | −0.013 | `Paris` | 4.1e-6 | 0.891 |
| 24 | layer_23 | −11.30 | **0.029** | `Paris` | 1.1e-5 | 0.869 |
| **25** | **layer_24** | **+2.95** | **0.631** | **`Tokyo`** ✓ | **0.777** | 0.041 |
| 26 | layer_25 | +3.23 | 0.643 | `Tokyo` | 0.799 | 0.031 |
| 27 | layer_26 | +6.12 | 0.764 | `Tokyo` | 0.895 | 2.0e-3 |
| 30 | layer_29 | +6.38 | 0.776 | `Tokyo` | 0.874 | 1.5e-3 |
| **32** | **layer_31** | **+11.45** | **0.990** | `Tokyo` | 0.900 | 9.6e-6 |
| 35 | layer_34 | +11.58 | 0.995 | `Tokyo` | 0.887 | 8.3e-6 |
| 36 | `norm` | +11.70 | **1.000** | `Tokyo` | 0.895 | 7.5e-6 |

### 5-3. 主要な観察

#### (a) 「Tokyo 情報の書き込み」は layer 24 で起きる

```text
k=24 (layer_23 output):  recovery = +0.029   top1 = "Paris"      P(Paris) = 0.87
k=25 (layer_24 output):  recovery = +0.631   top1 = "Tokyo"  ✓   P(Tokyo) = 0.78
                                              ↑ 急ジャンプ
```

**layer 24 の出力で初めて recovery が 60% を超え、top1 が `Tokyo` に flip する**。これが本実験の主要結果。

「`The capital of Japan is` → `Tokyo` の知識は、layer 24 の residual stream に書き込まれている」と言える。

#### (b) "Step function-like" な遷移

| layer 範囲 | recovery 帯 | 解釈 |
|---|---|---|
| 0–23 | ≈ 0（±0.05 程度のノイズ）| Tokyo 情報まだ無し |
| 24–30 | 0.63 → 0.78（緩やかな上昇）| 主要書き込み + refining |
| 31–34 | 0.99 → 0.995 | 残り 20% の "tail" |
| 35–36 | 0.995 → 1.0 | 完成 |

→ **layer 24 で 60% 一気に書き込まれ、layer 31 で残り 20% が確定**という 2 段構造。

#### (c) Layer 8–9 で negative recovery

```text
k=9 (layer_08): recovery = -0.080
k=10 (layer_09): recovery = -0.054
```

patching したら corrupt より **悪化**する（`Paris` の確率が 0.97 まで上がる）。これは「中間層では `Japan` の embedding 情報を patched で持ち込むと、まだ `Paris` 寄りに偏った後段計算と整合が取れず、噛み合って `Paris` がさらに強化される」結果。intermediate layer の patching が意味を持たない例として面白い。

#### (d) `norm` patch (k=36) で recovery = 1.0 完全一致

最終 RMSNorm 後の hidden state を全置換すると、`lm_head` 入力が完全に clean と同じになるので、定義上 recovery = 1.0 になる（sanity check として正しい）。

### 5-4. logit lens（[docs/08](08_logit_lens.md)）との関係

[docs/08](08_logit_lens.md) で観察した「答え `言` が top1 になるのは layer 34」というのは **logit lens（観察）の視点**。本実験は **patching（介入）の視点**で「layer 24 で答えが書き込まれる」と言っている。

両者は矛盾しないと解釈できる:

- **layer 24**: `Tokyo` を生成する情報は揃ったが、unembedding に直接読める形ではない
- **layer 24 → 34**: residual stream は次第に `lm_head` で読みやすい方向にラインアップされる
- **layer 34**: ようやく logit lens で `Tokyo` (`言`) が top1 に上がる

つまり「**情報の書き込みは layer 24、unembedding への射影完成は layer 34**」という 10 層の "polishing" 期間が存在する。これは [docs/14](14_qwen3_4b_transcoder_layers23_24_25.md) の MLP transcoder 解析でも観察された "pos=3 の Japan/France 区別 feature の活発化 → last position への伝播" 過程と整合的。

---

## 6. 図

このスクリプト自体は PNG を出力しません（CSV/JSON のみ）。可視化は [notebooks/02_residual_stream_logit_lens_patching.ipynb](../notebooks/02_residual_stream_logit_lens_patching.ipynb) が行います:

- [outputs/nb02_recovery_curve.png](../outputs/nb02_recovery_curve.png) — recovery 曲線（本実験データを直接 plot）
- [outputs/nb02_patching_probs.png](../outputs/nb02_patching_probs.png) — P(Tokyo) / P(Paris) の遷移
- [outputs/nb02_activation_patching_grid_recovery.png](../outputs/nb02_activation_patching_grid_recovery.png) — 全 (layer × position) のグリッド版

---

## 7. 応用への示唆

- **nb02 への直接寄与**: notebook 02 の patching セクションは本実験のデータをそのまま使う。「layer 24 が critical」という主張の数値的根拠。
- **講義デモ映え**: recovery 曲線で「**ある layer で急に top1 が `Paris` から `Tokyo` にひっくり返る**」が見える。Transformer の知識局在性を伝える絵として強力。
- **関連実験**:
  - [docs/14](14_qwen3_4b_transcoder_layers23_24_25.md): layer 23/24/25 を mwhanna MLP transcoder で詳細解析。本実験で同定された "critical layer 24" の前後を sparse feature 視点で確認。
  - [docs/15](15_qwen3_4b_transcoder_layer_sweep.md): 全 36 layer の sweep で、layer 23-25 が pos=3 (Japan/France 区別) のピークであることを確認。
- **講義での説明の流れ**:
  1. logit lens で「答えが top1 になるのは layer 34」を示す（[docs/08](08_logit_lens.md)）
  2. patching で「答えの情報が書き込まれるのは layer 24」を示す（本実験）
  3. その 10 層の差を「情報の書き込み vs unembedding への調整」として説明

---

## 8. 出力ファイル

- [outputs/prelim_residual_patching_summary.json](../outputs/prelim_residual_patching_summary.json) — 実験設定 + clean/corrupt baseline + patch_site mapping
- [outputs/prelim_residual_patching_by_layer.csv](../outputs/prelim_residual_patching_by_layer.csv) — 37 行 × 12 列。`layer_k`, `patch_site`, `clean_metric`, `corrupt_metric`, `patched_metric`, `recovery`, `patched_top1_piece` 等
- [outputs/prelim_residual_patching_topk.csv](../outputs/prelim_residual_patching_topk.csv) — 37 × top-10 = 370 行。各 patch site の next-token top-10
- [outputs/prelim_residual_patching_baseline_topk.csv](../outputs/prelim_residual_patching_baseline_topk.csv) — clean / corrupt baseline の top-10
- [outputs/prelim_residual_patching_prompt_tokens.csv](../outputs/prelim_residual_patching_prompt_tokens.csv) — clean / corrupt のトークン展開

---

## 9. 注意事項

- **Position 4 のみ patch**: 全 position × 全 layer のグリッドは notebook 02 が行う。本実験は計算量を抑えた予備調査。
- **`out = out.clone()` 必須**: hook 内で `out[0, pos, :] = patch_vec` を直接書くと、view への inplace write になり、後続の autograd で問題（本実験は no_grad なので動くが、安全のため clone）。Transformers 5.x では `Qwen3DecoderLayer.forward` が plain tensor を返すため、`hook(module, inp, out)` の `out` がそのまま tensor として使える。
- **fp16 数値の不安定さ**: 中間層 (k=8-15) で negative recovery (-0.02 〜 -0.08) が出る。本実験では fp32 で再実験しても同じ trend が出たため、これは fp16 量子化由来というより「intermediate patching が意味を持たない」現象と考えられる（本実験固有の観察に基づく解釈で、外部文献による裏付けではない点に注意）。
- **clean / corrupt の seq_len が同じ前提**: 両 prompt とも 5 token なので position 4 が両方の `is` に対応する。違う seq_len の prompt 間で patching すると position の対応が壊れる。
- **single-token answer 前提**: `" Tokyo"` / `" Paris"` が **単一トークン**で encode される偶然に依存。多トークン答えなら最初の token だけを使うか、log-prob を集約するか、追加の工夫が必要。
- **CLAUDE.md 違反でないこと**: `output_hidden_states=True` は clean run のみで使用、メモリ消費は 5 token × 37 hs × 2560 d_model × 2 bytes ≈ 1 MB と小さい。
