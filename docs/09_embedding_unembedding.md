# Experiment 09: Embedding $W_E$ と unembedding $W_U$ の関係 — tie_word_embeddings と effective unembedding

Script: [`scripts/09_embedding_unembedding.py`](../scripts/09_embedding_unembedding.py)
最終更新: 2026-05-11
ステータス: ✅ `data_ptr_equal = True`（同一 tensor）を確認。743 token subset の PCA / t-SNE coord も生成済み。

---

## 1. 目的

Qwen3-4B の **embedding 行列 $W_E$** と **unembedding 行列 $W_U$**（= `lm_head.weight`）の関係を数値で確かめる。具体的には:

1. **`tie_word_embeddings = True`** の意味を実装レベルで確認: 別の tensor がコピーされているのか、それとも同一メモリを共有しているのか
2. **「effective unembedding」** $W_U^{\text{eff}} = W_U \odot g$（$g$ は final RMSNorm の学習 gain）の意味と、$W_E$ / $W_U$ / $W_U^{\text{eff}}$ の **norm と方向の相互関係**
3. デモ用に、選択した 743 token subset の **PCA / t-SNE 2D 座標**を生成

これは [docs/08_logit_lens.md](08_logit_lens.md) で観察した「層 0 で input identity が top1 になる」現象や、後段の patching ([docs/12](12_residual_stream_patching.md)) の解釈の基礎になります。

---

## 2. 背景

### 2-1. `tie_word_embeddings` とは

Transformer 系では vocabulary が大きい場合 (Qwen3-4B では $V = 151{,}936$)、$W_E \in \mathbb{R}^{V \times d}$ と $W_U \in \mathbb{R}^{V \times d}$ を別パラメータにすると、それだけで $2 V d = 2 \times 151936 \times 2560 \approx 778$M params を消費します（モデル全体の 19%）。

**`tie_word_embeddings = True`** の設定では、$W_U \equiv W_E$（同一テンソル）として共有し、メモリと容量を半減できます。Llama / Qwen 系の小さめモデル（7B 以下）でよく使われます。

```python
W_E = model.model.embed_tokens.weight      # [V, d] = [151936, 2560]
W_U = model.lm_head.weight                 # [V, d] = [151936, 2560]
W_E.data_ptr() == W_U.data_ptr()            # True なら同一テンソル
```

### 2-2. Final RMSNorm gain $g$ と "effective unembedding"

`Qwen3Model.forward` の最終 RMSNorm は次の形:

$$
\mathrm{RMSNorm}(h)_j \;=\; g_j \cdot \frac{h_j}{\sqrt{\frac{1}{d}\sum_k h_k^2 + \varepsilon}}, \quad g \in \mathbb{R}^{d}, \ \varepsilon = 10^{-6}
$$

ここで $g$ は学習される **element-wise gain**（`model.model.norm.weight`）。

logit lens で「層 $k$ の hidden state $h^{(k)}$ から token $i$ の予測スコア」を出すには:

$$
\text{score}_i(h^{(k)}) \;=\; W_U[i] \cdot \mathrm{RMSNorm}(h^{(k)})
\;=\; \underbrace{(W_U[i] \odot g)}_{=: \ W_U^{\text{eff}}[i]} \cdot \frac{h^{(k)}}{\sqrt{\frac{1}{d}\sum_m (h^{(k)})_m^2 + \varepsilon}}
$$

つまり、**RMSNorm の gain $g$ を吸収した実効的な readout 方向**は $W_U^{\text{eff}}[i] = W_U[i] \odot g$ で表せます（分母の RMS スカラは hidden state ごとに異なるので分離できない）。本実験ではこれを `effective_unembedding` と呼びます。

「**この effective unembedding が、`tied` な $W_E$ や生 $W_U$ に対してどれだけ方向が一致しているか**」を観察するのが本実験の核心の一つです。

### 2-3. なぜ 743 token subset で見るのか

$V = 151{,}936$ token 全部の embedding を可視化するのは不可能 (PCA / t-SNE の計算量・図の可読性ともに)。**意味的に関心のある subset** を構成して観察します:

| subset 種類 | 個数（重複除去後）| 内訳 |
|---|---:|---|
| **prompt tokens** | 23 | デフォルトプロンプト `京都大学の…` の 35 token から重複除去 |
| **special tokens** | 26 | `<\|im_start\|>` `<\|im_end\|>` `<think>` `</think>` 等 |
| **logit_lens_topk tokens** | ≈ 510 | [docs/08](08_logit_lens.md) で出てきた各層 top-20 の和集合 |
| **manual tokens** | 22 | `言語`, `モデル`, `京都`, `AI`, `Transformer`, `softmax` 等の手動指定 |
| **background tokens** | 300 | 全 vocab からランダム sample (seed=0) |
| **合計（和集合）** | **743** | |

各 subset は `sources` 列でタグ付けされ、可視化時に色分けできます。

---

## 3. 実験設定

| 項目 | 値 |
|---|---|
| 対象モデル | `Qwen/Qwen3-4B` (V=151936, d=2560) |
| device / dtype | mps / float16（重みアクセスのみ。座標計算は CPU float32）|
| `BACKGROUND_SIZE` | 300（random sample, seed=0） |
| subset 合計 | 743 token |
| 可視化 | PCA + t-SNE（`init="pca"`, `perplexity=30`） |
| 入力行列 | $[3N \times d] = [2229 \times 2560]$（input / unembedding / effective_unembedding を stack） |

PCA / t-SNE は **3 種の表現を縦に積んだ合成行列 $X_{\text{all}}$** に対して **1 回**だけ fit します。同じ 2D 空間に投影することで、「同じ token の input embedding と (effective) unembedding がどれだけ離れているか」を視覚的に対比できます。

---

## 4. 方法

### 4-1. $W_E$ と $W_U$ の同一性確認

```python
W_E = model.model.embed_tokens.weight       # [V, d]
W_U = model.lm_head.weight                  # [V, d]

same_data_ptr = W_E.data_ptr() == W_U.data_ptr()
# 値の差分を chunk-wise に計算（フル diff tensor を作らない）
```

`data_ptr()` が一致すれば **同じメモリ領域**を参照していることが確定。値比較は CHUNK=4096 でストリーム計算（[V, d] = [151936, 2560] フル diff だと 1.5 GB float32）。

### 4-2. Effective unembedding の構築

```python
g = model.model.norm.weight                  # [d]
W_U_eff = W_U * g.unsqueeze(0)              # [V, d]
```

`g` は element-wise gain なので broadcast で簡単に吸収できます。

### 4-3. Per-token metadata の計算

各 subset token $i$ について:

```python
e  = W_E[i]               # input embedding
u  = W_U[i]               # unembedding (tie のため e に等しい)
eu = u * g                # effective unembedding

# Norms
e.norm(), u.norm(), eu.norm()

# Pairwise cosines
cos(e, u)    # tie なら 1.0
cos(e, eu)   # = cos(u, eu)（u = e なので）
cos(u, eu)
```

### 4-4. 2D 座標

```python
X_all = np.concatenate([W_E_subset, W_U_subset, W_U_eff_subset])   # [3N, d]
# PCA (sklearn)
X_pca = PCA(n_components=2).fit_transform(X_all)
# t-SNE (sklearn)
X_tsne = TSNE(n_components=2, init="pca", perplexity=30).fit_transform(X_all)
```

3 表現を縦積みすることで、同じ 2D 空間に投影される（同じ basis）。

---

## 5. 結果

### 5-1. $W_E \equiv W_U$（tie 確認）

[outputs/prelim_embedding_unembedding_summary.json](../outputs/prelim_embedding_unembedding_summary.json):

| 項目 | 値 |
|---|---|
| `W_E_shape` | `[151936, 2560]` |
| `W_U_shape` | `[151936, 2560]` |
| `tie_word_embeddings` | True |
| **`data_ptr_equal`** | **True** |
| `max_abs_diff` | 0.0 |
| `mean_abs_diff` | 0.0 |
| `torch_allclose` | True |

→ **`W_E` と `W_U` は同じメモリを指す同一テンソル**。コピーされて別物として存在しているのではなく、`lm_head.weight = embed_tokens.weight` という参照になっている。よって `cos(W_E[i], W_U[i]) = 1.0` が全 token で成立（CSV で確認すると 1.000001 ± 1e-6、これは fp16 → fp32 cast の量子化誤差）。

### 5-2. Norm の統計（743 token subset）

[outputs/prelim_embedding_unembedding_tokens.csv](../outputs/prelim_embedding_unembedding_tokens.csv):

| 量 | mean | std | min | median | max |
|---|---:|---:|---:|---:|---:|
| `input_norm` ($\|W_E[i]\|$) | 1.134 | 0.183 | 0.365 | 1.155 | 1.493 |
| `unembedding_norm` | **1.134** | 0.183 | 0.365 | 1.155 | 1.493 |
| `effective_unembedding_norm` ($\|W_U[i] \odot g\|$) | **3.210** | 0.457 | 1.409 | 3.262 | 4.174 |

観察:

- `input_norm = unembedding_norm` ぴったり（tie の自明な帰結）
- **effective unembedding は ≈ 2.83x 大きい**（gain $g$ の効果）。$\|W_U^{\text{eff}}[i]\| / \|W_U[i]\| \approx 2.83$ が全 token でほぼ一定 → $g$ は方向ベクトルを少しいびつにストレッチするが、概ね uniform に拡大
- **特殊トークン（`<|im_start|>` 等）は input_norm が小さい** (0.36–0.42)。学習中の頻度や初期化の影響と推察。`<|object_ref_start|>`〜`<|video_pad|>` は数値が完全一致 (0.364955) — 初期化のままで実質的に更新されていない可能性

### 5-3. Cosine の統計（input vs effective unembedding）

`cos(W_E[i], W_U[i] \odot g)` の分布:

| 統計 | 値 |
|---|---|
| mean | **0.972** |
| median | 0.975 |
| min | **0.877**（一部の special token） |
| max | 0.991 |

→ **$g$ は方向をほとんど変えない** が、完全に同じ方向ではない（cos = 1.0 ではなく 0.97–0.99）。「$g$ は dimension ごとに $W_U$ をいびつに重み付けする」ことで、わずかに方向を回転させている。特殊トークンで cos が 0.876 と低めになるのは、$W_U[i]$ の特定 dimension に強い偏りがあり、$g$ の不均一性が拡大して効くため、と推察できる。

### 5-4. 2D 座標 — [outputs/prelim_embedding_unembedding_coords.csv](../outputs/prelim_embedding_unembedding_coords.csv)

PCA + t-SNE 各 $3N = 2229$ 行 = 4458 行。各行に `(method, representation_type, token_id, raw_token, piece, sources, x, y, vector_norm)`。

このスクリプト自体は PNG を出力しません。可視化は notebook 側で行います（subset 別に色分け、3 表現を異なるマーカーでプロット、etc.）。

参考: 同じ token の `input_embedding` 座標と `effective_unembedding` 座標は、PCA 空間では near-coincident（cos = 0.97 だから）、t-SNE 空間では perplexity 設定により局所近傍として表れる傾向。

---

## 6. 応用への示唆

- **[docs/08_logit_lens.md](08_logit_lens.md) の現象「層 0 で input identity が top1」の説明**: $h^{(0)} = W_E[x_t]$ に `lm_head` ($W_U$) を当てると、$W_U[v] \cdot W_E[x_t] = W_E[v] \cdot W_E[x_t]$（tie のため）。これは $v = x_t$ で最大値を取りやすいので、層 0 の top1 は入力トークン自身になる。
- **用語の整理**: 「embedding と unembedding は同じ行列なのか」という素朴な疑問に、**Qwen3-4B では Yes（同一テンソル、tie）** と即答できる。Llama 3 8B などは tied ではない（別パラメータ）ことと対比して説明できる。
- **logit lens の implementation note**: 中間層 readout で正しい結果を得るには **必ず `model.model.norm` を適用してから `lm_head` に通す**。これは「effective unembedding が gain $g$ を含んでいて、生の $W_U$ で readout すると scale が ≈ 2.83x ずれる」ことの裏返し。
- **special token の取り扱い注意**: vocab の末尾に並ぶ `<|object_ref_*|>` `<|box_*|>` `<|vision_*|>` 等は **初期化のままで実質的に学習されていない**可能性がある（input_norm = 0.365 で全て同値）。logit lens で top1 にこれらが出てきたら「未訓練トークンが偶然 readout 方向に近かった」可能性を疑う。

---

## 7. 出力ファイル

- [outputs/prelim_embedding_unembedding_summary.json](../outputs/prelim_embedding_unembedding_summary.json) — モデル設定 + W_E/W_U 同一性 + subset 統計
- [outputs/prelim_embedding_unembedding_tokens.csv](../outputs/prelim_embedding_unembedding_tokens.csv) — 743 行 × 15 列。`is_*_token` flag, `input_norm`, `unembedding_norm`, `effective_unembedding_norm`, 3 種の cosine
- [outputs/prelim_embedding_unembedding_coords.csv](../outputs/prelim_embedding_unembedding_coords.csv) — 4458 行 × 9 列。PCA + t-SNE × 3 表現 × 743 token

---

## 8. 注意事項

- **fp16 計算誤差**: `cos(W_E[i], W_U[i])` は理論的に 1.0 だが、CSV では 1.000001 ± 1e-6。これは `e` `u` を float32 に cast してから内積を計算しているが、元の値が fp16 量子化されているため。実用上は無視可能。
- **`tie` モデル特有**: 本実験の結論（input_norm = unembedding_norm、cos = 1.0 など）は **Qwen3-4B が tied だから成立する**。tie していないモデルでは値が分かれる。
- **`effective_unembedding` は近似指標**: RMSNorm の分母（hidden state のスカラ RMS）は token ごとに違うので、$W_U^{\text{eff}}$ だけで完全な readout 方向を表すわけではない。「**学習 gain $g$ を吸収した方向**」の意味で使う。
- **PCA / t-SNE の結果はランダム性に依存**: `random_state=0` で固定しているが、scikit-learn のバージョンや BLAS によって誤差が出る。「同じデータでも plot が完全一致しない」のは正常。
- **PCA + t-SNE の比較**: PCA は線形射影で global 構造を保つ。t-SNE は近傍を保つ非線形。同じ subset でも見える「クラスタ」の意味が異なるので併用すると相補的。
