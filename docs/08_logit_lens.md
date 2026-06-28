# Experiment 08: Logit lens — 各層 hidden state を `lm_head` に通して層別 next-token 予測を見る

Script: [`scripts/08_logit_lens.py`](../scripts/08_logit_lens.py)
最終更新: 2026-05-11
ステータス: ✅ 全 37 layer index の top-20 / 各種 metric を取得。「答え `言` は layer 34 で初めて top1 になる」が確認できた。

---

## 1. 目的

「`The capital of Japan is` → `Tokyo`」のような next-token 予測を、**Transformer の最終層が完成形を吐き出す前から、中間層でもおおまかに見えているのではないか**という仮説を Qwen3-4B で確認する。これを可視化する道具が **logit lens**（[nostalgebraist 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens), [Belrose et al. 2023](https://arxiv.org/abs/2303.08112)）。

本実験では、ある選択した position の hidden state $h^{(k)} \in \mathbb{R}^{d_{\text{model}}}$ ($k = 0, 1, \dots, K$) を **すべての層について** unembedding (`lm_head`) に通し、層ごとの top-20 予測トークンと確率分布の遷移を観察する。

---

## 2. 背景: Logit lens とは何か

### 2-1. Transformer の出力経路

[docs/07](07_hidden_state_mapping.md) で確認した通り、Qwen3-4B の forward 経路は以下:

```text
input_ids ──► embed_tokens ──► hidden_states[0]
                              │
                              ▼
                       DecoderLayer 0..K-1 (K = 36)
                              │
                              ▼ pre-norm
                       model.model.norm     ◄── final RMSNorm
                              │
                              ▼ post-norm
                       hidden_states[K]
                              │
                              ▼
                       model.lm_head        ◄── unembedding $W_U \in \mathbb{R}^{V \times d}$
                              │
                              ▼
                          outputs.logits ∈ ℝ^(1 × T × V)
```

通常の forward は **最後の hidden state $h^{(K)}$** にしか `lm_head` を当てない。

### 2-2. Logit lens の数式

Logit lens は「中間層 $k$ の hidden state $h^{(k)}$ にも、最終 RMSNorm と `lm_head` をかけてみたら、その層が "考えている" 次トークン分布が見えるはずだ」という仮説に基づきます:

$$
\text{lens}(k, t) \;=\; \mathrm{softmax}\bigl(W_U \cdot \mathrm{RMSNorm}\bigl(h^{(k)}_t\bigr)\bigr) \;\in\; \Delta^{V-1}
$$

ここで:
- $k \in \{0, 1, \dots, K\}$ — layer index（0 は embedding 出力、$K$ は最終 RMSNorm 後）
- $t$ — 観察したい token position
- $W_U \in \mathbb{R}^{V \times d_{\text{model}}}$ — `lm_head` の重み（Qwen3 は `tie_word_embeddings=True` なので $W_U = W_E$）
- $\Delta^{V-1}$ — vocab 上の確率単体

### 2-3. RMSNorm の適用ポイントが微妙

§ 2-1 のとおり、`hidden_states[K]` だけは **既に RMSNorm 後**の値です（`Qwen3Model.forward` が norm を適用してから保存するため）。中間層 `hidden_states[k]` ($k < K$) は **pre-norm** の値。よって logit lens を統一的に計算するには:

| $k$ | `readout = ...` |
|---|---|
| $k = 0, 1, \dots, K-1$ | `model.model.norm(hs[k])` — 自前で RMSNorm 適用 |
| $k = K$ | `hs[K]` — そのまま（既に post-norm） |

その後 `lm_head(readout)` で logits を得る。この区別を取り違えると中間層の予測が変な方向にズレるので、Script 内では `norm_applied` flag を CSV に出して明示しています。

### 2-4. logit lens を読むときの 3 つの指標

各 layer $k$ について、選択 position $t$ で計算する量:

| 指標 | 定義 | 何を見るか |
|---|---|---|
| **top1 piece / prob** | $\arg\max_v \mathrm{lens}(k, t)_v$ と該当確率 | 「この層は何を予測しようとしているか」 |
| **entropy** (nats) | $-\sum_v p_v \log p_v$（小さいほど確信） | 分布の集中度 |
| **final top1 rank in this layer** | 最終層 ($k = K$) の top1 トークン $v^\star$ が、layer $k$ の確率順で何位か | 「最終答え `v^\star` がこの層では何位扱いか」 |

特に「**final top1 rank**」が後段でじりじり 1 に向かって減ってくる遷移パターンが、logit lens の典型的な見せ場です。

---

## 3. 実験設定

| 項目 | 値 |
|---|---|
| 対象モデル | `Qwen/Qwen3-4B` (K=36) |
| プロンプト | デフォルト ([qwen3_4b_probe.json](../scripts/qwen3_4b_probe.json) の `default_prompt`、35 token) |
| 選択 position $t$ | **34**（最後の token、`\n\n`、`<think></think>` の直後の generation 開始点） |
| device / dtype | mps / float16 |
| `attn_implementation` | `eager` |
| 各層で出力する top-k | 20 |
| layer index 範囲 | $k = 0, 1, \dots, K = 36$（37 段階） |

「最後の token = `\n\n`」というのは Qwen3 の chat template 展開後の構造で、ここの hidden state に `lm_head` を当てると **次に来るべき token**（answer の最初の token）の分布が出ます。

---

## 4. 方法

### 4-1. Forward して全 hidden state を取得

```python
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
    )
hs = outputs.hidden_states            # tuple of length K+1 = 37
```

### 4-2. Sanity check: $k = K$ で `lm_head(hs[K]) = outputs.logits` を確認

```python
logits_K_full = model.lm_head(hs[K])
diff_full = (logits_K_full - outputs.logits).abs().max()
# → 0.0
```

これは [docs/07](07_hidden_state_mapping.md) で確認済みの関係の再確認。`max_abs_diff = 0.0` で完全一致。なお、選択 position $t = 34$ のスライスだけで `lm_head` を呼び直すと `max_abs_diff = 0.0078125 = 1/128` （fp16 の精度限界、無視できる）。

### 4-3. 各層で logit lens を計算

```python
for k in range(K + 1):
    if k < K:
        # RMSNorm は position-independent なので、選択 position のスライスだけ norm
        readout = model.model.norm(hs[k][:, t:t+1, :])[:, 0, :]
        norm_applied = True
    else:
        readout = hs[K][:, t, :]      # 既に post-norm
        norm_applied = False

    logits = model.lm_head(readout).float()   # [1, V]
    probs = torch.softmax(logits[0], dim=-1)

    # top-20 取得
    top_vals, top_ids = torch.topk(probs, k=20)

    # entropy
    entropy = -(probs * torch.log(probs + 1e-9)).sum()

    # final layer の top1 トークン v* が、ここでは何位か
    v_star = final_top1_token_id
    final_rank_here = int((logits[0] > logits[0][v_star]).sum()) + 1
```

選択 position のスライスだけに norm を当てているのは、メモリ削減のため（RMSNorm は position 独立なので結果に影響なし）。

---

## 5. 結果

### 5-1. Layer 別 top1 と final top1 のランク推移 — [outputs/prelim_logit_lens_layer_metrics.csv](../outputs/prelim_logit_lens_layer_metrics.csv)

position $t = 34$（最後の token `\n\n`）における **layer 別 top1 トークン**と、**最終 top1 トークン `言` のランク**:

| layer $k$ | top1 piece | top1 prob | entropy | final `言` rank |
|---:|---|---:|---:|---:|
| 0 | `\n\n` | **1.0000** | 2e-28 | 6931 |
| 1 | `安抚` | 0.0280 | 8.70 | 99132 |
| 2 | `obble` | 0.0229 | 8.58 | 117234 |
| 3 | `uckle` | 0.0420 | 8.11 | 79528 |
| … | （ノイズトークンが top1）| | | |
| 21 | `語` | 0.0233 | 8.50 | 13582 |
| 22 | ` briefly` | 0.1330 | 7.48 | 10020 |
| 23 | `回答` | 0.0529 | 7.93 | 3227 |
| 24 | ` Yes` | 0.0416 | 7.54 | 2068 |
| 25 | `当` | 0.0318 | 7.93 | 496 |
| 26 | `亲爱的` | 0.0729 | 6.16 | 312 |
| 27 | `当` | 0.1166 | 6.28 | 156 |
| 28 | `当` | 0.1595 | 6.15 | 259 |
| **29** | **`当然`** | **0.9955** | **0.052** | 196 |
| 30 | `当然` | 0.9947 | 0.058 | 20 |
| 31 | `当然` | 0.9495 | 0.29 | **2** |
| 32 | `当然` | 0.9608 | 0.21 | 2 |
| 33 | `当然` | 0.6693 | 0.94 | 2 |
| **34** | **`言`** | **0.8431** | 0.56 | **1** |
| 35 | `言` | 0.9639 | 0.41 | 1 |
| 36 | `言` | 0.9639 | 0.21 | 1 |

### 5-2. 観察

#### (a) 層 0 は「入力トークンそのもの」が top1（identity-like readout）

$k = 0$ で top1 が `\n\n` (prob = 1.0) になっているのは、$h^{(0)} = W_E e_{x_t}$ という embedding 出力が、`lm_head` $W_U = W_E$ (`tie_word_embeddings=True`) との内積を取ると、対角成分（$x_t$ 自身）で大きい値になるため。**「層 0 の logit lens は input identity readout に等しい」**と言える。これ自体は意味的予測ではない。

#### (b) 層 1–20 はほぼノイズ

`安抚`, `obble`, `uckle`, `ôm`, `潜`, `談`, `TL`, `性和`, `有意`, `孜`, ... と支離滅裂。entropy ≈ 8–9 nats と高く、確信のない分布。final 答え `言` のランクは **数万位**でほぼ最下位群。**前段では「次に何を吐くべきか」がまったく決まっていない**。

#### (c) 層 21–28 で関連語が浮上、入れ替わりつつ "Chinese-ish noise" 期

- 層 21: `語`（言語の片割れ）が top1 だが prob は 0.023 と低い
- 層 22: ` briefly`（「短く説明して」のニュアンス）
- 層 23: `回答`、層 24: ` Yes`
- 層 25–28: `当` が散発的に top1

→ プロンプトの意味理解は層 20 台半ばから動き出すが、**まだ「言語モデルとは…」を始めるための最初の token を決められていない**。

#### (d) 層 29 で急に `当然` が prob 0.995 で top1（"early lock-in"）

これが logit lens で最もよく出る現象の一つ。**層 29 で entropy が 8.7 → 0.052 nats と劇的に低下**し、`当然` (= 「もちろん」、中国語的応答開始フレーズ）が確信を持って top1 になる。

ただし `当然` は**最終答えではない**。最終答えは `言`。実際 layer 29 では `言` のランクは 196 位とまだ低い。

#### (e) 層 31–33 で `言` のランクが 2 位まで上昇

`当然` が top1 のまま、final 答え `言` が裏で 196 → 20 → 2 → 2 → 2 と上がってくる。「上位 2 候補が拮抗しているが top1 はまだ `当然`」という状態。

#### (f) 層 34 で top1 swap：`当然` → `言`

**最後の 2 層（34, 35, 36）で `言` が逆転して top1 に**。これが「最終答えが top1 になる layer」。

→ Qwen3-4B における「答えが logit lens で見えるようになる層 = **34**」と判明。Qwen3 は全 36 layer なので、**ほぼ最後の最後で答えが決まる**典型的パターン。

### 5-3. 観察まとめ

```text
layer 0    ──► input identity readout (\n\n)
layer 1-20 ──► noise (中国語 random tokens)
layer 21-28──► プロンプト関連語が散発浮上 (語, briefly, 回答, Yes, 当)
layer 29-33──► "当然" early lock-in。final 答え '言' は裏でランクが上がる
layer 34-36──► final 答え '言' が top1 に確定
```

このパターンは [Belrose et al. 2023]（Tuned Lens）が論じる、各層の予測が最終的な出力分布へ次第に収束していく現象の典型例。「最終予測が出るのは最後の 2-3 層」というのも、4B クラスの instruct model でしばしば観察されます。

### 5-4. なぜ `当然` が一旦 top1 になるのか（解釈）

プロンプト末尾は `\n<|im_start|>assistant\n<think>\n\n</think>\n\n` で、ここから assistant の応答が始まる位置。Qwen 系の中国語 instruct データには「`当然！...`」のような肯定的開始フレーズが多く、layer 29-33 はその「応答開始の慣用句」を準備していると推測される。最終 2 層で「日本語で `言語モデル` 始まりにすべき」と修正される、という流れ。

---

## 6. 図

このスクリプト自体は PNG を出力しません（CSV/JSON のみ）。可視化は **[lecture/02_residual_stream_logit_lens_patching.ipynb](../lecture/02_residual_stream_logit_lens_patching.ipynb)** が同じデータで行います（[outputs/nb02_logit_lens_clean.png](../outputs/nb02_logit_lens_clean.png) など）。

ターミナル出力としては、各層の top1 / final top1 ランクが printed されます。

---

## 7. 応用への示唆

- **nb02 への直接寄与**: notebook 02 の logit lens セクション（[outputs/nb02_logit_lens_clean.png](../outputs/nb02_logit_lens_clean.png) など）はここでの手法をそのまま使い、`The capital of Japan is` プロンプトで「答え `Tokyo` が浮上する layer」を可視化する。本実験で確認した「中間層 RMSNorm 適用ルール」「`final_top1_rank` 指標」がそのまま使われる。
- **デモ映え**: 「層が深くなるにつれ予測が定まる」「最後の 2-3 層で急に top1 が決まる」を、ランク推移表 / heatmap で示せる。読む側は「Transformer の途中では何が決まっているか」を直感的に掴める。
- **[docs/12_residual_stream_patching.md](12_residual_stream_patching.md) の前提**: patching で「layer $k$ 付近で答えが決まる」と言う前に、まず logit lens で「naturally どの層で答えが top1 になるか」を確認しておく必要がある。
- **再利用したい数値**: 「答え `言` が top1 になるのは layer 34」「`当然` early lock-in が layer 29」という具体的数字は、notebook 02 の解説で使える。

---

## 8. 出力ファイル

- [outputs/prelim_logit_lens_summary.json](../outputs/prelim_logit_lens_summary.json) — モデル設定 + 選択 position + final top1 + sanity check diff
- [outputs/prelim_logit_lens_layer_metrics.csv](../outputs/prelim_logit_lens_layer_metrics.csv) — 37 行（layer 0..36）× 12 列。`top1_piece`, `top1_prob`, `entropy`, `final_top1_rank_in_this_layer` など
- [outputs/prelim_logit_lens_topk.csv](../outputs/prelim_logit_lens_topk.csv) — 37 layer × top-20 = 740 行。各行 `layer_index, rank, token_id, raw_token, piece, logit, prob`

---

## 9. 注意事項

- **RMSNorm を中間層で忘れない**: $k < K$ の `hidden_states[k]` を直接 `lm_head` に通すと、scale がずれて意味の通らない結果になる。CSV に `norm_applied` flag を残しているのはこのため。
- **fp16 数値誤差**: 選択 position のスライスだけ再 `lm_head` を呼ぶと `max_abs_diff = 0.0078125 = 1/128`（fp16 精度の floor）。full sequence の reuse なら 0.0。これが嫌なら全部 float32 で計算するのが [docs/11](11_compare_logit_lens_float32.md) のアプローチ。
- **`tie_word_embeddings=True` 前提**: Qwen3-4B では $W_U = W_E$ なので「層 0 で input identity が top1 になる」現象が起きる。tie していないモデルではこの現象は弱まる（embedding と unembedding が別空間なので、$W_U \cdot W_E e_x$ が対角に集中する保証がない）。
- **logit lens は本来 affine probing で改善できる**: **Belrose et al. (Tuned Lens)** は、中間層 hidden state に**学習された affine 変換（translator）**をかけてから unembedding に通すことを提案している（nostalgebraist の原典 logit lens は学習なしで unembedding を直接当てる baseline で、Tuned Lens はそれを改善する手法）。本実験は **untuned logit lens**（学習なし、直接 `lm_head` を当てる）で、簡便だが各層の "意味" を過大評価しやすい。Tuned Lens は [docs/10](10_compare_logit_lens_transformerlens.md) で言及あり（Qwen3 未対応のため動作はしていない）。
