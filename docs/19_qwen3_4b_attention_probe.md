# Experiment 19: Qwen3-4B Attention probe — attention weights, residual updates, and component patching

> [!WARNING]
> **未検証・暫定版（preliminary / unverified）**
> このレポートは script 19 の出力を一通りまとめた **仮の草稿**です。数値・図・解釈のいずれも十分なレビューや再現確認を経ていません。とくに head scoring（どの head が効くか）と component-level activation patching の結論部は追加検証が必要で、今後変わる可能性があります。参考資料として読む際は「確定した結果ではなく作業中のメモ」として扱ってください。

Script: [`scripts/19_prelim_attention_probe.py`](../scripts/19_prelim_attention_probe.py)
最終更新: 2026-05-21
ステータス: 🚧 暫定版（未検証）。clean/corrupt forward + attention weights + head scoring + residual update + component-level activation patching まで一通り出したが、レビュー・再現確認は未実施。

---

## 1. 目的

note02 では Qwen3-4B を「外側から」観察した。具体的には:

- **logit lens** で各 layer の residual stream を読み出した。
- **residual stream patching** で、ある layer の hidden state を clean → corrupt に差し替えると最終出力がどう変わるかを見た。

note02 で見えてきたのは、Qwen3-4B が "The capital of Japan is" の文脈で **" Tokyo" を選ぶ判断は中盤〜後半の layer (とくに L24 周辺) に集中している** ということだった。ただし note02 はあくまで **residual stream の単位**で見ており、その layer の中で **attention sub-block と MLP sub-block のどちらが** 効いているのかは区別していなかった。

03 notebook では Transformer block の**中**に入り、Attention と MLP が residual stream に何をしているかを見る予定。script 19 はその準備として、以下を一気にまとめて調べる:

1. attention weights (どのトークンを見ているか) を全 layer × 全 head で取り出す。
2. それを定量的にスコアリングし、注目すべき head を選ぶ。
3. attention update / MLP update が residual stream の "logit(Tokyo) - logit(Paris)" metric をどう動かすかを見る。
4. **attention output と MLP output を個別に activation patching** し、note02 の residual stream patching を component-level に分解する。

---

## 2. 背景: Transformer block と Attention

### 2-1. residual stream

Qwen3 を含む現代の decoder-only Transformer の各 block $j$ は、次のように residual stream $h \in \mathbb{R}^{T \times d_{\text{model}}}$ を 2 段で更新する:

$$
\begin{aligned}
h' &= h + \mathrm{Attn}_j(\mathrm{RMSNorm}_1(h)) \\
h_{\text{out}} &= h' + \mathrm{MLP}_j(\mathrm{RMSNorm}_2(h'))
\end{aligned}
$$

$T$ = トークン数、$d_{\text{model}}$ = hidden size。**residual stream** とは layer をまたいで一貫した「token ごとの内部表現の通り道」のことで、各 block はその通り道に新しい情報を**足し算**で書き込む。最後に final RMSNorm + `lm_head` で語彙ロジットになる。

```text
embed  →  [block 0]  →  [block 1]  →  ...  →  [block 35]  →  RMSNorm  →  lm_head  →  logits
                ↑               ↑
               attention      MLP
               update         update
              (residual         (residual
               に足す)           に足す)
```

### 2-2. self-attention の役割と Q / K / V

self-attention sub-block は **「他の token を参照して、自分の residual に何を書き加えるか」** を計算する操作。

入力 $X \in \mathbb{R}^{T \times d_{\text{model}}}$ から、線形変換で 3 種類の表現を作る:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V
$$

直感的な役割:

| 名前 | 略 | 役割 |
|---|---|---|
| **Query** | $Q$ | 「自分は何を探しているか」を表す問い合わせベクトル。query token (=注目している現在 token) ごとに作る。 |
| **Key** | $K$ | 「自分は何者か / 何を提供できるか」を表す目印ベクトル。**参照される候補となる** すべての token に作る。 |
| **Value** | $V$ | 「実際に運ばれる中身」。query が key にマッチしたとき、対応する value が residual に取り込まれる。 |

attention weights は $Q$ と $K$ の内積から計算する:

$$
A = \mathrm{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_{\text{head}}}} + M \right) \in \mathbb{R}^{T \times T}
$$

- $A[q, k]$ は **query 位置 $q$ が key 位置 $k$ をどの程度参照するか** を表す確率分布 (行ごとに和が 1)。
- $M$ は **causal mask**。未来 token を見せないように上三角を $-\infty$ にする。結果として decoder の attention は下三角行列 (今と過去だけを見る)。
- $\sqrt{d_{\text{head}}}$ は scaling factor。

そして attention output は:

$$
\mathrm{AttnOut} = (A V) W_O \in \mathbb{R}^{T \times d_{\text{model}}}
$$

$W_O$ は output projection。要するに **「他の token の value を attention weights で重みづけて足し、最後に projection」**。これが residual stream に加算される。

これを **複数 head** で並列に行う (multi-head attention)。各 head は独立した $W_Q, W_K, W_V$ を持ち、別々のサブ空間で「探す / マッチする / 運ぶ」を行う。head 同士は事後的にチャンネル次元で結合される (`o_proj` の中で線形に混ぜられる)。

### 2-3. Qwen3-4B の GQA (Grouped-Query Attention)

Qwen3-4B は **GQA** を採用している:

$$
\text{Q heads} = 32, \quad \text{K/V heads} = 8, \quad \text{num\_kv\_groups} = \frac{32}{8} = 4
$$

これは「Q だけ 32 種類、K と V は 8 種類しかない (4 つの Q head が同じ K/V を共有する)」という構造。inference 時の KV cache を小さくするために広く使われている。  
ただし attention weights $A$ は **Q head ごとに別々** に計算されるので、本実験では head=0..31 の 32 個を独立に扱う (どの 4 Q heads が K/V を共有しているかは別問題)。

### 2-4. なぜ "logit(Tokyo) - logit(Paris)" を見るのか

clean prompt `"The capital of Japan is"` と corrupt prompt `"The capital of France is"` を比較する。両 prompt で唯一違うのは pos=3 の **国名 token (" Japan" / " France")**。

最終 token (pos=4, `" is"`) の隠れ状態を logit lens で読むと:

- clean: " Tokyo" の logit が " Paris" より大きい → metric > 0
- corrupt: " Paris" の logit が " Tokyo" より大きい → metric < 0

定義:

$$
\text{metric} = \mathrm{logit}(\text{" Tokyo"}) - \mathrm{logit}(\text{" Paris"})
$$

これが clean vs corrupt の**差分**を一つの scalar に圧縮した量。"対立する 2 つの解釈" のうちどちらに振れているかを符号で表せるため、attention 解析・residual update 解析・activation patching の **共通の物差し** として使える。

### 2-5. residual stream update と attention の関係

attention は residual に**情報を引き寄せる操作**だが、attention weights 自体は「どこを見るか」しか教えてくれない。実際にどんな**ベクトル**が residual に書き込まれるかは $V$ と $W_O$ 次第。よって本実験では:

- attention weights $A$ そのもの (どのトークンを見るか) と、
- attention sub-block の出力 $\Delta_{\text{attn}} = \mathrm{AttnOut}$ (residual に書き込まれるベクトル) と、
- それが metric をどう動かしたか ($\Delta\text{metric}_{\text{attn}}$)

の 3 つを **別々に** 観察する。

### 2-6. activation patching との関係

note02 で使った **residual stream patching** は、ある layer の residual を corrupt run の中で clean のものに**差し替え**たうえで forward を続け、最終 metric の回復度 (recovery) を見るというもの。これは因果的介入 (intervention) なので、相関だけを返す logit lens より一段強い「その layer がどれだけ最終答えに効いているか」を教えてくれる。

script 19 では同じ枠組みを **component-level** に分解する:

- **attention output patching**: 各 layer の `self_attn` の出力 (= $\mathrm{AttnOut}$) を pos=4 で clean のものに差し替え。
- **MLP output patching**: 同様に `mlp` の出力を差し替え。

これによって、note02 で見えた「L24 周辺で recovery が立ち上がる」現象が attention 由来か MLP 由来かを切り分けられる。

---

## 3. 実験設定

| 項目 | 値 |
|---|---|
| 対象モデル | `Qwen/Qwen3-4B` |
| `attn_implementation` | `"eager"` (attention weights を取り出すため) |
| device / dtype | mps / float16 (本実行) — cuda なら bfloat16、cpu なら float32 |
| `use_cache` | `False` |
| chat template | **使わない** (素の prompt 5 token をそのまま入力) |
| clean prompt | `"The capital of Japan is"`  →  期待答え `" Tokyo"` |
| corrupt prompt | `"The capital of France is"`  →  期待答え `" Paris"` |
| 注目位置 | `pos=3` (国名 token), `pos=4` (`" is"`, query 位置) |
| metric | $\mathrm{logit}(\text{" Tokyo"}) - \mathrm{logit}(\text{" Paris"})$ at pos=4 |

両 prompt は 5 token、`clean_answer` / `corrupt_answer` はそれぞれ単一 token (id=26194, 12095) であることを実行時に確認している。

### Token table (pos と token の対応)

| pos | clean piece | corrupt piece | 用途 |
|---|---|---|---|
| 0 | `The` | `The` | 共通 |
| 1 | ` capital` | ` capital` | 共通 |
| 2 | ` of` | ` of` | 共通 |
| 3 | ` Japan` | ` France` | **country (=参照されたい key)** |
| 4 | ` is` | ` is` | **query 位置 (=次トークン予測点)** |

chat template を使わないのは、生 prompt を 5 token に抑えて attention matrix を視認できる小ささに保つため。

### baseline ロジット (実測)

| run | top-1 | metric |
|---|---|---|
| clean | `" Tokyo"` | **+11.6953** |
| corrupt | `" Paris"` | **-11.9844** |

metric range (clean - corrupt) = **+23.6797**。これが component patching の recovery の分母になる。

---

## 4. Model architecture (実測サマリ)

`outputs/prelim_attention_architecture_summary.json` より:

| 項目 | 値 |
|---|---|
| num_hidden_layers $K$ | 36 |
| hidden_size $d_{\text{model}}$ | 2560 |
| vocab_size | 151936 |
| intermediate_size (MLP 中間幅) | 9728 |
| num_attention_heads (Q heads) | **32** |
| num_key_value_heads (K/V heads) | **8** |
| num_key_value_groups | 4 |
| head_dim | 128 |
| rms_norm_eps | 1e-06 |
| tie_word_embeddings | True |

decoder layer class: `Qwen3DecoderLayer` / attention: `Qwen3Attention` / mlp: `Qwen3MLP`。  
`Qwen3DecoderLayer.forward` / `Qwen3Attention.forward` / `Qwen3MLP.forward` のソースは `outputs/prelim_attention_source_snippets.txt` に保存している (今回の hook 設計の根拠)。

---

## 5. Attention weights: self-attention matrix

### 5-1. shape と読み方

全層・全 head の attention weights を `output_attentions=True` で取り、tuple of $K$ tensor, 各 $[1, \text{heads}, T, T] = [1, 32, 5, 5]$ を得た。

- $A_{L,H}[q, k]$: layer $L$, head $H$ で、query 位置 $q$ が key 位置 $k$ を見る確率 (行ごとに和が 1)。
- 今回は 5x5 と小さいので、すべてを long CSV (`outputs/prelim_attention_self_attention_matrix_long.csv`, 57,600 行) に保存した。

### 5-2. 図の見方

**主図 (last-query row)**: query=pos4 (`" is"`) の行だけを抜き出し、layer × head の grid に 1×5 のミニ heatmap として並べた。1 セルが「ある (layer, head) について、pos=4 がどの key を見ているか」の確率分布。color scale は 0..1 共通。  
左から right に key=0..4 (= `The`, ` capital`, ` of`, ` Japan`/` France`, ` is`)。

![attention grid clean (last-query row)](../outputs/nb03_attention_grid_last_query_clean.png)

**Figure 1**: clean run の attention `A[q=4, k]` を layer×head grid で並べたもの。各セルの内部は 5 個の cell 横並び (key=0..4)。明るいほど大きい attention。多くの head で右端 (`" is"` 自身) を強く見ているが、いくつかの head は pos=3 (` Japan`) を強く見ているのが分かる。

![attention grid corrupt (last-query row)](../outputs/nb03_attention_grid_last_query_corrupt.png)

**Figure 2**: corrupt run の同じ grid。pos=3 が ` France` に変わっている以外は同じ prompt。clean と並べると、ほとんどの head は同じパターンだが、いくつかの head の attention が顕著に変わっているのが見える。

![attention grid clean - corrupt (last-query row)](../outputs/nb03_attention_grid_last_query_clean_minus_corrupt.png)

**Figure 3**: 上 2 図の差 (clean - corrupt)。発散 colormap (RdBu_r) で中心 0、青=corrupt 側が強い、赤=clean 側が強い。**clean/corrupt で attention pattern が変わる head** が可視化される。

**補助図 (full 5×5 self-attention matrix)**: query=0..4 すべての行 (= full 5×5 attention matrix) を grid に並べたもの。記録用。

![attention grid clean (full 5x5)](../outputs/nb03_attention_grid_full_matrix_clean.png)

**Figure 4**: clean の full 5×5 self-attention matrix grid。causal mask により右上の三角が 0 になっているのが確認できる。

![attention grid corrupt (full 5x5)](../outputs/nb03_attention_grid_full_matrix_corrupt.png)

**Figure 5**: corrupt 側 full 5×5。

![attention grid clean - corrupt (full 5x5)](../outputs/nb03_attention_grid_full_matrix_clean_minus_corrupt.png)

**Figure 6**: full matrix の差。

### 5-3. 代表的 head の単独 heatmap

§5-2 の grid は全 36×32 head を俯瞰するためのもので、個々の head の attention pattern を細かく読むには小さすぎる。後述の §6 のスコアで特徴のある 5 個の head を選び、**clean / corrupt / clean−corrupt の 3 panel** で大きな heatmap を描いた。各 cell には attention の値も annotate している。  
横軸 = key token、縦軸 = query token、行ごとに値の和が 1 (causal mask により右上は 0)。clean と corrupt は pos=3 だけ token が違う (` Japan` vs ` France`) ので、x/y のラベルもそれぞれの prompt のものを表示している。差分 panel の軸ラベルは便宜上 clean 側を表示している点に注意 (pos=3 の対応は "国名 token どうし" として読む)。

#### (a) L8 H29 — 安定 country pointer

![head L8 H29](../outputs/nb03_attention_head_L08_H29.png)

**Figure 5-3a**: `mean_attn_to_country` 1 位の head。query=4 (` is`) → key=3 (国名) の attention は **clean 0.896 / corrupt 0.870** と非常に高く、しかも clean/corrupt でほぼ同じ。pos=3 の行も `The` を強く見る傾向 (clean 0.63 / corrupt 0.87) があり、**「文中の名詞句を attention sink (`The`) と組み合わせて見る」構造的な head**。answer の中身に関わらず動作する。

#### (b) L24 H26 — country pointer + context shift

![head L24 H26](../outputs/nb03_attention_head_L24_H26.png)

**Figure 5-3b**: query=4 → key=3 の attention が **clean 0.912 / corrupt 0.532**。clean では国名を集中的に見て、corrupt では半分以下に下がる。`attn_output_L24` patching が単独で recovery 0.55 を出した layer の中にあり、country lookup を担う候補 head。pos=3 の対角 (` Japan` / ` France` token が自分自身を見る) の割合も clean 0.41 → corrupt 0.27 と異なる。

#### (c) L17 H17 — context-sensitive チャンピオン

![head L17 H17](../outputs/nb03_attention_head_L17_H17.png)

**Figure 5-3c**: row L1 で唯一 > 1.3 を出した head。query=4 → key=3 が **clean 0.146 / corrupt 0.542**、まさに「clean では国名をほとんど見ない、corrupt では強く見る」極端な切り替え。query=4 → key=0 (`The`) の attention sink も clean 0.733 / corrupt 0.103 と逆転している。差分 panel の右下に巨大な赤/青の対立が出る。

#### (d) L14 H11 — 国名 attention が大反転 (見ない方向 → 見る方向)

![head L14 H11](../outputs/nb03_attention_head_L14_H11.png)

**Figure 5-3d**: `country_difference_by_abs_diff` 1 位。query=4 → key=3 が **clean 0.185 / corrupt 0.654**。L17 H17 と同じく「corrupt のときに国名を強く見る」性質。前段の処理で clean run では別の中間表現に頼り、corrupt 側だけ raw token を読み直しているように見える。

#### (e) L12 H18 — 逆方向の context shift (clean で見る、corrupt で見ない)

![head L12 H18](../outputs/nb03_attention_head_L12_H18.png)

**Figure 5-3e**: 上 2 個と反対方向の context-sensitive head。query=4 → key=3 が **clean 0.561 / corrupt 0.099**。さらにこの head は **対角性が強い**: query=1→key=1=0.945、query=2→key=2=0.938 と前段で自己 token を強く見る。"前トークン / 現トークン" を読みつつ pos=4 で `" Japan"` だけに強く反応する選択的 head。

### 5-4. これらの 5 個から見える定性的な役割の差

| head | 性質 |
|---|---|
| (L8, H29) | **安定 pointer**: clean/corrupt の両方で国名を見る。文構造の認識を担う。 |
| (L24, H26) | **causal pointer**: 国名を見て、かつ最終 answer に因果的に効く (component patching でも単独で recovery 0.55)。 |
| (L17, H17) | **対称 switcher**: corrupt のとき初めて国名を見にいく方向の context-sensitive head。 |
| (L14, H11) | **(L17, H17) と同方向の switcher**、もう少し浅い layer。 |
| (L12, H18) | **逆方向 switcher**: clean で国名、corrupt で別の場所。`" Japan"` 専用 detector に近い。 |

attention weights を見るだけでは「どれが answer に効くか」は決まらない (§9 参照) が、**どの head が prompt の何に反応しているか** は分かる。L24 H26 のように "attention でも特徴的 + component patching でも因果的" の二重一致がある head が、講義デモで紹介するのに最適な例。

### 5-5. Row L1 top-10 head のまとめて観察 (前半 vs 後半 layer)

Figure 9 (row L1 clean − corrupt) で「context によって attention pattern が変わる」と特定された **上位 10 個** をすべて単独 heatmap で見る。  
これによって、§5-3 で限られた 3 個 ((L17,H17), (L14,H11), (L12,H18)) しか見ていなかった top-L1 head 群の全体像を確認できる。さらに前半 layer (L<20) と後半 layer (L≥20) で性質が違うかを比較するため、2×5 grid の overview 図も作った。

#### (a) Row L1 top-10 head の一覧

| rank | layer | head | half | clean attn→country | corrupt attn→country | row L1 | 性質メモ |
|---:|---:|---:|---|---:|---:|---:|---|
| 1 | 17 | 17 | FRONT | 0.146 | 0.542 | 1.301 | corrupt-dominant country pointer (Fig. 5-3c) |
| 2 | 17 | 25 | FRONT | 0.028 | 0.009 | 1.208 | country は見ない。pos=4 → key=0 (`The`) sink の比率が clean 0.80 / corrupt 0.30 と崩れて self-attention (` is` → ` is`) が 0.07 → 0.67 に跳ねる。**"sink-vs-self の切り替え" head**。 |
| 3 | 26 | 17 | BACK  | 0.008 | 0.080 | 0.983 | pos=4 row が clean では key=2 (` of`) に 0.61 / sink に 0.05 (異常に "of" を見る)、corrupt では key=0 sink に戻る (0.47)。**"of"-token-specific head** で、clean 文 (Japan の前置 of) の文脈下だけ "of" を強く見る。 |
| 4 | 14 | 11 | FRONT | 0.185 | 0.654 | 0.938 | corrupt-dominant pointer (Fig. 5-3d) |
| 5 | 12 | 18 | FRONT | 0.561 | 0.099 | 0.936 | clean-dominant pointer; 対角強 (Fig. 5-3e) |
| 6 | 26 | 20 | BACK  | 0.343 | 0.125 | 0.901 | clean-dominant: pos=4 → key=3 country が clean 0.34、corrupt は 0.13 で sink に流れる。 |
| 7 | 16 | 25 | FRONT | 0.510 | 0.136 | 0.884 | clean-dominant: pos=4 → key=3 country が clean 0.51、corrupt は 0.14 で sink に戻る。 |
| 8 | 16 | 27 | FRONT | 0.008 | 0.008 | 0.851 | 対角型 head (各 token が自分自身を強く見る)。pos=4 self-attn が **clean 0.535 / corrupt 0.960** と大きく変わる。country attention は両方 ≒ 0。 |
| 9 | 31 | 1  | BACK  | 0.100 | 0.358 | 0.847 | corrupt-dominant: pos=4 → key=3 country が clean 0.10 / corrupt 0.36。pos=4 sink (`The`) も clean 0.80 / corrupt 0.38。 |
| 10 | 34 | 20 | BACK  | 0.200 | 0.621 | 0.843 | corrupt-dominant、より深い layer での switcher。query=3 (` Japan`/` France` 自身) はほぼ `The` を見るだけ (両方 0.99) で、pos=4 だけで切り替わる。 |

ここでの **FRONT** は layer<20、**BACK** は layer≥20。「ちょうど中ほど」と「後半」の境界は曖昧なので、L24 を境にする代わりに L20 で切っている (top-10 では境界に乗る head はない)。

#### (b) 個別の単独 heatmap (top-10 のうち §5-3 で出ていない 7 個)

§5-3 で出した 3 個 ((L17,H17), (L14,H11), (L12,H18)) は再掲しない。残り 7 個:

![head L17 H25](../outputs/nb03_attention_head_L17_H25.png)

**Figure 5-5a**: (L17, H25) FRONT. country はほぼ見ない。pos=4 row の diff は key=0 (`The`, sink) と key=4 (` is`, 自己) の間で大きく入れ替わる。

![head L26 H17](../outputs/nb03_attention_head_L26_H17.png)

**Figure 5-5b**: (L26, H17) BACK. pos=4 row が "clean では key=2 (` of`) を 0.61 で強く見、corrupt では key=0 sink に戻る" 特殊 head。前置詞 (` of`) を文脈依存で読みに行く。

![head L26 H20](../outputs/nb03_attention_head_L26_H20.png)

**Figure 5-5c**: (L26, H20) BACK. clean-dominant pointer。BACK で珍しく "clean 側で国名を強く見る" head。

![head L16 H25](../outputs/nb03_attention_head_L16_H25.png)

**Figure 5-5d**: (L16, H25) FRONT. clean-dominant pointer。L12 H18 と似た方向。

![head L16 H27](../outputs/nb03_attention_head_L16_H27.png)

**Figure 5-5e**: (L16, H27) FRONT. 対角型: q=k=1, 2, 3 で値 ≒ 0.7-0.95 (各 token が自分自身を強く見る) が prompt-invariant。pos=4 self だけが **clean 0.535 → corrupt 0.960** に跳ねる。country を直接見るのではなく、" is" 自身の representation の使われ方が変わる head。

![head L31 H1](../outputs/nb03_attention_head_L31_H01.png)

**Figure 5-5f**: (L31, H1) BACK. corrupt-dominant pointer。L24 以降で最も早く現れる context switcher。

![head L34 H20](../outputs/nb03_attention_head_L34_H20.png)

**Figure 5-5g**: (L34, H20) BACK. 最も深い corrupt-dominant pointer。query=0..3 の挙動はほぼ "全部 sink を見る" 退化的な状態で、変化は pos=4 一点だけ。

#### (c) 10 個まとめて並べた overview (前半 vs 後半 比較用)

![top10 row_l1 clean](../outputs/nb03_attention_top10_row_l1_clean.png)

**Figure 5-5h**: top-10 row_l1 heads の **clean** attention matrix を 2×5 grid に並べたもの。タイトルに `[FRONT]` / `[BACK]` を付けている。

![top10 row_l1 corrupt](../outputs/nb03_attention_top10_row_l1_corrupt.png)

**Figure 5-5i**: 同 **corrupt** 版。clean と corrupt を並べると、pos=3 行の入れ替わりが視認しやすい。

![top10 row_l1 diff](../outputs/nb03_attention_top10_row_l1_diff.png)

**Figure 5-5j**: 同 **clean − corrupt** 差分。差分が出ている cell の位置が、各 head ごとに見える。

#### (d) FRONT (L<20) vs BACK (L≥20) の性質の違い

top-10 を FRONT / BACK で集計すると:

| metric | FRONT 6 個 (L12,14,16×2,17×2) | BACK 4 個 (L26×2, L31, L34) |
|---|---|---|
| 国名 token attention の主役か | **半数以上が主役** (L14 H11, L17 H17, L16 H25, L12 H18 で `attn_to_country` > 0.4 を片側で出す) | 全 4 個のうち 3 個が `attn_to_country > 0.3` を片側で出す。やや弱め |
| sink (`The`) や self (` is`) と country の "三角関係" | sink ↔ country の入れ替えが目立つ ((L17,H17), (L17,H25), (L16,H27)) | sink ↔ country の入れ替えがメイン (L26 H20, L26 H17, L31 H1, L34 H20) |
| **prompt の浅い構造に対する反応** | (L12 H18) のような対角強 head, (L16 H27) のような self-attn head が混ざっており、country lookup と並行して "前段の token relations" を担っている | 浅い構造 (前段 token 間の attention) はもうほとんど更新されず、pos=4 一点だけが大きく動く (Fig. 5-5g がその極端な例) |
| clean-dominant / corrupt-dominant | clean 3 (L12 H18, L16 H25, L17 H25), corrupt 3 (L14 H11, L17 H17, L16 H27) で半々 | clean 1 (L26 H20), corrupt 3 (L26 H17, L31 H1, L34 H20)。**深い layer ほど corrupt で国名を見直す方向の head が多い** |

定性的にまとめると:

- **FRONT (L12-L17) の context-sensitive head は "重い"**: country attention 自体が clean/corrupt で 0.4-0.7 動く head が多く、しかも head の中で **同時に対角や前段構造への attention も変化** する。要するに「文の中の関係を組み立てている最中」の head 群。
- **BACK (L26-L34) の context-sensitive head は "軽い"**: 大半の row は両 prompt で同じ (前段の token 関係はもう確定) で、**変化は pos=4 一点に集中**。とくに L34 H20 では pos=4 を除く全 row が両 prompt でほぼ identical。これは「pos=4 で答えを出す直前の局所的読み直し」の段階。
- **(L24, H26) は本 top-10 には入らない** (row L1 ランキングでは 10 位までに入らないため) が、country attn の絶対値で見れば clean 0.91 / corrupt 0.53 と top-10 内のどれよりも大きく、しかも component patching で因果的に効く唯一の head。row L1 で測ったときに上位に来ない理由は、**「両 prompt で同程度に高い → 差は大きいが片方は 0 になるほどではない」** ため。差の大きさ (row L1) と因果的影響は別物であることを示す具体例。

つまり Figure 9 で同じ "context-sensitive" にカテゴライズされていた head のうちでも、**FRONT は "前段の token 関係を組み立てる head"、BACK は "answer 直前の局所読み直し head"** と性質が分かれている。

### 5-6. 国名以外への attention pointer (key position 別)

§5 と §6 の前半は **"国名 (pos=3) を見る head"** に視点が偏っていた。しかし prompt の他の token (` capital`, ` of`, `The`, ` is` 自身) もそれぞれ重要な文構造の要素であり、それらを強く見る head も存在する。  
ここでは pos=4 (` is`) を query としたとき、各 key position 0..4 について attention を強く向ける head を全 layer × head から抜き出して見る。

#### (a) 位置別 top-3 head の一覧

`outputs/prelim_attention_head_attn_by_keypos_ranked.csv` (key×rank×layer×head の long format CSV) より:

| key_pos | piece | rank | layer | head | clean attn | corrupt attn | 性質 |
|---:|---|---:|---:|---:|---:|---:|---|
| 0 | `'The'` (sink) | 1 | 7 | 4 | 1.000 | 1.000 | **pure attention sink head**。pos=4 から見て BOS-like な `The` だけを 100% 見る。 |
| 0 | `'The'` (sink) | 2 | 7 | 6 | 1.000 | 1.000 | 同上。L7 には sink head が複数並ぶ。 |
| 0 | `'The'` (sink) | 3 | 13 | 12 | 1.000 | 1.000 | 浅い層から深い層まで sink head は散在する。 |
| 1 | `' capital'` | 1 | **6** | **15** | **0.982** | **0.950** | clean/corrupt の両方で ` capital` に 95% 以上を集中する **"capital pointer" head**。 |
| 1 | `' capital'` | 2 | 6 | 27 | 0.882 | 0.870 | 同じく L6。 |
| 1 | `' capital'` | 3 | 6 | 22 | 0.886 | 0.829 | 同じく L6。 |
| 2 | `' of'` | 1 | 1 | 26 | 0.950 | 0.940 | clean/corrupt の両方で ` of` を 94% 以上見る **"of pointer" head**。 |
| 2 | `' of'` | 2 | 2 | 7 | 0.870 | 0.891 | L2 にも ` of` 専属 head がある。 |
| 2 | `' of'` | 3 | 5 | 20 | 0.820 | 0.794 | 浅い層に分布。 |
| 3 | country | 1 | 8 | 29 | 0.896 | 0.870 | (§6 country ランキング 1 位、Fig. 5-3a)。 |
| 3 | country | 2 | 6 | 21 | 0.909 | 0.796 |  |
| 3 | country | 3 | 0 | 1 | 0.866 | 0.773 |  |
| 4 | `' is'` (self) | 1 | 0 | 2 | 0.999 | 0.999 | pos=4 から自分自身を 99% 見る self-loop head (L0 のため事実上 trivial)。 |
| 4 | `' is'` (self) | 2 | 0 | 26 | 0.989 | 0.991 | 同上。 |
| 4 | `' is'` (self) | 3 | 14 | 30 | 0.943 | 0.978 | 深い層の self head。L14 で `' is'` が自分自身を強く見直す。 |

特筆すべき発見:

- **L6 は "capital pointer 層"**: H15, H27, H22 ともに ` capital` を強く見る。さらに H29 (0.79), H11 (0.55), H14 (0.43) もある (CSV 参照)。**1 つの layer に同じ key position を見る head が集中する** という構造が初めて見えた。
- **L1 H26 / L2 H7 は "of pointer"**: 「前置詞 `of` を読みに行く head」が浅い層に専属で存在する。
- **L7 や L13, L20, L29 の "pure sink head"** は attention 出力がほぼ `The` の value だけになる。これらは "情報を捨てる" head として知られる attention sink の典型例。
- **§5-5 で見た (L26, H17)** は実は "of pointer" の **clean-only 版**: clean では ` of` を 0.61 で見るのに corrupt では 0.27 に下がる。L1 H26 のような "両 prompt で安定" な of pointer とは違って、文脈で発火する of-reader。

#### (b) 位置別 scalar heatmap (layer × head)

各 key position について、`mean attn from pos=4 to key=N` を layer × head に並べたもの。各 head が "どの位置を見やすいか" を一枚ずつ確認できる。

![score mean attn to pos 0 (The/sink)](../outputs/nb03_attention_score_mean_attn_to_pos0.png)

**Figure 5-6a**: → `'The'` (pos=0, sink) への mean attention。多くの head が中程度 (0.3-0.6) の値を持ち、特定の head ((L7, H4) (L7, H6) (L13, H12) (L29, H20) など) が 1.0 に近い "pure sink" として浮かび上がる。**全体的に sink 寄り** は近年の LLM の典型 (attention sink 現象)。

![score mean attn to pos 1 (capital)](../outputs/nb03_attention_score_mean_attn_to_pos1.png)

**Figure 5-6b**: → `' capital'` (pos=1) への mean attention。**L6 の column が縞状に明るくなる** のが見える。L6 H15, H22, H27, H29 がはっきり浮かび上がる。他の layer には強い ` capital` pointer はほぼ存在しない。**" capital" lookup は L6 で集中的に行われている**。

![score mean attn to pos 2 (of)](../outputs/nb03_attention_score_mean_attn_to_pos2.png)

**Figure 5-6c**: → `' of'` (pos=2)。浅い層 (L1-L5) に強い head が散らばる。深い層では (L26, H17) がやや浮かぶ程度。

![score mean attn to pos 3 (country)](../outputs/nb03_attention_score_mean_attn_to_pos3.png)

**Figure 5-6d**: → country (pos=3)。§6 Figure 7 と同じ図 (mean_attn_to_country) と読み替えて参照。

![score mean attn to pos 4 ( is, self)](../outputs/nb03_attention_score_mean_attn_to_pos4.png)

**Figure 5-6e**: → `' is'` (pos=4, self)。L0 と L14, L24 周辺に集中する。pos=4 self-attention は "自分の情報を捨てない" head と読める。L0 の self 集中は単に「embed_tokens 直後では文脈情報がまだないので自己 token しか見るものがない」事情も大きい。

#### (c) 各位置の代表 head の単独 heatmap

![head L6 H15 (capital pointer)](../outputs/nb03_attention_head_L06_H15.png)

**Figure 5-6f**: **(L6, H15) — capital pointer のチャンピオン**。pos=4 → key=1 (` capital`) が clean 0.98 / corrupt 0.95。clean/corrupt でほぼ同じ (差は 0.03 のみ) なので、これは「文の `capital of X` 構造を読む head」 で、X の中身に左右されない。query=2 行 (` of`) も 0.71 で ` capital` を見ており、内部的に "capital ↔ of" pair を作っている可能性がある。

![head L6 H27 (capital pointer alt)](../outputs/nb03_attention_head_L06_H27.png)

**Figure 5-6g**: **(L6, H27) — capital pointer の 2 番手**。pos=4 → key=1 が clean 0.88 / corrupt 0.87。L6 H15 と同じ性質。L6 にはこのような capital pointer head が複数並ぶ。

![head L1 H26 (of pointer)](../outputs/nb03_attention_head_L01_H26.png)

**Figure 5-6h**: **(L1, H26) — ` of` pointer**。pos=4 → key=2 (` of`) が clean 0.95 / corrupt 0.94。これも prompt-invariant (差 0.01)。前置詞そのものを読みに行く head。

![head L13 H12 (pure sink)](../outputs/nb03_attention_head_L13_H12.png)

**Figure 5-6i**: **(L13, H12) — pure sink head**。すべての query position が `The` (pos=0) を 100% 見る (5×5 matrix の 1 列目だけが 1.0)。これは attention の典型的な "attention sink"。 sub-block の output は何にせよ "`The` の value を `o_proj` に通したベクトル + residual に注入" になる。

![head L14 H30 (self head)](../outputs/nb03_attention_head_L14_H30.png)

**Figure 5-6j**: **(L14, H30) — self head**。pos=4 → key=4 (` is`) が clean 0.94 / corrupt 0.98。query=3 と query=4 が自分自身を強く見る。`' is'` の representation を **その layer で局所的に refine する** 役割と解釈できる (residual 上の "self update")。

#### (d) 5 位置をまとめて並べた overview

![per keypos top3 overview](../outputs/nb03_attention_per_keypos_top3_overview.png)

**Figure 5-6k**: 5 つの key position × rank 1-3 の合計 15 head の clean attention を 1 枚に並べたもの。**列ごとに同じ "見ている位置"** が並ぶので、(a) の表よりも視覚的に位置別の性質差が掴みやすい。

#### (e) 統合的な観察

1. **prompt の各 token に "専属 pointer head" がある**: `The` (sink), ` capital`, ` of`, country, ` is` のすべてに、それを 80% 以上見る head が少なくとも 1 つは存在する。**Transformer は token ごとに "誰がそれを読むか" を分業している**。  
   - sink (`The`): 多数の head が pure sink。"情報を捨てる" バルブ。  
   - ` capital`: **L6 に集中** (>= 4 head)。  
   - ` of`: 浅い層 (L1-L5) に分散。  
   - country (pos=3): 浅い-深い両方に分散。L24 H26 が因果的にも効く唯一の head (§8)。  
   - ` is` (self): 浅い L0 に多数 + 深い L14, L24 にも refine 用 head。
2. **layer ごとの "役割" がぼんやり見える**: L6 = capital lookup, L1-L2 = `of` lookup, L7+ = sink バルブ, L24 周辺 = country lookup (causal)。これは綺麗に分業しているわけではなく、**同じ layer 内に多種類の head が並存** しているが、それぞれの **layer × head 平面で見ると "強い pointer" がクラスタを作っている**。
3. **"安定 pointer" と "context-sensitive pointer" の区別が key position によらず効く**: L6 H15 のように **clean/corrupt で差 < 0.05** の head は「文構造の認識」担当、§5-5 の top-L1 head のように **差 > 0.4** の head は「内容に応じた読み直し」担当、と整理できる。

---

## 6. Head scoring

### 6-1. 定義

query=pos4 の row $a := A_{L,H}[4, :]$ について、以下のスカラを定義する。

| 列 | 定義 | 直感 |
|---|---|---|
| `attn_to_country` | $a[3]$ | pos=3 国名 token への直接の重み |
| `country_rank` | row 内で $a[3]$ より大きい key の数 + 1 (1 = 最大) | 国名 token が "何位" に見られているか |
| `country_margin` | $a[3] - \max_{k \neq 3} a[k]$ | 国名 token が次点より何ポイント勝っているか |
| `self_attn_weight` | $a[4]$ | 自分自身 (` is`) を見ている割合 |
| `first_token_weight` | $a[0]$ | BOS-like な `The` を見ている割合 (attention sink ぽい挙動) |
| `row_entropy` | $-\sum_k a_k \log a_k$ | row の分布の散らばり (見えている key=4 個分のみ実質寄与) |
| `row_entropy_norm` | row_entropy / $\log(\text{visible})$ | causal mask で見える key の数で正規化 (0..1) |
| `focus_score` | `attn_to_country` × (1 − `row_entropy_norm`) | 「国名を見て、かつ集中している」の合成指標 |

さらに clean / corrupt を比較する指標:

| 列 | 定義 |
|---|---|
| `mean_attn_to_country` | 0.5 × (clean + corrupt) の attn_to_country |
| `abs_diff_attn_to_country` | |clean − corrupt| |
| `mean_focus_score` | 0.5 × (clean + corrupt) の focus |
| `row_l1_clean_corrupt` | $\sum_k |a^{\text{clean}}_k - a^{\text{corrupt}}_k|$ |
| `row_js_clean_corrupt` | clean / corrupt 分布間の Jensen-Shannon divergence (自然対数) |

これらは `outputs/prelim_attention_head_scores.csv` (2,304 行 = 2 × 36 × 32) に全て、`outputs/prelim_attention_head_scores_ranked.csv` に複数ランキング上位 50 件を保存した。

### 6-2. ランキング (top-10)

#### (a) `country_pointer_by_mean_attn` — 国名 token を強く見る head

| rank | layer | head | clean attn→country | corrupt attn→country | clean rank | corrupt rank |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 29 | 0.896 | 0.870 | 1 | 1 |
| 2 | 6 | 21 | 0.909 | 0.796 | 1 | 1 |
| 3 | 0 | 1 | 0.866 | 0.773 | 1 | 1 |
| 4 | 5 | 18 | 0.863 | 0.775 | 1 | 1 |
| 5 | 3 | 3 | 0.823 | 0.802 | 1 | 1 |
| 6 | 1 | 4 | 0.778 | 0.836 | 1 | 1 |
| 7 | 2 | 19 | 0.785 | 0.737 | 1 | 1 |
| 8 | 12 | 0 | 0.704 | 0.807 | 1 | 1 |
| 9 | 14 | 10 | 0.627 | 0.857 | 1 | 1 |
| 10 | 24 | 26 | 0.912 | 0.532 | 1 | 1 |

#### (b) `country_pointer_by_focus` — 国名を見て、かつ集中している head

(a) とほぼ同じ顔ぶれだが順序が微妙に違う。特に **(L24, H26)** が 7 位に上がる (focus = 0.456)。

| rank | layer | head | mean focus |
|---:|---:|---:|---:|
| 1 | 8 | 29 | 0.622 |
| 2 | 6 | 21 | 0.552 |
| 3 | 0 | 1 | 0.541 |
| 4 | 1 | 4 | 0.513 |
| 5 | 5 | 18 | 0.490 |
| 6 | 3 | 3 | 0.462 |
| 7 | 24 | 26 | 0.456 |
| 8 | 14 | 10 | 0.433 |
| 9 | 12 | 0 | 0.392 |
| 10 | 2 | 19 | 0.374 |

#### (c) `context_sensitive_by_l1` — clean/corrupt で attention pattern が変わる head (row L1)

| rank | layer | head | row L1 | row JS | clean attn→country | corrupt attn→country |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 17 | 1.301 | 0.238 | 0.146 | 0.542 |
| 2 | 17 | 25 | 1.208 | 0.220 | 0.028 | 0.009 |
| 3 | 26 | 17 | 0.983 | 0.171 | 0.008 | 0.080 |
| 4 | 14 | 11 | 0.938 | 0.124 | 0.185 | 0.654 |
| 5 | 12 | 18 | 0.936 | 0.136 | 0.561 | 0.099 |
| 6 | 26 | 20 | 0.901 | 0.113 | 0.343 | 0.125 |
| 7 | 16 | 25 | 0.884 | 0.112 | 0.510 | 0.136 |
| 8 | 16 | 27 | 0.851 | 0.142 | 0.008 | 0.008 |
| 9 | 31 | 1 | 0.847 | 0.098 | 0.100 | 0.358 |
| 10 | 34 | 20 | 0.843 | 0.095 | 0.200 | 0.621 |

#### (d) `country_difference_by_abs_diff` — 国名 token への attention 自体が clean/corrupt で大きく変わる head

| rank | layer | head | |Δ attn→country| | clean | corrupt |
|---:|---:|---:|---:|---:|---:|
| 1 | 14 | 11 | 0.469 | 0.185 | 0.654 |
| 2 | 12 | 18 | 0.461 | 0.561 | 0.099 |
| 3 | 34 | 20 | 0.421 | 0.200 | 0.621 |
| 4 | 14 | 23 | 0.420 | 0.313 | 0.733 |
| 5 | 11 | 0 | 0.412 | 0.126 | 0.538 |
| 6 | 13 | 17 | 0.399 | 0.451 | 0.052 |
| 7 | 17 | 17 | 0.397 | 0.146 | 0.542 |
| 8 | 0 | 9 | 0.393 | 0.484 | 0.091 |
| 9 | 14 | 22 | 0.384 | 0.066 | 0.449 |
| 10 | 24 | 26 | 0.380 | 0.912 | 0.532 |

### 6-3. scalar score heatmap

layer × head 全体の俯瞰図。

![mean attn to country](../outputs/nb03_attention_score_mean_attn_to_country.png)

**Figure 7**: `mean_attn_to_country` (layer × head, viridis)。前半 layer に「国名 token を強く見る head」が広く分布している (Figure 1, 2 と一致)。

![mean focus](../outputs/nb03_attention_score_mean_focus.png)

**Figure 8**: `mean_focus_score`。entropy が低くて国名を見ている head が浮かび上がる。

![row L1 clean - corrupt](../outputs/nb03_attention_score_row_l1_clean_corrupt.png)

**Figure 9**: `row_l1_clean_corrupt`。L17 H17 が突出。L12-L17 と L26 付近の少数 head に「context によって attention pattern が変わる」性質が偏っている。

![row JS clean - corrupt](../outputs/nb03_attention_score_row_js_clean_corrupt.png)

**Figure 10**: `row_js_clean_corrupt`。L1 とよく似た傾向。

### 6-4. 解釈

1. **「国名を見る head」と「文脈で変わる head」は別物**。  
   country_pointer_by_focus の top の多くは前半 layer (L0..L8) に集中し、しかも clean と corrupt の両方で attn≈0.8-0.9 と高い。つまりこれらの head は **prompt の構造として「直前の名詞句を見る」性質**を持っており、内容によらず安定。  
   一方、context_sensitive_by_l1 の top はもっと深い layer (L12-L17, L26 など) で、clean と corrupt で attention pattern 自体が変わる head。**「中身を見て振る舞いを変える」head** であり、country lookup → answer の橋渡しに関わっている可能性が高い。

2. **(L24, H26) は両方の性質を持つ希少な head**。  
   - `attn_to_country` (clean) = 0.912 で focus も top-10 入り。
   - 同時に `abs_diff_attn_to_country` = 0.380 で clean/corrupt で attention が大きく変わる。  
   後述の component patching で L24 attention が単独で recovery 0.55 を出す結果と整合的。

3. **L17 H17 は context-sensitive のチャンピオン**。  
   row L1 で唯一 > 1.3 を出す。clean では country token をあまり見ない (0.15) のに corrupt では強く見ている (0.54)。意味的に「対称な切り替え」を担う head の候補。

---

## 7. Residual stream update metric

### 7-1. ねらいと定義

attention weights は "**どこを見るか**" を教えるが、"**residual に何が書き込まれたか**" は output projection を通った後にしか分からない。さらに最終 metric ("logit(Tokyo) - logit(Paris)") に効くかどうかは、書き込まれたベクトルの**向き** (Tokyo - Paris 方向との内積) で決まる。

そこで script 19 では、各 layer $L$ について **pos=4** の residual stream の状態を、attention sub-block と MLP sub-block の前後で **3 点サンプリング**する:

```text
h_before_attn  --(+ attn_update)-->  h_after_attn  --(+ mlp_update)-->  h_after_mlp
```

各点で **logit lens** (final RMSNorm + lm_head) を通して metric を読み、増分 $\Delta$ を見る:

$$
\begin{aligned}
\Delta\text{metric}_{\text{attn}}(L) &= \text{metric}(h_{\text{after attn}}) - \text{metric}(h_{\text{before attn}}) \\
\Delta\text{metric}_{\text{mlp}}(L)  &= \text{metric}(h_{\text{after mlp}})  - \text{metric}(h_{\text{after attn}})
\end{aligned}
$$

ここで使うのは **final RMSNorm を介した logit lens**。各 layer の residual を**「もし最後の RMSNorm + unembedding に直結したらどう読めるか」**として読み出す通常の logit lens 規約。  
(注: hidden_states[k] (k<K) は post-norm ではなく **layer 出力の生 residual** なので、ここで final RMSNorm を 1 回だけかけても二重 norm にはならない。実行時に sanity check で確認している。`outputs/prelim_attention_residual_update_sanity.csv`。)

更新ベクトルの **L2 norm** とその residual 比 (`*_update_relative_norm`) も同時に記録する (sub-block の「大きさ」と「向き」を切り分けるため)。

### 7-2. 図

#### (a) metric が layer をどう動くか

![metric vs layer clean](../outputs/nb03_attention_residual_metric_clean.png)

**Figure 11**: clean run。横軸 = layer index、縦軸 = `logit(" Tokyo") - logit(" Paris")` を pos=4 で読んだ値。3 系列: `before attn` / `after attn` / `after mlp`。**L24 で attn が大きく持ち上げ (+7.7)** て、それまで $\pm 2$ 程度を漂っていた metric が +10 付近に乗る。その後、深い layer (L29, L31) でさらに上方修正される。

![metric vs layer corrupt](../outputs/nb03_attention_residual_metric_corrupt.png)

**Figure 12**: corrupt run。L24 attn が **逆向きに −5.2** 押し下げて metric を更に negative にする (Paris を強める)。同じ "L24 attn" が clean と corrupt で逆符号に動く点が note02 の "L24 が answer 決定の中心" と整合的。

#### (b) Δ metric: attn vs mlp

![delta metric attn vs mlp clean](../outputs/nb03_attention_delta_metric_attn_vs_mlp_clean.png)

**Figure 13**: clean run の Δ metric を attn (青) と mlp (緑) で 2 系列棒グラフ。**L0 attn (+10.9)**, **L24 attn (+7.7)**, **L29 mlp (+7.9)**, **L31 attn (+9.9)** が突出。L0 の +10.9 は "embedding を直後の attention で混ぜた瞬間に、後方 4 token への参照が混ざって token 統計的に Tokyo/Paris が動く" 程度の意味と思われる (baseline 動作)。**L24 attn と L29 mlp は note02 residual stream patching のピークと一致** する重要な点。

![delta metric attn vs mlp corrupt](../outputs/nb03_attention_delta_metric_attn_vs_mlp_corrupt.png)

**Figure 14**: corrupt 側。L24 attn = −5.2、L29 mlp = −6.0、L31 attn = −6.7。clean と対称的に逆向きに押している。

#### (c) update norm (大きさ自体)

![update norms clean](../outputs/nb03_attention_update_norms_clean.png)

**Figure 15**: clean。`||attn update||` と `||mlp update||` を residual norm と一緒にプロット。  
**重要**: residual norm は layer が深くなるにつれて単調に増加 (L0: 0.85 → L35: 638)。これは Transformer の典型的なふるまい (各 layer が足し算で書き込むので)。**update の絶対 norm は深い layer ほど大きい**が、それは residual が大きくなっているからで、metric への寄与は別の話。L24 attn の norm 24.4 は決して最大ではない (L34 attn = 80, L35 mlp = 240 はもっと大きい)。つまり **「effective な layer = norm が大きい layer」ではない**。

![update norms corrupt](../outputs/nb03_attention_update_norms_corrupt.png)

**Figure 16**: corrupt 側 (clean と大差なし)。

### 7-3. 解釈

- **L0 attn の +10.9 (clean) は "見かけ上の大きな寄与"**。pos=4 (`" is"`) はまだ context をほとんど持っていない token (1 layer 後の residual norm が 7.7) なので、logit lens で読んだとき token 統計に強く引きずられている可能性が高い。**初期 layer の logit lens 値はノイジー** であることに留意。
- **L24 attn (+7.7 / −5.2) と L29 mlp (+7.9 / −6.0) が真の "answer 決定の場"**。residual norm が安定してから大きく metric を動かしている。これは後述の component patching で attn_output_L24 / mlp_output_L29 が高 recovery を出すことと一致。
- attn と mlp は必ずしも同じ layer で同期しない。**L24 = attn が立つ layer / L29 = mlp が立つ layer**。residual stream で見ると同じ residual を共有しているが、書き込みのタイミングは別。

---

## 8. Component-level activation patching

### 8-1. 定義

note02 の **residual stream patching** は、ある layer の "residual stream そのもの" (= layer 出力) を pos=4 で `clean → corrupt run の中` に注入し、最終 metric の recovery を測るものだった:

$$
\text{recovery} = \frac{\text{metric}_{\text{patched}} - \text{metric}_{\text{corrupt}}}{\text{metric}_{\text{clean}} - \text{metric}_{\text{corrupt}}}
$$

`= 0` なら全く回復していない (corrupt のまま)、`= 1` なら完全に clean と同じ metric まで戻る。

script 19 ではこれを **2 種類の component に分解** する:

- **`attn_output` patching**: `model.model.layers[L].self_attn` の forward 出力 (attention sub-block の更新ベクトル $\mathrm{AttnOut}$) を pos=4 で clean に差し替える。
- **`mlp_output` patching**: `model.model.layers[L].mlp` の出力 (MLP 更新ベクトル) を pos=4 で clean に差し替える。

それ以外の component (前後の attention, mlp, ほかの位置) は corrupt run のまま。これにより **「この component 単体を clean にしたら answer は救えるか?」** を 36 layer × 2 component で測る (= 72 試行)。

技術メモ: `self_attn.forward` の出力は `(attn_output, attn_weights)` の tuple、`mlp.forward` の出力は tensor。hook では両方に対応している。tuple/tensor の差は `outputs/prelim_attention_source_snippets.txt` の Qwen3 ソースで確認できる。

### 8-2. 図

![component patching recovery](../outputs/nb03_attention_component_patching_recovery.png)

**Figure 17**: component patching の recovery 曲線。横軸 = layer index、縦軸 = recovery。実線 0 (corrupt) と破線 1 (clean) が基準。  
**attn_output (青)**: **L24 で recovery 0.55** が単独で立つ。次点が L31 (0.42)。それ以外は recovery < 0.12。  
**mlp_output (オレンジ)**: **L29 で recovery 0.55** が単独で立つ。次点が L31 (0.29)。それ以外は recovery < 0.10。

![component patching metric](../outputs/nb03_attention_component_patching_metric.png)

**Figure 18**: patched metric の生値。clean (上の破線, +11.7) / corrupt (下の破線, −11.98) を参考線にした絶対値プロット。L24 attn patching と L29 mlp patching で metric が中央 (≒ 0) まで戻り、それ以外の層では corrupt 付近に張り付いていることが視認できる。

![component patching top1](../outputs/nb03_attention_component_patching_top1.png)

**Figure 19**: 36 layer × 2 component (attn_output 上段 / mlp_output 下段) の patched top-1 token を **categorical color** で表示。  
**色**: 緑 = top-1 が `" Tokyo"` (= clean answer に flip 成功)、赤 = `" Paris"` (= corrupt のまま)、灰 = それ以外 (中間状態の token)。  
各セルには patched top-1 piece (太字、上段) と recovery 値 `r=...` (小さく、下段) を併記。  
**緑のセルは 2 つだけ — attn_output@L24 と mlp_output@L29** で、ここでだけ answer が clean に flip する。L31 attn / L31 mlp は recovery が中程度 (r ≃ +0.42 / +0.29) だが top-1 が flip しきれず、L31 mlp は `" in"` (灰) に着地している。「recovery > 0 = 部分的な押し戻し」と「top-1 が flip = 完全に answer を奪取」が別の現象であることが色で読み取れる。

### 8-3. 数値ハイライト

`outputs/prelim_attention_component_patching_by_layer.csv` から:

| component | layer | patched metric | recovery | patched top-1 | patched P(Tokyo) | patched P(Paris) |
|---|---:|---:|---:|---|---:|---:|
| attn_output | **24** | **+1.05** | **0.550** | **` Tokyo`** | 0.487 | 0.171 |
| attn_output | 31 | −2.00 | 0.422 | ` Paris` | 0.060 | 0.444 |
| attn_output | 35 | −9.34 | 0.112 | ` Paris` | < 1e−4 | 0.621 |
| attn_output | 34 | −9.98 | 0.085 | ` Paris` | < 1e−4 | 0.465 |
| mlp_output | **29** | **+0.95** | **0.546** | **` Tokyo`** | 0.446 | 0.172 |
| mlp_output | 31 | −5.11 | 0.290 | ` in` | 0.001 | 0.238 |
| mlp_output | 26 | −9.94 | 0.086 | ` in` | < 1e−4 | 0.207 |
| mlp_output | 22 | −10.45 | 0.065 | ` Paris` | < 1e−4 | 0.607 |

### 8-4. 解釈

1. **L24 attention sub-block が answer lookup の中心**。 単独で patch するだけで top-1 が ` Tokyo` に切り替わり、metric が +1.0 付近まで戻る (≒ neutral)。
2. **L29 MLP がもう一つの主要 component**。recovery と top-1 入れ替わりが L24 attention とほぼ同等で、note02 の residual stream patching が L24-L29 区間でピークを描く理由を component-level に分解できた。
3. **L31 では attn / mlp の両方で部分 recovery が見られる**が、top-1 は `Paris` か `in` のまま。L31 は L24/L29 で確定した answer を補強する役割 (cleanup) と解釈できる。
4. 一方で **attention weights ベースで上位だった head (例: L8 H29) は component patching では効かない**。これは "attention で見ている = 因果的に効く" ではないという attention 解釈の典型的な落とし穴。本実験では **attn_output_L24 を head 分解していない** (全 32 heads まとめて patch している) ので、L24 の中のどの head が決定的かは別実験 (script 20 候補) で詰める。

---

## 9. 解釈 (総合)

- **logit lens (note02) と residual update metric (script 19) はどちらも「読み出し」のみ** で、因果関係を示さない。一方で **residual stream patching (note02) と component patching (script 19) は介入** であり、因果的な情報を返す。両者を組み合わせると、note02 の "L24 周辺で recovery が立つ" を本実験は **"L24 = attention の貢献、L29 = MLP の貢献" にさらに分解** できた。
- **attention weights は注意の位置情報、AttnOut は実際の更新ベクトル、Δmetric は metric への射影**。3 つは別物で、必ずしも一致しない。L8 H29 のように「常に国名を強く見るが、最終答えに効かない head」と、L24 H26 のように「国名を見ていて、かつ最終答えにも因果的に効く layer の一部」がある。
- **head 単位の "因果性" は本実験では分かっていない**。今回の component patching は **全 head まとめ** ; head-level patching は script 20 候補。

---

## 10. 今後

- **script 20** (候補): selected head-level activation patching。L24 attention の中の各 head を個別に patch して、`H26` が単独で recovery を担うかを確認する。
- **03 notebook**: Attention と FFN/MLP の講義補助資料。本 script の図を直接持ち込み、学生に "head の attention pattern" → "component patching" → "final answer" の流れを見せる。
- **transcoder 実験 (script 14, 15) との接続**: 本実験で特定された L24 attention / L29 MLP のうち、L29 MLP は mwhanna transcoder で sparse feature 単位に分解できる。L29 transcoder の feature を `attn_output_L24` で patch した状況と組み合わせて見たい (= attention で残された情報を MLP がどの feature で読み出すか)。
- **`qwen3_4b_trace` workspace**: `Qwen3Attention.forward` を instrument し、head-level の AttnOut (∈ ℝ^{T×head_dim}, o_proj 前) を per-head tensor として取り出す。本 workspace では editable install を使わない方針なので、ここから先は trace workspace へ移す。

---

## 11. 出力ファイル

`outputs/` 直下に以下を出力。

### CSV

| ファイル | 行数 | 内容 |
|---|---:|---|
| `prelim_attention_prompt_tokens.csv` | 10 | clean/corrupt 各 5 token の (pos, token_id, piece) |
| `prelim_attention_baseline_topk.csv` | 20 | clean/corrupt baseline の top-10 next token |
| `prelim_attention_self_attention_matrix_long.csv` | 57,600 | 2 × K × heads × T × T の long-format attention weights |
| `prelim_attention_head_scores.csv` | 2,304 | 2 × 36 × 32 head ごとのスカラスコア (attn_to_country, focus, entropy, …) |
| `prelim_attention_head_scores_ranked.csv` | 250 | 5 つの rank_type × top-50 |
| `prelim_attention_residual_update_metrics.csv` | 72 | 2 × 36 layer の (Δmetric_attn, Δmetric_mlp, update norms) |
| `prelim_attention_residual_update_sanity.csv` | 72 | h_after_mlp[L] と hidden_states[L+1] の数値ずれ (fp16 sanity) |
| `prelim_attention_component_patching_by_layer.csv` | 72 | 36 layer × 2 component の (patched_metric, recovery, top-1) |
| `prelim_attention_component_patching_topk.csv` | 720 | 同 patch 試行ごとの top-10 token |
| `prelim_attention_head_attn_by_keypos.csv` | 11,520 | 2 × 36 × 32 × 5 の long: pos=4 row の各 key 位置 attention |
| `prelim_attention_head_attn_by_keypos_ranked.csv` | 75 | 5 key 位置 × top-15 head (mean attention) ranking |

### JSON

| ファイル | 内容 |
|---|---|
| `prelim_attention_architecture_summary.json` | model config の要約 (K, hidden, heads, GQA, head_dim, …) |
| `prelim_attention_baseline_summary.json` | clean/corrupt metric と top-1 piece |
| `prelim_attention_component_patching_summary.json` | patching 実験の定義と clean/corrupt metric |

### Text

| ファイル | 内容 |
|---|---|
| `prelim_attention_source_snippets.txt` | `Qwen3DecoderLayer.forward` / `Qwen3Attention.forward` / `Qwen3MLP.forward` のソース |

### PNG

| ファイル | 説明 |
|---|---|
| `nb03_attention_grid_last_query_clean.png` | last-query row 版 attention grid (clean) |
| `nb03_attention_grid_last_query_corrupt.png` | 同 (corrupt) |
| `nb03_attention_grid_last_query_clean_minus_corrupt.png` | 同 (差分) |
| `nb03_attention_grid_full_matrix_clean.png` | full 5×5 matrix 版 (clean) |
| `nb03_attention_grid_full_matrix_corrupt.png` | 同 (corrupt) |
| `nb03_attention_grid_full_matrix_clean_minus_corrupt.png` | 同 (差分) |
| `nb03_attention_score_mean_attn_to_country.png` | scalar head score: mean attn→country |
| `nb03_attention_score_mean_focus.png` | scalar head score: mean focus |
| `nb03_attention_score_row_l1_clean_corrupt.png` | scalar head score: row L1 |
| `nb03_attention_score_row_js_clean_corrupt.png` | scalar head score: row JS |
| `nb03_attention_residual_metric_clean.png` | metric (before/after attn/mlp) vs layer (clean) |
| `nb03_attention_residual_metric_corrupt.png` | 同 (corrupt) |
| `nb03_attention_delta_metric_attn_vs_mlp_clean.png` | Δmetric_attn と Δmetric_mlp 棒グラフ (clean) |
| `nb03_attention_delta_metric_attn_vs_mlp_corrupt.png` | 同 (corrupt) |
| `nb03_attention_update_norms_clean.png` | ||attn update|| / ||mlp update|| / ||residual|| (clean) |
| `nb03_attention_update_norms_corrupt.png` | 同 (corrupt) |
| `nb03_attention_component_patching_recovery.png` | component patching の recovery vs layer |
| `nb03_attention_component_patching_metric.png` | component patching の patched metric vs layer |
| `nb03_attention_component_patching_top1.png` | 各 patch 試行の patched top-1 token (strip plot) |
| `nb03_attention_head_L08_H29.png` | 単独 head 3-panel: clean / corrupt / diff (stable country pointer) |
| `nb03_attention_head_L24_H26.png` | 単独 head 3-panel (country pointer + context shift, causally important) |
| `nb03_attention_head_L17_H17.png` | 単独 head 3-panel (context-sensitive チャンピオン) |
| `nb03_attention_head_L14_H11.png` | 単独 head 3-panel (国名 attention 大反転) |
| `nb03_attention_head_L12_H18.png` | 単独 head 3-panel (逆方向 context shift) |
| `nb03_attention_head_L17_H25.png` | 単独 head 3-panel (top-L1 rank 2: sink ↔ self 切り替え) |
| `nb03_attention_head_L26_H17.png` | 単独 head 3-panel (top-L1 rank 3: " of" を文脈で読みに行く) |
| `nb03_attention_head_L26_H20.png` | 単独 head 3-panel (top-L1 rank 6: clean-dominant BACK) |
| `nb03_attention_head_L16_H25.png` | 単独 head 3-panel (top-L1 rank 7: clean-dominant FRONT) |
| `nb03_attention_head_L16_H27.png` | 単独 head 3-panel (top-L1 rank 8: 対角型 + pos=4 self flip) |
| `nb03_attention_head_L31_H01.png` | 単独 head 3-panel (top-L1 rank 9: corrupt-dominant BACK) |
| `nb03_attention_head_L34_H20.png` | 単独 head 3-panel (top-L1 rank 10: corrupt-dominant deep) |
| `nb03_attention_top10_row_l1_clean.png` | top-10 row_l1 heads の clean 5×5 を 2×5 grid に並べた overview |
| `nb03_attention_top10_row_l1_corrupt.png` | 同 corrupt |
| `nb03_attention_top10_row_l1_diff.png` | 同 clean − corrupt 差分 |
| `nb03_attention_head_L06_H15.png` | 単独 head 3-panel (capital pointer 1 位) |
| `nb03_attention_head_L06_H27.png` | 単独 head 3-panel (capital pointer 2 位) |
| `nb03_attention_head_L01_H26.png` | 単独 head 3-panel (`' of'` pointer 1 位) |
| `nb03_attention_head_L13_H12.png` | 単独 head 3-panel (pure `'The'` sink head) |
| `nb03_attention_head_L14_H30.png` | 単独 head 3-panel (` is` self head) |
| `nb03_attention_score_mean_attn_to_pos0.png` | scalar heatmap: mean attn q=4 → key=0 (`'The'`/sink) |
| `nb03_attention_score_mean_attn_to_pos1.png` | scalar heatmap: mean attn q=4 → key=1 (`' capital'`) |
| `nb03_attention_score_mean_attn_to_pos2.png` | scalar heatmap: mean attn q=4 → key=2 (`' of'`) |
| `nb03_attention_score_mean_attn_to_pos3.png` | scalar heatmap: mean attn q=4 → key=3 (country) |
| `nb03_attention_score_mean_attn_to_pos4.png` | scalar heatmap: mean attn q=4 → key=4 (`' is'`/self) |
| `nb03_attention_per_keypos_top3_overview.png` | 5 key位置 × top-3 head の overview (15 panels) |

---

## 12. 注意事項

- 本実行は MPS + fp16。`run_with_component_capture` が拾った fp32 residual と layer の fp16 出力の比較では、深い layer で 0.1〜数の max abs diff が出る (fp16 表現精度の蓄積)。これは `prelim_attention_residual_update_sanity.csv` から読める。  
  warning threshold は 5.0 に設定しており、今回は 1 件も warning を出さなかった。cuda/bfloat16 に切り替えれば差は変わる。
- `hidden_states[K]` (最終 entry) は **post-final-RMSNorm** で、`layers[K-1].output` ではない。sanity check は L=K-1 のみ post-norm 同士で比較するように分岐させている (`reference` 列で識別)。
- attention pattern の解釈は **小さな prompt の特定インスタンスに対する観察**。general claim ではない。
- component patching は **全 head まとめ** で patch しているので、L24 attn の "中の" どの head が決定的かは未確定。
- chat template を使っていないので、Qwen3-4B 本来の "instruct mode" とは挙動が違う可能性がある。本実験はあくまで pre-training 由来の next-token 統計を見ている。

---

## 13. 関連実験

- note02 ([残差ストリーム logit lens + activation patching](../notebooks/)): 本 script の前提となる residual stream patching を実装。L24 周辺で recovery がピークになる現象を可視化。
- [Experiment 14: Qwen3-4B × mwhanna MLP transcoder — layers 23/24/25](14_qwen3_4b_transcoder_layers23_24_25.md): note02 で着目した L24 周辺の MLP に sparse feature 分解を試みた実験。本 script で同じ層が attention 側からも因果的に効くと確認できた。
- [Experiment 15: Qwen3-4B × mwhanna MLP transcoder — layer sweep](15_qwen3_4b_transcoder_layer_sweep.md): 全 36 layer の MLP transcoder sweep。本 script の `Δmetric_mlp` で見えた L29 ピークと、transcoder feature の active fraction との関係を比較する素材になる。
- [Experiment 16/17: Qwen-Scope SAE smoke](16_qwenscope_sae_qwen3_1p7b_layer20.md): residual stream 側の SAE。本 script の `Δmetric_attn` で見えた "attention が answer を運ぶ" 現象を、SAE feature 側からも追える。
