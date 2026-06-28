# Experiment 15: Qwen3-4B × mwhanna MLP transcoder — 全 36 layer sweep

Script: [`scripts/15_prelim_qwen3_4b_transcoder_layer_sweep.py`](../scripts/15_prelim_qwen3_4b_transcoder_layer_sweep.py)
最終更新: 2026-05-21
ステータス: ✅ 36 layer 全完走、per-layer combined heatmap 36 枚 + aggregate plots を出力

---

## 1. 目的

[Experiment 14](14_qwen3_4b_transcoder_layers23_24_25.md) で 3 layer (23/24/25) について詳しく見た per-(layer, position) 解析を、**Qwen3-4B の全 36 layer (0..35)** に拡張する。狙い:

- `pos=3` (Japan vs France 識別) と `pos=4` (' is' の前文脈差) の指標が層方向にどう変化するかを完全把握
- 全 36 layer の combined sum+diff heatmap を「記録用」として残し、後で個別 layer 解析を行う際の出発点にする
- L23 でのピーク、L29 の巨大スパイク、L34-35 の出力直前異常など、layer-specific な現象を全 view で観察

## 2. 実験設定

実験設定（model、prompt、token positions、transcoder の数式、PyTorch hook、選択基準）は **[Experiment 14](14_qwen3_4b_transcoder_layers23_24_25.md) と完全に同一**。違うのは:

| 項目 | docs/14 | docs/15 (本書) |
|---|---|---|
| `LAYER_IDXS` | [23, 24, 25] | range(36) = [0, 1, ..., 35] |
| HF cache 取得サイズ | ~5 GB (3 layer 分) | **~60 GB** (36 × 1.68 GB) |
| 実行時間 | ~3 分 | ~15-20 分 |
| per-layer heatmap | 3 枚 | **36 枚** |
| per-(layer, position) CSV | 15 行 | **180 行** |

方法・指標定義・コード対応は **[docs/14 の section 2-5](14_qwen3_4b_transcoder_layers23_24_25.md#2-背景-mlp-transcoder-とは何か)** を参照。

### 指標の再掲（数式のみ）

各 (layer $\ell$, position $p$) について、$\mathbf{f}^{\text{clean}}_{\ell,p}, \mathbf{f}^{\text{corrupt}}_{\ell,p} \in \mathbb{R}^{163840}$ から以下を計算。本 doc では特に重要な 4 指標を line plot:

| 指標 | 数式 | 意味 |
|---|---|---|
| **max\|Δ\|** | $\max_j \|f^{\text{clean}}_j - f^{\text{corrupt}}_j\|$ | outlier 1 個の差分強度 |
| **‖Δ‖₂** | $\sqrt{\sum_j (f^{\text{clean}}_j - f^{\text{corrupt}}_j)^2}$ | 全 features 込みの L2 discrimination |
| **Tanimoto** | $\dfrac{\sum_j \min(f^{\text{clean}}_j, f^{\text{corrupt}}_j)}{\sum_j \max(f^{\text{clean}}_j, f^{\text{corrupt}}_j)}$ | 連続 Jaccard（ReLU の noise floor に robust） |
| **max single** | $\max_j \max(f^{\text{clean}}_j, f^{\text{corrupt}}_j)$ | 強い feature 単体の値 |

加えて、layer-level aggregate（per-position の集約ではない）として:
- **active fraction**: $P(f_j > 0)$ の clean / corrupt 平均
- **reconstruction quality**: 再構成 RMSE と mean cosine

---

## 3. 結果概要

### 3-1. Sanity check（全 36 layer 共通）

- clean top1 = `' Tokyo'` ✓、corrupt top1 = `' Paris'` ✓
- 全 layer で同じ Qwen3-4B model を 1 回 load し、全 36 MLP に hook を付けて forward 1 回ずつ実行
- 各 transcoder safetensors (1.68 GB) を順次 download / encode / del + gc

### 3-2. pos=3 (Japan vs France) の max\|Δ\| layer 軸 trend

```
L  0..4   :  ~1.4 - 3.2    （初期、低い）
L  5..7   :  ~4.6 - 4.8
L  8      :  10.48          ← 第一の中規模ピーク
L  9..21  :  3.4 - 7.2       （振動、中程度）
L 22      :  9.14
L 23      : **19.26**        ← 第一の大ピーク
L 24-25   :  12.07 → 9.26    ← note02 で着目した減衰
L 26-28   :  10.91 → 12.06 → 18.75
L 29      : **46.02**        ← 巨大スパイク（outlier）
L 30-32   :  20-22
L 33-35   :  21 → 25 → 33    ← 後段も高水準
```

→ **3 つの異なる layer regime**:
- 初期 (0-22): 語彙識別 features が断続的に立ち上がる
- 中盤 (23-28): 第一ピーク。note02 の k=24→25 transition と一致
- 後段 (29-35): 巨大化、特に L29 の outlier と L35 の出力層近傍

### 3-3. Tanimoto layer 軸 trend (pos=3)

```
L  0..28: 0.06 - 0.40   （概ね低い、clean と corrupt は active pattern が異なる）
L 24    : 0.375          ← 局所ピーク（feature pattern が一時的に似る）
L 29-32 : 0.22 - 0.50    （上昇傾向）
L 33-34 : 0.50, 0.59
L 35    : 0.789          ← 出力直前、ほぼ similar
```

→ Tanimoto は**最終層に向けて 1.0 に近づく**。後段 layer では「clean prompt と corrupt prompt の active features が 80% 重なる」状態に。これは「lm_head に近づくにつれて両 prompt の representation が最終的な next-token prediction 空間に収束しつつある」ことを示唆。

### 3-4. layer-aggregate metrics

`outputs/prelim_qwen3_4b_transcoder_layer_sweep_summary.csv` 参照。代表値:

| layer | active_frac (clean) | recon_rmse_clean | recon_mean_cos_clean |
|---|---|---|---|
| 0 | 0.0219 | 0.280 | 0.880 |
| 6 | 0.0098 | **40.881** ← 異常 | 0.691 |
| 16 | 0.0359 | **10.083** ← 異常 | 0.807 |
| 23-25 | 0.0006 / 0.0034 / 0.0086 | 0.32 / 0.71 / 0.97 | 0.76 / 0.74 / 0.74 |
| 33-35 | 0.0144 / 0.0135 / 0.0499 | **18.66 / 44.47 / 216.16** ← 後段で急増 | 0.704 / 0.881 / 0.751 |

→ **reconstruction RMSE は layer 6, 16, 33-35 で異常値**。一部 layer は transcoder 学習が不安定か、MLP output 自体のノルムが大きい layer がある可能性（特に layer 35 は lm_head 直前）。

---

## 4. 図（aggregate metrics）

> 注: Fig 1-4 は **pos=3 (Japan/France) と pos=4 (' is' clean vs corrupt) の 2 系列のみ**で描画。pos 0..2 は causal mask により厳密にゼロまたは 1 なので意味のあるトレンドが出ず、見やすさのため省略。全 5 position 版は `outputs/nb03_qwen3_4b_transcoder_layer_sweep_{max_abs_delta, l2_delta, tanimoto, max_single}.png` に記録として残してある。Fig 1-4 は [`scripts/15b_qwen3_4b_transcoder_layer_sweep_replots.py`](../scripts/15b_qwen3_4b_transcoder_layer_sweep_replots.py) が CSV から再生成。

### Figure 1 — max|Δ| 全 layer sweep (outlier discrimination, pos=3, 4)

![max abs delta pos34](images/nb03_qwen3_4b_transcoder_layer_sweep_max_abs_delta_pos34.png)

$$
\max_j \,\bigl|\,f^{\text{clean}}_{\ell, p, j} - f^{\text{corrupt}}_{\ell, p, j}\,\bigr|
$$

x = layer_idx 0..35、赤 = pos=3 (Japan/France)、青 = pos=4 (' is' の前文脈差)。紫破線が layer 24 (note02 reference)。pos=3 は L23 で **小ピーク** (19.26)、L29 で **巨大スパイク** (46.0)、後段で持続的に高い。pos=4 は後段で急増（L31 で 51.0）。

### Figure 2 — ‖Δ‖₂ 全 layer sweep (total L2 discrimination, pos=3, 4)

![l2 delta pos34](images/nb03_qwen3_4b_transcoder_layer_sweep_l2_delta_pos34.png)

$$
\bigl\|\,\mathbf{f}^{\text{clean}}_{\ell, p} - \mathbf{f}^{\text{corrupt}}_{\ell, p}\,\bigr\|_2
$$

max\|Δ\| (Fig 1) と類似する形状。outlier 1 個ではなく全 features 込み L2 ノルムで集約。L29 spike は max\|Δ\| が 46 に対し L2 が 60、滑らかさは増す。

### Figure 3 — Tanimoto 全 layer sweep (連続 Jaccard, pos=3, 4)

![tanimoto pos34](images/nb03_qwen3_4b_transcoder_layer_sweep_tanimoto_pos34.png)

$$
T = \frac{\sum_j \min(f^{\text{clean}}_j,\, f^{\text{corrupt}}_j)}{\sum_j \max(f^{\text{clean}}_j,\, f^{\text{corrupt}}_j)}
$$

非負 vector の連続 Jaccard。**pos=3 (赤) と pos=4 (青) の trend が大きく異なる**:

- **pos=4 (青、' is' clean vs corrupt)**: 前半 L0-5 で **0.76 〜 0.89 と高い**（同じトークン ' is' に対する transcoder feature が前半 layer ではほぼ同じ）、L6 以降に急落して 0.4 周辺で安定、L31-32 で局所的に 0.17-0.20 まで落ち、L35 で 0.40 へ戻る
- **pos=3 (赤、Japan vs France)**: 前半 L0-22 は 0.06 〜 0.33 で低水準、L24 で局所ピーク 0.375 → L25 で 0.155 に急落（[docs/14 で詳述した k=24→25 transition](14_qwen3_4b_transcoder_layers23_24_25.md#5-1-note02-の-k2425-変化との対応) と timing 整合）、L28 以降は **0.22 → 0.39 → 0.44 → 0.50 → 0.50 → 0.59 → 0.79** と単調に近い増加

→ メカニズム解釈は仮説段階だが、「pos=4 は同一トークン由来の features が前半で支配的だったのが、attention で異なる文脈が運ばれて中盤で乖離」「pos=3 は異なる国名 token から始まるが、後段で『首都名予測』に向けて共通 features に収束」。次の Figure 4 (binary Jaccard) と比較。

### Figure 4 — Jaccard (binary, active sets) 全 layer sweep (pos=3, 4)

![jaccard pos34](images/nb03_qwen3_4b_transcoder_layer_sweep_jaccard_pos34.png)

$$
J(A^{\text{clean}}, A^{\text{corrupt}}) = \frac{|A^{\text{clean}} \cap A^{\text{corrupt}}|}{|A^{\text{clean}} \cup A^{\text{corrupt}}|},
\quad A^{\text{clean}} = \{j : f^{\text{clean}}_j > 0\}
$$

magnitude を捨て、**発火 feature の set 一致度のみ**を測る binary 版。mwhanna は ReLU SAE なので **threshold 0 では noise floor も拾う**（特に pos=0 で active count が層を進むほど膨張）caveat あり。Tanimoto (Fig 3) と比較すると：

- **中盤までは Tanimoto と Jaccard はほぼ同じ shape** で動く → 両指標の差は magnitude weighting だけだが、中盤までは大きな寄与の features が active set 全体を支配しているため両指標が連動
- **後段で乖離**: 特に pos=3 (赤) で L33-35 を見ると、Tanimoto は 0.50 → 0.59 → 0.79 と急上昇する一方、Jaccard は 0.53 → 0.41 → 0.29 と寧ろ低下。これは「**重なる features の個数自体は減るが、その少数の重なる features が両 prompt で巨大な activation を持つようになる**」ことを意味する（最終層 lm_head 直前で「答えに直結する features」が両 prompt で支配的になる）
- デモで Tanimoto と Jaccard を**両方並べる**価値は、この後段の divergence にある

### Figure 5 — max single activation (pos=3, 4)

![max single pos34](images/nb03_qwen3_4b_transcoder_layer_sweep_max_single_pos34.png)

$$
\max_j \,\max\bigl(f^{\text{clean}}_{\ell, p, j},\, f^{\text{corrupt}}_{\ell, p, j}\bigr)
$$

「最も強く発火する single feature」。pos=3 (赤) は後段に向けて爆発的に増加（L35 で **291.91**）、pos=4 (青) も後段で 100 を超える。これは出力直前の features が「絶対値スケールで巨大化」する一般現象を反映。

### Figure 6 — max single activation 全 5 position 比較 (log y)

![max single log](images/nb03_qwen3_4b_transcoder_layer_sweep_max_single_log.png)

Figure 5 と同じ指標を **全 5 position + 縦軸 log スケール**で描画。pos 0..2 (グレー = causal mask、clean = corrupt) も後段で大きな値になる事実が見える（pos=0 ' The' は前段 ' Japan' / ' France' より小さいが、後段では同程度）。これは「**後段 layer の feature scale 自体が token 内容によらず大きくなる**」現象。Fig 5 の絶対値ではなく Fig 6 の log spread で「各 position の **相対的** な engagement」を観察できる。

### Figure 7 — active fraction (layer-level aggregate)

![active fraction sweep](images/nb03_qwen3_4b_transcoder_layer_sweep_active_fraction.png)

mean active fraction = $\mathbb{E}_{t, j}[\mathbb{1}[f_{t,j} > 0]]$、clean / corrupt 別に prompt 全体（5 token × 163840 feature）で平均。

→ **U 字 + 後段増加**: layer 0 (2.2%) → layer 3-7 (0.01-0.1%、極めて sparse) → 後段で再増 (L35 で 5.0%)。中段で features が最も「絞られる」。**L16 と L28 のピーク**が特徴的だが、L16 については `outputs/nb03_qwen3_4b_transcoder_layer16_feature_heatmap.png`（Appendix 参照）で pos=0 の特異な active 数（2731）が原因と推測される。

### Figure 8 — active feature count per (layer, position, prompt) — log y

![active count per position](images/nb03_qwen3_4b_transcoder_layer_sweep_active_count_per_position.png)

Figure 7 が「全 5 position × 163840 features 平均の active fraction」を 1 数値で見せたのに対し、本図は **(layer, position, prompt) 単位の active count** を分解。pos 0..2 (グレー、causal mask により clean=corrupt なので 1 線で表示) と pos=3, 4 (赤・青、clean=実線 / corrupt=破線) で計 7 線。

→ pos 0..2 が**指数的に増加** (L0 で 349 → L35 で数千 〜 数万) する一方、pos=3, 4 は数十〜数百のオーダーで比較的安定。「pos=0 'The' での active 数が後段で爆発」が dominant な現象であることが明確。

### Figure 9 — reconstruction quality (RMSE log y + mean cosine linear)

![reconstruction log](images/nb03_qwen3_4b_transcoder_layer_sweep_reconstruction_log.png)

上段 = RMSE（**log y スケール**）、下段 = mean cosine（linear）。log で見ると **layer 6 (40.9)、L16 (10.1)、L33-35 (18.7, 44.5, 216.2)** の異常層が明確、また異常でない中間層も 0.04 〜 2.3 の幅で構造的に layer-by-layer 変動する事実が読める。cosine は概ね 0.6-0.9 範囲、後段に向けて緩やかに上昇。後段で MLP output 自体の magnitude が大きくなるため RMSE 比較は単純でない（次の section 5 参照）。

---

## 5. 解釈

### 5-1. 「3 つの regime」

pos=3 max\|Δ\| の curve から、layer 軸で **3 つの regime** に分かれる:

1. **初期 (L0-22)**: 語彙レベル識別が散発的、量は 1-10 範囲
2. **transition (L23-28)**: 第一ピーク layer 23 (19.26)、その後 L24-25 で減衰（note02 の k=24→25 transition と整合）→ 再び立ち上がり
3. **後段 (L29-35)**: 出力近傍。outlier feature の影響が支配的、量も巨大化

### 5-2. note02 (residual stream patching) との対応

[docs/14 section 8](14_qwen3_4b_transcoder_layers23_24_25.md#8-1-note02-の-k2425-変化との対応) で議論した「pos=3 の discrimination がピーク → last position の discrimination 立ち上がり」物語が、全 36 layer view で確認できる:

- pos=3 max\|Δ\| は L23 でピーク (19.26)
- pos=4 (last ' is') の max\|Δ\| は後段で急増（L29: 44.40, L31: 51.01）
- 「pos=3 のピーク layer の後で、語彙情報が pos=4 へ流れる」trend が **layer 軸で明確**

### 5-3. Tanimoto / Jaccard の興味深い trend

Tanimoto (Fig 3) と binary Jaccard (Fig 4) はどちらも「重なり度」を測るが、性質が違う:
- Tanimoto: $\sum_j \min(c_j, k_j) / \sum_j \max(c_j, k_j)$、magnitude-aware
- Jaccard: $|A^c \cap A^k| / |A^c \cup A^k|$、binary（閾値 0、ReLU の noise floor 込み）

#### pos=3 (Japan vs France) — 後段で収束

L0-22 は 0.06 〜 0.33 の低水準を振動。L23 は 0.25 (Tan) / 0.35 (Jac)。**L24 で局所ピーク 0.375 (Tan) / 0.33 (Jac) → L25 で 0.155 / 0.22 に急落**（[docs/14](14_qwen3_4b_transcoder_layers23_24_25.md) で詳述した k=24→25 transition と timing 整合、note02 の patching jump も同位置）。L28 以降は Tanimoto が **0.22 → 0.39 → 0.44 → 0.50 → 0.50 → 0.59 → 0.79** と単調増加し、L35 で 0.79 まで到達。一方 Jaccard は L32-33 で 0.50-0.53 を一旦記録した後、L34-35 で 0.41, 0.29 と寧ろ減少。

→ 後段で「**重なる features の個数自体は減るが、残った少数の features が両 prompt で巨大な activation を持つ**」。Tanimoto は magnitude を反映して急上昇、Jaccard は count なので減少。

#### pos=4 (' is' clean vs corrupt) — 前半で高く、中盤で下降

L0-5 で Tanimoto が **0.76 〜 0.89 と高い**（同じトークン ' is' に対する transcoder feature が前半 layer ではほぼ同じ；attention で運ばれる上流の Japan/France 文脈の差が transcoder feature に反映される前段階）。L6 で急落 (0.59)、その後緩やかに減少して L13 で 0.29 / L15 で 0.20 まで下がり、中盤で 0.3-0.5 を振動。L31-32 で 0.17-0.20 と局所最小、L33-35 で 0.31, 0.41, 0.40 と回復。

→ pos=4 の前半が高いのは「上流文脈がまだ十分伝播していない、token ' is' そのもののスタンプが支配的」と解釈できる。

#### 図的に重要な観察

- **pos=3 と pos=4 の trend は対称的ではない**: pos=3 は前半低・後段高、pos=4 は前半高・中盤低・後段中。両者を重ね描きすることで「同じ Tanimoto 値でも layer によって意味する内容が違う」が見える
- **Tanimoto と Jaccard は中盤までは並行**、後段（特に L33-35 の pos=3）で divergence → 後段 layer の少数 features の magnitude-dominance が明確
- L24-25 の Tanimoto V 字（pos=3）は note02 patching jump と同 layer で起こる

### 5-4. active fraction のピーク (Fig 7)

active fraction は L16 と L28 に顕著なピークがある:

- **L16 のピーク (3.6%)**: Appendix A の `layer16` heatmap を見ると pos=0 ('The') で異常に多くの features が発火している。中段で「first-token attractor が特に強い layer」と推測される
- **L28 のピーク (2.3%)**: 後段の features の活発化開始。L29 の巨大スパイク（Fig 1 の max\|Δ\|=46）と timing がほぼ一致

これらは active count per position (Fig 8) で詳しく分解できる（pos=0 で支配的、pos=3, 4 は比較的安定）。

### 5-5. 後段 layer の異常値群

L29 spike (max\|Δ\|=46)、L33-35 の reconstruction RMSE 異常、L35 max_single 291.91 など。これらは:

- transcoder 学習時に出力近傍の MLP が捉えにくかった可能性
- 出力 layer 自体の magnitude scale が大きい
- 一部 outlier feature (lm_head に向けて) の影響

→ **デモではこれらの異常値は注意して扱う**。本質的な「discrimination が後段で大きい」trend は real だが、絶対値そのものは layer 間で直接比較しにくい。

---

## 6. 応用への示唆

- **デモ映え**: Figure 1 (max\|Δ\|) と Figure 3 (Tanimoto) を並べて見せると、「max は outlier、Tanimoto は set 類似度」という 2 つの異なる視点で同じデータを語れる
- **関連 notebook**: nb02 (residual stream patching) の k=24→25 transition との対応を、Figure 1 で全 36 layer 視点から補強。「特定 layer (23-25) だけでなく全 layer 範囲で見ると、その transition は『より大きな構造』の一部」と説明可能
- **再利用したい figure**: Figure 1 (max\|Δ\|) と Figure 3 (Tanimoto) が主役、Figure 9 (reconstruction RMSE log) は品質チェック / 注意事項として
- **per-layer detail**: 興味ある layer の combined heatmap は Appendix A 参照

## 7. 出力ファイル

aggregate (script 15 generated):
- `outputs/prelim_qwen3_4b_transcoder_layer_sweep_summary.csv` — 36 行 × 30 列（per-layer aggregate）
- `outputs/prelim_qwen3_4b_transcoder_layer_sweep_summary.json` — メタ情報込み
- `outputs/prelim_qwen3_4b_transcoder_layer_sweep_position_metrics.csv` — **180 行 × 22 列**（per-(layer, position) 詳細）
- `outputs/nb03_qwen3_4b_transcoder_layer_sweep_max_abs_delta.png` — 全 5 position 版（記録）
- `outputs/nb03_qwen3_4b_transcoder_layer_sweep_l2_delta.png` — 同上
- `outputs/nb03_qwen3_4b_transcoder_layer_sweep_tanimoto.png` — 同上
- `outputs/nb03_qwen3_4b_transcoder_layer_sweep_max_single.png` — 同上
- `outputs/nb03_qwen3_4b_transcoder_layer_sweep_active_fraction.png`
- `outputs/nb03_qwen3_4b_transcoder_layer_sweep_reconstruction.png`

aggregate (script 15b CSV-driven replots、本書 Fig 1-6, 8, 9):
- `outputs/nb03_qwen3_4b_transcoder_layer_sweep_max_abs_delta_pos34.png` — pos=3, 4 のみ
- `outputs/nb03_qwen3_4b_transcoder_layer_sweep_l2_delta_pos34.png`
- `outputs/nb03_qwen3_4b_transcoder_layer_sweep_tanimoto_pos34.png`
- `outputs/nb03_qwen3_4b_transcoder_layer_sweep_jaccard_pos34.png` — binary Jaccard、新規
- `outputs/nb03_qwen3_4b_transcoder_layer_sweep_max_single_pos34.png`
- `outputs/nb03_qwen3_4b_transcoder_layer_sweep_max_single_log.png` — 全 5 position、log y
- `outputs/nb03_qwen3_4b_transcoder_layer_sweep_active_count_per_position.png` — per (pos × prompt)、log y
- `outputs/nb03_qwen3_4b_transcoder_layer_sweep_reconstruction_log.png` — RMSE log y

per-layer (× 36 layer):
- `outputs/prelim_qwen3_4b_transcoder_layer{0..35}_feature_matrix.csv` — 各 11 行 × 301 列 (token_label + 300 features pool)
- `outputs/nb03_qwen3_4b_transcoder_layer{0..35}_feature_heatmap.png` — 各 layer の combined sum + diff heatmap

## 8. 注意事項

- HF cache 約 60 GB を消費（36 × 1.68 GB）。disk 容量事前確認
- 実行時間 ~15-20 分
- layer 間で feature_id を直接比較しない（各 layer の transcoder は独立学習で、feature 空間が違う）
- 後段 layer の指標値（特に max_single, RMSE）は magnitude が大きすぎるので、layer 間比較する際は normalize するか log scale 検討
- **mwhanna の transcoder は Qwen 公式の Qwen-Scope SAE ではなく、community が公開している MLP transcoder** である。residual stream SAE ではなく、MLP input から MLP output を sparse に近似する道具である点に注意。詳細は [docs/14 §12](14_qwen3_4b_transcoder_layers23_24_25.md) を参照。

## 9. 関連実験

- [Experiment 14](14_qwen3_4b_transcoder_layers23_24_25.md) — 同 transcoder の layer 23/24/25 詳細版（方法と指標定義を full 説明）
- [`scripts/15b_qwen3_4b_transcoder_layer_sweep_replots.py`](../scripts/15b_qwen3_4b_transcoder_layer_sweep_replots.py) — 本 doc の Fig 1-6, 8, 9 を CSV から再生成する replot script。transcoder weights や model を再 load せず、CSV から軽く再描画する
- script 16, 17 — Qwen-Scope (TopK SAE) との比較（別 transcoder ファミリー）
- script 18 — note02 (residual patching + logit lens) との対応

---

## Appendix A. 全 36 per-layer combined sum + diff heatmap

各 layer の combined heatmap を記録用に全て掲載する。レイアウトは [docs/14 の Figure 1](14_qwen3_4b_transcoder_layers23_24_25.md#7-図) と同じ:

- 上段 = sum (clean + corrupt, viridis)
- 下段 = diff (clean − corrupt, RdBu_r diverging)
- 列 = top 60 features by max-over-10-cells、同一順
- 行 = 5 token position

注意:
- pos 0..2 (`'The'`, `' capital'`, `' of'`) は causal mask により diff 行が真っ白
- pos 3 (`' Japan'` vs `' France'`) と pos 4 (`' is'` clean/corrupt) で diff に色が乗る
- feature_id は per-layer なので異なる layer の同じ id は別 feature を指す

### Layers 0-5

![layer 0](images/nb03_qwen3_4b_transcoder_layer0_feature_heatmap.png)
![layer 1](images/nb03_qwen3_4b_transcoder_layer1_feature_heatmap.png)
![layer 2](images/nb03_qwen3_4b_transcoder_layer2_feature_heatmap.png)
![layer 3](images/nb03_qwen3_4b_transcoder_layer3_feature_heatmap.png)
![layer 4](images/nb03_qwen3_4b_transcoder_layer4_feature_heatmap.png)
![layer 5](images/nb03_qwen3_4b_transcoder_layer5_feature_heatmap.png)

### Layers 6-11

![layer 6](images/nb03_qwen3_4b_transcoder_layer6_feature_heatmap.png)
![layer 7](images/nb03_qwen3_4b_transcoder_layer7_feature_heatmap.png)
![layer 8](images/nb03_qwen3_4b_transcoder_layer8_feature_heatmap.png)
![layer 9](images/nb03_qwen3_4b_transcoder_layer9_feature_heatmap.png)
![layer 10](images/nb03_qwen3_4b_transcoder_layer10_feature_heatmap.png)
![layer 11](images/nb03_qwen3_4b_transcoder_layer11_feature_heatmap.png)

### Layers 12-17

![layer 12](images/nb03_qwen3_4b_transcoder_layer12_feature_heatmap.png)
![layer 13](images/nb03_qwen3_4b_transcoder_layer13_feature_heatmap.png)
![layer 14](images/nb03_qwen3_4b_transcoder_layer14_feature_heatmap.png)
![layer 15](images/nb03_qwen3_4b_transcoder_layer15_feature_heatmap.png)
![layer 16](images/nb03_qwen3_4b_transcoder_layer16_feature_heatmap.png)
![layer 17](images/nb03_qwen3_4b_transcoder_layer17_feature_heatmap.png)

### Layers 18-23

![layer 18](images/nb03_qwen3_4b_transcoder_layer18_feature_heatmap.png)
![layer 19](images/nb03_qwen3_4b_transcoder_layer19_feature_heatmap.png)
![layer 20](images/nb03_qwen3_4b_transcoder_layer20_feature_heatmap.png)
![layer 21](images/nb03_qwen3_4b_transcoder_layer21_feature_heatmap.png)
![layer 22](images/nb03_qwen3_4b_transcoder_layer22_feature_heatmap.png)
![layer 23](images/nb03_qwen3_4b_transcoder_layer23_feature_heatmap.png)

### Layers 24-29

![layer 24](images/nb03_qwen3_4b_transcoder_layer24_feature_heatmap.png)
![layer 25](images/nb03_qwen3_4b_transcoder_layer25_feature_heatmap.png)
![layer 26](images/nb03_qwen3_4b_transcoder_layer26_feature_heatmap.png)
![layer 27](images/nb03_qwen3_4b_transcoder_layer27_feature_heatmap.png)
![layer 28](images/nb03_qwen3_4b_transcoder_layer28_feature_heatmap.png)
![layer 29](images/nb03_qwen3_4b_transcoder_layer29_feature_heatmap.png)

### Layers 30-35

![layer 30](images/nb03_qwen3_4b_transcoder_layer30_feature_heatmap.png)
![layer 31](images/nb03_qwen3_4b_transcoder_layer31_feature_heatmap.png)
![layer 32](images/nb03_qwen3_4b_transcoder_layer32_feature_heatmap.png)
![layer 33](images/nb03_qwen3_4b_transcoder_layer33_feature_heatmap.png)
![layer 34](images/nb03_qwen3_4b_transcoder_layer34_feature_heatmap.png)
![layer 35](images/nb03_qwen3_4b_transcoder_layer35_feature_heatmap.png)
