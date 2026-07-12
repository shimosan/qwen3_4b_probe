# Qwen3-4B のトークン埋め込みに刻まれた多言語構造

**English** → [Multilingual geometry encoded in Qwen3-4B's token embedding matrix](multilingual_geometry_en.md)

多言語モデル **Qwen3-4B** のトークン埋め込み（語彙埋め込み行列 $W_E$、$151936 \times 2560$、入力と出力で共有）に刻まれた多言語構造を観察する。各トークンのベクトルは、共通の出発点に「概念の回転」と「言語の回転」を重ねて作られると考える（回転は正確には直交変換）。回転を重ねる順番で **概念先行モデル** $v_L(w)=R(L)R(w)v_o$・**言語先行モデル** $v_L(w)=R(w)R(L)v_o$・**加法モデル** $v_L(w)=v_\text{en}(w)+a_L$ の 3 つを立て、どれが $W_E$ について最もよく支持されるかを、対訳辞書 **MUSE**（英語を軸にした en-XX、44 言語）を用いて図で確かめる。Qwen3-4B のトークン埋め込みでは概念先行モデルが他の 2 つよりよく支持される。概念先行モデルが当てはまると、ある言語で得たベクトルを概念によらず一つの変換で別言語へ移せる（概念非依存の言語転移）。

手続きの詳細・数式・関連研究は notebook 本体を参照:
[multilingual_geometry_demo.ipynb](lecture/multilingual_geometry_demo.ipynb) ・ [実行結果を見る](rendered/multilingual_geometry_demo.ipynb) ・ [![nbviewer](https://img.shields.io/badge/Render-nbviewer-orange)](https://nbviewer.org/github/shimosan/qwen3_4b_probe/blob/main/rendered/multilingual_geometry_demo.ipynb) ・ [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shimosan/qwen3_4b_probe/blob/main/lecture/multilingual_geometry_demo.ipynb)（CPU で可）

---

## 1. 英語は自語族から離れ CJK と束ねられる

![多言語の階層クラスタリング（Ward）](images/mling_demo_dendro_ward.png)

**Figure 1**: 38 言語の階層クラスタリング（Ward 法、距離 = $\sqrt{1 - \text{言語間の平均コサイン類似度}}$）。葉＝言語（色＝言語グループ）、縦軸＝Ward 連結距離（低いほど早く 1 つの塊になる＝近い）。注目は、**英語 (en) が自分の語族 Germanic（de, nl, af, sv, da, no）ではなく、中国語・日本語・韓国語（CJK）の塊に入る**こと。これは言語の系統では説明できず、Qwen の学習データで中国語・英語が主要言語であることを反映していると見られる（ただし英語ピボット辞書と選択基準の影響は分離できていない。Romance・Slavic などは概ね語族どおりまとまる）。

## 2. 英語–中国語が最も近い言語類似度行列

![言語類似度行列 M（Ward 順）](images/mling_demo_heatmap_ward.png)

**Figure 2**: 言語類似度行列 $M$（各セル＝2 言語間の平均コサイン類似度、濃い赤ほど大、対角は自明なので灰）。行・列は Figure 1 の Ward 順に並べ、ラベルを言語グループ色にした。ko-ja-en-zh が濃い赤のブロックをつくり、最も強いセルは **英語–中国語**（$m = 0.500$）。英語の行・列は多くの言語グループに広く赤みがあり、英語ピボット型データ上でハブ的に振る舞うことが行列の側からも見える。

## 3. 英語からの転移性能は生の近さと少しズレる

![en→L 転移性能 vs 生のコサイン](images/mling_demo_transfer_vs_raw.png)

**Figure 3**: 各点が実験対象の 1 言語（色＝言語グループ）。横軸＝英語との生のコサイン $m_{en,L}$（Figure 2 の $M$ 行）、縦軸＝英語→その言語への「転移性能」（回転 $R(L)$ の推定に使っていない held-out 対訳で測った整列度）。両者は強く相関する（Pearson $r = 0.83$, Spearman $\rho = 0.87$）が完全には一致せず、**Romance（es, pt, fr）は生の近さの割に転移が上位**に来る一方、CJK（zh, ja）は生の近さが最大。生の近さと「回転ひとつで別言語へ移せる度合い（転移性）」は別物で、後者は語族の整列しやすさも反映する（絶対値は作業空間と対の選別に依存するため、順位とズレだけを読む）。

## 4. 生の埋め込みは意味でまとまる（概念優位の表現）

![生の埋め込みの t-SNE](images/mling_demo_raw_tsne.png)

**Figure 4**: 生の埋め込み（変換を一切かけないベクトル）を t-SNE（コサイン距離）で 2 次元化。**ランダムに選んだ 48 個の英単語**とその各言語訳を、英語（黒＝ハブ）を中心に細い線で結んだ「星」で描く（枝＝英語→訳語。近い語だけを選ぶような恣意的な抽出はしていない）。**同じ意味の訳語どうし（別言語）が近くに来る**一方、同一言語グループだけが集まった領域は見当たらない。近さを決めているのは言語よりも意味であり、この生の埋め込みは概念でまとまる＝**概念優位の表現**といえる。

## 5. 逆概念変換で言語構造が顕在化する

![逆概念変換後の t-SNE](images/mling_demo_x_tsne.png)

**Figure 5**: 各言語ごとに対訳ペアから直交変換（回転）$R(L)$ を推定し、概念成分を打ち消す **逆概念変換** $C_L^{-1}(w) = R(L)\,R(w)^{-1}\,R(L)^{-1}$ を適用した後を t-SNE で描いたもの（Figure 4 と同じ表示概念・言語だが、作業空間は生 2560 次元でなく PCA-128。英語点は逆変換で基準 $v_o$ に厳密一致するため個別に描かず + で代表）。**点は概念ごとではなく言語（＝色）ごとにまとまり**、各言語の中心（大きな +）が互いに離れて置かれる＝生の埋め込みに隠れていた言語構造が顕在化した。この変化が明瞭に現れるのは概念先行モデル $v_L(w)=R(L)R(w)v_o$ のもとで共役変換を作る場合だけで、順序を入れ替えた言語先行モデルや、言語差を定数ベクトルで表す加法モデルではそうならない。多言語の対応は「翻訳ベクトルの足し算」ではなく、概念先行の回転（共役）でよく記述される、というのがこのノートの観察。
