# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: aidemo2026
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Qwen3-4B の語彙埋め込みに刻まれた多言語構造
#
# 多言語モデル **Qwen3-4B** は、文字列をトークンと呼ばれる語や部分語の単位に区切り、各トークンを 2560 次元のベクトルに対応づけて
# から計算を始める。この対応表を **トークン埋め込み** と呼び、その本体が大きさ $151936 \times 2560$ の **語彙埋め込み行列** $W_E$ である。
# $W_E$ は入力側と出力側で共有される。Qwen3-4B は多言語モデルなので、その語彙には英語や中国語、日本語、韓国語など、さまざまな言語の
# トークンが含まれる。本ノートでは、このトークン埋め込みの中で、言語の違いと、概念すなわち意味の違いが、どう組み合わさっているかを調べる。
#
# 各トークンのベクトルは、共通の出発点に「概念の回転」と「言語の回転」を重ねて作られる、と考える。ここでいう回転は、正確には直交変換の
# ことである。回転を重ねる順番には 2 通りある。概念を先に、言語をあとに作用させるものを **概念先行モデル** と呼び、言語を先に、概念を
# あとに作用させるものを **言語先行モデル** と呼ぶ。さらに単純な比較対象として、言語の違いを一定のベクトルの足し算で表す **加法モデル**
# も考える。
#
# どのモデルが実際の埋め込みをよく近似するかが問題になる。概念先行モデルか加法モデルが当てはまるなら、ある言語で得たトークンのベクトル
# を、概念ごとに変換を作り直すことなく一つの決まった変換で別の言語へ移せる。これは、言語をまたいで知識を効率よく持ち運べる構造を、多言語モデルが内部に備えて
# いることを示唆する。本ノートでは、Qwen3-4B のトークン埋め込みでは概念先行モデルが他の 2 つよりよく支持されることを、
# 図と数値で確かめる。
#
# 進め方は次のとおりである。まず語彙埋め込み行列 $W_E$ を概観し、多言語のクラスタリングによって、英語が自分の語族である Germanic では
# なく、中国語や日本語、韓国語からなる CJK と束ねられることを確かめる。次に、言語の回転を精度よく推定するために、対訳データが十分に
# そろう言語を実験対象として選ぶ。最後に、三つのモデルがそれぞれ予測する変換を適用し、概念の違いを抑えたときに、
# どのモデルで言語の構造が最も明瞭に現れるかを比較する。
# 
# モデル本体の 40 億パラメータは動かす必要がなく、埋め込み行列 1 枚だけを読む。テキスト生成もしないので、
# メモリの負担は小さく、安全に実行できる。
#
# %% [markdown]
# ## 0. 埋め込み行列 $W_E$ を読む
#
# Qwen3-4B は複数ファイル（shard）に分割保存されている。本ノートは $W_E$（`embed_tokens`）を含む 1 ファイルだけを
# ダウンロードして読む。モデル全体（約 8GB）を読み込まないので、ノート PC でも Google Colab でも動く。

# %%
# 3 環境（Mac / Win / Colab）を同一ファイルで動かす。Colab のみ pip / フォント導入（Mac/Win は環境に在るので skip）。
import sys, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers==5.9.0", "safetensors", "huggingface_hub",
                    "arabic-reshaper", "python-bidi"], check=True)
    # Colab(Linux) は日中韓・アラビアフォントが無いので apt で導入（図中の CJK が □ 豆腐になるのを防ぐ）
    subprocess.run(["apt-get", "-qq", "-y", "install",
                    "fonts-ipafont-gothic", "fonts-wqy-zenhei", "fonts-nanum", "fonts-noto-core"], check=False)
    print("Colab: pip / fonts done")
else:
    print("local(Mac/Win): pip skip (using existing environment)")

import json, os, glob, logging, itertools, urllib.request
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse, Circle

# 図の保存先（他 notebook と同じ規約）。Colab は outputs/、ローカルはノートの 1 つ上の outputs/。
outputs_dir = Path("outputs") if IN_COLAB else Path("../outputs")
outputs_dir.mkdir(parents=True, exist_ok=True)
print("outputs :", outputs_dir)

# 多言語フォント設定（3環境 Mac / Win / Colab）: CJK(日中韓ハングル) + アラビア語/ヘブライ語。
# 方式は 02 ノート（正本・何度も検証済み）に合わせる。matplotlib の既定 DejaVu Sans は CJK glyph を
# 持たないので、font.family にリストを与えて「字ごとに先頭から fallback」させる（matplotlib >= 3.6）。
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
# macOS: .ttc は自動検出が不安定なので明示登録（保険）
for fp in ["/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/AppleSDGothicNeo.ttc",
           "/Library/Fonts/AppleGothic.ttf"]:
    if Path(fp).exists():
        try:
            font_manager.fontManager.addfont(fp)
        except Exception:
            pass
# Colab(Linux): apt で入れた CJK/アラビアフォントを登録（apt 導入は上の IN_COLAB ブロック）
if IN_COLAB:
    for _pat in ("/usr/share/fonts/**/ipag*.ttf", "/usr/share/fonts/**/wqy-zenhei*.tt?",
                 "/usr/share/fonts/**/NanumGothic*.ttf", "/usr/share/fonts/**/NotoSansArabic*.ttf",
                 "/usr/share/fonts/**/NotoNaskhArabic*.ttf", "/usr/share/fonts/**/NotoSans-*.ttf"):
        for _fp in glob.glob(_pat, recursive=True):
            try:
                font_manager.fontManager.addfont(_fp)
            except Exception:
                pass
_available = {f.name for f in font_manager.fontManager.ttflist}
_font_candidates = [
    # 日本語
    "Hiragino Sans", "Hiragino Sans GB", "Yu Gothic", "Meiryo", "IPAGothic", "Noto Sans CJK JP", "Noto Sans JP",
    # 中国語(簡体)  Mac=PingFang / Win=Microsoft YaHei / Colab=WenQuanYi
    "PingFang SC", "Microsoft YaHei", "WenQuanYi Zen Hei", "Noto Sans CJK SC",
    # 韓国語(ハングル)  Mac=Apple SD Gothic Neo / Win=Malgun Gothic / Colab=Nanum
    "Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic", "NanumGothic", "Nanum Gothic",
    "Noto Sans CJK KR", "Noto Sans KR",
    # アラビア語/ヘブライ語(字形はフォント、RTL 並べ替えは _fix_rtl)  Mac=Geeza Pro / Win=Segoe UI / Colab=Noto
    "Geeza Pro", "Segoe UI", "Noto Sans Arabic", "Noto Naskh Arabic", "Noto Sans Hebrew",
    # 全部入り(Mac)
    "Arial Unicode MS",
]
plt.rcParams["font.family"] = [n for n in _font_candidates if n in _available] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
print("font.family =", plt.rcParams["font.family"])

# RTL(アラビア語/ヘブライ語) 整形。libraqm 付き matplotlib は native 整形するので raw で正しい。
# libraqm 非搭載(一部 Linux/Colab)だけ arabic-reshaper + python-bidi で手動整形（版でなくバイナリで決まる）。
RESHAPE_RTL = "auto"   # "auto"=libraqm有無で自動 / True=必ず手動 / False=整形しない
try:
    import matplotlib.ft2font as _ft2
    _MPL_HAS_RAQM = bool(getattr(_ft2, "__libraqm_version__", ""))
except Exception:
    _MPL_HAS_RAQM = False
def _need_manual_reshape():
    return (not _MPL_HAS_RAQM) if RESHAPE_RTL == "auto" else bool(RESHAPE_RTL)
try:
    import arabic_reshaper as _arsh
    from bidi.algorithm import get_display as _bidi_disp
    import re as _re_rtl
    _RTL_RE = _re_rtl.compile("[֐-ࣿﭐ-﷿ﹰ-﻿]")
    def _fix_rtl(_s):  # pyright: ignore[reportRedeclaration]
        if _need_manual_reshape() and _s and _RTL_RE.search(_s):
            return _bidi_disp(_arsh.reshape(_s))
        return _s
except Exception:
    def _fix_rtl(_s):
        return _s
print(f"[RTL] libraqm={_MPL_HAS_RAQM} RESHAPE_RTL={RESHAPE_RTL} -> manual shaping={_need_manual_reshape()}")

# %%
# 言語グループ（可視化用の色分け。Romance/Germanic は系統分類だが CJK/Viet/Thai は地域・文字圏・個別言語が混在し、必ずしも系統分類ではない）の統一カラーコード。以降の図（ヒートマップ・デンドロ・回転図）で共通に使う。
# 図中の文字は英語（あとで全英語版へ流用しやすくするため）。本文の説明は日本語。
# MUSE en-XX の全候補言語（44）の言語グループをすべて定義しておく。対象言語を変えても壊れないよう、
#   未知の言語は fam_of()/fam_color() が "Other"（灰色）にフォールバックする。
CANON_FAM = {
    "CJK": "#e41a1c", "Romance": "#ff7f00", "Germanic": "#f781bf", "Slavic": "#377eb8",
    "Semitic": "#4daf4a", "Iranian": "#a65628", "Turkic": "#984ea3", "Viet": "#999999",
    "Thai": "#bcbd22", "Austronesian": "#00ced1", "Uralic": "#1b9e77", "Baltic": "#8c6d31",
    "Albanian": "#7570b3", "Hellenic": "#66a61e", "Indic": "#e6ab02", "Dravidian": "#e7298a",
    "Other": "#cccccc",
}
# 言語コード -> 言語グループ（MUSE en-XX 全 44 候補を網羅）
FAM = {
    "en": "Germanic", "de": "Germanic", "nl": "Germanic", "sv": "Germanic", "da": "Germanic", "no": "Germanic", "af": "Germanic",
    "fr": "Romance", "es": "Romance", "it": "Romance", "pt": "Romance", "ro": "Romance", "ca": "Romance",
    "zh": "CJK", "ja": "CJK", "ko": "CJK",
    "ru": "Slavic", "pl": "Slavic", "uk": "Slavic", "cs": "Slavic", "sk": "Slavic", "sl": "Slavic",
    "hr": "Slavic", "bs": "Slavic", "bg": "Slavic", "mk": "Slavic",
    "ar": "Semitic", "he": "Semitic", "fa": "Iranian", "tr": "Turkic", "vi": "Viet", "th": "Thai",
    "id": "Austronesian", "ms": "Austronesian", "tl": "Austronesian", "fi": "Uralic", "hu": "Uralic", "et": "Uralic",
    "lv": "Baltic", "lt": "Baltic", "sq": "Albanian",
    "el": "Hellenic", "hi": "Indic", "bn": "Indic", "ta": "Dravidian",
}


def fam_of(L):
    """言語コード -> 言語グループ。未知の言語は "Other"（対象言語を変えても壊れないようにするため）。"""
    return FAM.get(L, "Other")


def fam_color(L):
    """言語コード -> 言語グループの色（統一パレット）。未知は灰色。"""
    return CANON_FAM.get(fam_of(L), "#cccccc")

# %%
# 埋め込み行列 W_E を「embed_tokens を含む shard だけ」ダウンロードして読む（モデル本体はロードしない）
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

MODEL_ID = "Qwen/Qwen3-4B"
# どの shard に埋め込みが入っているかは index ファイルに書いてある
weight_map = json.loads(open(hf_hub_download(MODEL_ID, "model.safetensors.index.json")).read())["weight_map"]
shard = hf_hub_download(MODEL_ID, weight_map["model.embed_tokens.weight"])   # 未取得ならこの 1 ファイルだけ DL
with safe_open(shard, framework="pt") as f:
    W_E = f.get_tensor("model.embed_tokens.weight").float().numpy()          # 形: 151936 x 2560
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("W_E:", W_E.shape, "  vocab size x dim")

_tok1_cache = {}


def tok1(s):
    """文字列 s が「ちょうど 1 トークン」になるならその id を返す。ならなければ None。
    英単語は語頭スペース付き（Qwen の流儀）を優先し、CJK は素の文字で試す。"""
    if s in _tok1_cache:
        return _tok1_cache[s]
    r = None
    for c in (" " + s, s):
        ids = tokenizer.encode(c, add_special_tokens=False)
        if len(ids) == 1:
            r = ids[0]; break
    _tok1_cache[s] = r
    return r


def u(v):
    """ベクトルを長さ 1 に正規化する（向きだけを見るため）。"""
    return v / (np.linalg.norm(v) + 1e-12)


def e(t):
    """トークン id t の埋め込みベクトル（長さ 1 に正規化済み）。"""
    return u(W_E[t].astype(np.float64))


# %% [markdown]
# ## Part 1　多言語をクラスタリングして全体像を見る
#
# **目的**: 言語どうしがどれだけ近いかを測り、階層クラスタリングで「どの言語が同じ塊になるか」を見る。
# あらかじめ言うと、英語は自分の語族（Germanic）ではなく **中国語・日本語・韓国語（CJK）と同じ塊**に入る。
# これは言語の系統（語族）では説明できない。Qwen の学習データで中国語・英語が主要言語であること
# を反映していると推測される（ただし学習コーパスは非公開のため、特定のメカニズムまでは確証がなく、英語ピボット辞書と選択基準の影響も分離できていない）。

# %% [markdown]
# ### 1.1 英語ピボットの MUSE 対訳辞書
#
# 言語間の「同じ意味の単語ペア」には **MUSE の対訳辞書**を使う。MUSE は Facebook AI Research が公開する
# 多言語単語埋め込みのライブラリで、評価・学習用の**対訳辞書**が付属している（本プロジェクトはこの辞書だけを使う）。
#
# - 論文: Conneau, Lample, Ranzato, Denoyer, Jégou, [*Word Translation Without Parallel Data*](https://arxiv.org/abs/1710.04087), ICLR 2018。
# - 本家リポジトリ: [github.com/facebookresearch/MUSE](https://github.com/facebookresearch/MUSE)
# - 配布元: `https://dl.fbaipublicfiles.com/arrival/dictionaries/en-{XX}.txt`（各言語ファイルを直接取得する。ディレクトリ一覧は不可）。英語を起点にした `en-XX` 辞書が **44 言語分**ある（英語自身を除く。次の 1.2 でこの全部を使う）。
#
# ★重要な前提: MUSE の辞書は **すべて英語を軸にしている**（`en-XX`＝英語→各言語）。
# 日本語↔中国語のように英語を介さない直接辞書は無い。したがって日中を比べるときも共通概念は英語経由で定義される
# （`en-ja` と `en-zh` の両方に載っている英単語を、日中で共通の概念とみなす）。この en-XX 構造から、本ノートでは
# **概念を 1 つの英単語 $w$ で表す**（概念 $w$ の実体は、各言語での $w$ の訳語）。概念変数は全パートで一貫して英単語 $w$ である。
# つまり **英語はデータの作り方の時点で構造的に中心**に置かれている。後で「英語がハブ」という結果が出るが、
# その一部はこの英語ピボットの帰結でもある。この点は結果を読むときに念頭に置く。

# %% [markdown]
# ### 1.2 対象言語を「Qwen が単一トークンにできるか」で選ぶ
#
# MUSE が出す **en-XX 全 44 言語**（英語自身を除く）をすべてダウンロードし、そこから対象言語を **1 つの定量基準で機械的に選ぶ**。
# 基準は「その言語の訳語が Qwen で **1 トークン**になる内容語の対訳ペアが、中頻度バンド（§1.3）内に **100 組以上**そろうこと」
# （＝ 各語のベクトルが一意に定まり、比較できること）。この基準はトークナイザだけで決まり、後段の結論（言語の近さ）に依存しない。
# どの言語が残り、どれがなぜ落ちるかは、次の棒グラフで見る。

# %%
# MUSE 対訳辞書を用意する（初回のみ DL）。
# ★注意: これは MUSE 辞書のキャッシュで、Hugging Face のモデルキャッシュ（~/.cache/huggingface）とは【別物】。
#   保存先 ~/.cache/muse_full/。en 自身を除く 44 言語・計 ~50MB。
MUSE_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-{}.txt"
MUSE_DIR = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "muse_full"
MUSE_DIR.mkdir(parents=True, exist_ok=True)
# MUSE 本家 README の en-XX 全リスト（en 自身は除く）＝ 44 言語
CAND_LANGS = ["af", "sq", "ar", "bn", "bs", "bg", "ca", "zh", "hr", "cs", "da", "nl", "et", "tl", "fi",
              "fr", "de", "el", "he", "hi", "hu", "id", "it", "ja", "ko", "lv", "lt", "mk", "ms", "no",
              "fa", "pl", "pt", "ro", "ru", "sk", "sl", "es", "sv", "ta", "th", "tr", "uk", "vi"]
n_new = 0
for L in CAND_LANGS:
    p = MUSE_DIR / f"en-{L}.txt"
    if not p.exists():
        try:
            urllib.request.urlretrieve(MUSE_URL.format(L), p); n_new += 1
        except Exception as ex:
            print(f"  en-{L}: download failed {ex}")
print(f"[MUSE dictionaries] {len(CAND_LANGS)} candidate languages (newly downloaded this run: {n_new})  cache: {str(MUSE_DIR).replace(str(Path.home()), '~')}")

# 機能語（内容語だけ使うための除外セット）
STOP = set("a an the this that these those and or but if then else for nor so yet of to in on at by with from up down out off over under again about into as is are was were be been being am do does did have has had having i you he she it we they me him her us them my your his its our their not no yes very can will just should would could may might must shall here there where when why how what who whom which than too also more most some any all each every both few many much other another such own same then once".split())
BAND_LO, BAND_HI = 2000, 12000    # 中頻度バンド（根拠は 1.3）


def n_single_token_pairs(L):
    """言語 L で「en も訳語も 1 トークン・内容語・相異トークン・band 内」の対訳ペア数（選定基準）。"""
    seen = set(); n = 0
    for line in (MUSE_DIR / f"en-{L}.txt").read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 2:
            continue
        en, tgt = p[0], p[1]
        if (en in STOP) or (not en.isalpha()) or len(en) < 3:
            continue
        et, tt = tok1(en), tok1(tgt)
        if et is None or tt is None or et == tt or (et, tt) in seen:
            continue
        if not (BAND_LO <= et < BAND_HI):
            continue
        seen.add((et, tt)); n += 1
    return n


MIN_PAIRS = 100    # この本数に満たない言語を落とす（下の図のとおり自然な崖の中に置く）
pair_count = {L: n_single_token_pairs(L) for L in CAND_LANGS}
ALL_LANGS = sorted([L for L in CAND_LANGS if pair_count[L] >= MIN_PAIRS], key=lambda L: -pair_count[L])
dropped = sorted([L for L in CAND_LANGS if pair_count[L] < MIN_PAIRS], key=lambda L: -pair_count[L])
print(f"selection funnel: {len(CAND_LANGS)} candidates -> {len(ALL_LANGS)} kept (single-token pairs >= {MIN_PAIRS}) / {len(dropped)} dropped")
print(f"  dropped (Qwen does not single-tokenize): " + ", ".join(f"{L}({pair_count[L]})" for L in dropped))


def plot_langsel():
    order = sorted(CAND_LANGS, key=lambda L: -pair_count[L])
    vals = [pair_count[L] for L in order]
    cols = ["#2a9d8f" if v >= MIN_PAIRS else "#e76f51" for v in vals]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.bar(range(len(order)), vals, color=cols)
    ax.axhline(MIN_PAIRS, color="black", ls="--", lw=1)
    ax.set_yscale("log"); ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=8.5)
    ax.set_ylabel("single-token translation pairs (log)")
    ax.set_title("Single-token translation pairs per MUSE language", fontsize=12, fontweight="bold")
    ax.legend(handles=[Line2D([0], [0], marker="s", color="w", markerfacecolor="#2a9d8f", ms=9, label=f"kept ({len(ALL_LANGS)})"),
                       Line2D([0], [0], marker="s", color="w", markerfacecolor="#e76f51", ms=9, label="dropped: " + ", ".join(dropped))],
              fontsize=9.5, loc="upper right")
    fig.tight_layout(); fig.savefig(outputs_dir / "mling_demo_langsel.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.show()


plot_langsel()

# %% [markdown]
# **図の内容**: 各棒は MUSE 1 言語の「バンド内・単一トークンの内容語対訳ペア数」（対数目盛）。緑＝採用（しきい値 100 以上）、
# 赤＝除外、黒破線＝しきい値 100。**観察される分布**: 候補 44 のうち 38 が緑（採用）・6 が赤（除外）。赤のうち fi(92)・lt(40) は
# しきい値のすぐ下、el(2)・hi(2)・ta(1)・bn(0) はほぼゼロで、この 4 言語とそれ以外の間に大きな段差がある。

# %% [markdown]
# **解釈**: ほぼゼロの el・hi・ta・bn は、今回の MUSE 訳語とフィルタ条件では、それらの文字体系（ギリシャ／デーヴァナーガリー／タミル／ベンガル）の語が
# Qwen で単一トークンになる例がほとんど得られないため対がほとんど作れない。**トークナイザの痕跡**と読める（≤2 と残りの間に大きな段差があり、この 4 言語の除外は
# しきい値の取り方に頑健）。一方 fi・lt は単一トークン化はできるがバンド内の対が薄く、しきい値 100 で追加的に落ちる
# （こちらの境界はしきい値に依存する softer なカット）。バンド内で数えるのは、続く言語類似度 $M$ を同じ指標で測るため。

# %% [markdown]
# ### 1.3 トークン頻度帯の選択
#
# 前のセクションでは、この帯の中で対訳ペアが十分に取れる 38 言語が選ばれた（図の緑の 38 言語。英語を加えて
# 39 言語になる）。対訳ペアを数えるときにも、英語トークンの id がこの中頻度帯 $2000 \le \mathrm{id} < 12000$ に
# あること、を既に条件として使っている。id はおおよそ頻度の順（小さいほど高頻度）で、最高頻度の語は綴りが
# 英語と衝突した外来語が多く紛らわしく、低頻度の語は雑音が多い。両端を外した中頻度が、意味のはっきりした語を
# きれいに映す。以降の概念選びと言語類似度もこの帯の中で行う。
#
# 2 言語 $i, j$ の近さは、両言語が共有する概念での訳語ベクトルの平均コサイン $m_{ij}$ で測る（正確な定義は 1.5）。
# この帯は結論に合わせて選んだ一点ではない。それを確かめるため、帯の両端を広く動かしながら、
# 全 $\binom{39}{2}=741$ 言語ペアの $m_{ij}$ のプロファイルがどう動くかを見る。

# %%
_uc = {}
def _uvec(t):
    v = _uc.get(t)
    if v is None:
        w = W_E[t].astype(np.float64); v = w / (np.linalg.norm(w) + 1e-12); _uc[t] = v
    return v

# バンド無しの概念プール（英語トークン id = 頻度 proxy を保持し、バンドは後で再フィルタ）
_pool = {}
for L in ALL_LANGS:
    _seen = set()
    for line in (MUSE_DIR / f"en-{L}.txt").read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 2:
            continue
        en, tgt = p[0], p[1]
        if (en in STOP) or (not en.isalpha()) or len(en) < 3:
            continue
        et, tt = tok1(en), tok1(tgt)
        if et is None or tt is None or et == tt or (et, tt) in _seen:
            continue
        _seen.add((et, tt))
        _pool.setdefault(en, {"tid": et, "tk": {}})["tk"].setdefault(L, tt)
for en, c in _pool.items():
    c["tk"]["en"] = c["tid"]
_pool = list(_pool.values())
_lang = ["en"] + [L for L in ALL_LANGS if L != "en"]; _ix = {L: i for i, L in enumerate(_lang)}
_ENZH = (0, _ix["zh"])   # en は index 0


def _pair_full(lo, hi):
    """バンド [lo,hi) での全言語ペアの平均 cos 辞書 {(i,j): m_ij}。"""
    acc = {}
    for c in _pool:
        if not (lo <= c["tid"] < hi):
            continue
        Ls = list(c["tk"])
        for a in range(len(Ls)):
            for b in range(a + 1, len(Ls)):
                x, y = Ls[a], Ls[b]
                if c["tk"][x] == c["tk"][y]:
                    continue
                k = (min(_ix[x], _ix[y]), max(_ix[x], _ix[y]))
                acc.setdefault(k, []).append(_uvec(c["tk"][x]) @ _uvec(c["tk"][y]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _sweep(fixed, vals, is_hi):
    """端の一方を固定し他方を vals で動かす。各ペアの生 cos と z-score の系列を返す（欠損は NaN）。"""
    dicts = [(_pair_full(fixed, s) if is_hi else _pair_full(s, fixed)) for s in vals]
    keys = sorted(set().union(*[set(d) for d in dicts]))
    raw = {k: np.array([d.get(k, np.nan) for d in dicts]) for k in keys}
    z = {k: [] for k in keys}
    for d in dicts:
        vv = np.array(list(d.values())); mu, sd = vv.mean(), vv.std()
        for k in keys:
            z[k].append((d[k] - mu) / sd if k in d else np.nan)
    z = {k: np.array(v) for k, v in z.items()}
    return raw, z, keys


_HIs = [4000, 6000, 8000, BAND_HI, 16000, 24000, 40000, 60000, 90000, 150000]   # LO=2000 固定
_LOs = [0, 1000, BAND_LO, 3000, 4000, 6000, 8000, 9000, 10000, 11000]           # HI=12000 固定
_rawH, _zH, _kH = _sweep(BAND_LO, _HIs, True)
_rawL, _zL, _kL = _sweep(BAND_HI, _LOs, False)


def _panel(ax, series, keys, xpos, xlabels, adopt_pos, ylabel, xlabel):
    span = max(xpos) - min(xpos)
    for k in keys:                                   # 全 741 ペアを薄線（欠損は gap）
        ax.plot(xpos, series[k], color="#cfcfcf", lw=0.5, alpha=0.5, zorder=1)
    order = sorted(keys, key=lambda k: -np.nanmean(series[k]))
    labeled = order[:4] if _ENZH in order[:4] else [_ENZH] + order[:3]
    pal = ["#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]; pi = 0
    for k in labeled:
        is_ez = (k == _ENZH); i, j = k; name = f"{_lang[i]}-{_lang[j]}"
        col = "#e41a1c" if is_ez else pal[pi % 4]; pi += 0 if is_ez else 1
        ax.plot(xpos, series[k], "-o", color=col, lw=2.6 if is_ez else 1.4,
                ms=5 if is_ez else 3, zorder=6 if is_ez else 3)
        nn = np.where(~np.isnan(series[k]))[0]
        if len(nn):
            ax.text(xpos[nn[-1]] + span * 0.012, series[k][nn[-1]], name, color=col, fontsize=8,
                    va="center", fontweight="bold" if is_ez else "normal")
    ax.axvline(adopt_pos, color="#1a1a1a", ls="--", lw=1.1, alpha=0.7)
    ax.set_xticks(xpos); ax.set_xticklabels(xlabels, fontsize=8, rotation=30)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.22)


_xH = [np.log10(x) for x in _HIs]; _lbH = [str(x) for x in _HIs]      # 右列: log
_xL = list(_LOs); _lbL = [str(x) for x in _LOs]                        # 左列: linear
fig, axs = plt.subplots(2, 2, figsize=(15, 9.5))
_panel(axs[0, 0], _rawL, _kL, _xL, _lbL, BAND_LO, "raw pair cosine  m_ij",
       f"lower edge LO on English token id  (HI={BAND_HI}; linear x; dashed=adopted)")
_panel(axs[0, 1], _rawH, _kH, _xH, _lbH, np.log10(BAND_HI), "raw pair cosine  m_ij",
       f"upper edge HI on English token id  (LO={BAND_LO}; log x; dashed=adopted)")
_panel(axs[1, 0], _zL, _kL, _xL, _lbL, BAND_LO, "z-score  (m_ij - mean)/std",
       f"lower edge LO on English token id  (HI={BAND_HI}; linear x)")
_panel(axs[1, 1], _zH, _kH, _xH, _lbH, np.log10(BAND_HI), "z-score  (m_ij - mean)/std",
       f"upper edge HI on English token id  (LO={BAND_LO}; log x)")
for ax in axs[1]:
    ax.axhline(0, color="gray", lw=0.6)
fig.suptitle("Is the band [2000, 12000) reasonable?  en-zh (red) is the top envelope over all 741 language pairs and stays so across the band\n"
             "left: lower edge LO swept (linear)   right: upper edge HI swept (log)   rows: raw cosine (top) / z-score (bottom)",
             fontsize=12, fontweight="bold", y=1.0)
fig.tight_layout(); fig.savefig(outputs_dir / "mling_demo_band_justification.png", dpi=150, bbox_inches="tight", facecolor="white"); plt.show()

# %% [markdown]
# **図の内容**: 灰色の細線は 39 言語（英語＋対象 38 言語）のすべての言語ペア $\binom{39}{2}=741$ 本それぞれのプロファイル。上段が生の平均コサイン $m_{ij}$、下段がそれを
# 全ペア分布で標準化した z-score $(m_{ij}-\mu)/\sigma$。左列は下端 LO を動かし（上端 12000 固定・linear 軸）、右列は
# 上端 HI を動かす（下端 2000 固定・log 軸）。赤＝en–zh、青・緑・紫＝次に高い 3 ペア、破線＝採用した 2000 / 12000。
# **観察される配置**: en–zh（赤）はどの帯でも最上位の包絡線で、生コサイン約 0.50、z-score 約 4。上端を広げると
# 生コサインはゆるく下がり、下端を上端へ近づけると帯の概念が枯渇して赤線は灰色の束へ沈む。採用帯 2000 / 12000 の
# 破線はこの平坦な頂上の内側にある。

# %% [markdown]
# **解釈**: en–zh は全ペアの中で約 4σ 突出した最強の結び付きであり、その順位も突出度も帯の端の取り方に依存しない。
# したがって $2000 \le \mathrm{id} < 12000$ は結論を作るために選んだ一点ではなく、広く頑健な範囲の内側の一点にすぎない。
# 一方で端を極端に振ると、上端では低頻度語による希釈で、下端では概念数の枯渇で崩れるので、中頻度帯に留めることには
# 客観的な理由がある。

# %% [markdown]
# ### 1.4 概念の選択手順
#
# 採用した各言語の辞書から、次の条件を満たす「概念」（英単語＋その各言語の訳）を集める。
#
# 1. 単純な stopword・文字種・文字数フィルタで**内容語候補**を選ぶ（`the` `and` などの機能語、記号、2 文字以下は除く）。
# 2. 英語も訳語も **Qwen で 1 トークン**になること（ベクトルが一意に定まるため）。
# 3. 英語トークンの id が **中頻度バンド 2000〜12000** にあること（この頻度帯を選ぶ理由は 1.3）。

# %%
concept = {}
for L in ALL_LANGS:
    seen = set()
    for line in (MUSE_DIR / f"en-{L}.txt").read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 2:
            continue
        en, tgt = p[0], p[1]
        if (en in STOP) or (not en.isalpha()) or len(en) < 3:
            continue
        et, tt = tok1(en), tok1(tgt)
        if et is None or tt is None or et == tt or (et, tt) in seen:   # et==tt: 同一トークン（同綴り・共有字・借用語を含む）を除く
            continue
        if not (BAND_LO <= et < BAND_HI):
            continue
        seen.add((et, tt))
        concept.setdefault(en, {"toks": {}})["toks"].setdefault(L, tt)
for en, c in concept.items():
    t = tok1(en)
    if t is not None:
        c["toks"]["en"] = t
print(f"concepts (en single-token, within band, {len(ALL_LANGS)} languages) = {len(concept)}")

# %% [markdown]
# ### 1.5 言語類似度行列 $M$ をつくって眺める
#
# **単語ベクトルの定義（以降ずっと使う）**。概念 $w$（＝英単語）の**言語 $L$ 訳トークン**を $t_L(w)$ と書き、その
# **長さ 1 に正規化した埋め込み**を単語ベクトルとする:
# $$ v_L(w) \;:=\; e\big(t_L(w)\big) \;=\; \frac{W_E[\,t_L(w)\,]}{\lVert W_E[\,t_L(w)\,]\rVert} \;\in\; \mathbb{R}^{2560}. $$
# 英語も 1 言語で、概念 $w$ の英語トークンは $t_\text{en}(w)$、その単語ベクトルは $v_\text{en}(w)=e(t_\text{en}(w))$。
# $W_E$ 全体はトークン埋め込み（部分語も含む）だが、本ノートでは 1 トークンになる語だけを選ぶので、そのベクトルを **単語ベクトル** と呼ぶ。
# この $v_L(w)$ が本ノートの主役で、
# Part 3 のモデル $v_L(w)=R(L)R(w)v_o$ の左辺もこれと同じ物である（回転を扱う Part 2.1 以降は 128 次元に落として測る。§3.2 で導線を示す）。
#
# 言語 $L, L'$ が訳を持つ概念の集合をそれぞれ $D_L, D_{L'}$ とする。2 言語の近さを、**両方に訳がある概念** $D_L \cap D_{L'}$ での
# 余弦類似度の平均で測る:
# $$ m_{LL'} = \operatorname{mean}_{\,w \,\in\, D_L \cap D_{L'},\ t_L(w)\neq t_{L'}(w)}\ \cos\!\big(v_L(w),\, v_{L'}(w)\big). $$
# $\cos$ で測るので $v_L(w)$ の正規化は値に影響しないが、後段の $v_L(w)$ と同じ量にそろえるため正規化版に統一する。
# $t_L(w)=t_{L'}(w)$（例: 日中で同じ漢字）は自明に $\cos = 1$ になる。これは同じトークンを共有しているだけで不正ではなく、本来は
# 含めてもよいが、CJK どうしの類似度を過度に押し上げないようここでは安全側で除いている。これを全ペアに並べた行列を
# $M=(m_{ij})$（$i,j$ は言語の番号。成分は上の $m_{LL'}$、対角は $m_{ii}=1$）とする。下では $M$ を階層クラスタリング（1.6）で得た順に
# 並べ替えて表示する。似た言語が隣に来るので塊が見やすい。

# %%
LANGS = ["en"] + ALL_LANGS
n = len(LANGS); idx = {L: i for i, L in enumerate(LANGS)}
acc = [[[] for _ in range(n)] for _ in range(n)]
for c in concept.values():
    Ls = list(c["toks"])
    for a in range(len(Ls)):
        for b in range(a + 1, len(Ls)):
            li, lj = Ls[a], Ls[b]                   # li, lj = 言語 i, 言語 j
            if c["toks"][li] == c["toks"][lj]:      # 同一トークン（例: 日中で同じ漢字）は cos=1 なので除外
                continue
            v = float(e(c["toks"][li]) @ e(c["toks"][lj]))
            acc[idx[li]][idx[lj]].append(v); acc[idx[lj]][idx[li]].append(v)

M = np.full((n, n), np.nan); Ncnt = np.zeros((n, n), int)   # M = (m_ij)
for i in range(n):
    M[i, i] = 1.0
    for j in range(n):
        if i != j and acc[i][j]:
            M[i, j] = float(np.mean(acc[i][j])); Ncnt[i, j] = len(acc[i][j])
off_n = Ncnt[~np.eye(n, dtype=bool)]
print(f"shared concepts per pair: min {off_n.min()} / median {int(np.median(off_n))} / max {off_n.max()}")
print(f"similarity of the strongest pair en-zh: m = {M[idx['en'], idx['zh']]:.3f}")

# 並べ替え用に階層クラスタリング（Ward 法）を先に計算しておく（図と説明は 1.6）。距離は sqrt(1 - m_ij)（単位ベクトルのユークリッド距離。Ward の前提に合わせる）。
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform
Dist = np.sqrt(np.clip(1.0 - M, 0, None))
np.fill_diagonal(Dist, 0.0)
Dist = np.nan_to_num(Dist, nan=float(np.nanmax(Dist)))
Dist = 0.5 * (Dist + Dist.T)
Zw = linkage(squareform(Dist, checks=False), method="ward", optimal_ordering=True)
order_w = list(leaves_list(Zw))

# 言語グループ凡例のハンドル（ヒートマップ・デンドロで共用）
_fams_present = [f for f in CANON_FAM if f != "Other" and f in {fam_of(L) for L in LANGS}]
_fam_handles = [Line2D([0], [0], marker="s", color="w", markerfacecolor=CANON_FAM[f], ms=9, label=f) for f in _fams_present]

# %%
_off = M[np.ix_(order_w, order_w)][~np.eye(n, dtype=bool)]; _off = _off[~np.isnan(_off)]
vmin, vmax = float(np.percentile(_off, 2)), float(np.nanmax(_off))


def plot_heatmap_ward():
    nm = [LANGS[i] for i in order_w]
    Mx = M[np.ix_(order_w, order_w)].copy(); np.fill_diagonal(Mx, np.nan)
    cmap = plt.cm.YlOrRd.copy(); cmap.set_bad("#eeeeee")
    fig, ax = plt.subplots(figsize=(12.5, 11)); im = ax.imshow(Mx, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(nm, rotation=90, fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(nm, fontsize=9)
    for k, L in enumerate(nm):
        ax.get_xticklabels()[k].set_color(fam_color(L)); ax.get_yticklabels()[k].set_color(fam_color(L))
        if L in ("en", "zh"):
            ax.get_xticklabels()[k].set_fontweight("bold"); ax.get_yticklabels()[k].set_fontweight("bold")
    ax.set_title("Language similarity matrix $M$ (Ward order)", fontsize=12.5, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="mean cosine")
    ax.legend(handles=_fam_handles, loc="upper left", bbox_to_anchor=(1.15, 1.0), fontsize=8.5, title="language group", frameon=False)
    fig.tight_layout(); fig.savefig(outputs_dir / "mling_demo_heatmap_ward.png", dpi=160, bbox_inches="tight", facecolor="white"); plt.show()


plot_heatmap_ward()

# %% [markdown]
# **図の内容**: 各セルは 2 言語間の平均コサイン類似度 $m_{ij}$（濃い赤ほど大、対角は自明なので灰色でマスク）。行・列は
# 1.6 の Ward クラスタリングの順（似た言語が隣）に並べ、ラベルを言語グループ色にした。
# **観察される配置**: ko-ja-en-zh が濃い赤のブロックをつくり、最強セルは en–zh（$m=0.500$）。その隣に
# Romance（it, es, pt, fr）の対角ブロックが続く。en の行・列は多くの言語グループにわたって赤みがある。

# %% [markdown]
# **解釈**: en が特定の言語グループだけでなく各言語グループに広く近く、英語は多くの言語に対して高い類似度を示す
# （この英語ピボット型データ上でハブ的な位置）と読める。CJK と en が最も濃いブロックを
# 作ることは、「英語は語族でなく CJK と束ねられる」を数値行列の側から見たものといえる。

# %% [markdown]
# ### 1.6 Ward 法による階層クラスタリングで語族を確かめる
#
# 上の行列 $M$ の並べ替えには**階層クラスタリング**（Ward 法）を使った。
# Ward 法には $d_{ij}=\sqrt{1 - m_{ij}}$ を用いる。この距離は、同じ概念集合上で平均を取る場合にはユークリッド距離として表現できるが、本実験では共通概念の集合が言語ペアごとに異なるため、厳密な保証はない。
# Ward 法はユークリッド距離を前提に分散を最小化する手法なので、これに合わせておく（$1 - m_{ij}$ を直接使っても、下記の結論は変わらない）。
# 近い言語から順にまとめていく手続きで、結果を樹形図（デンドログラム）にすると「どの言語が早く一つの塊になるか」が見える。
# 葉ラベルの色は言語グループ。注目は **英語が自分の Germanic 群（de, nl, sv, da, no, af）から離れ、CJK（zh, ja, ko）の塊に入る**こと。

# %%
def plot_dendro_ward():
    from collections import Counter
    # リンク（枝）の色 = その枝の配下の葉で「多数派の言語グループ」の色（例: CJK が多数なら赤）。
    # 言語グループをまたぐ上位の幹（しきい値超）は中立の灰色にして、図を客観的に保つ（矢印等の主張は入れない）。
    node_leaffams = {i: [fam_of(LANGS[i])] for i in range(n)}
    for i, row in enumerate(Zw):
        node_leaffams[n + i] = node_leaffams[int(row[0])] + node_leaffams[int(row[1])]
    dist_of = {n + i: float(Zw[i, 2]) for i in range(len(Zw))}
    thresh = 0.86 * Zw[:, 2].max()

    def link_color(node_id):
        if dist_of[node_id] > thresh:
            return "#bcbcbc"
        fam = Counter(node_leaffams[node_id]).most_common(1)[0][0]
        return CANON_FAM.get(fam, "#999999")

    fig, ax = plt.subplots(figsize=(16.5, 5.2))
    dendrogram(Zw, labels=LANGS, ax=ax, leaf_rotation=0, leaf_font_size=12, link_color_func=link_color)
    for c in ax.collections:
        c.set_linewidth(2.4)
    for lbl in ax.get_xmajorticklabels():
        L = lbl.get_text(); lbl.set_color(fam_color(L)); lbl.set_fontweight("bold")
    ax.set_ylabel("Ward linkage distance", fontsize=14)
    ax.set_ylim(Zw[:, 2].min() - 0.05, Zw[:, 2].max() * 1.02); ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=_fam_handles, loc="lower center", bbox_to_anchor=(0.5, 1.004), ncol=13,
              frameon=False, prop={"size": 10.5, "weight": "bold"}, handletextpad=0.3, columnspacing=0.9)
    fig.suptitle("Hierarchical clustering of languages (Ward)",
                 fontsize=15, fontweight="bold", y=1.035)
    fig.tight_layout(); fig.savefig(outputs_dir / "mling_demo_dendro_ward.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.show()


plot_dendro_ward()
leaf = [LANGS[i] for i in order_w]
ei = leaf.index("en")
print("leaf order:", " ".join(leaf))
print(f"neighbors of en: {leaf[max(0,ei-2):ei]} [en] {leaf[ei+1:ei+3]}")

# %% [markdown]
# **図の内容**: 葉＝言語（言語グループ色）、縦軸＝Ward 連結距離（低いほど早く 1 つの塊になる＝近い）。枝の色はその枝配下で多数派の
# 言語グループ、言語グループをまたぐ上位の幹は灰色。**観察される配置**: en の隣は zh で、en は **CJK（ko, ja, zh）の塊の中**に入る。本来の
# Germanic 群（右方の de, nl, af, sv, da, no）からは離れている。Romance・Slavic などは概ね語族どおりまとまる。

# %% [markdown]
# **解釈**: en が語族（Germanic）でなく CJK と早く 1 つになるのは、言語の系統では説明できない現象で、
# 冒頭で述べたとおり学習データの言語構成を反映したものと推測される。周辺の低リソース言語は共有概念が少なく
# 数値が不安定なので、細かい並びは深追いしない。

# %% [markdown]
# ### 1.7 どの言語がハブか
#
# 「他の言語すべてへの平均余弦類似度」を中心性とすると、辞書の小さい言語が上位に来る**選択バイアス**が出る
# （小辞書の言語は易しい高頻度概念に偏り、値がかさ上げされる）。そこで各言語の**最も近い上位 $k$ 言語だけ**の平均
# （上位 k 近傍類似度（top-k 近傍類似度））を見る。こちらは遠い言語を足しても変わりにくく、言語の選び方に左右されにくい。

# %%
def overall_cent(L):
    return float(np.nanmean([M[idx[L], idx[K]] for K in LANGS if K != L]))


def topk_cent(L, k):
    row = np.array([M[idx[L], j] for j in range(n) if j != idx[L] and not np.isnan(M[idx[L], j])])
    return float(np.mean(np.sort(row)[::-1][:k]))


_ov = sorted(LANGS, key=lambda L: -overall_cent(L))
print("[reference] overall mean similarity (selection bias; not adopted) top5: " + "  >  ".join(f"{L} {overall_cent(L):.3f}" for L in _ov[:5]))
print()
print("top-k nearest-neighbor similarity ranking (robust metric):")
for k in [1, 2, 3, 4, 5]:
    o = sorted(LANGS, key=lambda L: -topk_cent(L, k))
    print(f"  top-{k}: " + "  >  ".join(f"{L} {topk_cent(L, k):.3f}" for L in o[:6]) + f"    (en is rank {o.index('en')+1})")

# %% [markdown]
# **結論**: 上位 k 近傍類似度（top-k 近傍類似度）では $k=1$ から $5$ まで **英語が一貫して 1 位**（$k=1$ では英語と中国語が 0.500 で同点最強）。
# 続いて中国語・韓国語・日本語（CJK）とスペイン語・ポルトガル語（Romance）が並ぶ。
# **欧州言語を多数足しても英語–中国語が top-k で検討した k=1..5 の範囲で最も高い**。この近さはこの 38 言語集合において $k=1..5$ の範囲で安定して最も高く、
# 学習データ構成と整合的（ただし英語ピボット辞書と選択基準の影響は分離できていない）。
# 一方、全体平均類似度は小辞書の言語（例: ko, bg）を上位に押し上げる選択バイアスがあり、採用しない。

# %% [markdown]
# ## Part 2　言語を選び、生の埋め込みを見る
#
# ここからは「多言語の対応が埋め込みの中でどう表されているか」を見る。まず**回転に使う言語を選び**、
# 次に**生の埋め込みを t-SNE で可視化**する（Part 3 の変換後と対をなす中心的な図）。回転の理論と手続きの詳細は Part 3（理論編）にまとめる。

# %% [markdown]
# ### 2.1 回転に使う言語の選択
#
# あとで各言語ごとに「英語側から その言語側への回転 $R(L)$」を対訳ペアから推定する（詳細は Part 3）。
# ただし、推定に使える対訳ペアが少ない言語は、**推定に使った対にはよく合うのに、使っていない対には合わない**（過学習）。
# そこで各言語で対訳ペアを **train / valid に分けて当てはまりを比べ**、その差 **gap = train − valid** が小さい
# （＝過学習が小さい）言語だけを回転に使う。分割は、各言語の対訳ペアを **valid 3 割（最低 5 対）／ train 7 割**に
# ランダム分割し、**$R(L)$ は train だけで推定**して、train と held-out の valid の両方で整列度（コサイン）を測る、というもの。
# 乱数の種を変えて **5 回**繰り返した平均を tr / va とし、対訳ペアが **15 対未満**の言語は安定に分割できないので除外する。
# ここでは重い計算（128 次元への削減と $R(L)$ の推定）を先に済ませて
# 言語を選ぶだけにする。**何をしているかの中身は Part 3 で説明する。**

# %%
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

ALIGN_THR = 0.25       # 対訳品質フィルタ（英語とその訳語の cos がこれ以上）: 回転推定を安定させる
K = 128              # 回転を求める作業空間の次元（PCA で削減。根拠は Part 3）
GAP_CUT = 0.30       # gap がこれ以下の言語を回転に使う
M_MIN = 3            # 表示・指標には非英語 3 言語以上に訳がある概念を使う
VALID_FRAC = 0.30    # train/valid 分割の valid 割合（2.1 の言語選択）
N_SPLIT = 5          # 分割をやり直して平均する回数（2.1）
MIN_VALID = 5        # valid の最低対数（2.1）
MIN_FIT = 15         # これ未満の対数の言語は分割せず除外（2.1）

# 回転推定用の対訳ペア（英語とよく揃う訳 = filter5）。Part 1 で作った concept（band 内・単一トークン）を再利用。
fitpairs = {}
for L in ALL_LANGS:
    fitpairs[L] = [(c["toks"]["en"], c["toks"][L]) for c in concept.values()
                   if "en" in c["toks"] and L in c["toks"] and c["toks"]["en"] != c["toks"][L]
                   and float(e(c["toks"]["en"]) @ e(c["toks"][L])) >= ALIGN_THR]

# 作業空間 PCA-128（概念に出る全トークン）。以降この空間で回転を測る（詳細は Part 3）。
_allids = sorted({c["toks"][L] for c in concept.values() for L in c["toks"]})
pca = PCA(n_components=K, random_state=0).fit(np.array([e(t) for t in _allids]))
_pc = {}


def ep(t):
    """トークン t を 128 次元へ写して長さ 1 に正規化。"""
    if t not in _pc:
        _pc[t] = u(pca.transform(e(t)[None])[0])
    return _pc[t]


def _align(pairs, R):
    return float(np.mean([float(u(ep(a) @ R) @ ep(b)) for a, b in pairs])) if pairs else float("nan")


# 各言語で fit ペアを train/valid に分け（5 回平均）、R(L) の当てはまり gap を測る
gap_rows = []
for L in ALL_LANGS:
    pr = fitpairs[L]; nfit = len(pr)
    if nfit < MIN_FIT:
        gap_rows.append((L, nfit, np.nan, np.nan, np.nan)); continue
    trs, vas = [], []
    for seed in range(N_SPLIT):
        rng = np.random.default_rng(seed)
        ii = rng.permutation(nfit); nv = max(MIN_VALID, int(nfit * VALID_FRAC))
        vi, ti = ii[:nv], ii[nv:]
        A = np.array([ep(pr[i][0]) for i in ti]); B = np.array([ep(pr[i][1]) for i in ti])
        Rr, _ = orthogonal_procrustes(A, B)
        trs.append(_align([pr[i] for i in ti], Rr)); vas.append(_align([pr[i] for i in vi], Rr))
    tr, va = float(np.mean(trs)), float(np.mean(vas))
    gap_rows.append((L, nfit, tr, va, tr - va))

_gapd = {r[0]: r[4] for r in gap_rows}
LANGS_ROT = [r[0] for r in gap_rows if not np.isnan(r[4]) and r[4] <= GAP_CUT]
LANGS_ROT = sorted(LANGS_ROT, key=lambda L: _gapd[L])
print(f"languages used for the rotation (gap <= {GAP_CUT}): {len(LANGS_ROT)} languages (adding English, rotation over {len(LANGS_ROT)+1} languages)")
for L in LANGS_ROT:
    print(f"  {L:<4} gap={_gapd[L]:+.3f}  fit={dict((r[0], r[1]) for r in gap_rows)[L]:>4} pairs  ({fam_of(L)})")


def plot_overfit_lines():
    rows = sorted([r for r in gap_rows if not np.isnan(r[4])], key=lambda r: r[4])
    labels = [r[0] for r in rows]; train = [r[2] for r in rows]; valid = [r[3] for r in rows]
    xs = list(range(len(rows))); n_keep = sum(1 for r in rows if r[4] <= GAP_CUT); cut_x = n_keep - 0.5
    fig, ax = plt.subplots(figsize=(14.5, 6.0)); ax.set_axisbelow(True); ax.grid(axis="y", color="0.90", lw=0.7)
    ax.axvspan(cut_x, len(rows) - 0.4, color="#e76f51", alpha=0.05); ax.axvline(cut_x, color="0.35", ls="--", lw=1.2)
    ax.plot(xs, train, color="#264653", lw=1.7, marker="o", ms=6, label="train  (pairs used to fit R(L))")
    ax.plot(xs, valid, color="#e76f51", lw=1.7, marker="X", ms=7, label="valid  (held-out pairs)")
    ax.text(cut_x - 0.4, 0.145, f"kept: gap <= {GAP_CUT}  ({n_keep} languages)", ha="right", va="center", fontsize=11, color="0.25")
    ax.text(cut_x + 0.4, 0.145, "dropped (overfit)", ha="left", va="center", fontsize=11, color="#c1553b")
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=90, fontsize=10.5)
    for k, L in enumerate(labels):
        ax.get_xticklabels()[k].set_color(fam_color(L))
    ax.set_ylabel("R(L) alignment  cos", fontsize=12)
    ax.set_xlabel("language  (sorted by overfitting; label color = language group)", fontsize=11.5)
    ax.set_title("Choosing rotation languages: $R(L)$ fit vs held-out", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylim(0.10, 0.97); ax.set_xlim(-0.7, len(rows) - 0.4); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.9, edgecolor="0.8")
    fig.tight_layout(); fig.savefig(outputs_dir / "mling_demo_overfit.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.show()


plot_overfit_lines()

# %% [markdown]
# **図の内容**: 各言語について、回転 $R(L)$ の当てはまり（コサイン）を、推定に使った対（train, ●）と使っていない対
# （valid, ✕）で示す。言語は 2 本の線の縦の差＝**gap**（過学習量）が小さい順に並べ、破線より左（gap $\le 0.30$）が採用、
# 右（薄赤）が除外。**観察される配置**: 左では train と valid がほぼ重なり、右へ行くほど valid が下がって gap が開く。
# gap $\le 0.30$ を満たすのは 14 言語。

# %% [markdown]
# **解釈**: valid が落ちる（train にだけ合う）のは、対訳ペアが少ない言語で 128 次元の $R(L)$ が過学習しているためと解釈できる。
# そこで train–valid gap が比較的小さい（過学習の小さい）14 言語だけを回転に使う。なぜ過学習するのか・$R(L)$ をどう推定するのかは Part 3 で説明する。

# %% [markdown]
# **何が選ばれたか**: この基準（gap $\le 0.30$）を満たしたのは **14 言語**で、これに英語を加えた **15 言語**を回転に使う。
# 選ばれた 14 言語は次のとおり（上の print に各言語の gap も出る）:
#
# - **CJK**: 中国語(zh)・日本語(ja)・韓国語(ko)
# - **Romance**: スペイン語(es)・ポルトガル語(pt)・フランス語(fr)・イタリア語(it)
# - **Slavic**: ロシア語(ru)・ブルガリア語(bg)
# - **Semitic**: アラビア語(ar)・ヘブライ語(he)
# - **Germanic**: ドイツ語(de) ／ **Turkic**: トルコ語(tr) ／ **Viet**: ベトナム語(vi)
#
# これらは主要な言語グループを広くカバーしている。gap が最も小さいのは**中国語**（train と valid の当てはまりがほぼ同じ）で、
# 一般に**対訳ペアが多い言語ほど gap が小さい**。逆に、Part 1 では 38 言語をクラスタに使えたのに回転は 14 言語に減るのは、
# 「言語どうしの近さ（クラスタ）」は各ペアの平均 cos だけで測れるのに対し、「回転 $R(L)$」は言語ごとに 128 次元の変換を
# 対訳ペアから**推定**する必要があり、ペアが少ない言語では過学習してしまうからである（これが Part 1 と Part 2 で使える言語数が違う理由）。

# %% [markdown]
# ### 2.2 en→L の「転移性能」は生の近さと少しズレる
#
# 概念先行モデル（Part 3）では、英語を基準にすると $v_L(w)=R(L)\,v_\text{en}(w)$ となる。すなわち en から言語 $L$ への転移は
# **一つの回転 $R(L)$** そのものである。だから「en→L の転移がどれだけ効くか」は、上で言語選択に使った
# **held-out 整列 valid**（未知の概念で $R(L)v_\text{en}(w)$ が $v_L(w)$ にどれだけ合うか）がそのまま指標になる。
# これを Part 1.5 の**生の類似** $m_{en,L}$（回転をかけない M 行）と並べ、両者がどれだけ一致するかを見る。

# %%
# 転移性能（held-out 整列 valid, 上の gap_rows）を、生の en 類似 m_{en,L}（1.5 の M 行）と比較。
from scipy.stats import pearsonr, spearmanr
va_d = {r[0]: r[3] for r in gap_rows}                       # held-out 整列 valid = en→L 転移スコア
_xs = [float(M[idx["en"], idx[L]]) for L in LANGS_ROT]      # 生類似 m_{en,L}（raw 2560 次元）
_ys = [va_d[L] for L in LANGS_ROT]                          # 転移 valid（PCA-128 held-out）
pr_r = float(pearsonr(_xs, _ys)[0]); sp_r = float(spearmanr(_xs, _ys)[0])  # type: ignore[arg-type]


def plot_transfer_vs_raw():
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for L in LANGS_ROT:
        x, y = float(M[idx["en"], idx[L]]), va_d[L]
        ax.scatter(x, y, s=115, color=fam_color(L), edgecolors="black", linewidths=0.8, zorder=3)
        ax.annotate(L, (x, y), fontsize=11, fontweight="bold", ha="center", va="center",
                    xytext=(0, 11), textcoords="offset points", color=fam_color(L))
    ax.set_xlabel("raw cosine to English  $m_{en,L}$  (Part 1.5 M-matrix row)", fontsize=12)
    ax.set_ylabel("held-out valid alignment  (en→L transfer performance)", fontsize=12)
    ax.set_title(f"en→L transfer performance vs raw similarity   (Pearson r={pr_r:.2f}, Spearman ρ={sp_r:.2f})",
                 fontsize=12, fontweight="bold")
    _fams = [f for f in CANON_FAM if f != "Other" and f in {fam_of(L) for L in LANGS_ROT}]
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=CANON_FAM[f], ms=10, label=f) for f in _fams],
              fontsize=9.5, loc="best", title="language group")
    ax.grid(alpha=0.25); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(outputs_dir / "mling_demo_transfer_vs_raw.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.show()


plot_transfer_vs_raw()
print("en->L transfer score (valid) ranking (descending): " + "  >  ".join(f"{L} {va_d[L]:.2f}" for L in sorted(LANGS_ROT, key=lambda L: -va_d[L])))
print(f"correlation valid vs m_en:  Pearson r={pr_r:.3f}   Spearman rho={sp_r:.3f}")

# %% [markdown]
# **図の内容**: 各点が回転言語 1 つ。横軸＝英語との生のコサイン $m_{en,L}$（Part 1.5 の M 行・回転なし・2560 次元）、
# 縦軸＝held-out 整列 valid（$R(L)$ の推定に使っていない対で測った転移スコア・PCA-128）。色は言語グループ。
# **観察される配置**: 全体に強い正の相関（Spearman $\rho\approx0.87$）だが完全ではなく、**上位で順位が入れ替わる**。
# **中国語(zh) は生では英語に最も近い**（$m_{en,L}$ 最大）が転移では 3 位。**スペイン語・ポルトガル語(Romance) は生では
# それほど近くない**のに転移では 1・2 位に立つ。
#
# **解釈**: 「生の近さ」と「回転で転移できるか」は別物である。Part 1 の英語–中国語の近さは*生の配置*の話で、
# 中国語トークンが（学習後の埋め込み空間で）英語の近くに置かれていることを反映する。一方 Romance が転移で上位に来るのは、
# 英語の配置を**回転させると Romance になりやすく**、en→Romance の対応が単一回転でよく表せる、と読める
# （「意味が近い」ではなく「写像が回転的」）。
#
# **数値の読み方（注意）**: valid（held-out 整列）の**絶対値**は作業空間（PCA-128）と対の選別（$\ge$ ALIGN_THR）に依存するので、
# 絶対水準は解釈しない。読むのは**言語間の順位**と**生類似からのズレ**だけ。辞書が en-X（英語ピボット）なので
# 「英語からの転移」は構造的に有利なので、値は言語間の相対比較にのみ用いる。対象は gap $\le 0.30$ の 14 言語に限る。

# %% [markdown]
# ### 2.3 生の埋め込みを見る
#
# 選んだ言語＋英語について、**生の埋め込み**（変換を一切かけないベクトル）を 2 次元へ落として（t-SNE）眺める。
# 各概念は、英語トークンを中心に各言語の訳語を線でつないだ「星」の形に描く（枝＝英語→訳語）。
# 何が近くに来るか（同じ意味か・同じ言語か）を目で確かめるのがねらい。
#
# 表示する概念の数は下の **`N_DISPLAY`** で自由に変えられる（多いと賑やか・読みにくく、少ないとすっきり）。
# **`SEED`** を変えると、表示する概念の組が別の乱数で選び直され、別の絵になる。

# %%
from sklearn.manifold import TSNE

# 言語ごとの色 = 統一カラーコード（CANON_FAM の言語グループ色）。英語＝黒（ハブ）。
# 同じ言語グループの言語は同じ色（例: CJK の zh/ja/ko は赤）。個々の言語は点に付いた訳語ラベルで区別する。
LANG_COLOR = {L: fam_color(L) for L in LANGS_ROT}
LANG_COLOR["en"] = "#1a1a1a"  # 黒（(0.1, 0.1, 0.1) 相当のハブ色）

# 表示・指標に使える概念プール（回転言語, band, 単一トークン, 翻訳品質フィルタなし = 図で「近い語だけ選ぶ」チートを避ける）
cc = {}
for L in LANGS_ROT:
    seen = set()
    for line in (MUSE_DIR / f"en-{L}.txt").read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) < 2:
            continue
        en, tgt = p[0], p[1]
        if (en in STOP) or (not en.isalpha()) or len(en) < 3:
            continue
        et, tt = tok1(en), tok1(tgt)
        if et is None or tt is None or et == tt or not (BAND_LO <= et < BAND_HI) or (et, tt) in seen:
            continue
        seen.add((et, tt))
        cc.setdefault(en, {"en": et, "en_word": en, "langs": {}})["langs"].setdefault(L, (tt, tgt))
draw = [c for c in cc.values() if len(c["langs"]) >= M_MIN and len(set(c["langs"][L][0] for L in c["langs"])) == len(c["langs"])]
draw_en = {c["en_word"] for c in draw}
print(f"concept pool (among the {len(LANGS_ROT)} rotation languages, concepts with translations in >= {M_MIN} non-English languages): {len(draw)} concepts  <- N_DISPLAY of these are selected for display")

# %%
# 表示する概念を乱数で選ぶ（このセルで選択を確定。描画は次のセル。seed 固定なので再現できる）
SEED = 0            # 表示概念のランダム選択の種（変えると別の概念が選ばれ別の絵になる）
N_DISPLAY = 48      # t-SNE に表示する概念の数
rng = np.random.default_rng(SEED)
sub = [draw[i] for i in sorted(rng.choice(len(draw), size=min(N_DISPLAY, len(draw)), replace=False))]

from collections import Counter
print(f"concepts displayed in the t-SNE: {len(sub)} (drawn from a pool of {len(draw)} with SEED={SEED})")
print("  distribution of linked language counts: " + ", ".join(f"{k} languages={v} concepts" for k, v in sorted(Counter(len(c['langs']) for c in sub).items(), reverse=True)))
print("  concepts linked to many languages (many star branches) top 6:")
for c in sorted(sub, key=lambda c: -len(c["langs"]))[:6]:
    print(f"    {c['en_word']:<12}({len(c['langs'])} languages): " + ", ".join(f"{L}={c['langs'][L][1]}" for L in c["langs"]))

# %% [markdown]
# **上の print の読み方**: 選ばれた概念それぞれに、回転で使う言語のうち何言語の訳語が紐づくか（＝星の枝の数）を示した。
# 枝が多い概念ほど下の図で中心から多くの言語へ伸びる「大きな星」になる。どれが選ばれるかは `SEED` の乱数で決まるので、
# 具体的にどの概念かはこの print で確認する（`SEED` を変えれば別の概念が選ばれる）。次のセルでこの概念集合を t-SNE で描く。

# %%
# t-SNE 描画（上で選んだ sub を描く。dpi=300 の高解像度で保存）
def plot_star_raw():
    spts = []
    for ci, c in enumerate(sub):
        spts.append((ci, "en", c["en"], c["en_word"]))
        for L in c["langs"]:
            spts.append((ci, L, c["langs"][L][0], c["langs"][L][1]))
    uniq = sorted({p[2] for p in spts}); U = np.array([e(t) for t in uniq]); id2r = {t: r for r, t in enumerate(uniq)}
    XY = TSNE(n_components=2, metric="cosine", perplexity=min(30, max(5, (len(uniq) - 1) // 3)), random_state=0, init="pca").fit_transform(U)
    P = np.array([XY[id2r[p[2]]] for p in spts]); byc = {}
    for k, p in enumerate(spts):
        byc.setdefault(p[0], {})[p[1]] = k
    fig, ax = plt.subplots(figsize=(14, 12))
    for ci, dd in byc.items():
        for L, k in dd.items():
            if L != "en" and "en" in dd:
                ax.plot([P[dd["en"], 0], P[k, 0]], [P[dd["en"], 1], P[k, 1]], color="gray", lw=0.35, alpha=0.22, zorder=1)
    for L in ["en"] + LANGS_ROT:
        ks = [k for k in range(len(spts)) if spts[k][1] == L]
        ax.scatter(P[ks, 0], P[ks, 1], s=38, color=[LANG_COLOR[L]], alpha=0.75, edgecolors="none", zorder=(3 if L == "en" else 2))
    for k in range(len(spts)):
        ax.text(P[k, 0], P[k, 1], _fix_rtl(spts[k][3]), fontsize=6.5, color=LANG_COLOR[spts[k][1]], alpha=0.95, zorder=4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Raw token embeddings (English-hub star)", fontsize=13, fontweight="bold")
    _famset = [f for f in CANON_FAM if f != "Other" and f in {fam_of(L) for L in LANGS_ROT}]
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=(0.1, 0.1, 0.1, 1.0), ms=9, label="en (hub)")]
              + [Line2D([0], [0], marker="o", color="w", markerfacecolor=CANON_FAM[f], ms=9, label=f) for f in _famset],
              fontsize=9.5, loc="best", title="color = language group")
    fig.tight_layout(); fig.savefig(outputs_dir / "mling_demo_raw_tsne.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.show()


plot_star_raw()

# %% [markdown]
# **図の内容**: 各点は、表示概念×言語の訳語トークンの生の埋め込みベクトルを、コサイン距離の t-SNE で 2 次元化したもの。
# 英語（黒＝ハブ）を中心に、同じ概念の各言語訳を灰色の細い線でつないである（星の枝＝英語→訳語）。点の色は言語グループ
# （CANON_FAM）、各点の訳語ラベルで個々の語が読める。**観察される配置**: 同じ意味の訳語どうし（＝同じ星に属する
# 別言語の点）が互いに近くに来る一方、同じ色（同一言語グループ）だけが集まった領域は見当たらず、色は各星の中で混ざる。
# 概念によっては枝が長く伸びる。

# %% [markdown]
# **解釈**: 近さを決めているのは言語よりも意味であり、生の埋め込みは「**概念でまとまる**」（概念優位の表現）と解釈できる。
# 長く伸びる枝は、多義語など英語と意味がずれた訳語が効いた例と解釈できる。この概念が支配的な生の埋め込みを、Part 3 では
# 逆概念変換 $C_L^{-1}(w)$ で概念成分を抑えると、隠れていた言語構造が顕在化する（同じ言語でまとまる）ことを見る。

# %% [markdown]
# ## Part 3　多言語の対応は「回転」
#
# Part 2 で見た「生の空間は概念でまとまる」を、ここでは**回転**という言葉で理解し、
# **各言語・各概念に対応する**逆概念変換 $C_L^{-1}(w)$ を適用すると、概念依存の類似性が下がり、同一言語内の類似性が相対的に上がる
# （概念成分を抑えて言語構造を顕在化する）ことを示す。手続きの詳細もここで説明する。
#
# ★用語の注意: 本ノートでは（冒頭の「回転」も含め）分かりやすさのため一貫して「**回転**」「**回転行列**」と呼ぶが、厳密には鏡映も含む
# **直交変換**・**直交行列**のことである（Procrustes で求まる $R$ も直交行列で、行列式は $\pm 1$）。

# %% [markdown]
# ### 3.0 概念成分を抑えると言語構造が顕在化する
#
# 生のトークン埋め込みでは、同じ意味を表す対訳語が近く、概念構造が支配的に見える。共役型の逆概念変換
# $C_L^{-1}(w)=R(L)R(w)^{-1}R(L)^{-1}$ は、言語 $L$ の座標系で概念 $w$ の効果を打ち消すことを意図した変換である。
# その結果、同じ言語の語が相対的に近くなり、もとの埋め込みに含まれていた言語構造が見えやすくなる。
#
# 下の図は、この読み方を模式的に表したものである。左は生の埋め込み、中央は共役型の逆概念変換、右は変換後に同一言語の基準位置
# $\hat{x}_L=R(L)v_o$ が見える、という流れを表す。

# %%
def plot_concept_language_schematic():
    # 横長・上下を締めたアスペクトで、文字/線/マーカーを大きめに。縮小表示でも読める密度に。
    fig = plt.figure(figsize=(12.8, 4.9))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    panels = {
        "raw": (0.020, 0.105, 0.290, 0.700),
        "transform": (0.385, 0.105, 0.230, 0.700),
        "after": (0.690, 0.105, 0.290, 0.700),
    }
    lang_colors = {"en": "#333333", "zh": "#d62728", "fr": "#ff7f0e", "ja": "#2ca02c"}
    concept_markers = {"water": "o", "king": "s", "city": "^"}

    def to_ax(panel, x, y):
        x0, y0, w, h = panel
        return x0 + x * w, y0 + y * h

    def blob(cx, cy, rw, rh, face, edge, alpha=0.16, lw=2.0):
        ax.add_patch(Ellipse((cx, cy), rw, rh, facecolor=face, edgecolor="none",
                             alpha=alpha, zorder=1))
        ax.add_patch(Ellipse((cx, cy), rw, rh, facecolor="none", edgecolor=edge,
                             alpha=0.6, linewidth=lw, zorder=2))

    def draw_panel(key, title, subtitle, face):
        x0, y0, w, h = panels[key]
        ax.add_patch(FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.014,rounding_size=0.022",
            linewidth=1.6, edgecolor="#c4ccd4", facecolor=face, zorder=0
        ))
        ax.text(x0 + w / 2, y0 + h + 0.155, title, fontsize=20, weight="bold",
                color="#1f2328", ha="center")
        ax.text(x0 + w / 2, y0 + h + 0.070, subtitle, fontsize=13.5, weight="bold",
                color="#5b6570", ha="center")

    draw_panel("raw", "Raw token embeddings",
               "grouped by concept — languages mixed", "#fbfcff")
    draw_panel("transform", "Inverse concept transform",
               r"$C_L^{-1}(w)=R(L)\,R(w)^{-1}\,R(L)^{-1}$", "#fffdf7")
    draw_panel("after", "After the transform",
               "grouped by language — concepts collapsed", "#fbfffb")

    # ── 左パネル: 概念ごとの blob（同一マーカー形＝概念, 4 色＝言語） ───────────
    concept_centers = {"water": (0.32, 0.80), "king": (0.68, 0.50), "city": (0.32, 0.20)}
    lang_arrange = {"en": (-0.072, 0.070), "zh": (0.072, 0.070),
                    "fr": (-0.072, -0.070), "ja": (0.072, -0.070)}
    for concept, (cx, cy) in concept_centers.items():
        bx, by = to_ax(panels["raw"], cx, cy)
        blob(bx, by, 0.115, 0.215, face="#8c959f", edge="#8c959f")
        for lang, (dx, dy) in lang_arrange.items():
            x, y = to_ax(panels["raw"], cx + dx, cy + dy)
            ax.scatter(x, y, s=200, marker=concept_markers[concept],
                       color=lang_colors[lang], edgecolor="white", linewidth=1.7, zorder=4)
        ax.text(bx, by - 0.140, concept, ha="center", fontsize=15,
                color="#3a4048", weight="bold", zorder=5)

    # ── 中央パネル: 3 段変換（番号は上下中央揃え） ──────────────────────────
    tx0, ty0, tw, th = panels["transform"]
    steps = [
        ("1", r"$R(L)^{-1}$", "align to English"),
        ("2", r"$R(w)^{-1}$", "undo concept"),
        ("3", r"$R(L)$", "back to language $L$"),
    ]
    bh = 0.155
    for i, (num, label, desc) in enumerate(steps):
        cyl = 0.80 - i * 0.30
        _, cy = to_ax(panels["transform"], 0.0, cyl)
        bx = tx0 + 0.030
        bw = tw - 0.060
        ax.add_patch(FancyBboxPatch((bx, cy - bh / 2), bw, bh,
                                    boxstyle="round,pad=0.010,rounding_size=0.018",
                                    linewidth=1.5, edgecolor="#cbd2d9",
                                    facecolor="#ffffff", zorder=3))
        ax.add_patch(Circle((bx + 0.034, cy), 0.023, facecolor="#0969da",
                            edgecolor="none", zorder=4))
        ax.text(bx + 0.034, cy, num, ha="center", va="center", fontsize=13,
                color="white", weight="bold", zorder=5)
        ax.text(bx + 0.078, cy + 0.026, label, va="center", fontsize=18.5,
                color="#1f2328", zorder=5)
        ax.text(bx + 0.078, cy - 0.031, desc, va="center", fontsize=12.5,
                color="#57606a", zorder=5)
        if i < 2:
            midx = bx + bw / 2
            top_next = cy - 0.30 * th + bh / 2
            ax.add_patch(FancyArrowPatch((midx, cy - bh / 2 - 0.004),
                                         (midx, top_next + 0.004),
                                         arrowstyle="-|>", mutation_scale=18,
                                         linewidth=2.2, color="#8c959f", zorder=3))

    # ── 右パネル: 言語ごとの blob（同一色＝言語, 3 概念が基準点 x̂_L の近くに集まる） ──
    # fr/ja は楕円・マーカー・ラベルごと下へ寄せる（上の楕円と離し、下マージンを活用）。
    lang_centers = {"en": (0.22, 0.72), "zh": (0.78, 0.72),
                    "fr": (0.22, 0.23), "ja": (0.78, 0.23)}
    ghost_dirs = {"water": (-0.82, 0.55), "king": (0.82, 0.55), "city": (0.0, -1.05)}
    for lang, (cx, cy) in lang_centers.items():
        col = lang_colors[lang]
        bx, by = to_ax(panels["after"], cx, cy)
        blob(bx, by, 0.124, 0.220, face=col, edge=col, alpha=0.13)
        rad_x, rad_y = 0.090, 0.086
        ax.scatter(bx, by, s=430, marker="+", color=col, linewidth=3.6, zorder=4)
        for concept, (ux, uy) in ghost_dirs.items():
            gx, gy = to_ax(panels["after"], cx + ux * rad_x, cy + uy * rad_y)
            ax.scatter(gx, gy, s=130, marker=concept_markers[concept],
                       facecolor="white", edgecolor=col, linewidth=2.1,
                       alpha=0.95, zorder=6)
        ax.text(bx, by + 0.150, rf"$\hat{{x}}_{{\mathrm{{{lang}}}}}$",
                ha="center", fontsize=17, color=col, weight="bold", zorder=6)

    # ── パネル間の大矢印（太く大きく） ─────────────────────────────────────
    def arrow_between(left_key, right_key):
        lx, ly, lw, lh = panels[left_key]
        rx, ry, rw, rh = panels[right_key]
        ymid = ly + lh * 0.50
        ax.add_patch(FancyArrowPatch((lx + lw + 0.012, ymid), (rx - 0.012, ymid),
                                     arrowstyle="-|>", mutation_scale=42,
                                     linewidth=4.4, color="#57606a", zorder=5))

    arrow_between("raw", "transform")
    arrow_between("transform", "after")

    # ── 凡例 ───────────────────────────────────────────────────────────────
    legend_y = 0.040
    ax.text(0.068, legend_y, "color = language", fontsize=14, color="#3a4048",
            va="center", weight="bold")
    for i, (lang, color) in enumerate(lang_colors.items()):
        x = 0.238 + i * 0.058
        ax.scatter(x, legend_y, s=150, color=color, edgecolor="white", linewidth=1.2)
        ax.text(x + 0.016, legend_y, lang, fontsize=13, color="#57606a", va="center")
    ax.text(0.560, legend_y, "marker = concept", fontsize=14, color="#3a4048",
            va="center", weight="bold")
    for i, (concept, marker) in enumerate(concept_markers.items()):
        x = 0.730 + i * 0.080
        ax.scatter(x, legend_y, s=150, marker=marker, color="#6e7781",
                   edgecolor="white", linewidth=1.2)
        ax.text(x + 0.016, legend_y, concept, fontsize=13, color="#57606a", va="center")

    fig.savefig(outputs_dir / "mling_demo_concept_language_schematic.png",
                dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.show()


plot_concept_language_schematic()

# %% [markdown]
# ### 3.1 概念先行モデル
#
# Part 1.5 で定義した単語ベクトル $v_L(w)$（概念 $w$ の言語 $L$ 訳トークンの正規化埋め込み。英語は $v_\text{en}(w)$）が、
# 作業空間（3.2 の PCA-128）で次のように作られていると考える:
# $$ v_L(w) = R(L)\, R(w)\, v_o. $$
# - $v_o$: 基準方向（英語概念ベクトルの平均方向）。
# - $R(w)$: 概念ごとの直交変換。本ノートでは、基準方向 $v_o$ を英語概念ベクトル $v_\text{en}(w)$ へ写し、両者が張る 2 次元平面の
#   直交補空間では恒等に作用する **最小平面回転** として定める（後の `rinv` がこれを実装）。「言語に依らない」は発見された性質ではなく、
#   英語側でそう **構成した仮定** である。
# - $R(L)$: 言語ごとの直交変換。**英語側のベクトル配置を言語 $L$ 側へ整列する**回転で、対訳ペアから推定する（英語は基準＝$R(\text{en})=I$）。
#   別々の空間があるのではなく、単一の共有埋め込みの中での整列変換。
#
# 英語は回転なしなので $v_\text{en}(w) = R(w)\,v_o$。つまり各語は、共通の基準点 $v_o$ を「概念の回転」と「言語の回転」で
# 2 段に回した先にある。これは、概念の回転を先に・言語の回転を後に作用させる作り方であり、冒頭で **概念先行モデル** と呼んだものにあたる。
# もしこの概念先行モデルが正しければ、**言語の回転 $R(L)$ を打ち消せば概念だけが残り、
# 概念の回転 $R(w)$ を打ち消せば言語だけが残る**はず。
#
# **言語転移への含意（冒頭の動機）**: 概念先行モデルが正しければ、言語 $L$ の単語ベクトルから言語 $L'$ の単語ベクトルへは、
# 概念 $w$ によらない一つの変換で移せる。実際
# $$ v_{L'}(w) = R(L')\,R(w)\,v_o = R(L')\,R(L)^{-1}\,R(L)\,R(w)\,v_o = R(L')\,R(L)^{-1}\,v_L(w) $$
# であり、変換 $R(L')R(L)^{-1}$ は概念 $w$ を含まない。ある言語で得た知識を、意味ごとに作り直さず別の言語へ運べる。すなわち、冒頭で述べた
# 「言語をまたいで知識を効率よく運べる構造」がこれである。同じ概念非依存の転移は加法モデルも定数オフセットで可能であるが、
# 言語先行モデルでは概念ごとに異なる変換が必要になる。


# %% [markdown]
# ### 3.2 PCA による 128 次元の作業空間
#
# 回転 $R(L)$ は対訳ペアから推定する（次節）。ところが元の 2560 次元のまま推定すると、推定に使っていない語に当てはまらない
# （過学習）。そこで **PCA で 128 次元に落とした作業空間** で回転を扱う（2.1 で既に用意した空間。コードでは `ep(t)` が $e(t)$ の 128 次元版）。
# **以降、単語ベクトル $v_L(w)$ はこの 128 次元での値を指す**（記号は変えず、空間だけを 2560→128 に切り替える。生 2560 次元との対比は 3.5 で行う）。
# 次元 $K$ を 2560→512→…→32 と動かして言語の分離度を測ると **128 付近が最良**で、full 次元では変換後も
# 同一言語のペアの類似が対訳ペアの類似を上回らず、言語構造の顕在化が確認できない。だから 128 を使う。

# %% [markdown]
# ### 3.3 言語回転 $R(L)$ を対訳ペアから推定する
#
# 回転の推定に使う概念集合を $D_\text{fit}$ とする（**表示・評価に使う概念を含まない** fitting set。
# 同じ概念で $R(L)$ を作って同じ概念で近いと主張する in-sample fitting を避けるため）。
# $D_\text{fit}$ の各概念 $w$ について、英語ベクトル $v_\text{en}(w)$ を**列として横に並べた行列**を $A$、
# 対応する言語 $L$ のベクトル $v_L(w)$ を同じ順に並べた行列を $B$ とする（どちらも $128 \times |D_\text{fit}|$・**各列が 1 概念**）。
# 3.1 のモデル $v_L(w) = R(L)\,v_\text{en}(w)$（$R(L)$ を左から掛けて英語→言語 $L$）を全概念まとめて書くと
# $$ R(L)\,A \approx B $$
# となる。これを満たす**直交行列** $R(L)$ を最小二乗で求める
# （**直交 Procrustes**＝2 組の対応点をできるだけ重ねる回転・鏡映を求める標準手法）。英語自身は回転なし（基準 $R(\text{en})=I$）。
# 当てはまりの良い（過学習しない）14 言語だけを使うことは 2.1 で確認済み。ここでその 14 言語の $R(L)$ を実際に推定する。

# %%
# ── コード世界 ↔ 数式（3.3）の対応 ─────────────────────────────────────────────────────
# 本コードは実装の都合で「行ベクトル規約」（各概念ベクトルを行に積む）で計算する。これは 3.3 の
# 数式（各概念を列に積む列規約 R(L)·A ≈ B）を そっくり転置した世界 に当たる:
#   ・行列はすべて数式の転置:  コードの A = 数式 A^T,  コードの B = 数式 B^T（各行が 1 概念, |D_fit|×128）。
#   ・scipy.orthogonal_procrustes(A, B) は A·Ω ≈ B を満たす直交行列 Ω を返す。列規約に直すと Ω^T·A ≈ B、
#     すなわち数式の R(L) = Ω^T。よって R[L] に格納されるのは Ω = R(L)^T（＝数式 R(L) の転置）。
#   ・行ベクトル v（1×128）への作用:  v @ R[L] = 列規約の R(L)·v,   v @ R[L].T = R(L)^T·v（= R(L)^-1·v）。
# 数値は規約に依らず一致する（図・指標は同じ）。以降のコードはこの行規約で一貫させている。
# ──────────────────────────────────────────────────────────────────────────────────────
# R(L) 推定: 2.1 の作業空間 ep と対訳ペア fitpairs を再利用。表示概念（draw）を除いた held-out ペアで直交 Procrustes。
draw_en_tok = {c["en"] for c in draw}
R = {}
for L in LANGS_ROT:
    D_fit = [(et, lt) for et, lt in fitpairs[L] if et not in draw_en_tok]   # 表示概念を除いた held-out 対訳（D_fit）
    A = np.array([ep(et) for et, lt in D_fit]); B = np.array([ep(lt) for et, lt in D_fit])   # コード A=数式 A^T, コード B=数式 B^T（各行 1 概念）
    R[L], _ = orthogonal_procrustes(A, B)                                    # A·Ω ≈ B → R[L]=Ω=R(L)^T（数式では R(L)·A ≈ B）
v_o = u(np.mean([ep(c["en"]) for c in draw], axis=0))                        # 基準点ベクトル = 英語概念の平均方向
print(f"Estimated R(L) for {len(LANGS_ROT)} languages (held-out translation pairs, orthogonal Procrustes, 128 dimensions). English has no rotation (reference).")

# %% [markdown]
# ### 3.4 逆概念変換 $C_L^{-1}(w)$
#
# モデル $v_L(w)=R(L)R(w)v_o$ が厳密に成り立つ場合、以下に述べる変換によって、**概念成分を完全に消して言語成分だけ残す**ことができる。
# もちろん、実際の埋め込みでは変換は近似なので、正確には **概念成分を抑え、言語構造を相対的に顕在化する**。すなわち、
# 変換後も概念の寄与は完全にはゼロにならず、点は 1 点に潰れず散らばる。
#
# 概念の回転 $R(w)$ を**言語 $L$ の座標系で表した**ものを、**概念変換**として定義する:
# $$ C_L(w) = R(L)\,R(w)\,R(L)^{-1} \qquad(\textit{concept transform for language } L). $$
# その**逆変換**を
# $$ C_L^{-1}(w) = R(L)\,R(w)^{-1}\,R(L)^{-1} \qquad(\textit{inverse concept transform for language } L) $$
# と定義する。$C_L^{-1}(w)$ は、言語 $L$ の埋め込みから概念 $w$ による変化を取り除く変換である。
#
# モデル $v_L(w)=R(L)R(w)\,v_o$ は $C_L(w)$ を使うと $v_L(w)=C_L(w)\,R(L)\,v_o$ と書ける。この逆変換後のベクトルを
# $$ x_L(w) \;:=\; C_L^{-1}(w)\,v_L(w) \;\approx\; R(L)\,v_o \;=:\; \hat{x}_L $$
# と書く。右辺 $\hat{x}_L=R(L)v_o$ は単語 $w$ を含まない（言語 $L$ の基準位置だけ）ので、同じ言語の単語はすべて同じ点
# $\hat{x}_L$ に集まるはず（＝下図の +）。変換は次の 3 段階からなる:
# 1. $R(L)^{-1}$: 訳語ベクトルを英語側の配置へ写し戻す。
# 2. $R(w)^{-1}$: 概念回転を打ち消し、基準方向 $v_o$ の近くへ戻す。
# 3. $R(L)$: ベクトルを言語 $L$ 側の配置へ再び写す。
#
# なぜ足し算や全言語共通の 1 変換でなく、言語ごとの共役か: モデルは概念の回転を先に、言語の回転を後に作用させる
# （$v_L(w)=R(L)R(w)v_o$）。これら 2 つの変換が可換なら、共通の $R(w)^{-1}$ を適用することで $R(L)v_o$ に写せるが、一般に
# 直交変換は非可換なので言語ごとの共役が要る。この順序を入れ替えた **言語先行モデル** と、足し算による **加法モデル** では、同じ効果が得られないことを Part 4 で確かめる。

# %%
def rinv(a, b, x):
    """基準点 a から b へ向ける球面上の回転の逆を x に適用（概念回転 R(w)^{-1} に相当）。"""
    cth = float(np.dot(a, b))
    if cth > 1 - 1e-9:
        return x.copy()
    v = u(b - cth * a); ux, vx = float(np.dot(a, x)), float(np.dot(v, x))
    th = -np.arccos(np.clip(cth, -1, 1)); cs, sn = np.cos(th), np.sin(th)
    return x - ux * a - vx * v + (cs * ux - sn * vx) * a + (sn * ux + cs * vx) * v


def dvec_of(c, L):
    """逆概念変換 C_L^{-1}(w) を訳語ベクトルに適用: 英語側へ写し戻す -> 概念を打ち消す -> 言語 L 側へ再び写す。結果は言語 L の特徴ベクトル。
    行規約（R[L]=R(L)^T。R(L) 推定セルの「コード↔数式の対応」を参照）。"""
    x1 = ep(c["langs"][L][0]) @ R[L].T      # 1. 英語側へ写し戻す R(L)^{-1}（行規約: v @ R[L].T = R(L)^-1·v）
    x2 = rinv(v_o, ep(c["en"]), u(x1))      # 2. 概念を打ち消す R(w)^{-1}
    return u(u(x2) @ R[L])                   # 3. 言語 L 側へ再び写す R(L)（行規約: v @ R[L] = R(L)·v）


def plot_dspace():
    pts, D = [], []
    for c in sub:
        for L in c["langs"]:
            pts.append((L, c["langs"][L][1])); D.append(dvec_of(c, L))
    npts = len(D)
    center_langs = ["en"] + LANGS_ROT
    center = [u(v_o)] + [u(v_o @ R[L]) for L in LANGS_ROT]
    XY = TSNE(n_components=2, metric="cosine", perplexity=min(30, max(5, (len(pts) - 1) // 3)), random_state=0, init="pca").fit_transform(np.array(D + center))
    XYp, XYr = XY[:npts], XY[npts:]; lang = [p[0] for p in pts]
    fig, ax = plt.subplots(figsize=(13, 11))
    for L in LANGS_ROT:
        i = [k for k in range(len(pts)) if lang[k] == L]
        ax.scatter(XYp[i, 0], XYp[i, 1], s=34, color=[LANG_COLOR[L]], alpha=0.7, edgecolors="none", zorder=2)
    for k in range(len(pts)):
        ax.text(XYp[k, 0], XYp[k, 1], _fix_rtl(pts[k][1]), fontsize=6.5, color=LANG_COLOR[lang[k]], alpha=0.85, zorder=3)
    for i, L in enumerate(center_langs):
        ax.scatter([XYr[i, 0]], [XYr[i, 1]], marker="P", s=210, color=[LANG_COLOR[L]], edgecolors="black", linewidths=1.5, zorder=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("After inverse concept transform $C_L^{-1}(w)$ (PCA-128)", fontsize=13, fontweight="bold")
    _famset = [f for f in CANON_FAM if f != "Other" and f in {fam_of(L) for L in LANGS_ROT}]
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=(0.1, 0.1, 0.1, 1.0), ms=9, label="en (hub)")]
              + [Line2D([0], [0], marker="o", color="w", markerfacecolor=CANON_FAM[f], ms=9, label=f) for f in _famset]
              + [Line2D([0], [0], marker="P", color="w", markerfacecolor="lightgray", markeredgecolor="black", ms=13, label="language center $\\hat{x}_L=R(L)v_o$")],
              fontsize=9.5, loc="best", title="color = language group")
    fig.tight_layout(); fig.savefig(outputs_dir / "mling_demo_x_tsne.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.show()


plot_dspace()

# %% [markdown]
# **図の内容**: 各点は、表示中の概念×言語の訳語トークンに逆概念変換 $C_L^{-1}(w)$ を適用した
# ベクトルを、コサイン距離の t-SNE で 2 次元へ落としたもの。2.3 と同じ表示概念・同じ言語を用いるが、作業空間は
# 2.3 の生 2560 次元ではなく **PCA-128**（3.5 の (2) と同じ空間）である点に注意。英語の概念点は逆概念変換で
# すべて基準 $v_o$ に厳密一致する（基準言語では $C_\text{en}^{-1}(w)\,v_\text{en}(w)=R(w)^{-1}R(w)\,v_o=v_o$）ので、
# 個別の点としては描かず大きな + の中心 1 点で代表される（消えているのではなく全点が + に重なっている）。
# 点の色は言語グループ（CANON_FAM、英語＝黒のハブ）、各点の訳語ラベルで個々の語が読める。大きな + は各言語の中心
# $\hat{x}_L=R(L)\,v_o$（基準点 $v_o$ を各言語へ写した位置。英語では $\hat{x}_\text{en}=v_o$）。**観察される配置**: 点は概念ごとではなく
# 言語（＝色）ごとにまとまり、各言語の中心 + は互いに離れて置かれる。同じ言語グループの色（例: CJK の zh/ja/ko＝赤）どうしは
# 近くに来る。対照として、変換前の 2.3（生 2560 次元）では同じ概念の語が概念ごとにまとまっていた。

# %% [markdown]
# **手続き**: 3.3 で推定した言語回転 $R(L)$（表示概念を除いた held-out 対訳 $D_\text{fit}$ だけで推定）と、
# 英語側で定めた概念回転 $R(w)$ を用い、表示中の各訳語トークンに $C_L^{-1}(w)=R(L)R(w)^{-1}R(L)^{-1}$ を適用した
# （3 段階: 英語側へ写し戻す → 概念回転を打ち消す → 言語 $L$ 側へ再び写す）。その結果ベクトルを t-SNE にかけたのが上図である。次の 3.5 では、この変換前後の
# 違いを within / cross / translation の 3 群の余弦類似度の分布として数値化する。
#
# **解釈**: 変換前は同じ概念の語どうしの類似度が高かったのに対し、変換後は同じ言語の語どうしの類似度が相対的に高くなった。
# これは、概念依存の構造を抑えることで、もとの埋め込みに含まれていた言語依存の構造が顕在化したものと解釈できる。この結果は、
# 概念回転 $R(w)$ を先に、言語回転 $R(L)$ を後に作用させるモデル $v_L(w)=R(L)R(w)v_o$ と整合的である。加法モデルによる
# 説明可能性は Part 4.2 で検証する。
#
# - **点が「雲」になること**: モデル $v_L(w)=R(L)R(w)\,v_o$ が厳密なら、変換後は $C_L^{-1}(w)\,v_L(w)=R(L)\,v_o$ となり、
#   言語 $L$ の単語は 1 点（中心 +）に潰れるはず。実際の広がりは、データが純粋な回転からずれる残差（$R(w)$ の球面回転近似・
#   $R(L)$ の有限推定）と解釈する。
# - **同じ言語グループが近いこと**: 中心 $R(L)\,v_o$ どうしの近さは $R(L)$ の似方を反映する。英語との対応の付き方が近い言語
#   （例: CJK）は $R(L)$ が似るため隣り合う、と読める。これは Part 1 のクラスタと同じ幾何を別角度から見たことになる。
# - **循環の程度**: $R(L)$ の**直接推定**には表示概念を使っていない。ただし言語選択・PCA 基底・基準方向 $v_o$ の構成には評価概念が
#   関与するため、**完全な外部評価ではない**（$R(L)$ 推定に関する in-sample fitting を避けている、という限定的な意味）。

# %% [markdown]
# ### 3.5 3 群の類似度分布
#
# 3 種類のペアの余弦類似度の分布を比べる（回転 14 言語・概念プール draw）。
# - **within（青・同一言語）**: 同じ言語の別々の単語（同一言語・別概念）。
# - **cross（灰・基準分布）**: 異なる 2 言語から概念を独立に選んだペア（概念の一致・不一致は条件づけない）。言語・概念の特別な対応を課さない基準分布として用いる。
# - **translation（赤・対訳）**: 同じ意味の訳語ペア（同一概念・別言語）。
#
# 各分布の平均を $\bar w$（青）, $\bar c$（灰）, $\bar t$（赤）とする。基準分布 $\bar c$ に対する平均コサイン類似度の上昇量を
# **類似度ゲイン**（similarity gain）と呼び、同一言語という制約による類似度ゲインを **言語ゲイン**、同一概念という制約による
# 類似度ゲインを **概念ゲイン** と定義する（本ノートで定義する量。既存研究の指標ではない）:
# $$ \Delta_\text{lang} = \bar w - \bar c \quad(\textit{language gain}), \qquad
#    \Delta_\text{concept} = \bar t - \bar c \quad(\textit{concept gain}). $$
# 前者は「同一言語のペアが基準よりどれだけ近いか」、後者は「同一概念（対訳）ペアが基準よりどれだけ近いか」。この 2 つの差を
# **言語–概念類似度コントラスト**（language–concept similarity contrast）と定義し、モデルの単一スコアとして使う:
# $$ \text{contrast} = \Delta_\text{lang} - \Delta_\text{concept} = \bar w - \bar t. $$
# $\text{contrast}>0$ なら平均で同一言語のペアが対訳ペアより近い（**言語優位の表現**）、$<0$ なら対訳ペアの方が近い（**概念優位の表現**）。
# 表には 3 平均と 2 ゲインを内訳として示し、概念優位か言語優位かの **分類は contrast の符号に基づいて行う**。

# %%
# 3 群 cos ヘルパ: (concept c, lang L)->ベクトル を受け、draw 全体・LANGS_ROT で within/cross/translation を返す。
# vec_of に何を渡すかで測る空間が決まる（生 2560 次元 / PCA-128 変換前 / PCA-128 変換後）。Part 4 も同じ関数を使う。
# 基準点である英語ハブ（全概念で共有）は群統計から除く。
# 3 群はいずれも全ペアを全数列挙する（ランダムサンプリングは無く決定的）。今回は draw=855 と小さいので、
# 各群を行列積 1 発（V@V.T, byL[li]@byL[lj].T）でまとめて計算できる。ただし cross は言語ペアごとに |D_li|x|D_lj|
# の外積で概念数の 2 乗に増えるため、辞書がもっと大きい場合はこの全数列挙をやめ、各群からランダムサンプリングした
# 部分集合で平均を推定する形に置き換えるべき。
def three_groups_vec(vec_of):
    dv = {(i, L): vec_of(c, L) for i, c in enumerate(draw) for L in c["langs"]}
    byL = {L: [dv[(i, L)] for i, c in enumerate(draw) if L in c["langs"]] for L in LANGS_ROT}
    within, cross, trans = [], [], []
    for L in LANGS_ROT:                                       # within: 同一言語・異概念の全ペア cos（上三角）
        V = np.array(byL[L]); within += list((V @ V.T)[np.triu_indices(len(V), 1)])
    for li, lj in itertools.combinations(LANGS_ROT, 2):       # cross: 異言語ペアの全概念 x 全概念 cos（外積を丸ごと）
        cross += list((np.array(byL[li]) @ np.array(byL[lj]).T).ravel())
    for i, c in enumerate(draw):                              # translation: 同一概念・異言語の cos
        for li, lj in itertools.combinations(list(c["langs"]), 2):
            trans.append(float(dv[(i, li)] @ dv[(i, lj)]))
    return np.array(within), np.array(cross), np.array(trans)


def sim_scores(w, c, s):
    """3 群平均 (within=青, cross=灰, translation=赤) から言語/概念ゲインとコントラスト（スコア）を返す。
    lang_gain=w̄−c̄, concept_gain=s̄−c̄, contrast=w̄−s̄(=lang_gain−concept_gain)。"""
    wm, cm, sm = float(np.mean(w)), float(np.mean(c)), float(np.mean(s))
    return dict(within=wm, cross=cm, translation=sm, lang_gain=wm - cm, concept_gain=sm - cm, contrast=wm - sm)


def print_scores(name, groups):
    d = sim_scores(*groups)
    print(f"  {name:<22}: within {d['within']:+.3f} / cross {d['cross']:+.3f} / translation {d['translation']:+.3f}"
          f"  |  lang_gain {d['lang_gain']:+.3f} / concept_gain {d['concept_gain']:+.3f}  =>  contrast {d['contrast']:+.3f}")


g_raw = three_groups_vec(lambda c, L: e(c["langs"][L][0]))    # (1) 生 2560 次元（2.3 の t-SNE に対応）
g_pre = three_groups_vec(lambda c, L: ep(c["langs"][L][0]))   # (2) PCA-128・変換前（新規）
g_post = three_groups_vec(dvec_of)                            # (3) PCA-128・変換後 = 逆概念変換 C_L^{-1}（3.4 に対応）
bins = np.linspace(-0.6, 1.0, 121); ctr = 0.5 * (bins[1:] + bins[:-1]); HC = {"w": "#1f77b4", "c": "#7f7f7f", "s": "#d62728"}


def draw_hist(ax, within, cross, trans, title):
    for x, cc_ in [(within, HC["w"]), (cross, HC["c"]), (trans, HC["s"])]:
        h = np.histogram(x, bins=bins, density=True)[0]
        ax.fill_between(ctr, h, color=cc_, alpha=0.28, step="mid"); ax.plot(ctr, h, color=cc_, lw=2.2, drawstyle="steps-mid"); ax.axvline(float(np.mean(x)), color=cc_, ls="--", lw=1.4)
    ax.set_xlabel("cosine"); ax.set_title(title, fontsize=11, fontweight="bold"); ax.set_xlim(-0.6, 1.0); ax.spines[["top", "right"]].set_visible(False)


def plot_hist_one(groups, title, fname):
    """1 パターンの 3 群ヒストを 1 枚で描く（別々の図にして後で再利用しやすくする）。"""
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    draw_hist(ax, groups[0], groups[1], groups[2], title)
    ax.legend(handles=[Line2D([0], [0], color=HC["w"], lw=3, label="within (blue, same lang, diff concept)"),
                       Line2D([0], [0], color=HC["c"], lw=3, label="cross (gray, diff lang = baseline)"),
                       Line2D([0], [0], color=HC["s"], lw=3, label="translation (red, same concept, diff lang)")], fontsize=8.5, loc="upper left")
    fig.tight_layout(); fig.savefig(outputs_dir / fname, dpi=170, bbox_inches="tight", facecolor="white"); plt.show()


plot_hist_one(g_raw, "(1) Raw embeddings (2560-dim)", "mling_demo_raw_hist.png")
plot_hist_one(g_pre, "(2) Before transform (PCA-128)", "mling_demo_rawpca_hist.png")
plot_hist_one(g_post, "(3) After $C_L^{-1}$ (PCA-128)", "mling_demo_x_hist.png")

# %% [markdown]
# **図の内容**: 3 枚とも 3 群コサインのヒスト（within 青／cross 灰＝ベースライン／translation 赤・破線＝各群平均）。測る空間だけを
# 変えて並べた: (1) 生 2560 次元（native、2.3 の t-SNE に対応）、(2) PCA-128・変換前、(3) PCA-128・変換後＝逆概念変換
# $C_L^{-1}$（3.4 に対応）。**観察される分布**: (1)(2) はどちらも赤（translation）が最も右、(3) だけ青（within）が最も右へ動く。
# 各群の平均・ゲイン・contrast は次のセルの表にまとめる。

# %%
# 大きな表: 3 パターン × (3 群平均・2 ゲイン・contrast)。数値は対象言語 LANGS_ROT と概念プール draw 全体の
# 全ペア（全数列挙・サンプリングなし）で決まる決定的な量。SEED は t-SNE 表示の 48 概念を選ぶだけで、この表には効かない。
print("3-group means and derived quantities  [baseline=cross(gray); lang_gain=within-cross, concept_gain=translation-cross, contrast=within-translation=score]")
print_scores("(1) raw 2560-dim", g_raw)
print_scores("(2) before transform PCA-128", g_pre)
print_scores("(3) after transform C_L^-1", g_post)

# %% [markdown]
# **表: 3 パターンの 3 群平均と派生量**（数値は上のセルの print と一致）。
# この数値は対象言語 $\mathcal{L}$（14 言語）と概念プール `draw`（855 概念）の全ペアを全数列挙して平均した決定的な量で、
# within / cross / translation のいずれにもランダムサンプリングは入らない（`SEED` は t-SNE に表示する 48 概念を選ぶだけで、
# この表の数値は変えない）。辞書がこの規模なので全ペアを行列積で一括計算しているが、もっと大きい辞書では、とくに全概念×全概念の
# 外積になる cross が概念数の 2 乗で増えるため、各群からランダムサンプリングした部分集合で平均を推定する形に置き換えるのが適切である。
#
# | パターン（測る空間） | within (青) | cross (灰・基準) | translation (赤) | $\Delta_\text{lang}$ | $\Delta_\text{concept}$ | contrast |
# |---|---|---|---|---|---|---|
# | (1) 生 raw 2560 次元 | 0.109 | 0.092 | 0.246 | 0.017 | 0.154 | **−0.137** |
# | (2) 変換前 PCA-128 | 0.091 | 0.011 | 0.339 | 0.079 | 0.328 | **−0.248** |
# | (3) 変換後 $C_L^{-1}$ PCA-128 | 0.253 | 0.018 | 0.105 | 0.235 | 0.087 | **+0.148** |
#
# **観察**（事実）: contrast は (1) −0.137・(2) −0.248 と負（同一概念のペアが最も近い＝概念優位の表現）で、(3) のみ +0.148 と正
# （同一言語のペアが最も近い＝言語優位の表現）。符号が反転するのは (3) だけ。
#
# **解釈**: (1)→(2) の PCA-128 射影では cross（基準）が 0.092→0.011 に下がり translation が 0.246→0.339 に上がり、概念構造が
# むしろ際立つ（が contrast は負のままで概念優位）。**概念優位から言語優位へ変わる**のは (2)→(3) の逆概念変換で、$\Delta_\text{lang}$ が
# 0.079→0.235 へ大きく上がり $\Delta_\text{concept}$ が 0.328→0.087 に下がる。**つまり言語構造の顕在化は次元削減でなく $C_L^{-1}$ 変換そのものの効果**。
# 手法の公平な前後比較は同一空間の (2)→(3) で見る（(1) は native な原空間で 2.3 の t-SNE に対応）。

# %% [markdown]
# ## Part 4　二つのベースラインモデル
#
# Part 3 では、モデル $v_L(w)=R(L)R(w)v_o$ のもとで**共役** $C_L^{-1}(w)=R(L)R(w)^{-1}R(L)^{-1}$ を作ると
# 概念優位の表現から言語優位の表現へ変わった（言語構造が顕在化した）。では、別の 2 つのモデル、**言語先行モデル**と**加法モデル**は
# $W_E$ の観測された幾何によって同じくらいよく支持されるだろうか。
#
# 1. **言語先行モデル**（4.1）: $v_L(w)=R(w)R(L)v_o$。概念消去は $R(w)^{-1}$ を掛けるだけ（言語ごとの $R(L)$ は不要）。
# 2. **加法モデル**（4.2）: $v_L(w)=v_\text{en}(w)+a_L$。翻訳を回転でなく定数ベクトルの足し算で表す。
#
# どちらも Part 3 と**同じデータ・同じ指標**で試す。指標は Part 3.5 で定義した 3 群（within/cross/translation）と、その要約スコア
# **言語–概念コントラスト** $\text{contrast}=\bar w-\bar t$（$>0$ なら平均で同一言語のペアが最も近く言語優位、$<0$ なら同一概念の
# ペアが最も近く概念優位）。すべて PCA-128・回転 14 言語・概念プール `draw` でそろえ、どちらのモデルでも言語構造が顕在化しない
# （言語優位にならない）ことを確かめる。

# %% [markdown]
# ### 4.1 言語先行モデル $v_L(w)=R(w)R(L)v_o$
#
# もし**言語の回転 $R(L)$ を先に、概念の回転 $R(w)$ を後に**作用させる順序（$v_L(w)=R(w)R(L)v_o$）なら、概念を消すのは簡単で、
# 英語側で定めた概念回転の逆 $R(w)^{-1}$ を訳語ベクトルに掛けるだけでよい。この逆変換後のベクトルを
# $$ y_L(w) \;:=\; R(w)^{-1}\,v_L(w) $$
# と書く。もしこの言語先行モデルが正しければ $y_L(w)=R(L)v_o$（語 $w$ 非依存）で、言語ごとの $R(L)$ を推定する必要すらない。
# これを実際の埋め込みに適用し、Part 3 の共役変換と同じように言語ごとに集まるかを見る。
#
# **参照点** ×: t-SNE には各言語の $y_L(w)$ の平均 $\hat{y}_L=\text{mean}_{w\in D_\text{fit}}\,y_L(w)$
# （held-out $D_\text{fit}$、正規化なし、コサインで測る）を × で重ねる。もし言語先行モデルが正しければ $\hat{y}_L=R(L)v_o$（＝3.4 の $\hat{x}_L$）となり、
# 点が × に潰れるはず。実データでそうなるかを見る。
#
# **言語転移への含意**: 概念先行モデルでは、言語間の転移が概念によらない一つの変換 $R(L')R(L)^{-1}$ で書けた（3.1）。言語先行モデルでは
# そうならない。$v_L(w)=R(w)R(L)v_o$ と $v_{L'}(w)=R(w)R(L')v_o$ から $v_o$ を消すと
# $$ v_{L'}(w) = R(w)\,R(L')\,R(L)^{-1}\,R(w)^{-1}\,v_L(w) $$
# となり、転移の変換は $R(w)$ による共役を含む。すなわち **概念 $w$ ごとに変わり**、言語先行モデルは概念非依存の言語転移を与えない。

# %%
# 言語先行モデルの逆変換 y_L(w)=R(w)^{-1}v_L(w): R(w)^{-1} を訳語ベクトルに直接掛ける（rinv=R(w)^{-1}, v_o は 3.3。R(L) 不要）。
def dvec_swap(c, L):
    """言語先行モデル v_L(w)=R(w)R(L)v_o の概念消去 y_L(w)=R(w)^{-1}v_L(w)。R(w)^{-1} のみ（言語ごとの R(L) を使わない）。rinv は単位ベクトルの回転なので結果も単位（単位化不要・参照点 ŷ_L と対称）。"""
    return rinv(v_o, ep(c["en"]), ep(c["langs"][L][0]))


def yhat_of(L):
    """言語先行の参照点 ŷ_L = mean_{w in D_fit} y_L(w)（held-out D_fit）。生の平均が基準 R(L)v_o の推定（重心）。点 dvec_swap（単位）の平均なので対称。"""
    D_fit = [(et, lt) for et, lt in fitpairs[L] if et not in draw_en_tok]
    return np.mean([rinv(v_o, ep(et), ep(lt)) for et, lt in D_fit], axis=0)

# %%
# 言語先行の逆変換後を t-SNE（3.4 と同じ PCA-128 空間・同じ t-SNE 設定・sub の概念）。× = ŷ_L（言語先行の逆変換 y_L(w) の平均）だけを重ねる。
def plot_swap_tsne():
    pts, D = [], []
    for c in sub:
        for L in c["langs"]:
            pts.append((L, c["langs"][L][1])); D.append(dvec_swap(c, L))
    npts = len(D); lang = [p[0] for p in pts]
    yhat = [yhat_of(L) for L in LANGS_ROT]                 # ŷ_L = 言語先行の逆変換 y_L(w) の平均（参照点, held-out D_fit）
    XY = TSNE(n_components=2, metric="cosine", perplexity=min(30, max(5, (len(pts) - 1) // 3)), random_state=0, init="pca").fit_transform(np.array(D + yhat))
    XYp = XY[:npts]; XYy = XY[npts:]
    fig, ax = plt.subplots(figsize=(13, 11))
    for L in LANGS_ROT:
        i = [k for k in range(len(pts)) if lang[k] == L]
        ax.scatter(XYp[i, 0], XYp[i, 1], s=34, color=[LANG_COLOR[L]], alpha=0.7, edgecolors="none", zorder=2)
    for k in range(len(pts)):
        ax.text(XYp[k, 0], XYp[k, 1], _fix_rtl(pts[k][1]), fontsize=6.5, color=LANG_COLOR[lang[k]], alpha=0.85, zorder=3)
    for i, L in enumerate(LANGS_ROT):
        ax.scatter([XYy[i, 0]], [XYy[i, 1]], marker="X", s=150, color=[LANG_COLOR[L]], edgecolors="black", linewidths=1.3, zorder=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Language-first model: $R(w)^{-1}$ only (PCA-128)", fontsize=12.5, fontweight="bold")
    _famset = [f for f in CANON_FAM if f != "Other" and f in {fam_of(L) for L in LANGS_ROT}]
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=CANON_FAM[f], ms=9, label=f) for f in _famset]
              + [Line2D([0], [0], marker="X", color="w", markerfacecolor="lightgray", markeredgecolor="black", ms=12, label="$\\hat{y}_L$ (language-first centroid)")],
              fontsize=9, loc="best", title="color = language group")
    fig.tight_layout(); fig.savefig(outputs_dir / "mling_demo_y_tsne.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.show()


plot_swap_tsne()

# %% [markdown]
# **図の内容**: 各点は、表示概念×言語の訳語トークンの言語先行の逆変換 $y_L(w)=R(w)^{-1}v_L(w)$（**PCA-128**）をコサイン
# 距離の t-SNE で 2 次元化したもの（3.4 と同じ PCA-128 空間・同じ t-SNE 設定。レイアウトは各図で独立に再最適化される。色＝言語グループ）。× は各言語の平均 $\hat{y}_L$（逆変換の
# 参照点、held-out $D_\text{fit}$）。**観察される配置**: 3.4（共役後）では点が広く離れた各言語中心へ集まったが、ここでは点は
# 概念ごとに散らばり × に潰れない。同じ意味の訳語（city＝ciudad/cidade/città/都市/град、comment＝comentario/コメント/댓글
# など）が言語グループをまたいで近いままで、概念優位の様相が残る。しかも参照点 × ($\hat{y}_L$) はおおむね中央付近に集まり、言語先行の
# 逆変換では言語中心そのものがほとんど分離しない（3.4 の言語中心が広く離れていたのと対照的）。

# %% [markdown]
# **解釈**: 意味（概念）がなお支配的で、$R(w)^{-1}$ だけでは言語構造が顕在化しない。言語先行モデル（言語の回転を先に・概念の回転を
# 後に作用させる $v_L(w)=R(w)R(L)v_o$）はこの診断では支持されない、と解釈できる。定量的な確認（同一言語の構造が実質増えていないこと）は
# 次のヒストグラムと contrast（言語–概念コントラスト）で行う。

# %%
# 言語先行モデルの 3 群分布（この 1 枚＝言語先行モデルのみ。数値もこの図に対応する 1 行だけ出す）。
sw = three_groups_vec(dvec_swap)     # 言語先行 R(w)^{-1}（PCA-128・回転 14 言語・概念プール draw）


def plot_hist_swap():
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    draw_hist(ax, sw[0], sw[1], sw[2], "Language-first $R(w)^{-1}$ (PCA-128)")
    ax.legend(handles=[Line2D([0], [0], color=HC["w"], lw=3, label="within (blue)"),
                       Line2D([0], [0], color=HC["c"], lw=3, label="cross (gray, baseline)"),
                       Line2D([0], [0], color=HC["s"], lw=3, label="translation (red)")], fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(outputs_dir / "mling_demo_y_hist.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.show()


plot_hist_swap()
print("3-group means and derived quantities (this figure = the language-first model only)  [baseline=cross(gray); contrast=within-translation=score]")
print_scores("language-first R(w)^-1 (PCA-128)", sw)

# %% [markdown]
# **図の内容**: 言語先行 $R(w)^{-1}$ 後の 3 群コサイン分布（PCA-128・回転 14 言語・概念プール draw。within 青／cross 灰＝
# ベースライン／translation 赤・破線＝各群平均）。この図に対応する数値（3.5 と同形式・1 行）:
#
# | パターン（測る空間） | within (青) | cross (灰・基準) | translation (赤) | $\Delta_\text{lang}$ | $\Delta_\text{concept}$ | contrast |
# |---|---|---|---|---|---|---|
# | 言語先行 $R(w)^{-1}$ PCA-128 | 0.220 | 0.135 | 0.339 | 0.085 | 0.205 | **−0.119** |
#
# **観察される分布**: 赤（translation, 平均 0.34）が最も右のままで、言語構造は顕在化しない（within 0.22 は cross 0.14 より右だが translation を
# 越えない）。この図の contrast は **−0.119**（負＝概念優位に留まる）。3.5 の変換前 −0.248・変換後 +0.148 と並べると、
# 言語先行は変換前からほぼ動かず、正（言語優位）にならない。

# %% [markdown]
# **解釈**: 言語先行の $R(w)^{-1}$ は同一概念の 2 言語に共通に掛かる直交変換なので **translation の cos を保存**する
# （変換前 0.339 ＝ 言語先行 0.339, 完全一致）。within は 0.091→0.220 と上がるが、cross（基準）も 0.011→0.135 と一緒に
# 上がる（回転後の部分空間での全体的な混み合い）ため、**言語ゲイン $\Delta_\text{lang}$（within−cross）はほぼ横ばい**
# （0.079→0.085）で、同一言語の構造は実質ほとんど増えていない。translation が高いままなので contrast は負のまま（−0.25→−0.12）
# 言語優位にならない。対して共役 $C_L^{-1}$ は言語ごとの $R(L)$ で translation を 0.339→0.105 に崩し、かつ cross を低く保って
# $\Delta_\text{lang}$ を 0.079→0.235 へ大きく上げる（contrast +0.15）。同じデータで言語優位に転じるのは概念先行モデルの共役 $C_L^{-1}$ の側だけで、
# 言語先行モデルの $R(w)^{-1}$ では同じ効果は得られない。

# %% [markdown]
# ### 4.2 加法モデル $v_L(w)=v_\text{en}(w)+a_L$
#
# 別の仮説として、言語差を一定のベクトル加算で表す**加法モデル**を考える: $v_L(w)=v_\text{en}(w)+a_L$
# （$a_L$ は言語 $L$ ごとの平行移動ベクトル）。もしこれが正しければ、訳語から英語ベクトルを引いた差分ベクトル $z_L(w)$ は
# 概念によらず一定 $a_L$ になる:
# $$ z_L(w) \;:=\; v_L(w)-v_\text{en}(w) \;\approx\; a_L, \qquad \hat{z}_L \;:=\; \operatorname{mean}_{w\in D_\text{fit}}\,z_L(w). $$
# つまり同一言語の $z_L(w)$ が 1 方向にそろえば言語でまとまる（向きをコサインで比べるので正規化はしない）。これを実データ（PCA-128）で確かめる。t-SNE には各言語の
# 平均 $\hat{z}_L$ を **△** で重ねる。
#
# **言語転移への含意**: 加法モデルは、概念非依存の言語転移を与える点では概念先行モデルと同じである。$v_L(w)=v_\text{en}(w)+a_L$ から
# $$ v_{L'}(w) = v_L(w) + (a_{L'}-a_L) $$
# となり、転移は概念 $w$ によらない定数オフセット $a_{L'}-a_L$ の足し算になる。ただし本節で見るように加法モデルは実データの言語構造を
# 顕在化しない（言語優位にならない）ので、転移が概念非依存でも、この診断では概念先行モデルほど支持されない。

# %%
# 加法モデルの差分 z_L(w)=v_L(w)-v_en(w)（訳語−英語, PCA-128）。差なので非単位。生のまま返し（参照点 ẑ_L も生の平均で対称）、単位化は必要な所だけ: t-SNE は cosine なので不要、数値表 three_groups_vec だけ内積=cos のため呼び出し側で u() する。
def dvec_add(c, L):
    """加法モデル v_L(w)=v_en(w)+a_L の生の差分 z_L(w)=v_L(w)-v_en(w)（単位化しない。数値表に渡すときだけ呼び出し側で u()）。"""
    return ep(c["langs"][L][0]) - ep(c["en"])


def zhat_of(L):
    """加法モデルの参照点 ẑ_L = mean_{w in D_fit} z_L(w)（held-out D_fit）。生の平均が加法オフセット a_L の推定（重心）。点 dvec_add も生なので対称。"""
    D_fit = [(et, lt) for et, lt in fitpairs[L] if et not in draw_en_tok]
    return np.mean([ep(lt) - ep(et) for et, lt in D_fit], axis=0)

# %%
# 加法の差分後を t-SNE（△ = ẑ_L）。en は z=0 で向きが未定義なので除く。
def plot_add_tsne():
    pts, D = [], []
    for c in sub:
        for L in c["langs"]:
            pts.append((L, c["langs"][L][1])); D.append(dvec_add(c, L))
    npts = len(D); lang = [p[0] for p in pts]
    zhat = [zhat_of(L) for L in LANGS_ROT]                 # ẑ_L = 差分ベクトルの平均（held-out D_fit）
    XY = TSNE(n_components=2, metric="cosine", perplexity=min(30, max(5, (len(pts) - 1) // 3)), random_state=0, init="pca").fit_transform(np.array(D + zhat))
    XYp = XY[:npts]; XYz = XY[npts:]
    fig, ax = plt.subplots(figsize=(13, 11))
    for L in LANGS_ROT:
        i = [k for k in range(len(pts)) if lang[k] == L]
        ax.scatter(XYp[i, 0], XYp[i, 1], s=34, color=[LANG_COLOR[L]], alpha=0.7, edgecolors="none", zorder=2)
    for k in range(len(pts)):
        ax.text(XYp[k, 0], XYp[k, 1], _fix_rtl(pts[k][1]), fontsize=6.5, color=LANG_COLOR[lang[k]], alpha=0.85, zorder=3)
    for i, L in enumerate(LANGS_ROT):
        ax.scatter([XYz[i, 0]], [XYz[i, 1]], marker="^", s=170, color=[LANG_COLOR[L]], edgecolors="black", linewidths=1.3, zorder=7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Additive model: $z_L(w)=v_L(w)-v_{en}(w)$ (PCA-128)", fontsize=11.5, fontweight="bold")
    _famset = [f for f in CANON_FAM if f != "Other" and f in {fam_of(L) for L in LANGS_ROT}]
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=CANON_FAM[f], ms=9, label=f) for f in _famset]
              + [Line2D([0], [0], marker="^", color="w", markerfacecolor="lightgray", markeredgecolor="black", ms=12, label="$\\hat{z}_L$ (additive centroid)")],
              fontsize=9, loc="best", title="color = language group")
    fig.tight_layout(); fig.savefig(outputs_dir / "mling_demo_z_tsne.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.show()


plot_add_tsne()

# %% [markdown]
# **図の内容**: 各点は、表示概念×言語の訳語トークンの差分ベクトル $z_L(w)=v_L(w)-v_\text{en}(w)$（PCA-128・向きをコサインで測る）を
# t-SNE（コサイン距離）で 2 次元化したもの（色＝言語グループ）。△ は各言語の平均 $\hat{z}_L$（held-out $D_\text{fit}$）。**観察される配置**: 点は言語グループ
# （色）ごとに固まらず全体に混ざり、差分ベクトルは言語でクラスタしない。△（$\hat{z}_L$）はどれも中央付近に集中し、点はそこへ
# 潰れず、言語中心もほとんど分離しない。

# %% [markdown]
# **解釈**: もし加法モデルが正しければ、同一言語の差分ベクトル $z_L(w)$ は概念によらず 1 方向 $a_L$ にそろい、点は △ に集まる
# はず。実際は言語ではそろわない（次の指標で $\Delta_\text{lang}\approx0$ を確認）。今回の選択データと PCA-128 表現では、言語差を単一の定数オフセットでは十分に説明できない。

# %%
# 加法モデルの 3 群分布（この 1 枚＝加法モデルのみ。数値もこの図に対応する 1 行だけ）。
za = three_groups_vec(lambda c, L: u(dvec_add(c, L)))  # 加法 z_L(w)（PCA-128・回転 14 言語・draw）。生の差分を内積=cos のため単位化して渡す


def plot_hist_add():
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    draw_hist(ax, za[0], za[1], za[2], "Additive $z_L(w)$ (PCA-128)")
    ax.legend(handles=[Line2D([0], [0], color=HC["w"], lw=3, label="within (blue)"),
                       Line2D([0], [0], color=HC["c"], lw=3, label="cross (gray, baseline)"),
                       Line2D([0], [0], color=HC["s"], lw=3, label="translation (red)")], fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(outputs_dir / "mling_demo_z_hist.png", dpi=170, bbox_inches="tight", facecolor="white"); plt.show()


plot_hist_add()
print("3-group means and derived quantities (this figure = the additive model only)  [baseline=cross(gray); contrast=within-translation=score]")
print_scores("additive z_L=v_L-v_en (PCA-128)", za)

# %% [markdown]
# **図の内容**: 加法の差分 $z_L(w)$ の 3 群コサイン分布（PCA-128・回転 14 言語・概念プール draw。within 青／cross 灰＝
# ベースライン／translation 赤・破線＝各群平均）。この図に対応する数値（3.5 と同形式・1 行）:
#
# | パターン（測る空間） | within (青) | cross (灰・基準) | translation (赤) | $\Delta_\text{lang}$ | $\Delta_\text{concept}$ | contrast |
# |---|---|---|---|---|---|---|
# | 加法 $z_L=v_L-v_\text{en}$ PCA-128 | 0.279 | 0.223 | 0.460 | 0.056 | 0.237 | **−0.181** |
#
# **観察される分布**: 赤（translation, 平均 0.46）が最も右で、同一概念の差分ベクトルが最もそろう。青（within 0.28）は灰（cross 0.22）
# とほとんど差がなく（$\Delta_\text{lang}$ わずか 0.056）、contrast は **−0.181**（負＝概念優位）。
#
# **解釈**: 差分ベクトルは「概念」方向にはそろう（translation 高）が「言語」方向にはほぼそろわない（$\Delta_\text{lang}\approx0$）。
# 加法モデルは言語構造を顕在化せず、共役のように言語優位にはならない。

# %% [markdown]
# ### 4.3 モデル比較のまとめ
#
# 3 つの逆変換を同じ指標（PCA-128・回転 14 言語・概念プール draw の言語–概念 contrast）で並べる。ベースラインは変換前。

# %%
# 全モデルの contrast 比較（この表だけは全モデルを一度に並べる）。
print("Compare the models by the language-concept contrast (score)  [PCA-128, rotation 14 languages, concept pool draw]")
print_scores("before transform (baseline)", g_pre)
print_scores("conjugation C_L^-1 (Part 3)", g_post)
print_scores("language-first R(w)^-1 (4.1)", sw)
print_scores("additive z_L=v_L-v_en (4.2)", za)

# %% [markdown]
# **表: 逆変換モデルの比較**（contrast > 0 で言語優位の表現に変わる）。
#
# | モデル（逆変換） | within | cross | translation | $\Delta_\text{lang}$ | $\Delta_\text{concept}$ | **contrast** |
# |---|---|---|---|---|---|---|
# | 変換前（ベースライン） | 0.091 | 0.011 | 0.339 | 0.079 | 0.328 | −0.248 |
# | **共役 $C_L^{-1}$**（概念を先に・言語を後にする順序＋共役） | 0.253 | 0.018 | 0.105 | 0.235 | 0.087 | **+0.148** |
# | 言語先行 $R(w)^{-1}$（4.1） | 0.220 | 0.135 | 0.339 | 0.085 | 0.205 | −0.119 |
# | 加法 $z_L=v_L-v_\text{en}$（4.2） | 0.279 | 0.223 | 0.460 | 0.056 | 0.237 | −0.181 |
#
# **観察**: **今回比較した 3 手法の中では**、contrast が正になる（言語優位の表現に変わる）のは **共役 $C_L^{-1}$ だけ**（+0.148）。言語先行（−0.119）も加法（−0.181）も
# 負のままで概念優位に留まる。両者とも $\Delta_\text{lang}$ が小さく（言語先行 0.085・加法 0.056）、同一言語の構造をほとんど作れない。
#
# **解釈**: 今回比較した 3 手法では、言語構造を顕在化できる（contrast>0）のは、概念の回転を先に・言語の回転を後に作用させる順序
# $v_L=R(L)R(w)v_o$ を仮定し、言語ごとに推定した $R(L)$ で **共役** $C_L^{-1}=R(L)R(w)^{-1}R(L)^{-1}$ を作る場合だけだった。言語先行
# （$R(w)^{-1}$ だけ）も足し算（$z_L$）も言語構造を顕在化しない。この結果は、多言語の対応を言語整列変換と概念回転の合成として記述するモデルと整合的である。

# %% [markdown]
# ## まとめ
#
# - **多言語のクラスタリング**（Part 1）: 英語は自分の言語グループ（Germanic）ではなく CJK と束ねられる。欧州言語を多数足しても
#   英語–中国語が上位 $k$ 近傍類似度で最も高い。この傾向は学習データ構成と整合的（ただし英語ピボット辞書と選択基準の影響は分離できていない）。
# - **回転で言語を選ぶ**（Part 2.1）: 各言語の回転 $R(L)$ は対訳ペアが少ないと過学習する。過学習しない 14 言語だけを回転に使う。
# - **生の埋め込み**（Part 2.3）: 同じ意味の訳語が近く、**概念優位の表現**。
# - **逆概念変換 $C_L^{-1}(w)$**（Part 3）: **概念先行モデル** $v_L(w)=R(L)R(w)v_o$ のもとで概念成分を打ち消す共役変換 $C_L^{-1}(w)=R(L)R(w)^{-1}R(L)^{-1}$
#   を作ると、概念優位から言語優位へ変わる（言語構造が顕在化し、同じ言語が集まる）。この結果は、多言語対応を回転（直交変換）の合成として記述するモデルと整合的。
# - **対照実験**（Part 4）: **言語先行モデル**（$R(w)^{-1}$ だけ, 4.1）も、言語差を定数の足し算で表す**加法モデル**
#   （$z_L=v_L-v_\text{en}$, 4.2）も、言語–概念 contrast が負のままで言語優位にならない（4.3 の比較表）。今回比較した 3 手法では、言語優位に転じるのは
#   **概念先行モデル**の共役 $C_L^{-1}$ だけだった。
# - **言語転移への含意**（3.1・4.1/4.2）: 概念先行モデルなら言語間（$L\to L'$）の転移は概念によらない一つの変換 $R(L')R(L)^{-1}$ で書ける
#   （加法も概念非依存、言語先行は概念ごとに変換が変わる）。概念先行が支持されることは、冒頭で述べた**概念に依らない言語転移**という
#   応用上の含意につながる。転移性能自体は生の近さと概ね相関するが完全には一致しない（2.2）。

# %% [markdown]
# ## Part 5　関連研究と本研究の位置づけ
#
# （ここからは研究上の位置づけ、すなわち関連研究です。デモの本体は Part 4・まとめで完結しています。背景に関心がある人向けの補足として読んでください。）
#
# 本研究の面白さは手法ではなく、ひとつの**経験的観察**にある: 多言語の対応を「言語の回転 $R(L)$ と概念の回転 $R(w)$ の合成」と
# みたとき、Qwen3-4B のトークン埋め込みに対する言語–概念コントラストの診断で最もよく支持されたのは、概念を先に・言語を後に作用させる**概念先行モデル** $R(L)R(w)v_o$ だった。
# このモデルのもとで共役変換により概念成分を打ち消すと言語構造が顕在化するが、順序を入れ替えた**言語先行モデル** $R(w)R(L)v_o$ ではそうならない。以下では、この観察を支える道具立て（多言語の直交整列・語彙関係の作用素表現・
# 言語と意味の分離・トークン埋め込みの幾何）がいずれも既知であることを整理し、そのうえで**言語作用素と概念作用素をどの順序で合成するか（どちらのモデルが埋め込みを説明するか）を
# 正面から扱った先行研究は調べた範囲で見当たらない**ことを述べ、本研究をそこに位置づける。手法の各部品は既知なので、**手法の新規性は主張しない**。

# %% [markdown]
# ### 5.1 多言語埋め込みの直交整列
#
# 別々に学習した単語埋め込み空間を線形写像で対応づける研究は [Mikolov ら (2013a)](https://arxiv.org/abs/1309.4168) に始まる。[Xing ら (2015)](https://aclanthology.org/N15-1104/) は単位長正規化と
# 直交制約を導入し、[Artetxe ら (2016)](https://aclanthology.org/D16-1250/) は直交性を「単言語内の内積を保つ条件」として原理化、[Smith ら (2017)](https://arxiv.org/abs/1702.03859) は自己整合な写像が
# **直交であるべき**ことを示した（直交 Procrustes・SVD 解）。[Conneau ら (2018, MUSE)](https://arxiv.org/abs/1710.04087) は教師なしでの対訳辞書誘導を実用化し、その
# **対訳辞書**を本研究は利用する。[Jawanpuria ら (2019, GeoMM)](https://aclanthology.org/Q19-1007/) は各言語固有の回転で共通空間へ整列する（language-as-rotation）。
# すなわち**言語を回転で表すこと自体が標準的**であり、本研究の $R(L)$ はこの系譜の道具である。相違点は、これらが**別々に学習した 2 空間**を
# 整列するのに対し、本研究は**単一の共有トークン埋め込み行列 $W_E$** の部分集合を言語別空間とみなす点にある。

# %% [markdown]
# ### 5.2 作用素としての語彙関係
#
# 語彙の意味関係は伝統的に**加法**の差分ベクトルで表されてきた（[Mikolov ら 2013b](https://aclanthology.org/N13-1090/) の king−man+woman≈queen）。[Ethayarajh (2019)](https://aclanthology.org/D19-1354/) は
# これを**直交作用素**で表し（$R\,\vec{\mathrm{king}}\approx\vec{\mathrm{queen}}$）、直交変換は加法とほぼ同等・一般線形はやや上、と報告した。
# [Reif ら (2026)](https://aclanthology.org/2026.findings-acl.1618/) は LLM の語彙埋め込みでも語形変化（時制・大小文字）を加法の変換ベクトルで表せることを示し、[Park ら (2024)](https://arxiv.org/abs/2311.03658) は概念が
# 線形方向として表れるという仮説を定式化した。すなわち**関係・概念を回転や線形作用素で表すことも既知**。ただし先行研究の多くは**複数の
# 関係作用素の合成**（$R(w_1)R(w_2)$ 的な概念×概念）を扱うのに対し、本研究が概念に用いるのは**単一の $R(w)$** である。また埋め込みを
# 単位球面上で扱う本研究では、**加法モデルは初めから退化する対照（ベースライン）**である（球面上で定ベクトルの平行移動は退化する。局所的には回転が
# 加法の一次近似になるので誤りではない）。Part 4.2 の加法モデルは「言語がそろわない」ことを明示的に確認するための対照に留まる。

# %% [markdown]
# ### 5.3 言語と意味の分離とトークン埋め込みの幾何
#
# 多言語モデルの表現を言語成分と意味成分に分ける研究は本研究に近い。[Gonen ら (2020)](https://aclanthology.org/2020.blackboxnlp-1.5/) は mBERT の「言語部分空間」を同定し、射影で
# 言語情報を出し入れした。[Chang ら (2022)](https://aclanthology.org/2022.emnlp-main.9/) は XLM-R で平均中心化後に言語が似た部分空間を占め、**言語敏感軸と言語中立軸がほぼ直交**する
# ことを示した。入力語彙埋め込みそのものの多言語幾何は [Wen-Yi と Mimno (2023)](https://aclanthology.org/2023.emnlp-main.71/) が直接扱い（文字体系で線形分離でき、幾何がモデル系列で
# 異なる）、[Kim と Lee (2025)](https://arxiv.org/abs/2511.16693) は語彙埋め込みの言語方向と訓練データ構成の関係を調べた。[Mathewson (2026)](https://arxiv.org/abs/2603.02258) は翻訳モデルで概念差ベクトルが
# 言語間で一貫する「オフセット不変性」を報告した。これらを見渡すと、先行研究は概ね 3 つに分かれる: 概念を回転で表す (Ethayarajh)、
# 言語を回転で表す (GeoMM)、言語を**加法**の平均シフトで表す (Chang, Mathewson)。**両者を合成した作用素として結び、その順序を
# 問うたものは、調べた範囲で見当たらない**。本研究はこの隙間に入り、decoder-only の Qwen3-4B へ対象を広げる。

# %% [markdown]
# ### 5.4 作用素合成の非可換性
#
# 「作用素の積の順序が結果を変える」こと自体は、知識グラフ埋め込みで扱われてきた（[Xu と Li 2019](https://aclanthology.org/P19-1026/) ほか。関係合成の非可換性を
# 明示モデル化）。ただし対象は語彙でなく KG の関係である。むしろ本研究と対照的なのは、可換性を**主張・仮定**する研究である:
# [Freenor と Alvarez (2026, RISE)](https://arxiv.org/abs/2510.09790) は意味変換が**順序に依らず可換**（二次まで）と示し、[Liu ら (2017)](https://arxiv.org/abs/1705.02426) は関係行列を**可換族**
# $A_iA_j=A_jA_i$ と仮定する。これらが可換を前提とするのに対し、本研究では**概念先行モデルと言語先行モデルが経験的に等価でなく**、言語と概念という異種の回転が可換でない可能性と整合的である。

# %% [markdown]
# ### 5.5 「潜在言語」研究との区別
#
# 「LLM は内部で英語（や中国語）を経由して考えるか」を、層をまたぐ隠れ状態で調べる研究（[Wendler ら 2024](https://aclanthology.org/2024.acl-long.820/), [Zhong ら 2025](https://aclanthology.org/2025.findings-acl.1350/),
# [Schut ら 2025](https://arxiv.org/abs/2502.15603)）とは**問いが異なる**。それらは logit lens で**計算過程**を追う。本研究が扱うのは、動かさない**語彙埋め込み行列
# $W_E$ の静的な幾何**であり、「モデルが何語で考えるか」という主張はしない。Part 1 で見た英語ハブ・CJK 束は静的埋め込みの性質であって、
# 計算過程の主張ではない。

# %% [markdown]
# ### 5.6 本研究の位置づけ
#
# 以上より、本研究で用いた部品（直交整列・作用素としての語彙意味・言語/意味の分離・トークン埋め込みの幾何）はいずれも既知であり、
# **手法の新規性は主張しない**。本研究の寄与は、ひとつの**経験的観察**にある: 単一の LLM の共有トークン埋め込みにおいて、言語と概念を
# 直交作用素とみなすと、言語–概念コントラストによる診断で最もよく支持されたのは、概念を先に・言語を後に作用させる**概念先行モデル** $R(L)R(w)v_o$ だった。
# このモデルのもとで共役 $C_L^{-1}(w)=R(L)R(w)^{-1}R(L)^{-1}$ により概念成分を打ち消すと言語構造が顕在化するが、順序を入れ替えた**言語先行モデル**ではそうならない。調べた範囲では
# （to the best of our knowledge, in the surveyed literature）、単一の共有 LLM 語彙埋め込みを言語作用素と概念作用素へ**合成順序まで含めて**
# 分解し、共役で概念成分を除いた先行研究は見当たらない。結果は Qwen3-4B に対するものであり、他モデルへ**外挿はしない**。ただし
# 本ノートを実行するだけで任意のモデルに同じ分析を適用でき、多くのモデルの傾向を横断的に調べる土台となる。

# %% [markdown]
# ### 参考文献
#
# 1. Mikolov, Le, Sutskever. 2013. [*Exploiting Similarities among Languages for Machine Translation*](https://arxiv.org/abs/1309.4168). arXiv:1309.4168.
# 2. Xing, Wang, Liu, Lin. 2015. [*Normalized Word Embedding and Orthogonal Transform for Bilingual Word Translation*](https://aclanthology.org/N15-1104/). NAACL-HLT.
# 3. Artetxe, Labaka, Agirre. 2016. [*Learning principled bilingual mappings of word embeddings while preserving monolingual invariance*](https://aclanthology.org/D16-1250/). EMNLP.
# 4. Smith, Turban, Hamblin, Hammerla. 2017. [*Offline Bilingual Word Vectors, Orthogonal Transformations and the Inverted Softmax*](https://arxiv.org/abs/1702.03859). ICLR.
# 5. Conneau, Lample, Ranzato, Denoyer, Jégou. 2018. [*Word Translation Without Parallel Data*](https://arxiv.org/abs/1710.04087). ICLR. (MUSE)
# 6. Jawanpuria, Balgovind, Kunchukuttan, Mishra. 2019. [*Learning Multilingual Word Embeddings in Latent Metric Space: A Geometric Approach*](https://aclanthology.org/Q19-1007/). TACL. (GeoMM)
# 7. Mikolov, Yih, Zweig. 2013. [*Linguistic Regularities in Continuous Space Word Representations*](https://aclanthology.org/N13-1090/). NAACL-HLT.
# 8. Ethayarajh. 2019. [*Rotate King to get Queen: Word Relationships as Orthogonal Transformations in Embedding Space*](https://aclanthology.org/D19-1354/). EMNLP-IJCNLP.
# 9. Reif, Kaplan, Schwartz. 2026. [*Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic*](https://aclanthology.org/2026.findings-acl.1618/). Findings of ACL.
# 10. Park, Choe, Veitch. 2024. [*The Linear Representation Hypothesis and the Geometry of Large Language Models*](https://arxiv.org/abs/2311.03658). ICML.
# 11. Gonen, Ravfogel, Elazar, Goldberg. 2020. [*It's not Greek to mBERT: Inducing Word-Level Translations from Multilingual BERT*](https://aclanthology.org/2020.blackboxnlp-1.5/). BlackboxNLP.
# 12. Chang, Tu, Bergen. 2022. [*The Geometry of Multilingual Language Model Representations*](https://aclanthology.org/2022.emnlp-main.9/). EMNLP.
# 13. Wen-Yi, Mimno. 2023. [*Hyperpolyglot LLMs: Cross-Lingual Interpretability in Token Embeddings*](https://aclanthology.org/2023.emnlp-main.71/). EMNLP.
# 14. Kim, Lee. 2025. [*How Language Directions Align with Token Geometry in Multilingual LLMs*](https://arxiv.org/abs/2511.16693). arXiv:2511.16693.
# 15. Mathewson. 2026. [*Universal Conceptual Structure in Neural Translation: Probing NLLB-200's Multilingual Geometry*](https://arxiv.org/abs/2603.02258). arXiv:2603.02258.
# 16. Xu, Li. 2019. [*Relation Embedding with Dihedral Group in Knowledge Graph*](https://aclanthology.org/P19-1026/). ACL.
# 17. Freenor, Alvarez. 2026. [*Mapping Semantic & Syntactic Relationships with Geometric Rotation*](https://arxiv.org/abs/2510.09790). ICLR. (RISE)
# 18. Liu, Wu, Yang. 2017. [*Analogical Inference for Multi-relational Embeddings*](https://arxiv.org/abs/1705.02426). ICML.
# 19. Wendler, Veselovsky, Monea, West. 2024. [*Do Llamas Work in English? On the Latent Language of Multilingual Transformers*](https://aclanthology.org/2024.acl-long.820/). ACL.
# 20. Zhong et al. 2025. [*What Language Do Non-English-Centric Large Language Models Think in?*](https://aclanthology.org/2025.findings-acl.1350/). Findings of ACL.
# 21. Schut, Gal, Farquhar. 2025. [*Do Multilingual LLMs Think In English?*](https://arxiv.org/abs/2502.15603). arXiv:2502.15603.
# 22. Facebook Research. [*MUSE: Multilingual Unsupervised and Supervised Embeddings*](https://github.com/facebookresearch/MUSE) （対訳辞書）.
