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
# # Multilingual geometry encoded in Qwen3-4B's token embedding matrix
#
# The multilingual language model **Qwen3-4B** first splits a string into units called tokens (words or subwords) and maps each token to a
# 2560-dimensional vector before it begins computing. This lookup table is the **token embedding matrix** $W_E$, of size $151936 \times 2560$,
# shared between the input and output sides. Because Qwen3-4B is a multilingual model, its vocabulary contains tokens from many languages:
# English, Chinese, Japanese, Korean, and more. In this notebook we examine how differences of language and differences of concept (that is,
# of meaning) are combined within this token embedding matrix.
#
# We model each token's vector as the result of composing a "concept rotation" and a "language rotation," starting from a common reference
# direction. The rotation here is, strictly speaking, an orthogonal transform. There are two orders in which to compose the rotations.
# Applying the concept first and the language afterward we call the **concept-first model**; applying the language first and the concept
# afterward we call the **language-first model**. As a simpler point of comparison we also consider the **additive model**, which represents
# the difference between languages by adding a fixed vector.
#
# The question is which model best approximates the actual embedding. If the concept-first model or the additive model holds, a token vector
# obtained in one language can be mapped to another by a single fixed transform, without rebuilding a transform for each concept. This suggests
# that the embedding may contain a structure that supports efficient transfer of information across languages. In this notebook we provide
# numerical and visual evidence that, for Qwen3-4B's token embedding matrix, the concept-first model is better supported than the other two.
#
# The plan is as follows. We first survey the token embedding matrix $W_E$ and, through multilingual clustering, confirm that English clusters
# not with its own language family Germanic but with CJK (Chinese, Japanese, and Korean). Next, to estimate the language rotation accurately,
# we select as experimental targets the languages with enough translation data. Finally, we apply the transform predicted by each of the three
# models and compare, once the concept differences are suppressed, which model makes the language structure most apparent.
#
# We do not need to run the model's 4 billion parameters; we read only a single embedding matrix. We also do not generate text, so
# memory usage is light and it runs safely.
#
# %% [markdown]
# ## 0. Reading the embedding matrix $W_E$
#
# Qwen3-4B is stored split across multiple files (shards). This notebook downloads and reads only the single file containing $W_E$
# (`embed_tokens`). Because it does not load the whole model (about 8 GB), it runs on a laptop or on Google Colab.

# %%
# Run the same file across 3 environments (Mac / Win / Colab). Only Colab needs pip / font installation (Mac/Win already have them, so skip).
import sys, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "transformers==5.9.0", "safetensors", "huggingface_hub",
                    "arabic-reshaper", "python-bidi"], check=True)
    # Colab(Linux) has no CJK / Arabic fonts, so install them via apt (prevents CJK in figures from turning into tofu boxes)
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

# Where figures are saved (same convention as the other notebooks). Colab uses outputs/, local uses outputs/ one level above the notebook.
outputs_dir = Path("outputs") if IN_COLAB else Path("../outputs")
outputs_dir.mkdir(parents=True, exist_ok=True)
print("outputs :", outputs_dir)

# Multilingual font setup (3 environments Mac / Win / Colab): CJK (Chinese/Japanese/Korean/Hangul) + Arabic/Hebrew.
# The approach follows notebook 02 (the canonical, repeatedly verified one). matplotlib's default DejaVu Sans has no CJK
# glyphs, so we give font.family a list and let it "fall back from the front, character by character" (matplotlib >= 3.6).
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
# macOS: .ttc auto-detection is unreliable, so register explicitly (as a safeguard)
for fp in ["/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/AppleSDGothicNeo.ttc",
           "/Library/Fonts/AppleGothic.ttf"]:
    if Path(fp).exists():
        try:
            font_manager.fontManager.addfont(fp)
        except Exception:
            pass
# Colab(Linux): register the CJK/Arabic fonts installed via apt (apt install is in the IN_COLAB block above)
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
    # Japanese
    "Hiragino Sans", "Hiragino Sans GB", "Yu Gothic", "Meiryo", "IPAGothic", "Noto Sans CJK JP", "Noto Sans JP",
    # Chinese (Simplified)  Mac=PingFang / Win=Microsoft YaHei / Colab=WenQuanYi
    "PingFang SC", "Microsoft YaHei", "WenQuanYi Zen Hei", "Noto Sans CJK SC",
    # Korean (Hangul)  Mac=Apple SD Gothic Neo / Win=Malgun Gothic / Colab=Nanum
    "Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic", "NanumGothic", "Nanum Gothic",
    "Noto Sans CJK KR", "Noto Sans KR",
    # Arabic/Hebrew (glyph shapes from the font, RTL reordering by _fix_rtl)  Mac=Geeza Pro / Win=Segoe UI / Colab=Noto
    "Geeza Pro", "Segoe UI", "Noto Sans Arabic", "Noto Naskh Arabic", "Noto Sans Hebrew",
    # all-in-one (Mac)
    "Arial Unicode MS",
]
plt.rcParams["font.family"] = [n for n in _font_candidates if n in _available] + ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
print("font.family =", plt.rcParams["font.family"])

# RTL (Arabic/Hebrew) shaping. matplotlib built with libraqm shapes natively, so raw text is correct.
# Only when libraqm is absent (some Linux/Colab) do we shape manually with arabic-reshaper + python-bidi (determined by the binary, not the version).
RESHAPE_RTL = "auto"   # "auto"=decide automatically from libraqm presence / True=always manual / False=no shaping
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
# Unified color codes for language groups (for coloring the figures. Romance/Germanic are genealogical, but CJK/Viet/Thai mix region, script, and individual languages and are not necessarily genealogical). Shared across the later figures (heatmap, dendrogram, rotation plots).
# The text inside the figures is in English; so are the prose explanations.
# We define the language group of every candidate language in MUSE en-XX (44). So that changing the target languages does not break anything,
#   unknown languages fall back to "Other" (gray) in fam_of()/fam_color().
CANON_FAM = {
    "CJK": "#e41a1c", "Romance": "#ff7f00", "Germanic": "#f781bf", "Slavic": "#377eb8",
    "Semitic": "#4daf4a", "Iranian": "#a65628", "Turkic": "#984ea3", "Viet": "#999999",
    "Thai": "#bcbd22", "Austronesian": "#00ced1", "Uralic": "#1b9e77", "Baltic": "#8c6d31",
    "Albanian": "#7570b3", "Hellenic": "#66a61e", "Indic": "#e6ab02", "Dravidian": "#e7298a",
    "Other": "#cccccc",
}
# language code -> language group (covers all 44 MUSE en-XX candidates)
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
    """language code -> language group. Unknown languages map to "Other" (so changing the target languages does not break anything)."""
    return FAM.get(L, "Other")


def fam_color(L):
    """language code -> language group color (unified palette). Unknown = gray."""
    return CANON_FAM.get(fam_of(L), "#cccccc")

# %%
# Download and read only "the shard containing embed_tokens" of the embedding matrix W_E (do not load the model itself)
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

MODEL_ID = "Qwen/Qwen3-4B"
# Which shard holds the embeddings is written in the index file
weight_map = json.loads(open(hf_hub_download(MODEL_ID, "model.safetensors.index.json")).read())["weight_map"]
shard = hf_hub_download(MODEL_ID, weight_map["model.embed_tokens.weight"])   # if not yet fetched, download only this one file
with safe_open(shard, framework="pt") as f:
    W_E = f.get_tensor("model.embed_tokens.weight").float().numpy()          # shape: 151936 x 2560
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print("W_E:", W_E.shape, "  vocab size x dim")

_tok1_cache = {}


def tok1(s):
    """If string s becomes "exactly 1 token", return its id. Otherwise return None.
    For English words, prefer a leading space (Qwen's convention); for CJK, try the bare character."""
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
    """Normalize a vector to length 1 (to look only at its direction)."""
    return v / (np.linalg.norm(v) + 1e-12)


def e(t):
    """Embedding vector for token id t (already normalized to length 1)."""
    return u(W_E[t].astype(np.float64))


# %% [markdown]
# ## Part 1  Clustering the languages
#
# **Goal**: measure how close languages are to one another and, with hierarchical clustering, see which languages form the same cluster.
# In brief, English clusters not with the Germanic languages but with Chinese, Japanese, and Korean (CJK).
# This cannot be explained by the languages' genealogy. We surmise it reflects Chinese and English being the main languages in Qwen's
# training data (but since the training corpus is not public, the specific mechanism is not firmly confirmed, and the effects of the English-pivot dictionary and the selection criteria cannot be separated out either).

# %% [markdown]
# ### 1.1 The English-pivot MUSE bilingual dictionaries
#
# For "same-meaning word pairs" between languages we use the **MUSE bilingual dictionaries**. MUSE is a multilingual word-embedding library
# released by Facebook AI Research, which ships with **bilingual dictionaries** for evaluation and training (this project uses only these dictionaries).
#
# - Paper: Conneau, Lample, Ranzato, Denoyer, Jégou, [*Word Translation Without Parallel Data*](https://arxiv.org/abs/1710.04087), ICLR 2018.
# - Upstream repository: [github.com/facebookresearch/MUSE](https://github.com/facebookresearch/MUSE)
# - Distribution: `https://dl.fbaipublicfiles.com/arrival/dictionaries/en-{XX}.txt` (fetch each language file directly; directory listing is not available). Starting from English there are `en-XX` dictionaries for **44 languages** (excluding English itself; we use all of them in 1.2 below).
#
# **Important prerequisite**: the MUSE dictionaries are **all anchored on English** (`en-XX` = English → each language).
# There is no direct dictionary that bypasses English, such as Japanese↔Chinese. So even when comparing Japanese and Chinese, the shared concept is defined via English
# (an English word listed in both `en-ja` and `en-zh` is treated as a concept shared by Japanese and Chinese). From this en-XX structure, in this notebook we
# **represent a concept by a single English word $w$** (concept $w$ is represented by its translation in each language). The concept variable is consistently the English word $w$ throughout all parts.
# In other words, **English is structurally central by construction**. Later a result of "English is the hub" appears, but
# part of that is also a consequence of this English pivot. Keep this point in mind when reading the results.

# %% [markdown]
# ### 1.2 Selecting target languages by whether Qwen tokenizes them as a single token
#
# We download all **44 en-XX languages** produced by MUSE (excluding English itself) and select the target languages from them **mechanically, by a single quantitative criterion**.
# The criterion is "the language has **at least 100** content-word translation pairs whose translation becomes a **single token** in Qwen, within the mid-frequency band (§1.3)"
# (so that each word has a unique, well-defined vector and can be compared). This criterion is determined by the tokenizer alone and does not depend on the downstream conclusion (language closeness).
# The next bar chart shows which languages are retained or excluded, and why.

# %%
# Prepare the MUSE bilingual dictionaries (download only on first run).
# Note: this is the cache for the MUSE dictionaries, and is a [separate thing] from the Hugging Face model cache (~/.cache/huggingface).
#   Saved to ~/.cache/muse_full/. 44 languages excluding en itself, ~50MB total.
MUSE_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-{}.txt"
MUSE_DIR = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "muse_full"
MUSE_DIR.mkdir(parents=True, exist_ok=True)
# The full en-XX list from the MUSE upstream README (excluding en itself) = 44 languages
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

# Function words (exclusion set so we use only content words)
STOP = set("a an the this that these those and or but if then else for nor so yet of to in on at by with from up down out off over under again about into as is are was were be been being am do does did have has had having i you he she it we they me him her us them my your his its our their not no yes very can will just should would could may might must shall here there where when why how what who whom which than too also more most some any all each every both few many much other another such own same then once".split())
BAND_LO, BAND_HI = 2000, 12000    # mid-frequency band (rationale in 1.3)


def n_single_token_pairs(L):
    """For language L, the number of translation pairs where "both en and the translation are 1 token, content words, distinct tokens, within band" (the selection criterion)."""
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


MIN_PAIRS = 100    # drop languages below this count (it sits within a natural cliff, as the figure below shows)
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
# **What the figure shows**: each bar is one MUSE language's "count of within-band, single-token content-word translation pairs" (log scale). Green = kept (threshold 100 or more),
# red = dropped, black dashed line = threshold 100. **Observed distribution**: of the 44 candidates, 38 are green (kept) and 6 are red (dropped). Among the red, fi(92) and lt(40) are
# just below the threshold, while el(2), hi(2), ta(1), bn(0) are nearly zero, with a large step between these 4 languages and the rest.

# %% [markdown]
# **Interpretation**: for el, hi, ta, bn near zero, under this run's MUSE translations and filter conditions, words in their writing systems (Greek / Devanagari / Tamil / Bengali)
# almost never become a single token in Qwen, so almost no pairs can be formed; we read this as a **tokenizer footprint** (there is a large step between ≤2 and the rest, so dropping these 4 languages is
# robust to the choice of threshold). fi and lt, on the other hand, can be single-tokenized but have few within-band pairs, so they drop out additionally at the threshold of 100
# (this boundary is a softer cut that depends on the threshold). We count within the band so that the subsequent language similarity $M$ is evaluated under the same selection criteria.

# %% [markdown]
# ### 1.3 Selecting the token frequency band
#
# The previous section kept the 38 languages that have enough translation pairs within this band (the 38 green languages in the figure; adding English makes
# 39 languages). When counting the translation pairs we already used, as a condition, that the English token's id lies in this mid-frequency band $2000 \le \mathrm{id} < 12000$.
# The id is roughly in frequency order (smaller = higher frequency): the highest-frequency words are often confusing loanwords whose spelling collides with
# English, and the low-frequency words are noisy. The mid-frequency range, excluding both ends, cleanly reflects words with clear meaning. The concept selection and
# language similarity below are also carried out within this band.
#
# The closeness of two languages $i, j$ is measured by the mean cosine $m_{ij}$ of the translation vectors over the concepts the two share (precise definition in 1.5).
# This band is not a single point chosen to fit the conclusion. To confirm that, we sweep both ends of the band widely and watch how the profile of
# $m_{ij}$ over all $\binom{39}{2}=741$ language pairs moves.

# %%
_uc = {}
def _uvec(t):
    v = _uc.get(t)
    if v is None:
        w = W_E[t].astype(np.float64); v = w / (np.linalg.norm(w) + 1e-12); _uc[t] = v
    return v

# Band-free concept pool (keep the English token id = frequency proxy; the band is re-filtered later)
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
_ENZH = (0, _ix["zh"])   # en is index 0


def _pair_full(lo, hi):
    """Dict of mean cos over all language pairs in the band [lo,hi): {(i,j): m_ij}."""
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
    """Fix one edge and sweep the other over vals. Return each pair's raw cos and z-score series (missing = NaN)."""
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


_HIs = [4000, 6000, 8000, BAND_HI, 16000, 24000, 40000, 60000, 90000, 150000]   # LO=2000 fixed
_LOs = [0, 1000, BAND_LO, 3000, 4000, 6000, 8000, 9000, 10000, 11000]           # HI=12000 fixed
_rawH, _zH, _kH = _sweep(BAND_LO, _HIs, True)
_rawL, _zL, _kL = _sweep(BAND_HI, _LOs, False)


def _panel(ax, series, keys, xpos, xlabels, adopt_pos, ylabel, xlabel):
    span = max(xpos) - min(xpos)
    for k in keys:                                   # all 741 pairs as thin lines (missing = gap)
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


_xH = [np.log10(x) for x in _HIs]; _lbH = [str(x) for x in _HIs]      # right column: log
_xL = list(_LOs); _lbL = [str(x) for x in _LOs]                        # left column: linear
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
# **What the figure shows**: each thin gray line is the profile of one of the $\binom{39}{2}=741$ language pairs among the 39 languages (English + the 38 target languages). The top row is the raw mean cosine $m_{ij}$, the bottom row is its
# z-score $(m_{ij}-\mu)/\sigma$ standardized over the all-pairs distribution. The left column sweeps the lower edge LO (upper edge fixed at 12000, linear axis), the right column
# sweeps the upper edge HI (lower edge fixed at 2000, log axis). Red = en–zh, blue/green/purple = the next 3 highest pairs, dashed line = the adopted 2000 / 12000.
# **Observed configuration**: en–zh (red) is the top envelope in every band, with a raw cosine of about 0.50 and a z-score of about 4. Widening the upper edge lowers
# the raw cosine gently, and moving the lower edge toward the upper edge exhausts the band's concepts so the red line sinks into the gray bundle. The dashed lines of the adopted band 2000 / 12000
# lie inside this flat plateau.

# %% [markdown]
# **Interpretation**: en–zh is the strongest bond among all pairs, standing out by about 4σ, and neither its rank nor its prominence depends on how the band's edges are taken.
# Therefore $2000 \le \mathrm{id} < 12000$ is not a single point chosen to manufacture the conclusion but merely one point inside a broadly robust range.
# Swinging the edges to the extremes, on the other hand, breaks down at the upper end through dilution by low-frequency words and at the lower end through exhaustion of concepts, so keeping to the
# mid-frequency band has an objective justification.

# %% [markdown]
# ### 1.4 The concept selection procedure
#
# From each kept language's dictionary, we collect "concepts" (an English word plus its translation in each language) that satisfy the following conditions.
#
# 1. Select **content-word candidates** with a simple stopword / character-type / length filter (exclude function words like `the` `and`, symbols, and words of 2 characters or fewer).
# 2. Both the English word and the translation become **a single token in Qwen** (so that each word has a unique, well-defined vector).
# 3. The English token's id lies in the **mid-frequency band 2000-12000** (the rationale for choosing this frequency band is §1.3).

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
        if et is None or tt is None or et == tt or (et, tt) in seen:   # et==tt: exclude identical tokens (incl. cognates / shared scripts / loanwords)
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
# ### 1.5 Building and inspecting the language similarity matrix $M$
#
# **Definition of the word vector (used throughout from here on)**. Write the **language-$L$ translation token** of concept $w$ (= an English word) as $t_L(w)$, and take its
# **length-1 normalized embedding** as the word vector:
# $$ v_L(w) \;:=\; e\big(t_L(w)\big) \;=\; \frac{W_E[\,t_L(w)\,]}{\lVert W_E[\,t_L(w)\,]\rVert} \;\in\; \mathbb{R}^{2560}. $$
# English is one language too: the English token of concept $w$ is $t_\text{en}(w)$, and its word vector is $v_\text{en}(w)=e(t_\text{en}(w))$.
# All of $W_E$ is a token embedding (subwords included), but in this notebook we select only words that become a single token, so we call their vectors **word vectors**.
# This $v_L(w)$ is the central object of analysis, and it is the same object as the left-hand side of the Part 3 model $v_L(w)=R(L)R(w)v_o$
# (from Part 2.1 on, where rotations are handled, we work in a 128-dimensional PCA subspace; §3.2 gives the bridge).
#
# Let $D_L, D_{L'}$ be the sets of concepts for which languages $L, L'$ have a translation. We measure the closeness of two languages by the mean cosine similarity over
# **concepts for which both languages have translations**, $D_L \cap D_{L'}$:
# $$ m_{LL'} = \operatorname{mean}_{\,w \,\in\, D_L \cap D_{L'},\ t_L(w)\neq t_{L'}(w)}\ \cos\!\big(v_L(w),\, v_{L'}(w)\big). $$
# Since we measure with $\cos$, normalizing $v_L(w)$ does not affect the value, but we standardize on the normalized version to match the later $v_L(w)$.
# $t_L(w)=t_{L'}(w)$ (e.g., the same kanji in Japanese and Chinese) is trivially $\cos = 1$. Such pairs simply share the same token; including them is not inherently invalid (in principle it could be
# included), but to avoid artificially inflating the similarity among CJK languages we exclude it here. Arranging this for all pairs gives the matrix
# $M=(m_{ij})$ ($i,j$ are language indices; the entries are the $m_{LL'}$ above, and the diagonal is $m_{ii}=1$). Below we display $M$ reordered by the order obtained from
# hierarchical clustering (1.6). Since similar languages come next to each other, the clusters are easy to see.

# %%
LANGS = ["en"] + ALL_LANGS
n = len(LANGS); idx = {L: i for i, L in enumerate(LANGS)}
acc = [[[] for _ in range(n)] for _ in range(n)]
for c in concept.values():
    Ls = list(c["toks"])
    for a in range(len(Ls)):
        for b in range(a + 1, len(Ls)):
            li, lj = Ls[a], Ls[b]                   # li, lj = language i, language j
            if c["toks"][li] == c["toks"][lj]:      # identical token (e.g., same kanji in Japanese/Chinese) is cos=1, so exclude
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

# Precompute hierarchical clustering (Ward linkage) for reordering (figure and explanation in 1.6). Distance is sqrt(1 - m_ij) (Euclidean distance of unit vectors; matches Ward's assumption).
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform
Dist = np.sqrt(np.clip(1.0 - M, 0, None))
np.fill_diagonal(Dist, 0.0)
Dist = np.nan_to_num(Dist, nan=float(np.nanmax(Dist)))
Dist = 0.5 * (Dist + Dist.T)
Zw = linkage(squareform(Dist, checks=False), method="ward", optimal_ordering=True)
order_w = list(leaves_list(Zw))

# language group legend handles (shared by the heatmap and dendrogram)
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
# **What the figure shows**: each cell is the mean cosine similarity $m_{ij}$ between two languages (darker red = larger; the diagonal is trivial, so masked in gray). Rows and columns are
# ordered by the Ward clustering of 1.6 (similar languages adjacent), and the labels are colored by language group.
# **Observed layout**: ko-ja-en-zh form a dark red block, and the strongest cell is en–zh ($m=0.500$). Next to it a
# diagonal block of Romance (it, es, pt, fr) follows. The English row and column contain relatively high similarity values across many language groups.

# %% [markdown]
# **Interpretation**: English is broadly similar not only to one language group but across many, suggesting a hub-like position in this
# English-pivot data. That CJK and English form the darkest block provides a matrix-level view of the same English–CJK clustering pattern
# seen in the dendrogram.

# %% [markdown]
# ### 1.6 Confirming the families with hierarchical clustering (Ward linkage)
#
# To reorder the matrix $M$ above, we used **hierarchical clustering** (Ward linkage).
# For Ward linkage we use the distance $d_{ij}=\sqrt{1 - m_{ij}}$. This distance can be written as a Euclidean distance when the average is taken over the same set of concepts, but in this experiment the set of shared concepts differs for each language pair, so there is no strict guarantee.
# Ward linkage minimizes variance assuming Euclidean distance, so we match it here (using $1 - m_{ij}$ directly does not change the conclusions below).
# The procedure merges the closest languages in order, and rendering the result as a tree (dendrogram) shows which languages merge early.
# The color of a leaf label is its language group. The point of interest is that **English departs from its own Germanic group (de, nl, sv, da, no, af) and joins the CJK (zh, ja, ko) cluster**.

# %%
def plot_dendro_ward():
    from collections import Counter
    # link (branch) color = the color of the "majority language group" among the leaves under that branch (e.g., red if CJK is the majority).
    # Higher trunks that span language groups (above the threshold) are neutral gray, to keep the figure objective (no assertions like arrows).
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
# **What the figure shows**: leaf = language (language group color), vertical axis = Ward linkage distance (lower = merges earlier = closer). Branch colors are the majority
# language group under that branch; higher trunks spanning language groups are gray. **Observed layout**: English is adjacent to Chinese in the leaf order, and English falls **inside the CJK (ko, ja, zh) cluster**,
# separated from its own Germanic group (de, nl, af, sv, da, no on the right). Romance, Slavic, and the others mostly cluster by family.

# %% [markdown]
# **Interpretation**: that English merges early with CJK rather than with its family (Germanic) cannot be explained by the languages' genealogy, and
# as stated at the outset, we surmise it reflects the language composition of the training data. The peripheral low-resource languages have few shared concepts and
# unstable numbers, so we do not chase the fine-grained ordering.

# %% [markdown]
# ### 1.7 Which language is the hub
#
# If we define centrality as "the mean cosine similarity to all other languages," a **selection bias** arises that pushes small-dictionary languages to the top
# (small-dictionary languages are biased toward easy high-frequency concepts, inflating their values). So we look at the mean over only the **nearest top $k$ languages** for each language
# (top-k nearest-neighbor similarity). This one is robust (adding distant languages barely changes it) and is not swayed by the choice of languages.

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
# **Conclusion**: in top-k nearest-neighbor similarity, from $k=1$ to $5$, **English is consistently rank 1** (at $k=1$, English and Chinese tie at 0.500 as the strongest).
# It is followed by Chinese, Korean, Japanese (CJK) and Spanish, Portuguese (Romance).
# Even with many European languages included, English–Chinese stays the highest over the examined range $k=1,\ldots,5$; over this set of 38 languages this closeness is stably the highest in the range $k=1,\ldots,5$,
# and is consistent with the training-data composition (though the effects of the English-pivot dictionary and the selection criteria are not separated out).
# Overall mean similarity, by contrast, has a selection bias that pushes small-dictionary languages (e.g., ko, bg) to the top, so we do not adopt it.

# %% [markdown]
# ## Part 2  Selecting languages and viewing the raw embeddings
#
# From here we look at "how the multilingual correspondence is represented inside the embeddings." First we **select the languages to use for the rotation**,
# then we **visualize the raw embeddings with t-SNE** (the central figure, paired with Part 3's after-transform figure). The theory and detailed procedure of the rotation are collected in Part 3 (theory).

# %% [markdown]
# ### 2.1 Selecting the languages used for the rotation
#
# Later, for each language, we estimate "the rotation $R(L)$ from the English side into that language" from the translation pairs (details in Part 3).
# However, a language with few usable translation pairs **fits the pairs used for estimation well but does not fit the pairs not used** (overfitting).
# So for each language we **split the translation pairs into train / valid and compare the fits**, and use for the rotation only the languages whose difference **gap = train − valid** is small
# (= little overfitting). The split randomly divides each language's translation pairs into **valid 30% (at least 5 pairs) / train 70%**,
# **estimates $R(L)$ on train only**, and measures the alignment (cosine) on both train and the held-out valid.
# We change the random seed and average over **5 runs** for tr / va, and exclude languages with **fewer than 15 translation pairs** since they cannot be split stably.
# This section performs the computationally intensive steps (PCA reduction to 128 dimensions and estimation of $R(L)$) before selecting the languages. **What is actually being done is explained in Part 3.**

# %%
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

ALIGN_THR = 0.25       # translation-quality filter (cos between English and its translation must be at least this): stabilizes rotation estimation
K = 128              # dimension of the working space in which the rotation is found (reduced by PCA; rationale in Part 3)
GAP_CUT = 0.30       # use languages whose gap is at most this for the rotation
M_MIN = 3            # for display and metrics, use concepts with translations in at least 3 non-English languages
VALID_FRAC = 0.30    # valid fraction of the train/valid split (language selection in 2.1)
N_SPLIT = 5          # number of times to redo the split and average (2.1)
MIN_VALID = 5        # minimum number of valid pairs (2.1)
MIN_FIT = 15         # languages with fewer pairs than this are excluded without splitting (2.1)

# translation pairs for rotation estimation (translations that align well with English = filter5). Reuse the concept from Part 1 (within band, single token).
fitpairs = {}
for L in ALL_LANGS:
    fitpairs[L] = [(c["toks"]["en"], c["toks"][L]) for c in concept.values()
                   if "en" in c["toks"] and L in c["toks"] and c["toks"]["en"] != c["toks"][L]
                   and float(e(c["toks"]["en"]) @ e(c["toks"][L])) >= ALIGN_THR]

# working-space PCA-128 (all tokens appearing in the concepts). We measure the rotation in this space hereafter (details in Part 3).
_allids = sorted({c["toks"][L] for c in concept.values() for L in c["toks"]})
pca = PCA(n_components=K, random_state=0).fit(np.array([e(t) for t in _allids]))
_pc = {}


def ep(t):
    """Map token t into 128 dimensions and normalize to length 1."""
    if t not in _pc:
        _pc[t] = u(pca.transform(e(t)[None])[0])
    return _pc[t]


def _align(pairs, R):
    return float(np.mean([float(u(ep(a) @ R) @ ep(b)) for a, b in pairs])) if pairs else float("nan")


# For each language, split the fit pairs into train/valid (averaged over 5 runs) and measure the fit gap of R(L)
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
# **What the figure shows**: for each language, the fit (cosine) of the rotation $R(L)$, shown on the pairs used for estimation (train, ●) and the pairs not used
# (valid, ✕). Languages are ordered by the vertical gap between the two lines = **gap** (amount of overfitting), smallest first; left of the dashed line (gap $\le 0.30$) is kept,
# right (light red) is dropped. **Observed layout**: on the left, train and valid nearly overlap; the further right, the more valid drops and the gap widens.
# 14 languages satisfy gap $\le 0.30$.

# %% [markdown]
# **Interpretation**: the drop in validation relative to training can be interpreted as the 128-dimensional $R(L)$ overfitting in languages with few translation pairs.
# So we use for the rotation only the 14 languages with a relatively small train–validation gap (little overfitting). Why they overfit and how $R(L)$ is estimated is explained in Part 3.

# %% [markdown]
# **What was selected**: **14 languages** satisfied this criterion (gap $\le 0.30$), and adding English gives **15 languages** used for the rotation.
# The 14 selected languages are as follows (the print above also shows each language's gap):
#
# - **CJK**: Chinese (zh), Japanese (ja), Korean (ko)
# - **Romance**: Spanish (es), Portuguese (pt), French (fr), Italian (it)
# - **Slavic**: Russian (ru), Bulgarian (bg)
# - **Semitic**: Arabic (ar), Hebrew (he)
# - **Germanic**: German (de) / **Turkic**: Turkish (tr) / **Viet**: Vietnamese (vi)
#
# These broadly cover the major language groups. The smallest gap is for **Chinese** (train and valid fits are nearly equal), and
# in general **the more translation pairs a language has, the smaller its gap**. Conversely, Part 1 could use 38 languages for clustering while the rotation shrinks to 14, because
# "closeness between languages (clustering)" can be measured with just the mean cosine similarity of each pair, whereas the "rotation $R(L)$" must **estimate** a 128-dimensional transform per language
# from the translation pairs, and languages with few pairs overfit (this is the reason the number of usable languages differs between Part 1 and Part 2).

# %% [markdown]
# ### 2.2 en→L transfer performance differs slightly from raw similarity
#
# In the concept-first model (Part 3), taking English as the reference gives $v_L(w)=R(L)\,v_\text{en}(w)$. That is, the transfer from en to language $L$ is
# **a single rotation $R(L)$** itself. So "how well the en→L transfer works" is directly measured by the **held-out valid alignment** used above for language selection
# (how well $R(L)v_\text{en}(w)$ matches $v_L(w)$ on unseen concepts). We place this alongside the **raw similarity** $m_{en,L}$ from Part 1.5 (the M-matrix row, no rotation) and see how well the two agree.

# %%
# Compare the transfer performance (held-out valid alignment, gap_rows above) with the raw en similarity m_{en,L} (the M row from 1.5).
from scipy.stats import pearsonr, spearmanr
va_d = {r[0]: r[3] for r in gap_rows}                       # held-out valid alignment = en→L transfer score
_xs = [float(M[idx["en"], idx[L]]) for L in LANGS_ROT]      # raw similarity m_{en,L} (raw 2560-dim)
_ys = [va_d[L] for L in LANGS_ROT]                          # transfer valid (PCA-128 held-out)
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
# **What the figure shows**: each point is one rotation language. Horizontal axis = raw cosine to English $m_{en,L}$ (the M row from Part 1.5, no rotation, 2560-dim),
# vertical axis = held-out valid alignment (the transfer score measured on pairs not used to estimate $R(L)$, PCA-128). Color is language group.
# **Observed layout**: a strong positive correlation overall (Spearman $\rho\approx0.87$) but not perfect, with **rank swaps at the top**.
# **Chinese (zh) is closest to English in the raw space** (largest $m_{en,L}$) but ranks 3rd in transfer. **Spanish and Portuguese (Romance), though not
# that close in the raw space**, stand at ranks 1 and 2 in transfer.
#
# **Interpretation**: "raw closeness" and "transferability by rotation" are different things. The English-Chinese closeness in Part 1 is about the *raw layout*,
# reflecting that Chinese tokens are placed (in the learned embedding space) near English. That Romance ranks high in transfer, on the other hand, can be read as: **rotating
# the English layout tends to yield Romance**, i.e. the en→Romance correspondence is well represented by a single rotation ("the mapping is rotational," not "the meaning is close").
#
# **How to read the numbers (caution)**: the **absolute value** of valid (held-out alignment) depends on the working space (PCA-128) and the pair selection ($\ge$ ALIGN_THR), so
# we do not interpret the absolute level. We read only the **ranking among languages** and the **deviation from raw similarity**. Since the dictionary is en-X (English pivot),
# "transfer from English" is structurally favored, so the values are used only for relative comparison among languages. The scope is limited to the 14 languages with gap $\le 0.30$.

# %% [markdown]
# ### 2.3 Viewing the raw embeddings
#
# For the selected languages plus English, we look at the **raw embeddings** (vectors with no transform applied at all) reduced to 2D (t-SNE).
# Each concept is drawn as a "star" with the English token at the center and each language's translation connected by lines (branch = English → translation).
# The aim is to visually check what comes close together (same meaning, or same language).
#
# The number of concepts displayed can be changed freely via **`N_DISPLAY`** below (more is busier and harder to read, fewer is cleaner).
# Changing **`SEED`** re-selects the set of displayed concepts with a different random draw, producing a different picture.

# %%
from sklearn.manifold import TSNE

# per-language color = unified color code (the language group color from CANON_FAM). English = black (hub).
# Languages of the same language group share a color (e.g., CJK's zh/ja/ko are red). Individual languages are distinguished by the translation label on each point.
LANG_COLOR = {L: fam_color(L) for L in LANGS_ROT}
LANG_COLOR["en"] = "#1a1a1a"  # black (hub color equivalent to (0.1, 0.1, 0.1))

# concept pool usable for display and metrics (rotation languages, band, single token, no translation-quality filter = avoids the "pick only close words" cheat in the figure)
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
# Randomly select the concepts to display (this cell fixes the selection; drawing is in the next cell. Reproducible since the seed is fixed).
SEED = 0            # seed for the random selection of displayed concepts (changing it selects different concepts and a different picture)
N_DISPLAY = 48      # number of concepts displayed in the t-SNE
rng = np.random.default_rng(SEED)
sub = [draw[i] for i in sorted(rng.choice(len(draw), size=min(N_DISPLAY, len(draw)), replace=False))]

from collections import Counter
print(f"concepts displayed in the t-SNE: {len(sub)} (drawn from a pool of {len(draw)} with SEED={SEED})")
print("  distribution of linked language counts: " + ", ".join(f"{k} languages={v} concepts" for k, v in sorted(Counter(len(c['langs']) for c in sub).items(), reverse=True)))
print("  concepts linked to many languages (many star branches) top 6:")
for c in sorted(sub, key=lambda c: -len(c["langs"]))[:6]:
    print(f"    {c['en_word']:<12}({len(c['langs'])} languages): " + ", ".join(f"{L}={c['langs'][L][1]}" for L in c["langs"]))

# %% [markdown]
# **How to read the print above**: for each selected concept, we show how many of the rotation languages have a linked translation (= the number of star branches).
# The more branches a concept has, the larger the resulting star in the figure below, extending from the center to many languages. Which ones are selected is determined by the `SEED` random draw, so
# check the specific concepts in this print (changing `SEED` selects different concepts). The next cell draws this concept set with t-SNE.

# %%
# t-SNE drawing (draw the sub selected above; save at high resolution dpi=300)
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
# **What the figure shows**: each point is the raw embedding vector of a (displayed concept × language) translation token, reduced to 2D by t-SNE with cosine distance.
# English (black = hub) is at the center, and each language's translation of the same concept is connected by thin gray lines (star branch = English → translation). Point color is the language group
# (CANON_FAM), and the translation label on each point lets individual words be read. **Observed layout**: translations of the same meaning (= points of different languages
# belonging to the same star) come close to each other, while no region is found where only the same color (same language group) gathers; the colors mix within each star.
# For some concepts a branch extends far.

# %% [markdown]
# **Interpretation**: what determines closeness is meaning rather than language, so the raw embedding clusters by concept (a concept-dominated representation).
# A long branch may reflect semantic divergence from the English word, for example because of polysemy. In Part 3 we see that when we suppress the concept
# component of this concept-dominated raw embedding with the inverse concept transform $C_L^{-1}(w)$, the hidden language structure becomes apparent (words cluster by language).

# %% [markdown]
# ## Part 3  Multilingual correspondence as rotation
#
# Here we understand "the raw space clusters by concept," seen in Part 2, in terms of **rotation**, and show that applying the inverse concept transform
# $C_L^{-1}(w)$ **specific to each language and each concept** lowers concept-dependent similarity and relatively raises within-language similarity
# (suppressing the concept component surfaces the language structure). The detailed procedure is also explained here.
#
# **Terminology**: throughout this notebook (including the "rotation" at the outset), for clarity we consistently say "**rotation**" and "**rotation matrix**," but strictly these are
# **orthogonal transforms** and **orthogonal matrices** that also include reflections (the $R$ found by Procrustes is likewise an orthogonal matrix, with determinant $\pm 1$).

# %% [markdown]
# ### 3.0 Suppressing the concept component surfaces the language structure
#
# In the raw token embeddings, translation-equivalent words of the same meaning are close, and the concept structure looks dominant. The conjugate inverse concept transform
# $C_L^{-1}(w)=R(L)R(w)^{-1}R(L)^{-1}$ is a transform intended to cancel the effect of concept $w$ in the coordinate system of language $L$.
# As a result, words of the same language become relatively closer, and the language structure that was present in the original embedding becomes easier to see.
#
# The figure below is a schematic of this reading. It shows the flow: raw embeddings on the left, the conjugate inverse concept transform in the middle, and on the right the base position
# of the same language $\hat{x}_L=R(L)v_o$ becoming visible after the transform.

# %%
def plot_concept_language_schematic():
    # Wide aspect with tight top/bottom, and larger text/lines/markers. Density that stays readable even when shrunk.
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

    # ── left panel: per-concept blobs (same marker shape = concept, 4 colors = language) ───────────
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

    # ── middle panel: 3-stage transform (numbers vertically centered) ──────────────────────────
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

    # ── right panel: per-language blobs (same color = language, 3 concepts gather near the base point x̂_L) ──
    # push fr/ja (ellipse, markers, label) downward (away from the upper ellipse, using the bottom margin).
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

    # ── large arrows between panels (thick and big) ─────────────────────────────────────
    def arrow_between(left_key, right_key):
        lx, ly, lw, lh = panels[left_key]
        rx, ry, rw, rh = panels[right_key]
        ymid = ly + lh * 0.50
        ax.add_patch(FancyArrowPatch((lx + lw + 0.012, ymid), (rx - 0.012, ymid),
                                     arrowstyle="-|>", mutation_scale=42,
                                     linewidth=4.4, color="#57606a", zorder=5))

    arrow_between("raw", "transform")
    arrow_between("transform", "after")

    # ── legend ───────────────────────────────────────────────────────────────
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
# ### 3.1 The concept-first model
#
# We think of the word vector $v_L(w)$ defined in Part 1.5 (the normalized embedding of the language-$L$ translation token of concept $w$; English is $v_\text{en}(w)$) as being
# built in the working space (the PCA-128 of 3.2) as follows:
# $$ v_L(w) = R(L)\, R(w)\, v_o. $$
# - $v_o$: the reference direction (the mean direction of the English concept vectors).
# - $R(w)$: the per-concept orthogonal transform. In this notebook we define it as the **minimal plane rotation** that maps the reference direction $v_o$ to the English concept vector
#   $v_\text{en}(w)$ and acts as the identity on the orthogonal complement of the 2D plane the two span (implemented later by `rinv`). "Independent of language" is not an empirical finding but
#   a **modeling assumption** imposed through its construction on the English side.
# - $R(L)$: the per-language orthogonal transform. It is the rotation that **aligns the English-side vector layout to the language-$L$ side**, estimated from the translation pairs (English is the reference = $R(\text{en})=I$).
#   There are not separate spaces; it is an alignment transform within a single shared embedding.
#
# English has no rotation, so $v_\text{en}(w) = R(w)\,v_o$. That is, each word lies where the common base point $v_o$ has been rotated in two stages, by the "concept rotation" and the
# "language rotation." This is a construction that applies the concept rotation first and the language rotation afterward, and it is what we called the **concept-first model** at the outset.
# If this concept-first model is correct, then **canceling the language rotation $R(L)$ should leave only the concept, and
# canceling the concept rotation $R(w)$ should leave only the language**.
#
# **Implication for language transfer (the motivation from the outset)**: if the concept-first model is correct, the word vector of language $L$ can be moved to that of language $L'$
# by a single transform independent of concept $w$. Indeed,
# $$ v_{L'}(w) = R(L')\,R(w)\,v_o = R(L')\,R(L)^{-1}\,R(L)\,R(w)\,v_o = R(L')\,R(L)^{-1}\,v_L(w) $$
# and the transform $R(L')R(L)^{-1}$ does not contain concept $w$. Knowledge obtained in one language can be carried to another without rebuilding it for each meaning. This is the
# "structure for efficiently carrying knowledge across languages" stated at the outset. The same concept-independent transfer is also possible for the additive model with a constant offset, but
# the language-first model requires a different transform for each concept.


# %% [markdown]
# ### 3.2 The 128-dimensional working space by PCA
#
# The rotation $R(L)$ is estimated from the translation pairs (next section). But estimating it at the original 2560 dimensions fails to fit words not used for estimation
# (overfitting). So we handle the rotation in the **working space reduced to 128 dimensions by PCA** (the space already prepared in 2.1; in the code `ep(t)` is the 128-dimensional version of $e(t)$).
# **Hereafter, the word vector $v_L(w)$ refers to its value in these 128 dimensions** (the symbol is unchanged; only the space switches from 2560 to 128. The contrast with the raw 2560 dimensions is made in 3.5).
# Sweeping the dimension $K$ as 2560→512→…→32 and measuring the language separation, **around 128 is best**; at full dimension the within-language pair similarity does not exceed the
# translation-pair similarity even after the transform, and the surfacing of the language structure cannot be confirmed. That is why we use 128.

# %% [markdown]
# ### 3.3 Estimating the language rotation $R(L)$ from the translation pairs
#
# Let the set of concepts used for estimating the rotation be $D_\text{fit}$ (a fitting set that **does not include the concepts used for display or evaluation**,
# to avoid estimating $R(L)$ and evaluating it on the same concepts).
# For each concept $w$ in $D_\text{fit}$, let $A$ be the matrix with the English vectors $v_\text{en}(w)$ **arranged side by side as columns**,
# and $B$ the matrix with the corresponding language-$L$ vectors $v_L(w)$ arranged in the same order (both $128 \times |D_\text{fit}|$, **each column being one concept**).
# Writing the model of 3.1, $v_L(w) = R(L)\,v_\text{en}(w)$ (multiply $R(L)$ from the left, English → language $L$), for all concepts together gives
# $$ R(L)\,A \approx B $$
# We find the **orthogonal matrix** $R(L)$ satisfying this by least squares
# (**orthogonal Procrustes** = the standard method for finding the rotation/reflection that best overlaps two sets of corresponding points). English itself has no rotation (reference $R(\text{en})=I$).
# That we use only the 14 well-fitting (non-overfitting) languages was confirmed in 2.1. Here we actually estimate $R(L)$ for those 14 languages.

# %%
# ── Correspondence between the code world ↔ the math (3.3) ─────────────────────────────────────────
# For implementation convenience this code computes in a "row-vector convention" (stacking each concept vector as a row). This corresponds to the
# world of the math of 3.3 (the column convention R(L)·A ≈ B, stacking each concept as a column) transposed wholesale:
#   - All matrices are the transpose of the math:  code A = math A^T,  code B = math B^T (each row one concept, |D_fit|×128).
#   - scipy.orthogonal_procrustes(A, B) returns the orthogonal matrix Ω satisfying A·Ω ≈ B. Recast to the column convention, Ω^T·A ≈ B,
#     i.e., the math's R(L) = Ω^T. So what is stored in R[L] is Ω = R(L)^T (= the transpose of the math R(L)).
#   - Action on a row vector v (1×128):  v @ R[L] = the column convention's R(L)·v,   v @ R[L].T = R(L)^T·v (= R(L)^-1·v).
# The numbers agree regardless of convention (figures and metrics are the same). The code hereafter is consistent in this row convention.
# ──────────────────────────────────────────────────────────────────────────────────────
# R(L) estimation: reuse the working space ep from 2.1 and the translation pairs fitpairs. Orthogonal Procrustes on held-out pairs excluding the display concepts (draw).
draw_en_tok = {c["en"] for c in draw}
R = {}
for L in LANGS_ROT:
    D_fit = [(et, lt) for et, lt in fitpairs[L] if et not in draw_en_tok]   # held-out translations excluding the display concepts (D_fit)
    A = np.array([ep(et) for et, lt in D_fit]); B = np.array([ep(lt) for et, lt in D_fit])   # code A=math A^T, code B=math B^T (each row one concept)
    R[L], _ = orthogonal_procrustes(A, B)                                    # A·Ω ≈ B → R[L]=Ω=R(L)^T (in the math, R(L)·A ≈ B)
v_o = u(np.mean([ep(c["en"]) for c in draw], axis=0))                        # base-point vector = mean direction of the English concepts
print(f"Estimated R(L) for {len(LANGS_ROT)} languages (held-out translation pairs, orthogonal Procrustes, 128 dimensions). English has no rotation (reference).")

# %% [markdown]
# ### 3.4 The inverse concept transform $C_L^{-1}(w)$
#
# When the model $v_L(w)=R(L)R(w)v_o$ holds exactly, the transform described below can **completely erase the concept component and leave only the language component**.
# Of course, in the actual embedding the transform is an approximation, so more precisely it **suppresses the concept component and relatively surfaces the language structure**. That is,
# even after the transform the concept contribution does not become exactly zero, and the points do not collapse to a single point but scatter.
#
# We define the concept rotation $R(w)$ **expressed in the coordinate system of language $L$** as the **concept transform**:
# $$ C_L(w) = R(L)\,R(w)\,R(L)^{-1} \qquad(\textit{concept transform for language } L). $$
# Its **inverse transform** we define as
# $$ C_L^{-1}(w) = R(L)\,R(w)^{-1}\,R(L)^{-1} \qquad(\textit{inverse concept transform for language } L) $$
# $C_L^{-1}(w)$ is the transform that removes the change due to concept $w$ from the language-$L$ embedding.
#
# Using $C_L(w)$, the model $v_L(w)=R(L)R(w)\,v_o$ can be written $v_L(w)=C_L(w)\,R(L)\,v_o$. We write the vector after this inverse transform as
# $$ x_L(w) \;:=\; C_L^{-1}(w)\,v_L(w) \;\approx\; R(L)\,v_o \;=:\; \hat{x}_L $$
# The right-hand side $\hat{x}_L=R(L)v_o$ does not contain the word $w$ (only the base position of language $L$), so all words of the same language should gather at the same point
# $\hat{x}_L$ (= the + in the figure below). The transform consists of the following 3 stages:
# 1. $R(L)^{-1}$: map the translation vector back to the English-side layout.
# 2. $R(w)^{-1}$: cancel the concept rotation, back to near the reference direction $v_o$.
# 3. $R(L)$: map the vector again to the language-$L$ side layout.
#
# Why a per-language conjugate rather than addition or a single transform common to all languages: the model applies the concept rotation first and the language rotation afterward
# ($v_L(w)=R(L)R(w)v_o$). If these two transforms commuted, applying a common $R(w)^{-1}$ would map to $R(L)v_o$, but in general orthogonal transforms do not commute, so a per-language conjugate
# is needed. Part 4 shows that neither the order-swapped **language-first model** nor the **additive model** produces the same effect.

# %%
def rinv(a, b, x):
    """Apply to x the inverse of the on-sphere rotation that turns base point a toward b (corresponds to the concept rotation R(w)^{-1})."""
    cth = float(np.dot(a, b))
    if cth > 1 - 1e-9:
        return x.copy()
    v = u(b - cth * a); ux, vx = float(np.dot(a, x)), float(np.dot(v, x))
    th = -np.arccos(np.clip(cth, -1, 1)); cs, sn = np.cos(th), np.sin(th)
    return x - ux * a - vx * v + (cs * ux - sn * vx) * a + (sn * ux + cs * vx) * v


def dvec_of(c, L):
    """Apply the inverse concept transform C_L^{-1}(w) to the translation vector: map back to the English side -> cancel the concept -> map again to the language-L side. The result is a language-L feature vector.
    Row convention (R[L]=R(L)^T; see "correspondence between the code and the math" in the R(L) estimation cell)."""
    x1 = ep(c["langs"][L][0]) @ R[L].T      # 1. map back to the English side, R(L)^{-1} (row convention: v @ R[L].T = R(L)^-1·v)
    x2 = rinv(v_o, ep(c["en"]), u(x1))      # 2. cancel the concept, R(w)^{-1}
    return u(u(x2) @ R[L])                   # 3. map again to the language-L side, R(L) (row convention: v @ R[L] = R(L)·v)


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
# **What the figure shows**: each point is the vector obtained by applying the inverse concept transform $C_L^{-1}(w)$ to a currently displayed (concept × language) translation token,
# reduced to 2D by t-SNE with cosine distance. It uses the same displayed concepts and languages as 2.3, but note that the working space is
# **PCA-128** (the same space as (2) in 3.5), not the raw 2560 dimensions of 2.3. The English concept points all map exactly to the base point $v_o$ under the inverse
# concept transform (for the reference language, $C_\text{en}^{-1}(w)\,v_\text{en}(w)=R(w)^{-1}R(w)\,v_o=v_o$), so they are not drawn individually but are represented by the
# single large + at the center (they have not disappeared; all of them coincide with the +). Point color is the language group (CANON_FAM, English = the black hub), and the
# translation label on each point lets individual words be read. The large plus signs mark each language's center $\hat{x}_L=R(L)\,v_o$ (the position the base point $v_o$ is
# mapped to for each language; for English, $\hat{x}_\text{en}=v_o$). **Observed layout**: the points cluster not by concept but by language (color),
# and the language centers + are placed far apart from one another. Colors of the same language group (e.g., CJK's zh/ja/ko = red) come close together.
# By contrast, before the transform, in 2.3 (raw 2560 dimensions) same-concept words clustered by concept.

# %% [markdown]
# **Procedure**: using the language rotation $R(L)$ estimated in 3.3 (estimated from only the held-out translations $D_\text{fit}$ excluding the display concepts) and
# the concept rotation $R(w)$ defined on the English side, we applied $C_L^{-1}(w)=R(L)R(w)^{-1}R(L)^{-1}$ to each currently displayed translation token
# (3 stages: map back to the English side → cancel the concept rotation → map again to the language-$L$ side). Passing the resulting vectors through t-SNE gives the figure above. In the next section 3.5 we quantify the
# difference before and after this transform as the distributions of cosine similarity over the 3 groups within / cross / translation.
#
# **Interpretation**: before the transform, words of the same concept had high similarity, whereas after the transform words of the same language became relatively more similar.
# This can be interpreted as the language-dependent structure present in the original embedding surfacing once the concept-dependent structure is suppressed. This result is
# consistent with the model $v_L(w)=R(L)R(w)v_o$ that applies the concept rotation $R(w)$ first and the language rotation $R(L)$ afterward. The explanatory power of the additive model
# is examined in Part 4.2.
#
# - **The points forming a "cloud"**: if the model $v_L(w)=R(L)R(w)\,v_o$ were exact, then after the transform $C_L^{-1}(w)\,v_L(w)=R(L)\,v_o$, and
#   the words of language $L$ should collapse to a single point (the center +). The actual spread is interpreted as the residual by which the data departs from a pure rotation
#   (the spherical-rotation approximation of $R(w)$ and the finite estimation of $R(L)$).
# - **Same language groups being close**: the closeness among the centers $R(L)\,v_o$ reflects how similar the $R(L)$ are. Languages whose correspondence with English is formed similarly
#   (e.g., CJK) have similar $R(L)$, so they sit adjacent. This is the same geometry as the Part 1 clusters seen from a different angle.
# - **Degree of circularity**: the **direct estimation** of $R(L)$ does not use the display concepts. However, language selection, the PCA basis, and the construction of the reference direction $v_o$
#   involve the evaluation concepts, so this is **not a fully external evaluation** (in the limited sense that we avoid in-sample fitting for the $R(L)$ estimation).

# %% [markdown]
# ### 3.5 The three-group similarity distributions
#
# We compare the cosine-similarity distributions of three types of pairs (rotation 14 languages, concept pool draw).
# - **within (blue, same language)**: separate words of the same language (same language, different concepts).
# - **cross (gray, baseline distribution)**: pairs formed by independently choosing concepts from two different languages (not conditioned on whether the concepts match). Used as a baseline distribution that imposes no special correspondence of language or concept.
# - **translation (red, translation pairs)**: same-meaning translation pairs (same concept, different languages).
#
# Let the mean of each distribution be $\bar w$ (blue), $\bar c$ (gray), $\bar t$ (red). The increase in mean cosine similarity relative to the baseline distribution $\bar c$ we call the
# **similarity gain**, and we define the similarity gain from the same-language constraint as the **language gain** and that from the same-concept constraint as the
# **concept gain** (quantities defined in this notebook; not metrics from existing research):
# $$ \Delta_\text{lang} = \bar w - \bar c \quad(\textit{language gain}), \qquad
#    \Delta_\text{concept} = \bar t - \bar c \quad(\textit{concept gain}). $$
# The former is "how much closer same-language pairs are than the baseline," the latter "how much closer same-concept (translation) pairs are than the baseline." The difference of these two we
# define as the **language–concept similarity contrast** and use as the model's single score:
# $$ \text{contrast} = \Delta_\text{lang} - \Delta_\text{concept} = \bar w - \bar t. $$
# If $\text{contrast}>0$, same-language pairs are on average closer than translation pairs (a **language-dominated representation**); if $<0$, translation pairs are closer (a **concept-dominated representation**).
# The table shows the 3 means and 2 gains as a breakdown, and the **classification of concept-dominated vs language-dominated is made from the sign of contrast**.

# %%
# 3-group cos helper: takes (concept c, lang L)->vector and returns within/cross/translation over all of draw and LANGS_ROT.
# What is passed to vec_of determines the space measured (raw 2560-dim / PCA-128 before transform / PCA-128 after transform). Part 4 uses the same function.
# The English hub (the base point, shared by all concepts) is excluded from the group statistics.
# All three groups enumerate every pair in full (no random sampling, so the result is deterministic). Since draw=855 is small here,
# each group is computed in a single matrix product (V@V.T, byL[li]@byL[lj].T). But cross grows as the outer product |D_li|x|D_lj| per language
# pair, i.e. quadratically in the number of concepts, so for a much larger dictionary this full enumeration should be replaced by estimating
# each group's mean from a randomly sampled subset.
def three_groups_vec(vec_of):
    dv = {(i, L): vec_of(c, L) for i, c in enumerate(draw) for L in c["langs"]}
    byL = {L: [dv[(i, L)] for i, c in enumerate(draw) if L in c["langs"]] for L in LANGS_ROT}
    within, cross, trans = [], [], []
    for L in LANGS_ROT:                                       # within: all same-language, different-concept pair cos (upper triangle)
        V = np.array(byL[L]); within += list((V @ V.T)[np.triu_indices(len(V), 1)])
    for li, lj in itertools.combinations(LANGS_ROT, 2):       # cross: all concept x all concept cos across a language pair (whole outer product)
        cross += list((np.array(byL[li]) @ np.array(byL[lj]).T).ravel())
    for i, c in enumerate(draw):                              # translation: same-concept, different-language cos
        for li, lj in itertools.combinations(list(c["langs"]), 2):
            trans.append(float(dv[(i, li)] @ dv[(i, lj)]))
    return np.array(within), np.array(cross), np.array(trans)


def sim_scores(w, c, s):
    """From the 3-group means (within=blue, cross=gray, translation=red), return the language/concept gains and the contrast (score).
    lang_gain=w̄−c̄, concept_gain=s̄−c̄, contrast=w̄−s̄(=lang_gain−concept_gain)."""
    wm, cm, sm = float(np.mean(w)), float(np.mean(c)), float(np.mean(s))
    return dict(within=wm, cross=cm, translation=sm, lang_gain=wm - cm, concept_gain=sm - cm, contrast=wm - sm)


def print_scores(name, groups):
    d = sim_scores(*groups)
    print(f"  {name:<22}: within {d['within']:+.3f} / cross {d['cross']:+.3f} / translation {d['translation']:+.3f}"
          f"  |  lang_gain {d['lang_gain']:+.3f} / concept_gain {d['concept_gain']:+.3f}  =>  contrast {d['contrast']:+.3f}")


g_raw = three_groups_vec(lambda c, L: e(c["langs"][L][0]))    # (1) raw 2560-dim (corresponds to the t-SNE of 2.3)
g_pre = three_groups_vec(lambda c, L: ep(c["langs"][L][0]))   # (2) PCA-128, before transform (new)
g_post = three_groups_vec(dvec_of)                            # (3) PCA-128, after transform = the inverse concept transform C_L^{-1} (corresponds to 3.4)
bins = np.linspace(-0.6, 1.0, 121); ctr = 0.5 * (bins[1:] + bins[:-1]); HC = {"w": "#1f77b4", "c": "#7f7f7f", "s": "#d62728"}


def draw_hist(ax, within, cross, trans, title):
    for x, cc_ in [(within, HC["w"]), (cross, HC["c"]), (trans, HC["s"])]:
        h = np.histogram(x, bins=bins, density=True)[0]
        ax.fill_between(ctr, h, color=cc_, alpha=0.28, step="mid"); ax.plot(ctr, h, color=cc_, lw=2.2, drawstyle="steps-mid"); ax.axvline(float(np.mean(x)), color=cc_, ls="--", lw=1.4)
    ax.set_xlabel("cosine"); ax.set_title(title, fontsize=11, fontweight="bold"); ax.set_xlim(-0.6, 1.0); ax.spines[["top", "right"]].set_visible(False)


def plot_hist_one(groups, title, fname):
    """Draw the 3-group histogram of one pattern in a single figure (kept as separate figures for easy later reuse)."""
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
# **What the figure shows**: all three panels show histograms of cosine similarities for the three pair types (within blue / cross gray = baseline / translation red, dashed = each group's mean). Only the measured space is
# varied across them: (1) raw 2560-dim (native, corresponds to the t-SNE of 2.3), (2) PCA-128 before transform, (3) PCA-128 after transform = the inverse concept transform
# $C_L^{-1}$ (corresponds to 3.4). **Observed distribution**: in both (1) and (2), red (translation) lies farthest to the right; only in (3) does blue (within) move farthest to the right.
# Each group's mean, gains, and contrast are summarized in the table of the next cell.

# %%
# Large table: 3 patterns × (3 group means, 2 gains, contrast). The numbers are deterministic quantities computed over all pairs (full enumeration, no sampling)
# of the target languages LANGS_ROT and the whole concept pool draw. SEED only selects the 48 concepts displayed in the t-SNE and does not affect this table.
print("3-group means and derived quantities  [baseline=cross(gray); lang_gain=within-cross, concept_gain=translation-cross, contrast=within-translation=score]")
print_scores("(1) raw 2560-dim", g_raw)
print_scores("(2) before transform PCA-128", g_pre)
print_scores("(3) after transform C_L^-1", g_post)

# %% [markdown]
# **Table: 3-group means and derived quantities for the 3 patterns** (the numbers match the print in the cell above).
# These numbers are deterministic quantities, averaged over all pairs of the target languages $\mathcal{L}$ (14 languages) and the concept pool `draw` (855 concepts);
# none of within / cross / translation involves random sampling (`SEED` only selects the 48 concepts displayed in the t-SNE and does not change these numbers). At this
# dictionary size we compute all pairs in a single matrix product, but for a much larger dictionary, where cross in particular grows quadratically in the number of concepts
# as an all-concept × all-concept outer product, it would be appropriate to estimate each group's mean from a randomly sampled subset.
#
# | pattern (measured space) | within (blue) | cross (gray, baseline) | translation (red) | $\Delta_\text{lang}$ | $\Delta_\text{concept}$ | contrast |
# |---|---|---|---|---|---|---|
# | (1) raw 2560-dim | 0.109 | 0.092 | 0.246 | 0.017 | 0.154 | **−0.137** |
# | (2) before transform PCA-128 | 0.091 | 0.011 | 0.339 | 0.079 | 0.328 | **−0.248** |
# | (3) after transform $C_L^{-1}$ PCA-128 | 0.253 | 0.018 | 0.105 | 0.235 | 0.087 | **+0.148** |
#
# **Observation** (fact): the contrast is negative in (1) −0.137 and (2) −0.248 (same-concept pairs are closest = concept-dominated representation), and positive only in (3) at +0.148
# (same-language pairs are closest = language-dominated representation). The sign changes only in (3).
#
# **Interpretation**: in the (1)→(2) PCA-128 projection, cross (baseline) drops 0.092→0.011 and translation rises 0.246→0.339, so the concept structure
# actually stands out more (but the contrast stays negative, concept-dominated). The change **from concept-dominated to language-dominated** happens at the (2)→(3) inverse concept transform, where $\Delta_\text{lang}$
# rises sharply 0.079→0.235 and $\Delta_\text{concept}$ drops 0.328→0.087. **That is, the surfacing of language structure is the effect of the $C_L^{-1}$ transform itself, not of dimensionality reduction.**
# A fair before/after comparison of the method is seen in the same space, (2)→(3) ((1) is the native original space, corresponding to the t-SNE of 2.3).

# %% [markdown]
# ## Part 4  Two baseline models
#
# In Part 3, under the model $v_L(w)=R(L)R(w)v_o$, building the **conjugate** $C_L^{-1}(w)=R(L)R(w)^{-1}R(L)^{-1}$ changed the representation
# from concept-dominated to language-dominated (the language structure surfaced). Now, are the two other models, the **language-first model** and the **additive model**,
# equally well supported by the observed geometry of $W_E$?
#
# 1. **Language-first model** (4.1): $v_L(w)=R(w)R(L)v_o$. Erasing the concept is just multiplying by $R(w)^{-1}$ (no per-language $R(L)$ needed).
# 2. **Additive model** (4.2): $v_L(w)=v_\text{en}(w)+a_L$. Represents translation by adding a constant vector rather than by a rotation.
#
# Both are tested on the **same data and the same metric** as Part 3. The metric is the 3 groups defined in Part 3.5 (within/cross/translation) and their summary score, the
# **language–concept contrast** $\text{contrast}=\bar w-\bar t$ ($>0$: same-language pairs are on average closest = language-dominated; $<0$: same-concept pairs are closest =
# concept-dominated). All comparisons use the same PCA-128 space, the same 14 rotation languages, and the same concept pool `draw`. Under these matched conditions, we confirm that neither model makes the language structure apparent
# (neither becomes language-dominated).

# %% [markdown]
# ### 4.1 Language-first model $v_L(w)=R(w)R(L)v_o$
#
# If the order were **the language rotation $R(L)$ first and the concept rotation $R(w)$ afterward** ($v_L(w)=R(w)R(L)v_o$), erasing the concept would be easy:
# just multiply the translation vector by the inverse $R(w)^{-1}$ of the concept rotation defined on the English side. We write the vector after this inverse transform as
# $$ y_L(w) \;:=\; R(w)^{-1}\,v_L(w) $$
# If this language-first model were correct, then $y_L(w)=R(L)v_o$ (independent of word $w$), and we would not even need to estimate a per-language $R(L)$.
# We apply this transform to the observed embeddings and ask whether the transformed points cluster by language as they did under the conjugate in Part 3.
#
# **Reference points** ×: on the t-SNE we overlay, as ×, the mean of each language's $y_L(w)$, $\hat{y}_L=\text{mean}_{w\in D_\text{fit}}\,y_L(w)$
# (held-out $D_\text{fit}$; no normalization; evaluated using cosine similarity). If the language-first model were correct, $\hat{y}_L=R(L)v_o$ (= the $\hat{x}_L$ of 3.4), and
# the points should collapse onto ×. We see whether that happens on the actual data.
#
# **Implication for language transfer**: in the concept-first model, transfer between languages could be written with a single concept-independent transform $R(L')R(L)^{-1}$ (3.1). In the language-first model
# it cannot. Eliminating $v_o$ from $v_L(w)=R(w)R(L)v_o$ and $v_{L'}(w)=R(w)R(L')v_o$ gives
# $$ v_{L'}(w) = R(w)\,R(L')\,R(L)^{-1}\,R(w)^{-1}\,v_L(w) $$
# and the transfer transform includes a conjugation by $R(w)$. That is, it **changes with each concept $w$**, and the language-first model does not give concept-independent language transfer.

# %%
# Language-first model inverse transform y_L(w)=R(w)^{-1}v_L(w): multiply R(w)^{-1} directly onto the translation vector (rinv=R(w)^{-1}, v_o from 3.3. No R(L) needed).
def dvec_swap(c, L):
    """Concept erasure y_L(w)=R(w)^{-1}v_L(w) for the language-first model v_L(w)=R(w)R(L)v_o. R(w)^{-1} only (does not use a per-language R(L)). rinv is a rotation of a unit vector, so the result is already unit (no normalization needed; symmetric with the reference point ŷ_L)."""
    return rinv(v_o, ep(c["en"]), ep(c["langs"][L][0]))


def yhat_of(L):
    """Language-first reference point ŷ_L = mean_{w in D_fit} y_L(w) (held-out D_fit). The raw mean estimates the reference R(L)v_o (the centroid). It is the mean of the (unit) dvec_swap points, so it is symmetric."""
    D_fit = [(et, lt) for et, lt in fitpairs[L] if et not in draw_en_tok]
    return np.mean([rinv(v_o, ep(et), ep(lt)) for et, lt in D_fit], axis=0)

# %%
# Language-first after inverse transform, t-SNE (same PCA-128 space, same t-SNE settings, sub concepts as 3.4). Overlay only × = ŷ_L (mean of the language-first inverse transform y_L(w)).
def plot_swap_tsne():
    pts, D = [], []
    for c in sub:
        for L in c["langs"]:
            pts.append((L, c["langs"][L][1])); D.append(dvec_swap(c, L))
    npts = len(D); lang = [p[0] for p in pts]
    yhat = [yhat_of(L) for L in LANGS_ROT]                 # ŷ_L = mean of the language-first inverse transform y_L(w) (reference point, held-out D_fit)
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
# **What the figure shows**: each point is the language-first inverse transform $y_L(w)=R(w)^{-1}v_L(w)$ (**PCA-128**) of a (displayed concept × language) translation token, reduced to 2D by
# t-SNE with cosine distance (the same PCA-128 space and t-SNE settings as 3.4; the layout is re-optimized independently for each figure; color = language group). × is each language's mean $\hat{y}_L$ (the reference point of the inverse
# transform, held-out $D_\text{fit}$). **Observed layout**: in 3.4 (after the conjugate) the points clustered onto widely separated language centers, but here the points
# scatter by concept and do not collapse onto ×. Same-meaning translations (city = ciudad/cidade/città/都市/град, comment = comentario/コメント/댓글,
# etc.) stay close across language groups, indicating that the representation remains concept-dominated. Moreover the reference points × ($\hat{y}_L$) mostly cluster near the center,
# so with the language-first inverse transform the language centers themselves barely separate (in contrast to 3.4, where the language centers were widely separated).

# %% [markdown]
# **Interpretation**: meaning (concept) is still dominant, and $R(w)^{-1}$ alone does not surface the language structure. The language-first model (applying the language rotation first and the
# concept rotation afterward, $v_L(w)=R(w)R(L)v_o$) can be read as not supported by this diagnostic. A quantitative confirmation (that the same-language structure barely increases) is
# done with the next histogram and the contrast (language–concept contrast).

# %%
# 3-group distribution of the language-first model (this single figure = the language-first model only; the numbers are also just the one row corresponding to this figure).
sw = three_groups_vec(dvec_swap)     # language-first R(w)^{-1} (PCA-128, rotation 14 languages, concept pool draw)


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
# **What the figure shows**: the 3-group cosine distribution after language-first $R(w)^{-1}$ (PCA-128, rotation 14 languages, concept pool draw; within blue / cross gray =
# baseline / translation red, dashed = each group's mean). The numbers corresponding to this figure (same format as 3.5, one row):
#
# | pattern (measured space) | within (blue) | cross (gray, baseline) | translation (red) | $\Delta_\text{lang}$ | $\Delta_\text{concept}$ | contrast |
# |---|---|---|---|---|---|---|
# | language-first $R(w)^{-1}$ PCA-128 | 0.220 | 0.135 | 0.339 | 0.085 | 0.205 | **−0.119** |
#
# **Observed distribution**: red (translation, mean 0.34) stays farthest to the right, so the language structure does not become apparent (within 0.22 is right of cross 0.14 but does not
# exceed translation). The contrast of this figure is **−0.119** (negative = stays concept-dominated). Placing it next to 3.5's before-transform −0.248 and after-transform +0.148,
# language-first barely moves from before-transform and does not become positive (language-dominated).

# %% [markdown]
# **Interpretation**: language-first $R(w)^{-1}$ is an orthogonal transform applied identically to both vectors in each translation pair, so it **preserves the translation-pair cosine**
# (before transform 0.339 = language-first 0.339, an exact match). within rises 0.091→0.220, but cross (baseline) also rises 0.011→0.135
# (overall crowding in the post-rotation subspace), so the **language gain $\Delta_\text{lang}$ (within−cross) is nearly flat**
# (0.079→0.085), and the within-language structure barely increases. Since translation stays high, the contrast stays negative (−0.25→−0.12),
# so it does not become language-dominated. The conjugate $C_L^{-1}$, by contrast, uses per-language $R(L)$ to reduce the translation-pair similarity 0.339→0.105 while keeping cross low, and
# raises $\Delta_\text{lang}$ sharply 0.079→0.235 (contrast +0.15). On the same data, only the concept-first model's conjugate $C_L^{-1}$ turns language-dominated;
# the language-first model's $R(w)^{-1}$ does not achieve the same effect.

# %% [markdown]
# ### 4.2 Additive model $v_L(w)=v_\text{en}(w)+a_L$
#
# As another hypothesis, consider the **additive model**, which represents the language difference by a fixed vector addition: $v_L(w)=v_\text{en}(w)+a_L$
# ($a_L$ is a shift vector for each language $L$). If this were correct, the offset vector $z_L(w)$, obtained by subtracting the English vector from the translation, would be
# constant $a_L$ independent of concept:
# $$ z_L(w) \;:=\; v_L(w)-v_\text{en}(w) \;\approx\; a_L, \qquad \hat{z}_L \;:=\; \operatorname{mean}_{w\in D_\text{fit}}\,z_L(w). $$
# That is, if the same-language $z_L(w)$ align in one direction, they cluster by language (we compare direction by cosine, so we do not normalize). We check this on the actual data (PCA-128). On the t-SNE we overlay
# each language's mean $\hat{z}_L$ as a **△**.
#
# **Implication for language transfer**: the additive model is the same as the concept-first model in that it gives concept-independent language transfer. From $v_L(w)=v_\text{en}(w)+a_L$,
# $$ v_{L'}(w) = v_L(w) + (a_{L'}-a_L) $$
# so transfer is the addition of a constant offset $a_{L'}-a_L$ independent of concept $w$. However, as we see in this section, the additive model does not surface the language structure
# of the actual data (does not become language-dominated), so even though the transfer is concept-independent, it is less well supported under this diagnostic than the concept-first model.

# %%
# Additive model offset z_L(w)=v_L(w)-v_en(w) (translation − English, PCA-128). A difference, so non-unit. Return it raw (the reference point ẑ_L is also a raw mean, so symmetric); normalize only where needed: t-SNE uses cosine so not needed, only the numeric table three_groups_vec measures inner product = cos, so the caller applies u().
def dvec_add(c, L):
    """Raw offset z_L(w)=v_L(w)-v_en(w) of the additive model v_L(w)=v_en(w)+a_L (not normalized; the caller applies u() only when passing to the numeric table)."""
    return ep(c["langs"][L][0]) - ep(c["en"])


def zhat_of(L):
    """Additive model reference point ẑ_L = mean_{w in D_fit} z_L(w) (held-out D_fit). The raw mean estimates the additive offset a_L (the centroid). The dvec_add points are also raw, so it is symmetric."""
    D_fit = [(et, lt) for et, lt in fitpairs[L] if et not in draw_en_tok]
    return np.mean([ep(lt) - ep(et) for et, lt in D_fit], axis=0)

# %%
# Additive after the offset, t-SNE (△ = ẑ_L). Exclude en, since z=0 and its direction is undefined.
def plot_add_tsne():
    pts, D = [], []
    for c in sub:
        for L in c["langs"]:
            pts.append((L, c["langs"][L][1])); D.append(dvec_add(c, L))
    npts = len(D); lang = [p[0] for p in pts]
    zhat = [zhat_of(L) for L in LANGS_ROT]                 # ẑ_L = mean of the offset vectors (held-out D_fit)
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
# **What the figure shows**: each point is the offset vector $z_L(w)=v_L(w)-v_\text{en}(w)$ (PCA-128, direction measured by cosine) of a (displayed concept × language) translation token,
# reduced to 2D by t-SNE (cosine distance) (color = language group). △ marks each language's mean $\hat{z}_L$ (held-out $D_\text{fit}$). **Observed layout**: the points do not cluster by language group
# (color) but mix throughout, so the offset vectors do not cluster by language. The triangular markers ($\hat{z}_L$) are all concentrated near the center, and the points do not
# collapse there, so the language centers barely separate either.

# %% [markdown]
# **Interpretation**: if the additive model were correct, the same-language offset vectors $z_L(w)$ would align in one direction $a_L$ independent of concept, and the points would gather at △.
# In fact they do not align by language (confirmed by $\Delta_\text{lang}\approx0$ in the next metric). With the selected data and the PCA-128 representation, the language difference cannot be sufficiently explained by a single constant offset.

# %%
# 3-group distribution of the additive model (this single figure = the additive model only; the numbers are also just the one row corresponding to this figure).
za = three_groups_vec(lambda c, L: u(dvec_add(c, L)))  # additive z_L(w) (PCA-128, rotation 14 languages, draw). Unit-normalize the raw offset before passing since inner product = cos


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
# **What the figure shows**: the 3-group cosine distribution of the additive offset $z_L(w)$ (PCA-128, rotation 14 languages, concept pool draw; within blue / cross gray =
# baseline / translation red, dashed = each group's mean). The numbers corresponding to this figure (same format as 3.5, one row):
#
# | pattern (measured space) | within (blue) | cross (gray, baseline) | translation (red) | $\Delta_\text{lang}$ | $\Delta_\text{concept}$ | contrast |
# |---|---|---|---|---|---|---|
# | additive $z_L=v_L-v_\text{en}$ PCA-128 | 0.279 | 0.223 | 0.460 | 0.056 | 0.237 | **−0.181** |
#
# **Observed distribution**: red (translation, mean 0.46) lies farthest to the right, so the same-concept offset vectors align best. Blue (within 0.28) barely differs from gray (cross 0.22)
# ($\Delta_\text{lang}$ only 0.056), and the contrast is **−0.181** (negative = concept-dominated).
#
# **Interpretation**: the offset vectors align in the "concept" direction (high translation) but hardly align in the "language" direction ($\Delta_\text{lang}\approx0$).
# The additive model does not surface the language structure and, unlike the conjugate, does not become language-dominated.

# %% [markdown]
# ### 4.3 Summary of the model comparison
#
# We line up the 3 inverse transforms by the same metric (the language–concept contrast on PCA-128, rotation 14 languages, concept pool draw). The baseline is before transform.

# %%
# Contrast comparison across all models (this table alone lines up all models at once).
print("Compare the models by the language-concept contrast (score)  [PCA-128, rotation 14 languages, concept pool draw]")
print_scores("before transform (baseline)", g_pre)
print_scores("conjugation C_L^-1 (Part 3)", g_post)
print_scores("language-first R(w)^-1 (4.1)", sw)
print_scores("additive z_L=v_L-v_en (4.2)", za)

# %% [markdown]
# **Table: comparison of the inverse-transform models** (with contrast > 0 the representation turns language-dominated).
#
# | model (inverse transform) | within | cross | translation | $\Delta_\text{lang}$ | $\Delta_\text{concept}$ | **contrast** |
# |---|---|---|---|---|---|---|
# | before transform (baseline) | 0.091 | 0.011 | 0.339 | 0.079 | 0.328 | −0.248 |
# | **conjugate $C_L^{-1}$** (concept-first-then-language order + conjugate) | 0.253 | 0.018 | 0.105 | 0.235 | 0.087 | **+0.148** |
# | language-first $R(w)^{-1}$ (4.1) | 0.220 | 0.135 | 0.339 | 0.085 | 0.205 | −0.119 |
# | additive $z_L=v_L-v_\text{en}$ (4.2) | 0.279 | 0.223 | 0.460 | 0.056 | 0.237 | −0.181 |
#
# **Observation**: **among the three methods compared here**, the only one whose contrast becomes positive (turns language-dominated) is the **conjugate $C_L^{-1}$** (+0.148). Both language-first (−0.119) and additive (−0.181)
# stay negative, remaining concept-dominated. Both have small $\Delta_\text{lang}$ values (language-first 0.085, additive 0.056), indicating only weak same-language structure.
#
# **Interpretation**: among the three methods compared here, the only one that could surface the language structure (contrast>0) was the one assuming the order that applies the concept rotation first and the language rotation afterward,
# $v_L=R(L)R(w)v_o$, and building the **conjugate** $C_L^{-1}=R(L)R(w)^{-1}R(L)^{-1}$ from the per-language estimated $R(L)$. Neither language-first
# (only $R(w)^{-1}$) nor addition ($z_L$) surfaces the language structure. This result is consistent with a model that describes the multilingual correspondence as a composition of a language-alignment transform and a concept rotation.

# %% [markdown]
# ## Summary
#
# - **Multilingual clustering** (Part 1): English clusters not with its own language group (Germanic) but with CJK. Even adding many European languages,
#   English–Chinese is the highest in top-k nearest-neighbor similarity. This tendency is consistent with the training-data composition (though the effects of the English-pivot dictionary and the selection criteria are not separated out).
# - **Selecting languages for the rotation** (Part 2.1): each language's rotation $R(L)$ overfits when the translation pairs are few. We use only the 14 non-overfitting languages for the rotation.
# - **Raw embeddings** (Part 2.3): same-meaning translations are close, indicating a **concept-dominated representation**.
# - **Inverse concept transform $C_L^{-1}(w)$** (Part 3): under the **concept-first model** $v_L(w)=R(L)R(w)v_o$, building the conjugate transform $C_L^{-1}(w)=R(L)R(w)^{-1}R(L)^{-1}$
#   that cancels the concept component changes the representation from concept-dominated to language-dominated (the language structure becomes apparent and words from the same language form distinct clusters). This result is consistent with a model that describes multilingual correspondence as a composition of rotations (orthogonal transforms).
# - **Control experiments** (Part 4): both the **language-first model** (only $R(w)^{-1}$, 4.1) and the **additive model** representing the language difference by a constant addition
#   ($z_L=v_L-v_\text{en}$, 4.2) keep the language–concept contrast negative, so they do not become language-dominated (the comparison table in 4.3). Among the three methods compared here, the only one that turned language-dominated was
#   the **concept-first model**'s conjugate $C_L^{-1}$.
# - **Implication for language transfer** (3.1, 4.1/4.2): with the concept-first model, transfer between languages ($L\to L'$) can be written with a single concept-independent transform $R(L')R(L)^{-1}$
#   (the additive is also concept-independent, while the language-first changes its transform per concept). That the concept-first model is supported connects to the applied implication of **concept-independent language transfer** stated at the outset.
#   The transfer performance itself broadly correlates with raw closeness but does not fully coincide (2.2).

# %% [markdown]
# ## Part 5  Related work and where this study stands
#
# (From here on is the research positioning, i.e. related work. The main body of the demo concludes with Part 4 and the Summary. Read this as a supplement for those interested in the background.)
#
# The main contribution of this study is empirical rather than methodological. When we view the multilingual correspondence as "a composition of a language rotation $R(L)$ and a concept rotation $R(w)$,"
# under the language–concept contrast diagnostic on Qwen3-4B's token embedding, the best-supported model was the **concept-first model** $R(L)R(w)v_o$, applying the concept first and the language afterward.
# Under this model, canceling the concept component by a conjugate transform makes the language structure apparent, whereas the order-swapped **language-first model** $R(w)R(L)v_o$ does not. Below we organize how the methodological components underlying this observation (multilingual orthogonal alignment, operator representations of lexical relations,
# separation of language and meaning, the geometry of token embeddings) are all known, and then state that **to the best of our knowledge, no prior work in the surveyed literature has directly addressed the order in which the language and concept operators are composed (which model explains the embedding)**, and position this study there. Since the components of the method are all known, **we do not claim methodological novelty**.

# %% [markdown]
# ### 5.1 Orthogonal alignment of multilingual embeddings
#
# Research on aligning separately trained word-embedding spaces with a linear map began with [Mikolov et al. (2013a)](https://arxiv.org/abs/1309.4168). [Xing et al. (2015)](https://aclanthology.org/N15-1104/) introduced unit-length normalization and
# an orthogonality constraint, [Artetxe et al. (2016)](https://aclanthology.org/D16-1250/) grounded orthogonality as "the condition preserving inner products within a language," and [Smith et al. (2017)](https://arxiv.org/abs/1702.03859) showed that a self-consistent map
# **should be orthogonal** (orthogonal Procrustes, SVD solution). [Conneau et al. (2018, MUSE)](https://arxiv.org/abs/1710.04087) made unsupervised bilingual-dictionary induction practical, and this study uses its
# **bilingual dictionaries**. [Jawanpuria et al. (2019, GeoMM)](https://aclanthology.org/Q19-1007/) align to a common space with a language-specific rotation (language-as-rotation).
# In short, **representing a language by a rotation is itself standard**, and this study's $R(L)$ is a tool in this lineage. The difference is that whereas these align **two separately trained spaces**,
# this study treats subsets of **a single shared token embedding matrix $W_E$** as per-language spaces.

# %% [markdown]
# ### 5.2 Lexical relations as operators
#
# Lexical semantic relations have traditionally been represented by **additive** difference vectors ([Mikolov et al. 2013b](https://aclanthology.org/N13-1090/)'s king−man+woman≈queen). [Ethayarajh (2019)](https://aclanthology.org/D19-1354/) represented
# these by an **orthogonal operator** ($R\,\vec{\mathrm{king}}\approx\vec{\mathrm{queen}}$) and reported that orthogonal transforms are nearly on par with additive, and general linear slightly better.
# [Reif et al. (2026)](https://aclanthology.org/2026.findings-acl.1618/) showed that even in LLM token embeddings, morphological and orthographic changes (tense, capitalization) can be represented by additive transform vectors, and [Park et al. (2024)](https://arxiv.org/abs/2311.03658) formalized the hypothesis that concepts
# appear as linear directions. In sum, **representing relations and concepts by rotations or linear operators is also known**. However, whereas much prior work handles the **composition of multiple
# relation operators** (concept × concept, like $R(w_1)R(w_2)$), what this study uses for concepts is a **single $R(w)$**. Also, since this study treats embeddings on the
# unit sphere, the **additive model is a control (baseline) that degenerates from the start** (translating by a constant vector degenerates on the sphere; locally a rotation is
# a first-order approximation of addition, so it is not wrong). The additive model of Part 4.2 remains a control for explicitly confirming that "the languages do not align."

# %% [markdown]
# ### 5.3 Separation of language and meaning, and the geometry of token embeddings
#
# Work that separates a multilingual model's representation into a language component and a meaning component is close to this study. [Gonen et al. (2020)](https://aclanthology.org/2020.blackboxnlp-1.5/) identified mBERT's "language subspace" and
# used projection to add and remove language information. [Chang et al. (2022)](https://aclanthology.org/2022.emnlp-main.9/) showed that in XLM-R, after mean-centering, languages occupy similar subspaces and the **language-sensitive axes and language-neutral axes are nearly orthogonal**.
# The multilingual geometry of the input token embeddings themselves is directly addressed by [Wen-Yi and Mimno (2023)](https://aclanthology.org/2023.emnlp-main.71/) (linearly separable by writing system, with the geometry differing across model families),
# and [Kim and Lee (2025)](https://arxiv.org/abs/2511.16693) examined the relation between the language directions of token embeddings and the training-data composition. [Mathewson (2026)](https://arxiv.org/abs/2603.02258) reported, in a translation model, an "offset invariance" in which concept-difference vectors
# are consistent across languages. Surveying these, prior work roughly splits into three: representing concepts by rotation (Ethayarajh),
# representing languages by rotation (GeoMM), and representing languages by an **additive** mean shift (Chang, Mathewson). **In the surveyed literature, we found no prior work that combines them as a composition of operators and explicitly asks how the order matters**.
# This study addresses this gap and extends this line of analysis to the decoder-only Qwen3-4B.

# %% [markdown]
# ### 5.4 Non-commutativity of operator composition
#
# That "the order of an operator product changes the result" has itself been handled in knowledge-graph embeddings ([Xu and Li 2019](https://aclanthology.org/P19-1026/) and others, explicitly modeling the non-commutativity of relation composition).
# However, that work concerns relations in knowledge graphs rather than lexical relations. More directly relevant for comparison are studies that **claim or assume** commutativity:
# [Freenor and Alvarez (2026, RISE)](https://arxiv.org/abs/2510.09790) showed that semantic transforms are **order-independent and commutative** (to second order), and [Liu et al. (2017)](https://arxiv.org/abs/1705.02426) assume the relation matrices form a **commutative family**
# $A_iA_j=A_jA_i$. Whereas these presuppose commutativity, in this study **the concept-first and language-first models are empirically not equivalent**, consistent with the possibility that the language and concept rotations, being heterogeneous, do not commute.

# %% [markdown]
# ### 5.5 Distinction from "latent language" studies
#
# This study asks a different question from work investigating whether an LLM internally reasons through English (or Chinese), using hidden states across layers ([Wendler et al. 2024](https://aclanthology.org/2024.acl-long.820/), [Zhong et al. 2025](https://aclanthology.org/2025.findings-acl.1350/),
# [Schut et al. 2025](https://arxiv.org/abs/2502.15603)). Those trace the **computation process** with the logit lens. What this study addresses is the static geometry of the **fixed token embedding matrix
# $W_E$**, and it makes no claim about "what language the model thinks in." The English hub and CJK cluster seen in Part 1 are properties of the static embedding,
# not claims about the computation process.

# %% [markdown]
# ### 5.6 Where this study stands
#
# From the above, the parts used in this study (orthogonal alignment, lexical meaning as operators, separation of language/meaning, the geometry of token embeddings) are all known, and
# **we do not claim methodological novelty**. The contribution of this study is a single **empirical observation**: in a single LLM's shared token embedding, when language and concept are
# viewed as orthogonal operators, the best-supported ordering under the language–concept contrast diagnostic was the **concept-first model** $R(L)R(w)v_o$, applying the concept first and the language afterward.
# Under this model, canceling the concept component by the conjugate $C_L^{-1}(w)=R(L)R(w)^{-1}R(L)^{-1}$ surfaces the language structure, whereas the order-swapped **language-first model** does not. To the best of our knowledge, in the surveyed literature,
# no prior work has decomposed a single shared LLM token embedding into a language operator and a concept operator **including the composition order** and removed the concept component by conjugation. The results are for Qwen3-4B, and we **do not extrapolate** to other models. However,
# simply running this notebook applies the same analysis to any model, providing a basis for systematic comparisons across models.

# %% [markdown]
# ### References
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
# 22. Facebook Research. [*MUSE: Multilingual Unsupervised and Supervised Embeddings*](https://github.com/facebookresearch/MUSE) (bilingual dictionaries).
