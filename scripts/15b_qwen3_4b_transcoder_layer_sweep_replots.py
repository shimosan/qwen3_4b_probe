# script 15 (layer_sweep) の出力 CSV から、追加プロットだけを生成する。
# 元 script は再実行しない (60 GB の transcoder weights 読み直し不要)。
#
# 入力:
#   outputs/prelim_qwen3_4b_transcoder_layer_sweep_position_metrics.csv  (180 行 × 22 列)
#   outputs/prelim_qwen3_4b_transcoder_layer_sweep_summary.csv           (36 行 × 30 列)
#
# 生成する PNG:
#   1. _max_abs_delta_pos34.png        pos=3, 4 のみ (Fig 1 差し替え用)
#   2. _l2_delta_pos34.png             pos=3, 4 のみ (Fig 2 差し替え用)
#   3. _tanimoto_pos34.png             pos=3, 4 のみ (Fig 3 差し替え用)
#   4. _jaccard_pos34.png              pos=3, 4 のみ、binary Jaccard (新規、Fig 3 比較用)
#   5. _max_single_pos34.png           pos=3, 4 のみ (Fig 4 差し替え用)
#   6. _max_single_log.png             pos=0..4 全部、縦軸 log スケール (Fig 4 補完)
#   7. _active_count_per_position.png  pos × prompt 別 active count (Fig 5 補完)
#   8. _reconstruction_rmse_log.png    RMSE 縦軸 log スケール (Fig 6 補完)
#
# 環境: llm2026-dev (matplotlib + pandas + numpy だけで動く)

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 文字化け対策
plt.rcParams["font.family"] = ["Hiragino Sans", "Apple SD Gothic Neo", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "outputs"

PM_CSV  = OUTDIR / "prelim_qwen3_4b_transcoder_layer_sweep_position_metrics.csv"
AGG_CSV = OUTDIR / "prelim_qwen3_4b_transcoder_layer_sweep_summary.csv"
LAYER_REFERENCE = 24  # note02 の対象 layer

pm  = pd.read_csv(PM_CSV)
agg = pd.read_csv(AGG_CSV)
xs  = sorted(pm["layer_idx"].unique())
n_positions = int(pm["position"].max()) + 1

print(f"loaded: {PM_CSV.name}  shape={pm.shape}")
print(f"loaded: {AGG_CSV.name} shape={agg.shape}")
print(f"layers: {xs[0]}..{xs[-1]} ({len(xs)} layers)")
print(f"positions: 0..{n_positions - 1}")


def _series_for(metric: str, position: int) -> list[float]:
    sub = pm[pm["position"] == position].sort_values("layer_idx")
    return sub[metric].tolist()


def _position_label(p: int) -> str:
    sub = pm[pm["position"] == p].iloc[0]
    c, k = sub["clean_token"], sub["corrupt_token"]
    return f"pos {p}: {c}" if c == k else f"pos {p}: {c} / {k}"


def _add_reference_line(ax):
    ax.axvline(LAYER_REFERENCE, color="gray", linestyle="--", alpha=0.6, linewidth=1.0)
    ax.text(
        LAYER_REFERENCE, ax.get_ylim()[1], "  L24 (note02 ref)",
        rotation=90, va="top", ha="left", fontsize=7, color="gray",
    )


# ────────────────────────────────────────────────────────────────────
# Group A: pos=3, 4 だけの 5 figure (max|Δ|, ‖Δ‖₂, Tanimoto, Jaccard, max_single)
# ────────────────────────────────────────────────────────────────────
POS34_COLORS = {3: "tab:red", 4: "tab:blue"}
POS34_MARKERS = {3: "o", 4: "s"}


def render_pos34_only(metric: str, title: str, ylabel: str, out_name: str) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.5))
    for p in [3, 4]:
        ys = _series_for(metric, p)
        ax.plot(
            xs, ys,
            color=POS34_COLORS[p],
            marker=POS34_MARKERS[p],
            markersize=5,
            linewidth=1.5,
            label=_position_label(p),
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("layer_idx")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(xs[::2])
    ax.grid(True, alpha=0.3)
    _add_reference_line(ax)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out_path = OUTDIR / out_name
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  saved: {out_path.name}")


render_pos34_only(
    "max_abs_delta",
    "max ⱼ |cleanⱼ − corruptⱼ|  (pos=3, 4 のみ)",
    "max |Δactivation|",
    "nb03_qwen3_4b_transcoder_layer_sweep_max_abs_delta_pos34.png",
)
render_pos34_only(
    "l2_delta",
    "‖clean − corrupt‖₂  (pos=3, 4 のみ)",
    "L2 of Δactivation",
    "nb03_qwen3_4b_transcoder_layer_sweep_l2_delta_pos34.png",
)
render_pos34_only(
    "tanimoto",
    "Tanimoto(clean, corrupt)  (pos=3, 4 のみ、連続 Jaccard)",
    "∑ⱼ min / ∑ⱼ max",
    "nb03_qwen3_4b_transcoder_layer_sweep_tanimoto_pos34.png",
)
render_pos34_only(
    "jaccard_active",
    "Jaccard(active sets)  (pos=3, 4 のみ、閾値 0 の binary Jaccard)",
    "|active∩| / |active∪|",
    "nb03_qwen3_4b_transcoder_layer_sweep_jaccard_pos34.png",
)
render_pos34_only(
    "max_single",
    "max ⱼ max(cleanⱼ, corruptⱼ)  (pos=3, 4 のみ)",
    "max single activation",
    "nb03_qwen3_4b_transcoder_layer_sweep_max_single_pos34.png",
)


# ────────────────────────────────────────────────────────────────────
# Group B: max_single 全 5 position を log y で見る
# ────────────────────────────────────────────────────────────────────
position_colors_full = {
    p: "tab:gray" if p < 3 else ("tab:red" if p == 3 else "tab:blue")
    for p in range(n_positions)
}
position_linestyles_full = {p: ":" if p < 3 else "-" for p in range(n_positions)}
position_alphas_full = {p: 0.5 if p < 3 else 1.0 for p in range(n_positions)}
position_markers_full = {
    p: "o" if p == 3 else ("s" if p == 4 else "x")
    for p in range(n_positions)
}

fig, ax = plt.subplots(figsize=(11.0, 5.0))
for p in range(n_positions):
    ys = _series_for("max_single", p)
    ax.plot(
        xs, ys,
        color=position_colors_full[p],
        linestyle=position_linestyles_full[p],
        alpha=position_alphas_full[p],
        marker=position_markers_full[p],
        markersize=5,
        linewidth=1.4,
        label=_position_label(p),
    )
ax.set_yscale("log")
ax.set_title("max ⱼ max(cleanⱼ, corruptⱼ)  (全 5 position、縦軸 log スケール)", fontsize=12)
ax.set_xlabel("layer_idx")
ax.set_ylabel("max single activation (log)")
ax.set_xticks(xs[::2])
ax.grid(True, alpha=0.3, which="both")
_add_reference_line(ax)
ax.legend(loc="best", fontsize=9, framealpha=0.9)
fig.tight_layout()
out = OUTDIR / "nb03_qwen3_4b_transcoder_layer_sweep_max_single_log.png"
fig.savefig(out, dpi=140)
plt.close(fig)
print(f"  saved: {out.name}")


# ────────────────────────────────────────────────────────────────────
# Group C: pos × prompt 別 active count (5 positions × {clean, corrupt} = 10 lines)
# ────────────────────────────────────────────────────────────────────
# pos 0..2 は causal mask により active_clean == active_corrupt なので、線がほぼ重なる
fig, ax = plt.subplots(figsize=(11.0, 5.5))
for p in range(n_positions):
    color = position_colors_full[p]
    marker = position_markers_full[p]
    base_label = _position_label(p)
    # clean
    ys_c = _series_for("active_clean", p)
    ax.plot(
        xs, ys_c,
        color=color, marker=marker, markersize=4,
        linewidth=1.4, linestyle="-",
        alpha=position_alphas_full[p],
        label=f"{base_label}  (clean)",
    )
    # corrupt
    ys_k = _series_for("active_corrupt", p)
    # pos 0..2 では clean と完全重複するので legend には出さない
    if p >= 3:
        ax.plot(
            xs, ys_k,
            color=color, marker=marker, markersize=4,
            linewidth=1.4, linestyle="--",
            alpha=position_alphas_full[p],
            label=f"{base_label}  (corrupt)",
        )

ax.set_yscale("log")
ax.set_title("Active feature count (f > 0) per (layer, position, prompt) — log y", fontsize=12)
ax.set_xlabel("layer_idx")
ax.set_ylabel("# active features (log)")
ax.set_xticks(xs[::2])
ax.grid(True, alpha=0.3, which="both")
_add_reference_line(ax)
ax.legend(loc="best", fontsize=8, framealpha=0.9, ncol=2)
fig.tight_layout()
out = OUTDIR / "nb03_qwen3_4b_transcoder_layer_sweep_active_count_per_position.png"
fig.savefig(out, dpi=140)
plt.close(fig)
print(f"  saved: {out.name}")


# ────────────────────────────────────────────────────────────────────
# Group D: Reconstruction RMSE 縦軸 log スケール
# ────────────────────────────────────────────────────────────────────
xs_agg = agg["layer_idx"].tolist()
fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11.0, 7.0), sharex=True)

# Top: RMSE log scale
ax_top.plot(xs_agg, agg["reconstruction_rmse_clean"],   marker="o", color="tab:green",  label="clean")
ax_top.plot(xs_agg, agg["reconstruction_rmse_corrupt"], marker="s", color="tab:orange", label="corrupt")
ax_top.set_yscale("log")
ax_top.set_ylabel("reconstruction RMSE (log)")
ax_top.set_title("Reconstruction quality across layers — RMSE log scale", fontsize=12)
ax_top.grid(True, alpha=0.3, which="both")
_add_reference_line(ax_top)
ax_top.legend(loc="best")

# Bottom: mean cosine (linear、parity check 用に残す)
ax_bot.plot(xs_agg, agg["reconstruction_mean_cos_clean"],   marker="o", color="tab:green",  label="clean")
ax_bot.plot(xs_agg, agg["reconstruction_mean_cos_corrupt"], marker="s", color="tab:orange", label="corrupt")
ax_bot.set_ylabel("reconstruction mean cosine (linear)")
ax_bot.set_xlabel("layer_idx")
ax_bot.set_xticks(xs_agg[::2])
ax_bot.grid(True, alpha=0.3)
_add_reference_line(ax_bot)
ax_bot.legend(loc="best")

fig.tight_layout()
out = OUTDIR / "nb03_qwen3_4b_transcoder_layer_sweep_reconstruction_log.png"
fig.savefig(out, dpi=140)
plt.close(fig)
print(f"  saved: {out.name}")

print("\nDone.")
