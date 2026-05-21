# 04_probe_forward.py が保存した probe_forward_compact.pt を読み込み、
# attention heatmap（PNG）と attention 重みの CSV を outputs/ に保存する。
# 引数: --head <head番号> --label-mode <both|piece|position>
# 依存: 04_probe_forward.py を先に実行しておく必要がある。
# 環境: llm2026

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
from pandas.api.types import is_scalar
import torch

from common import resolve_outputs_dir


def set_japanese_font_if_available() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in [
        "Hiragino Sans",
        "Hiragino Maru Gothic Pro",
        "AppleGothic",
        "Noto Sans CJK JP",
        "Yu Gothic",
    ]:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def choose_token_text(row: pd.Series) -> str:
    for col in ["decoded", "piece", "token"]:
        if col not in row.index:
            continue
        val = row[col]
        if not is_scalar(val):
            continue
        if cast(bool, pd.isna(val)):
            continue
        s = str(val)
        if s:
            return s
    return ""


def make_labels(token_table: pd.DataFrame, label_mode: str) -> list[str]:
    labels = []
    for i, row in token_table.iterrows():
        piece = choose_token_text(row).replace("\n", "\\n")
        if len(piece) > 12:
            piece = piece[:12] + "…"

        if label_mode == "position":
            labels.append(str(i))
        elif label_mode == "piece":
            labels.append(piece if piece else str(i))
        else:
            labels.append(f"{i}:{piece}" if piece else str(i))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=int, default=0, help="attention head index")
    parser.add_argument(
        "--label-mode",
        choices=["both", "position", "piece"],
        default="both",
        help="tick label style",
    )
    args = parser.parse_args()

    outputs_dir = resolve_outputs_dir()
    compact_path = outputs_dir / "probe_forward_compact.pt"
    token_table_path = outputs_dir / "token_table.csv"

    if not compact_path.exists():
        raise FileNotFoundError(f"compact tensor file not found: {compact_path}")
    if not token_table_path.exists():
        raise FileNotFoundError(f"token table not found: {token_table_path}")

    obj = torch.load(compact_path, map_location="cpu")
    attention = obj["attention_layer0"]

    if attention.ndim != 4:
        raise ValueError(f"expected attention shape [batch, heads, seq, seq], got {attention.shape}")

    num_heads = attention.shape[1]
    if not (0 <= args.head < num_heads):
        raise ValueError(f"head must be in [0, {num_heads - 1}], got {args.head}")

    attn = attention[0, args.head].float().numpy()
    seq_len = attn.shape[0]

    token_table = pd.read_csv(token_table_path)
    labels = make_labels(token_table, args.label_mode)

    if len(labels) != seq_len:
        labels = [str(i) for i in range(seq_len)]

    set_japanese_font_if_available()

    fig_size = max(8, 0.32 * seq_len + 3)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(attn, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(f"Qwen3-4B attention heatmap: layer 0, head {args.head}")
    ax.set_xlabel("Key token")
    ax.set_ylabel("Query token")

    ax.set_xticks(range(seq_len))
    ax.set_yticks(range(seq_len))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    fig.tight_layout()

    out_png = outputs_dir / f"attention_layer0_head{args.head}_{args.label_mode}.png"
    out_csv = outputs_dir / f"attention_layer0_head{args.head}_{args.label_mode}.csv"

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    pd.DataFrame(attn, index=labels, columns=labels).to_csv(out_csv)

    print(f"saved: {out_png}")
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
