# スクリプト間で共有するユーティリティ関数モジュール。
# load_config()        : scripts/qwen3_4b_probe.json を読み込む
# resolve_outputs_dir(): outputs/ ディレクトリのパスを返す（なければ作成）
# ensure_dir()         : 指定パスのディレクトリを作成して返す

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config() -> dict[str, Any]:
    cfg_path = Path(__file__).parent / "qwen3_4b_probe.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_outputs_dir() -> Path:
    return ensure_dir(project_root() / "outputs")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
