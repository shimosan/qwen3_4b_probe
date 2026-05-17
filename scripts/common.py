from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_WORKSPACE = "qwen3_4b_probe"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config() -> dict[str, Any]:
    cfg_path = Path(__file__).parent / "qwen3_4b_probe.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_outputs_dir() -> Path:
    return ensure_dir(project_root() / "outputs")


# 現在 scripts/notebooks は outputs/ に統一済み。
# GPU サーバーとの分離が必要になった時点で再導入すれば良い。削除しても可。
def resolve_scratch_dir() -> Path:
    explicit = os.environ.get("AIDEMO_SCRATCH_DIR")
    if explicit:
        return Path(explicit).expanduser()

    root = os.environ.get("AIDEMO_SCRATCH_ROOT")
    if root:
        return Path(root).expanduser() / _WORKSPACE

    return Path.home() / "scratch" / "aidemo2026" / _WORKSPACE


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
