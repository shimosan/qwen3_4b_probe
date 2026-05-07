from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config() -> dict[str, Any]:
    cfg_path = project_root() / "configs" / "qwen3_4b_probe.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_scratch_dir(workspace_name: str) -> Path:
    explicit = os.environ.get("AIDEMO_SCRATCH_DIR")
    if explicit:
        return Path(explicit).expanduser()

    root = os.environ.get("AIDEMO_SCRATCH_ROOT")
    if root:
        return Path(root).expanduser() / workspace_name

    return Path.home() / "scratch" / "aidemo2026" / workspace_name


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
