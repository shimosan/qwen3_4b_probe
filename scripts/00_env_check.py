from __future__ import annotations

import platform
from pathlib import Path

import torch
import transformers
import huggingface_hub

from common import load_config, resolve_outputs_dir

cfg = load_config()
outputs_dir = resolve_outputs_dir()

print("python:", platform.python_version())
print("platform:", platform.platform())
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("huggingface_hub:", huggingface_hub.__version__)

print("mps built:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("cuda:", torch.version.cuda)
    print("cuda device:", torch.cuda.get_device_name(0))

print("workspace:", Path.cwd())
print("outputs_dir:", outputs_dir)
print("model_id:", cfg["model_id"])
