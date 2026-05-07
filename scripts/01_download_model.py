from __future__ import annotations

from huggingface_hub import snapshot_download

from common import load_config

cfg = load_config()
model_id = cfg["model_id"]

path = snapshot_download(
    repo_id=model_id,
    local_dir=None,
)

print("Downloaded model:")
print(model_id)
print("Cache path:")
print(path)
