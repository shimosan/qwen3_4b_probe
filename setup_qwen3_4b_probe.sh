#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_NAME="qwen3_4b_probe"
PROJECT_ROOT="$HOME/.../aidemo2026"
WORKSPACE="$PROJECT_ROOT/$WORKSPACE_NAME"
SCRATCH_ROOT="${AIDEMO_SCRATCH_ROOT:-$HOME/scratch/aidemo2026}"
SCRATCH="$SCRATCH_ROOT/$WORKSPACE_NAME"
VENV="$HOME/.venvs/llm2026"
MODEL_ID="Qwen/Qwen3-4B"

echo "===== setup: $WORKSPACE_NAME ====="

mkdir -p "$WORKSPACE"/{scripts,outputs,docs,configs,.cursor/rules,.vscode}
mkdir -p "$SCRATCH"
mkdir -p "$HOME/.venvs"

cd "$WORKSPACE"

echo "===== initialize git repo if needed ====="
if [ ! -d .git ]; then
  git init
fi

echo "===== write README / docs ====="
cat > README.md <<'TXT'
# qwen3_4b_probe

Qwen3-4B を用いた LLM 内部可視化の軽量調査 workspace。

## Purpose

- tokenizer の確認
- token table の作成
- 既存 Transformers API による hidden states / attentions / logits の取得
- next-token distribution の確認
- attention heatmap 作成の準備

## Model

- `Qwen/Qwen3-4B`

## Environment

- probe venv: `~/.venvs/llm2026`
- model cache: Hugging Face cache
- large runtime outputs: scratch directory resolved by scripts

## Notes

This probe workspace should not modify Transformers source code.
Use `qwen3_4b_trace` for source-level tracing or modification.
TXT

cat > docs/experiment_log.md <<'TXT'
# Experiment log

## Initial setup

- workspace: `~/.../aidemo2026/qwen3_4b_probe`
- venv: `~/.venvs/llm2026`
- model: `Qwen/Qwen3-4B`
- purpose: probe Qwen3-4B using existing Transformers APIs

TXT

echo "===== write .gitignore / .cursorignore ====="
cat > .gitignore <<'TXT'
# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# Local environments
.venv/
venv/
env/

# Runtime outputs
outputs/
runs/
logs/
cache/

# Local scratch links or local-only dirs inside workspace, if any
scratch/
tmp/

# Secrets / local env
.env
.env.*
huggingface_token*

# OS/editor noise
.DS_Store
TXT

cp .gitignore .cursorignore

echo "===== write Cursor rules ====="
cat > .cursor/rules/project.mdc <<'TXT'
---
description: Project-wide rules for AI lecture demo workspaces
globs:
  - "**/*"
alwaysApply: true
---

# Project rules

- This workspace is part of the 2026 AI lecture demo project.
- Keep code simple, explicit, and suitable for lecture demonstrations.
- Do not place model weights, Hugging Face cache files, virtual environments, or large runtime outputs in the Git-managed workspace.
- Runtime outputs should be written to runtime folders such as `outputs/`, `runs/`, `logs/`, or to the scratch directory resolved by project scripts.
- Large outputs should use the standard scratch resolver:
  1. `AIDEMO_SCRATCH_DIR` if set.
  2. `AIDEMO_SCRATCH_ROOT / <workspace_name>` if `AIDEMO_SCRATCH_ROOT` is set.
  3. `Path.home() / "scratch" / "aidemo2026" / <workspace_name>` otherwise.
- Do not hard-code machine-specific absolute paths in Python scripts.
- Keep README and `docs/experiment_log.md` synchronized with meaningful changes.
TXT

cat > .cursor/rules/python.mdc <<'TXT'
---
description: Python coding rules
globs:
  - "**/*.py"
alwaysApply: true
---

# Python rules

- Use clear, minimal Python scripts rather than large notebooks for reproducible demos.
- Use `pathlib.Path` where practical.
- Put user-adjustable parameters near the top of each script or in `configs/*.json`.
- Save large tensors and generated media outside the workspace, under the configured scratch directory.
- Print concise progress messages so terminal logs are understandable.
- Avoid network downloads except in scripts explicitly named for downloading or setup.
TXT

cat > .cursor/rules/llm_probe.mdc <<'TXT'
---
description: Qwen3-4B probe workspace rules
globs:
  - "scripts/**/*.py"
  - "configs/**/*.json"
alwaysApply: true
---

# Qwen3-4B probe rules

- This workspace uses `Qwen/Qwen3-4B`.
- The default probe environment is named `llm2026`.
- Use the pip-installed Transformers package in this probe workspace.
- Do not modify Transformers source code here.
- Use existing APIs such as `output_hidden_states=True`, `output_attentions=True`, and PyTorch hooks.
- Use `attn_implementation="eager"` when attention weights or Python-level debugging are needed.
- Keep sequence lengths short when saving hidden states or attentions.
- Save large tensor outputs to the scratch directory resolved by project scripts.
TXT

echo "===== write VS Code / Cursor settings ====="
cat > .vscode/settings.json <<'TXT'
{
  "python.defaultInterpreterPath": "${userHome}/.venvs/llm2026",
  "python.terminal.activateEnvironment": true,

  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.diagnosticMode": "openFilesOnly",

  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true
  },

  "search.exclude": {
    "outputs": true,
    "runs": true,
    "logs": true,
    "cache": true,
    "scratch": true,
    "tmp": true,
    "**/__pycache__": true
  }
}
TXT

cat > .vscode/launch.json <<'TXT'
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug generate smoke",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/scripts/03_generate_smoke.py",
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "Debug forward probe",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/scripts/04_probe_forward.py",
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
TXT

echo "===== write config ====="
cat > configs/qwen3_4b_probe.json <<'TXT'
{
  "workspace_name": "qwen3_4b_probe",
  "model_id": "Qwen/Qwen3-4B",
  "default_prompt": "京都大学の情報学科1回生に、言語モデルとは何かを短く説明してください。",
  "max_new_tokens": 64,
  "attn_implementation": "eager",
  "output_hidden_states": true,
  "output_attentions": true,
  "use_cache": false
}
TXT

echo "===== create venv ====="
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"

# Keep setuptools compatible with torch 2.11.0.
python -m pip install -U pip wheel "setuptools<82"

echo "===== install packages ====="
python -m pip install -U torch torchvision torchaudio
python -m pip install -U "transformers>=4.51.0" accelerate safetensors sentencepiece protobuf pandas matplotlib huggingface_hub

echo "===== write helper module ====="
cat > scripts/common.py <<'PY'
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
PY

echo "===== write Python scripts ====="
cat > scripts/00_env_check.py <<'PY'
from __future__ import annotations

import platform
from pathlib import Path

import torch
import transformers
import huggingface_hub

from common import load_config, resolve_scratch_dir, ensure_dir

cfg = load_config()
workspace_name = cfg["workspace_name"]
scratch_dir = ensure_dir(resolve_scratch_dir(workspace_name))

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
print("scratch_dir:", scratch_dir)
print("model_id:", cfg["model_id"])
PY

cat > scripts/01_download_model.py <<'PY'
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
PY

cat > scripts/02_tokenizer_probe.py <<'PY'
from __future__ import annotations

import pandas as pd
from transformers import AutoTokenizer

from common import load_config

cfg = load_config()
model_id = cfg["model_id"]
prompt = cfg["default_prompt"]

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": prompt}]
chat_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

enc = tokenizer(chat_text, return_tensors="pt")
ids = enc["input_ids"][0].tolist()

rows = []
cumulative_ids = []
for pos, token_id in enumerate(ids):
    cumulative_ids.append(token_id)
    raw_token = tokenizer.convert_ids_to_tokens([token_id])[0]
    decoded_piece = tokenizer.decode([token_id])
    cumulative_decoded_text = tokenizer.decode(cumulative_ids)
    rows.append(
        {
            "position": pos,
            "token_id": token_id,
            "raw_token": raw_token,
            "decoded_piece": decoded_piece,
            "cumulative_decoded_text": cumulative_decoded_text,
        }
    )

df = pd.DataFrame(rows)
df.to_csv("outputs/token_table.csv", index=False)

print("MODEL_ID:", model_id)
print("input text:")
print(prompt)
print()
print("chat template text:")
print(chat_text)
print()
print("num tokens:", len(ids))
print()
print(df[["position", "token_id", "raw_token", "decoded_piece"]].head(80).to_string(index=False))
print()
print("saved: outputs/token_table.csv")
PY

cat > scripts/03_generate_smoke.py <<'PY'
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_config

cfg = load_config()
model_id = cfg["model_id"]
prompt = cfg["default_prompt"]

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print("model_id:", model_id)
print("device:", device)
print("dtype:", dtype)

tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = tokenizer(text, return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    attn_implementation=cfg["attn_implementation"],
)
model.to(device)
model.eval()

with torch.no_grad():
    generated = model.generate(
        **inputs,
        max_new_tokens=cfg["max_new_tokens"],
        do_sample=False,
    )

out = tokenizer.decode(generated[0], skip_special_tokens=False)
print(out)

with open("outputs/generate_smoke.txt", "w", encoding="utf-8") as f:
    f.write(out)
    f.write("\n")
PY

cat > scripts/04_probe_forward.py <<'PY'
from __future__ import annotations

import json

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import load_config, resolve_scratch_dir, ensure_dir

cfg = load_config()
model_id = cfg["model_id"]
prompt = cfg["default_prompt"]
scratch_dir = ensure_dir(resolve_scratch_dir(cfg["workspace_name"]))

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print("model_id:", model_id)
print("device:", device)
print("dtype:", dtype)
print("scratch_dir:", scratch_dir)

tokenizer = AutoTokenizer.from_pretrained(model_id)
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

inputs = tokenizer(text, return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    attn_implementation=cfg["attn_implementation"],
)
model.to(device)
model.eval()

with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=cfg["output_hidden_states"],
        output_attentions=cfg["output_attentions"],
        use_cache=cfg["use_cache"],
    )

shape_info = {
    "logits": list(outputs.logits.shape),
    "num_hidden_states": len(outputs.hidden_states) if outputs.hidden_states is not None else 0,
    "hidden_state_shapes": [list(x.shape) for x in outputs.hidden_states] if outputs.hidden_states is not None else [],
    "num_attentions": len(outputs.attentions) if outputs.attentions is not None else 0,
    "attention_shapes": [list(x.shape) for x in outputs.attentions] if outputs.attentions is not None else [],
}

with open("outputs/shape_info.json", "w", encoding="utf-8") as f:
    json.dump(shape_info, f, indent=2, ensure_ascii=False)

last_logits = outputs.logits[0, -1].float()
probs = torch.softmax(last_logits, dim=-1)
top = torch.topk(probs, k=20)

rows = []
for rank, (idx, prob) in enumerate(zip(top.indices.tolist(), top.values.tolist()), start=1):
    rows.append(
        {
            "rank": rank,
            "token_id": idx,
            "raw_token": tokenizer.convert_ids_to_tokens([idx])[0],
            "decoded_piece": tokenizer.decode([idx]),
            "prob": prob,
        }
    )

pd.DataFrame(rows).to_csv("outputs/next_token_top20.csv", index=False)

tensor_path = scratch_dir / "probe_forward_compact.pt"
payload = {
    "input_ids": inputs["input_ids"].detach().cpu(),
    "logits_last": outputs.logits[:, -1, :].detach().cpu(),
}

if outputs.hidden_states is not None:
    payload["hidden_last_layer"] = outputs.hidden_states[-1].detach().cpu()

if outputs.attentions is not None and len(outputs.attentions) > 0:
    payload["attention_layer0"] = outputs.attentions[0].detach().cpu()

torch.save(payload, tensor_path)

print(json.dumps(shape_info, indent=2, ensure_ascii=False))
print("saved: outputs/shape_info.json")
print("saved: outputs/next_token_top20.csv")
print("saved:", tensor_path)
PY

cat > scripts/05_show_transformers_source.py <<'PY'
from __future__ import annotations

import inspect

import transformers
from transformers.models.qwen3 import modeling_qwen3

print("transformers version:", transformers.__version__)
print("transformers file:", transformers.__file__)
print("qwen3 modeling file:", modeling_qwen3.__file__)
print()

for name in ["Qwen3ForCausalLM", "Qwen3Model", "Qwen3DecoderLayer", "Qwen3Attention", "Qwen3MLP"]:
    obj = getattr(modeling_qwen3, name, None)
    if obj is None:
        print(name, "not found")
    else:
        print(name, "->", inspect.getfile(obj))
PY

echo "===== freeze requirements ====="
python -m pip freeze > requirements.txt

echo "===== pip check ====="
python -m pip check | tee outputs/pip_check.txt

echo "===== run lightweight checks ====="
python scripts/00_env_check.py | tee outputs/env_check.txt
python scripts/01_download_model.py | tee outputs/download_model.txt
python scripts/02_tokenizer_probe.py | tee outputs/tokenizer_probe.txt
python scripts/05_show_transformers_source.py | tee outputs/transformers_source.txt

echo "===== done: $WORKSPACE_NAME ====="
echo "Workspace: $WORKSPACE"
echo "Scratch:   $SCRATCH"
echo "Venv:      $VENV"
echo
echo "Open in Cursor:"
echo "  cursor $WORKSPACE"
echo
echo "Activate venv:"
echo "  source $VENV/bin/activate"
