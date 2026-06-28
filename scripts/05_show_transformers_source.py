# インストール済み Transformers パッケージの Qwen3 実装ファイルのパスを表示する。
# pip install 版の transformers がどこにあるかを確認するためのユーティリティスクリプト。
# 環境: aidemo2026

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
