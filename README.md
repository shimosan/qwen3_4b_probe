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

## Structure

| Path | Contents |
|---|---|
| `scripts/` | 番号付き再現スクリプトと設定ファイル（JSON） |
| `notebooks/` | 探索・補助用 Jupyter Notebook |
| `outputs/` | 軽量な出力ファイル（PNG / CSV 等） |

## Setup

notebooks と core scripts（00–06）：

```bash
python3 -m venv ~/.venvs/llm2026
source ~/.venvs/llm2026/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
python scripts/01_download_model.py
```

実験スクリプト（07 以降）：

```bash
python3 -m venv ~/.venvs/llm2026-dev
source ~/.venvs/llm2026-dev/bin/activate
pip install -U pip wheel
pip install -r requirements-dev.txt
```

## Notes

This probe workspace should not modify Transformers source code.
Use `qwen3_4b_trace` for source-level tracing or modification.
