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
