# qwen3_4b_probe

Qwen3-4B を用いた LLM 内部可視化の軽量調査 workspace。

## Purpose

2026 年度「情報AI基礎」講義デモ向けに、Qwen3-4B の内部計算を可視化する workspace。
主成果物は `notebooks/` 配下の Jupyter Notebook 群で、各ノートは単体で完結する設計。
`scripts/` はその前段として行った事前探査・再現確認のためのスクリプト群。

主題（notebooks）:

- `00_intro_chat.ipynb` — Qwen3-4B の読み込み、tokenizer / chat template、シングル・マルチターン chat による動作確認
- `01_tokenizer.ipynb` — 文字コード（Unicode / UTF-8）の基礎、tokenizer の `encode` / `decode`、token 分割の観察
- `02_residual_stream_logit_lens_patching.ipynb` — 入口（embedding）と出口（`lm_head` + softmax）の対応、residual stream と `hidden_states`、Logit Lens、Activation Patching

## Model

- `Qwen/Qwen3-4B`

## Environment

- probe venv: `~/.venvs/llm2026`
- model cache: Hugging Face cache
- large runtime outputs: scratch directory resolved by scripts

## Structure

| Path | Contents |
|---|---|
| `notebooks/` | 講義デモ用 Jupyter Notebook（主成果物。各ノートが自己完結） |
| `scripts/` | 事前探査・補助用の番号付きスクリプト（tokenizer / shape 確認 / attention heatmap / logit lens 検証 等）と設定ファイル |
| `outputs/` | 軽量な出力ファイル（PNG / CSV 等、Git 管理外） |

## Setup

notebooks と core scripts（00–06）：

```bash
python3 -m venv ~/.venvs/llm2026
source ~/.venvs/llm2026/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
nbstripout --install --keep-id   # notebook output を commit 時に自動除去（clone 後 1 回だけ）
python scripts/01_download_model.py
```

`nbstripout --install --keep-id` は `.git/config` に filter コマンド、`.git/info/attributes` に `*.ipynb filter=nbstripout` を登録します。clone ごとに 1 回だけ実行すれば、以降の `git commit` で notebook の output が自動的に除去されます。

実験スクリプト（07 以降）：

```bash
python3 -m venv ~/.venvs/llm2026-dev
source ~/.venvs/llm2026-dev/bin/activate
pip install -U pip wheel
pip install -r requirements-dev.txt
```

## Usage

### notebooks

`llm2026` venv を activate した状態で、`notebooks/` を作業ディレクトリ (cwd) として起動してください。VS Code の Jupyter 拡張で `.ipynb` を直接開く場合は、cwd がそのファイルのあるディレクトリになるため自動で OK です。

```bash
source ~/.venvs/llm2026/bin/activate
cd notebooks
jupyter lab          # またはお好みの Jupyter 環境
```

### scripts

`scripts/` 配下の番号付きスクリプトは、どこから実行しても動作します（出力先はプロジェクトルート直下の `outputs/` に解決されます）。

```bash
# core scripts (00-06) は llm2026 venv で
source ~/.venvs/llm2026/bin/activate
python scripts/00_env_check.py
python scripts/04_probe_forward.py

# 実験スクリプト (07 以降) は llm2026-dev venv で
source ~/.venvs/llm2026-dev/bin/activate
python scripts/08_logit_lens.py
```

実行順序や依存関係（例: 06 は 04 の後に実行）は `CLAUDE.md` を参照してください。

## Notes

This probe workspace should not modify Transformers source code.
Use `qwen3_4b_trace` for source-level tracing or modification.
