# CLAUDE.md

このファイルは、Claude Code がこのリポジトリで作業するときのための作業方針をまとめたものです。

## プロジェクト概要

このリポジトリは、2026年度「情報AI基礎」講義デモ用の **Qwen3-4B probe workspace** です。

対象モデルは以下です。

```text
Qwen/Qwen3-4B
```

目的は、Hugging Face Transformers の既存 API を使って、Qwen3-4B の内部状態を観察・可視化することです。

主に扱う対象は以下です。

```text
- tokenization
- chat template 展開
- hidden states
- attention weights
- logits
- next-token distribution
- attention heatmap
```

この workspace は **probe 用**です。  
Transformers のソースコードを改変して内部を追跡するための workspace ではありません。

---

## この workspace の役割

このリポジトリは以下の workspace です。

```text
qwen3_4b_probe
```

役割は以下です。

```text
probe:
  pip install 版 Transformers を使い、
  既存 API、
  output_hidden_states=True、
  output_attentions=True、
  必要に応じた軽い PyTorch hook により、
  モデル内部を観察する。
```

このリポジトリでは、Transformers のソースコードを改変しないでください。

Qwen3 の実装ファイル `modeling_qwen3.py` に breakpoint を張る、`Qwen3Attention` や `Qwen3MLP` を改変する、RoPE や attention score の内部計算を追う、などの作業は、別 workspace で行います。

```text
qwen3_4b_trace
```

Transformers の改変可能な source tree は以下です。

```text
~/.../.../transformers_qwen
```

---

## 基本モデル設定

デフォルトモデルは以下です。

```text
Qwen/Qwen3-4B
```

重要な設定は以下です。

```text
attn_implementation = "eager"
output_hidden_states = true
output_attentions = true
use_cache = false
```

attention weights を取得する必要がある場合は、原則として以下を使います。

```python
attn_implementation="eager"
```

高速化された attention 実装では、講義デモに必要な形で attention weights が返らない場合があります。

hidden states や attentions を保存するとメモリを使うため、prompt と sequence length は短く保ってください。

---

## Python 環境

標準の Python 環境は以下です。

```text
~/.venvs/llm2026
```

実行前に activate します。

```bash
cd ~/.../aidemo2026/qwen3_4b_probe
source ~/.venvs/llm2026/bin/activate
```

この workspace では、原則として **pip install 版 Transformers** を使います。

想定する主要パッケージは以下です。

```text
Python 3.12.x
torch
transformers
huggingface_hub
accelerate
safetensors
sentencepiece
protobuf
pandas
matplotlib
```

Mac では MPS、Windows / GPU サーバーでは CUDA が使える場合はそれを利用します。

---

## scratch directory の方針

大きな中間ファイルは Git 管理下の workspace に保存しないでください。

標準の scratch 解決規則は以下です。

```text
1. AIDEMO_SCRATCH_DIR が設定されていれば、それを使う。
2. AIDEMO_SCRATCH_ROOT が設定されていれば、AIDEMO_SCRATCH_ROOT / <workspace_name> を使う。
3. それ以外では、Path.home() / "scratch" / "aidemo2026" / <workspace_name> を使う。
```

この workspace の標準 scratch は以下です。

```text
~/scratch/aidemo2026/qwen3_4b_probe
```

大きめの tensor は、例えば以下に保存します。

```text
~/scratch/aidemo2026/qwen3_4b_probe/probe_forward_compact.pt
```

モデル重み、Hugging Face cache、大きな tensor、大量の生成物を workspace 内に保存しないでください。

---

## Hugging Face cache の方針

モデル本体は workspace に置きません。

通常は Hugging Face cache に置きます。

```text
~/.cache/huggingface/hub/models--Qwen--Qwen3-4B
```

モデルのダウンロードは、明示的に download 用と分かる script でのみ行います。

例：

```text
scripts/01_download_model.py
```

通常の解析 script や可視化 script に、暗黙のダウンロード処理を追加しないでください。

---

## リポジトリ構成

想定する構成は以下です。

```text
qwen3_4b_probe/
  CLAUDE.md
  README.md
  requirements.txt
  .gitignore
  .cursorignore
  .cursor/
    rules/
      project.mdc
      python.mdc
      llm_probe.mdc
  .vscode/
    settings.json
    launch.json
  configs/
    qwen3_4b_probe.json
  scripts/
    00_env_check.py
    01_download_model.py
    02_tokenizer_probe.py
    03_generate_smoke.py
    04_probe_forward.py
    05_show_transformers_source.py
    06_attention_heatmap.py
  outputs/
  docs/
    experiment_log.md
```

`outputs/` は runtime output 用のフォルダであり、Git 管理対象ではありません。

script で `outputs/` に保存する場合は、必ず事前にディレクトリを作成してください。

```python
from pathlib import Path

Path("outputs").mkdir(parents=True, exist_ok=True)
```

---

## script の実行順序

script の番号には意味があります。  
番号順の構成をなるべく保ってください。

標準的な実行順序は以下です。

```bash
python scripts/00_env_check.py
python scripts/01_download_model.py
python scripts/02_tokenizer_probe.py
python scripts/03_generate_smoke.py
python scripts/04_probe_forward.py
python scripts/06_attention_heatmap.py --head 0 --label-mode both
```

依存関係は特に以下に注意してください。

```text
04_probe_forward.py
  -> scratch に probe_forward_compact.pt を保存する

06_attention_heatmap.py
  -> probe_forward_compact.pt を読む
  -> outputs/ に attention heatmap の PNG / CSV を保存する
```

したがって、`06_attention_heatmap.py` は `04_probe_forward.py` の後に実行します。

---

## 主な出力ファイル

軽量な出力は `outputs/` に保存して構いません。

例：

```text
token_table.csv
generate_smoke.txt
shape_info.json
next_token_top20.csv
attention_layer0_head0_both.csv
attention_layer0_head0_both.png
```

大きめの tensor は scratch に保存します。

例：

```text
probe_forward_compact.pt
```

compact tensor には、講義デモに必要な最小限の情報だけを入れます。

例：

```text
input_ids
logits_last
hidden_last_layer
attention_layer0
```

全 layer の hidden states や attentions を丸ごと保存することは、明示的に必要な場合を除いて避けます。

---

## コーディング方針

Python script は、講義デモで説明しやすいように、簡潔で明示的に書いてください。

基本方針：

```text
- notebook よりも再現しやすい Python script を優先する。
- pathlib.Path を使う。
- 調整可能なパラメータは configs/*.json または script 冒頭にまとめる。
- scripts/common.py に既存 utility がある場合はそれを使う。
- 出力先ディレクトリは保存前に作成する。
- 進捗が分かる簡潔な print を入れる。
- 大きな出力は scratch に保存する。
- マシン固有の絶対パスをハードコードしない。
- download / setup 用 script 以外に暗黙のネットワークアクセスを入れない。
```

講義デモ用なので、技巧的な実装よりも、読んで分かる実装を優先してください。

---

## forward probe の基本形

Qwen3-4B の内部状態を見るときは、概ね以下の形を使います。

```python
outputs = model(
    **inputs,
    output_hidden_states=True,
    output_attentions=True,
    use_cache=False,
)
```

短い日本語 prompt の代表的な shape は以下です。

```text
input length:
  35 tokens

logits:
  [1, 35, 151936]

hidden states:
  37 tensors
  each [1, 35, 2560]

attentions:
  36 tensors
  each [1, 32, 35, 35]
```

prompt やモデル設定を変えれば shape も変わります。

---

## attention heatmap の方針

attention heatmap は講義デモ用の図です。

軸の意味を明確にしてください。

```text
横軸:
  key token / 参照される token

縦軸:
  query token / 参照する token
```

label mode は以下を想定します。

```text
both
piece
position
```

token label が混みすぎる場合は、無理に全部表示せず、`position` 表示にするか、prompt を短くしてください。

autoregressive LM では、未来 token を見られないため、通常は causal mask により右上三角が暗く見えます。  
この性質は講義で説明しやすいので、heatmap 作成時には意識してください。

---

## Git の注意

明示的に指示されない限り、広い範囲をまとめて stage しないでください。

避ける例：

```bash
git add .
```

推奨：

```bash
git add CLAUDE.md
git add scripts/06_attention_heatmap.py
```

特に以下には注意してください。

```text
.vscode/settings.json
outputs/
scratch/
```

`.vscode/settings.json` は Cursor / Pyright 由来のローカル変更が入りやすいです。  
エディタ設定を変更するタスクでない限り、commit しないでください。

以下は commit しないでください。

```text
outputs/
runs/
logs/
cache/
scratch/
tmp/
*.pt
大きな tensor
モデル重み
Hugging Face cache
.env
.env.*
token や secret
```

---

## やってはいけないこと

この repository では、以下を避けてください。

```text
- Transformers ソースコードを改変する。
- モデル重みを repository に入れる。
- 大きな tensor を workspace 内に保存する。
- /Users/shimo/... や C:\Users\<name>\... のような絶対パスを script に直書きする。
- 通常の probe script に暗黙の download 処理を追加する。
- 明示的な指示なしに model_id を Qwen/Qwen3-4B から変える。
- attention 保存時に sequence length を大きくする。
- 生成物を commit する。
- 番号付き script 群を、理由なく巨大な単一 script にまとめる。
```

---

## 他 workspace との関係

関連 workspace は以下です。

```text
qwen3_4b_probe:
  この repository
  pip install 版 Transformers
  既存 API と軽い hook による観察

qwen3_4b_trace:
  source tracing 用 workspace
  editable install 版 Transformers
  Qwen3 実装内部への breakpoint / 必要に応じた改変

qwen3_8b_probe:
  将来の Qwen3-8B 横展開用

llmjp4_8b_probe:
  将来の日本語 LLM 比較用
```

この repository での変更は、Qwen3-4B probe workflow に集中させてください。

---

## 講義デモとしての優先事項

この code は、情報学科1回生向けの講義デモに使うことを想定しています。

優先するもの：

```text
- 分かりやすさ
- 再現性
- 出力ファイルの意味の明確さ
- 短い prompt での安定動作
- 図や CSV による説明しやすさ
```

よい demo output の例：

```text
- chat template の special token を含む token table
- next-token top-k distribution
- hidden-state shape summary
- attention tensor shape summary
- attention heatmap
- 短い日本語生成結果
```

この workspace の目的は、生成品質を最大化することではありません。  
目的は、言語モデルの内部計算を、実際の tensor や図を通して見える形にすることです。
