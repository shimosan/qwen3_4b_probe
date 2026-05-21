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

この workspace では venv を用途別に分けます。

```text
~/.venvs/llm2026      notebooks + core scripts (00-06) 用
~/.venvs/llm2026-dev  実験スクリプト (07 以降) 用（llm2026 の上位互換）
```

管理ファイル：

```text
requirements.txt      llm2026 の pip freeze
requirements-dev.txt  llm2026-dev の pip freeze
```

notebooks のカーネルには `llm2026` を使います。  
scripts/07 以降は `llm2026-dev` を前提とします。

activate：

```bash
source ~/.venvs/llm2026/bin/activate      # notebooks / core scripts
source ~/.venvs/llm2026-dev/bin/activate  # 実験スクリプト
```

この workspace では、原則として **pip install 版 Transformers** を使います。

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
  requirements-dev.txt
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
  scripts/
    common.py
    qwen3_4b_probe.json
    00_env_check.py
    01_download_model.py
    02_tokenizer_probe.py
    03_generate_smoke.py
    04_probe_forward.py
    05_show_transformers_source.py
    06_attention_heatmap.py
    07_hidden_state_mapping.py
    08_logit_lens.py
    09_embedding_unembedding.py
    10_prelim_compare_logit_lens_transformerlens.py
    11_prelim_compare_logit_lens_float32.py
    12_residual_stream_patching.py
  notebooks/
  outputs/
  logs/
  docs/
    images/
```

`outputs/` は runtime output 用のフォルダで、Git 管理対象**外**です（PNG / CSV / JSON など）。

`logs/` は実行ログおよび drafts 用の Git 管理対象**外**フォルダです（`*.log` と作業中の `*.md`）。

`docs/` は完成品の実験レポート md を置く Git 管理対象フォルダです。`docs/images/` には report が参照する figure を `outputs/` から **cp**（mv ではない）してきます。学生配布や永続的なドキュメントは docs/ に置きます。

script で `outputs/` に保存する場合は、必ず事前にディレクトリを作成してください。

```python
from pathlib import Path

Path("outputs").mkdir(parents=True, exist_ok=True)
```

---

## script の実行順序

script の番号には意味があります。  
番号順の構成をなるべく保ってください。

core scripts（`llm2026` で実行）：

```bash
python scripts/00_env_check.py
python scripts/01_download_model.py
python scripts/02_tokenizer_probe.py
python scripts/03_generate_smoke.py
python scripts/04_probe_forward.py
python scripts/06_attention_heatmap.py --head 0 --label-mode both
```

依存関係：

```text
04_probe_forward.py  ->  outputs/ に各種 CSV / JSON を保存
06_attention_heatmap.py  ->  04 の後に実行
```

実験スクリプト（`llm2026-dev` で実行、07 以降）：

```bash
python scripts/07_hidden_state_mapping.py
python scripts/08_logit_lens.py
python scripts/09_embedding_unembedding.py
python scripts/12_residual_stream_patching.py
# 以下は TransformerLens との比較検証
python scripts/10_prelim_compare_logit_lens_transformerlens.py
python scripts/11_prelim_compare_logit_lens_float32.py
```

scripts と notebooks は独立しています。notebooks は `llm2026` で動作します。

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

## 実験レポート md (docs/) の方針

実験スクリプトを実行した後に、実験者の報告書として md を書きます。
log file（テキスト実行ログ）とは別物で、これは**実行後に整理して作成する完成品**です。

### 三層の役割分担

```text
outputs/   git 管理外。script の生成物（PNG / CSV / JSON）。再生成可能、永続性なし。
logs/      git 管理外。実行ログ (*.log) と作業中の md ドラフト。
docs/      git 管理。完成品の実験レポート md。学生配布対象。
docs/images/  git 管理。docs/*.md が参照する figure を outputs/ から cp する。
```

- `outputs/` と `logs/` は再生成可能で永続性がないので、**完成版のレポート**は必ず `docs/` に置く。
- figure は `outputs/` から `docs/images/` へ **cp**（mv ではない）。outputs/ にも原本を残す。
- 1 script = 1 docs md。同じ script を複数回更新しても**新ファイルを作らず**、md を更新する。

### ファイル命名

```text
docs/{script番号}_{短い slug}.md
docs/images/{outputs と同じファイル名}.png
```

例: `docs/14_qwen3_4b_transcoder_layers23_24_25.md`、`docs/images/nb03_qwen3_4b_transcoder_layer24_feature_heatmap.png`

### ルール A — すべての report に共通する標準セクション

最低限以下の章立てを満たす：

1. **概要**（script リンク、最終更新日、ステータス）
2. **目的**
3. **実験設定**
4. **結果概要**（数値表は列の意味を明記）
5. **図**（`![caption](images/xxx.png)` で embed、`**Figure N**:` キャプション + 軸 / colormap / 系列の説明）
6. **解釈**
7. **応用への示唆**（講義デモへの活用、関連する notebook（nb02 / nb03 など）への寄与、再利用したい figure の指針）
8. **開発の経緯**（複数 stage の更新があれば）
9. **出力ファイル**（`outputs/` の manifest）
10. **注意事項**
11. **関連実験**（他 report への cross-link）

その他:

- frontmatter は使わない（Obsidian notes と違って素 md）
- script への link は `[scripts/14_xx.py](../scripts/14_xx.py)` のように workspace 相対
- 画像 path は `images/...`（docs/ 直下から見て）
- 数式は KaTeX 記法 `$ ... $` / `$$ ... $$`。Cursor 内蔵 Preview は非対応、`Cmd+Shift+V` の正規 Markdown Preview / Obsidian / GitHub で render される

### ルール B — 入門レベルの説明を強化する場合（実験の最初の md など）

ルール A の章の前半に、以下の解説セクションを挿入する：

- **背景**: ルール A の **目的 (2)** と **実験設定 (3)** の間に挿入。
  その実験で使う解析道具（SAE、transcoder、logit lens 等）が何か、数式で定義する。Transformer block の中での位置づけ、対象モデルとの関係を明示。
- **方法**: ルール A の **実験設定 (3)** と **結果概要 (4)** の間に挿入。
  パイプライン全体を**数式とコード併記**で説明（hook の役割、encode/decode の式とコードの対応）。読者が「結果テーブルを見る前に、何がどう計算されているか」を理解できるようにする。
- **測定する指標の定義**: 同じく **3-4 の間**、方法の直後に置く。
  表で各指標を**数式 + コード**で定義（`active_fraction`, `pos3 max|Δ|`, `reconstruction RMSE` など）。
- **「position」が何を意味するか**の表: 実験設定の中、または指標定義の直前に置く。
  pos=0..N-1 がどのトークンに対応し、どの比較に使うか。
- **結果テーブルの直前に列の意味を箇条書き**で明示。

→ B 適用後の章立て例:

1. 概要
2. 目的
3. **背景** ← B
4. 実験設定
5. **方法** ← B
6. **測定する指標の定義** ← B
7. 結果概要
8. 図
9. 解釈
10. 応用への示唆
11. 開発の経緯
12. 出力ファイル
13. 注意事項
14. 関連実験

これは「同じ workspace の文脈を知らない読者（学生）」が単独で読めるレベルまで踏み込む方針。
全 report に B を適用する必要はない。実験の中核となる report、講義デモに直接使う report に適用する。

### docs/ への作業フロー

```text
1. 実験スクリプトを実行 → outputs/ に PNG / CSV ができる
2. logs/{番号}_{slug}.md にドラフト作成（実験中で繰り返し更新する場合はここ）
3. レポートが固まったら docs/{番号}_{slug}.md に cp
4. docs/{番号}_{slug}.md の中の画像 path を「../outputs/xxx」から「images/xxx」に書き換え
5. 参照されている PNG / CSV を outputs/ から docs/images/ に cp
6. git add docs/ で commit
```

logs/ に書かずいきなり docs/ に書く運用も可。実験中のドラフト保存場所として logs/ を活用するかは個別判断。

---

## コーディング方針

Python script は、講義デモで説明しやすいように、簡潔で明示的に書いてください。

基本方針：

```text
- notebook よりも再現しやすい Python script を優先する。
- pathlib.Path を使う。
- 調整可能なパラメータは scripts/qwen3_4b_probe.json または script 冒頭にまとめる。
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

## notebook の作業フロー

原則として `notebooks/` で直接作業してよい。  
大規模な改修や、壊れるリスクが高い変更の場合は、`sandbox/` のコピーで検証してから `notebooks/` に反映してもよい。

```text
1. 通常の修正は notebooks/ で直接行う
2. 大きな改修・破壊的変更が心配なときは sandbox/ で先に検証してから notebooks/ に反映する
3. notebook の実行・出力上書きは、ユーザーから明示的に指示があったときのみ行う
```

`sandbox/` は git 管理外（`.gitignore` に登録済み）の自由な作業領域。  
`notebooks/` は git 管理下なので、変更後は `git status` / `git diff` で内容を確認できるようにする。

## notebook の設計原則

各 notebook は **単体で完結する**設計にしてください。

```text
- scripts/ フォルダのファイルを import・実行・参照しない
- common.py や qwen3_4b_probe.json を notebook から読み込まない
- モデルのロード・トークナイズ・forward など必要な処理はすべて notebook 内に記述する
- notebook を開けばそのまま実行できる状態を保つ
```

scripts と notebooks は独立した成果物であり、scripts が notebook の前提条件にならないようにしてください。

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

## 自律実行の禁止

明示的な指示があるまで以下を実行しない：

- `git commit` / `git commit --amend`
- `git push` / `git push --force`（force push は絶対に実行しない）
- `git rebase` / `git reset --hard`
- `git restore` / `git checkout -- .`（未コミット変更が消える）
- `git clean -f` / `git clean -fd`
- ブランチの作成・削除・リネーム

## commit の作業フロー

1. `git diff` / `git status` で変更内容を確認してユーザーに提示する
2. ユーザーの承認を得てから `git add`（対象ファイルを明示）
3. commit メッセージ案を提示する
4. ユーザーの承認を得てから `git commit`
5. `git push` はユーザーが明示的に要求した場合のみ実行する

## commit メッセージのスタイル

シンプルに1行で。プレフィックス例：`add:` `update:` `fix:` `remove:`

```text
add: cursor rule for git safety
update: probe config for short sequences
fix: scratch path resolution
```

署名（Co-Authored-By 等）は追記しない。

---

## やってはいけないこと

この repository では、以下を避けてください。

```text
- Transformers ソースコードを改変する。
- モデル重みを repository に入れる。
- 大きな tensor を workspace 内に保存する。
- `/Users/<name>/...` や `C:\Users\<name>\...` のようなマシン固有絶対パスを script に直書きする。
- 通常の probe script に暗黙の download 処理を追加する。
- 明示的な指示なしに model_id を Qwen/Qwen3-4B から変える。
- attention 保存時に sequence length を大きくする。
- 生成物を commit する。ただし `docs/` は例外で、完成品レポートと figure (docs/images/) は意図的に git 管理対象。
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
