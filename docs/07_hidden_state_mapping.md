# Experiment 07: hidden state の hook 取得が `output_hidden_states=True` と一致することの確認

Script: [`scripts/07_hidden_state_mapping.py`](../scripts/07_hidden_state_mapping.py)
最終更新: 2026-05-11
ステータス: ✅ 全 layer で `max_abs_diff = 0.0`（完全一致）を確認。

---

## 1. 目的

`output_hidden_states=True` で返ってくる Hugging Face Transformers の `outputs.hidden_states` タプルが、**PyTorch hook で各 decoder layer の出力を捕まえた tensor と数値的に同一**であることを確認する。

これは後続の実験（[docs/08_logit_lens.md](08_logit_lens.md), [docs/12_residual_stream_patching.md](12_residual_stream_patching.md)）の前提条件です。logit lens / activation patching では：

- **observe（観察）**: `output_hidden_states=True` の方が手軽
- **intervene（介入・置き換え）**: hook 経由しかできない

両方の経路で扱う tensor が同じものを指すことを最初に保証しておかないと、「実は別物だった」という後の混乱を生みます。

---

## 2. 背景: Qwen3Model.forward の構造

Qwen3 を含む現代の decoder-only Transformer は、おおむね以下の流れです:

```text
input_ids
    │
    ▼
embed_tokens         ──► hidden_states[0]   (= embedding output)
    │
    ▼
DecoderLayer 0       ──► hidden_states[1]
    │
    ▼
DecoderLayer 1       ──► hidden_states[2]
    │
    ⋮
    │
    ▼
DecoderLayer K-1     ──► (pre-norm output)
    │
    ▼
final RMSNorm        ──► hidden_states[K]   (= last hidden state)
    │
    ▼
lm_head              ──► logits
```

ここで:

- $K = 36$ は decoder layer の数（Qwen3-4B の場合）
- `hidden_states` タプルの長さは **$K + 1 = 37$**（embedding 出力 + 各 layer 出力 + 最終 norm 後）
- **最終 RMSNorm は `Qwen3Model.forward` の中**で `model.model.norm` として適用される。**DecoderLayer の中ではない**点に注意

最後の 1 点が、最終層の hook 出力と `hidden_states[-1]` の関係を考えるときに重要になります。

---

## 3. 実験設定

| 項目 | 値 |
|---|---|
| 対象モデル | `Qwen/Qwen3-4B` |
| プロンプト | デフォルト ([qwen3_4b_probe.json](../scripts/qwen3_4b_probe.json) の `default_prompt`、35 token) |
| device / dtype | mps / float16 |
| `attn_implementation` | `eager` |
| `output_hidden_states` | True（hook 取得と並行） |
| `use_cache` | False |

---

## 4. 方法

### 4-1. Forward hook を全 decoder layer に登録

```python
K = len(model.model.layers)           # 36
hook_outputs: dict[int, torch.Tensor] = {}

def make_hook(j: int):
    def hook(module, inp, out):
        # out は tuple または tensor。最初の要素を取る。
        raw = out[0] if isinstance(out, (tuple, list)) else out
        hook_outputs[j] = raw.detach()
    return hook

handles = [
    layer.register_forward_hook(make_hook(j))
    for j, layer in enumerate(model.model.layers)
]
```

各 `Qwen3DecoderLayer` の forward が呼ばれるたびに、**その layer の出力 tensor** が `hook_outputs[j]` に格納されます。出力が tuple の場合は最初の要素を取る（Qwen3 では `(hidden_states,)` だけが返る設定）。

### 4-2. 1 回 forward して両方の経路を取得

```python
with torch.no_grad():
    outputs_hook = model(
        **inputs,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
    )

for h in handles:
    h.remove()                       # hook は必ず外す

hs_hook = outputs_hook.hidden_states  # tuple of length K+1 = 37
```

### 4-3. 3 段階の一致確認

#### (a) embedding 出力

```python
embed_out = model.model.embed_tokens(inputs["input_ids"])
diff = (embed_out - hs_hook[0]).abs().max()
```

確認: `embed_tokens(input_ids)` $\overset{?}{=}$ `hidden_states[0]`

#### (b) 各 decoder layer

```python
for j in range(K):
    if j < K - 1:
        # 中間 layer: hook 出力をそのまま比較
        ref = hs_hook[j + 1]
        d = (hook_outputs[j] - ref).abs().max()
    else:
        # 最終 layer: hook は pre-norm 出力。final RMSNorm を通してから比較
        normed = model.model.norm(hook_outputs[j])
        ref = hs_hook[j + 1]
        d = (normed - ref).abs().max()
```

確認:
- 中間 layer ($j = 0, 1, \dots, K-2$): hook 出力 $\overset{?}{=}$ `hidden_states[j+1]`
- 最終 layer ($j = K-1 = 35$): **`model.norm(hook 出力)`** $\overset{?}{=}$ `hidden_states[K]`

**最終層だけ扱いが違う**のは、§ 2 で述べた「final RMSNorm が `Qwen3Model.forward` 側にある」設計から。`hidden_states[-1]` は norm 後の値が入っている一方、layer 35 の forward hook は norm 前の出力を見るため、片側に norm を適用してから比較する必要があります。

#### (c) lm_head

```python
lm_out = model.lm_head(hs_hook[-1])
diff = (lm_out - outputs_hook.logits).abs().max()
```

確認: `lm_head(hidden_states[-1])` $\overset{?}{=}$ `outputs.logits`

---

## 5. 結果

### 5-1. サマリ — [outputs/prelim_hidden_state_mapping_summary.json](../outputs/prelim_hidden_state_mapping_summary.json)

| 項目 | 値 |
|---|---|
| `num_parameters` | 4,022,468,096（≒ 4.02 B）|
| `num_decoder_layers` | 36 |
| `num_hidden_states` | 37 |
| `hidden_size` | 2560 |
| `vocab_size` | 151936 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | **8**（GQA: 32 query heads → 8 KV heads にグルーピング）|
| `tie_word_embeddings` | **True**（embedding と unembedding が同じ重み。詳細は [docs/09_embedding_unembedding.md](09_embedding_unembedding.md)）|
| `rms_norm_eps` | 1e-06 |

### 5-2. 一致確認 — [outputs/prelim_hidden_state_mapping_diffs.csv](../outputs/prelim_hidden_state_mapping_diffs.csv)

**全 38 比較で `max_abs_diff = 0.0`**（embedding + 36 layer + lm_head）。

| 比較 | max abs diff |
|---|---|
| `embed_tokens(input_ids)` vs `hidden_states[0]` | **0.0** |
| layer 0 hook vs `hidden_states[1]` | 0.0 |
| layer 1 hook vs `hidden_states[2]` | 0.0 |
| … (layers 2–34 全て 0.0) | 0.0 |
| **`norm(layer 35 hook)` vs `hidden_states[36]`** | **0.0** |
| `lm_head(hidden_states[-1])` vs `outputs.logits` | **0.0** |

→ 完全一致。`output_hidden_states=True` で返ってくる tensor と hook で捕まえた tensor は、**bit-exact** で同じ。

### 5-3. 何が確認できたか

1. **`hidden_states[j]` は `model.model.layers[j-1]` の出力に等しい**（$j \geq 1$）。`j = 0` だけは embedding 出力。
2. **最終層 (layer 35) の hook 出力は post-norm ではなく pre-norm**。`hidden_states[-1]` を取るときは内部で `model.model.norm` が適用されているので、両者をそのまま比較すると違って見える。`norm(hook output)` で揃う。
3. **`lm_head(hidden_states[-1])` で logits が再現できる**。「最後の hidden state に unembedding をかけて softmax を取れば next-token 分布が出る」という logit lens の原理 ([docs/08](08_logit_lens.md)) がここから直接導かれる。
4. **数値が完全一致 (`= 0.0`)**: float16 / MPS でも、同じ tensor を 2 経路から取り出しているだけなので誤差は出ない。これは「hook と `output_hidden_states` が **同一の tensor 参照**を返している」ことを意味する（実装的にも transformers/modeling_qwen3.py で確認可能）。

---

## 6. 応用への示唆

- **[docs/08_logit_lens.md](08_logit_lens.md)**: 「各層の hidden state に lm_head をかける」logit lens は、ここで確認した `lm_head(hs[-1]) = logits` の関係を全 layer に拡張したもの。
- **[docs/12_residual_stream_patching.md](12_residual_stream_patching.md)**: activation patching は「clean run の hook 出力を corrupt run の hook 経路に書き込む」操作。hook 出力と `hidden_states[j+1]` が同じものを指すと確認できたので、観察と介入の整合性が取れる。
- **最終層の norm 扱い**: 自前で logit lens を実装するときに、`hidden_states[-1]` をそのまま `lm_head` に通せばよい（norm はすでに済んでいる）。中間層 `hidden_states[j]` ($j < K$) は **norm 前** なので、自前で `model.model.norm` を適用してから `lm_head` に通す必要がある（[docs/08](08_logit_lens.md) で詳述）。
- **nb02 への寄与**: 同 notebook で hook を仕掛けるときに、ここで確認した「hook 出力 = `hidden_states[j+1]`」関係に依拠している。

---

## 7. 出力ファイル

- [outputs/prelim_hidden_state_mapping_diffs.csv](../outputs/prelim_hidden_state_mapping_diffs.csv) — 38 行（embedding + 36 layer + lm_head）の `label, max_abs_diff`
- [outputs/prelim_hidden_state_mapping_summary.json](../outputs/prelim_hidden_state_mapping_summary.json) — モデル設定 + 各比較の max diff
- [outputs/prelim_qwen3_source_paths.txt](../outputs/prelim_qwen3_source_paths.txt) — modeling_qwen3.py のパス
- [outputs/prelim_qwen3_source_snippets.txt](../outputs/prelim_qwen3_source_snippets.txt) — `Qwen3ForCausalLM.forward`, `Qwen3Model.forward`, `Qwen3DecoderLayer.forward` のソース抜粋

---

## 8. 注意事項

- **GQA の存在**: `num_attention_heads = 32`、`num_key_value_heads = 8`。Qwen3 は **Grouped-Query Attention** を使い、query は 32 head、KV は 8 head で計算する（4 query head が 1 KV head を共有）。attention 行列の shape は `[batch, num_attention_heads, T, T] = [1, 32, 35, 35]` で、KV head 数は heatmap 上では見えない。
- **hook 出力が tuple か tensor かはモジュール依存**: `Qwen3DecoderLayer.forward` は単一 tensor を返す（output_attentions / use_cache が両方 False のとき）。tuple で返るケースに備えて `out[0] if isinstance(out, (tuple, list)) else out` を入れてある。
- **hook は必ず外す**: 外し忘れると後続の forward が遅くなる / メモリリークの原因。`try/finally` で囲む方が安全（このスクリプトでは順次実行で短いので素直な `for h in handles: h.remove()`）。
- **scripts 内のファイル名は `07_hidden_state_mapping.py` だが、過去 commit では `07_prelim_hidden_state_mapping.py` だった**。途中で `prelim_` prefix を外したリネームがあり、出力 CSV / JSON の名前にだけ `prelim_` が残っている。
