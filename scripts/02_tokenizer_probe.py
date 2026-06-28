# chat template を適用したプロンプトをトークナイズし、各トークンの情報を表示・保存する。
# position / token_id / piece / decoded_piece を一覧化した CSV を outputs/token_table.csv に保存する。
# 環境: aidemo2026

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
