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
