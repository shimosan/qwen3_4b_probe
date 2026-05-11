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
    dtype=dtype,
    attn_implementation=cfg["attn_implementation"],
)
model.to(device)  # type: ignore[union-attr]
model.eval()

with torch.no_grad():
    generated = model.generate(  # type: ignore[union-attr]
        **inputs,
        max_new_tokens=cfg["max_new_tokens"],
        do_sample=False,
    )

out = tokenizer.decode(generated[0], skip_special_tokens=False)
print(out)

with open("outputs/generate_smoke.txt", "w", encoding="utf-8") as f:
    f.write(out)
    f.write("\n")
