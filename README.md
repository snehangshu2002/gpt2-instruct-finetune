# GPT-2 Medium Instruct 🧠

A **355M parameter GPT-2 Medium** model fine-tuned from scratch on the Alpaca instruction dataset — built end-to-end using a custom PyTorch architecture, trained with PyTorch Lightning, and deployed as an interactive Gradio app on Hugging Face Spaces.

[![Live Demo](https://img.shields.io/badge/🤗%20Spaces-Live%20Demo-blue)](https://snehangshu511-gpt2-medium-instruct-demo.hf.space/)
[![Model on HF](https://img.shields.io/badge/🤗%20Hub-Model%20Weights-yellow)](https://huggingface.co/snehangshu511/gpt2-medium-instruct)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🔗 Links

| Resource | Link |
|---|---|
| 🚀 Live Demo (Gradio) | https://snehangshu511-gpt2-medium-instruct-demo.hf.space/ |
| 🤗 Model Weights | https://huggingface.co/snehangshu511/gpt2-medium-instruct |

---

## What is this?

This project fine-tunes GPT-2 Medium for **instruction following** — you give it a task like *"Explain machine learning simply"* or *"Write a poem about the ocean"* and it generates a response.

The entire pipeline is built from scratch:
- Custom GPT-2 architecture in PyTorch (no HF `AutoModel` during training)
- Pretrained weights loaded manually from `openai-community/gpt2-medium`
- Fine-tuned using PyTorch Lightning on the `yahma/alpaca-cleaned` dataset
- Weights converted to HF-compatible format and pushed to the Hub
- Deployed as a Gradio web app on HF Spaces

Built as a learning project following Sebastian Raschka's *"Build a Large Language Model from Scratch"*.

---

## Demo

> Try it live: **https://snehangshu511-gpt2-medium-instruct-demo.hf.space/**

![Gradio UI showing instruction input and model response](https://img.shields.io/badge/UI-Gradio%206.0-orange)

Example prompts you can try:
- `Explain artificial intelligence simply.`
- `Write a poem about the ocean.`
- `What are the benefits of exercise?`
- `Summarize this:` + any input text

---

## Model Details

| Property | Value |
|---|---|
| Base model | `openai-community/gpt2-medium` |
| Parameters | ~355M |
| Architecture | GPT-2 (decoder-only transformer) |
| Fine-tuning dataset | `yahma/alpaca-cleaned` (10,000 samples) |
| Context length | 1,024 tokens |
| Embedding dim | 1,024 |
| Layers | 24 transformer blocks |
| Attention heads | 16 |
| Tokenizer | GPT-2 BPE |

---

## Training Pipeline

```
yahma/alpaca-cleaned (52K rows)
        ↓  use 10K for training
Alpaca prompt formatting  (### Instruction / Input / Response)
        ↓
tiktoken BPE tokenization  +  -100 prompt masking
        ↓
Custom PyTorch Dataset + DataLoader (dynamic padding)
        ↓
GPT-2 Medium pretrained weights loaded
        ↓
PyTorch Lightning fine-tuning
  · AdamW  lr=3e-5  weight_decay=0.1
  · FP16 mixed precision
  · Gradient accumulation (effective batch = 8)
  · Checkpoint on best val_loss  +  early stopping
        ↓
Lightning prefix stripped → clean GPTModel state dict
        ↓
Converted to HF GPT2LMHeadModel key format
        ↓
Saved as model.safetensors + pytorch_model.bin
        ↓
Pushed to HF Hub  →  Gradio Space deployed
```

### Key Training Details

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | `3e-5` |
| Weight decay | `0.1` |
| Betas | `(0.9, 0.95)` |
| Epochs | 2 |
| Batch size | 2 (× 4 gradient accumulation = 8 effective) |
| Precision | FP16 mixed (`16-mixed`) |
| Gradient clip | `1.0` |
| Early stopping patience | 3 checks |

---

## Project Structure

```
gpt2-medium-instruct/
├── GPT.py                        # Custom GPTModel architecture + Lightning wrapper
├── app_instruct.py               # Gradio web app (Gradio 6.0 compatible)
├── gpt2_chat_instruct_notebook.ipynb  # Full training notebook
└── README.md
```

---

## Running Locally

### Inference only (from HF Hub)

```bash
pip install transformers torch gradio
```

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_id  = "snehangshu511/gpt2-medium-instruct"
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
model     = GPT2LMHeadModel.from_pretrained(model_id)
model.eval()

def build_prompt(instruction, input_text=""):
    base = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
    )
    if input_text.strip():
        return f"{base}### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    return f"{base}### Instruction:\n{instruction}\n\n### Response:\n"

prompt = build_prompt("Explain what machine learning is in simple terms.")
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

input_len = inputs["input_ids"].shape[1]
print(tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True))
```

### Run the Gradio app locally

```bash
pip install transformers torch gradio
python app_instruct.py
```

---

## Prompt Format

The model expects the Alpaca instruction format:

```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
<your instruction here>

### Input:
<optional context>

### Response:
```

If there is no additional input, the `### Input:` block is omitted.

---

## Limitations

- Fine-tuned on only 10K of the available ~52K Alpaca samples — a full-data run would improve quality noticeably
- GPT-2 Medium is much smaller than modern LLMs; responses can drift or be inconsistent on complex tasks
- No RLHF or safety alignment — do not deploy in production without additional safeguards
- Hard context limit of 1,024 tokens

---

## References

- [Build a Large Language Model From Scratch](https://github.com/rasbt/LLMs-from-scratch) — Sebastian Raschka
- [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) — instruction fine-tuning dataset
- [openai-community/gpt2-medium](https://huggingface.co/openai-community/gpt2-medium) — pretrained base weights

---

## Author

**Snehangshu Bhuin**  
GitHub: [@snehangshu2002](https://github.com/snehangshu2002)