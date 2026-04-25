import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ── Load model ────────────────────────────────────────────────
MODEL_ID = "snehangshu511/gpt2-medium-instruct"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
model     = GPT2LMHeadModel.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

# ── Prompt builder ────────────────────────────────────────────
def build_prompt(instruction: str, input_text: str = "") -> str:
    base = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
    )
    if input_text.strip():
        return f"{base}### Instruction:\n{instruction.strip()}\n\n### Input:\n{input_text.strip()}\n\n### Response:\n"
    return f"{base}### Instruction:\n{instruction.strip()}\n\n### Response:\n"

# ── Generate ──────────────────────────────────────────────────
def generate(instruction, input_text, max_new_tokens, temperature, top_p, top_k, rep_penalty):
    if not instruction.strip():
        return "⚠️ Please enter an instruction."

    prompt    = build_prompt(instruction, input_text)
    inputs    = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    if input_len > 900:
        return "⚠️ Prompt too long. Please shorten your instruction."

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens     = int(max_new_tokens),
            do_sample          = temperature > 0.0,
            temperature        = float(temperature),
            top_p              = float(top_p),
            top_k              = int(top_k),
            repetition_penalty = float(rep_penalty),
            pad_token_id       = tokenizer.eos_token_id,
            eos_token_id       = tokenizer.eos_token_id,
        )

    raw = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    return raw if raw else "⚠️ Empty response. Try different settings."

# ── UI ────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="violet"),
    title="GPT-2 Medium Instruct"
) as demo:

    gr.Markdown(
        """
        # GPT-2 Medium Instruct  
        406M parameter GPT-2 fine-tuned for instruction following.

        Enter a task → get a response.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            instruction_box = gr.Textbox(
                label="Instruction",
                placeholder="Write a haiku about the moon...",
                lines=4
            )

            input_box = gr.Textbox(
                label="Optional Input",
                placeholder="Add context if needed...",
                lines=3
            )

            generate_btn = gr.Button("Generate", variant="primary")

            with gr.Accordion("Generation Settings", open=False):
                max_tokens  = gr.Slider(50, 512, value=200, step=10, label="Max tokens")
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                top_p       = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p")
                top_k       = gr.Slider(0, 100, value=50, step=5, label="Top-k")
                rep_penalty = gr.Slider(1.0, 2.0, value=1.2, step=0.1, label="Repetition penalty")

        with gr.Column(scale=1):
            output_box = gr.Textbox(
                label="Response",
                lines=18,
                show_copy_button=True
            )

            clear_btn = gr.Button("Clear")

    # Examples
    gr.Markdown("### Examples")
    gr.Examples(
        examples=[
            ["Explain artificial intelligence simply.", "", 200, 0.7, 0.9, 50, 1.2],
            ["Write a poem about the ocean.", "", 150, 0.9, 0.95, 60, 1.1],
            ["Benefits of exercise?", "", 200, 0.7, 0.9, 50, 1.2],
            ["Summarize this:", "The Industrial Revolution began...", 200, 0.6, 0.9, 40, 1.2],
        ],
        inputs=[instruction_box, input_box, max_tokens, temperature, top_p, top_k, rep_penalty],
        outputs=output_box,
        fn=generate,
        cache_examples=False
    )

    # ── Events ────────────────────────────────────────────────
    inputs_list = [instruction_box, input_box, max_tokens, temperature, top_p, top_k, rep_penalty]

    generate_btn.click(fn=generate, inputs=inputs_list, outputs=output_box)
    instruction_box.submit(fn=generate, inputs=inputs_list, outputs=output_box)

    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[instruction_box, input_box, output_box]
    )

# launch with queue for better UX
demo.queue().launch()