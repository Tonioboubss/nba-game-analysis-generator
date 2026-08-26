"""Gradio demo: pick a test-set game, compare the generated recap to the real one.

Run locally or deploy as-is to a Hugging Face Space (CPU Basic is enough for
inference on a LoRA-adapted small/base encoder-decoder model).

    python src/app.py --model-dir models/winner --test data/processed/test.jsonl

Later (see project recap): swap the --test file for live/recent games pulled
via nba_api, once they've been reshaped through preprocess.py's linearization
format — the model was trained on that exact input shape and will degrade on
anything structurally different.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import gradio as gr
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

_STATE = {}


def load_test_games(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def generate_recap(source_text: str, num_candidates: int, temperature: float) -> list[str]:
    tokenizer = _STATE["tokenizer"]
    model = _STATE["model"]
    inputs = tokenizer(source_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        num_return_sequences=num_candidates,
    )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


def on_pick_game(game_id: str):
    game = next(g for g in _STATE["games"] if g["id"] == game_id)
    return game["source"], game["target"]


def on_generate(source_text: str, num_candidates: int, temperature: float):
    candidates = generate_recap(source_text, int(num_candidates), temperature)
    return "\n\n---\n\n".join(f"Candidate {i+1}:\n{c}" for i, c in enumerate(candidates))


def build_ui() -> gr.Blocks:
    games = _STATE["games"]
    choices = [g["id"] for g in games]

    with gr.Blocks(title="RotoWire recap generator") as demo:
        gr.Markdown(
            "# NBA box-score → recap generator\n"
            "Pick a test-set game, see the linearized box-score fed to the model, "
            "generate one or more candidate recaps, and compare against the real "
            "human-written summary."
        )
        with gr.Row():
            game_dropdown = gr.Dropdown(choices=choices, label="Test game")
        with gr.Row():
            source_box = gr.Textbox(label="Linearized box-score (model input)", lines=6)
        with gr.Row():
            with gr.Column():
                num_candidates = gr.Slider(1, 5, value=3, step=1, label="Candidates to sample")
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Sampling temperature")
                generate_btn = gr.Button("Generate")
            with gr.Column():
                real_summary = gr.Textbox(label="Real human-written summary", lines=6)
        generated_box = gr.Textbox(label="Generated candidate(s)", lines=10)

        game_dropdown.change(on_pick_game, inputs=game_dropdown, outputs=[source_box, real_summary])
        generate_btn.click(on_generate, inputs=[source_box, num_candidates, temperature], outputs=generated_box)

        if choices:
            demo.load(lambda: on_pick_game(choices[0]), outputs=[source_box, real_summary])

    return demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=None,
                         help="Base model id the LoRA adapter was trained from (e.g. t5-base). "
                              "If omitted, read from the adapter's own config.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Path to the saved LoRA adapter")
    parser.add_argument("--test", type=Path, required=True, help="Path to data/processed/test.jsonl")
    parser.add_argument("--sample", type=int, default=20, help="Only load N random test games into the dropdown")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    base_model_id = args.base_model
    if base_model_id is None:
        with open(args.model_dir / "adapter_config.json") as f:
            base_model_id = json.load(f)["base_model_name_or_path"]
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    model.eval()

    games = load_test_games(args.test)
    random.Random(0).shuffle(games)
    _STATE["games"] = games[: args.sample]
    _STATE["tokenizer"] = tokenizer
    _STATE["model"] = model

    build_ui().launch()


if __name__ == "__main__":
    main()
