"""Batch-generate recaps for a whole split, in the format evaluate_generation.py expects.

Usage:
    python src/predict.py --base-model t5-base --model-dir models/winner \
        --input data/processed/test.jsonl --output data/processed/test_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=None,
                         help="Base model id the adapter was trained from. If omitted, "
                              "read from the adapter's own adapter_config.json.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-source-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    base_model_id = args.base_model
    if base_model_id is None:
        with open(args.model_dir / "adapter_config.json") as f:
            base_model_id = json.load(f)["base_model_name_or_path"]
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base_model, args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]

    with open(args.output, "w", encoding="utf-8") as out:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            inputs = tokenizer(
                [r["source"] for r in batch],
                return_tensors="pt", truncation=True,
                max_length=args.max_source_len, padding=True,
            ).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens,
                    num_beams=4,  # greedy/beam here: this is for evaluation, not
                                  # the demo's sampled-variety candidates.
                )
            generated = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for row, gen in zip(batch, generated):
                out.write(json.dumps({**row, "generated": gen}, ensure_ascii=False) + "\n")
            print(f"[{start + len(batch)}/{len(rows)}] generated")

    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
