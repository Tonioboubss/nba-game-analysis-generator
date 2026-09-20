"""LoRA fine-tuning of an encoder-decoder model on RotoWire (box-score -> recap).

Designed to run on a single free Colab T4 GPU. Meant to be called once per
candidate during the model bake-off (see project recap: T5-small/base vs
BART-base), then again for a full run on the winner.

Usage (Colab):
    !pip install -q transformers datasets peft accelerate evaluate sacrebleu
    !python src/finetune.py --model t5-small --train data/processed/train.jsonl \
        --valid data/processed/valid.jsonl --output models/t5-small-bakeoff \
        --max-train-examples 500 --epochs 3

For the bake-off, pass --max-train-examples to cap it to a small common
subset (a few hundred examples, per the recap's methodology) so the 2-3
candidates can be compared quickly before committing to a full run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# BART and T5 use different LoRA target module names internally (T5: q/v of
# self and cross attention; BART: same names but different module tree) —
# transformers' naming is consistent enough that targeting "q_proj"/"v_proj"
# (BART-style) or "q"/"v" (T5-style) needs to match the actual model. We
# detect by model type rather than hardcoding one architecture.
LORA_TARGET_MODULES = {
    "t5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
}


def load_jsonl(path: Path) -> Dataset:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return Dataset.from_list(rows)


def build_lora_model(model_name: str):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model_type = model.config.model_type  # e.g. "t5", "bart"
    target_modules = LORA_TARGET_MODULES.get(model_type)
    if target_modules is None:
        raise ValueError(
            f"No known LoRA target modules for model_type={model_type!r} — "
            f"add an entry to LORA_TARGET_MODULES after checking the model's "
            f"named_modules() for its attention projection layer names."
        )
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    return get_peft_model(model, lora_config)


def preprocess_batch(batch, tokenizer, max_source_len: int, max_target_len: int):
    model_inputs = tokenizer(
        batch["source"], max_length=max_source_len, truncation=True
    )
    labels = tokenizer(
        text_target=batch["target"], max_length=max_target_len, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                         help="HF model id, e.g. t5-small, t5-base, facebook/bart-base")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-examples", type=int, default=None,
                         help="Cap training set size — use a small common number across "
                              "candidates for the bake-off, omit for the full run.")
    parser.add_argument("--max-source-len", type=int, default=1024,
                         help="Verified empirically (2026-09-02): with top-3-players/team "
                              "linearization, the median source is ~987 tokens and only ~5%% "
                              "exceed 1024 (bart-base's hard position-embedding limit).")
    parser.add_argument("--max-target-len", type=int, default=512,
                         help="Verified empirically (2026-09-02): the real RotoWire summaries "
                              "have a median of 348 tokens and 72%% exceed the old 256 default — "
                              "512 covers ~88%% of them, still well under bart-base's 1024 limit.")
    parser.add_argument("--epochs", type=int, default=3,
                         help="Upper bound on epochs — with --early-stopping-patience set, "
                              "training usually stops before reaching this, so it's fine to "
                              "set this generously for the full run (e.g. 8).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                         help="Accumulate gradients over this many steps before each optimizer "
                              "update, so --batch-size x this = the effective batch size. Use "
                              "this to lower --batch-size (and so peak activation memory) for a "
                              "model that OOMs — e.g. t5-base has 24 transformer layers (12+12) "
                              "vs bart-base's 12 (6+6), roughly double the activation memory at "
                              "the same batch size and sequence length, and was the one candidate "
                              "that OOM'd on a T4 at --batch-size 8 with 1024-token inputs. "
                              "--batch-size 2 --gradient-accumulation-steps 4 keeps the same "
                              "effective batch of 8 (so the bake-off comparison stays fair) while "
                              "cutting per-step memory roughly 4x.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42,
                         help="For a reproducible final run, worth reporting in the portfolio writeup.")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                         help="Stop once eval_loss hasn't improved for this many epochs, and keep "
                              "the best checkpoint (not necessarily the last one). Recommended for "
                              "the full run (e.g. 2); omit for the bake-off, where --epochs is "
                              "already small and fixed on purpose so candidates are compared "
                              "on equal footing.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = build_lora_model(args.model)
    model.print_trainable_parameters()

    train_ds = load_jsonl(args.train)
    if args.max_train_examples is not None:
        train_ds = train_ds.select(range(min(args.max_train_examples, len(train_ds))))
    valid_ds = load_jsonl(args.valid)

    train_ds = train_ds.map(
        lambda b: preprocess_batch(b, tokenizer, args.max_source_len, args.max_target_len),
        batched=True, remove_columns=train_ds.column_names,
    )
    valid_ds = valid_ds.map(
        lambda b: preprocess_batch(b, tokenizer, args.max_source_len, args.max_target_len),
        batched=True, remove_columns=valid_ds.column_names,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        fp16=True,  # T4 supports fp16; drop this if training on CPU.
        logging_steps=20,
        report_to="none",
        seed=args.seed,
        # Needed for EarlyStoppingCallback to be able to restore the best
        # checkpoint (lowest eval_loss), which isn't necessarily the last one.
        load_best_model_at_end=args.early_stopping_patience is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    callbacks = []
    if args.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train()

    # Save only the LoRA adapter (a few MB) — cheap to checkpoint, cheap to
    # resume across Colab session disconnects.
    model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"Saved LoRA adapter + tokenizer to {args.output}")


if __name__ == "__main__":
    main()