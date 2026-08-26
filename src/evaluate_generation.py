"""Evaluate generated recaps: BLEU (surface form) + an approximate content-selection score.

Honest scoping note (read this before trusting the numbers):
The RotoWire paper (Wiseman et al., 2017) defines three data-to-text-specific
metrics — Relation Generation (RG), Content Selection (CS), Content Ordering
(CO) — computed by re-extracting (entity, type, value) relations from the
GENERATED text using a separately trained information-extraction (IE) model,
then comparing those extracted relations to the gold box-score records.
Reproducing that IE model faithfully is a real sub-project on its own (it
was itself a trained classifier in the original paper), not something to
improvise correctly in a first pass.

What this script gives instead, clearly labeled as an approximation, not a
faithful reimplementation:
  - BLEU: standard surface-form n-gram overlap against the reference summary.
  - approx_content_selection: for each ground-truth record value in the
    source (e.g. "24" for Harrison Barnes' PTS), checks whether that exact
    value token appears anywhere in the generated text. This is a coarse,
    entity-blind recall proxy for CS — it will overcount (a value can appear
    coincidentally, e.g. a jersey number) and it says nothing about whether
    the value is attached to the RIGHT entity in the generated text (which
    is exactly what CS/RG are designed to check). Report it as "approximate
    content coverage", not as "CS" — don't relabel it as the paper's metric
    in a README or a portfolio write-up.

Before presenting results as a comparison to the published literature (as
planned in the project recap), either budget real time to train a proper IE
model on this same data (the original repo's approach, would need adapting
from its old Lua/Torch codebase to PyTorch), or be explicit in the write-up
that only BLEU is a like-for-like comparison and the content-coverage number
is this project's own simplified proxy.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import evaluate

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_record_values(source_line: str) -> list[str]:
    """Pull the "value" field out of each <record> entity | type | value | HA </record> span."""
    values = []
    for record in re.findall(r"<record>\s*(.*?)\s*</record>", source_line):
        parts = [p.strip() for p in record.split("|")]
        if len(parts) >= 3:
            values.append(parts[2])
    return values


def approx_content_selection(source_line: str, generated_text: str) -> float:
    """Fraction of source record values that show up verbatim in the generated text."""
    values = [v for v in extract_record_values(source_line) if NUMBER_RE.fullmatch(v)]
    if not values:
        return float("nan")
    gen_numbers = set(NUMBER_RE.findall(generated_text))
    hits = sum(1 for v in values if v in gen_numbers)
    return hits / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True,
                         help="JSONL with one {'id', 'source', 'target', 'generated'} row per line")
    parser.add_argument("--out", type=Path, default=None,
                         help="Optional path to write per-example scores as JSONL")
    args = parser.parse_args()

    bleu = evaluate.load("sacrebleu")
    rows = [json.loads(line) for line in open(args.predictions, encoding="utf-8")]

    predictions = [r["generated"] for r in rows]
    references = [[r["target"]] for r in rows]
    bleu_result = bleu.compute(predictions=predictions, references=references)

    per_example = []
    coverage_scores = []
    for r in rows:
        score = approx_content_selection(r["source"], r["generated"])
        if score == score:  # not NaN
            coverage_scores.append(score)
        per_example.append({"id": r.get("id"), "approx_content_coverage": score})

    avg_coverage = sum(coverage_scores) / len(coverage_scores) if coverage_scores else float("nan")

    print(f"BLEU: {bleu_result['score']:.2f}")
    print(f"Approx. content coverage (NOT the paper's CS metric — see module docstring): "
          f"{avg_coverage:.3f} over {len(coverage_scores)} examples")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for row in per_example:
                f.write(json.dumps(row) + "\n")
        print(f"Per-example scores written to {args.out}")


if __name__ == "__main__":
    main()
