"""Preprocess the RotoWire dataset into (source, target) pairs for seq2seq fine-tuning.

Source format: a flat sequence of linearized records, one per team stat and per
player stat, following the record representation used in the data-to-text
literature (Wiseman et al., 2017 and follow-ups): each record is a
(entity, type, value, home_or_visitor) tuple, serialized as a short tagged
span. Example excerpt of a linearized source:

    <record> Utah Jazz | TEAM-PTS | 97 | HOME </record>
    <record> Dallas Mavericks | TEAM-PTS | 81 | VIS </record>
    <record> Harrison Barnes | PTS | 24 | VIS </record>
    ...

Target: the human-written recap, detokenized into a plain string.

IMPORTANT — schema note (read before running):
This script was written against the verified schema of the bilingual GEM
re-packaging of RotoWire (GEM/RotoWire_English-German on Hugging Face), which
confirmed field names: home_line/vis_line (TEAM-* keys), box_score (per-stat
dict keyed by player index string), home_name/vis_name/home_city/vis_city,
and a tokenized summary list. The original monolingual files
(harvardnlp/boxscore-data, rotowire/{train,valid,test}.json) are widely
reported to use the same home_line/vis_line/box_score structure, but this
was not re-verified directly against that exact file in this session (it
ships as a .tar.bz2 archive, not a browsable raw file). The loader below is
deliberately schema-driven (it reads whatever stat keys are actually present
in box_score rather than hardcoding an assumed list) and the summary-field
lookup tries both "summary" and "summary_en" so it degrades gracefully
either way — but the first time you run this against the real
rotowire/train.json in Colab, sanity-check one record by hand
(see `if __name__ == "__main__"` below) before trusting the full conversion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Per-player fields that are names/labels, not numeric stats — used to build
# the player's display name, never emitted as a stat record themselves.
NAME_FIELDS = {"PLAYER_NAME", "FIRST_NAME", "SECOND_NAME"}

# Fields inside home_line / vis_line that identify the team rather than
# describing a stat — excluded from the emitted team stat records.
TEAM_IDENTITY_FIELDS = {"TEAM-NAME", "TEAM-CITY"}


def load_games(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _team_records(line: dict, team_label: str) -> list[str]:
    """Linearize one team's line score (home_line or vis_line) into records."""
    team_name = line.get("TEAM-NAME", "")
    team_city = line.get("TEAM-CITY", "")
    entity = f"{team_city} {team_name}".strip()
    records = []
    for key, value in line.items():
        if key in TEAM_IDENTITY_FIELDS:
            continue
        stat_type = key.removeprefix("TEAM-")
        records.append(f"<record> {entity} | {stat_type} | {value} | {team_label} </record>")
    return records


def _player_records(box_score: dict, home_city: str, vis_city: str) -> list[str]:
    """Linearize every player's stats into records.

    box_score is a dict of {stat_name: {player_index: value}}. We first
    recover the set of player indices, then emit one record per
    (player, stat) pair, tagging each player HOME or VIS based on their
    TEAM_CITY field.
    """
    # Recover all player indices from any stat sub-dict (they share the same keys).
    any_field = next((v for v in box_score.values() if isinstance(v, dict)), {})
    player_indices = sorted(any_field.keys(), key=lambda i: int(i))

    records = []
    for idx in player_indices:
        name = box_score.get("PLAYER_NAME", {}).get(idx, "").strip()
        if not name:
            continue
        player_city = box_score.get("TEAM_CITY", {}).get(idx, "")
        if player_city == home_city:
            label = "HOME"
        elif player_city == vis_city:
            label = "VIS"
        else:
            label = "UNK"  # TEAM_CITY missing/unexpected — flag rather than guess.

        for stat_name, values in box_score.items():
            if stat_name in NAME_FIELDS or stat_name == "TEAM_CITY":
                continue
            value = values.get(idx)
            if value in (None, "N/A"):
                continue
            records.append(f"<record> {name} | {stat_name} | {value} | {label} </record>")
    return records


def linearize_source(game: dict) -> str:
    records = []
    records += _team_records(game["home_line"], "HOME")
    records += _team_records(game["vis_line"], "VIS")
    records += _player_records(
        game["box_score"], game.get("home_city", ""), game.get("vis_city", "")
    )
    return " ".join(records)


def build_target(game: dict) -> str:
    tokens = game.get("summary") or game.get("summary_en")
    if tokens is None:
        raise KeyError(
            "Neither 'summary' nor 'summary_en' found in this record — "
            "check the actual key name in your downloaded file and adjust build_target()."
        )
    return " ".join(tokens)


def convert_split(in_path: Path, out_path: Path) -> int:
    games = load_games(in_path)
    with open(out_path, "w", encoding="utf-8") as out:
        for game in games:
            row = {
                "id": game.get("id", f"{game.get('day', '')}-{game.get('home_name', '')}-{game.get('vis_name', '')}"),
                "source": linearize_source(game),
                "target": build_target(game),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(games)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/rotowire"),
                         help="Directory containing train.json / valid.json / test.json")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        in_path = args.raw_dir / f"{split}.json"
        if not in_path.exists():
            print(f"[skip] {in_path} not found — did you extract rotowire.tar.bz2 into {args.raw_dir}?")
            continue
        out_path = args.out_dir / f"{split}.jsonl"
        n = convert_split(in_path, out_path)
        print(f"[ok] {split}: {n} games -> {out_path}")


if __name__ == "__main__":
    main()
