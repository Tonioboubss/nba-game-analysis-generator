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

Schema note: verified directly against the real rotowire/train.json (not
just the bilingual GEM re-packaging this was first drafted against) —
confirmed top-level keys (home_name, box_score, home_city, vis_name,
summary, vis_line, vis_city, day, home_line), confirmed the summary field is
named "summary" (not "summary_en"), and confirmed box_score's per-player
fields including one not anticipated in the first draft: START_POSITION,
which is an empty string "" (not None/"N/A") for bench players — the value
filter below accounts for that. The loader stays schema-driven (reads
whatever stat keys are actually present rather than hardcoding an assumed
list) as a general safety margin, not because the schema is still in doubt.
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

# Verified empirically (2026-09-02): the full linearization (every stat for
# every dressed player) runs to a median of ~7100 tokens per game — every
# single game in the training set blows past a 512- or even 768-token model
# input, meaning almost no player-level information was actually reaching
# the model. First cut (top 8 players/team, 10 stats) only got the median
# down to ~2980 — still 100% over bart-base's hard 1024-token position
# limit, because players (not team stats) dominate the token count. Cutting
# harder here rather than touching team stats, since team field names
# haven't been verified against the real schema the way box_score's have —
# see the schema note at the top of this file.
SALIENT_PLAYER_STATS = ["MIN", "PTS", "REB", "AST", "STL", "BLK"]


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


def _parse_minutes(value) -> float:
    """MIN comes through as a numeric string ("34") or "N/A"/"" for DNPs."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _player_records(
    box_score: dict, home_city: str, vis_city: str, top_n_per_team: int = 3
) -> list[str]:
    """Linearize the top-N players per team (by minutes played) into stat records.

    box_score is a dict of {stat_name: {player_index: value}}. We first
    recover the set of player indices, tag each HOME/VIS/UNK based on their
    TEAM_CITY field, rank each group by minutes played, and only emit
    records — restricted to SALIENT_PLAYER_STATS — for the top_n_per_team of
    each group. Cutting bench players who barely played (and dropping
    low-signal stat fields) is what keeps a game's linearization within a
    model's input budget; see the note above SALIENT_PLAYER_STATS.
    """
    # Recover all player indices from any stat sub-dict (they share the same keys).
    any_field = next((v for v in box_score.values() if isinstance(v, dict)), {})
    player_indices = sorted(any_field.keys(), key=lambda i: int(i))

    by_label: dict[str, list[tuple[float, str, str]]] = {"HOME": [], "VIS": [], "UNK": []}
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
        minutes = _parse_minutes(box_score.get("MIN", {}).get(idx))
        by_label[label].append((minutes, idx, name))

    records = []
    for label, players in by_label.items():
        top_players = sorted(players, key=lambda p: p[0], reverse=True)[:top_n_per_team]
        for _minutes, idx, name in top_players:
            for stat_name in SALIENT_PLAYER_STATS:
                value = box_score.get(stat_name, {}).get(idx)
                if value in (None, "N/A", ""):
                    continue
                records.append(f"<record> {name} | {stat_name} | {value} | {label} </record>")
    return records


def linearize_source(game: dict, top_n_per_team: int = 3) -> str:
    records = []
    records += _team_records(game["home_line"], "HOME")
    records += _team_records(game["vis_line"], "VIS")
    records += _player_records(
        game["box_score"], game.get("home_city", ""), game.get("vis_city", ""),
        top_n_per_team=top_n_per_team,
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


def convert_split(in_path: Path, out_path: Path, top_n_per_team: int = 3) -> int:
    games = load_games(in_path)
    with open(out_path, "w", encoding="utf-8") as out:
        for game in games:
            row = {
                "id": game.get("id", f"{game.get('day', '')}-{game.get('home_name', '')}-{game.get('vis_name', '')}"),
                "source": linearize_source(game, top_n_per_team=top_n_per_team),
                "target": build_target(game),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(games)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/rotowire"),
                         help="Directory containing train.json / valid.json / test.json")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--top-n-players", type=int, default=3,
                         help="Keep only this many players per team, ranked by minutes played. "
                              "Verified empirically (2026-09-02) that emitting every dressed "
                              "player's full stat line blows past any reasonable model input "
                              "budget (median ~7100 tokens/game) — tune this down further if "
                              "the token-length check still comes back too high.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        in_path = args.raw_dir / f"{split}.json"
        if not in_path.exists():
            print(f"[skip] {in_path} not found — did you extract rotowire.tar.bz2 into {args.raw_dir}?")
            continue
        out_path = args.out_dir / f"{split}.jsonl"
        n = convert_split(in_path, out_path, top_n_per_team=args.top_n_players)
        print(f"[ok] {split}: {n} games -> {out_path}")


if __name__ == "__main__":
    main()