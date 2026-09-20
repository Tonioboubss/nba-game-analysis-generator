"""Gradio demo: pick a test-set game, compare the generated recap to the real one.

Run locally against a local checkpoint:
    python src/app.py --model-dir models/bart-base-final --test data/processed/test.jsonl

Deploy as a Hugging Face Space: a Space starts this script with NO CLI args
(just `python app.py`), so --model-dir/--test fall back to MODEL_ID/TEST_FILE
below — set those to your pushed Hub repo id and a small bundled test-set
sample (see the project recap's Phase 7/8 notes for the exact push steps).
Both can also be overridden via the MODEL_ID/TEST_FILE environment variables
(Space Settings > Variables) without touching this file.

Later (see project recap): swap the --test file for live/recent games pulled
via nba_api, once they've been reshaped through preprocess.py's linearization
format — the model was trained on that exact input shape and will degrade on
anything structurally different.

Module-level `demo`, and why: a Hugging Face "Gradio SDK" Space does not run
your script with plain `python app.py` under the hood — it launches it through
Gradio's own `gradio` CLI wrapper, which turns on Gradio's file-watch/hot-reload
tooling by default (visible in a Space's container logs as a line like
"GRADIO_HOT_RELOAD: Launching demo not found in __main__. Using 'demo'"). That
tooling statically scans this file for a variable literally named `demo` at
module level (not nested inside any function) to know which Blocks app is "the"
app to watch/relaunch. Without one, on Gradio 6.28.0 the fallback path is what
caused the deploy crash this file used to have — a stray internal "ping"
sentinel got dispatched into the first real event handler it found
(game_dropdown.change) instead of being safely no-op'd, surfacing as
`gradio.exceptions.Error: "Value: ping is not in the list of choices: [...]"`.
Defining `demo` at column 0 (below) is Gradio's own documented fix for
"Cannot statically find a gradio demo called demo".

UI note: the raw linearized source (`<record> entity | stat | value | HOME|VIS
</record>` spans, see preprocess.py) is never shown to the user directly — it's
reparsed here (see parse_boxscore_html / _group_records) into a proper
scoreboard + stat tables, so the demo reads as "box-score in, analysis out"
rather than exposing the model's internal input format. The raw string is
still what's actually fed to the model (via a hidden gr.State) — only the
*display* changes.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from html import escape as _esc
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

try:
    # Only importable inside an actual Hugging Face Space — ZeroGPU allocates
    # a GPU on demand for the duration of a function decorated with this.
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    # Local run / Colab: no-op decorator, generate_recap runs on whatever
    # device _load_state() already moved the model to (CPU or a real Colab GPU).
    def gpu_decorator(fn):
        return fn

MODEL_ID = os.environ.get("MODEL_ID", "antoineboubouu/bart-base-rotowire-recap")
TEST_FILE = os.environ.get("TEST_FILE", "data/test_sample.jsonl")

_STATE = {}

# Matches one linearized record span: "<record> entity | stat | value | label </record>"
# — see preprocess.py's _team_records / _player_records, which is the only
# place this exact format is produced.
_RECORD_RE = re.compile(r"<record>\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*</record>")

# "10_30_14-Clippers-Thunder" -> ("10", "30", "14", "Clippers-Thunder")
_GAME_ID_RE = re.compile(r"^(\d{2})_(\d{2})_(\d{2})-(.+)$")


def load_test_games(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def pretty_game_label(game_id: str) -> str:
    """Turn a raw RotoWire game id into a readable dropdown label. Best-effort
    only (falls back to the raw id) — the id's own team-name order (which
    team is listed first) was never verified against home/vis, so this never
    claims a "home vs away" order itself; that distinction is drawn instead
    from the actual HOME/VIS tags in the box-score records themselves, which
    *are* sourced directly from the structured data (see render_boxscore_html)."""
    m = _GAME_ID_RE.match(game_id)
    if not m:
        return game_id
    month, day, yy = m.group(1), m.group(2), m.group(3)
    teams = m.group(4).split("-")
    matchup = " vs ".join(teams) if len(teams) == 2 else m.group(4).replace("-", " / ")
    return f"{day}/{month}/20{yy} — {matchup}"


def _group_records(source: str) -> list[dict]:
    """Parse the flat "<record> ..." string back into ordered groups — one
    group per contiguous run of records sharing the same entity. Because
    preprocess.py always emits, in order, all of the home team's stat
    records, then all of the away team's, then each player's stats grouped
    together, group 0 is always the home team, group 1 the away team, and
    everything after is a player, tagged HOME/VIS via its own records'
    4th field (not guessed) — schema-driven, not hardcoded to specific stat
    names, since home_line/vis_line's exact field set is documented
    elsewhere as unverified."""
    groups: list[dict] = []
    for entity, stat, value, label in _RECORD_RE.findall(source):
        if groups and groups[-1]["entity"] == entity:
            groups[-1]["stats"].append((stat, value))
        else:
            groups.append({"entity": entity, "label": label, "stats": [(stat, value)]})
    return groups


_PLAYER_STAT_COLUMNS = ["MIN", "PTS", "REB", "AST", "STL", "BLK"]


def render_boxscore_html(source: str) -> str:
    """Render the linearized source as a scoreboard + team stat table + two
    per-team player tables, instead of showing the tagged model-input string
    verbatim."""
    groups = _group_records(source)
    if len(groups) < 2:
        return "<p><em>Box-score illisible (format inattendu).</em></p>"

    home_team, vis_team = groups[0], groups[1]
    home_players = [g for g in groups[2:] if g["label"] == "HOME"]
    vis_players = [g for g in groups[2:] if g["label"] == "VIS"]

    def stat(group: dict, name: str):
        for k, v in group["stats"]:
            if k.upper() == name:
                return v
        return None

    home_pts = stat(home_team, "PTS") or "?"
    vis_pts = stat(vis_team, "PTS") or "?"

    scoreboard = f'''
    <div style="display:flex;justify-content:space-between;align-items:center;
                background:#1a1f2b;color:#f5f5f5;border-radius:10px;
                padding:14px 20px;margin-bottom:14px;">
      <div style="text-align:left;">
        <div style="font-size:11px;opacity:.65;letter-spacing:.05em;">DOMICILE</div>
        <div style="font-weight:700;font-size:15px;">{_esc(home_team["entity"])}</div>
      </div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:28px;font-weight:700;
                  letter-spacing:1px;padding:0 16px;">{_esc(home_pts)} – {_esc(vis_pts)}</div>
      <div style="text-align:right;">
        <div style="font-size:11px;opacity:.65;letter-spacing:.05em;">EXTÉRIEUR</div>
        <div style="font-weight:700;font-size:15px;">{_esc(vis_team["entity"])}</div>
      </div>
    </div>
    '''

    home_stats = dict(home_team["stats"])
    vis_stats = dict(vis_team["stats"])
    stat_order = [k for k, _ in home_team["stats"]]
    team_rows = "".join(
        f'<tr><td style="padding:4px 8px;font-family:monospace;font-size:11px;color:#888;">{_esc(k)}</td>'
        f'<td style="padding:4px 8px;text-align:right;">{_esc(home_stats.get(k, "–"))}</td>'
        f'<td style="padding:4px 8px;text-align:right;">{_esc(vis_stats.get(k, "–"))}</td></tr>'
        for k in stat_order
    )
    team_table = f'''
    <table style="width:100%;border-collapse:collapse;margin-bottom:16px;font-size:13px;">
      <thead><tr>
        <th style="text-align:left;padding:4px 8px;font-size:11px;color:#888;">STAT ÉQUIPE</th>
        <th style="text-align:right;padding:4px 8px;font-size:11px;color:#888;">DOM.</th>
        <th style="text-align:right;padding:4px 8px;font-size:11px;color:#888;">EXT.</th>
      </tr></thead>
      <tbody>{team_rows}</tbody>
    </table>
    '''

    def player_table(title: str, players: list[dict]) -> str:
        if not players:
            return ""
        header_cells = "".join(
            f'<th style="text-align:right;padding:4px 8px;font-size:11px;color:#888;">{c}</th>'
            for c in _PLAYER_STAT_COLUMNS
        )
        body_rows = ""
        for p in players:
            pstats = dict(p["stats"])
            cells = "".join(
                f'<td style="padding:4px 8px;text-align:right;">{_esc(pstats.get(c, "–"))}</td>'
                for c in _PLAYER_STAT_COLUMNS
            )
            body_rows += f'<tr><td style="padding:4px 8px;">{_esc(p["entity"])}</td>{cells}</tr>'
        return f'''
        <div style="margin-bottom:14px;">
          <div style="font-size:11px;font-weight:700;color:#888;text-transform:uppercase;
                      letter-spacing:.04em;margin-bottom:4px;">{_esc(title)}</div>
          <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <thead><tr>
              <th style="text-align:left;padding:4px 8px;font-size:11px;color:#888;">Joueur</th>
              {header_cells}
            </tr></thead>
            <tbody>{body_rows}</tbody>
          </table>
        </div>
        '''

    players_html = player_table(f"{home_team['entity']} — meneurs (par minutes jouées)", home_players)
    players_html += player_table(f"{vis_team['entity']} — meneurs (par minutes jouées)", vis_players)

    return scoreboard + team_table + players_html


def format_candidates_markdown(candidates: list[str]) -> str:
    if len(candidates) == 1:
        return candidates[0]
    return "\n\n---\n\n".join(f"**Candidat {i}**\n\n{c}" for i, c in enumerate(candidates, start=1))


@gpu_decorator
def generate_recap(source_text: str, num_candidates: int, temperature: float) -> list[str]:
    tokenizer = _STATE["tokenizer"]
    model = _STATE["model"]
    inputs = tokenizer(source_text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(_STATE["device"]) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        num_return_sequences=num_candidates,
        no_repeat_ngram_size=3,  # sampling reduces looping but doesn't rule it out
        repetition_penalty=1.3,  # on a weak/undertrained checkpoint
    )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


def on_pick_game(game_id: str):
    game = next(g for g in _STATE["games"] if g["id"] == game_id)
    box_html = render_boxscore_html(game["source"])
    # Order matches build_ui()'s outputs=[box_score_html, source_state, real_summary, generated_box]
    return box_html, game["source"], game["target"], ""


def on_generate(source_text: str, num_candidates: int, temperature: float):
    candidates = generate_recap(source_text, int(num_candidates), temperature)
    return format_candidates_markdown(candidates)


def build_ui() -> gr.Blocks:
    games = _STATE["games"]
    choices = [(g["label"], g["id"]) for g in games]
    default_id = games[0]["id"] if games else None

    # Named `ui` here (a purely local variable) so it can't be confused with
    # the module-level `demo` name that Gradio's reload tooling looks for —
    # that one is assigned once, below, from this function's return value.
    with gr.Blocks(title="RotoWire recap generator", theme=gr.themes.Soft()) as ui:
        gr.Markdown(
            "# 🏀 NBA box-score → recap generator\n"
            "Un modèle **BART-base + LoRA** (fine-tuné sur RotoWire) génère l'analyse d'un "
            "match à partir de son seul box-score structuré, sans jamais voir le résumé "
            "humain de référence. Choisis un match du test set (jamais vu à l'entraînement), "
            "génère un ou plusieurs résumés, et compare au vrai résumé écrit par un humain."
        )
        game_dropdown = gr.Dropdown(
            choices=choices, value=default_id,
            label="Match du test set",
        )

        # Holds the raw "<record> ..." string actually fed to the model —
        # never rendered as-is (see render_boxscore_html), just passed through.
        source_state = gr.State(value="")

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("### 📊 Box-score — ce que voit le modèle")
                box_score_html = gr.HTML()
            with gr.Column(scale=6):
                gr.Markdown("### ✍️ Résumé généré")
                with gr.Row():
                    num_candidates = gr.Slider(1, 5, value=3, step=1, label="Candidats à échantillonner")
                    temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Température d'échantillonnage")
                generate_btn = gr.Button("🎲 Générer", variant="primary")
                generated_box = gr.Markdown()
                gr.Markdown("### 📰 Résumé réel (rédigé par un humain)")
                real_summary = gr.Markdown()

        game_dropdown.change(
            on_pick_game, inputs=game_dropdown,
            outputs=[box_score_html, source_state, real_summary, generated_box],
        )
        generate_btn.click(
            on_generate, inputs=[source_state, num_candidates, temperature], outputs=generated_box,
        )

        if choices:
            default_game_id = choices[0][1]
            ui.load(
                lambda: on_pick_game(default_game_id),
                outputs=[box_score_html, source_state, real_summary, generated_box],
            )

    return ui


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=None,
                         help="Base model id the LoRA adapter was trained from (e.g. t5-base). "
                              "If omitted, read from the adapter's own config.")
    parser.add_argument("--model-dir", default=MODEL_ID,
                         help="Local path OR Hugging Face Hub repo id of the saved LoRA adapter. "
                              "Defaults to MODEL_ID above so a Space works with no CLI args.")
    parser.add_argument("--test", type=Path, default=Path(TEST_FILE),
                         help="Path to a jsonl file of {id, source, target} rows. Defaults to "
                              "TEST_FILE above (a small bundled sample, not the full test set, "
                              "to keep the Space repo lightweight).")
    parser.add_argument("--sample", type=int, default=20, help="Only load N random test games into the dropdown")
    return parser.parse_args()


def _load_state(args) -> None:
    """Load the tokenizer/model/test games into _STATE. Split out of the old
    main() so module-level code (just below) can call it before build_ui() —
    see the file's top-level docstring for why `demo` needs to exist at
    module scope rather than inside a function."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    base_model_id = args.base_model
    if base_model_id is None:
        local_config = Path(args.model_dir) / "adapter_config.json"
        if local_config.exists():
            config_path = local_config
        else:
            # args.model_dir is a Hub repo id, not a local folder — download
            # just this one small file to read the base model id from it.
            config_path = hf_hub_download(args.model_dir, "adapter_config.json")
        with open(config_path) as f:
            base_model_id = json.load(f)["base_model_name_or_path"]
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)
    # torch_device="cpu" forced explicitly: on a ZeroGPU Space, torch.cuda.is_available()
    # is patched to report True even here, at plain startup, before any real GPU is
    # attached — left to auto-infer, PEFT tries to load the adapter straight onto a
    # "cuda" device that doesn't physically exist yet and crashes with "No CUDA GPUs
    # are available". Loading on CPU first and moving the whole model with .to(device)
    # right below is the deferred move ZeroGPU actually knows how to handle.
    model = PeftModel.from_pretrained(base_model, args.model_dir, torch_device="cpu")
    model.eval()

    # On a ZeroGPU Space, CUDA is only actually attached during a call to a
    # function decorated with @spaces.GPU (generate_recap) — .to("cuda") here
    # still works (ZeroGPU intercepts it) and is what the docs recommend
    # doing at load time, not inside the decorated function itself.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    games = load_test_games(args.test)
    random.Random(0).shuffle(games)
    sampled = games[: args.sample]
    for g in sampled:
        g["label"] = pretty_game_label(g["id"])

    _STATE["games"] = sampled
    _STATE["tokenizer"] = tokenizer
    _STATE["model"] = model
    _STATE["device"] = device


# --- Module-level setup: runs once at import, both for a local `python
# src/app.py ...` run and inside a Space (which also imports/runs this file
# as __main__, just via the `gradio` wrapper instead of plain `python`). This
# is what gives Gradio's reload tooling a real, literal top-level `demo`
# name to find — see the file's top-level docstring for the full story.
_args = _parse_args()
_load_state(_args)
demo = build_ui()

if __name__ == "__main__":
    demo.launch()