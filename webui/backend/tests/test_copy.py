"""The plain-language layer: complete, jargon-free, and with no orphans.

`copy.json` is what the interface says; `options.schema.json` is what h3
accepts. These tests keep the first honest about the second.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COPY = json.loads((ROOT / "webui/shared/copy.json").read_text())
SCHEMA = json.loads((ROOT / "webui/shared/options.schema.json").read_text())
JOBSPEC = (ROOT / "webui/backend/app/jobspec.py").read_text()

# Words that describe how the engine is built, not what a person controls.
JARGON = {
    "dit", "denoise", "denoising", "denoiser", "reuse", "token", "tokens",
    "rope", "int8", "bf16", "vae", "latent", "canvas", "sampler", "inference",
    "checkpoint", "transformer", "quantization", "ref2va", "fl2va", "adaln",
}


def _engine_errors() -> set[str]:
    """Every message the validator can produce, as written in jobspec.py."""
    found = set()
    for match in re.finditer(r"errors\.append\(\s*(.*?)\s*\)\n", JOBSPEC, re.DOTALL):
        raw = match.group(1)
        parts = re.findall(r'f?"([^"]*)"', raw)
        if parts:
            found.add("".join(parts))
    return found


def test_every_option_a_person_can_set_has_a_plain_name():
    server_side = {"show", "zoom", "info", "help"}
    for option in SCHEMA["options"]:
        key = option["key"]
        if key in server_side:
            continue
        if option["group"] == "parity":
            assert "slower" in COPY["options"], "the parity flags share one entry"
            continue
        if option["type"] == "reference":
            assert key in COPY["options"], key
            continue
        assert key in COPY["options"], key
        assert COPY["options"][key]["help"], key


def test_no_plain_name_contains_engine_jargon():
    for key, entry in COPY["options"].items():
        words = {word.strip(",.()").lower() for word in entry["name"].split()}
        assert not words & JARGON, f"{key}: {entry['name']}"


def test_every_engine_error_has_a_plain_translation():
    translations = COPY["errors"]
    for message in _engine_errors():
        assert any(entry["match"] in message for entry in translations), message


def test_no_translation_is_orphaned():
    engine = _engine_errors()
    for entry in COPY["errors"]:
        assert any(entry["match"] in message for message in engine), entry["match"]


def test_every_translation_says_what_to_do():
    for entry in COPY["errors"]:
        assert entry["title"].endswith((".", "?")), entry["match"]
        assert len(entry["fix"].split()) >= 6, entry["match"]
        # An instruction, not an apology.
        assert "sorry" not in entry["fix"].lower()


def test_every_progress_phase_has_a_plain_name():
    """The phases h3 actually emits, read from the C sources."""
    emitted = set()
    for source in ("h3.c", "h3_dit.c"):
        text = (ROOT / source).read_text()
        emitted |= set(re.findall(r'progress[^"]*"([a-zA-Z0-9 ]+)", ', text))
    known = {phase for phase in COPY["phases"]}
    missing = {
        phase for phase in emitted
        if phase not in known and len(phase.split()) <= 4
    }
    assert not missing, sorted(missing)


def test_job_states_are_named_for_people():
    from app.runner import TERMINAL_STATES

    for state in TERMINAL_STATES | {"queued", "running"}:
        assert state in COPY["states"], state
