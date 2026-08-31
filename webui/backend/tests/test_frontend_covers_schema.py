"""Every CLI option must be reachable from the UI, and nothing may drift.

The frontend has no test runner of its own: these checks read its sources, so
a flag added to h3.c and to the schema cannot quietly miss a control.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = json.loads((ROOT / "webui/shared/options.schema.json").read_text())
FRONTEND = ROOT / "webui/frontend/src"
SOURCES = "\n".join(
    path.read_text()
    for path in FRONTEND.rglob("*.ts*")
    if "generated" not in path.parts
)
GENERATED = (FRONTEND / "generated/options.ts").read_text()
# JSX wraps prose across lines: compare on a single normalized line.
FLAT = re.sub(r"\s+", " ", SOURCES)


def _flags(text: str) -> set[str]:
    return set(re.findall(r"--[a-z0-9-]+", text))


def test_every_visible_option_has_a_control_in_the_ui():
    """Named in a component, or rendered from the generated list (parity flags)."""
    visible = {o["flag"] for o in SCHEMA["options"] if o["ui"] != "hidden"}
    parity = {o["flag"] for o in SCHEMA["options"] if o["group"] == "parity"}
    named = _flags(SOURCES)
    assert visible - parity <= named, sorted(visible - parity - named)
    # The parity checkboxes are rendered one per flag from the generated list.
    assert "ALL_SLOWER_FLAGS.map" in FLAT
    assert parity <= _flags(GENERATED), sorted(parity - _flags(GENERATED))


def test_every_option_key_reaches_the_job_spec_type():
    types = (FRONTEND / "types.ts").read_text()
    server_side = {"model_dir", "output", "show", "zoom", "info", "help"}
    # Flags whose directory the server assigns: the UI exposes a toggle.
    toggles = {"frames_dir": "write_frames", "preview": "preview"}
    for option in SCHEMA["options"]:
        key = option["key"]
        if key in server_side or option["type"] == "reference":
            continue
        if option["group"] == "parity":
            # The ten --use-slower-* flags travel together in `slower`.
            assert re.search(r"^  slower: string\[\];", types, re.MULTILINE)
            continue
        field = toggles.get(key, key)
        assert re.search(rf"^  {field}[?:]", types, re.MULTILINE), key


def test_the_generated_options_file_is_current():
    for option in SCHEMA["options"]:
        assert f'"flag": "{option["flag"]}"' in GENERATED, option["flag"]
    assert '"max_pixels": ' + str(SCHEMA["constants"]["max_pixels"]) in GENERATED


def test_the_parity_flags_are_listed_individually():
    parity = [o["flag"][2:] for o in SCHEMA["options"] if o["group"] == "parity"]
    for flag in parity:
        assert flag in GENERATED, flag
    assert len(parity) == 10


def test_blocking_constraints_reach_the_person_in_plain_words():
    """The engine's wording is translated, not repeated."""
    app = (FRONTEND / "App.tsx").read_text()
    # Whatever h3 refuses, the browser shows the translation first.
    assert "explain(problems[0]).title" in app
    assert "explain(problems[0]).fix" in app
    assert "what h3 reported" in app, "the technical message stays available"
    for needle in [
        "fixed lengths",              # frame alignment
        "Multiples of 32",            # canvas grid
        "two different ways to work", # anchors versus references
        "2 to 15 seconds",            # reference audio
        "at least 56 frames",         # soundtrack length
    ]:
        assert needle in FLAT, needle


def test_the_post_processing_section_is_driven_by_the_api():
    """A plugin that becomes available must not need a UI change."""
    expert = (FRONTEND / "components/Expert.tsx").read_text()
    assert "plugins.map" in expert
    assert "disabled={!plugin.available}" in expert
    assert "plugin.reason" in expert and "plugin.notice" in expert
    assert "faceswap" not in expert, "the plugin name must come from the API"


def test_the_composing_view_says_nothing_in_the_engine_s_words():
    """Rubric M5 criterion 1, on the code that actually ships.

    R23 kept every CLI flag visible in Create as a secondary label. R28 moved
    them out: composing shows a sentence about the shot, and the flags live
    under Expert with the exact name they have on the command line.
    """
    create = (FRONTEND / "components/Create.tsx").read_text()
    # The whole surface a person composes on, not only the first screen.
    for name in ["Create", "FineTune", "References", "PhotoSlot"]:
        source = (FRONTEND / f"components/{name}.tsx").read_text()
        assert 'className="flag"' not in source, f"a flag chip is back in {name}"
        assert "<code>--" not in source, f"a flag is back in {name}"
    named = re.search(r'["\'`][^"\'`]*--[a-z]', create)
    assert not named, "a flag name is back in Create"

    # Everything a person reads there, with the imports and the interpolations
    # taken out.
    prose = "\n".join(
        line for line in create.splitlines() if not line.startswith("import ")
    )
    literals = re.findall(r'"([^"\n]{3,})"|\u0060([^\u0060]{3,})\u0060', prose)
    words = set()
    for double, backtick in literals:
        text = re.sub(r"\$\{[^}]*\}", " ", double or backtick)
        words |= {word.strip(",.:;()").lower() for word in text.split()}
    jargon = {
        "dit", "denoise", "reuse", "token", "rope", "int8", "canvas", "vae",
        "latent", "sampler", "steps", "layers", "seed", "checkpoint",
        "inference", "tensor",
        # "frame" is not on this list: it is the subject's own word, and the
        # first frame of a video is a thing anyone can picture.
    }
    assert not words & jargon, sorted(words & jargon)

    # And the flags are somewhere: under Expert, named exactly.
    expert = (FRONTEND / "components/Expert.tsx").read_text()
    for flag in ["--steps", "--layers", "--seed", "--first-frame", "--last-frame"]:
        assert flag in expert, flag


def test_every_choice_that_changes_the_wait_shows_it():
    """Rubric M4 criterion 2."""
    create = (FRONTEND / "components/Create.tsx").read_text()
    assert "shapeSeconds" in create and "qualitySeconds" in create
    assert "totalSeconds" in create
    app = (FRONTEND / "App.tsx").read_text()
    assert "useEstimates" in app
    fine = (FRONTEND / "components/FineTune.tsx").read_text()
    assert "Delta" in fine and "saves" in fine and "adds" in fine
