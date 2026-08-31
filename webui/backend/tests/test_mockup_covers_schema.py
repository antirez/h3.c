"""The mockup must show every CLI option, so no flag is silently dropped.

The covering rule (design rubric M1): every option of the schema that the UI
does not exclude appears in a screen with its CLI flag visible, and every
excluded one is listed as excluded.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = json.loads((ROOT / "webui/shared/options.schema.json").read_text())
MOCKUP = (ROOT / "docs/mockup/index.html").read_text()
MOCKUP_V2 = (ROOT / "docs/mockup/v2.html").read_text()


def _flags_in(html: str) -> set[str]:
    return set(re.findall(r"--[a-z0-9-]+", html))


def test_every_visible_option_appears_with_its_flag():
    visible = {o["flag"] for o in SCHEMA["options"] if o["ui"] != "hidden"}
    assert visible <= _flags_in(MOCKUP), sorted(visible - _flags_in(MOCKUP))


def test_the_redesign_still_reaches_every_option():
    """Rubric M4 criterion 5: nothing disappears in the new design."""
    visible = {o["flag"] for o in SCHEMA["options"] if o["ui"] != "hidden"}
    assert visible <= _flags_in(MOCKUP_V2), sorted(visible - _flags_in(MOCKUP_V2))
    for flag in ("--show", "--zoom"):
        assert flag in MOCKUP_V2, f"{flag} must be listed as not exposed"


def test_the_redesign_names_controls_in_plain_language():
    """Rubric M4 criterion 1: no jargon as a primary label in Create."""
    create = MOCKUP_V2[
        MOCKUP_V2.index('<section class="stage composing">') :
        MOCKUP_V2.index('<details class="level">')
    ]
    headings = re.findall(r'<div class="label">([^<]+)', create)
    jargon = ("dit", "denoise", "reuse", "token", "rope", "int8", "canvas", "vae",
              "latent", "sampler", "inference")
    for heading in headings:
        words = heading.lower().split()
        assert not any(word.strip(",.") in jargon for word in words), heading


def test_hidden_options_are_declared_as_not_exposed():
    section = re.search(
        r"<summary>Not exposed by this UI.*?</details>", MOCKUP, re.DOTALL
    )
    assert section, "the 'Not exposed by this UI' section is missing"
    hidden = {o["flag"] for o in SCHEMA["options"] if o["ui"] == "hidden"}
    assert hidden <= _flags_in(section.group(0)), sorted(
        hidden - _flags_in(section.group(0))
    )


def test_every_blocking_constraint_is_shown_before_submitting():
    """Constraints that make h3 refuse a job must be visible in the form."""
    for needle in [
        "multiples of 32",           # canvas_multiple
        "768 × 1344",                 # max_pixels
        "Set both or neither",        # render_pair / render_shape
        "5 + 17n",                    # frame_range
        "22 frames",                  # frame_minimum
        "2 … 1000",                   # steps_range
        "35 … 50",                    # layers_range
        "Mutually exclusive",         # reuse vs core-reuse, seconds vs frames
        "cannot be combined with ordered references",  # anchors vs references
        "at least 2 s at 32 kHz",     # audio minimum
        "≤ 15 s",                     # audio total
        "at least 56 output frames",  # soundtrack duration
        "9 images, 3 videos, 3 audio inputs",  # reference limits
    ]:
        assert needle in MOCKUP, needle


def test_all_job_states_are_represented():
    for state in ["running", "queued", "failed", "done"]:
        assert f'class="st {state[:3]}"' in MOCKUP or f">{state}<" in MOCKUP, state
