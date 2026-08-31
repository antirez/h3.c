"""Fail if webui/shared/options.schema.json drifts from the h3.c CLI.

The schema is the single source of truth for the web UI. It is hand-maintained,
so this test re-reads the C sources and compares what can be compared
mechanically: the set of long options, the short options, and the numeric
constants and defaults the UI depends on.
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = json.loads((ROOT / "webui/shared/options.schema.json").read_text())


def _cli_long_options() -> set[str]:
    source = (ROOT / "main.c").read_text()
    block = re.search(
        r"static const struct option options\[\] = \{(.*?)\n    \};",
        source,
        re.DOTALL,
    )
    assert block, "option table not found in main.c"
    return {name for name in re.findall(r'\{"([a-z0-9-]+)",', block.group(1))}


def _cli_short_options() -> set[str]:
    source = (ROOT / "main.c").read_text()
    spec = re.search(r'getopt_long\(argc, argv, "([^"]+)"', source)
    assert spec, "getopt_long spec not found in main.c"
    return {c for c in spec.group(1) if c.isalnum()}


def _c_text(name: str) -> str:
    """Source with adjacent C string literals joined, so split messages match."""
    return re.sub(r'"\s*"', "", (ROOT / name).read_text())


def _define(header: str, name: str) -> str:
    source = (ROOT / header).read_text()
    match = re.search(rf"^#define {name}\s+(.+)$", source, re.MULTILINE)
    assert match, f"{name} not found in {header}"
    return match.group(1).strip()


def test_every_cli_long_option_is_in_the_schema():
    assert _cli_long_options() == {o["flag"][2:] for o in SCHEMA["options"]}


def test_short_options_match():
    schema_short = {o["short"][1:] for o in SCHEMA["options"] if "short" in o}
    assert schema_short == _cli_short_options()


def test_keys_and_flags_are_unique():
    flags = [o["flag"] for o in SCHEMA["options"]]
    keys = [o["key"] for o in SCHEMA["options"]]
    assert len(flags) == len(set(flags))
    assert len(keys) == len(set(keys))


def test_every_option_declares_a_group_that_exists():
    groups = {g["id"] for g in SCHEMA["groups"]}
    assert {o["group"] for o in SCHEMA["options"]} <= groups
    assert {o["ui"] for o in SCHEMA["options"]} <= {"simple", "advanced", "hidden"}


def test_constants_match_the_c_headers():
    constants = SCHEMA["constants"]
    assert constants["fps"] == int(_define("h3_host.h", "H3_FPS"))
    assert constants["canvas_multiple"] == int(
        _define("h3_host.h", "H3_CANVAS_MULTIPLE")
    )
    assert constants["max_steps"] == int(_define("h3_host.h", "H3_MAX_STEPS"))
    assert constants["max_pixels"] == eval(_define("h3_host.h", "H3_MAX_PIXELS"))
    assert constants["dit_layers"]["min"] == int(
        _define("h3.h", "H3_MIN_DIT_LAYERS")
    )
    assert constants["dit_layers"]["max"] == int(
        _define("h3.h", "H3_DEFAULT_DIT_LAYERS")
    )


def test_defaults_match_h3_header():
    by_key = {o["key"]: o for o in SCHEMA["options"]}
    assert by_key["width"]["default"] == int(_define("h3.h", "H3_DEFAULT_WIDTH"))
    assert by_key["height"]["default"] == int(_define("h3.h", "H3_DEFAULT_HEIGHT"))
    assert by_key["frames"]["default"] == int(_define("h3.h", "H3_DEFAULT_FRAMES"))
    assert by_key["steps"]["default"] == int(_define("h3.h", "H3_DEFAULT_STEPS"))
    assert by_key["dit_layers"]["default"] == int(
        _define("h3.h", "H3_DEFAULT_DIT_LAYERS")
    )


def test_frame_alignment_matches_h3_host():
    source = (ROOT / "h3_host.c").read_text()
    body = re.search(
        r"int h3_align_frame_count\(int requested\) \{(.*?)\n\}", source, re.DOTALL
    )
    assert body, "h3_align_frame_count not found"
    frames = SCHEMA["constants"]["frames"]
    assert f"< {frames['align_base']} ?" in body.group(1)
    assert f"% {frames['align_stride']}" in body.group(1)


def test_validation_messages_are_quoted_from_h3_c():
    source = _c_text("h3.c")
    for entry in SCHEMA["constraints"]:
        assert entry["message"] in source, entry["id"]
    for entry in SCHEMA["mutual_exclusions"]:
        if entry.get("source") == "h3.c":
            assert entry["message"] in source, entry["message"]


def test_reference_limits_match_h3_c():
    source = _c_text("h3.c")
    refs = SCHEMA["references"]
    assert "Ref2VA supports at most 12 references" in source
    assert refs["max_total"] == 12
    assert (
        f"Ref2VA limits are {refs['max_images']} images, "
        f"{refs['max_videos']} videos, and {refs['max_audio_inputs']} audio inputs"
        in source
    )
