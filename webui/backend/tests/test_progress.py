"""The weighted progress model: monotonic, scaled, and honest about gaps."""

import pytest

from app.config import Settings
from app.jobspec import JobSpec
from app.progress import ProgressModel, load_weights

WEIGHTS = {
    "reference": {"width": 256, "height": 256, "frames": 22, "steps": 8, "layers": 50},
    "phase_seconds": {
        "tokenizer": 2.0,
        "text encoder": 16.0,
        "load transformer core": 40.0,
        "denoise": 8.0,
        "video VAE load": 14.0,
    },
    "total_seconds": 80.0,
}
SPEC = JobSpec(prompt="x", width=256, height=256, frames=22, steps=8)


@pytest.fixture
def model():
    return ProgressModel(WEIGHTS)


def test_the_shipped_weights_load_and_cover_the_main_phases(model):
    shipped = ProgressModel(load_weights(Settings().progress_weights_path))
    phases = dict(shipped.plan(SPEC))
    assert {"text encoder", "load transformer core", "denoise"} <= set(phases)
    assert all(seconds > 0 for seconds in phases.values())


def test_progress_never_goes_backwards_across_a_whole_run(model):
    timeline = [
        ("tokenizer", 0, 1), ("tokenizer", 1, 1),
        ("text encoder", 0, 50), ("text encoder", 25, 50), ("text encoder", 50, 50),
        ("load transformer core", 1, 50), ("load transformer core", 50, 50),
        ("denoise", 0, 8), ("denoise", 4, 8), ("denoise", 8, 8),
        ("video VAE load", 1, 36), ("video VAE load", 36, 36),
    ]
    seen = [model.fraction(SPEC, *point) for point in timeline]
    assert seen == sorted(seen)
    assert seen[0] == 0.0
    assert seen[-1] == pytest.approx(1.0)


def test_a_heavier_job_gives_denoise_a_bigger_share(model):
    light = model.fraction(SPEC, "denoise", 4, 8)
    heavy_spec = SPEC.model_copy(update={"steps": 50})
    heavy = model.fraction(heavy_spec, "denoise", 25, 50)
    # At the same relative point of denoising, more steps means the phases
    # before it are worth proportionally less.
    assert heavy < light


def test_resolution_scales_the_denoise_and_decode_budget(model):
    small = dict(model.plan(SPEC))
    large = dict(model.plan(SPEC.model_copy(update={"width": 512, "height": 512})))
    assert large["denoise"] == pytest.approx(small["denoise"] * 4)
    assert large["video VAE load"] == pytest.approx(small["video VAE load"] * 4)
    assert large["text encoder"] == small["text encoder"]


def test_the_internal_canvas_is_what_counts_for_scaling(model):
    spec = SPEC.model_copy(update={"width": 512, "height": 512,
                                   "render_width": 256, "render_height": 256})
    assert dict(model.plan(spec))["denoise"] == pytest.approx(
        dict(model.plan(SPEC))["denoise"]
    )


def test_an_unknown_phase_reports_nothing_rather_than_guessing(model):
    # The runner keeps the highest value seen, so 0 means "no news", not a
    # regression. Reporting the sum of the known phases would claim 100%.
    assert model.fraction(SPEC, "Qwen vision", 3, 10) == 0.0


def test_missing_phases_in_the_weights_still_produce_a_full_bar():
    sparse = ProgressModel(
        {"reference": WEIGHTS["reference"], "phase_seconds": {"denoise": 1.0}}
    )
    assert sparse.fraction(SPEC, "denoise", 8, 8) == pytest.approx(1.0)
    assert sparse.fraction(SPEC, "tokenizer", 1, 1) == 0.0


def test_the_eta_corrects_itself_from_the_observed_pace(model):
    # Half way through by weight, having taken 100 s: expect about 100 s left.
    half = model.fraction(SPEC, "load transformer core", 33, 50)
    assert 0.4 < half < 0.6
    remaining = model.remaining_seconds(
        SPEC, "load transformer core", 33, 50, elapsed=100.0
    )
    assert 60 < remaining < 160


def test_the_eta_before_any_progress_falls_back_to_the_budget(model):
    budget = sum(seconds for _, seconds in model.plan(SPEC))
    assert model.remaining_seconds(SPEC, None, 0, 0, elapsed=0.0) == pytest.approx(
        budget
    )


def test_enabling_the_preview_adds_a_load_phase(model):
    plain = dict(model.plan(SPEC))
    with_preview = dict(model.plan(SPEC.model_copy(update={"preview": True})))
    assert "preview VAE load" not in plain
    assert with_preview["preview VAE load"] > 0


def test_the_stored_progress_only_ever_grows(tmp_path):
    """End to end: the runner must not let the bar step back."""
    import stat
    import time

    from conftest import authed_client

    script = (
        "#!/bin/sh\n"
        "printf '\\rload transformer core        50/50  ' >&2\n"
        "printf '\\rQwen vision                   1/4   ' >&2\n"
        "printf '\\rdenoise                       1/2   ' >&2\n"
        "printf '\\rdenoise                       2/2   \\n' >&2\n"
        "exit 0\n"
    )
    binary = tmp_path / "h3"
    binary.write_text(script)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    config = Settings(binary=binary, model_dir=tmp_path, data_dir=tmp_path / "data")
    seen: list[float] = []
    with authed_client(config) as client:
        client.app.state.runner.add_listener(lambda job: seen.append(job["progress"]))
        job_id = client.post(
            "/api/jobs",
            json={"prompt": "x", "width": 256, "height": 256, "frames": 22, "steps": 2},
        ).json()["id"]
        deadline = time.time() + 20
        while time.time() < deadline:
            if client.get(f"/api/jobs/{job_id}").json()["state"] not in (
                "queued",
                "running",
            ):
                break
            time.sleep(0.05)
    assert seen == sorted(seen)
    assert seen[-1] == 1.0


# ── fixed cost versus cost that scales ──────────────────────────────────────

TWO_SAMPLES = {
    "reference": {"width": 256, "height": 256, "frames": 22, "steps": 8, "layers": 50},
    "phase_seconds": {"text encoder": 16.0, "denoise": 8.0, "video VAE load": 14.0},
    "samples": [
        {
            "reference": {"width": 256, "height": 256, "frames": 22, "steps": 8},
            "phase_seconds": {
                "text encoder": 16.0, "denoise": 8.0, "video VAE load": 14.0
            },
        },
        {
            "reference": {"width": 512, "height": 512, "frames": 56, "steps": 20},
            # text encoder does not depend on the video; denoise and the decoder do.
            "phase_seconds": {
                "text encoder": 16.4, "denoise": 130.0, "video VAE load": 40.0
            },
        },
    ],
}


def test_a_second_sample_separates_fixed_cost_from_work():
    model = ProgressModel(TWO_SAMPLES)
    fixed, variable = model.fit["text encoder"]
    assert variable == 0.0 and 15 < fixed < 17, model.fit["text encoder"]
    assert model.fit["denoise"][1] > 0
    assert model.fit["video VAE load"][1] > 0


def test_with_one_sample_every_phase_still_scales():
    model = ProgressModel(
        {
            "reference": TWO_SAMPLES["reference"],
            "phase_seconds": TWO_SAMPLES["phase_seconds"],
        }
    )
    assert model.fit == {}
    assert dict(model.plan(SPEC))["text encoder"] == 16.0


def test_the_quality_presets_do_not_all_cost_the_same():
    model = ProgressModel(TWO_SAMPLES)
    base = {"prompt": "x", "width": 512, "height": 512, "frames": 107}
    quick = JobSpec(**base, steps=20, dit_layers=40, denoise_reuse=3)
    balanced = JobSpec(
        **base, steps=20, dit_layers=45, denoise_reuse=2, token_reduction=True
    )
    best = JobSpec(**base, steps=50, dit_layers=50, denoise_reuse=1)
    times = [
        sum(seconds for _, seconds in model.plan(spec))
        for spec in (quick, balanced, best)
    ]
    assert times[0] < times[1] < times[2]
    # A preset that skips work must be visibly cheaper, not a rounding away.
    assert times[1] - times[0] > 20


def test_reading_the_model_from_disk_costs_time_per_pass():
    # The shipped weights carry the measured streaming penalty.
    model = ProgressModel(load_weights(Settings().progress_weights_path))
    base = {"prompt": "x", "width": 512, "height": 512, "frames": 107, "steps": 20}
    resident = sum(s for _, s in model.plan(JobSpec(**base)))
    streamed = sum(s for _, s in model.plan(JobSpec(**base, ssd_streaming=True)))
    assert streamed > resident


def test_the_internal_canvas_is_what_the_work_is_measured_on():
    model = ProgressModel(TWO_SAMPLES)
    full = JobSpec(prompt="x", width=512, height=512, frames=107, steps=20)
    smaller = full.model_copy(update={"render_width": 256, "render_height": 256})
    assert sum(s for _, s in model.plan(smaller)) < sum(
        s for _, s in model.plan(full)
    )
