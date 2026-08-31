"""Time estimates: one request labels every choice on screen."""

import pytest
from conftest import authed_client

from app.config import Settings

SPEC = {
    "prompt": "a fox",
    "width": 512,
    "height": 512,
    "frames": 22,
    "steps": 20,
}


@pytest.fixture
def client(tmp_path):
    config = Settings(model_dir=tmp_path, data_dir=tmp_path / "data")
    with authed_client(config) as client:
        yield client


def estimate(client, **body):
    return client.post("/api/jobs/estimate", json=body).json()


def test_a_job_is_estimated_in_seconds(client):
    body = estimate(client, spec=SPEC)
    assert body["seconds"] > 0
    assert body["variants"] == []


def test_more_detail_passes_take_longer(client):
    body = estimate(
        client,
        spec=SPEC,
        variants=[{"steps": 4}, {"steps": 20}, {"steps": 50}],
    )
    times = [variant["seconds"] for variant in body["variants"]]
    assert times == sorted(times)
    assert times[0] < times[-1]


def test_a_bigger_picture_takes_longer(client):
    body = estimate(
        client,
        spec=SPEC,
        variants=[
            {"width": 256, "height": 256},
            {"width": 512, "height": 512},
            {"width": 768, "height": 768},
        ],
    )
    times = [variant["seconds"] for variant in body["variants"]]
    assert times == sorted(times)


def test_a_longer_video_takes_longer(client):
    body = estimate(
        client, spec=SPEC, variants=[{"frames": 22}, {"frames": 107}, {"frames": 243}]
    )
    times = [variant["seconds"] for variant in body["variants"]]
    assert times == sorted(times)


def test_working_smaller_is_faster_than_the_output_size(client):
    body = estimate(
        client,
        spec={**SPEC, "width": 512, "height": 512},
        variants=[{"render_width": 256, "render_height": 256}],
    )
    assert body["variants"][0]["seconds"] < body["seconds"]


def test_the_preview_costs_something_but_not_much(client):
    body = estimate(client, spec=SPEC, variants=[{"preview": True}])
    with_preview = body["variants"][0]["seconds"]
    assert with_preview > body["seconds"]
    assert with_preview < body["seconds"] * 1.5


def test_an_impossible_variant_is_reported_not_guessed(client):
    body = estimate(client, spec=SPEC, variants=[{"steps": "many"}])
    assert "seconds" not in body["variants"][0]
    assert body["variants"][0]["error"] >= 1


def test_validation_carries_the_estimate_so_one_call_is_enough(client):
    body = client.post("/api/jobs/validate", json=SPEC).json()
    assert body["estimate_seconds"] > 0
    assert body["frames"] == 22


# ── the estimate learns from what this machine actually did ─────────────────

def _finished_job(client, spec, seconds):
    """Insert a completed job that took a known amount of time."""
    import json

    db = client.app.state.db
    db.run(
        "INSERT INTO jobs (state, prompt, params, started_at, finished_at, progress) "
        "VALUES ('completed', 'x', ?, datetime('now'), "
        "datetime('now', ?), 1.0)",
        (json.dumps(spec), f"+{seconds} seconds"),
    )


def test_with_no_history_the_estimate_is_the_model_alone(client):
    body = estimate(client, spec=SPEC)
    assert body["learned_from"] == 0


def test_jobs_that_took_longer_push_the_estimate_up(client):
    before = estimate(client, spec=SPEC)["seconds"]
    for _ in range(3):
        _finished_job(client, SPEC, int(before * 2))
    after = estimate(client, spec=SPEC)
    assert after["learned_from"] == 3
    assert 1.6 * before < after["seconds"] < 2.4 * before


def test_one_job_is_not_enough_to_learn_from(client):
    before = estimate(client, spec=SPEC)["seconds"]
    _finished_job(client, SPEC, 10_000)
    after = estimate(client, spec=SPEC)
    assert after["learned_from"] == 1
    assert after["seconds"] == before


def test_a_wild_history_cannot_pull_the_estimate_beyond_reason(client):
    plain = estimate(client, spec=SPEC)["seconds"]
    for _ in range(4):
        _finished_job(client, SPEC, 100_000)
    assert estimate(client, spec=SPEC)["seconds"] <= plain * 4.0 + 1


def test_failed_and_cancelled_jobs_teach_nothing(client):
    import json

    before = estimate(client, spec=SPEC)["seconds"]
    for state in ("failed", "cancelled", "failed"):
        client.app.state.db.run(
            "INSERT INTO jobs (state, prompt, params, started_at, finished_at) "
            "VALUES (?, 'x', ?, datetime('now'), datetime('now', '+9000 seconds'))",
            (state, json.dumps(SPEC)),
        )
    assert estimate(client, spec=SPEC)["seconds"] == before
