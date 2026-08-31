"""SSE progress stream and the media endpoints."""

import json
import shutil
import stat
import time

import pytest
from conftest import authed_client

from app.config import Settings

JOB = {"prompt": "a fox", "width": 512, "height": 512, "frames": 22, "steps": 2}

PROGRESS = r"""#!/bin/sh
out=""
while [ $# -gt 0 ]; do
  case "$1" in -o) out="$2"; shift;; esac
  shift
done
printf '\rtext encoder                 50/50   ' >&2
printf '\rdenoise                       1/2    ' >&2
sleep 0.2
printf '\rdenoise                       2/2    \n' >&2
printf 'h3: done\n' >&2
if [ -n "$out" ]; then
  ffmpeg -y -loglevel error -f lavfi -i testsrc=size=64x64:rate=24:duration=1 \
    -pix_fmt yuv420p "$out"
fi
exit 0
"""


def _client(tmp_path, script=PROGRESS):
    binary = tmp_path / "h3"
    binary.write_text(script)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    config = Settings(
        binary=binary, model_dir=tmp_path, data_dir=tmp_path / "data"
    )
    return authed_client(config)


def _parse(stream_text: str) -> list[dict]:
    events = []
    for block in stream_text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def _wait(client, job_id, states, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in states:
            return job
        time.sleep(0.05)
    pytest.fail("job never reached a terminal state")


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="FFmpeg is required for the fixture"
)


@needs_ffmpeg
def test_the_stream_reports_progress_and_closes_on_completion(tmp_path):
    with _client(tmp_path) as client:
        job_id = client.post("/api/jobs", json=JOB).json()["id"]
        with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
            assert response.headers["content-type"].startswith("text/event-stream")
            body = "".join(response.iter_text())
    events = _parse(body)
    assert events[0]["id"] == job_id
    assert events[-1]["state"] == "completed"
    phases = [(e["phase"], e["completed"], e["total"]) for e in events]
    assert ("denoise", 2, 2) in phases


@needs_ffmpeg
def test_streaming_a_finished_job_returns_one_snapshot(tmp_path):
    with _client(tmp_path) as client:
        job_id = client.post("/api/jobs", json=JOB).json()["id"]
        _wait(client, job_id, {"completed", "failed"})
        with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
            events = _parse("".join(response.iter_text()))
    assert len(events) == 1
    assert events[0]["state"] == "completed"


def test_streaming_an_unknown_job_reports_an_error(tmp_path):
    with (
        _client(tmp_path) as client,
        client.stream("GET", "/api/jobs/404/events") as response,
    ):
        body = "".join(response.iter_text())
    assert "unknown job" in body


@needs_ffmpeg
def test_video_poster_and_log_are_served(tmp_path):
    with _client(tmp_path) as client:
        job_id = client.post("/api/jobs", json=JOB).json()["id"]
        _wait(client, job_id, {"completed", "failed"})
        video = client.get(f"/api/jobs/{job_id}/video")
        poster = client.get(f"/api/jobs/{job_id}/poster")
        log = client.get(f"/api/jobs/{job_id}/log")
    assert video.status_code == 200
    assert video.headers["content-type"] == "video/mp4"
    assert poster.status_code == 200
    assert poster.content[:3] == b"\xff\xd8\xff"
    assert "h3: done" in log.text


def test_media_endpoints_are_404_before_the_job_produces_anything(tmp_path):
    with _client(tmp_path, "#!/bin/sh\nexit 1\n") as client:
        job_id = client.post("/api/jobs", json=JOB).json()["id"]
        _wait(client, job_id, {"failed", "completed"})
        assert client.get(f"/api/jobs/{job_id}/video").status_code == 404
        assert client.get(f"/api/jobs/{job_id}/poster").status_code == 404
        assert client.get("/api/jobs/999/log").status_code == 404


@needs_ffmpeg
def test_the_poster_is_built_once_and_reused(tmp_path):
    with _client(tmp_path) as client:
        job_id = client.post("/api/jobs", json=JOB).json()["id"]
        _wait(client, job_id, {"completed", "failed"})
        client.get(f"/api/jobs/{job_id}/poster")
        poster = tmp_path / f"data/jobs/{job_id}/poster.jpg"
        assert poster.is_file()
        stamp = poster.stat().st_mtime_ns
        client.get(f"/api/jobs/{job_id}/poster")
        assert poster.stat().st_mtime_ns == stamp
