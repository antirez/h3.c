"""Live preview: the newest complete step reaches the browser as a JPEG."""

import json
import shutil
import stat
import time

import pytest
from conftest import authed_client

from app.config import Settings

JOB = {
    "prompt": "a fox",
    "width": 64,
    "height": 64,
    "frames": 22,
    "steps": 3,
    "preview": True,
}

# Writes one PPM per step into --preview-dir, the way h3 does.
PREVIEWS = r"""#!/bin/sh
dir=""
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --preview-dir) dir="$2"; shift;;
    -o) out="$2"; shift;;
  esac
  shift
done
step=0
while [ $step -lt 3 ]; do
  printf '\rdenoise                       %d/3    ' "$step" >&2
  ffmpeg -y -loglevel error -f lavfi -i "color=c=blue:s=64x64:d=1" \
    -frames:v 1 "$dir/.step.ppm"
  mv "$dir/.step.ppm" "$(printf '%s/step-%04d.ppm' "$dir" "$step")"
  step=$((step + 1))
  sleep 0.15
done
printf '\rdenoise                       3/3    \n' >&2
[ -n "$out" ] && printf 'mp4' > "$out"
exit 0
"""

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="FFmpeg is required for the fixture"
)


def _client(tmp_path):
    binary = tmp_path / "h3"
    binary.write_text(PREVIEWS)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    config = Settings(binary=binary, model_dir=tmp_path, data_dir=tmp_path / "data")
    return authed_client(config)


def _wait(client, job_id, states, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in states:
            return job
        time.sleep(0.05)
    pytest.fail("job never reached a terminal state")


def test_the_newest_preview_is_served_as_a_jpeg(tmp_path):
    with _client(tmp_path) as client:
        job_id = client.post("/api/jobs", json=JOB).json()["id"]
        job = _wait(client, job_id, {"completed", "failed"})
        assert job["state"] == "completed"
        assert job["preview_step"] == 2
        response = client.get(f"/api/jobs/{job_id}/preview")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:3] == b"\xff\xd8\xff"


def test_the_stream_carries_the_preview_step(tmp_path):
    with _client(tmp_path) as client:
        job_id = client.post("/api/jobs", json=JOB).json()["id"]
        with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
            body = "".join(response.iter_text())
    steps = [
        json.loads(line[6:])["preview_step"]
        for line in body.splitlines()
        if line.startswith("data: ")
    ]
    assert steps[0] is None
    assert max(step for step in steps if step is not None) == 2


def test_no_preview_before_the_first_step(tmp_path):
    with _client(tmp_path) as client:
        response = client.post("/api/jobs", json={**JOB, "preview": False})
        job_id = response.json()["id"]
        assert client.get(f"/api/jobs/{job_id}/preview").status_code == 404
        _wait(client, job_id, {"completed", "failed"})
        assert client.get(f"/api/jobs/{job_id}").json()["preview_step"] is None


def test_a_partial_file_is_never_served(tmp_path):
    """Only renamed step-*.ppm files are considered, never the staging name."""
    from app.media import latest_preview

    directory = tmp_path / "preview"
    directory.mkdir()
    (directory / ".step.ppm").write_bytes(b"partial")
    assert latest_preview(directory) is None
    (directory / "step-0007.ppm").write_bytes(b"whole")
    assert latest_preview(directory)[0] == 7
