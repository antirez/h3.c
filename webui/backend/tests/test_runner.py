"""Queue and runner, driven by a stand-in for the h3 binary.

The fake reproduces what matters about h3: progress written to stderr with a
carriage return, an mp4 written where -o points, and h3-prefixed error lines.
No GPU and no checkpoint are involved.
"""

import os
import stat
import time

import pytest
from conftest import authed_client

from app.config import Settings

JOB = {"prompt": "a fox", "width": 512, "height": 512, "frames": 22, "steps": 2}

SUCCESS = r"""#!/bin/sh
out=""
while [ $# -gt 0 ]; do
  case "$1" in -o) out="$2"; shift;; esac
  shift
done
printf 'h3: starting\n' >&2
printf '\rtext encoder                 %d/50   ' 50 >&2
printf '\rdenoise                       1/2    ' >&2
printf '\rdenoise                       2/2    ' >&2
printf '\n' >&2
[ -n "$out" ] && printf 'fake mp4' > "$out"
exit 0
"""

FAILURE = r"""#!/bin/sh
printf 'h3: canvas exceeds the released 768*1344 pixel limit\n' >&2
exit 1
"""

SLOW = r"""#!/bin/sh
trap 'exit 143' TERM
printf '\rdenoise                       1/900  ' >&2
sleep 30 &
wait $!
"""


def _binary(tmp_path, script):
    path = tmp_path / "h3"
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _client(tmp_path, script):
    config = Settings(
        binary=_binary(tmp_path, script),
        model_dir=tmp_path,
        data_dir=tmp_path / "data",
    )
    return authed_client(config)


def _wait(client, job_id, states, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in states:
            return job
        time.sleep(0.05)
    pytest.fail(f"job stayed in {job['state']}")


def _wait_pid(client, job_id, timeout=20.0):
    """The pid is recorded in the write right after the spawn (T105), so a
    job can legitimately be 'running' with the pid still absent: wait it
    out. The invariant the restart sweep relies on is that a running job
    ends up with its process recorded, not that both flip in one write."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pid = client.app.state.db.query_one(
            "SELECT pid FROM jobs WHERE id = ?", (job_id,)
        )["pid"]
        if pid is not None:
            return pid
        time.sleep(0.05)
    pytest.fail("the running job never recorded its pid")


def test_a_job_runs_to_completion_and_keeps_its_artifacts(tmp_path):
    with _client(tmp_path, SUCCESS) as client:
        created = client.post("/api/jobs", json=JOB)
        assert created.status_code == 201
        job = _wait(client, created.json()["id"], {"completed", "failed"})
    assert job["state"] == "completed"
    assert job["progress"] == 1.0
    assert (tmp_path / "data/jobs/1/out.mp4").read_text() == "fake mp4"
    assert "h3: starting" in (tmp_path / "data/jobs/1/job.log").read_text()


def test_progress_is_parsed_from_carriage_return_updates(tmp_path):
    seen = []
    with _client(tmp_path, SUCCESS) as client:
        client.app.state.runner.add_listener(
            lambda job: seen.append((job["phase"], job["completed"], job["total"]))
        )
        created = client.post("/api/jobs", json=JOB)
        _wait(client, created.json()["id"], {"completed", "failed"})
    assert ("text encoder", 50, 50) in seen
    assert ("denoise", 1, 2) in seen
    assert ("denoise", 2, 2) in seen


def test_a_failing_run_keeps_the_h3_error_line(tmp_path):
    with _client(tmp_path, FAILURE) as client:
        created = client.post("/api/jobs", json=JOB)
        job = _wait(client, created.json()["id"], {"completed", "failed"})
    assert job["state"] == "failed"
    assert job["error"] == "h3: canvas exceeds the released 768*1344 pixel limit"


def test_a_running_job_can_be_cancelled(tmp_path):
    with _client(tmp_path, SLOW) as client:
        created = client.post("/api/jobs", json=JOB)
        job_id = created.json()["id"]
        _wait(client, job_id, {"running"})
        cancelled = client.post(f"/api/jobs/{job_id}/cancel").json()
        assert cancelled["state"] in {"cancelled", "running"}
        job = _wait(client, job_id, {"cancelled", "failed", "completed"})
    assert job["state"] == "cancelled"


def test_a_queued_job_can_be_cancelled_before_it_starts(tmp_path):
    with _client(tmp_path, SLOW) as client:
        first = client.post("/api/jobs", json=JOB).json()["id"]
        second = client.post("/api/jobs", json=JOB).json()["id"]
        _wait(client, first, {"running"})
        assert client.get(f"/api/jobs/{second}").json()["state"] == "queued"
        cancelled = client.post(f"/api/jobs/{second}/cancel").json()
        assert cancelled["state"] == "cancelled"
        client.post(f"/api/jobs/{first}/cancel")
        _wait(client, first, {"cancelled", "failed", "completed"})


def test_jobs_run_one_at_a_time(tmp_path):
    with _client(tmp_path, SLOW) as client:
        ids = [client.post("/api/jobs", json=JOB).json()["id"] for _ in range(3)]
        _wait(client, ids[0], {"running"})
        states = [client.get(f"/api/jobs/{i}").json()["state"] for i in ids]
        assert states.count("running") == 1
        assert states.count("queued") == 2
        for job_id in ids:
            client.post(f"/api/jobs/{job_id}/cancel")
        for job_id in ids:
            _wait(client, job_id, {"cancelled", "failed", "completed"})


def test_an_invalid_job_is_refused_before_it_is_queued(tmp_path):
    with _client(tmp_path, SUCCESS) as client:
        response = client.post("/api/jobs", json={**JOB, "width": 500})
        assert response.status_code == 422
        assert (
            "width and height must be multiples of 32 and at least 32"
            in response.json()["detail"]["errors"]
        )
        assert client.get("/api/jobs").json() == []


def test_validate_endpoint_reports_the_resolved_duration(tmp_path):
    with _client(tmp_path, SUCCESS) as client:
        body = client.post(
            "/api/jobs/validate", json={**JOB, "frames": None, "seconds": 10.0}
        ).json()
    assert body["errors"] == []
    assert body["frames"] == 243
    assert body["seconds"] == 10.125


def test_the_recorded_argv_is_the_command_that_ran(tmp_path):
    with _client(tmp_path, SUCCESS) as client:
        created = client.post("/api/jobs", json=JOB)
        job = _wait(client, created.json()["id"], {"completed", "failed"})
    assert job["argv"][1:3] == ["-d", str(tmp_path)]
    assert job["argv"][job["argv"].index("--frames") + 1] == "22"


def test_missing_binary_fails_the_job_with_a_reason(tmp_path):
    config = Settings(
        binary=tmp_path / "absent", model_dir=tmp_path, data_dir=tmp_path / "data"
    )
    with authed_client(config) as client:
        created = client.post("/api/jobs", json=JOB)
        job = _wait(client, created.json()["id"], {"failed", "completed"})
    assert job["state"] == "failed"
    assert "cannot start h3" in job["error"]


def test_unknown_job_is_a_404(tmp_path):
    with _client(tmp_path, SUCCESS) as client:
        assert client.get("/api/jobs/404").status_code == 404


def test_a_job_interrupted_by_a_restart_is_marked_failed(tmp_path):
    config = Settings(
        binary=_binary(tmp_path, SUCCESS),
        model_dir=tmp_path,
        data_dir=tmp_path / "data",
    )
    with authed_client(config) as client:
        created = client.post("/api/jobs", json=JOB)
        _wait(client, created.json()["id"], {"completed", "failed"})
        # Simulate a backend killed while a job was running.
        client.app.state.db.run("UPDATE jobs SET state='running' WHERE id=1")
    with authed_client(config) as client:
        job = client.get("/api/jobs/1").json()
    assert job["state"] == "failed"
    assert job["error"] == "interrupted by a backend restart"


def test_shutdown_cancels_the_running_job_instead_of_failing_it(tmp_path):
    config = Settings(
        binary=_binary(tmp_path, SLOW),
        model_dir=tmp_path,
        data_dir=tmp_path / "data",
    )
    with authed_client(config) as client:
        job_id = client.post("/api/jobs", json=JOB).json()["id"]
        _wait(client, job_id, {"running"})
    with authed_client(config) as client:
        job = client.get(f"/api/jobs/{job_id}").json()
    assert job["state"] == "cancelled"


def test_a_finished_job_can_be_deleted_with_everything_it_wrote(tmp_path):
    with _client(tmp_path, SUCCESS) as client:
        job_id = client.post("/api/jobs", json=JOB).json()["id"]
        _wait(client, job_id, {"completed", "failed"})
        directory = tmp_path / "data/jobs" / str(job_id)
        assert (directory / "out.mp4").is_file()

        assert client.delete(f"/api/jobs/{job_id}").status_code == 204

        assert not directory.exists()
        assert client.get(f"/api/jobs/{job_id}").status_code == 404
        assert client.get(f"/api/jobs/{job_id}/video").status_code == 404
        assert client.get(f"/api/jobs/{job_id}/log").status_code == 404
        assert client.get("/api/jobs").json() == []


def test_deleting_a_running_job_is_refused_and_keeps_its_files(tmp_path):
    with _client(tmp_path, SLOW) as client:
        job_id = client.post("/api/jobs", json=JOB).json()["id"]
        _wait(client, job_id, {"running"})

        # The row flips to running before h3 is even spawned, so the job's
        # files are compared against themselves rather than against a name
        # that may not have been written yet.
        directory = tmp_path / "data/jobs" / str(job_id)
        before = {entry.name for entry in directory.iterdir()}

        refused = client.delete(f"/api/jobs/{job_id}")
        assert refused.status_code == 409
        assert refused.json()["detail"] == "stop this video before deleting it"
        assert directory.is_dir()
        assert before <= {entry.name for entry in directory.iterdir()}
        assert client.get(f"/api/jobs/{job_id}").json()["state"] == "running"

        client.post(f"/api/jobs/{job_id}/cancel")
        _wait(client, job_id, {"cancelled", "failed", "completed"})


def test_a_queued_job_is_removed_by_stopping_it_not_by_deleting_it(tmp_path):
    with _client(tmp_path, SLOW) as client:
        running = client.post("/api/jobs", json=JOB).json()["id"]
        queued = client.post("/api/jobs", json=JOB).json()["id"]
        _wait(client, running, {"running"})
        assert client.get(f"/api/jobs/{queued}").json()["state"] == "queued"

        assert client.delete(f"/api/jobs/{queued}").status_code == 409

        client.post(f"/api/jobs/{queued}/cancel")
        _wait(client, queued, {"cancelled", "failed", "completed"})
        assert client.delete(f"/api/jobs/{queued}").status_code == 204
        client.post(f"/api/jobs/{running}/cancel")
        _wait(client, running, {"cancelled", "failed", "completed"})


def test_deleting_an_unknown_job_is_a_404(tmp_path):
    with _client(tmp_path, SUCCESS) as client:
        assert client.delete("/api/jobs/404").status_code == 404


def test_the_event_stream_of_a_deleted_job_ends_instead_of_hanging(tmp_path):
    with _client(tmp_path, SUCCESS) as client:
        job_id = client.post("/api/jobs", json=JOB).json()["id"]
        _wait(client, job_id, {"completed", "failed"})
        client.delete(f"/api/jobs/{job_id}")

        # R30: an unknown job is a 404 on every endpoint, the stream too —
        # a foreign id and a deleted one are indistinguishable on purpose.
        assert client.get(f"/api/jobs/{job_id}/events").status_code == 404


def test_a_running_job_records_its_process(tmp_path):
    with _client(tmp_path, SLOW) as client:
        job_id = client.post("/api/jobs", json=JOB).json()["id"]
        _wait(client, job_id, {"running"})
        pid = _wait_pid(client, job_id)
        assert pid > 0
        client.post(f"/api/jobs/{job_id}/cancel")
        _wait(client, job_id, {"cancelled", "failed"})


def test_a_restart_stops_a_live_h3_before_failing_its_job(tmp_path):
    config = Settings(
        binary=_binary(tmp_path, SLOW),
        model_dir=tmp_path,
        data_dir=tmp_path / "data",
    )
    with authed_client(config) as client:
        job_id = client.post("/api/jobs", json=JOB).json()["id"]
        _wait(client, job_id, {"running"})
        pid = _wait_pid(client, job_id)

        # A crash, not a shutdown: the first backend gets no graceful stop,
        # and a second startup sweeps over the job while h3 is still alive.
        second_config = Settings(
            binary=tmp_path / "h3", model_dir=tmp_path, data_dir=tmp_path / "data"
        )
        with authed_client(second_config) as second:
            job = second.get(f"/api/jobs/{job_id}").json()
            assert job["state"] == "failed"

    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the live h3 survived the restart sweep")


def test_the_restart_sweep_names_a_live_process_it_stopped(tmp_path):
    import subprocess

    from app.db import Database

    config = Settings(
        binary=tmp_path / "absent", model_dir=tmp_path, data_dir=tmp_path / "data"
    )
    child = subprocess.Popen(
        ["sleep", "30"], stdout=subprocess.DEVNULL, start_new_session=True
    )
    database = Database(config.data_dir / "h3.sqlite3")
    database.run(
        "INSERT INTO jobs (state, prompt, params, pid)"
        " VALUES ('running', 'x', '{}', ?)",
        (child.pid,),
    )
    database.close()

    with authed_client(config) as client:
        job = client.get("/api/jobs/1").json()
        assert job["state"] == "failed"
        assert "still running" in (job["error"] or "")

    deadline = time.time() + 10
    while time.time() < deadline:
        if child.poll() is not None:
            break
        time.sleep(0.05)
    assert child.poll() is not None, "the process survived the sweep"


def test_the_restart_sweep_lets_a_dead_process_go(tmp_path):
    import subprocess

    from app.db import Database

    config = Settings(
        binary=tmp_path / "absent", model_dir=tmp_path, data_dir=tmp_path / "data"
    )
    child = subprocess.Popen(["true"], stdout=subprocess.DEVNULL)
    child.wait()
    database = Database(config.data_dir / "h3.sqlite3")
    database.run(
        "INSERT INTO jobs (state, prompt, params, pid)"
        " VALUES ('running', 'x', '{}', ?)",
        (child.pid,),
    )
    database.close()

    with authed_client(config) as client:
        job = client.get("/api/jobs/1").json()
        assert job["state"] == "failed"
        assert job["error"] == "interrupted by a backend restart"
