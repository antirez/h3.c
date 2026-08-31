"""Serial job queue over the h3 CLI.

One GPU, one job at a time. A worker thread pulls queued jobs, runs `./h3` as
a subprocess and mirrors its stderr progress into SQLite. h3 rewrites the
current progress line with a carriage return, so the reader splits on both
CR and LF instead of iterating over lines.
"""

import contextlib
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .argv import build_argv
from .config import Settings
from .db import Closed, Database
from .jobspec import JobSpec
from .media import latest_preview
from .postprocess import PluginError, run_stage
from .progress import ProgressModel, load_weights

# "denoise                     7/20 "
PROGRESS = re.compile(r"^(?P<phase>\S.*?)\s{2,}(?P<completed>\d+)/(?P<total>\d+)\s*$")

TERMINAL_STATES = {"completed", "failed", "cancelled"}
Listener = Callable[[dict[str, Any]], None]


class JobRunner:
    """Owns the worker thread, the current process and the event listeners."""

    def __init__(self, database: Database, config: Settings) -> None:
        self.db = database
        self.config = config
        self.model = ProgressModel(load_weights(config.progress_weights_path))
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._process: subprocess.Popen[str] | None = None
        # start_new_session makes the child its own group leader, so the group
        # id equals its pid and stays valid even after the leader is reaped.
        self._pgid: int | None = None
        self._current: int | None = None
        # Cancellation is recorded, not inferred: a killed child may still
        # exit with a normal status if it traps the signal.
        self._cancelled: set[int] = set()
        self._listeners: list[Listener] = []
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> None:
        # A previous backend may have died mid-job, but h3 is born with
        # start_new_session and outlives it: before declaring anything,
        # check the recorded pid and stop whatever is still alive (T105).
        for row in self.db.query_all(
            "SELECT id, pid FROM jobs WHERE state = 'running'"
        ):
            pid = row["pid"]
            if pid is not None and _process_alive(pid):
                # The pid can in principle have been reused since the crash;
                # that risk is accepted, because leaving a live writer on a
                # directory the UI may delete is worse.
                _signal_group(pid, signal.SIGKILL)
                error = (
                    "interrupted by a backend restart "
                    "(its process was still running, and was stopped)"
                )
            else:
                error = "interrupted by a backend restart"
            self.db.run(
                "UPDATE jobs SET state='failed', finished_at=datetime('now'), "
                "error=?, pid=NULL WHERE id=?",
                (error, row["id"]),
            )
        self._thread.start()
        self._wake.set()

    def shutdown(self) -> None:
        self._stop.set()
        self._wake.set()
        with self._lock:
            if self._current is not None:
                self._cancelled.add(self._current)
        self.cancel_current()
        self._thread.join(timeout=30)
        # If the worker is still winding down, record the outcome here: the
        # database is about to close and a job must never stay 'running'.
        with self._lock:
            pending = self._current
        if pending is not None:
            with contextlib.suppress(Closed):
                self._finish(pending, "cancelled", error="backend shutting down")

    # ── public API ───────────────────────────────────────────────────────
    def submit(self, spec: JobSpec, owner: int | None = None) -> dict[str, Any]:
        job_id = self.db.run(
            "INSERT INTO jobs (state, prompt, params, owner)"
            " VALUES ('queued', ?, ?, ?)",
            (spec.prompt, spec.model_dump_json(), owner),
        )
        self._wake.set()
        job = self.get(job_id)
        self._emit(job)
        return job

    def get(self, job_id: int) -> dict[str, Any] | None:
        row = self.db.query_one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        return self._decorate(_row(row)) if row else None

    def job_dir(self, job_id: int) -> Path:
        return self.config.data_dir / "jobs" / str(job_id)

    def preview_dir(self, job_id: int) -> Path:
        return self.job_dir(job_id) / "preview"

    def delete(self, job_id: int) -> str | None:
        """Remove a finished job and everything it wrote to disk.

        Only a job in a terminal state can be deleted, which keeps the worker
        from writing into a directory that is being removed. That is a guard,
        not a guarantee: a job the restart sweep declared failed may still have
        a live h3 of its own, because h3 outlives a crash of this service.

        The directory goes first. If it cannot be removed the row stays and the
        video is still listed: a visible remnant that can be deleted again is
        better than gigabytes nothing points at any more. The path is derived
        from the job id, never from anything the client sent.
        """
        job = self.get(job_id)
        if job is None:
            return None
        if job["state"] not in TERMINAL_STATES:
            return "unfinished"
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(self.job_dir(job_id))
        self.db.run("DELETE FROM jobs WHERE id = ?", (job_id,))
        return "deleted"

    def _decorate(self, job: dict[str, Any]) -> dict[str, Any]:
        """Attach the newest preview and the weighted progress estimate."""
        newest = (
            latest_preview(self.preview_dir(job["id"]))
            if job["params"].get("preview")
            else None
        )
        job["preview_step"] = newest[0] if newest else None
        job["elapsed"] = _elapsed(job)
        job["remaining"] = (
            _remaining(job["progress"], job["elapsed"])
            if job["state"] == "running"
            else None
        )
        return job

    def listing(
        self, limit: int = 100, owner: int | None = None
    ) -> list[dict[str, Any]]:
        """Newest first; `owner` filters to one person's takes (R30)."""
        if owner is None:
            rows = self.db.query_all(
                "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)
            )
        else:
            rows = self.db.query_all(
                "SELECT * FROM jobs WHERE owner = ? ORDER BY id DESC LIMIT ?",
                (owner, limit),
            )
        return [self._decorate(_row(row)) for row in rows]

    def cancel(self, job_id: int) -> dict[str, Any] | None:
        job = self.get(job_id)
        if job is None or job["state"] in TERMINAL_STATES:
            return job
        with self._lock:
            # Recorded first: a job claimed but not yet spawned would otherwise
            # slip through and leave an orphan process behind.
            self._cancelled.add(job_id)
            is_current = self._current == job_id and self._process is not None
        if is_current:
            self.cancel_current()
        elif job["state"] == "queued":
            # The request stays recorded: the worker may already be claiming
            # this job, and it checks the set right after spawning.
            self._finish(job_id, "cancelled", error="cancelled before it started")
        return self.get(job_id)

    def cancel_current(self) -> None:
        """Signal the child and return: the worker thread owns its lifecycle."""
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return
        with self._lock:
            pgid = self._pgid
        if pgid is None:
            return
        _signal_group(pgid, signal.SIGTERM)
        killer = threading.Timer(
            self.config.kill_grace, _signal_group, args=(pgid, signal.SIGKILL)
        )
        killer.daemon = True
        killer.start()

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def remove_listener(self, listener: Listener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    # ── worker ───────────────────────────────────────────────────────────
    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            while not self._stop.is_set():
                try:
                    row = self.db.query_one(
                        "SELECT * FROM jobs WHERE state = 'queued' ORDER BY id LIMIT 1"
                    )
                except Closed:
                    return
                if row is None:
                    break
                try:
                    self._run(_row(row))
                except Closed:
                    return

    def _run(self, job: dict[str, Any]) -> None:
        job_id = job["id"]
        spec = JobSpec.model_validate(job["params"])
        directory = self.config.data_dir / "jobs" / str(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "out.mp4"
        preview_dir = directory / "preview" if spec.preview else None
        frames_dir = directory / "frames" if spec.write_frames else None
        for extra in (preview_dir, frames_dir):
            if extra is not None:
                extra.mkdir(parents=True, exist_ok=True)
        argv = build_argv(
            spec,
            self.config.binary,
            self.config.model_dir,
            output,
            frames_dir=frames_dir,
            preview_dir=preview_dir,
        )
        log_path = directory / "job.log"
        self.db.run(
            "UPDATE jobs SET state='running', started_at=datetime('now'), "
            "argv=?, output_path=?, log_path=?, phase=NULL, completed=0, total=0, "
            "progress=0.0 WHERE id=?",
            (json.dumps(argv), str(output), str(log_path), job_id),
        )
        self._emit(self.get(job_id))

        try:
            process = subprocess.Popen(  # noqa: S603 - argv list, no shell
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                start_new_session=True,
            )
        except OSError as error:
            self._finish(job_id, "failed", error=f"cannot start h3: {error}")
            return

        # Recorded for the restart sweep (T105): h3 runs in its own session,
        # so the pid doubles as the group id.
        self.db.run("UPDATE jobs SET pid = ? WHERE id = ?", (process.pid, job_id))

        with self._lock:
            self._process = process
            self._pgid = process.pid
            self._current = job_id
            # A shutdown that lands between the claim and this point must not
            # leave the child running: treat it as a cancellation.
            if self._stop.is_set():
                self._cancelled.add(job_id)
            cancel_pending = job_id in self._cancelled
        if cancel_pending:
            self.cancel_current()

        # h3 spawns FFmpeg, which inherits stderr: a lingering grandchild would
        # keep the pipe open forever, so the reader lives in its own thread and
        # the pipe is closed from this side if it outlives the process.
        tail: list[str] = []
        reader = threading.Thread(
            target=self._pump,
            args=(process, job_id, log_path, tail, spec),
            daemon=True,
        )
        reader.start()
        code = process.wait()
        # Reap anything h3 left behind (FFmpeg, in practice) before waiting on
        # the reader: an orphan would otherwise hold the stderr pipe open.
        _signal_group(process.pid, signal.SIGKILL)
        reader.join(timeout=5)
        if reader.is_alive() and process.stderr is not None:
            with contextlib.suppress(OSError, ValueError):
                process.stderr.close()
            reader.join(timeout=5)
        with self._lock:
            self._process = None
            self._pgid = None
            self._current = None

        with self._lock:
            was_cancelled = job_id in self._cancelled
            self._cancelled.discard(job_id)

        if code == 0 and not was_cancelled:
            try:
                if spec.postprocess and output.is_file():
                    run_stage(self.config, output, spec.postprocess)
            except PluginError as error:
                # The raw video stays where it is: the generation succeeded.
                self._finish(job_id, "failed", error=str(error))
                return
            self._finish(job_id, "completed")
        elif was_cancelled:
            self._finish(job_id, "cancelled", error="cancelled")
        else:
            self._finish(job_id, "failed", error=_reason(tail, code))

    def _pump(
        self,
        process: subprocess.Popen[str],
        job_id: int,
        log_path: Path,
        tail: list[str],
        spec: JobSpec,
    ) -> None:
        """Mirror stderr into the log and turn progress lines into updates."""
        buffer = ""
        if process.stderr is None:
            return
        with log_path.open("w", encoding="utf-8") as log:
            while True:
                try:
                    chunk = process.stderr.read(1)
                except (OSError, ValueError):
                    break
                if not chunk:
                    break
                log.write(chunk)
                if chunk in "\r\n":
                    line, buffer = buffer, ""
                    if line.strip():
                        self._consume(job_id, line, tail, spec)
                    log.flush()
                else:
                    buffer += chunk
        if buffer.strip():
            self._consume(job_id, buffer, tail, spec)

    def _consume(
        self, job_id: int, line: str, tail: list[str], spec: JobSpec
    ) -> None:
        match = PROGRESS.match(line.strip("\r\n"))
        if match:
            phase = match["phase"].strip()
            completed = int(match["completed"])
            total = int(match["total"])
            # max(): an unknown phase reports 0, and the bar must never regress.
            self.db.run(
                "UPDATE jobs SET phase=?, completed=?, total=?, "
                "progress=max(progress, ?) WHERE id=?",
                (
                    phase,
                    completed,
                    total,
                    self.model.fraction(spec, phase, completed, total),
                    job_id,
                ),
            )
            self._emit(self.get(job_id))
            return
        tail.append(line.strip())
        del tail[:-20]

    def _finish(self, job_id: int, state: str, error: str | None = None) -> None:
        # progress is NOT NULL: a cancelled or failed job keeps what it reached.
        if state == "completed":
            self.db.run(
                "UPDATE jobs SET state=?, error=?, finished_at=datetime('now'), "
                "progress=1.0, pid=NULL WHERE id=?",
                (state, error, job_id),
            )
        else:
            self.db.run(
                "UPDATE jobs SET state=?, error=?, finished_at=datetime('now'), "
                "pid=NULL WHERE id=?",
                (state, error, job_id),
            )
        self._emit(self.get(job_id))

    def _emit(self, job: dict[str, Any] | None) -> None:
        if job is None:
            return
        for listener in list(self._listeners):
            try:
                listener(job)
            except Exception:  # noqa: BLE001 - a broken listener must not stop a job
                self.remove_listener(listener)


def _signal_group(pgid: int, number: int) -> None:
    """Signal the whole group, including children the leader left behind."""
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pgid, number)


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # Exists but belongs to someone else: not ours to stop.
        return False
    return True


def _reason(tail: list[str], code: int) -> str:
    for line in reversed(tail):
        if line.startswith("h3:"):
            return line
    return tail[-1] if tail else f"h3 exited with code {code}"


def _remaining(progress: float, elapsed: float | None) -> float | None:
    """Correct the estimate with the pace this run is actually keeping."""
    if not elapsed or progress <= 0.02:
        return None
    return max(elapsed / progress - elapsed, 0.0)


def _elapsed(job: dict[str, Any]) -> float | None:
    started = job.get("started_at")
    if not started:
        return None
    ended = job.get("finished_at")
    start = datetime.strptime(started, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
    stop = (
        datetime.strptime(ended, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
        if ended
        else datetime.now(UTC)
    )
    return max((stop - start).total_seconds(), 0.0)


def _row(row: sqlite3.Row) -> dict[str, Any]:
    job = dict(row)
    job["params"] = json.loads(job["params"]) if job["params"] else {}
    job["argv"] = json.loads(job["argv"]) if job["argv"] else None
    return job
