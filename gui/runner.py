from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    model_dir: Path
    prompt: str
    output_path: Path
    reference_image: Path | None
    width: int
    height: int
    render_width: int | None
    render_height: int | None
    seconds: int
    steps: int
    layers: int
    reuse: int
    core_reuse: int
    seed: int
    ssd_streaming: bool
    live_preview: bool
    preview_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    phase: str
    completed: int
    total: int
    percent: float
    eta_seconds: float | None


class ProgressTracker:
    _pattern = re.compile(r"\s*(.*?)\s+(\d+)\s*/\s*(\d+)\s*")

    def __init__(self) -> None:
        self._phase: str | None = None
        self._phase_started = 0.0
        self._phase_started_at = 0

    def consume(self, text: str, *, now: float) -> ProgressUpdate | None:
        match = self._pattern.fullmatch(text.strip("\r\n"))
        if not match:
            return None
        phase = match.group(1).strip()
        completed = int(match.group(2))
        total = int(match.group(3))
        if total <= 0 or completed > total:
            return None
        if phase != self._phase or completed < self._phase_started_at:
            self._phase = phase
            self._phase_started = now
            self._phase_started_at = completed
        delta = completed - self._phase_started_at
        eta_seconds = None
        if delta > 0:
            seconds_per_unit = (now - self._phase_started) / delta
            eta_seconds = max(0.0, seconds_per_unit * (total - completed))
        return ProgressUpdate(
            phase=phase,
            completed=completed,
            total=total,
            percent=(completed / total) * 100.0,
            eta_seconds=eta_seconds,
        )


@dataclass(frozen=True, slots=True)
class RunResult:
    exit_code: int
    cancelled: bool
    output_path: Path


@dataclass(frozen=True, slots=True)
class RunnerCallbacks:
    on_progress: Callable[[ProgressUpdate], Any]
    on_output: Callable[[str], Any]
    on_preview: Callable[[Path], Any]
    on_finished: Callable[[RunResult], Any]


def _spawn_process(command: list[str], cwd: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,
    )


class H3Runner:
    def __init__(
        self,
        executable: Path,
        *,
        process_factory: Callable[[list[str], Path], Any] = _spawn_process,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._executable = executable
        self._process_factory = process_factory
        self._clock = clock
        self._process: Any | None = None
        self._thread: threading.Thread | None = None
        self._cancel_requested = False
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        with self._lock:
            return self._process is not None

    def start(
        self,
        settings: GenerationSettings,
        callbacks: RunnerCallbacks,
    ) -> None:
        with self._lock:
            if self._process is not None:
                raise RuntimeError("una generazione è già in corso")
            settings.output_path.parent.mkdir(parents=True, exist_ok=True)
            if settings.preview_dir is not None:
                settings.preview_dir.mkdir(parents=True, exist_ok=True)
            command = build_h3_command(self._executable, settings)
            process = self._process_factory(command, self._executable.parent)
            self._process = process
            self._cancel_requested = False
        self._thread = threading.Thread(
            target=self._consume_process,
            args=(process, settings, callbacks),
            name="h3-runner",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, force_after: float = 8.0) -> None:
        with self._lock:
            process = self._process
            if process is None:
                return
            self._cancel_requested = True
        process.terminate()
        timer = threading.Timer(force_after, self._kill_if_running, args=(process,))
        timer.daemon = True
        timer.start()

    def _kill_if_running(self, process: Any) -> None:
        if process.poll() is None:
            process.kill()

    def _consume_process(
        self,
        process: Any,
        settings: GenerationSettings,
        callbacks: RunnerCallbacks,
    ) -> None:
        tracker = ProgressTracker()
        buffer = ""
        stream = process.stdout
        if stream is not None:
            while True:
                character = stream.read(1)
                if character == "":
                    if buffer:
                        self._consume_line(buffer, tracker, callbacks)
                    break
                if character in "\r\n":
                    if buffer:
                        self._consume_line(buffer, tracker, callbacks)
                        buffer = ""
                else:
                    buffer += character
        exit_code = process.wait()
        with self._lock:
            cancelled = self._cancel_requested
            if self._process is process:
                self._process = None
        callbacks.on_finished(
            RunResult(
                exit_code=exit_code,
                cancelled=cancelled,
                output_path=settings.output_path,
            )
        )

    def _consume_line(
        self,
        line: str,
        tracker: ProgressTracker,
        callbacks: RunnerCallbacks,
    ) -> None:
        callbacks.on_output(line)
        update = tracker.consume(line, now=self._clock())
        if update is not None:
            callbacks.on_progress(update)
        preview_prefix = "h3: preview-file "
        if line.startswith(preview_prefix):
            callbacks.on_preview(Path(line[len(preview_prefix) :]))


def build_h3_command(executable: Path, settings: GenerationSettings) -> list[str]:
    command = [str(executable), "--profile", "-d", str(settings.model_dir)]
    if settings.reference_image is not None:
        command.extend(
            (
                "--ref-image",
                str(settings.reference_image),
                "--ref-image-size",
                "max",
            )
        )
    command.extend(("-p", settings.prompt))
    command.extend(("--width", str(settings.width), "--height", str(settings.height)))
    if settings.render_width is not None and settings.render_height is not None:
        command.extend(
            (
                "--render-width",
                str(settings.render_width),
                "--render-height",
                str(settings.render_height),
            )
        )
    command.extend(
        (
            "--seconds",
            str(settings.seconds),
            "--steps",
            str(settings.steps),
            "--layers",
            str(settings.layers),
            "--reuse",
            str(settings.reuse),
        )
    )
    if settings.core_reuse > 1:
        command.extend(("--core-reuse", str(settings.core_reuse)))
    command.extend(("--seed", str(settings.seed)))
    if settings.ssd_streaming:
        command.append("--ssd-streaming")
    if settings.live_preview and settings.preview_dir is not None:
        command.extend(("--preview-dir", str(settings.preview_dir)))
    command.extend(("-o", str(settings.output_path)))
    return command
