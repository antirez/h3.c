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
    stage: str
    stage_index: int
    completed: int
    total: int
    percent: float
    eta_seconds: float | None


class ProgressTracker:
    _pattern = re.compile(r"\s*(.*?)\s+(\d+)\s*/\s*(\d+)\s*")
    _third = 100.0 / 3.0
    _phase_bands = {
        "tokenizer": (0, 0.0, 2.0),
        "audio VAE encoder": (0, 2.0, 8.0),
        "video VAE encoder": (0, 2.0, 12.0),
        "Qwen vision": (0, 12.0, 18.0),
        "text encoder": (0, 18.0, 23.0),
        "load transformer core": (0, 23.0, 30.0),
        "precompute AdaLN": (0, 30.0, 31.0),
        "refine text": (0, 31.0, 32.0),
        "preview VAE load": (0, 32.0, _third),
        "denoise enqueue": (1, _third, 2 * _third),
        "denoise": (1, _third, 2 * _third),
        "audio VAE": (2, 2 * _third, 74.0),
        "video VAE load": (2, 74.0, 92.0),
        "FFmpeg": (2, 92.0, 100.0),
    }
    _stage_names = ("Preparation", "Generation", "Decode & export")

    def __init__(self) -> None:
        self._started: float | None = None
        self._overall_percent = 0.0
        self._stage_index = 0

    def consume(self, text: str, *, now: float) -> ProgressUpdate | None:
        match = self._pattern.fullmatch(text.strip("\r\n"))
        if not match:
            return None
        phase = match.group(1).strip()
        completed = int(match.group(2))
        total = int(match.group(3))
        if total <= 0 or completed > total:
            return None
        if self._started is None:
            self._started = now
        stage_index, band_start, band_end = self._band_for(phase)
        self._stage_index = max(self._stage_index, stage_index)
        phase_fraction = completed / total
        mapped_percent = band_start + (band_end - band_start) * phase_fraction
        self._overall_percent = max(self._overall_percent, mapped_percent)
        eta_seconds: float | None = None
        elapsed = now - self._started
        if self._overall_percent >= 100.0:
            eta_seconds = 0.0
        elif elapsed > 0.0 and self._overall_percent > 0.0:
            eta_seconds = elapsed * (100.0 - self._overall_percent) / self._overall_percent
        return ProgressUpdate(
            phase=phase,
            stage=self._stage_names[self._stage_index],
            stage_index=self._stage_index,
            completed=completed,
            total=total,
            percent=self._overall_percent,
            eta_seconds=eta_seconds,
        )

    def _band_for(self, phase: str) -> tuple[int, float, float]:
        known = self._phase_bands.get(phase)
        if known is not None:
            return known
        if "denoise" in phase.lower():
            return (1, self._third, 2 * self._third)
        if self._stage_index >= 1:
            return (2, 2 * self._third, 92.0)
        return (0, 0.0, self._third)


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
                raise RuntimeError("a generation is already running")
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
