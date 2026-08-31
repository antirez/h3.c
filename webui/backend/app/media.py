"""Small FFmpeg helpers for the gallery."""

import re
import subprocess
from pathlib import Path

from .config import Settings

_STEP = re.compile(r"^step-(\d+)\.ppm$")


def extract_poster(video: Path, poster: Path, config: Settings) -> bool:
    """Grab the first frame as a JPEG thumbnail. Best effort."""
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                config.ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-frames:v",
                "1",
                str(poster),
            ],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0 and poster.is_file()


def latest_preview(directory: Path) -> tuple[int, Path] | None:
    """Newest complete denoising preview, or None if there is not one yet.

    h3 writes to a staging name and renames, so every step-*.ppm here is whole.
    """
    if not directory.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for entry in directory.iterdir():
        match = _STEP.match(entry.name)
        if match and (best is None or int(match.group(1)) > best[0]):
            best = (int(match.group(1)), entry)
    return best


def preview_jpeg(directory: Path, config: Settings) -> Path | None:
    """Convert the newest preview to JPEG once, then reuse it."""
    newest = latest_preview(directory)
    if newest is None:
        return None
    step, source = newest
    target = directory / f"step-{step:04d}.jpg"
    if target.is_file():
        return target
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [config.ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
             "-q:v", "3", str(target)],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return target if done.returncode == 0 and target.is_file() else None
