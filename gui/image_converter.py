from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path


class ImageConversionError(RuntimeError):
    """A reference image could not be converted into a format H3 accepts."""


CommandRunner = Callable[[tuple[str, ...]], None]


def _run_command(command: tuple[str, ...]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def convert_reference_image(
    source: Path,
    output_dir: Path,
    *,
    run: CommandRunner = _run_command,
) -> Path:
    """Convert an iPhone HEIC/HEIF reference to PNG, preserving the original."""
    source = source.expanduser().resolve()
    if source.suffix.lower() not in (".heic", ".heif"):
        return source
    if not source.is_file():
        raise ImageConversionError(f"Immagine HEIC non trovata: {source}")

    output_dir = output_dir.expanduser().resolve()
    destination = output_dir / f"{source.stem}.png"
    command = (
        "/usr/bin/sips",
        "-s",
        "format",
        "png",
        str(source),
        "--out",
        str(destination),
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        run(command)
        signature = destination.read_bytes()[:8]
    except (OSError, subprocess.SubprocessError) as error:
        raise ImageConversionError(
            "Non sono riuscito a convertire l’immagine HEIC con sips."
        ) from error
    if signature != b"\x89PNG\r\n\x1a\n":
        raise ImageConversionError(
            "La conversione HEIC non ha prodotto un file PNG valido."
        )
    return destination
