from __future__ import annotations

import hashlib
import subprocess
import sys
import uuid
from collections.abc import Callable
from pathlib import Path


class ImageConversionError(RuntimeError):
    """A reference image could not be converted into a format H3 accepts."""


CommandRunner = Callable[[tuple[str, ...]], None]
CONVERTIBLE_REFERENCE_SUFFIXES = (".heic", ".heif")
REFERENCE_IMAGE_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    *CONVERTIBLE_REFERENCE_SUFFIXES,
    ".webp",
)
REFERENCE_IMAGE_FILE_FILTER = "Immagini (" + " ".join(
    f"*{suffix}" for suffix in REFERENCE_IMAGE_SUFFIXES
) + ")"


def requires_png_conversion(source: Path) -> bool:
    return source.suffix.lower() in CONVERTIBLE_REFERENCE_SUFFIXES


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
    if not requires_png_conversion(source):
        return source
    if not source.is_file():
        raise ImageConversionError(f"Immagine HEIC non trovata: {source}")

    output_dir = output_dir.expanduser().resolve()
    source_id = hashlib.sha256(str(source).encode()).hexdigest()[:8]
    destination = output_dir / f"{source.stem}-{source_id}.png"
    temporary = output_dir / f".{destination.stem}-{uuid.uuid4().hex}.tmp.png"
    command = (
        "/usr/bin/sips",
        "-s",
        "format",
        "png",
        str(source),
        "--out",
        str(temporary),
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        run(command)
        signature = temporary.read_bytes()[:8]
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ImageConversionError(
                "La conversione HEIC non ha prodotto un file PNG valido."
            )
        temporary.replace(destination)
        return destination
    except ImageConversionError:
        raise
    except (OSError, subprocess.SubprocessError) as error:
        raise ImageConversionError(
            "Non sono riuscito a convertire o salvare l’immagine HEIC."
        ) from error
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            temporary.unlink(missing_ok=True)
        except OSError as cleanup_error:
            if not active_error:
                raise ImageConversionError(
                    "Non sono riuscito a rimuovere il file temporaneo HEIC."
                ) from cleanup_error
