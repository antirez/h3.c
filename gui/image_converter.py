from __future__ import annotations

import hashlib
import subprocess
import sys
import uuid
from collections.abc import Callable
from pathlib import Path

from PySide6.QtGui import (
    QImage,
    QImageIOHandler,
    QImageReader,
    QImageWriter,
    QTransform,
)


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
REFERENCE_IMAGE_FILE_FILTER = "Images (" + " ".join(
    f"*{suffix}" for suffix in REFERENCE_IMAGE_SUFFIXES
) + ")"


def requires_png_conversion(source: Path) -> bool:
    return source.suffix.lower() in CONVERTIBLE_REFERENCE_SUFFIXES


def _run_command(command: tuple[str, ...]) -> None:
    subprocess.run(command, check=True, capture_output=True, text=True)


def _apply_orientation(
    image: QImage,
    transformation: QImageIOHandler.Transformation,
) -> QImage:
    if transformation == QImageIOHandler.Transformation.TransformationNone:
        return image
    if transformation == QImageIOHandler.Transformation.TransformationMirror:
        return image.mirrored(True, False)
    if transformation == QImageIOHandler.Transformation.TransformationFlip:
        return image.mirrored(False, True)
    if transformation == QImageIOHandler.Transformation.TransformationRotate180:
        return image.transformed(QTransform().rotate(180))
    if (
        transformation
        == QImageIOHandler.Transformation.TransformationMirrorAndRotate90
    ):
        image = image.mirrored(True, False)
    elif (
        transformation
        == QImageIOHandler.Transformation.TransformationFlipAndRotate90
    ):
        image = image.mirrored(False, True)
    angle = (
        270
        if transformation == QImageIOHandler.Transformation.TransformationRotate270
        else 90
    )
    return image.transformed(QTransform().rotate(angle))


def convert_reference_image(
    source: Path,
    output_dir: Path,
    *,
    run: CommandRunner = _run_command,
) -> Path:
    """Decode an iPhone photo upright and save a metadata-neutral PNG copy."""
    source = source.expanduser().resolve()
    if not requires_png_conversion(source):
        return source
    if not source.is_file():
        raise ImageConversionError(f"HEIC image not found: {source}")

    output_dir = output_dir.expanduser().resolve()
    source_id = hashlib.sha256(str(source).encode()).hexdigest()[:8]
    destination = output_dir / f"{source.stem}-{source_id}.png"
    temporary = output_dir / f".{destination.stem}-{uuid.uuid4().hex}.tmp.png"
    decoded = output_dir / f".{destination.stem}-{uuid.uuid4().hex}.sips.png"
    command = (
        "/usr/bin/sips",
        "-s",
        "format",
        "png",
        str(source),
        "--out",
        str(decoded),
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        orientation_reader = QImageReader(str(source))
        transformation = orientation_reader.transformation()
        run(command)
        image = QImage(str(decoded))
        if image.isNull():
            raise ImageConversionError(
                "HEIC conversion did not produce a readable PNG image."
            )
        image = _apply_orientation(image, transformation)
        writer = QImageWriter(str(temporary), b"png")
        if not writer.write(image):
            raise ImageConversionError(
                "Could not save the orientation-normalized PNG image: "
                f"{writer.errorString()}"
            )
        signature = temporary.read_bytes()[:8]
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ImageConversionError(
                "HEIC conversion did not produce a valid PNG file."
            )
        temporary.replace(destination)
        return destination
    except ImageConversionError:
        raise
    except (OSError, subprocess.SubprocessError) as error:
        raise ImageConversionError(
            "Could not convert or save the HEIC image."
        ) from error
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            temporary.unlink(missing_ok=True)
            decoded.unlink(missing_ok=True)
        except OSError as cleanup_error:
            if not active_error:
                raise ImageConversionError(
                    "Could not remove the temporary HEIC file."
                ) from cleanup_error
