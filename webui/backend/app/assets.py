"""Uploaded images, clips and soundtracks.

Files are stored by content hash, so re-uploading the same photo reuses the
existing entry and the library stays free of duplicates. Extensions are
whitelisted and every file is probed with ffprobe: what the browser calls a
PNG is only accepted if ffprobe agrees.
"""

import hashlib
import json
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .db import Database

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
ALLOWED_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES | AUDIO_SUFFIXES

# h3 accepts reference audio between 2 and 15 seconds.
MIN_AUDIO_SECONDS = 2.0
MAX_AUDIO_SECONDS = 15.0


class AssetError(ValueError):
    """The upload cannot be stored, with a reason meant for the user."""


@dataclass
class Probe:
    kind: str
    seconds: float | None
    width: int | None
    height: int | None
    has_audio: bool


def kind_from_suffix(suffix: str) -> str:
    lowered = suffix.lower()
    if lowered in IMAGE_SUFFIXES:
        return "image"
    if lowered in VIDEO_SUFFIXES:
        return "video"
    if lowered in AUDIO_SUFFIXES:
        return "audio"
    raise AssetError(f"unsupported file type: {suffix or 'no extension'}")


def probe(path: Path, ffprobe: str = "ffprobe") -> Probe:
    """Ask ffprobe what this file really is."""
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "stream=codec_type,width,height:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise AssetError(f"cannot probe the upload: {error}") from error
    if done.returncode != 0:
        raise AssetError("the file is not a readable image, video or audio track")
    report = json.loads(done.stdout or "{}")
    streams = report.get("streams", [])
    if not streams:
        raise AssetError("the file has no decodable stream")
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    duration = report.get("format", {}).get("duration")
    seconds = float(duration) if duration not in (None, "N/A") else None
    if video is None:
        kind = "audio"
    elif seconds is None or seconds == 0 or _is_still(seconds, video):
        kind = "image"
    else:
        kind = "video"
    return Probe(
        kind=kind,
        seconds=seconds,
        width=video.get("width") if video else None,
        height=video.get("height") if video else None,
        has_audio=has_audio,
    )


def _is_still(seconds: float, video: dict[str, Any]) -> bool:
    # ffprobe reports a tiny synthetic duration for single-frame images.
    return seconds < 0.1 and not video.get("nb_frames", "").isdigit()


def store(
    database: Database,
    source: Path,
    filename: str,
    root: Path,
    max_bytes: int,
    ffprobe: str = "ffprobe",
    owner: int | None = None,
) -> dict[str, Any]:
    """Validate, deduplicate and record one upload.

    Deduplication is per owner (R30): the same bytes may be held by two
    people, each seeing it in their own library, while the file on disk —
    named by its hash — stays shared.
    """
    declared = kind_from_suffix(Path(filename).suffix)
    size = source.stat().st_size
    if size == 0:
        raise AssetError("the upload is empty")
    if size > max_bytes:
        raise AssetError(
            f"the upload is {size / 1e6:.1f} MB, over the "
            f"{max_bytes / 1e6:.0f} MB limit"
        )
    detected = probe(source, ffprobe)
    if detected.kind != declared:
        raise AssetError(
            f"the extension says {declared} but the file is {detected.kind}"
        )

    digest = _sha256(source)
    existing = database.query_one(
        "SELECT * FROM assets WHERE sha256 = ? AND owner IS ?", (digest, owner)
    )
    if existing:
        return _row_to_dict(existing) | {"duplicate": True}

    target = root / digest[:2] / f"{digest}{Path(filename).suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    metadata = {
        "seconds": detected.seconds,
        "width": detected.width,
        "height": detected.height,
        "has_audio": detected.has_audio,
        "notes": _notes(detected),
    }
    asset_id = database.run(
        "INSERT INTO assets (sha256, kind, filename, path, bytes, metadata, owner) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            digest,
            detected.kind,
            filename,
            str(target),
            size,
            json.dumps(metadata),
            owner,
        ),
    )
    row = database.query_one("SELECT * FROM assets WHERE id = ?", (asset_id,))
    return _row_to_dict(row) | {"duplicate": False}


def listing(database: Database, owner: int | None = None) -> list[dict[str, Any]]:
    if owner is None:
        rows = database.query_all("SELECT * FROM assets ORDER BY id DESC")
    else:
        rows = database.query_all(
            "SELECT * FROM assets WHERE owner = ? ORDER BY id DESC", (owner,)
        )
    return [_row_to_dict(row) for row in rows]


def _notes(detected: Probe) -> list[str]:
    """Usage limits worth showing next to the file, not reasons to reject it."""
    notes: list[str] = []
    if detected.seconds is None:
        return notes
    if detected.kind == "audio" and detected.seconds < MIN_AUDIO_SECONDS:
        notes.append("shorter than the 2 s minimum for a reference audio track")
    elif detected.kind == "audio" and detected.seconds > MAX_AUDIO_SECONDS:
        notes.append(
            "longer than the 15 s total budget for reference audio; "
            "h3 will use the first 15 s"
        )
    elif detected.kind == "video" and detected.seconds < MIN_AUDIO_SECONDS:
        notes.append("shorter than 2 s: usable only as a silent video reference")
    return notes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["metadata"] = json.loads(item["metadata"] or "{}")
    return item
