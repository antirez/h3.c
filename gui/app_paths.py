from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path


def locate_engine_dir(
    *,
    source_root: Path,
    executable: Path | None = None,
    environ: Mapping[str, str] = os.environ,
) -> Path:
    candidates: list[Path] = []
    configured = environ.get("H3_ENGINE_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(source_root)
    executable = executable or Path(sys.executable)
    if len(executable.parents) >= 2:
        candidates.append(executable.resolve().parents[1] / "Resources")
    for candidate in candidates:
        if (candidate / "h3").is_file() and (
            candidate / "h3_shaders.metal"
        ).is_file():
            return candidate.resolve()
    return source_root.resolve()
