"""Optional post-processing stage: an extension point, not an integration.

A plugin is an external executable, not a Python import: FaceFusion and its
kin live in their own virtualenv with pinned onnxruntime builds, and importing
them here would inherit those constraints.

Contract, so a third party can implement one without reading this file:

    $H3_<NAME>_CMD --input IN.mp4 --output OUT.mp4 [--param value ...]

    exit 0  and OUT.mp4 written  -> the job's video is replaced
    exit != 0                    -> the job fails, IN.mp4 is kept

This repository ships no models, no weights and no download URLs. Plugins are
unavailable until the operator points the environment variable at an
executable they installed themselves.
"""

import contextlib
import os
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings


@dataclass
class Plugin:
    name: str
    label: str
    description: str
    env_var: str
    command: str
    notice: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.command) and (
            Path(self.command).is_file() or shutil.which(self.command) is not None
        )

    @property
    def reason(self) -> str | None:
        if self.available:
            return None
        if not self.command:
            return (
                f"no model and no runtime installed: set {self.env_var} to an "
                "executable to enable it"
            )
        return f"{self.env_var} points at {self.command}, which is not executable"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "env_var": self.env_var,
            "available": self.available,
            "reason": self.reason,
            "notice": self.notice,
        }


def registry(config: Settings) -> list[Plugin]:
    return [
        Plugin(
            name="faceswap",
            label="Face replacement",
            description=(
                "Replaces faces in the generated video using an external "
                "face-swapping runtime."
            ),
            env_var="H3_FACESWAP_CMD",
            command=config.faceswap_cmd,
            notice=(
                "This repository ships neither models nor download URLs. Known "
                "model licences are non-commercial/research only, and the "
                "operator is responsible for checking them. Do not use it on "
                "images of real people without their consent."
            ),
        )
    ]


def by_name(config: Settings, name: str) -> Plugin | None:
    return next((plugin for plugin in registry(config) if plugin.name == name), None)


class PluginError(RuntimeError):
    pass


def run_stage(
    config: Settings, video: Path, requested: list[str], timeout: float = 3600.0
) -> Path:
    """Run each requested plugin in order, replacing the video each time."""
    for name in requested:
        plugin = by_name(config, name)
        if plugin is None:
            raise PluginError(f"unknown post-processing plugin: {name}")
        if not plugin.available:
            raise PluginError(f"post-processing plugin {name} is unavailable")
        produced = video.with_name(f"{video.stem}-{name}{video.suffix}")
        try:
            process = subprocess.Popen(  # noqa: S603 - argv list, no shell
                [plugin.command, "--input", str(video), "--output", str(produced)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as error:
            raise PluginError(
                f"post-processing {name} could not run: {error}"
            ) from error
        try:
            out, err = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Kill the whole session, not just the plugin: it may have
            # spawned helpers of its own, and an orphan would keep working
            # on the video after the job is declared failed (T106).
            with contextlib.suppress(OSError):
                os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(OSError):
                process.wait(timeout=10)
            raise PluginError(
                f"post-processing {name} timed out after {timeout:.0f} s"
            ) from None
        if process.returncode != 0 or not produced.is_file():
            detail = (err or out or "").strip().splitlines()
            last = detail[-1] if detail else "no output"
            raise PluginError(f"post-processing {name} failed: {last}")
        produced.replace(video)
    return video
