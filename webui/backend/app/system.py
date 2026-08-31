"""Device and checkpoint inventory, read from `h3 --info`.

`h3 --info` only reads checkpoint headers, so it is cheap enough to call on
demand and cache. Everything degrades to `available: false` with a reason
instead of raising, so the UI can explain what is missing.
"""

import re
import subprocess
from pathlib import Path
from typing import Any

_COMPONENT = re.compile(
    r"^\s{2}(?P<label>.+?)\s{2,}(?P<files>\d+) files\s+(?P<tensors>\d+) tensors"
    r"\s+(?P<gib>[\d.]+) GiB\s*$"
)
_MEMORY = re.compile(r"^\s{2}(?P<label>[a-zA-Z0-9 ]+?)\s{2,}(?P<value>.+?)\s*$")


def parse_info(text: str) -> dict[str, Any]:
    """Turn the plain-text `h3 --info` report into structured data."""
    info: dict[str, Any] = {"device": {}, "components": {}}
    in_inventory = False
    for line in text.splitlines():
        if line.startswith("h3-"):
            engine, _, version = line.partition(" ")
            info["engine"] = engine
            info["version"] = version.strip()
            continue
        if line.startswith("Device:"):
            device = line[len("Device:") :].strip()
            match = re.match(r"^(?P<name>.*?)\s*\((?P<architecture>[^)]*)\)$", device)
            info["device"] = (
                match.groupdict() if match else {"name": device, "architecture": ""}
            )
            continue
        if line.startswith("Native checkpoint inventory"):
            in_inventory = True
            continue
        if in_inventory:
            match = _COMPONENT.match(line)
            if match:
                info["components"][match["label"].strip()] = {
                    "files": int(match["files"]),
                    "tensors": int(match["tensors"]),
                    "gib": float(match["gib"]),
                }
            continue
        match = _MEMORY.match(line)
        if match:
            key = match["label"].strip().replace(" ", "_")
            info["device"][key] = match["value"].strip()
    return info


def read_system(binary: Path, model_dir: Path, timeout: float) -> dict[str, Any]:
    """Run `h3 --info`, or explain why it could not run."""
    if not binary.exists():
        return _unavailable(f"h3 binary not found at {binary}")
    if not model_dir.is_dir():
        return _unavailable(f"model directory not found at {model_dir}")
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [str(binary), "--info", "-d", str(model_dir)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _unavailable(f"h3 --info timed out after {timeout:g}s")
    except OSError as error:
        return _unavailable(f"cannot run h3 --info: {error}")
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip().splitlines()
        return _unavailable(detail[-1] if detail else "h3 --info failed")
    info = parse_info(done.stdout)
    info["available"] = True
    info["model_dir"] = str(model_dir)
    info["has_ref2va"] = any(
        "Ref2VA" in label and entry["files"] > 0
        for label, entry in info["components"].items()
    )
    return info


def _unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason, "device": {}, "components": {}}
