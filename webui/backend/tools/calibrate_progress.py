"""Measure how long each h3 phase takes, to weight the progress bar.

Usage:
    webui/backend/.venv/bin/python webui/backend/tools/calibrate_progress.py \
        --model-dir ./MiniMax-H3 [h3 options...]

Prints a JSON summary and, with --write, updates
webui/shared/progress_weights.json. Re-run it when the hardware changes: a
progress bar calibrated on another machine is a guess, not a measurement.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROGRESS = re.compile(r"^(?P<phase>\S.*?)\s{2,}(?P<completed>\d+)/(?P<total>\d+)\s*$")


def measure(argv: list[str]) -> dict[str, float]:
    started = time.monotonic()
    marks: dict[str, float] = {}
    order: list[str] = []
    current: str | None = None
    process = subprocess.Popen(argv, stderr=subprocess.PIPE, text=True)
    buffer = ""
    assert process.stderr is not None
    while True:
        chunk = process.stderr.read(1)
        if not chunk:
            break
        if chunk in "\r\n":
            line, buffer = buffer, ""
            match = PROGRESS.match(line.strip())
            if match and match["phase"].strip() != current:
                current = match["phase"].strip()
                if current not in marks:
                    marks[current] = time.monotonic()
                    order.append(current)
                    print(f"  {current}", file=sys.stderr)
        else:
            buffer += chunk
    process.wait()
    ended = time.monotonic()
    durations: dict[str, float] = {}
    for index, phase in enumerate(order):
        following = marks[order[index + 1]] if index + 1 < len(order) else ended
        durations[phase] = round(following - marks[phase], 3)
    durations["_total"] = round(ended - started, 3)
    durations["_returncode"] = float(process.returncode)
    return durations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=str(ROOT / "h3"))
    parser.add_argument("--model-dir", default=str(ROOT / "MiniMax-H3"))
    parser.add_argument("--prompt", default="A bright red cube on a white background.")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--frames", type=int, default=22)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--layers", type=int, default=50)
    parser.add_argument("--output", default="/dev/null")
    parser.add_argument("--write", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="add this run as another sample instead of replacing the file, so "
             "fixed and variable cost can be told apart",
    )
    args = parser.parse_args()

    argv = [
        args.binary, "-d", args.model_dir, "-p", args.prompt,
        "--width", str(args.width), "--height", str(args.height),
        "--frames", str(args.frames), "--steps", str(args.steps),
        "--layers", str(args.layers), "--seed", "42", "-o", args.output,
    ]
    durations = measure(argv)
    report = {
        "reference": {
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "steps": args.steps,
            "layers": args.layers,
        },
        "phase_seconds": {k: v for k, v in durations.items() if not k.startswith("_")},
        "total_seconds": durations["_total"],
    }
    print(json.dumps(report, indent=2))
    if args.write:
        target = ROOT / "webui/shared/progress_weights.json"
        existing = json.loads(target.read_text()) if target.exists() else {}
        sample = {
            "reference": report["reference"],
            "phase_seconds": report["phase_seconds"],
            "total_seconds": report["total_seconds"],
        }
        if args.append:
            samples = existing.get("samples", [])
            if not samples and "phase_seconds" in existing:
                samples = [
                    {
                        "reference": existing["reference"],
                        "phase_seconds": existing["phase_seconds"],
                        "total_seconds": existing["total_seconds"],
                    }
                ]
            samples.append(sample)
            existing["samples"] = samples
        else:
            existing = {**existing, **sample, "samples": [sample]}
        target.write_text(json.dumps(existing, indent=2) + "\n")
        print(f"wrote {target}", file=sys.stderr)
    return int(durations["_returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
