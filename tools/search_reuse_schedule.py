#!/usr/bin/env python3
"""Benchmark non-uniform H3_REUSE_STEPS schedules against a full reference.

The tool intentionally uses the existing public CLI and environment override.
It can be prepared before model weights are installed with --dry-run, then run
unchanged after the checkpoint returns.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


def exact_schedule(steps, evaluations, curve):
    """Return exactly `evaluations` increasing indices including both ends."""
    if steps < 2 or evaluations < 2 or evaluations > steps:
        raise ValueError("evaluations must be in [2, steps]")
    selected = []
    for ordinal in range(evaluations):
        remaining = evaluations - ordinal - 1
        low = selected[-1] + 1 if selected else 0
        high = steps - remaining - 1
        position = int(round(curve(ordinal / (evaluations - 1)) * (steps - 1)))
        selected.append(min(high, max(low, position)))
    selected[0] = 0
    selected[-1] = steps - 1
    return tuple(selected)


def candidate_schedules(steps, counts, gammas):
    """Generate deterministic uniform, power, and endpoint-heavy schedules."""
    candidates = {}
    for count in counts:
        for gamma in gammas:
            name = f"power-{gamma:g}-n{count}"
            candidates[name] = exact_schedule(
                steps, count, lambda value, exponent=gamma: value ** exponent
            )
        candidates[f"cosine-n{count}"] = exact_schedule(
            steps, count, lambda value: (1.0 - math.cos(math.pi * value)) / 2.0
        )
    unique = {}
    for name, schedule in candidates.items():
        unique.setdefault(schedule, name)
    return [(name, schedule) for schedule, name in unique.items()]


def parse_csv_numbers(value, cast, label):
    try:
        values = [cast(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid {label}: {value}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{label} must not be empty")
    return values


def video_ssim(ffmpeg, reference, candidate):
    command = [
        ffmpeg, "-hide_banner", "-nostdin", "-i", str(reference),
        "-i", str(candidate), "-lavfi", "[0:v][1:v]ssim",
        "-f", "null", "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    match = re.search(r"All:([0-9.]+)", result.stderr)
    return float(match.group(1)) if result.returncode == 0 and match else None


def run_case(command, environment, log_path, timeout):
    started = time.monotonic()
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
        timeout=timeout,
        check=False,
    )
    elapsed = time.monotonic() - started
    log_path.write_text(result.stdout, encoding="utf-8")
    return result.returncode, elapsed


def write_summary(path, results):
    lines = [
        "# H3 reuse schedule search",
        "",
        "| Schedule | Evaluations | Wall time | Speedup | SSIM | Result |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in results:
        speedup = item.get("speedup")
        ssim = item.get("ssim")
        lines.append(
            f"| {item['name']} | {item['evaluations']} | "
            f"{item['elapsed_seconds']:.2f}s | "
            f"{speedup:.3f}x | " if speedup is not None else
            f"| {item['name']} | {item['evaluations']} | "
            f"{item['elapsed_seconds']:.2f}s | — | "
        )
        lines[-1] += f"{ssim:.6f} | " if ssim is not None else "— | "
        lines[-1] += "ok |" if item["return_code"] == 0 else "failed |"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h3", default="./h3")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", default="outputs/reuse-search")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--counts", default="8,9,11")
    parser.add_argument("--gammas", default="0.55,0.75,1,1.35,1.8")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "extra", nargs=argparse.REMAINDER,
        help="additional h3 arguments after --, excluding steps/reuse/output",
    )
    args = parser.parse_args()

    counts = parse_csv_numbers(args.counts, int, "counts")
    gammas = parse_csv_numbers(args.gammas, float, "gammas")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if any(count < 2 or count > args.steps for count in counts):
        parser.error("every evaluation count must be in [2, steps]")
    forbidden = {"--steps", "--reuse", "-o", "--output", "-d", "--model-dir"}
    if any(item in forbidden for item in args.extra):
        parser.error("extra arguments must not override model, steps, reuse, or output")

    output_dir = Path(args.output_dir)
    schedules = candidate_schedules(args.steps, counts, gammas)
    cases = [("reference", None)] + schedules
    if args.dry_run:
        for name, schedule in cases:
            value = "full" if schedule is None else ",".join(map(str, schedule))
            print(f"{name}: {value}")
        return 0

    h3 = Path(args.h3).resolve()
    model_dir = Path(args.model_dir).resolve()
    if not h3.is_file():
        parser.error(f"h3 executable not found: {h3}")
    if not model_dir.is_dir():
        parser.error(f"model directory not found: {model_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = os.environ.get("H3_FFMPEG") or shutil.which("ffmpeg")
    results = []
    reference_output = output_dir / "reference.mp4"

    for name, schedule in cases:
        output = reference_output if schedule is None else output_dir / f"{name}.mp4"
        command = [
            str(h3), "--profile", "-d", str(model_dir), "-p", args.prompt,
            "--steps", str(args.steps), "--reuse", "1" if schedule is None else "2",
            "-o", str(output), *args.extra,
        ]
        environment = os.environ.copy()
        if schedule is None:
            environment.pop("H3_REUSE_STEPS", None)
        else:
            environment["H3_REUSE_STEPS"] = ",".join(map(str, schedule))
        print(f"[{name}] {' '.join(command)}", flush=True)
        return_code, elapsed = run_case(
            command, environment, output_dir / f"{name}.log", args.timeout
        )
        item = {
            "name": name,
            "schedule": list(schedule) if schedule is not None else None,
            "evaluations": args.steps if schedule is None else len(schedule),
            "elapsed_seconds": elapsed,
            "return_code": return_code,
            "output": str(output),
            "ssim": None,
            "speedup": None,
        }
        if return_code == 0 and schedule is not None and ffmpeg:
            item["ssim"] = video_ssim(ffmpeg, reference_output, output)
        if results and return_code == 0 and results[0]["return_code"] == 0:
            item["speedup"] = results[0]["elapsed_seconds"] / elapsed
        results.append(item)

    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_summary(output_dir / "results.md", results)
    return 0 if all(item["return_code"] == 0 for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
