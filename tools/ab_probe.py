#!/usr/bin/env python3
"""Automated A/B quality/speed probe for stacked h3 acceleration options.

Renders a short clip with the same prompt/seed across a progressive stack:

    ref          reference schedule (default 50 steps, no speed flags)
    turbo6       folded/runtime Turbo LoRA at 6 steps
    turbo6_l45   + --layers 45
    turbo6_l45_red    + --token-reduction (spatial)
    turbo6_l45_tred   + --token-reduction with H3_TOKEN_REDUCTION_AXIS=time
                        (optional, --temporal)

Each variant is timed, compared against the reference and the previous
stage with ffmpeg SSIM/PSNR, and summarized in a markdown report with
first/middle/last contact sheets for human inspection of composition
drift. If the reduction stage costs markedly more SSIM than the stage
before it, the report recommends falling back to turbo6+layers45.

Requires ffmpeg/ffprobe on PATH. LPIPS is computed only with --lpips and
an importable torch+lpips. Audio is intentionally not evaluated.

Example:

    python3 tools/ab_probe.py \
        -d ./MiniMax-H3 --turbo-lora turbo_v4.safetensors \
        -p "A red fox walks through fresh snow in a pine forest." \
        --width 768 --height 768 --seconds 3 --out-dir outputs/probe
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time


def fail(message):
    print(f"ab_probe: {message}", file=sys.stderr)
    sys.exit(1)


def run(command, env=None, log_path=None, dry_run=False):
    printable = " ".join(command)
    if env:
        printable = " ".join(f"{k}={v}" for k, v in env.items()) + " " + printable
    print(f"+ {printable}", flush=True)
    if dry_run:
        return 0.0, ""
    merged = dict(os.environ)
    if env:
        merged.update(env)
    start = time.monotonic()
    result = subprocess.run(command, env=merged, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    elapsed = time.monotonic() - start
    if log_path:
        with open(log_path, "w") as log:
            log.write(result.stdout)
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-15:])
        fail(f"command failed ({result.returncode}):\n{tail}")
    return elapsed, result.stdout


def ffprobe_frames(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
    return int(out)


def video_metrics(distorted, reference):
    """Return (ssim, psnr) of distorted vs reference via ffmpeg filters."""
    out = subprocess.run(
        ["ffmpeg", "-nostats", "-i", distorted, "-i", reference,
         "-lavfi", "[0:v][1:v]ssim;[0:v][1:v]psnr", "-f", "null", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    ssim = psnr = None
    match = re.search(r"SSIM.*All:\s*([0-9.]+)", out)
    if match:
        ssim = float(match.group(1))
    match = re.search(r"PSNR.*average:\s*([0-9.]+|inf)", out)
    if match:
        psnr = float("inf") if match.group(1) == "inf" else float(match.group(1))
    return ssim, psnr


def lpips_metric(distorted, reference, work_dir, samples=8):
    try:
        import lpips  # type: ignore
        import torch  # type: ignore
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return None
    frames = ffprobe_frames(reference)
    indices = sorted({round(i * (frames - 1) / (samples - 1)) for i in range(samples)})
    model = lpips.LPIPS(net="alex", verbose=False)
    values = []
    for tag, path in (("a", distorted), ("b", reference)):
        select = "+".join(f"eq(n\\,{i})" for i in indices)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", path,
             "-vf", f"select='{select}'", "-vsync", "0",
             os.path.join(work_dir, f"lpips_{tag}_%03d.png")], check=True)
    for index in range(1, len(indices) + 1):
        pair = []
        for tag in ("a", "b"):
            image = Image.open(
                os.path.join(work_dir, f"lpips_{tag}_{index:03d}.png")).convert("RGB")
            array = np.asarray(image).astype("float32") / 127.5 - 1.0
            pair.append(torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0))
        with torch.no_grad():
            values.append(model(pair[0], pair[1]).item())
    for name in os.listdir(work_dir):
        if name.startswith("lpips_"):
            os.remove(os.path.join(work_dir, name))
    return sum(values) / len(values)


def contact_sheet(video, out_png):
    frames = ffprobe_frames(video)
    picks = [0, frames // 2, frames - 1]
    select = "+".join(f"eq(n\\,{i})" for i in picks)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", video,
         "-vf", f"select='{select}',tile=3x1", "-frames:v", "1", out_png],
        check=True)


def comparison_grid(sheets, out_png):
    existing = [s for s in sheets if os.path.exists(s)]
    if len(existing) < 2:
        return
    inputs = []
    for sheet in existing:
        inputs += ["-i", sheet]
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error"] + inputs +
        ["-filter_complex", f"vstack=inputs={len(existing)}", out_png],
        check=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-d", "--model-dir", required=True)
    parser.add_argument("-p", "--prompt", required=True)
    parser.add_argument("--turbo-lora",
                        help="Turbo step-distillation adapter (.safetensors); "
                             "without it fast variants stay at --fast-steps")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--h3", default="./h3", help="h3 binary (default ./h3)")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seconds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-steps", type=int, default=50)
    parser.add_argument("--turbo-steps", type=int, default=6)
    parser.add_argument("--fast-steps", type=int, default=20,
                        help="steps for fast variants when no Turbo adapter")
    parser.add_argument("--out-dir", default="outputs/ab_probe")
    parser.add_argument("--temporal", action="store_true",
                        help="also probe temporal token reduction")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--conflict-margin", type=float, default=0.05,
                        help="SSIM drop vs previous stage that flags a conflict")
    parser.add_argument("--severe-floor", type=float, default=0.55,
                        help="SSIM vs reference below this flags severe divergence")
    parser.add_argument("--resume", action="store_true",
                        help="reuse existing MP4s in --out-dir")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_variants(args):
    turbo = []
    steps_fast = args.fast_steps
    if args.turbo_lora:
        turbo = ["--lora", args.turbo_lora,
                 "--lora-scale", str(args.lora_scale),
                 "--lora-scope", "fl2va"]
        steps_fast = args.turbo_steps
    label = "turbo6" if args.turbo_lora else f"steps{steps_fast}"
    variants = [
        ("ref", ["--steps", str(args.reference_steps)], {}),
        (label, turbo + ["--steps", str(steps_fast)], {}),
        (f"{label}_l45", turbo + ["--steps", str(steps_fast), "--layers", "45"],
         {}),
        (f"{label}_l45_red",
         turbo + ["--steps", str(steps_fast), "--layers", "45",
                  "--token-reduction"], {}),
    ]
    if args.temporal:
        variants.append(
            (f"{label}_l45_tred",
             turbo + ["--steps", str(steps_fast), "--layers", "45",
                      "--token-reduction"],
             {"H3_TOKEN_REDUCTION_AXIS": "time"}))
    return variants


def main():
    args = parse_args()
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            fail(f"{tool} is required on PATH")
    if not args.dry_run and not os.path.exists(args.h3):
        fail(f"h3 binary not found at {args.h3} (build with make)")
    if args.turbo_lora and not args.dry_run and not os.path.exists(args.turbo_lora):
        fail(f"Turbo adapter not found: {args.turbo_lora}")
    if not args.turbo_lora:
        print("ab_probe: no --turbo-lora; fast variants use "
              f"--steps {args.fast_steps} (base schedule)", file=sys.stderr)
    os.makedirs(args.out_dir, exist_ok=True)

    variants = build_variants(args)
    results = []
    for name, extra, env in variants:
        output = os.path.join(args.out_dir, f"{name}.mp4")
        log = os.path.join(args.out_dir, f"{name}.log")
        command = [args.h3, "-d", args.model_dir, "-p", args.prompt,
                   "--width", str(args.width), "--height", str(args.height),
                   "--seconds", str(args.seconds), "--seed", str(args.seed),
                   "--profile", "-o", output] + extra
        if args.resume and os.path.exists(output):
            print(f"= {name}: reusing {output}")
            elapsed = None
        else:
            elapsed, _ = run(command, env=env, log_path=log,
                             dry_run=args.dry_run)
        results.append({"name": name, "output": output, "wall": elapsed,
                        "env": env, "args": extra})

    if args.dry_run:
        return

    reference = results[0]
    sheets = []
    for index, entry in enumerate(results):
        entry["frames"] = ffprobe_frames(entry["output"])
        sheet = os.path.join(args.out_dir, f"{entry['name']}_sheet.png")
        contact_sheet(entry["output"], sheet)
        sheets.append(sheet)
        if index == 0:
            entry["ssim_ref"] = entry["psnr_ref"] = None
            entry["ssim_prev"] = None
            continue
        entry["ssim_ref"], entry["psnr_ref"] = video_metrics(
            entry["output"], reference["output"])
        previous = results[index - 1]
        entry["ssim_prev"], _ = video_metrics(
            entry["output"], previous["output"])
        if args.lpips:
            entry["lpips_ref"] = lpips_metric(
                entry["output"], reference["output"], args.out_dir)
    comparison_grid(sheets, os.path.join(args.out_dir, "compare_all.png"))

    verdicts = []
    stage_ssims = [(e["name"], e["ssim_ref"]) for e in results[1:]
                   if e["ssim_ref"] is not None]
    for position, (name, ssim) in enumerate(stage_ssims):
        if ssim < args.severe_floor:
            verdicts.append(f"SEVERE: {name} SSIM vs ref {ssim:.3f} < "
                            f"{args.severe_floor} — inspect for artifacts")
        if position > 0:
            prev_name, prev_ssim = stage_ssims[position - 1]
            if prev_ssim - ssim > args.conflict_margin:
                verdicts.append(
                    f"CONFLICT: {name} loses {prev_ssim - ssim:.3f} SSIM over "
                    f"{prev_name} — recommend falling back to {prev_name}")
    if not verdicts:
        verdicts.append("No SSIM-detectable conflict; confirm composition "
                        "drift by eye in compare_all.png before scaling to 10s.")

    report = os.path.join(args.out_dir, "report.md")
    with open(report, "w") as out:
        out.write(f"# h3 A/B probe\n\nprompt: `{args.prompt}`\n\n"
                  f"seed {args.seed}, {args.width}x{args.height}, "
                  f"{args.seconds}s, reference {args.reference_steps} steps\n\n")
        out.write("| variant | wall s | speedup | SSIM vs ref | PSNR vs ref "
                  "| SSIM vs prev |" +
                  (" LPIPS vs ref |" if args.lpips else "") + "\n")
        out.write("|---|---:|---:|---:|---:|---:|" +
                  ("---:|" if args.lpips else "") + "\n")
        base_wall = reference["wall"]
        for entry in results:
            wall = f"{entry['wall']:.1f}" if entry["wall"] else "reused"
            speed = (f"{base_wall / entry['wall']:.2f}x"
                     if entry["wall"] and base_wall else "-")
            ssim_ref = (f"{entry['ssim_ref']:.4f}"
                        if entry.get("ssim_ref") is not None else "-")
            psnr_ref = (f"{entry['psnr_ref']:.2f}"
                        if entry.get("psnr_ref") is not None else "-")
            ssim_prev = (f"{entry['ssim_prev']:.4f}"
                         if entry.get("ssim_prev") is not None else "-")
            row = (f"| {entry['name']} | {wall} | {speed} | {ssim_ref} "
                   f"| {psnr_ref} | {ssim_prev} |")
            if args.lpips:
                lp = entry.get("lpips_ref")
                row += f" {lp:.4f} |" if lp is not None else " - |"
            out.write(row + "\n")
        out.write("\n## Verdicts\n\n")
        for verdict in verdicts:
            out.write(f"- {verdict}\n")
        out.write("\nAudio is not evaluated (accepted lossy). Token "
                  "reduction legitimately changes composition, so judge "
                  "artifact severity (ringing, ghosted limbs, identity "
                  "drift) from compare_all.png, not SSIM alone.\n")
    print(f"\nreport: {report}")
    for verdict in verdicts:
        print(verdict)
    print(json.dumps(
        [{k: v for k, v in e.items() if k not in ("env", "args")}
         for e in results], indent=2, default=str))


if __name__ == "__main__":
    main()
