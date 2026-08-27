#!/usr/bin/env python3
"""Chain multiple h3 shots into one long video with synchronized audio.

H3 renders one shot of at most ~15 seconds. This tool renders a written
scene as a chain of shots: each shot after the first is anchored on the
previous shot's handoff frame through --first-frame (FL2VA), which keeps
subjects, wardrobe, and setting continuous across boundaries. The first
shot may instead use ordered --ref-image identity references (Ref2VA);
H3 does not allow mixing references with frame anchors, so references
apply to the first shot only and continuity afterwards rides on the
handoff frames.

Shots are given one per line (file or stdin). A line may set its own
duration with a leading [Ns] tag:

    [4s] A red fox trots through fresh snow, wide tracking shot.
    [6s] The fox stops at a wooden fence and looks back at the camera.
    The fox walks away toward the treeline in soft morning light.

Each shot renders with the caller's engine flags (window attention, Turbo
steps, and the folded checkpoint are the recommended stack), the handoff
frame is extracted with an optional backoff from the exact last frame,
duplicated seam frames are trimmed, and video plus audio are concatenated
with ffmpeg. A manifest.json records prompts, seeds, timings, and trims.

Example:

    H3_ATTN_WINDOW=2 python3 tools/shot_chain.py \
        -d MiniMax-H3/turbo-v4-step600-ema --shots-file scene.txt \
        --width 1344 --height 768 --seconds-per-shot 5 \
        --steps 8 --layers 45 --out outputs/chains/scene.mp4
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SHOT_TAG = re.compile(r"^\s*\[(\d+(?:\.\d+)?)s\]\s*(.*)$")


def fail(message):
    print(f"shot_chain: {message}", file=sys.stderr)
    sys.exit(1)


def run_logged(command, log_path, env=None, replace=False):
    if replace:
        merged = dict(env or {})
    else:
        merged = dict(os.environ)
        if env:
            merged.update(env)
    started = time.monotonic()
    with open(log_path, "w") as log:
        result = subprocess.run(command, env=merged, stdout=log,
                                stderr=subprocess.STDOUT, text=True)
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        tail = Path(log_path).read_text(errors="replace").splitlines()[-12:]
        fail("shot failed ({}):\n{}".format(result.returncode,
                                            "\n".join(tail)))
    return elapsed


def probe(path, entries, stream="v:0"):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", stream,
         "-count_frames", "-show_entries", entries, "-of", "csv=p=0",
         str(path)],
        stdout=subprocess.PIPE, text=True, check=True).stdout.strip()
    return out


def parse_shots(text):
    shots = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = SHOT_TAG.match(line)
        if match:
            shots.append({"seconds": float(match.group(1)),
                          "prompt": match.group(2).strip()})
        else:
            shots.append({"seconds": None, "prompt": line})
    return shots


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--model-dir", required=True)
    parser.add_argument("--shots-file",
                        help="one shot prompt per line; '-' reads stdin. "
                             "Optional [Ns] duration prefix per line.")
    parser.add_argument("--shot", action="append", default=[],
                        help="add one shot prompt (repeatable, ordered)")
    parser.add_argument("--out", default="outputs/chains/chain.mp4")
    parser.add_argument("--h3", default="./h3")
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seconds-per-shot", type=float, default=5.0,
                        help="duration for shots without a [Ns] tag")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--layers", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vary-seed", action="store_true",
                        help="use seed+index per shot instead of one seed")
    parser.add_argument("--ref-image", action="append", default=[],
                        help="identity reference for the FIRST shot only "
                             "(Ref2VA; repeatable)")
    parser.add_argument("--lora", action="append", default=[],
                        help="runtime LoRA safetensors (repeatable)")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--handoff-offset", type=int, default=1,
                        help="anchor on the Nth frame before the end of the "
                             "previous shot (skips end-of-shot motion blur)")
    parser.add_argument("--trim-seam", type=int, default=1,
                        help="frames dropped from the start of every chained "
                             "shot, whose anchor frame repeats the handoff")
    parser.add_argument("--token-reduction", action="store_true")
    parser.add_argument("--extra", action="append", default=[],
                        help="extra raw h3 flag (repeatable), e.g. "
                             "--extra=--use-int8-row-fc2")
    parser.add_argument("--keep-shots", action="store_true",
                        help="keep per-shot MP4s next to the final output")
    args = parser.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            fail(f"{tool} is required on PATH")
    if not os.path.exists(args.h3):
        fail(f"h3 binary not found at {args.h3}")

    shots = []
    if args.shots_file:
        text = (sys.stdin.read() if args.shots_file == "-" else
                Path(args.shots_file).read_text())
        shots.extend(parse_shots(text))
    shots.extend({"seconds": None, "prompt": prompt}
                 for prompt in args.shot)
    if len(shots) < 1:
        fail("no shots given (use --shots-file or --shot)")
    for shot in shots:
        if shot["seconds"] is None:
            shot["seconds"] = args.seconds_per_shot

    out_path = Path(args.out)
    work = out_path.parent / (out_path.stem + ".shots")
    work.mkdir(parents=True, exist_ok=True)

    manifest = {"shots": [], "settings": {
        "width": args.width, "height": args.height, "steps": args.steps,
        "layers": args.layers, "seed": args.seed,
        "handoff_offset": args.handoff_offset, "trim_seam": args.trim_seam,
        "attn_window": os.environ.get("H3_ATTN_WINDOW"),
        "model_dir": args.model_dir,
    }}
    total_started = time.monotonic()
    handoff = None
    clips = []
    for index, shot in enumerate(shots):
        clip = work / f"shot_{index:02d}.mp4"
        log = work / f"shot_{index:02d}.log"
        seed = args.seed + index if args.vary_seed else args.seed
        command = [args.h3, "-d", args.model_dir, "-p", shot["prompt"],
                   "--width", str(args.width), "--height", str(args.height),
                   "--seconds", str(shot["seconds"]),
                   "--steps", str(args.steps), "--layers", str(args.layers),
                   "--seed", str(seed), "--profile", "-o", str(clip)]
        if args.token_reduction:
            command.append("--token-reduction")
        if index == 0:
            for reference in args.ref_image:
                command += ["--ref-image", reference]
        else:
            command += ["--first-frame", str(handoff)]
        for lora in args.lora:
            command += ["--lora", lora, "--lora-scale",
                        str(args.lora_scale), "--lora-scope", "both"]
        command += args.extra
        print(f"SHOT {index + 1}/{len(shots)} rendering "
              f"({shot['seconds']:g}s): {shot['prompt'][:70]}", flush=True)
        # An unanchored first shot is plain text-to-video: windowed
        # attention breaks object permanence there (background drift), so
        # drop it; every anchored shot keeps the inherited environment.
        if index == 0 and not args.ref_image and "H3_ATTN_WINDOW" in os.environ:
            shot_env = dict(os.environ)
            shot_env.pop("H3_ATTN_WINDOW")
            elapsed = run_logged(command, log, env=shot_env, replace=True)
        else:
            elapsed = run_logged(command, log)
        frames = int(probe(clip, "stream=nb_read_frames"))
        print(f"SHOT {index + 1}/{len(shots)} done in {elapsed:.0f}s "
              f"({frames} frames)", flush=True)

        handoff = work / f"handoff_{index:02d}.png"
        anchor = max(0, frames - 1 - max(0, args.handoff_offset))
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(clip),
             "-vf", f"select='eq(n\\,{anchor})'", "-frames:v", "1",
             str(handoff)], check=True)
        manifest["shots"].append({
            "prompt": shot["prompt"], "seconds": shot["seconds"],
            "seed": seed, "frames": frames, "render_seconds": round(elapsed, 1),
            "anchor_frame": anchor, "clip": clip.name,
        })
        clips.append({"path": clip, "frames": frames, "anchor": anchor})

    # Concatenate: each chained clip starts at trim_seam (its first frames
    # repeat the handoff anchor), and every clip ends at its anchor frame so
    # the next shot continues exactly where this one was sampled.
    parts = []
    for index, clip in enumerate(clips):
        start = 0 if index == 0 else max(0, args.trim_seam)
        stop = clip["anchor"] + 1 if index + 1 < len(clips) else clip["frames"]
        if stop <= start:
            fail(f"shot {index} has no frames left after trimming")
        parts.append((clip["path"], start, stop))
    inputs = []
    filters = []
    for position, (path, start, stop) in enumerate(parts):
        inputs += ["-i", str(path)]
        filters.append(
            f"[{position}:v]trim=start_frame={start}:end_frame={stop},"
            f"setpts=PTS-STARTPTS[v{position}];"
            f"[{position}:a]atrim=start={start / 24:.6f}:end={stop / 24:.6f},"
            f"asetpts=PTS-STARTPTS[a{position}]")
    joined = "".join(f"[v{i}][a{i}]" for i in range(len(parts)))
    filters.append(f"{joined}concat=n={len(parts)}:v=1:a=1[v][a]")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error"] + inputs +
        ["-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "17",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         str(out_path)], check=True)

    total = time.monotonic() - total_started
    manifest["total_seconds"] = round(total, 1)
    manifest["output"] = str(out_path)
    manifest["output_duration"] = probe(out_path, "format=duration",
                                        stream="v:0")
    (work / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if not args.keep_shots:
        for clip in clips:
            clip["path"].unlink(missing_ok=True)
    print(f"CHAIN done: {out_path} ({manifest['output_duration']}s) "
          f"in {total / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
