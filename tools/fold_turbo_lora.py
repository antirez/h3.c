#!/usr/bin/env python3
"""Fold a MiniMax-H3 LoRA into the bf16 checkpoint by in-place byte patching.

h3.c has no LoRA runtime, and does not need one: a LoRA is W' = W + scale*(B@A),
which can be baked into the checkpoint once, offline. This tool clones the
original shards (copy-on-write where the filesystem supports it) and rewrites
only the byte ranges of the targeted tensors, so headers, tensor order, and
alignment stay byte-identical to the originals.

Folding the 4-step Turbo distillation adapter this way enables 5-6 step
sampling at zero runtime cost. Measured on an M5 Max (960x544, identical
prompt/seed): 39 frames 87s -> 65s and a 5s clip 8.8min -> 6.2min versus the
--steps 20 --reuse 2 --layers 45 preset, at comparable visual quality.

Requires only numpy. Adapter tensors must be named <target>.lora_A.weight /
<target>.lora_B.weight over the checkpoint's own key space, as
larryvrh/MiniMax-H3-Turbo-Lora's packaged .safetensors files are.

Usage:
  python3 tools/fold_turbo_lora.py \
      --checkpoint MiniMax-H3/FL2VA/transformer \
      --lora minimax_h3_turbo_v4_step600_ema.safetensors \
      --out MiniMax-H3-turbo/FL2VA/transformer
Then point h3 at a model directory whose transformer is the folded tree and
sample with --steps 5 or 6 (4 shows motion smear; do not combine with --reuse).
"""
import argparse
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np


def read_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    return hlen, header


def load_tensor(path, info, base):
    start, end = info["data_offsets"]
    dtype = info["dtype"]
    with open(path, "rb") as f:
        f.seek(base + start)
        raw = f.read(end - start)
    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
        return (u16 << 16).view(np.float32).reshape(info["shape"])
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(info["shape"]).copy()
    raise SystemExit(f"unsupported dtype {dtype} for {path}")


def f32_to_bf16_bytes(x):
    u32 = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = (u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16  # round to nearest even
    return rounded.astype("<u2").tobytes()


def clone(src, dst):
    if subprocess.run(["cp", "-c", str(src), str(dst)],
                      capture_output=True).returncode != 0:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="directory of original bf16 transformer shards")
    ap.add_argument("--lora", required=True, type=Path,
                    help="LoRA .safetensors with <key>.lora_A/.lora_B pairs")
    ap.add_argument("--out", required=True, type=Path,
                    help="output directory for the folded shards")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    lhl, lora_hdr = read_header(args.lora)
    lora_base = 8 + lhl
    pairs = {}
    for key in lora_hdr:
        if key.endswith(".lora_A.weight"):
            target = key[: -len(".lora_A.weight")] + ".weight"
            pairs[target] = (key, key[: -len(".lora_A.weight")] + ".lora_B.weight")
    if not pairs:
        raise SystemExit("no .lora_A/.lora_B pairs found in the adapter")
    print(f"{len(pairs)} adapter pairs")

    args.out.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.checkpoint.glob("model-*.safetensors"))
    if not shards:
        raise SystemExit(f"no shards in {args.checkpoint}")
    for extra in ("config.json", "model.safetensors.index.json"):
        src = args.checkpoint / extra
        if src.exists():
            shutil.copy2(src, args.out / extra)

    patched = 0
    for shard in shards:
        dst = args.out / shard.name
        clone(shard, dst)
        hlen, hdr = read_header(dst)
        base = 8 + hlen
        todo = [k for k in hdr if k in pairs]
        if not todo:
            continue
        with open(dst, "r+b") as f:
            for key in todo:
                info = hdr[key]
                if info["dtype"] != "BF16":
                    raise SystemExit(f"{key}: expected BF16, got {info['dtype']}")
                a_key, b_key = pairs[key]
                A = load_tensor(args.lora, lora_hdr[a_key], lora_base)
                B = load_tensor(args.lora, lora_hdr[b_key], lora_base)
                W = load_tensor(dst, info, base)
                folded = W + args.scale * (B.astype(np.float32) @ A.astype(np.float32))
                # parity check on a random probe before committing bytes
                x = np.random.default_rng(0).standard_normal(W.shape[1]).astype(np.float32)
                direct = folded @ x
                composed = W @ x + args.scale * (B @ (A @ x))
                rel = np.max(np.abs(direct - composed)) / (np.max(np.abs(composed)) + 1e-9)
                if rel > 1e-4:
                    raise SystemExit(f"{key}: parity check failed (rel={rel:.2e})")
                buf = f32_to_bf16_bytes(folded)
                start, end = info["data_offsets"]
                if len(buf) != end - start:
                    raise SystemExit(f"{key}: byte count mismatch")
                f.seek(base + start)
                f.write(buf)
                patched += 1
        print(f"{shard.name}: cumulative {patched}/{len(pairs)}")

    if patched != len(pairs):
        raise SystemExit(f"only {patched} of {len(pairs)} adapter pairs matched "
                         "checkpoint tensors - wrong checkpoint or adapter?")
    print("done")


if __name__ == "__main__":
    sys.exit(main())
