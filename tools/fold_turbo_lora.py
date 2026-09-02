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

Requires only numpy. Supported adapter layouts are native
`<target>.lora_A/B.weight`, Diffusers/AI Toolkit
`diffusion_model.<target>.lora_A/B.weight`, and Kohya/ComfyUI
`lora_unet_<target>.lora_down/up.weight` with optional alpha tensors.

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
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(info["shape"])
    raise SystemExit(f"unsupported dtype {dtype} for {path}")


def f32_to_bf16_bytes(x):
    u32 = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = (u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16  # round to nearest even
    return rounded.astype("<u2").tobytes()


def require_finite(label, values):
    """Reject real NaN/Inf while tolerating spurious Accelerate FP flags."""
    if not np.isfinite(values).all():
        raise SystemExit(f"{label}: non-finite values produced during folding")


def clone(src, dst):
    if subprocess.run(["cp", "-c", str(src), str(dst)],
                      capture_output=True).returncode != 0:
        shutil.copy2(src, dst)


def checkpoint_tensors(shard_headers):
    tensors = {}
    for _, header in shard_headers:
        for name, info in header.items():
            if name in tensors:
                raise SystemExit(f"duplicate checkpoint tensor: {name}")
            tensors[name] = info
    return tensors


def kohya_target_map(tensor_names):
    result = {}
    collisions = set()
    for name in tensor_names:
        if not name.endswith(".weight"):
            continue
        encoded = name[:-len(".weight")].replace(".", "_")
        if encoded in result and result[encoded] != name:
            collisions.add(encoded)
        else:
            result[encoded] = name
    for encoded in collisions:
        result.pop(encoded, None)
    return result


def scalar_tensor(path, header, base, name):
    values = load_tensor(path, header[name], base).reshape(-1)
    if values.size != 1 or not np.isfinite(values[0]):
        raise SystemExit(f"{name}: alpha must contain one finite value")
    return float(values[0])


def resolve_pairs(lora_path, lora_header, lora_base, checkpoint):
    """Map common H3 LoRA key conventions to checkpoint tensor names."""
    pairs = {}
    formats = {}
    kohya_map = kohya_target_map(checkpoint)

    def add_pair(target, a_key, b_key, alpha_key, format_name):
        if target not in checkpoint:
            raise SystemExit(f"{a_key}: target {target} is absent from checkpoint")
        if b_key not in lora_header:
            raise SystemExit(f"{a_key}: missing paired tensor {b_key}")
        if target in pairs:
            raise SystemExit(f"multiple adapter pairs map to {target}")
        a_info = lora_header[a_key]
        b_info = lora_header[b_key]
        target_info = checkpoint[target]
        a_shape = a_info.get("shape", [])
        b_shape = b_info.get("shape", [])
        target_shape = target_info.get("shape", [])
        if len(a_shape) != 2 or len(b_shape) != 2 or len(target_shape) != 2:
            raise SystemExit(f"{target}: LoRA and checkpoint weights must be matrices")
        rank = a_shape[0]
        if not rank or b_shape[1] != rank or b_shape[0] != target_shape[0] or \
                a_shape[1] != target_shape[1]:
            raise SystemExit(
                f"{target}: incompatible shapes A={a_shape}, B={b_shape}, "
                f"checkpoint={target_shape}"
            )
        pair_scale = 1.0
        if alpha_key and alpha_key in lora_header:
            pair_scale = scalar_tensor(
                lora_path, lora_header, lora_base, alpha_key
            ) / rank
        pairs[target] = (a_key, b_key, pair_scale)
        formats[format_name] = formats.get(format_name, 0) + 1

    for key in lora_header:
        if key.endswith(".lora_A.weight"):
            stem = key[:-len(".lora_A.weight")]
            target_stem = stem[len("diffusion_model."):] \
                if stem.startswith("diffusion_model.") else stem
            format_name = "diffusion_model" \
                if stem.startswith("diffusion_model.") else "native"
            add_pair(
                target_stem + ".weight",
                key,
                stem + ".lora_B.weight",
                stem + ".alpha",
                format_name,
            )
        elif key.startswith("lora_unet_") and key.endswith(".lora_down.weight"):
            stem = key[len("lora_unet_"):-len(".lora_down.weight")]
            target = kohya_map.get(stem)
            if not target:
                raise SystemExit(f"{key}: cannot map Kohya key to checkpoint")
            prefix = key[:-len(".lora_down.weight")]
            add_pair(
                target,
                key,
                prefix + ".lora_up.weight",
                prefix + ".alpha",
                "kohya",
            )
    if not pairs:
        raise SystemExit("no supported LoRA pairs found in the adapter")
    return pairs, formats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="directory of original bf16 transformer shards")
    ap.add_argument("--lora", required=True, type=Path,
                    help="H3 LoRA in native, Diffusers, or Kohya key format")
    ap.add_argument("--out", type=Path,
                    help="output directory for the folded shards")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and report matches without writing output")
    args = ap.parse_args()
    if not args.dry_run and args.out is None:
        ap.error("--out is required unless --dry-run is used")
    if not np.isfinite(args.scale):
        ap.error("--scale must be finite")

    lhl, lora_hdr = read_header(args.lora)
    lora_base = 8 + lhl
    shards = sorted(args.checkpoint.glob("model-*.safetensors"))
    if not shards:
        raise SystemExit(f"no shards in {args.checkpoint}")
    shard_headers = [(shard, read_header(shard)[1]) for shard in shards]
    checkpoint = checkpoint_tensors(shard_headers)
    pairs, formats = resolve_pairs(
        args.lora, lora_hdr, lora_base, checkpoint
    )
    summary = ", ".join(f"{name}={count}" for name, count in sorted(formats.items()))
    print(f"{len(pairs)} adapter pairs ({summary})")
    if args.dry_run:
        for shard, header in shard_headers:
            matched = sum(name in pairs for name in header)
            if matched:
                print(f"{shard.name}: {matched} matched tensors")
        print("dry run: all adapter pairs match the checkpoint")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
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
                a_key, b_key, pair_scale = pairs[key]
                A = load_tensor(args.lora, lora_hdr[a_key], lora_base)
                B = load_tensor(args.lora, lora_hdr[b_key], lora_base)
                W = load_tensor(dst, info, base)
                # NumPy 2.0 on Apple Accelerate can emit overflow/invalid
                # RuntimeWarnings for these large, finite GEMMs even when the
                # resulting array contains no NaN/Inf. Suppress the stale FP
                # status flags, then validate every result explicitly.
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    update = B.astype(np.float32) @ A.astype(np.float32)
                    effective_scale = args.scale * pair_scale
                    folded = W + effective_scale * update
                require_finite(f"{key} update", update)
                require_finite(f"{key} folded weight", folded)
                # parity check on a random probe before committing bytes
                x = np.random.default_rng(0).standard_normal(W.shape[1]).astype(np.float32)
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    direct = folded @ x
                    composed = W @ x + effective_scale * (B @ (A @ x))
                    rel = (np.max(np.abs(direct - composed)) /
                           (np.max(np.abs(composed)) + 1e-9))
                require_finite(f"{key} direct parity probe", direct)
                require_finite(f"{key} composed parity probe", composed)
                if not np.isfinite(rel) or rel > 1e-4:
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
