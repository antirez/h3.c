#!/usr/bin/env python3
"""Fold LoRA adapters into the bf16 MiniMax-H3 checkpoint by in-place byte patching.

h3.c has no LoRA runtime, and does not need one: a LoRA is W' = W + scale*(B@A),
which can be baked into the checkpoint once, offline. This tool clones the
original shards (copy-on-write where the filesystem supports it) and rewrites
only the byte ranges of the targeted tensors, so headers, tensor order, and
alignment stay byte-identical to the originals.

Several adapters can be folded in one pass, each with its own strength; every
targeted tensor is read once from the base and written once with the sum of all
deltas. The effective scale of an adapter is strength * alpha / rank, where
alpha comes from a per-pair `.alpha` tensor (ComfyUI / kohya files) and
defaults to rank (scale 1) when absent.

Accepted naming, all resolved against the checkpoint's own key space:
  <target>.lora_A.weight / .lora_B.weight             (native layout)
  diffusion_model.<target>.lora_A / .lora_B + .alpha   (ComfyUI, e.g. lightx2v distill LoRAs)
  lora_unet_<target with _>.lora_down / .lora_up + .alpha  (kohya / musubi)
Diffusers-layout files (to_q/to_k/to_v, ff.net.*) use split q/k/v and a
[value;gate] fc1 order and are not supported; use the ComfyUI conversion of
those adapters instead.

Pairs whose A matrix takes the 8-dim time curve (blocks.N.adaln_proj.linear)
instead of the adaln input cannot be folded into the weight, which means the
adapter was trained against a low-rank-AdaLN base. The tool aborts on them.
Pass --skip-curve-pairs to fold the rest and leave those targets unchanged.

Output is written to `<out>.tmp` and renamed to `<out>` only on success. All
targets are validated (BF16 dtype, shape) before anything is written. Each
written tensor is read back and compared with W + dW, computed in float64,
within BF16 rounding tolerance.

Requires only numpy. Example, folding the public lightx2v MiniMax-H3 distill
LoRA (ComfyUI naming) into the Ref2VA transformer; the adapter filename is a
placeholder for the file you downloaded:
  python3 tools/fold_turbo_lora.py \
      --checkpoint MiniMax-H3/Ref2VA/transformer \
      --lora distill_comfyui.safetensors:1.0 \
      --out MiniMax-H3-distill/Ref2VA/transformer
Then point h3 at a model directory whose transformer is the folded tree.
Do not combine distilled schedules with --reuse or --core-reuse.
"""
import argparse
import json
import os
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


def bf16_to_f32(raw, shape):
    u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
    return (u16 << 16).view(np.float32).reshape(shape)


def load_tensor(path, info, base):
    start, end = info["data_offsets"]
    dtype = info["dtype"]
    with open(path, "rb") as f:
        f.seek(base + start)
        raw = f.read(end - start)
    if dtype == "BF16":
        return bf16_to_f32(raw, info["shape"])
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(info["shape"])
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(info["shape"]).copy()
    raise SystemExit(f"unsupported dtype {dtype} for {path}")


def f32_to_bf16_bytes(x):
    xf = np.ascontiguousarray(x, dtype=np.float32)
    u32 = xf.view(np.uint32)
    rounded = (u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16  # round to nearest even
    nan_bits = ((u32 >> 16) & 0x8000) | 0x7FC0  # quiet NaN, sign preserved
    rounded = np.where(np.isnan(xf), nan_bits, rounded)
    return rounded.astype("<u2").tobytes()


def clone(src, dst):
    # /bin/cp: a GNU cp earlier on PATH has no -c and would silently fall back to a full copy.
    if subprocess.run(["/bin/cp", "-c", str(src), str(dst)],
                      capture_output=True).returncode != 0:
        shutil.copy2(src, dst)


SUFFIXES = ((".lora_A.weight", ".lora_B.weight"),
            (".lora_down.weight", ".lora_up.weight"))
PREFIXES = ("diffusion_model.", "transformer.", "model.")
CURVE_TARGET = "adaln_proj.linear.weight"
CURVE_DIM = 8


def resolve_target(stem, targets, flattened):
    for prefix in PREFIXES:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    if stem + ".weight" in targets:
        return stem + ".weight"
    if stem.startswith("lora_unet_"):
        return flattened.get(stem[len("lora_unet_"):])
    return None


def load_adapter(path, strength, targets, skip_curve_pairs):
    """Return {checkpoint key: (A, B, scale)} for one adapter file.

    targets maps each checkpoint key to its shape."""
    hlen, header = read_header(path)
    base = 8 + hlen
    flattened = {k[: -len(".weight")].replace(".", "_"): k for k in targets}
    pairs, unmatched, curves = {}, [], 0
    for key in header:
        for a_suffix, b_suffix in SUFFIXES:
            if not key.endswith(a_suffix):
                continue
            stem = key[: -len(a_suffix)]
            target = resolve_target(stem, targets, flattened)
            if target is None:
                unmatched.append(stem)
                continue
            if target.endswith(CURVE_TARGET) and header[key]["shape"][1] == CURVE_DIM:
                curves += 1
                continue
            A = load_tensor(path, header[key], base)
            B = load_tensor(path, header[stem + b_suffix], base)
            if A.ndim != 2 or B.ndim != 2 or B.shape[1] != A.shape[0] \
                    or [B.shape[0], A.shape[1]] != list(targets[target]):
                raise SystemExit(f"{path.name}: {stem}: B{list(B.shape)} @ A{list(A.shape)} "
                                 f"does not match {target} {list(targets[target])}")
            rank = A.shape[0]
            alpha_info = header.get(stem + ".alpha")
            alpha = float(load_tensor(path, alpha_info, base).reshape(-1)[0]) \
                if alpha_info else float(rank)
            pairs[target] = (A, B, strength * alpha / rank)
    if unmatched:
        raise SystemExit(f"{path.name}: {len(unmatched)} pairs match no checkpoint "
                         f"tensor (first: {unmatched[0]}) - diffusers layout?")
    if curves and not skip_curve_pairs:
        raise SystemExit(f"{path.name}: {curves} adaln_proj curve pairs (adapter trained "
                         f"against a low-rank-AdaLN base, not foldable); "
                         f"pass --skip-curve-pairs to fold the rest")
    if curves:
        print(f"{path.name}: skipped {curves} adaln_proj curve pairs (not foldable)")
    if not pairs:
        raise SystemExit(f"{path.name}: no LoRA pairs found")
    scales = sorted({round(s, 6) for _, _, s in pairs.values()})
    print(f"{path.name}: {len(pairs)} pairs, strength {strength}, "
          f"effective scale {scales[0]}..{scales[-1]}")
    return pairs


def parse_lora(spec):
    path, _, tail = spec.rpartition(":")
    try:
        return Path(path), float(tail)
    except ValueError:
        return Path(spec), 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="directory of original bf16 transformer shards")
    ap.add_argument("--lora", required=True, action="append",
                    help="adapter .safetensors[:strength], repeatable")
    ap.add_argument("--out", required=True, type=Path,
                    help="output directory for the folded shards (must not exist)")
    ap.add_argument("--skip-curve-pairs", action="store_true",
                    help="fold the adapter without its adaln_proj curve pairs "
                         "(default: abort when they are present)")
    args = ap.parse_args()

    if args.out.exists():
        raise SystemExit(f"{args.out} already exists")
    shards = sorted(args.checkpoint.glob("model-*.safetensors"))
    if not shards:
        raise SystemExit(f"no shards in {args.checkpoint}")
    targets = {}
    for shard in shards:
        targets.update({k: v["shape"] for k, v in read_header(shard)[1].items()})

    deltas = {}
    for spec in args.lora:
        path, strength = parse_lora(spec)
        for key, pair in load_adapter(path, strength, targets, args.skip_curve_pairs).items():
            deltas.setdefault(key, []).append(pair)

    # Validate every target before writing anything.
    found = 0
    for shard in shards:
        for key, info in read_header(shard)[1].items():
            if key not in deltas:
                continue
            found += 1
            if info["dtype"] != "BF16":
                raise SystemExit(f"{key} in {shard.name}: expected BF16, got {info['dtype']}; "
                                 f"nothing written")
            if list(info["shape"]) != list(targets[key]):
                raise SystemExit(f"{key} in {shard.name}: shape {info['shape']} differs from "
                                 f"{targets[key]}; nothing written")
    if found != len(deltas):
        raise SystemExit(f"only {found} of {len(deltas)} targets exist in the checkpoint; "
                         f"nothing written")

    tmp = args.out.with_name(args.out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        for extra in ("config.json", "model.safetensors.index.json"):
            src = args.checkpoint / extra
            if src.exists():
                shutil.copy2(src, tmp / extra)

        patched = 0
        for shard in shards:
            dst = tmp / shard.name
            clone(shard, dst)
            hlen, hdr = read_header(dst)
            base = 8 + hlen
            todo = [k for k in hdr if k in deltas]
            if not todo:
                continue
            with open(dst, "r+b") as f:
                for key in todo:
                    info = hdr[key]
                    W = load_tensor(shard, info, base)
                    folded = W.copy()
                    expected = W.astype(np.float64)
                    for A, B, scale in deltas[key]:
                        folded += scale * (B @ A)
                        expected += scale * (B.astype(np.float64) @ A.astype(np.float64))
                    buf = f32_to_bf16_bytes(folded)
                    start, end = info["data_offsets"]
                    if len(buf) != end - start:
                        raise SystemExit(f"{key}: byte count mismatch")
                    f.seek(base + start)
                    f.write(buf)
                    f.flush()
                    f.seek(base + start)
                    readback = bf16_to_f32(f.read(end - start), info["shape"]).astype(np.float64)
                    tol = 2.0 ** -8 * np.abs(expected) + np.finfo(np.float32).tiny
                    ok = (np.abs(readback - expected) <= tol) | \
                        (np.isnan(readback) & np.isnan(expected))
                    if not ok.all():
                        raise SystemExit(f"{key}: re-read BF16 tensor differs from W+dW "
                                         f"at {int((~ok).sum())} elements")
                    patched += 1
            print(f"{shard.name}: cumulative {patched}/{len(deltas)}")

        if patched != len(deltas):
            raise SystemExit(f"only {patched} of {len(deltas)} targets were patched")
        os.rename(tmp, args.out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print("done")


if __name__ == "__main__":
    sys.exit(main())
