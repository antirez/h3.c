#!/usr/bin/env python3
"""Convert an alibaba-pai MiniMax-H3 PDD adapter into h3-native artifacts.

The PDD (Parallel Decoding Distillation) release stores Diffusers-style
module paths with bare `.lora_down`/`.lora_up` keys, separate Q/K/V
projections, and 32 per-interval replacement output heads. h3.c consumes
the released checkpoint layout instead: fused, per-head interleaved QKV
rows ([Q 128 | K 128 | V 128] per head), fused FC1, and one final head.

This tool emits two files:

1. `<out>.lora.safetensors` - native `<checkpoint-tensor>.lora_A/B.weight`
   pairs for tools/fold_turbo_lora.py. Q/K/V pairs are fused into one
   rank-192 pair whose B rows follow the checkpoint's per-head interleave,
   so the folded delta is exactly `concat(Bq@Aq, Bk@Ak, Bv@Av)` per head.
   The PDD alpha/rank scale (1.0 for the release) is baked into A.

2. `<out>.heads.safetensors` - eight premixed per-step final heads for
   video and audio. PDD arms its 32 interval heads with a per-block plan
   (step sizes over the block span) before every forward; at the released
   32/4 grid an 8-evaluation run uses eight fixed mixtures, and h3's
   serving schedule evaluates the same shifted-sigma boundaries, so the
   mixtures can be precomputed offline. Weights are emitted in F32 like
   the checkpoint's final_layer tensors, with `.stepK` name suffixes.

Usage:
  python3 tools/convert_pdd_lora.py \
      --adapter MiniMax-H3-Ref2VA-Acc-8Step.safetensors --out pdd_ref2va
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

HEADS = 56
HEAD_DIM = 128
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0

MODULE_MAP = {
    "attn.to_out.0": "attn.out_proj",
    "ff.net.0.proj": "mlp.fc1",
    "ff.net.2": "mlp.fc2",
    "adaln_proj.linear": "adaln_proj.linear",
}


def read_safetensors(path):
    with open(path, "rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        payload = stream.read()
    metadata = header.pop("__metadata__", {})
    tensors = {}
    for name, info in header.items():
        start, stop = info["data_offsets"]
        raw = payload[start:stop]
        if info["dtype"] == "BF16":
            bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
            values = bits.view(np.float32).astype(np.float32)
        elif info["dtype"] == "F32":
            values = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        else:
            raise ValueError(f"{name}: unsupported dtype {info['dtype']}")
        tensors[name] = values.reshape(info["shape"])
    return tensors, metadata


def write_safetensors(path, tensors, metadata=None):
    entries = {}
    blobs = []
    offset = 0
    for name, value in tensors.items():
        raw = np.ascontiguousarray(value, dtype="<f4").tobytes()
        entries[name] = {"dtype": "F32", "shape": list(value.shape),
                         "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    if metadata:
        entries["__metadata__"] = {k: str(v) for k, v in metadata.items()}
    encoded = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for raw in blobs:
            stream.write(raw)


def checkpoint_stem(module):
    """Map one adapter module path to the checkpoint tensor stem."""
    match = re.match(r"^transformer_blocks\.(\d+)\.(.+)$", module)
    if match:
        block, tail = match.groups()
        prefix = f"blocks.{block}"
    else:
        match = re.match(r"^token_refiner\.refiner_blocks\.(\d+)\.(.+)$",
                         module)
        if not match:
            return None
        block, tail = match.groups()
        prefix = f"token_refiner.blocks.{block}"
    mapped = MODULE_MAP.get(tail)
    return f"{prefix}.{mapped}" if mapped else None


def fuse_qkv(down_up, module_prefix):
    """Build the rank-192 fused-QKV pair with per-head interleaved B rows."""
    parts = []
    for name in ("to_q", "to_k", "to_v"):
        down = down_up.get(f"{module_prefix}.attn.{name}.lora_down")
        up = down_up.get(f"{module_prefix}.attn.{name}.lora_up")
        if down is None or up is None:
            return None, None
        parts.append((down, up))
    rank = parts[0][0].shape[0]
    hidden = parts[0][0].shape[1]
    inner = parts[0][1].shape[0]
    if inner != HEADS * HEAD_DIM:
        raise ValueError(f"{module_prefix}: unexpected inner width {inner}")
    fused_a = np.vstack([down for down, _ in parts])
    fused_b = np.zeros((3 * inner, 3 * rank), dtype=np.float32)
    for head in range(HEADS):
        row = head * 3 * HEAD_DIM
        source = head * HEAD_DIM
        for lane, (_, up) in enumerate(parts):
            fused_b[row + lane * HEAD_DIM:row + (lane + 1) * HEAD_DIM,
                    lane * rank:(lane + 1) * rank] = \
                up[source:source + HEAD_DIM]
    return fused_a, fused_b


def pdd_step_sizes(shift, num_steps):
    sigma = np.linspace(1.0, 0.0, num_steps + 1, dtype=np.float64)
    shifted = shift * sigma / (1.0 + (shift - 1.0) * sigma)
    grid = 1.0 - shifted
    return np.diff(grid)


def premix_heads(weight, bias, shift, block_size):
    """Collapse the 32 interval heads into one head per evaluation block."""
    num_steps = weight.shape[0]
    sizes = pdd_step_sizes(shift, num_steps)
    mixed_w, mixed_b = [], []
    for start in range(0, num_steps, block_size):
        plan = sizes[start:start + block_size]
        plan = plan / plan.sum()
        mixed_w.append(np.einsum("n,noi->oi", plan,
                                 weight[start:start + block_size]))
        mixed_b.append(np.einsum("n,no->o", plan,
                                 bias[start:start + block_size]))
    return mixed_w, mixed_b


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True,
                        help="output prefix (writes <out>.lora.safetensors "
                             "and <out>.heads.safetensors)")
    args = parser.parse_args()

    tensors, metadata = read_safetensors(args.adapter)
    rank = int(metadata.get("lora_rank", 64))
    alpha = float(metadata.get("lora_alpha", rank))
    num_steps = int(metadata.get("pdd_num_steps", 32))
    block_size = int(metadata.get("pdd_block_size", 4))
    scale = alpha / rank
    print(f"adapter: rank={rank} alpha={alpha} scale={scale} "
          f"grid={num_steps}/{block_size} tensors={len(tensors)}")

    modules = {}
    for key in tensors:
        if key.endswith(".lora_down") or key.endswith(".lora_up"):
            modules[key.rsplit(".", 1)[0]] = True

    native = {}
    fused_blocks = set()
    skipped = []
    for module in sorted(modules):
        if re.search(r"\.attn\.to_[qkv]$", module):
            prefix = module.rsplit(".attn.", 1)[0]
            if prefix in fused_blocks:
                continue
            fused_blocks.add(prefix)
            fused_a, fused_b = fuse_qkv(tensors, prefix)
            stem = checkpoint_stem(f"{prefix}.attn.to_out.0")
            if fused_a is None or stem is None:
                skipped.append(module)
                continue
            qkv_stem = stem.rsplit(".", 1)[0] + ".qkv_proj"
            native[f"{qkv_stem}.lora_A.weight"] = fused_a * scale
            native[f"{qkv_stem}.lora_B.weight"] = fused_b
            continue
        stem = checkpoint_stem(module)
        if stem is None:
            skipped.append(module)
            continue
        native[f"{stem}.lora_A.weight"] = \
            tensors[f"{module}.lora_down"] * scale
        native[f"{stem}.lora_B.weight"] = tensors[f"{module}.lora_up"]
    if skipped:
        print(f"warning: skipped {len(skipped)} unmapped modules, e.g. "
              f"{skipped[:3]}", file=sys.stderr)

    heads = {}
    for source, target, shift in (
            ("proj_out", "final_layer.video_out", VIDEO_SHIFT),
            ("audio_proj_out", "final_layer.audio_out", AUDIO_SHIFT)):
        mixed_w, mixed_b = premix_heads(
            tensors[f"{source}.weight"], tensors[f"{source}.bias"],
            shift, block_size)
        for step, (w, b) in enumerate(zip(mixed_w, mixed_b)):
            heads[f"{target}.weight.step{step}"] = w
            heads[f"{target}.bias.step{step}"] = b

    lora_path = Path(f"{args.out}.lora.safetensors")
    heads_path = Path(f"{args.out}.heads.safetensors")
    write_safetensors(lora_path, native,
                      {"source": Path(args.adapter).name,
                       "format": "h3-native-lora"})
    write_safetensors(heads_path, heads,
                      {"source": Path(args.adapter).name,
                       "format": "h3-pdd-heads-v1",
                       "evaluations": num_steps // block_size,
                       "video_shift": VIDEO_SHIFT,
                       "audio_shift": AUDIO_SHIFT})
    evaluations = num_steps // block_size
    print(f"wrote {lora_path} ({len(native)} tensors) and {heads_path} "
          f"({len(heads)} tensors, {evaluations} evaluations)")


if __name__ == "__main__":
    main()
