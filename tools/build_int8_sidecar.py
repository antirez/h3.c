#!/usr/bin/env python3
"""Build row-wise INT8 H3 transformer weights for a future direct loader.

The output mirrors the online Metal quantizer: every selected BF16 matrix gets
an I8 tensor with the original name and an F32 `<name>.h3_scale` vector. The
tool streams row batches, so it does not materialize a full H3 matrix as F32.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
from pathlib import Path

import numpy as np


FORMAT = "h3-int8-sidecar-v1"
TARGET = re.compile(
    r"^blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))\.weight$"
)


def read_header(path):
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        header_size = struct.unpack("<Q", raw)[0]
        header = json.loads(stream.read(header_size))
    header.pop("__metadata__", None)
    return header_size, header


def bf16_to_f32(values):
    bits = np.asarray(values, dtype="<u2").astype(np.uint32) << 16
    return bits.view(np.float32)


def quantize_bf16_rows(values):
    """Match h3_quantize_bf16_int8_rows for a two-dimensional BF16 array."""
    f32 = bf16_to_f32(values)
    maximum = np.max(np.abs(f32), axis=1).astype(np.float32)
    positive = maximum > 0
    scales = np.full(maximum.shape, np.float32(1.0 / 127.0), dtype=np.float32)
    np.divide(maximum, np.float32(127.0), out=scales, where=positive)
    inverse = np.full(maximum.shape, np.float32(127.0), dtype=np.float32)
    np.divide(np.float32(127.0), maximum, out=inverse, where=positive)
    scales = scales.astype("<f4", copy=False)
    quantized = np.clip(
        np.rint(f32 * inverse[:, None]), -127, 127
    ).astype(np.int8)
    return quantized, scales


def padded_header(entries):
    encoded = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return encoded


def sidecar_entries(selected):
    entries = {}
    offset = 0
    for name, info in selected:
        rows, columns = info["shape"]
        weight_bytes = rows * columns
        entries[name] = {
            "dtype": "I8",
            "shape": [rows, columns],
            "data_offsets": [offset, offset + weight_bytes],
        }
        offset += weight_bytes
        scale_bytes = rows * 4
        entries[f"{name}.h3_scale"] = {
            "dtype": "F32",
            "shape": [rows],
            "data_offsets": [offset, offset + scale_bytes],
        }
        offset += scale_bytes
    return entries, offset


def quantize_shard(source, destination, batch_rows):
    header_size, header = read_header(source)
    selected = sorted(
        (name, info) for name, info in header.items() if TARGET.match(name)
    )
    for name, info in selected:
        if info.get("dtype") != "BF16" or len(info.get("shape", [])) != 2:
            raise ValueError(f"{name}: expected a two-dimensional BF16 tensor")
    if not selected:
        return []
    entries, payload_size = sidecar_entries(selected)
    encoded = padded_header(entries)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        output.truncate(8 + len(encoded) + payload_size)

    source_base = 8 + header_size
    output_base = 8 + len(encoded)
    with destination.open("r+b") as output:
        for name, info in selected:
            rows, columns = info["shape"]
            start, end = info["data_offsets"]
            expected = rows * columns * 2
            if end - start != expected:
                raise ValueError(f"{name}: BF16 payload size does not match shape")
            source_values = np.memmap(
                source, dtype="<u2", mode="r", offset=source_base + start,
                shape=(rows, columns), order="C",
            )
            weight_offset = entries[name]["data_offsets"][0]
            scale_offset = entries[f"{name}.h3_scale"]["data_offsets"][0]
            for row_start in range(0, rows, batch_rows):
                row_stop = min(rows, row_start + batch_rows)
                quantized, scales = quantize_bf16_rows(
                    source_values[row_start:row_stop]
                )
                output.seek(output_base + weight_offset + row_start * columns)
                output.write(quantized.tobytes(order="C"))
                output.seek(output_base + scale_offset + row_start * 4)
                output.write(scales.tobytes(order="C"))
            del source_values
    return [name for name, _ in selected]


def source_fingerprint(path):
    header_size, _ = read_header(path)
    with path.open("rb") as stream:
        header_prefix = stream.read(8 + header_size)
    status = path.stat()
    return {
        "name": path.name,
        "size": status.st_size,
        "mtime_ns": status.st_mtime_ns,
        "header_sha256": hashlib.sha256(header_prefix).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-rows", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_rows < 1:
        parser.error("--batch-rows must be positive")
    shards = sorted(args.checkpoint.glob("*.safetensors"))
    if not shards:
        parser.error(f"no safetensors shards in {args.checkpoint}")

    plan = []
    estimated_bytes = 0
    for shard in shards:
        _, header = read_header(shard)
        selected = sorted(
            (name, info) for name, info in header.items() if TARGET.match(name)
        )
        if not selected:
            continue
        entries, payload_size = sidecar_entries(selected)
        estimated_bytes += payload_size + len(padded_header(entries)) + 8
        plan.append((shard, selected))
    if not plan:
        parser.error("no H3 transformer matrices matched the sidecar schema")
    print(f"{sum(len(items) for _, items in plan)} tensors, "
          f"{estimated_bytes / 1024**3:.2f} GiB output")
    if args.dry_run:
        for shard, selected in plan:
            print(f"{shard.name}: {len(selected)} tensors")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    available = shutil.disk_usage(args.out).free
    if available < int(estimated_bytes * 1.05):
        parser.error("insufficient free space for INT8 sidecar")
    manifest = {
        "format": FORMAT,
        "source": str(args.checkpoint.resolve()),
        "sources": [],
        "tensors": [],
    }
    for index, (shard, _) in enumerate(plan, 1):
        destination = args.out / f"int8-{index:05d}-of-{len(plan):05d}.safetensors"
        names = quantize_shard(shard, destination, args.batch_rows)
        manifest["sources"].append(source_fingerprint(shard))
        manifest["tensors"].extend(names)
        print(f"{shard.name}: {len(names)} tensors -> {destination.name}", flush=True)
    temporary = args.out / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.out / "manifest.json")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
