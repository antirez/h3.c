#!/usr/bin/env python3
"""Page-align the safetensors files of a packed INT8 DiT (<variant>/transformer-int8) for zero-copy loading.

Each file is rewritten as an ordinary safetensors file (tensors back to back, header padded to a page) with the
tensors whose size is a whole number of 16 KiB pages first, so every one of them starts on a page and h3 can map
it straight from the page cache (H3_ZERO_COPY_WEIGHTS). The small rest (per-row scales, norms) follows and is
copied at load time as before. Every tensor is compared byte for byte with the original before the original is
replaced; a file that is already aligned is left alone.

Usage: tools/align_int8_pack.py DIR [DIR...]
"""

import json
import os
import struct
import sys

PAGE = 16384


def read_header(path):
    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(length)), 8 + length


def size(spec):
    return spec["data_offsets"][1] - spec["data_offsets"][0]


def aligned(path):
    header, start = read_header(path)
    specs = [s for k, s in header.items() if k != "__metadata__"]
    spans = sorted(s["data_offsets"] for s in specs)
    contiguous = spans[0][0] == 0 and all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    pages = all((start + s["data_offsets"][0]) % PAGE == 0 for s in specs if size(s) % PAGE == 0)
    return start % PAGE == 0 and contiguous and pages


def align(path):
    if aligned(path):
        print(f"{path}: already aligned")
        return
    header, start = read_header(path)
    metadata = header.pop("__metadata__", None)
    keys = list(header)
    keys = [k for k in keys if size(header[k]) % PAGE == 0] + [k for k in keys if size(header[k]) % PAGE]
    out, offset = {}, 0
    for key in keys:
        spec = header[key]
        out[key] = {"dtype": spec["dtype"], "shape": spec["shape"], "data_offsets": [offset, offset + size(spec)]}
        offset += size(spec)
    if metadata is not None:
        out = {"__metadata__": metadata, **out}
    blob = json.dumps(out, separators=(",", ":")).encode()
    blob += b" " * ((-(8 + len(blob))) % PAGE)
    data_start = 8 + len(blob)
    target = path + ".aligning"
    with open(path, "rb") as src, open(target, "wb") as dst:
        dst.write(struct.pack("<Q", len(blob)) + blob)
        for key in keys:
            a, b = header[key]["data_offsets"]
            src.seek(start + a)
            remaining = b - a
            while remaining:
                chunk = src.read(min(remaining, 64 << 20))
                dst.write(chunk)
                remaining -= len(chunk)
    new_header, new_start = read_header(target)
    with open(path, "rb") as src, open(target, "rb") as dst:
        for key in keys:
            a, b = header[key]["data_offsets"]
            na, nb = new_header[key]["data_offsets"]
            src.seek(start + a)
            dst.seek(new_start + na)
            remaining = b - a
            while remaining:
                n = min(remaining, 64 << 20)
                if src.read(n) != dst.read(n):
                    os.unlink(target)
                    sys.exit(f"{path}: verification failed for {key}")
                remaining -= n
    if not aligned(target):
        os.unlink(target)
        sys.exit(f"{path}: result is not aligned")
    os.replace(target, path)
    mapped = sum(1 for k in keys if size(header[k]) % PAGE == 0)
    print(f"{path}: {len(keys)} tensors, {mapped} page-aligned for zero-copy")


if __name__ == "__main__":
    for directory in sys.argv[1:]:
        for name in sorted(os.listdir(directory)):
            if name.endswith(".safetensors"):
                align(os.path.join(directory, name))
