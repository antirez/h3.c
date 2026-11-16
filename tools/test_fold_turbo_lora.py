#!/usr/bin/env python3
"""Tests for fold_turbo_lora.py on tiny synthetic safetensors.

  python3 tools/test_fold_turbo_lora.py
Needs numpy.
"""
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

FOLD = Path(__file__).with_name("fold_turbo_lora.py")
BLOCKS = 50
ADALN = [12, 16]
QKV = [24, 8]


def bf16_bytes(x):
    u32 = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype("<u2").tobytes()


def from_bf16(raw, shape):
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)


def write_st(path, tensors):
    """tensors: {name: (dtype, ndarray)} with dtype BF16, F16 or F32."""
    header, blobs, off = {}, [], 0
    for name, (dtype, arr) in tensors.items():
        raw = bf16_bytes(arr) if dtype == "BF16" else \
            np.ascontiguousarray(arr, dtype={"F16": "<f2", "F32": "<f4"}[dtype]).tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape),
                        "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hdr = json.dumps(header).encode()
    hdr += b" " * (-len(hdr) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)) + hdr + b"".join(blobs))


def read_st(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
        data = f.read()
    return {k: from_bf16(data[v["data_offsets"][0]:v["data_offsets"][1]], v["shape"])
            for k, v in header.items()}


class FoldTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.rng = np.random.default_rng(1)
        self.ckpt = self.tmp / "ckpt"
        self.ckpt.mkdir()
        self.base = {}
        for shard, blocks in ((1, range(0, 25)), (2, range(25, BLOCKS))):
            tensors = {}
            for n in blocks:
                for name, shape in ((f"blocks.{n}.adaln_proj.linear.weight", ADALN),
                                    (f"blocks.{n}.attn.qkv.weight", QKV)):
                    w = self.rng.standard_normal(shape).astype(np.float32)
                    tensors[name] = ("BF16", w)
                    self.base[name] = from_bf16(bf16_bytes(w), shape)
            write_st(self.ckpt / f"model-0000{shard}-of-00002.safetensors", tensors)

    def mat(self, *shape):
        return (0.1 * self.rng.standard_normal(shape)).astype(np.float16).astype(np.float32)

    def comfy_like(self, curves, bad_qkv=False):
        """ComfyUI-style adapter with time-curve pairs on every block."""
        t = {}
        for n in range(curves):
            p = f"diffusion_model.blocks.{n}.adaln_proj.linear"
            t[p + ".lora_A.weight"] = ("F16", self.mat(4, 8))
            t[p + ".lora_B.weight"] = ("F16", self.mat(ADALN[0], 4))
        for n in range(0, BLOCKS, 7):
            p = f"diffusion_model.blocks.{n}.attn.qkv"
            t[p + ".lora_A.weight"] = ("F16", self.mat(4, QKV[1]))
            t[p + ".lora_B.weight"] = ("F16", self.mat(QKV[0] - (4 if bad_qkv else 0), 4))
        return t

    def kohya_like(self):
        t = {}
        for n in range(0, BLOCKS, 3):
            p = f"lora_unet_blocks_{n}_attn_qkv"
            t[p + ".lora_down.weight"] = ("F16", self.mat(32, QKV[1]))
            t[p + ".lora_up.weight"] = ("F16", self.mat(QKV[0], 32))
            t[p + ".alpha"] = ("F32", np.array(16.0, dtype=np.float32))
            p = f"lora_unet_blocks_{n}_adaln_proj_linear"
            t[p + ".lora_down.weight"] = ("F16", self.mat(4, ADALN[1]))
            t[p + ".lora_up.weight"] = ("F16", self.mat(ADALN[0], 4))
            t[p + ".alpha"] = ("F32", np.array(4.0, dtype=np.float32))
        return t

    def fold(self, adapter, out="out"):
        lora = self.tmp / "lora.safetensors"
        write_st(lora, adapter)
        r = subprocess.run([sys.executable, "-I", str(FOLD), "--checkpoint", str(self.ckpt),
                            "--lora", f"{lora}:0.75", "--out", str(self.tmp / out)],
                           capture_output=True, text=True)
        return r, self.tmp / out

    def folded(self, out):
        merged = {}
        for shard in sorted(out.glob("model-*.safetensors")):
            merged.update(read_st(shard))
        return merged

    def expected(self, adapter, a_suffix, b_suffix, key_of):
        exp = dict(self.base)
        for k, (_, A) in adapter.items():
            if not k.endswith(a_suffix):
                continue
            stem = k[: -len(a_suffix)]
            key = key_of(stem)
            if key.endswith("adaln_proj.linear.weight") and A.shape[1] == 8:
                continue
            B = adapter[stem + b_suffix][1]
            alpha = float(adapter[stem + ".alpha"][1]) if stem + ".alpha" in adapter \
                else float(A.shape[0])
            w = exp[key].copy()
            w += 0.75 * alpha / A.shape[0] * (B @ A)
            exp[key] = from_bf16(bf16_bytes(w), w.shape)
        return exp

    def assert_folded(self, out, exp):
        got = self.folded(out)
        self.assertEqual(set(got), set(exp))
        for k in exp:
            np.testing.assert_array_equal(got[k], exp[k], err_msg=k)

    def test_a_curve_pairs_skipped_rest_folded(self):
        adapter = self.comfy_like(BLOCKS)
        r, out = self.fold(adapter)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("skipped 50 adaln_proj curve pairs", r.stdout)
        self.assertIn("8 pairs", r.stdout)
        exp = self.expected(adapter, ".lora_A.weight", ".lora_B.weight",
                            lambda s: s[len("diffusion_model."):] + ".weight")
        self.assert_folded(out, exp)
        got = self.folded(out)
        for n in range(BLOCKS):
            k = f"blocks.{n}.adaln_proj.linear.weight"
            np.testing.assert_array_equal(got[k], self.base[k])

    def test_b_shape_mismatch_aborts(self):
        r, _ = self.fold(self.comfy_like(BLOCKS, bad_qkv=True))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not match blocks.0.attn.qkv.weight [24, 8]", r.stderr)

    def test_c_kohya_no_curve_pairs(self):
        adapter = self.kohya_like()
        r, out = self.fold(adapter)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("skipped", r.stdout)
        exp = self.expected(adapter, ".lora_down.weight", ".lora_up.weight",
                            lambda s: s[len("lora_unet_"):].replace("_attn_qkv", ".attn.qkv")
                            .replace("_adaln_proj_linear", ".adaln_proj.linear")
                            .replace("blocks_", "blocks.") + ".weight")
        self.assert_folded(out, exp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
