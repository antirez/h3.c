import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


def write_safetensors(path, tensors):
    header = {}
    payload = bytearray()
    for name, (dtype, shape, data) in tensors.items():
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def bf16_bytes(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype("<u2").tobytes()


def f16_bytes(values):
    return np.asarray(values, dtype="<f2").tobytes()


def read_bf16(path, name):
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        start, end = header[name]["data_offsets"]
        stream.seek(8 + header_size + start)
        values = np.frombuffer(stream.read(end - start), dtype="<u2").astype(np.uint32)
    return (values << 16).view(np.float32).reshape(header[name]["shape"])


class FoldTurboLoraTests(unittest.TestCase):
    @staticmethod
    def run_tool(checkpoint, lora, output):
        tool = Path(__file__).parents[1] / "tools/fold_turbo_lora.py"
        return subprocess.run(
            [
                sys.executable, str(tool),
                "--checkpoint", str(checkpoint),
                "--lora", str(lora),
                "--out", str(output),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_folds_matching_pair_and_copies_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            checkpoint = temp / "checkpoint"
            output = temp / "output"
            checkpoint.mkdir()
            weight = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
            down = np.array([[0.5, 1.0, -0.5]], dtype=np.float32)
            up = np.array([[2.0], [-1.0]], dtype=np.float32)
            write_safetensors(
                checkpoint / "model-00001-of-00001.safetensors",
                {"blocks.0.weight": ("BF16", weight.shape, bf16_bytes(weight))},
            )
            write_safetensors(
                temp / "turbo.safetensors",
                {
                    "blocks.0.lora_A.weight": ("F32", down.shape, down.tobytes()),
                    "blocks.0.lora_B.weight": ("F32", up.shape, up.tobytes()),
                },
            )
            (checkpoint / "config.json").write_text('{"test":true}')
            (checkpoint / "model.safetensors.index.json").write_text("{}")

            result = self.run_tool(checkpoint, temp / "turbo.safetensors", output)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = read_bf16(
                output / "model-00001-of-00001.safetensors", "blocks.0.weight"
            )
            expected = read_bf16(
                checkpoint / "model-00001-of-00001.safetensors", "blocks.0.weight"
            ) + up @ down
            np.testing.assert_allclose(actual, expected, rtol=0.01, atol=0.02)
            self.assertTrue((output / "config.json").is_file())
            self.assertTrue((output / "model.safetensors.index.json").is_file())

    def test_diffusion_model_prefix_and_f16_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            checkpoint = temp / "checkpoint"
            output = temp / "output"
            checkpoint.mkdir()
            target = "token_refiner.blocks.0.attn.out_proj.weight"
            weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            down = np.array([[0.5, -0.25]], dtype=np.float32)
            up = np.array([[2.0], [1.0]], dtype=np.float32)
            write_safetensors(
                checkpoint / "model-00001-of-00001.safetensors",
                {target: ("BF16", weight.shape, bf16_bytes(weight))},
            )
            stem = "diffusion_model.token_refiner.blocks.0.attn.out_proj"
            write_safetensors(
                temp / "diffusers.safetensors",
                {
                    stem + ".lora_A.weight": ("F16", down.shape, f16_bytes(down)),
                    stem + ".lora_B.weight": ("F16", up.shape, f16_bytes(up)),
                },
            )
            result = self.run_tool(
                checkpoint, temp / "diffusers.safetensors", output
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = read_bf16(output / "model-00001-of-00001.safetensors", target)
            np.testing.assert_allclose(actual, weight + up @ down, atol=0.02)
            self.assertIn("diffusion_model=1", result.stdout)

    def test_kohya_down_up_mapping_applies_alpha_over_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            checkpoint = temp / "checkpoint"
            output = temp / "output"
            checkpoint.mkdir()
            target = "blocks.0.mlp.fc2.weight"
            weight = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            down = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            up = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
            write_safetensors(
                checkpoint / "model-00001-of-00001.safetensors",
                {target: ("BF16", weight.shape, bf16_bytes(weight))},
            )
            stem = "lora_unet_blocks_0_mlp_fc2"
            write_safetensors(
                temp / "kohya.safetensors",
                {
                    stem + ".alpha": ("BF16", (), bf16_bytes(np.array(1.0))),
                    stem + ".lora_down.weight": (
                        "BF16", down.shape, bf16_bytes(down)
                    ),
                    stem + ".lora_up.weight": (
                        "BF16", up.shape, bf16_bytes(up)
                    ),
                },
            )
            result = self.run_tool(checkpoint, temp / "kohya.safetensors", output)
            self.assertEqual(result.returncode, 0, result.stderr)
            actual = read_bf16(output / "model-00001-of-00001.safetensors", target)
            np.testing.assert_allclose(
                actual, weight + (1.0 / down.shape[0]) * (up @ down), atol=0.02
            )
            self.assertIn("kohya=1", result.stdout)

    def test_dry_run_does_not_require_or_create_output(self):
        root = Path(__file__).parents[1]
        tool = root / "tools/fold_turbo_lora.py"
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            checkpoint = temp / "checkpoint"
            checkpoint.mkdir()
            values = np.eye(2, dtype=np.float32)
            write_safetensors(
                checkpoint / "model-00001-of-00001.safetensors",
                {"blocks.0.weight": ("BF16", values.shape, bf16_bytes(values))},
            )
            write_safetensors(
                temp / "adapter.safetensors",
                {
                    "blocks.0.lora_A.weight": (
                        "F32", values.shape, values.tobytes()
                    ),
                    "blocks.0.lora_B.weight": (
                        "F32", values.shape, values.tobytes()
                    ),
                },
            )
            result = subprocess.run(
                [
                    sys.executable, str(tool),
                    "--checkpoint", str(checkpoint),
                    "--lora", str(temp / "adapter.safetensors"),
                    "--dry-run",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("dry run: all adapter pairs match", result.stdout)


if __name__ == "__main__":
    unittest.main()
