import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.build_int8_sidecar import (
    bf16_to_f32,
    quantize_bf16_rows,
    quantize_shard,
)


def f32_to_bf16(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return rounded.astype("<u2")


def write_bf16_safetensors(path, name, values):
    payload = f32_to_bf16(values).tobytes()
    header = {
        name: {
            "dtype": "BF16",
            "shape": list(values.shape),
            "data_offsets": [0, len(payload)],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def read_tensor(path, name):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(size))
        info = header[name]
        start, end = info["data_offsets"]
        stream.seek(8 + size + start)
        dtype = {"I8": np.int8, "F32": "<f4"}[info["dtype"]]
        data = np.frombuffer(stream.read(end - start), dtype=dtype)
    return data.reshape(info["shape"])


class Int8SidecarTests(unittest.TestCase):
    def test_row_quantization_matches_expected_scales(self):
        values = np.array([
            [-1.0, -0.5, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float32)
        quantized, scales = quantize_bf16_rows(f32_to_bf16(values))
        np.testing.assert_array_equal(
            quantized[0], np.array([-127, -64, 0, 127], dtype=np.int8)
        )
        np.testing.assert_array_equal(quantized[1], np.zeros(4, dtype=np.int8))
        self.assertAlmostEqual(float(scales[0]), 1.0 / 127.0, places=8)
        self.assertAlmostEqual(float(scales[1]), 1.0 / 127.0, places=8)

    def test_quantizes_only_transformer_matrices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "model.safetensors"
            destination = root / "int8.safetensors"
            name = "blocks.0.mlp.fc2.weight"
            values = np.array([
                [1.0, -0.5, 0.25, -1.0],
                [2.0, 1.0, -2.0, 0.0],
            ], dtype=np.float32)
            write_bf16_safetensors(source, name, values)
            self.assertEqual(quantize_shard(source, destination, 1), [name])
            quantized = read_tensor(destination, name)
            scales = read_tensor(destination, f"{name}.h3_scale")
            restored = quantized.astype(np.float32) * scales[:, None]
            np.testing.assert_allclose(
                restored, bf16_to_f32(f32_to_bf16(values)), atol=0.01
            )


if __name__ == "__main__":
    unittest.main()
