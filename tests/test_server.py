import os
import struct
import tempfile
import textwrap
import unittest
from pathlib import Path

import server


class VideoParameterTests(unittest.TestCase):
    def test_normalizes_defaults_and_blank_seed(self):
        params = server.normalize_video_params({"seed": ""})
        self.assertEqual(params["seed"], 42)
        self.assertEqual((params["width"], params["height"]), (864, 480))
        self.assertFalse(params["match_input_aspect"])

    def test_rejects_invalid_canvas_and_conflicting_memory_modes(self):
        with self.assertRaisesRegex(ValueError, "multiples of 32"):
            server.normalize_video_params({"width": 513})
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            server.normalize_video_params({
                "ssd_streaming": True,
                "int8_row_fc2": True,
            })
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            server.normalize_video_params({"ssd_streaming": "false"})

    def test_source_aspect_selects_legal_canvas_near_target_area(self):
        self.assertEqual(
            server.legal_canvas_for_aspect(1920, 1080, 1024 * 576),
            (1024, 576),
        )
        self.assertEqual(
            server.legal_canvas_for_aspect(3024, 4032, 1024 * 576),
            (672, 896),
        )
        width, height = server.legal_canvas_for_aspect(1000, 667, 1024 * 576)
        self.assertEqual(width % 32, 0)
        self.assertEqual(height % 32, 0)
        self.assertLess(abs(width / height - 1000 / 667), 0.02)

    def test_primary_aspect_image_follows_workflow_order(self):
        files = {
            "first_frame": ["first.jpg"],
            "last_frame": ["last.jpg"],
            "ref_image": ["reference.jpg"],
        }
        self.assertEqual(
            server.primary_image_reference("frame-anchor", files), "first.jpg"
        )
        self.assertEqual(
            server.primary_image_reference("ref-image", files), "reference.jpg"
        )
        self.assertIsNone(server.primary_image_reference("text", files))

    def test_maps_real_video_vae_decode_progress(self):
        job = {
            "kind": "video",
            "status": "running",
            "started_timestamp": 0.0,
            "estimated_total_seconds": 100.0,
            "progress_percent": 0.0,
            "log": "",
        }
        server.record_process_line(job, "video VAE decode 18/36")
        self.assertEqual(job["phase"], "video VAE decode")
        self.assertGreaterEqual(job["progress_percent"], 95.0)

    def test_worker_command_encodes_multiline_utf8_and_paths(self):
        request = {
            "prompt": "人在走\n镜头跟随",
            "mode": "frame-anchor",
            "params": server.normalize_video_params({}),
            "files": {"first_frame": ["/tmp/a b.jpg"], "last_frame": []},
        }
        commands = server.build_worker_commands(
            request, "job_1", "/tmp/output file.mp4"
        )
        run = commands[-1].split()
        self.assertEqual(run[:2], ["!worker-run", "job_1"])
        self.assertEqual(bytes.fromhex(run[2]).decode(), "/tmp/output file.mp4")
        self.assertEqual(bytes.fromhex(run[3]).decode(), request["prompt"])
        anchor = next(line for line in commands if line.startswith("!worker-anchor"))
        self.assertEqual(bytes.fromhex(anchor.split()[2]).decode(), "/tmp/a b.jpg")

    def test_turbo_profile_selects_safe_mode_specific_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "turbo-v4-step600-ema"
            (root / "FL2VA/transformer/.h3-int8").mkdir(parents=True)
            (root / "FL2VA/transformer/.h3-int8/source.tsv").write_text("ok")
            profile = server.generation_profile(root)
            self.assertTrue(profile["turbo"])
            self.assertTrue(profile["int8_sidecar"])
            self.assertEqual(profile["text_defaults"], {
                "steps": 5, "layers": 50, "reuse": 1,
            })
            self.assertEqual(profile["reference_defaults"], {
                "steps": 20, "layers": 45, "reuse": 2,
            })
            normalized = server.normalize_video_params(
                {}, profile["text_defaults"]
            )
            self.assertEqual(
                (normalized["steps"], normalized["layers"], normalized["reuse"]),
                (5, 50, 1),
            )


class PersistentWorkerTests(unittest.TestCase):
    @staticmethod
    def _write_minimal_model(root):
        model = root / "model"
        for relative in (
            "FL2VA/text_encoder",
            "FL2VA/transformer",
            "FL2VA/video_vae/source",
            "FL2VA/audio_vae",
            "FL2VA/tokenizer",
        ):
            (model / relative).mkdir(parents=True, exist_ok=True)
        (model / "FL2VA/transformer/config.json").write_text("{}")
        (model / "FL2VA/tokenizer/tokenizer.json").write_text("{}")
        empty_safetensors = struct.pack("<Q", 2) + b"{}"
        for relative in (
            "FL2VA/text_encoder/empty.safetensors",
            "FL2VA/transformer/empty.safetensors",
            "FL2VA/video_vae/source/empty.safetensors",
            "FL2VA/audio_vae/empty.safetensors",
        ):
            (model / relative).write_bytes(empty_safetensors)
        return model

    def test_two_jobs_reuse_one_process(self):
        script_source = textwrap.dedent("""\
            #!/usr/bin/env python3
            import pathlib
            import sys

            print("H3_WORKER_READY", flush=True)
            runs = 0
            for raw in sys.stdin:
                line = raw.strip()
                if not line.startswith("!worker-run "):
                    continue
                _, job_id, output_hex, _ = line.split()
                output = bytes.fromhex(output_hex).decode()
                print(f"H3_WORKER_START {job_id}", flush=True)
                print("video VAE cache hit" if runs else
                      "video VAE cache miss; decoder retained", flush=True)
                pathlib.Path(output).write_bytes(b"fake mp4")
                runs += 1
                print(f"H3_WORKER_DONE {job_id}", flush=True)
        """)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "fake-h3"
            executable.write_text(script_source)
            executable.chmod(0o755)
            worker = server.H3PersistentWorker(
                executable, root / "model", os.environ.copy(), startup_timeout=2
            )
            request = {
                "prompt": "walk",
                "mode": "text",
                "params": server.normalize_video_params({}),
                "files": {},
            }
            try:
                first = {"job_id": "job1", "kind": "video", "status": "running", "log": ""}
                second = {"job_id": "job2", "kind": "video", "status": "running", "log": ""}
                self.assertEqual(worker.run(first, request, root / "one.mp4", 2), 0)
                first_status = worker.status()
                self.assertEqual(worker.run(second, request, root / "two.mp4", 2), 0)
                second_status = worker.status()
                self.assertEqual(first_status["pid"], second_status["pid"])
                self.assertEqual(second_status["starts"], 1)
                self.assertIn("cache miss", first["log"])
                self.assertIn("cache hit", second["log"])
            finally:
                worker.stop()

    def test_native_worker_survives_failed_generation(self):
        executable = Path(__file__).parents[1] / "h3"
        if not executable.is_file():
            self.skipTest("h3 binary has not been built")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self._write_minimal_model(root)
            worker = server.H3PersistentWorker(
                executable, model, os.environ.copy(), startup_timeout=5
            )
            request = {
                "prompt": "walk",
                "mode": "text",
                "params": server.normalize_video_params({}),
                "files": {},
            }
            try:
                first = {"job_id": "native1", "kind": "video", "status": "running", "log": ""}
                second = {"job_id": "native2", "kind": "video", "status": "running", "log": ""}
                self.assertEqual(worker.run(first, request, root / "one.mp4", 5), 1)
                first_pid = worker.status()["pid"]
                self.assertEqual(worker.run(second, request, root / "two.mp4", 5), 1)
                self.assertEqual(worker.status()["pid"], first_pid)
                self.assertEqual(worker.status()["starts"], 1)
                self.assertIn("tokenizer", first["log"])
            finally:
                worker.stop()


if __name__ == "__main__":
    unittest.main()
