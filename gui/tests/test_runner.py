import io
import threading
import unittest
from pathlib import Path

from gui.runner import (
    GenerationSettings,
    H3Runner,
    ProgressTracker,
    RunnerCallbacks,
    build_h3_command,
)


def example_settings() -> GenerationSettings:
    return GenerationSettings(
        model_dir=Path("/models/MiniMax-H3"),
        prompt="Un test",
        output_path=Path("/tmp/h3-test-output.mp4"),
        reference_image=None,
        width=512,
        height=512,
        render_width=320,
        render_height=320,
        seconds=2,
        steps=6,
        layers=40,
        reuse=1,
        core_reuse=1,
        seed=42,
        ssd_streaming=True,
        live_preview=True,
        preview_dir=Path("/tmp/h3-previews"),
    )


class FakeProcess:
    def __init__(self, output: str, *, blocked: bool = False) -> None:
        self.stdout = io.StringIO(output)
        self.terminated = False
        self.killed = False
        self._done = threading.Event()
        self.returncode: int | None = None
        if not blocked:
            self.returncode = 0
            self._done.set()

    def wait(self) -> int:
        self._done.wait(2)
        return self.returncode if self.returncode is not None else -1

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 130
        self._done.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._done.set()


class CommandTests(unittest.TestCase):
    def test_builds_reference_video_command_without_a_shell(self) -> None:
        settings = GenerationSettings(
            model_dir=Path("/models/MiniMax-H3"),
            prompt="Michela cammina sulla spiaggia e sorride.",
            output_path=Path("/videos/michela.mp4"),
            reference_image=Path("/images/michela.png"),
            width=512,
            height=512,
            render_width=320,
            render_height=320,
            seconds=2,
            steps=6,
            layers=40,
            reuse=1,
            core_reuse=1,
            seed=42,
            ssd_streaming=True,
            live_preview=False,
            additional_reference_images=(
                Path("/images/michela-profile.png"),
                Path("/images/michela-full-body.png"),
            ),
            reference_image_size="match",
            token_reduction=True,
            use_int8_row_fc2=True,
        )

        command = build_h3_command(Path("/repo/h3"), settings)

        self.assertEqual(
            command,
            [
                "/repo/h3",
                "--profile",
                "-d",
                "/models/MiniMax-H3",
                "--ref-image",
                "/images/michela.png",
                "--ref-image",
                "/images/michela-profile.png",
                "--ref-image",
                "/images/michela-full-body.png",
                "--ref-image-size",
                "match",
                "-p",
                "Michela cammina sulla spiaggia e sorride.",
                "--width",
                "512",
                "--height",
                "512",
                "--render-width",
                "320",
                "--render-height",
                "320",
                "--seconds",
                "2",
                "--steps",
                "6",
                "--layers",
                "40",
                "--reuse",
                "1",
                "--seed",
                "42",
                "--ssd-streaming",
                "--token-reduction",
                "--use-int8-row-fc2",
                "-o",
                "/videos/michela.mp4",
            ],
        )


class ProgressTests(unittest.TestCase):
    def test_estimates_remaining_time_from_completed_steps(self) -> None:
        tracker = ProgressTracker()

        first = tracker.consume("\rdenoise                    0/6   ", now=100.0)
        second = tracker.consume("\rdenoise                    1/6   ", now=110.0)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert second is not None
        self.assertEqual(second.phase, "denoise")
        self.assertEqual(second.stage, "Generation")
        self.assertEqual((second.completed, second.total), (1, 6))
        self.assertAlmostEqual(second.percent, 350 / 9)
        self.assertAlmostEqual(second.eta_seconds or 0.0, 110 / 7)

    def test_ignores_regular_diagnostic_output(self) -> None:
        tracker = ProgressTracker()

        self.assertIsNone(
            tracker.consume("h3: video VAE cache miss; decoder retained", now=2.0)
        )

    def test_maps_log_phases_to_three_global_progress_segments(self) -> None:
        tracker = ProgressTracker()

        tracker.consume("tokenizer 0/1", now=100.0)
        preparation = tracker.consume("video VAE encoder 154/154", now=120.0)
        generation = tracker.consume("denoise 3/6", now=160.0)
        export_started = tracker.consume("FFmpeg 0/96", now=190.0)
        completed = tracker.consume("FFmpeg 96/96", now=200.0)

        assert preparation is not None
        assert generation is not None
        assert export_started is not None
        assert completed is not None
        self.assertEqual(preparation.stage, "Preparation")
        self.assertLess(preparation.percent, 100 / 3)
        self.assertGreater(preparation.eta_seconds or 0.0, 0.0)
        self.assertEqual(generation.stage, "Generation")
        self.assertGreaterEqual(generation.percent, 100 / 3)
        self.assertLess(generation.percent, 200 / 3)
        self.assertEqual(export_started.stage, "Decode & export")
        self.assertGreaterEqual(export_started.percent, 200 / 3)
        self.assertEqual(completed.percent, 100.0)
        self.assertEqual(completed.eta_seconds, 0.0)

    def test_consecutive_reference_phase_resets_continue_to_advance(self) -> None:
        tracker = ProgressTracker()

        updates = [
            tracker.consume("tokenizer 0/1", now=0.0),
            tracker.consume("video VAE encoder 154/154", now=10.0),
            tracker.consume("video VAE encoder 0/154", now=20.0),
            tracker.consume("video VAE encoder 77/154", now=30.0),
            tracker.consume("video VAE encoder 154/154", now=40.0),
            tracker.consume("Qwen vision 80/80", now=50.0),
        ]
        progress = [item for item in updates if item is not None]

        self.assertGreater(progress[3].percent, progress[1].percent)
        self.assertGreater(progress[4].percent, progress[3].percent)
        self.assertGreater(progress[5].percent, progress[4].percent)

    def test_eta_recovers_from_an_early_underestimate(self) -> None:
        tracker = ProgressTracker()

        tracker.consume("tokenizer 0/1", now=0.0)
        early = tracker.consume("tokenizer 1/1", now=1.0)
        delayed = tracker.consume("video VAE encoder 1/154", now=100.0)

        assert early is not None
        assert delayed is not None
        self.assertGreater(early.eta_seconds or 0.0, 1.0)
        self.assertGreater(delayed.eta_seconds or 0.0, 1.0)


class RunnerTests(unittest.TestCase):
    def test_reports_progress_preview_and_completion(self) -> None:
        process = FakeProcess(
            "\rdenoise                    0/6   "
            "\rdenoise                    1/6   \n"
            "h3: preview-file /tmp/h3-previews/preview-0001.ppm\n"
        )
        done = threading.Event()
        progress = []
        previews = []
        results = []
        runner = H3Runner(
            Path("/repo/h3"),
            process_factory=lambda command, cwd: process,
        )

        runner.start(
            example_settings(),
            RunnerCallbacks(
                on_progress=progress.append,
                on_output=lambda line: None,
                on_preview=previews.append,
                on_finished=lambda result: (results.append(result), done.set()),
            ),
        )

        self.assertTrue(done.wait(2))
        self.assertEqual([(item.completed, item.total) for item in progress], [(0, 6), (1, 6)])
        self.assertEqual(previews, [Path("/tmp/h3-previews/preview-0001.ppm")])
        self.assertEqual(results[0].exit_code, 0)
        self.assertFalse(results[0].cancelled)

    def test_stop_requests_process_termination(self) -> None:
        process = FakeProcess("", blocked=True)
        done = threading.Event()
        runner = H3Runner(
            Path("/repo/h3"),
            process_factory=lambda command, cwd: process,
        )
        callbacks = RunnerCallbacks(
            on_progress=lambda update: None,
            on_output=lambda line: None,
            on_preview=lambda path: None,
            on_finished=lambda result: done.set(),
        )

        runner.start(example_settings(), callbacks)
        runner.stop()

        self.assertTrue(done.wait(2))
        self.assertTrue(process.terminated)


if __name__ == "__main__":
    unittest.main()
