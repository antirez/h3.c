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
                "--ref-image-size",
                "max",
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
        self.assertEqual((second.completed, second.total), (1, 6))
        self.assertAlmostEqual(second.percent, 100 / 6)
        self.assertAlmostEqual(second.eta_seconds or 0.0, 50.0)

    def test_ignores_regular_diagnostic_output(self) -> None:
        tracker = ProgressTracker()

        self.assertIsNone(
            tracker.consume("h3: video VAE cache miss; decoder retained", now=2.0)
        )


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
