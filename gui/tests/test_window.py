import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover - exercised without the optional GUI deps
    QApplication = None
    QSettings = None

from gui.hardware import MacInfo


@unittest.skipIf(QApplication is None, "PySide6 is not installed")
class MainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert QApplication is not None
        cls.app = QApplication.instance() or QApplication([])

    def test_fast_preset_produces_expected_generation_settings(self) -> None:
        from gui.window import MainWindow

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            runner = FakeRunner()
            window = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal 3"),
                runner=runner,
                load_preferences=False,
            )
            window.set_paths(
                model_dir=repo / "MiniMax-H3",
                reference_image=repo / "michela.png",
                output_path=repo / "outputs" / "video.mp4",
            )
            window.set_prompt("Michela cammina sulla spiaggia.")
            window.apply_preset("fast")

            settings = window.generation_settings()

            self.assertEqual((settings.width, settings.height), (512, 512))
            self.assertEqual(
                (settings.render_width, settings.render_height), (320, 320)
            )
            self.assertEqual((settings.seconds, settings.steps), (2, 6))
            self.assertEqual((settings.layers, settings.reuse), (40, 1))
            self.assertTrue(settings.ssd_streaming)
            self.assertFalse(settings.live_preview)
            window.close()

    def test_fast_portrait_render_preserves_output_aspect_ratio(self) -> None:
        from gui.window import MainWindow

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            window = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal supportato"),
                runner=FakeRunner(),
                load_preferences=False,
            )
            window.apply_preset("fast")
            window.select_format(480, 864)

            settings = window.generation_settings()

            assert settings.render_width is not None
            assert settings.render_height is not None
            self.assertEqual(
                settings.width * settings.render_height,
                settings.height * settings.render_width,
            )
            self.assertEqual(
                (settings.render_width, settings.render_height), (320, 576)
            )
            window.close()

    def test_blank_output_is_rejected_before_settings_are_built(self) -> None:
        from gui.window import MainWindow

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            window = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal supportato"),
                runner=FakeRunner(),
                load_preferences=False,
            )
            window.output_edit.clear()

            with self.assertRaisesRegex(ValueError, "output"):
                window.generation_settings()
            window.close()

    def test_selecting_heic_reference_uses_the_converted_png(self) -> None:
        from gui.window import MainWindow

        conversions: list[tuple[Path, Path]] = []
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = repo / "IMG_4621.HEIC"
            source.write_bytes(b"heic")
            converted = repo / "outputs" / "reference-images" / "IMG_4621.png"

            def fake_converter(image: Path, output_dir: Path) -> Path:
                conversions.append((image, output_dir))
                converted.parent.mkdir(parents=True)
                converted.write_bytes(b"png")
                return converted

            window = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal supportato"),
                runner=FakeRunner(),
                load_preferences=False,
                reference_converter=fake_converter,
            )
            chosen_output = repo / "custom-output" / "video.mp4"
            window.output_edit.setText(str(chosen_output))

            selected = window.set_reference_image(source)

            self.assertEqual(selected, converted)
            self.assertEqual(window.reference_edit.text(), str(converted))
            self.assertEqual(
                conversions,
                [(source, chosen_output.resolve().parent / "reference-images")],
            )
            self.assertIn("HEIC convertito", window.reference_status.text())
            window.close()

    def test_changing_video_output_reconverts_heic_beside_new_output(self) -> None:
        from gui.window import MainWindow

        conversions: list[Path] = []
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            (repo / "h3").touch()
            model = repo / "MiniMax-H3"
            model.mkdir()
            source = repo / "IMG_4621.HEIC"
            source.write_bytes(b"heic")

            def fake_converter(image: Path, output_dir: Path) -> Path:
                conversions.append(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                converted = output_dir / "IMG_4621-converted.png"
                converted.write_bytes(b"png")
                return converted

            runner = FakeRunner()
            window = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal supportato"),
                runner=runner,
                load_preferences=False,
                reference_converter=fake_converter,
            )
            first_output = repo / "first" / "video.mp4"
            second_output = repo / "second" / "video.mp4"
            window.set_paths(
                model_dir=model,
                reference_image=None,
                output_path=first_output,
            )
            window.set_prompt("Michela cammina sulla spiaggia.")
            window.set_reference_image(source)
            window.output_edit.setText(str(second_output))

            window.generate_button.click()

            self.assertEqual(
                conversions,
                [
                    first_output.parent / "reference-images",
                    second_output.parent / "reference-images",
                ],
            )
            self.assertEqual(
                runner.settings.reference_image,
                second_output.parent
                / "reference-images"
                / "IMG_4621-converted.png",
            )
            runner.running = False
            window.close()

    def test_persists_complete_generation_configuration(self) -> None:
        from gui.window import MainWindow

        assert QSettings is not None
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            preferences_path = repo / "preferences.ini"
            preferences = QSettings(
                str(preferences_path), QSettings.Format.IniFormat
            )
            first = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal supportato"),
                runner=FakeRunner(),
                settings_store=preferences,
            )
            first.apply_preset("fast")
            first.select_format(480, 864)
            first.select_duration(4)
            first.live_preview_check.setChecked(True)
            first.steps_spin.setValue(12)
            first.seed_spin.setValue(99)
            first.ssd_streaming_check.setChecked(False)
            first.close()
            preferences.sync()

            restored = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal supportato"),
                runner=FakeRunner(),
                settings_store=QSettings(
                    str(preferences_path), QSettings.Format.IniFormat
                ),
            )
            settings = restored.generation_settings()

            self.assertEqual((settings.width, settings.height), (480, 864))
            self.assertEqual(settings.seconds, 4)
            self.assertEqual(settings.steps, 12)
            self.assertEqual(settings.seed, 99)
            self.assertTrue(settings.live_preview)
            self.assertFalse(settings.ssd_streaming)
            restored.close()


class FakeRunner:
    running = False

    def start(self, settings, callbacks) -> None:
        self.settings = settings
        self.callbacks = callbacks
        self.running = True

    def stop(self) -> None:
        self.stopped = True


if __name__ == "__main__":
    unittest.main()
