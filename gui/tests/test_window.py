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
