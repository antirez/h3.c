import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QApplication
except ImportError:  # pragma: no cover - exercised without the optional GUI deps
    QApplication = None
    QColor = None
    QImage = None
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
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal supported"),
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

    def test_additional_reference_images_keep_their_picture_order(self) -> None:
        from gui.window import MainWindow

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            primary = repo / "front.png"
            profile = repo / "profile.png"
            full_body = repo / "full-body.png"
            for image in (primary, profile, full_body):
                image.touch()
            window = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal 4"),
                runner=FakeRunner(),
                load_preferences=False,
            )
            window.set_paths(
                model_dir=repo / "MiniMax-H3",
                reference_image=primary,
                output_path=repo / "outputs" / "video.mp4",
            )

            window.add_reference_images((profile, full_body))
            settings = window.generation_settings()

            self.assertEqual(
                settings.reference_images,
                (primary, profile.resolve(), full_body.resolve()),
            )
            self.assertEqual(window.additional_references_list.count(), 2)

            window.additional_references_list.setCurrentRow(1)
            window.move_reference_up_button.click()
            self.assertEqual(
                window.generation_settings().reference_images,
                (primary, full_body.resolve(), profile.resolve()),
            )
            window.remove_reference_button.click()
            self.assertEqual(
                window.generation_settings().reference_images,
                (primary, profile.resolve()),
            )
            window.close()

    def test_input_preview_shows_all_selected_reference_images(self) -> None:
        from gui.window import MainWindow

        assert QApplication is not None
        assert QColor is not None
        assert QImage is not None
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            first = repo / "front.png"
            second = repo / "profile.png"
            for path, color in ((first, "#42c98c"), (second, "#e9b866")):
                image = QImage(320, 240, QImage.Format.Format_RGB32)
                image.fill(QColor(color))
                self.assertTrue(image.save(str(path), "PNG"))
            window = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal 4"),
                runner=FakeRunner(),
                load_preferences=False,
            )

            window.set_reference_image(first)
            window.add_reference_images((second,))
            QApplication.processEvents()

            self.assertEqual(
                window.input_preview.reference_paths,
                (first.resolve(), second.resolve()),
            )
            self.assertIs(window.preview_tabs.currentWidget(), window.input_preview)
            window.close()

    def test_primary_reference_cannot_exceed_the_nine_picture_limit(self) -> None:
        from gui.window import MainWindow

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            images = tuple(repo / f"reference-{index}.png" for index in range(10))
            for image in images:
                image.touch()
            window = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal supported"),
                runner=FakeRunner(),
                load_preferences=False,
            )
            window.add_reference_images(images[:9])

            with self.assertRaisesRegex(ValueError, "at most 9"):
                window.set_reference_image(images[9])

            self.assertEqual(window.reference_edit.text(), "")
            self.assertEqual(len(window.generation_settings().reference_images), 9)
            self.assertIn(
                "Picture 9",
                window.additional_references_list.item(8).text(),
            )
            window.close()

    def test_blank_output_is_rejected_before_settings_are_built(self) -> None:
        from gui.window import MainWindow

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            window = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal supported"),
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
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal supported"),
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
            self.assertIn("HEIC converted", window.reference_status.text())
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
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal supported"),
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

    def test_all_heic_references_follow_the_final_output_folder(self) -> None:
        from gui.window import MainWindow

        conversions: list[tuple[str, Path]] = []
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            (repo / "h3").touch()
            model = repo / "MiniMax-H3"
            model.mkdir()
            front = repo / "front.HEIC"
            profile = repo / "profile.HEIC"
            front.write_bytes(b"heic")
            profile.write_bytes(b"heic")

            def fake_converter(image: Path, output_dir: Path) -> Path:
                conversions.append((image.name, output_dir))
                output_dir.mkdir(parents=True, exist_ok=True)
                converted = output_dir / f"{image.stem}.png"
                converted.write_bytes(b"png")
                return converted

            runner = FakeRunner()
            window = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal 4"),
                runner=runner,
                load_preferences=False,
                reference_converter=fake_converter,
            )
            first_output = repo / "first" / "video.mp4"
            final_output = repo / "final" / "video.mp4"
            window.set_paths(
                model_dir=model,
                reference_image=None,
                output_path=first_output,
            )
            window.set_reference_image(front)
            window.add_reference_images((profile,))
            window.set_prompt("Picture 1 and Picture 2 show the same person.")
            window.output_edit.setText(str(final_output))

            window.generate_button.click()

            self.assertEqual(
                runner.settings.reference_images,
                (
                    final_output.parent / "reference-images" / "front.png",
                    final_output.parent / "reference-images" / "profile.png",
                ),
            )
            self.assertEqual(
                conversions[-2:],
                [
                    ("front.HEIC", final_output.parent / "reference-images"),
                    ("profile.HEIC", final_output.parent / "reference-images"),
                ],
            )
            runner.running = False
            window.close()

    def test_persists_technical_settings_but_starts_with_fresh_content(self) -> None:
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
                mac_info=MacInfo("Apple M5 Max", 48.0, "arm64", "Metal 4"),
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
            first.reference_size_combo.setCurrentIndex(1)
            first.token_reduction_check.setChecked(True)
            first.int8_row_fc2_check.setChecked(True)
            reference = repo / "previous-reference.png"
            reference.touch()
            first.set_reference_image(reference)
            first.set_prompt("This prompt belongs only to the current session.")
            first.close()
            preferences.sync()

            restored = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M5 Max", 48.0, "arm64", "Metal 4"),
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
            self.assertEqual(settings.reference_image_size, "match")
            self.assertTrue(settings.token_reduction)
            self.assertTrue(settings.use_int8_row_fc2)
            self.assertEqual(restored.prompt_edit.toPlainText(), "")
            self.assertEqual(restored.reference_edit.text(), "")
            self.assertEqual(settings.reference_images, ())
            self.assertEqual(restored.input_preview.reference_paths, ())
            restored.close()

    def test_legacy_prompt_and_reference_preferences_are_erased_on_load(self) -> None:
        from gui.window import MainWindow

        assert QSettings is not None
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            preferences_path = repo / "preferences.ini"
            preferences = QSettings(
                str(preferences_path),
                QSettings.Format.IniFormat,
            )
            preferences.setValue("prompt", "A private previous prompt")
            preferences.setValue("reference_image", "/private/previous.png")
            preferences.setValue("reference_source", "/private/previous.HEIC")
            preferences.sync()

            window = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal 4"),
                runner=FakeRunner(),
                settings_store=preferences,
            )

            self.assertEqual(window.prompt_edit.toPlainText(), "")
            self.assertEqual(window.reference_edit.text(), "")
            for key in ("prompt", "reference_image", "reference_source"):
                self.assertIsNone(preferences.value(key))
            window.close()

    def test_int8_row_fc2_requires_m5_metal4_and_ssd_streaming_off(self) -> None:
        from gui.window import MainWindow

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            m4_window = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal 4"),
                runner=FakeRunner(),
                load_preferences=False,
            )
            self.assertFalse(m4_window.int8_row_fc2_check.isEnabled())
            self.assertIn("M5", m4_window.int8_row_fc2_check.toolTip())
            m4_window.close()

            m5_window = MainWindow(
                repo_root=repo,
                mac_info=MacInfo(
                    "Apple M5 Max",
                    64.0,
                    "arm64",
                    "Metal supported",
                ),
                runner=FakeRunner(),
                load_preferences=False,
            )
            self.assertFalse(m5_window.int8_row_fc2_check.isEnabled())
            m5_window.ssd_streaming_check.setChecked(False)
            self.assertTrue(m5_window.int8_row_fc2_check.isEnabled())
            m5_window.int8_row_fc2_check.setChecked(True)
            m5_window.ssd_streaming_check.setChecked(True)
            self.assertFalse(m5_window.int8_row_fc2_check.isChecked())
            self.assertFalse(m5_window.int8_row_fc2_check.isEnabled())
            m5_window.close()

    def test_live_preview_appears_and_resizes_with_the_window(self) -> None:
        from gui.window import MainWindow

        assert QApplication is not None
        assert QColor is not None
        assert QImage is not None
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            (repo / "h3").touch()
            model = repo / "MiniMax-H3"
            model.mkdir()
            reference = repo / "reference.png"
            reference.touch()
            output = repo / "outputs" / "video.mp4"
            preview = repo / "preview.ppm"
            image = QImage(800, 600, QImage.Format.Format_RGB32)
            image.fill(QColor("#42c98c"))
            self.assertTrue(image.save(str(preview), "PPM"))
            runner = FakeRunner()
            window = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal 4"),
                runner=runner,
                load_preferences=False,
            )
            window.set_paths(
                model_dir=model,
                reference_image=reference,
                output_path=output,
            )
            window.set_prompt("A person powers up.")
            window.live_preview_check.setChecked(True)
            window.resize(820, 650)
            window.show()
            QApplication.processEvents()
            window.generate_button.click()

            runner.callbacks.on_preview(preview)
            QApplication.processEvents()
            first_pixmap = window.preview_label.pixmap()
            self.assertIs(window.preview_stack.currentWidget(), window.preview_label)
            self.assertFalse(first_pixmap.isNull())
            first_width = first_pixmap.width()

            window.resize(1200, 900)
            QApplication.processEvents()
            resized_pixmap = window.preview_label.pixmap()

            self.assertGreater(resized_pixmap.width(), first_width)
            runner.running = False
            window.close()

    def test_custom_opens_advanced_controls_and_layout_is_responsive(self) -> None:
        from PySide6.QtCore import Qt

        from gui.window import MainWindow

        assert QApplication is not None
        with tempfile.TemporaryDirectory() as directory:
            window = MainWindow(
                repo_root=Path(directory),
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal 4"),
                runner=FakeRunner(),
                load_preferences=False,
            )
            window.show()
            original_steps = window.steps_spin.value()

            window.custom_button.click()

            self.assertEqual(window.windowTitle(), "H3 Studio - by pierpaolov")
            self.assertTrue(window.custom_button.isChecked())
            self.assertTrue(window.advanced_check.isChecked())
            self.assertTrue(window.advanced_group.isVisible())
            self.assertEqual(window.steps_spin.value(), original_steps)

            window.resize(700, 600)
            QApplication.processEvents()
            self.assertEqual(
                window.content_splitter.orientation(), Qt.Orientation.Vertical
            )
            window.resize(1200, 800)
            QApplication.processEvents()
            self.assertEqual(
                window.content_splitter.orientation(), Qt.Orientation.Horizontal
            )
            window.close()

    def test_preview_placeholder_reports_failure_before_first_frame(self) -> None:
        from gui.runner import RunResult
        from gui.window import MainWindow

        assert QApplication is not None
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory).resolve()
            (repo / "h3").touch()
            model = repo / "MiniMax-H3"
            model.mkdir()
            output = repo / "outputs" / "video.mp4"
            runner = FakeRunner()
            window = MainWindow(
                repo_root=repo,
                mac_info=MacInfo("Apple M4 Pro", 48.0, "arm64", "Metal 4"),
                runner=runner,
                load_preferences=False,
            )
            window.set_paths(
                model_dir=model,
                reference_image=None,
                output_path=output,
            )
            window.set_prompt("A person powers up.")
            window.live_preview_check.setChecked(True)
            window.generate_button.click()

            runner.callbacks.on_finished(
                RunResult(exit_code=1, cancelled=False, output_path=output)
            )
            QApplication.processEvents()

            self.assertIs(
                window.preview_stack.currentWidget(), window.preview_placeholder
            )
            self.assertIn("failed", window.preview_placeholder.text().lower())
            runner.running = False
            window.close()


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
