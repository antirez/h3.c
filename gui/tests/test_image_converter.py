import tempfile
import unittest
from pathlib import Path


class ReferenceImageConversionTests(unittest.TestCase):
    def test_converts_heic_to_png_without_modifying_the_source(self) -> None:
        from gui.image_converter import convert_reference_image

        commands: list[tuple[str, ...]] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "IMG_4621.HEIC"
            source.write_bytes(b"original-heic")

            def fake_run(command: tuple[str, ...]) -> None:
                commands.append(command)
                Path(command[-1]).write_bytes(b"\x89PNG\r\n\x1a\nconverted")

            converted = convert_reference_image(
                source,
                root / "converted",
                run=fake_run,
            )

            self.assertEqual(source.read_bytes(), b"original-heic")
            self.assertEqual(converted.suffix, ".png")
            self.assertEqual(converted.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(
                commands,
                [
                    (
                        "/usr/bin/sips",
                        "-s",
                        "format",
                        "png",
                        str(source.resolve()),
                        "--out",
                        str(converted),
                    )
                ],
            )


if __name__ == "__main__":
    unittest.main()
