import struct
import tempfile
import unittest
from pathlib import Path

from PySide6.QtGui import QColor, QImage, QImageIOHandler, QImageReader


def write_rotated_jpeg(path: Path) -> None:
    image = QImage(40, 20, QImage.Format.Format_RGB32)
    image.fill(QColor("#42c98c"))
    if not image.save(str(path), "JPEG"):
        raise AssertionError("Could not create the orientation test image")
    jpeg = path.read_bytes()
    tiff = (
        b"MM\x00\x2a\x00\x00\x00\x08"
        b"\x00\x01"
        b"\x01\x12\x00\x03\x00\x00\x00\x01\x00\x06\x00\x00"
        b"\x00\x00\x00\x00"
    )
    payload = b"Exif\x00\x00" + tiff
    app1 = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    path.write_bytes(jpeg[:2] + app1 + jpeg[2:])


class ReferenceImageConversionTests(unittest.TestCase):
    def test_converts_heic_to_physically_oriented_png_without_modifying_source(
        self,
    ) -> None:
        from gui.image_converter import convert_reference_image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "IMG_4621.HEIC"
            write_rotated_jpeg(source)
            original = source.read_bytes()

            converted = convert_reference_image(
                source,
                root / "converted",
            )

            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(converted.suffix, ".png")
            self.assertEqual(converted.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            reader = QImageReader(str(converted))
            self.assertEqual((reader.size().width(), reader.size().height()), (20, 40))
            self.assertEqual(
                reader.transformation(),
                QImageIOHandler.Transformation.TransformationNone,
            )

    def test_same_iphone_filename_from_different_folders_does_not_collide(
        self,
    ) -> None:
        from gui.image_converter import convert_reference_image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "trip-one" / "IMG_4621.HEIC"
            second = root / "trip-two" / "IMG_4621.HEIC"
            first.parent.mkdir()
            second.parent.mkdir()
            write_rotated_jpeg(first)
            write_rotated_jpeg(second)

            first_png = convert_reference_image(first, root / "converted")
            second_png = convert_reference_image(second, root / "converted")

            self.assertNotEqual(first_png, second_png)
            self.assertTrue(first_png.is_file())
            self.assertTrue(second_png.is_file())


if __name__ == "__main__":
    unittest.main()
