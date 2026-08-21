import os
import struct
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QLabel


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


class ReferenceInputPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_applies_embedded_orientation_before_showing_reference(self) -> None:
        from gui.widgets import ReferenceInputPreview

        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "portrait.jpg"
            write_rotated_jpeg(reference)
            preview = ReferenceInputPreview()

            preview.set_references((reference,))
            QApplication.processEvents()

            card_item = preview.cards_layout.itemAtPosition(0, 0)
            assert card_item is not None
            card = card_item.widget()
            assert card is not None
            image_label = card.findChildren(QLabel)[0]
            pixmap = image_label.pixmap()
            self.assertFalse(pixmap.isNull())
            self.assertGreater(pixmap.height(), pixmap.width())


if __name__ == "__main__":
    unittest.main()
