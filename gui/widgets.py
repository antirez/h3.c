from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


class ScalablePreviewLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self._source_pixmap: QPixmap | None = None

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self._fit_source()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._fit_source()

    def _fit_source(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        self.setPixmap(
            self._source_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class ReferenceInputPreview(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.reference_paths: tuple[Path, ...] = ()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        self.placeholder = QLabel(
            "No reference images selected.\n\n"
            "Selected inputs will appear here as Picture 1, Picture 2, …"
        )
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setObjectName("previewPlaceholder")
        layout.addWidget(self.placeholder, 1)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("inputPreviewScroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.cards = QWidget()
        self.cards.setObjectName("inputPreviewCards")
        self.cards_layout = QGridLayout(self.cards)
        self.cards_layout.setContentsMargins(4, 4, 4, 4)
        self.cards_layout.setSpacing(10)
        self.scroll_area.setWidget(self.cards)
        self.scroll_area.setVisible(False)
        layout.addWidget(self.scroll_area, 1)

    def set_references(self, paths: tuple[Path, ...]) -> None:
        self.reference_paths = paths
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.hide()
                widget.setParent(None)
                widget.deleteLater()
        self.placeholder.setVisible(not paths)
        self.scroll_area.setVisible(bool(paths))
        for index, path in enumerate(paths):
            card = QFrame()
            card.setObjectName("inputCard")
            card_layout = QVBoxLayout(card)
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                image_label.setText("Preview unavailable")
            else:
                image_label.setPixmap(
                    pixmap.scaled(
                        240,
                        180,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            caption = QLabel(f"Picture {index + 1} · {path.name}")
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            caption.setWordWrap(True)
            card_layout.addWidget(image_label, 1)
            card_layout.addWidget(caption)
            row, column = divmod(index, 2)
            self.cards_layout.addWidget(card, row, column)
        self.cards_layout.setColumnStretch(0, 1)
        self.cards_layout.setColumnStretch(1, 1)
        if paths:
            self.cards_layout.setRowStretch((len(paths) - 1) // 2 + 1, 1)


class SegmentedProgressBar(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.segments: list[QProgressBar] = []
        for _ in range(3):
            segment = QProgressBar()
            segment.setRange(0, 1000)
            segment.setValue(0)
            segment.setTextVisible(False)
            layout.addWidget(segment, 1)
            self.segments.append(segment)

    def set_overall_percent(self, percent: float) -> None:
        bounded = min(100.0, max(0.0, percent))
        for index, segment in enumerate(self.segments):
            segment_start = index * (100.0 / 3.0)
            local_percent = (bounded - segment_start) * 3.0
            segment.setValue(round(min(100.0, max(0.0, local_percent)) * 10))

    def reset(self) -> None:
        self.set_overall_percent(0.0)

    def complete(self) -> None:
        self.set_overall_percent(100.0)
