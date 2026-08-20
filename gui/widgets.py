from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QWidget


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
