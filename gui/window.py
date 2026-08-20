from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QSettings, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .hardware import MacInfo
from .image_converter import (
    REFERENCE_IMAGE_FILE_FILTER,
    ImageConversionError,
    convert_reference_image,
    requires_png_conversion,
)
from .presets import PRESET_ORDER, preset_for, recommended_preset_name
from .runner import (
    GenerationSettings,
    H3Runner,
    ProgressUpdate,
    RunResult,
    RunnerCallbacks,
)
from .widgets import ScalablePreviewLabel, SegmentedProgressBar


class RunnerBridge(QObject):
    progress = Signal(object)
    output = Signal(str)
    preview = Signal(str)
    finished = Signal(object)


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        repo_root: Path,
        mac_info: MacInfo,
        runner: H3Runner | Any,
        load_preferences: bool = True,
        default_output_dir: Path | None = None,
        settings_store: QSettings | None = None,
        reference_converter: Callable[[Path, Path], Path] = convert_reference_image,
    ) -> None:
        super().__init__()
        self.repo_root = repo_root.resolve()
        self.mac_info = mac_info
        self.runner = runner
        self.default_output_dir = default_output_dir or self.repo_root / "outputs"
        self.reference_converter = reference_converter
        self._persistence_enabled = load_preferences
        self._settings = settings_store or QSettings("h3-metal", "H3 Studio")
        self._preview_temp: tempfile.TemporaryDirectory[str] | None = None
        self._last_output: Path | None = None
        self._preview_received = False
        self._reference_source: Path | None = None
        self._converted_reference: Path | None = None
        self._active_preset = recommended_preset_name(mac_info)
        self._preset_buttons: dict[str, QPushButton] = {}
        self._bridge = RunnerBridge(self)
        self._bridge.progress.connect(self._on_progress)
        self._bridge.output.connect(self._append_log)
        self._bridge.preview.connect(self._on_preview)
        self._bridge.finished.connect(self._on_finished)

        self.setWindowTitle("H3 Studio - by pierpaolov")
        self.setMinimumSize(640, 480)
        self.resize(1080, 760)
        self._build_ui()
        self._update_responsive_layout(self.width())
        self._apply_style()
        self.apply_preset(self._active_preset)
        self._set_default_paths()
        if load_preferences:
            self._load_preferences()

    def _build_ui(self) -> None:
        content = QWidget(self)
        content.setObjectName("contentRoot")
        root = QVBoxLayout(content)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)
        self.content_scroll = QScrollArea(self)
        self.content_scroll.setObjectName("contentScroll")
        self.content_scroll.setWidgetResizable(True)
        self.content_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.content_scroll.setWidget(content)
        self.setCentralWidget(self.content_scroll)

        hardware = QFrame()
        hardware.setObjectName("hardwarePanel")
        hardware_layout = QHBoxLayout(hardware)
        hardware_layout.setContentsMargins(14, 10, 14, 10)
        hardware_text = QVBoxLayout()
        title = QLabel(self.mac_info.chip)
        title.setObjectName("hardwareTitle")
        details = QLabel(
            f"{self.mac_info.memory_gib:.0f} GB unified memory · "
            f"{self.mac_info.architecture} · {self.mac_info.metal_support}"
        )
        details.setObjectName("secondaryText")
        hardware_text.addWidget(title)
        hardware_text.addWidget(details)
        hardware_layout.addLayout(hardware_text)
        hardware_layout.addStretch()
        self.recommendation_label = QLabel()
        self.recommendation_label.setObjectName("recommendation")
        hardware_layout.addWidget(self.recommendation_label)
        root.addWidget(hardware)

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.setHandleWidth(8)
        root.addWidget(self.content_splitter, 1)

        controls = QWidget()
        controls.setObjectName("controlsPanel")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)
        controls.setMinimumWidth(390)
        self.content_splitter.addWidget(controls)

        controls_layout.addWidget(self._section_label("Preset"))
        preset_row = QHBoxLayout()
        preset_row.setSpacing(6)
        preset_group = QButtonGroup(self)
        preset_group.setExclusive(True)
        for name in PRESET_ORDER:
            button = QPushButton(preset_for(name, self.mac_info).label)
            button.setCheckable(True)
            button.setObjectName("presetButton")
            button.clicked.connect(lambda checked=False, key=name: self.apply_preset(key))
            preset_group.addButton(button)
            preset_row.addWidget(button)
            self._preset_buttons[name] = button
        self.custom_button = QPushButton("Custom")
        self.custom_button.setCheckable(True)
        self.custom_button.setObjectName("presetButton")
        self.custom_button.clicked.connect(self.activate_custom)
        preset_group.addButton(self.custom_button)
        preset_row.addWidget(self.custom_button)
        self._preset_buttons["custom"] = self.custom_button
        controls_layout.addLayout(preset_row)
        self.preset_description = QLabel()
        self.preset_description.setWordWrap(True)
        self.preset_description.setObjectName("secondaryText")
        controls_layout.addWidget(self.preset_description)

        self.model_edit = self._path_field(
            controls_layout,
            "Model folder",
            directory=True,
        )
        self.reference_edit = self._path_field(
            controls_layout,
            "Reference image",
            image=True,
            optional=True,
        )
        self.reference_edit.editingFinished.connect(self._convert_reference_field)
        self.reference_status = QLabel(
            "iPhone HEIC/HEIF photos are converted to PNG automatically."
        )
        self.reference_status.setWordWrap(True)
        self.reference_status.setObjectName("secondaryText")
        controls_layout.addWidget(self.reference_status)

        controls_layout.addWidget(self._field_label("Prompt"))
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setAcceptRichText(False)
        self.prompt_edit.setPlaceholderText(
            "Describe the scene, action, camera, and audio…"
        )
        self.prompt_edit.setMinimumHeight(105)
        controls_layout.addWidget(self.prompt_edit)

        format_row = QHBoxLayout()
        format_form = QFormLayout()
        self.format_combo = QComboBox()
        self.format_combo.addItem("Compact · 256 × 256", (256, 256))
        self.format_combo.addItem("Square · 512 × 512", (512, 512))
        self.format_combo.addItem("Portrait · 480 × 864", (480, 864))
        self.format_combo.addItem("Landscape · 864 × 480", (864, 480))
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        format_form.addRow("Format", self.format_combo)
        duration_form = QFormLayout()
        self.duration_combo = QComboBox()
        for seconds in (1, 2, 4, 6, 10):
            self.duration_combo.addItem(f"{seconds} seconds", seconds)
        duration_form.addRow("Duration", self.duration_combo)
        format_row.addLayout(format_form, 1)
        format_row.addLayout(duration_form, 1)
        controls_layout.addLayout(format_row)

        self.live_preview_check = QCheckBox(
            "Live preview during denoising (about +10 GB memory)"
        )
        self.live_preview_check.setToolTip("Enable this before starting generation.")
        self.live_preview_check.toggled.connect(self._update_preview_warning)
        controls_layout.addWidget(self.live_preview_check)
        self.preview_warning = QLabel()
        self.preview_warning.setWordWrap(True)
        self.preview_warning.setObjectName("warningText")
        controls_layout.addWidget(self.preview_warning)

        self.advanced_check = QCheckBox("Show advanced parameters")
        self.advanced_check.toggled.connect(self._set_advanced_visible)
        controls_layout.addWidget(self.advanced_check)
        self.advanced_group = QGroupBox("Advanced parameters")
        advanced = QGridLayout(self.advanced_group)
        self.steps_spin = self._spin(2, 1000)
        self.layers_spin = self._spin(1, 50)
        self.reuse_spin = self._spin(1, 3)
        self.core_reuse_spin = self._spin(1, 20)
        self.render_width_spin = self._spin(0, 2048, "Native")
        self.render_height_spin = self._spin(0, 2048, "Native")
        self.seed_spin = self._spin(0, 2_147_483_647)
        self.ssd_streaming_check = QCheckBox("SSD streaming")
        fields = (
            ("Step", self.steps_spin),
            ("Layer", self.layers_spin),
            ("Reuse", self.reuse_spin),
            ("Core reuse", self.core_reuse_spin),
            ("Render W", self.render_width_spin),
            ("Render H", self.render_height_spin),
            ("Seed", self.seed_spin),
        )
        for index, (label, widget) in enumerate(fields):
            row, column = divmod(index, 3)
            box = QVBoxLayout()
            box.addWidget(self._field_label(label))
            box.addWidget(widget)
            advanced.addLayout(box, row, column)
        advanced.addWidget(self.ssd_streaming_check, 3, 0, 1, 3)
        self.advanced_group.setVisible(False)
        controls_layout.addWidget(self.advanced_group)
        controls_layout.addStretch()

        right = QWidget()
        right.setObjectName("previewColumn")
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        right.setMinimumWidth(360)
        self.content_splitter.addWidget(right)
        self.content_splitter.setStretchFactor(0, 11)
        self.content_splitter.setStretchFactor(1, 9)
        self.content_splitter.setSizes((560, 460))
        right_layout.addWidget(self._section_label("Preview"))

        self.preview_stack = QStackedWidget()
        self.preview_stack.setObjectName("previewPanel")
        self.preview_placeholder = QLabel(
            "No generation started\n\n"
            "Enable Live preview before generating to see denoising frames here."
        )
        self.preview_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_placeholder.setWordWrap(True)
        self.preview_placeholder.setObjectName("previewPlaceholder")
        self.preview_label = ScalablePreviewLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self.preview_stack.addWidget(self.preview_placeholder)
        self.preview_stack.addWidget(self.preview_label)
        right_layout.addWidget(self.preview_stack, 1)

        status_row = QHBoxLayout()
        self.phase_label = QLabel("Ready")
        self.eta_label = QLabel("ETA —")
        self.eta_label.setObjectName("secondaryText")
        status_row.addWidget(self.phase_label)
        status_row.addStretch()
        status_row.addWidget(self.eta_label)
        right_layout.addLayout(status_row)
        self.progress_bar = SegmentedProgressBar()
        right_layout.addWidget(self.progress_bar)

        self.output_edit = self._path_field(
            right_layout,
            "Output video",
            save_file=True,
        )

        log_header = QHBoxLayout()
        log_header.addWidget(self._section_label("Log"))
        log_header.addStretch()
        self.log_toggle = QPushButton("Show")
        self.log_toggle.setCheckable(True)
        self.log_toggle.toggled.connect(self._toggle_log)
        log_header.addWidget(self.log_toggle)
        right_layout.addLayout(log_header)
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumBlockCount(500)
        self.log_edit.setVisible(False)
        self.log_edit.setMaximumHeight(150)
        right_layout.addWidget(self.log_edit)

        action_row = QHBoxLayout()
        self.generate_button = QPushButton("Generate video")
        self.generate_button.setObjectName("primaryButton")
        self.generate_button.clicked.connect(self._start_generation)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_generation)
        self.open_button = QPushButton("Open video")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_output)
        action_row.addWidget(self.generate_button, 1)
        action_row.addWidget(self.stop_button)
        action_row.addWidget(self.open_button)
        right_layout.addLayout(action_row)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #171816; }
            QWidget#contentRoot, QWidget#controlsPanel, QWidget#previewColumn,
            QScrollArea#contentScroll, QScrollArea#contentScroll > QWidget > QWidget {
                background: #171816; border: 0;
            }
            QWidget { color: #f1f3ed; font-size: 13px; }
            QFrame#hardwarePanel, QGroupBox, QStackedWidget#previewPanel {
                background: #252723; border: 1px solid #41463e; border-radius: 9px;
            }
            QLabel#hardwareTitle { font-weight: 600; font-size: 15px; }
            QLabel#secondaryText { color: #aeb5a7; }
            QLabel#recommendation { color: #65c58e; }
            QLabel#warningText { color: #e9b866; }
            QLabel#previewPlaceholder { color: #aeb5a7; padding: 24px; }
            QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox {
                background: #222320; border: 1px solid #41463e; border-radius: 6px;
                padding: 7px; selection-background-color: #4a9c6d;
            }
            QPushButton { background: #2c2e2a; border: 1px solid #41463e;
                border-radius: 6px; padding: 8px 12px; }
            QPushButton:hover { background: #353832; }
            QPushButton#presetButton:checked, QPushButton#primaryButton {
                background: #65c58e; color: #102218; border-color: #65c58e;
            }
            QProgressBar { background: #353832; border: 0; border-radius: 3px; }
            QProgressBar::chunk { background: #65c58e; border-radius: 3px; }
            QSplitter::handle { background: #171816; }
            QCheckBox { spacing: 8px; }
            QGroupBox { margin-top: 8px; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            """
        )

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        return label

    def _field_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("secondaryText")
        return label

    def _spin(self, minimum: int, maximum: int, special: str = "") -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        if special:
            spin.setSpecialValueText(special)
        return spin

    def _path_field(
        self,
        layout: QVBoxLayout,
        label: str,
        *,
        directory: bool = False,
        image: bool = False,
        save_file: bool = False,
        optional: bool = False,
    ) -> QLineEdit:
        suffix = " (optional)" if optional else ""
        layout.addWidget(self._field_label(label + suffix))
        row = QHBoxLayout()
        edit = QLineEdit()
        browse = QPushButton("Choose…")

        def choose() -> None:
            if directory:
                selected = QFileDialog.getExistingDirectory(self, label, edit.text())
            elif save_file:
                selected, _ = QFileDialog.getSaveFileName(
                    self, label, edit.text(), "MP4 video (*.mp4)"
                )
            elif image:
                selected, _ = QFileDialog.getOpenFileName(
                    self,
                    label,
                    edit.text(),
                    REFERENCE_IMAGE_FILE_FILTER,
                )
            else:
                selected, _ = QFileDialog.getOpenFileName(self, label, edit.text())
            if selected:
                edit.setText(selected)
                if image:
                    self._convert_reference_field()

        browse.clicked.connect(choose)
        row.addWidget(edit, 1)
        row.addWidget(browse)
        layout.addLayout(row)
        return edit

    def _set_default_paths(self) -> None:
        model = self.repo_root / "MiniMax-H3"
        reference = self.repo_root.parent / "reference-images" / "michela-cutout-transparent.png"
        output = self.default_output_dir / "h3-studio.mp4"
        self.set_paths(
            model_dir=model,
            reference_image=reference if reference.exists() else None,
            output_path=output,
        )

    def set_paths(
        self,
        *,
        model_dir: Path,
        reference_image: Path | None,
        output_path: Path,
    ) -> None:
        self._reference_source = None
        self._converted_reference = None
        self.model_edit.setText(str(model_dir))
        self.reference_edit.setText(str(reference_image) if reference_image else "")
        self.output_edit.setText(str(output_path))
        self.model_edit.setCursorPosition(0)
        self.reference_edit.setCursorPosition(0)
        self.output_edit.setCursorPosition(0)

    def set_prompt(self, prompt: str) -> None:
        self.prompt_edit.setPlainText(prompt)

    def set_reference_image(self, source: Path) -> Path:
        is_heic = requires_png_conversion(source)
        converted = self.reference_converter(
            source,
            self._reference_output_dir(),
        )
        self._reference_source = source.expanduser() if is_heic else None
        self._converted_reference = converted.expanduser() if is_heic else None
        self.reference_edit.setText(str(converted))
        self.reference_edit.setCursorPosition(0)
        if is_heic:
            self.reference_status.setText(
                f"HEIC converted automatically: {converted}"
            )
        else:
            self.reference_status.setText(
                "iPhone HEIC/HEIF photos are converted to PNG automatically."
            )
        return converted

    def _reference_output_dir(self) -> Path:
        output_text = self.output_edit.text().strip()
        output_dir = (
            Path(output_text).expanduser().resolve().parent
            if output_text
            else self.default_output_dir
        )
        return output_dir / "reference-images"

    def _convert_reference_field(self) -> bool:
        text = self.reference_edit.text().strip()
        if not text:
            self._reference_source = None
            self._converted_reference = None
            return True
        current = Path(text).expanduser()
        if not requires_png_conversion(current):
            if (
                self._reference_source is None
                or self._converted_reference is None
                or current.resolve() != self._converted_reference.resolve()
            ):
                self._reference_source = None
                self._converted_reference = None
                return True
            if current.resolve().parent == self._reference_output_dir().resolve():
                return True
        source = (
            current if requires_png_conversion(current) else self._reference_source
        )
        if source is None:
            return True
        try:
            self.set_reference_image(source)
        except ImageConversionError as error:
            QMessageBox.critical(self, "HEIC conversion failed", str(error))
            return False
        return True

    def apply_preset(self, name: str) -> None:
        preset = preset_for(name, self.mac_info)
        self._active_preset = name
        self._preset_buttons[name].setChecked(True)
        self.select_format(preset.width, preset.height)
        self.preset_description.setText(preset.description)
        self.recommendation_label.setText(
            f"Recommended preset: {preset_for(recommended_preset_name(self.mac_info), self.mac_info).label}"
        )
        self.steps_spin.setValue(preset.steps)
        self.layers_spin.setValue(preset.layers)
        self.reuse_spin.setValue(preset.reuse)
        self.core_reuse_spin.setValue(preset.core_reuse)
        self.render_width_spin.setValue(preset.render_width or 0)
        self.render_height_spin.setValue(preset.render_height or 0)
        self.seed_spin.setValue(42)
        self.ssd_streaming_check.setChecked(preset.ssd_streaming)
        index = self.duration_combo.findData(preset.seconds)
        if index >= 0:
            self.duration_combo.setCurrentIndex(index)
        self.live_preview_check.setChecked(preset.live_preview)
        self._on_format_changed()

    def activate_custom(self, checked: bool = False) -> None:
        del checked
        self._active_preset = "custom"
        self.custom_button.setChecked(True)
        self.preset_description.setText(
            "Adjust every generation setting in the advanced panel."
        )
        self.advanced_check.setChecked(True)

    def select_format(self, width: int, height: int) -> None:
        for index in range(self.format_combo.count()):
            if self.format_combo.itemData(index) == (width, height):
                self.format_combo.setCurrentIndex(index)
                return
        raise ValueError(f"unsupported format: {width}x{height}")

    def select_duration(self, seconds: int) -> None:
        index = self.duration_combo.findData(seconds)
        if index < 0:
            raise ValueError(f"unsupported duration: {seconds}")
        self.duration_combo.setCurrentIndex(index)

    def _on_format_changed(self) -> None:
        if self._active_preset != "fast":
            return
        width, height = self.format_combo.currentData()
        if width == height:
            render = (256, 256) if width == 256 else (320, 320)
        elif width > height:
            render = (576, 320)
        else:
            render = (320, 576)
        self.render_width_spin.setValue(render[0])
        self.render_height_spin.setValue(render[1])

    def _update_preview_warning(self, enabled: bool) -> None:
        self.preview_warning.setText(
            "On a 48 GB Mac this can increase memory pressure and slow generation."
            if enabled and self.mac_info.memory_gib <= 48
            else ""
        )

    def _set_advanced_visible(self, visible: bool) -> None:
        self.advanced_group.setVisible(visible)

    def _toggle_log(self, visible: bool) -> None:
        self.log_edit.setVisible(visible)
        self.log_toggle.setText("Hide" if visible else "Show")

    def generation_settings(
        self, preview_dir: Path | None = None
    ) -> GenerationSettings:
        width, height = self.format_combo.currentData()
        render_width = self.render_width_spin.value() or None
        render_height = self.render_height_spin.value() or None
        reference_text = self.reference_edit.text().strip()
        model_text = self.model_edit.text().strip()
        output_text = self.output_edit.text().strip()
        if not model_text:
            raise ValueError("Choose the model folder.")
        if not output_text:
            raise ValueError("Choose the output video file.")
        return GenerationSettings(
            model_dir=Path(model_text).expanduser(),
            prompt=self.prompt_edit.toPlainText().strip(),
            output_path=Path(output_text).expanduser(),
            reference_image=Path(reference_text).expanduser() if reference_text else None,
            width=width,
            height=height,
            render_width=render_width,
            render_height=render_height,
            seconds=int(self.duration_combo.currentData()),
            steps=self.steps_spin.value(),
            layers=self.layers_spin.value(),
            reuse=self.reuse_spin.value(),
            core_reuse=self.core_reuse_spin.value(),
            seed=self.seed_spin.value(),
            ssd_streaming=self.ssd_streaming_check.isChecked(),
            live_preview=self.live_preview_check.isChecked(),
            preview_dir=preview_dir,
        )

    def _validate(self, settings: GenerationSettings) -> str | None:
        executable = self.repo_root / "h3"
        if not executable.is_file():
            return "The h3 executable was not found. Run ‘make h3’ first."
        if not settings.model_dir.is_dir():
            return "The model folder does not exist."
        if not settings.prompt:
            return "Enter a prompt."
        if settings.reference_image is not None and not settings.reference_image.is_file():
            return "The reference image does not exist."
        if settings.reuse > 1 and settings.core_reuse > 1:
            return "Reuse and core reuse cannot both be greater than 1."
        return None

    def _start_generation(self) -> None:
        if self.runner.running:
            return
        if not self._convert_reference_field():
            return
        preview_dir = None
        if self.live_preview_check.isChecked():
            self._preview_temp = tempfile.TemporaryDirectory(prefix="h3-studio-preview-")
            preview_dir = Path(self._preview_temp.name)
        try:
            settings = self.generation_settings(preview_dir)
        except ValueError as settings_error:
            self._cleanup_preview_temp()
            QMessageBox.warning(self, "Cannot generate", str(settings_error))
            return
        validation_error = self._validate(settings)
        if validation_error:
            self._cleanup_preview_temp()
            QMessageBox.warning(self, "Cannot generate", validation_error)
            return
        self.log_edit.clear()
        self.progress_bar.reset()
        self.phase_label.setText("Starting…")
        self.eta_label.setText("ETA —")
        self._preview_received = False
        self.preview_stack.setCurrentWidget(self.preview_placeholder)
        self.preview_placeholder.setText(
            "Preparing live preview…\n\n"
            "The first frame appears during stage 2/3: Generation."
            if settings.live_preview
            else "Live preview is off for this generation."
        )
        self.generate_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.open_button.setEnabled(False)
        self.live_preview_check.setEnabled(False)
        callbacks = RunnerCallbacks(
            on_progress=self._bridge.progress.emit,
            on_output=self._bridge.output.emit,
            on_preview=lambda path: self._bridge.preview.emit(str(path)),
            on_finished=self._bridge.finished.emit,
        )
        try:
            self.runner.start(settings, callbacks)
        except (OSError, RuntimeError) as error:
            self._on_start_error(str(error))

    def _on_start_error(self, message: str) -> None:
        self.generate_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.live_preview_check.setEnabled(True)
        self.phase_label.setText("Start error")
        if self.live_preview_check.isChecked() and not self._preview_received:
            self.preview_placeholder.setText(
                "Live preview is unavailable because generation could not start."
            )
        self._cleanup_preview_temp()
        QMessageBox.critical(self, "Error", message)

    def _stop_generation(self) -> None:
        self.phase_label.setText("Stopping…")
        self.stop_button.setEnabled(False)
        self.runner.stop()

    def _on_progress(self, update: ProgressUpdate) -> None:
        self.phase_label.setText(
            f"{update.stage_index + 1}/3 {update.stage} · "
            f"{update.phase} {update.completed}/{update.total}"
        )
        self.progress_bar.set_overall_percent(update.percent)
        if update.eta_seconds is None:
            self.eta_label.setText("Calculating ETA…")
        else:
            self.eta_label.setText(f"ETA {self._format_duration(update.eta_seconds)}")

    def _append_log(self, line: str) -> None:
        self.log_edit.appendPlainText(line)

    def _on_preview(self, path_text: str) -> None:
        pixmap = QPixmap(path_text)
        if pixmap.isNull():
            return
        self.preview_label.set_source_pixmap(pixmap)
        self.preview_stack.setCurrentWidget(self.preview_label)
        self._preview_received = True

    def _on_finished(self, result: RunResult) -> None:
        self.generate_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.live_preview_check.setEnabled(True)
        if result.cancelled:
            self.phase_label.setText("Generation stopped")
        elif result.exit_code == 0:
            self.progress_bar.complete()
            self.phase_label.setText("Video complete")
            self.eta_label.setText("ETA 0s")
            self._last_output = result.output_path
            self.open_button.setEnabled(result.output_path.exists())
        else:
            self.phase_label.setText(f"Error · exit code {result.exit_code}")
            self.log_edit.setVisible(True)
            self.log_toggle.setChecked(True)
        if self.live_preview_check.isChecked() and not self._preview_received:
            if result.cancelled:
                preview_message = "Generation stopped before a preview frame was ready."
            elif result.exit_code == 0:
                preview_message = "Generation completed without a preview frame."
            else:
                preview_message = "Live preview failed because generation ended with an error."
            self.preview_placeholder.setText(preview_message)
            self.preview_stack.setCurrentWidget(self.preview_placeholder)
        self._cleanup_preview_temp()

    def _open_output(self) -> None:
        if self._last_output and self._last_output.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output)))

    def _cleanup_preview_temp(self) -> None:
        if self._preview_temp is not None:
            self._preview_temp.cleanup()
            self._preview_temp = None

    def _format_duration(self, seconds: float) -> str:
        rounded = max(0, round(seconds))
        minutes, remaining = divmod(rounded, 60)
        if minutes:
            return f"{minutes}m {remaining:02d}s"
        return f"{remaining}s"

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_responsive_layout(event.size().width())

    def _update_responsive_layout(self, width: int) -> None:
        if not hasattr(self, "content_splitter"):
            return
        orientation = (
            Qt.Orientation.Vertical if width < 900 else Qt.Orientation.Horizontal
        )
        if self.content_splitter.orientation() == orientation:
            return
        self.content_splitter.setOrientation(orientation)
        self.content_splitter.setSizes(
            (620, 540) if orientation == Qt.Orientation.Vertical else (560, 460)
        )

    def _load_preferences(self) -> None:
        model = self._setting_text("model_dir", self.model_edit.text())
        reference = self._setting_text(
            "reference_image", self.reference_edit.text()
        )
        reference_source = self._setting_text("reference_source", "")
        output = self._setting_text("output_path", self.output_edit.text())
        prompt = self._setting_text("prompt", "")
        preset = self._setting_text("preset", self._active_preset)
        if preset == "custom":
            self.activate_custom()
        elif preset in self._preset_buttons:
            self.apply_preset(preset)
        width = self._setting_int("format_width", self.format_combo.currentData()[0])
        height = self._setting_int("format_height", self.format_combo.currentData()[1])
        try:
            self.select_format(width, height)
        except ValueError:
            pass
        try:
            self.select_duration(
                self._setting_int("seconds", int(self.duration_combo.currentData()))
            )
        except ValueError:
            pass
        self.steps_spin.setValue(self._setting_int("steps", self.steps_spin.value()))
        self.layers_spin.setValue(self._setting_int("layers", self.layers_spin.value()))
        self.reuse_spin.setValue(self._setting_int("reuse", self.reuse_spin.value()))
        self.core_reuse_spin.setValue(
            self._setting_int("core_reuse", self.core_reuse_spin.value())
        )
        self.render_width_spin.setValue(
            self._setting_int("render_width", self.render_width_spin.value())
        )
        self.render_height_spin.setValue(
            self._setting_int("render_height", self.render_height_spin.value())
        )
        self.seed_spin.setValue(self._setting_int("seed", self.seed_spin.value()))
        self.ssd_streaming_check.setChecked(
            self._setting_bool("ssd_streaming", self.ssd_streaming_check.isChecked())
        )
        self.live_preview_check.setChecked(
            self._setting_bool("live_preview", self.live_preview_check.isChecked())
        )
        self.advanced_check.setChecked(
            self._setting_bool("advanced_visible", self.advanced_check.isChecked())
        )
        self.model_edit.setText(model)
        self.reference_edit.setText(reference)
        self.output_edit.setText(output)
        self.model_edit.setCursorPosition(0)
        self.reference_edit.setCursorPosition(0)
        self.output_edit.setCursorPosition(0)
        self.prompt_edit.setPlainText(prompt)
        if reference_source and requires_png_conversion(Path(reference_source)):
            self._reference_source = Path(reference_source).expanduser()
            self._converted_reference = Path(reference).expanduser()
        else:
            self._reference_source = None
            self._converted_reference = None

    def _setting_text(self, key: str, default: str) -> str:
        value = self._settings.value(key, default)
        return value if isinstance(value, str) else default

    def _setting_int(self, key: str, default: int) -> int:
        value = self._settings.value(key, default)
        if not isinstance(value, (int, str)):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _setting_bool(self, key: str, default: bool) -> bool:
        value = self._settings.value(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return default

    def _save_preferences(self) -> None:
        if not self._persistence_enabled:
            return
        self._settings.setValue("model_dir", self.model_edit.text())
        self._settings.setValue("reference_image", self.reference_edit.text())
        self._settings.setValue(
            "reference_source",
            str(self._reference_source) if self._reference_source else "",
        )
        self._settings.setValue("output_path", self.output_edit.text())
        self._settings.setValue("prompt", self.prompt_edit.toPlainText())
        self._settings.setValue("preset", self._active_preset)
        width, height = self.format_combo.currentData()
        self._settings.setValue("format_width", width)
        self._settings.setValue("format_height", height)
        self._settings.setValue("seconds", self.duration_combo.currentData())
        self._settings.setValue("steps", self.steps_spin.value())
        self._settings.setValue("layers", self.layers_spin.value())
        self._settings.setValue("reuse", self.reuse_spin.value())
        self._settings.setValue("core_reuse", self.core_reuse_spin.value())
        self._settings.setValue("render_width", self.render_width_spin.value())
        self._settings.setValue("render_height", self.render_height_spin.value())
        self._settings.setValue("seed", self.seed_spin.value())
        self._settings.setValue(
            "ssd_streaming", self.ssd_streaming_check.isChecked()
        )
        self._settings.setValue("live_preview", self.live_preview_check.isChecked())
        self._settings.setValue("advanced_visible", self.advanced_check.isChecked())
        self._settings.sync()

    def closeEvent(self, event) -> None:
        if self.runner.running:
            answer = QMessageBox.question(
                self,
                "Generation in progress",
                "Stop generation and close H3 Studio?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.runner.stop()
        self._save_preferences()
        self._cleanup_preview_temp()
        event.accept()
