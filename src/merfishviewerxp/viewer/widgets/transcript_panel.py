from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QCheckBox, QComboBox, QFormLayout, QGroupBox, QLabel, QSlider


class TranscriptPanel(QGroupBox):
    def __init__(
        self,
        *,
        initial_point_size: float,
        initial_point_opacity: float,
        initial_include_blanks: bool,
        on_visible_changed: Callable[[bool], None],
        on_point_size_changed: Callable[[float], None],
        on_point_opacity_changed: Callable[[float], None],
        on_include_blanks_changed: Callable[[bool], None],
        on_display_mode_changed: Callable[[str], None],
        parent=None,
    ) -> None:
        super().__init__("Transcripts", parent)
        layout = QFormLayout()

        self.visible_checkbox = QCheckBox("Visible")
        self.visible_checkbox.setChecked(True)
        self.visible_checkbox.toggled.connect(on_visible_changed)
        layout.addRow(self.visible_checkbox)

        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(1, 50)
        self.size_slider.setValue(int(initial_point_size))
        self.size_slider.valueChanged.connect(lambda v: on_point_size_changed(float(v)))
        layout.addRow("Point size", self.size_slider)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(initial_point_opacity * 100))
        self.opacity_slider.valueChanged.connect(lambda v: on_point_opacity_changed(v / 100.0))
        layout.addRow("Opacity", self.opacity_slider)

        self.include_blanks_checkbox = QCheckBox("Include blanks")
        self.include_blanks_checkbox.setChecked(initial_include_blanks)
        self.include_blanks_checkbox.toggled.connect(on_include_blanks_changed)
        layout.addRow(self.include_blanks_checkbox)

        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems(["auto", "points"])
        self.display_mode_combo.currentTextChanged.connect(on_display_mode_changed)
        layout.addRow("Display mode", self.display_mode_combo)

        self.count_label = QLabel("Visible: 0 / 0")
        layout.addRow(self.count_label)

        self.setLayout(layout)

    def set_counts(self, shown: int, total_in_view: int, sampled: bool) -> None:
        suffix = " (sampled)" if sampled else ""
        self.count_label.setText(f"Visible: {shown} / {total_in_view}{suffix}")
