from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class ChannelControls(QGroupBox):
    def __init__(
        self,
        channel_id: str,
        *,
        initial_visible: bool,
        initial_opacity: float,
        on_visible_changed: Callable[[str, bool], None],
        on_opacity_changed: Callable[[str, float], None],
        on_contrast_changed: Callable[[str, float, float], None],
        parent=None,
    ) -> None:
        super().__init__(channel_id.capitalize(), parent)
        self.channel_id = channel_id
        layout = QFormLayout()

        self.visible_checkbox = QCheckBox("Visible")
        self.visible_checkbox.setChecked(initial_visible)
        self.visible_checkbox.toggled.connect(lambda v: on_visible_changed(channel_id, v))
        layout.addRow(self.visible_checkbox)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(initial_opacity * 100))
        self.opacity_slider.valueChanged.connect(lambda v: on_opacity_changed(channel_id, v / 100.0))
        layout.addRow("Opacity", self.opacity_slider)

        self.contrast_min = QDoubleSpinBox()
        self.contrast_min.setRange(0, 1e9)
        self.contrast_max = QDoubleSpinBox()
        self.contrast_max.setRange(0, 1e9)
        self.contrast_max.setValue(1000.0)
        def _emit_contrast_changed(_v):
            on_contrast_changed(channel_id, self.contrast_min.value(), self.contrast_max.value())

        for w in (self.contrast_min, self.contrast_max):
            w.valueChanged.connect(_emit_contrast_changed)
        layout.addRow("Contrast min", self.contrast_min)
        layout.addRow("Contrast max", self.contrast_max)

        self.setLayout(layout)


class ImagePanel(QGroupBox):
    def __init__(
        self,
        *,
        channels: list[str],
        z_count: int,
        on_visible_changed: Callable[[str, bool], None],
        on_opacity_changed: Callable[[str, float], None],
        on_contrast_changed: Callable[[str, float, float], None],
        on_z_mode_changed: Callable[[str], None],
        on_z_index_changed: Callable[[int], None],
        on_z_range_changed: Callable[[int, int], None],
        parent=None,
    ) -> None:
        super().__init__("Images", parent)
        layout = QVBoxLayout()

        self.channel_controls: dict[str, ChannelControls] = {}
        for channel_id in channels:
            ctrl = ChannelControls(
                channel_id,
                initial_visible=True,
                initial_opacity=1.0,
                on_visible_changed=on_visible_changed,
                on_opacity_changed=on_opacity_changed,
                on_contrast_changed=on_contrast_changed,
            )
            self.channel_controls[channel_id] = ctrl
            layout.addWidget(ctrl)

        z_widget = QWidget()
        z_form = QFormLayout()
        self.z_mode_combo = QComboBox()
        self.z_mode_combo.addItems(["max_projection", "single", "max_projection_range"])
        self.z_mode_combo.currentTextChanged.connect(on_z_mode_changed)
        z_form.addRow("Z mode", self.z_mode_combo)

        self.z_index_spin = QSpinBox()
        self.z_index_spin.setRange(0, max(z_count - 1, 0))
        self.z_index_spin.valueChanged.connect(on_z_index_changed)
        z_form.addRow("Z index", self.z_index_spin)

        self.z_range_min = QSpinBox()
        self.z_range_min.setRange(0, max(z_count - 1, 0))
        self.z_range_max = QSpinBox()
        self.z_range_max.setRange(0, max(z_count - 1, 0))
        self.z_range_max.setValue(max(z_count - 1, 0))
        for w in (self.z_range_min, self.z_range_max):
            w.valueChanged.connect(lambda _v: on_z_range_changed(self.z_range_min.value(), self.z_range_max.value()))
        z_form.addRow("Z range min", self.z_range_min)
        z_form.addRow("Z range max", self.z_range_max)

        z_widget.setLayout(z_form)
        layout.addWidget(z_widget)

        layout.addWidget(QLabel(f"{z_count} z-plane(s) available"))
        self.setLayout(layout)
