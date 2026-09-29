from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
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


class ColorButton(QPushButton):
    """A swatch button that opens a color dialog and reports the chosen hex color."""

    def __init__(self, color_hex: str, *, title: str, on_color_changed: Callable[[str], None], parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self._on_color_changed = on_color_changed
        self.setFixedWidth(48)
        self.set_color(color_hex)
        self.clicked.connect(self._choose)

    @property
    def color(self) -> str:
        return self._color

    def set_color(self, color_hex: str) -> None:
        self._color = color_hex
        self.setStyleSheet(f"background-color: {color_hex}; border: 1px solid #888;")
        self.setToolTip(f"{self._title}: {color_hex} (click to change)")

    def _choose(self) -> None:
        color = QColorDialog.getColor(QColor(self._color), self, self._title)
        if not color.isValid():
            return
        self.set_color(color.name())
        self._on_color_changed(color.name())


class CellBoundaryControls(QGroupBox):
    def __init__(
        self,
        *,
        initial_visible: bool,
        initial_opacity: float,
        initial_default_color: str,
        initial_highlight_color: str,
        initial_colored_cell_count: int,
        on_visible_changed: Callable[[bool], None],
        on_opacity_changed: Callable[[float], None],
        on_default_color_changed: Callable[[str], None],
        on_highlight_color_changed: Callable[[str], None],
        on_coloring_mode_changed: Callable[[bool], None],
        on_clear_cell_colors: Callable[[], None],
        parent=None,
    ) -> None:
        super().__init__("Cell boundaries", parent)
        layout = QFormLayout()

        self.visible_checkbox = QCheckBox("Show segmentation boundaries")
        self.visible_checkbox.setChecked(initial_visible)
        self.visible_checkbox.toggled.connect(on_visible_changed)
        layout.addRow(self.visible_checkbox)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(int(initial_opacity * 100))
        self.opacity_slider.valueChanged.connect(lambda v: on_opacity_changed(v / 100.0))
        layout.addRow("Opacity", self.opacity_slider)

        self.default_color_button = ColorButton(
            initial_default_color, title="Cell boundary color", on_color_changed=on_default_color_changed
        )
        layout.addRow("Boundary color", self.default_color_button)

        self.highlight_color_button = ColorButton(
            initial_highlight_color, title="Color for clicked cells", on_color_changed=on_highlight_color_changed
        )
        self.color_cells_button = QPushButton("Click cells to color")
        self.color_cells_button.setCheckable(True)
        self.color_cells_button.setToolTip(
            "While on, clicking a cell gives its boundary the highlight color; clicking it again restores "
            "the boundary color."
        )
        self.color_cells_button.toggled.connect(on_coloring_mode_changed)
        color_row = QHBoxLayout()
        color_row.addWidget(self.highlight_color_button)
        color_row.addWidget(self.color_cells_button, 1)
        layout.addRow("Highlight", color_row)

        self.colored_count_label = QLabel()
        self.clear_button = QPushButton("Clear cell colors")
        self.clear_button.clicked.connect(on_clear_cell_colors)
        clear_row = QHBoxLayout()
        clear_row.addWidget(self.colored_count_label, 1)
        clear_row.addWidget(self.clear_button)
        layout.addRow(clear_row)
        self.set_colored_cell_count(initial_colored_cell_count)

        self.setLayout(layout)

    def set_colored_cell_count(self, n: int) -> None:
        self.colored_count_label.setText(f"{n} cell(s) recolored")
        self.clear_button.setEnabled(n > 0)


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
        cell_boundaries_available: bool = False,
        cell_boundary_options: dict | None = None,
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

        self.cell_boundary_controls: CellBoundaryControls | None = None
        if cell_boundaries_available:
            # `cell_boundary_options` holds CellBoundaryControls' keyword arguments.
            self.cell_boundary_controls = CellBoundaryControls(**(cell_boundary_options or {}))
            layout.addWidget(self.cell_boundary_controls)

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
