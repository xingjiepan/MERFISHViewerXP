from __future__ import annotations

from collections.abc import Callable

from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class QCPanel(QGroupBox):
    def __init__(
        self,
        *,
        max_fov_id: int,
        on_show_boundaries_changed: Callable[[bool], None],
        on_show_ids_changed: Callable[[bool], None],
        on_jump_to_fov: Callable[[int], None],
        on_jump_to_xy: Callable[[float, float], None],
        parent=None,
    ) -> None:
        super().__init__("Debug / QC", parent)
        layout = QVBoxLayout()

        self.boundaries_checkbox = QCheckBox("Show FOV boundaries")
        self.boundaries_checkbox.toggled.connect(on_show_boundaries_changed)
        layout.addWidget(self.boundaries_checkbox)

        self.ids_checkbox = QCheckBox("Show FOV IDs")
        self.ids_checkbox.toggled.connect(on_show_ids_changed)
        layout.addWidget(self.ids_checkbox)

        self.cursor_label = QLabel("world x/y: -")
        layout.addWidget(self.cursor_label)

        jump_fov_row = QHBoxLayout()
        self.fov_spin = QSpinBox()
        self.fov_spin.setRange(0, max(max_fov_id, 0))
        jump_fov_btn = QPushButton("Jump to FOV")
        jump_fov_btn.clicked.connect(lambda: on_jump_to_fov(self.fov_spin.value()))
        jump_fov_row.addWidget(self.fov_spin)
        jump_fov_row.addWidget(jump_fov_btn)
        layout.addLayout(jump_fov_row)

        jump_xy_form = QFormLayout()
        self.x_spin = QDoubleSpinBox()
        self.x_spin.setRange(-1e9, 1e9)
        self.y_spin = QDoubleSpinBox()
        self.y_spin.setRange(-1e9, 1e9)
        jump_xy_form.addRow("x (um)", self.x_spin)
        jump_xy_form.addRow("y (um)", self.y_spin)
        layout.addLayout(jump_xy_form)

        jump_xy_btn = QPushButton("Jump to x/y")
        jump_xy_btn.clicked.connect(lambda: on_jump_to_xy(self.x_spin.value(), self.y_spin.value()))
        layout.addWidget(jump_xy_btn)

        self.setLayout(layout)

    def set_cursor_world(self, x_um: float | None, y_um: float | None) -> None:
        if x_um is None:
            self.cursor_label.setText("world x/y: -")
        else:
            self.cursor_label.setText(f"world x/y: {x_um:.2f}, {y_um:.2f}")
