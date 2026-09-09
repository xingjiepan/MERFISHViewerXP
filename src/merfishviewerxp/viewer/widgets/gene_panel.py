from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

GENE_ID_ROLE = Qt.ItemDataRole.UserRole

# napari's supported Points symbols (napari.layers.points._points_constants.Symbol)
AVAILABLE_SYMBOLS = [
    "disc",
    "square",
    "diamond",
    "triangle_up",
    "triangle_down",
    "star",
    "cross",
    "x",
    "ring",
    "arrow",
    "tailed_arrow",
    "clobber",
    "hbar",
    "vbar",
]


def _swatch_icon(color_hex: str) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(color_hex))
    return QIcon(pixmap)


class GenePanel(QGroupBox):
    """Per-codebook controls: visible, spot symbol, gene search/selection.

    One instance is created per codebook so that, when a dataset has
    multiple codebooks, each gets its own independent panel and its own
    decoded-spot symbol (spec-adjacent extension requested by the user).
    """

    def __init__(
        self,
        *,
        codebook_id: str,
        genes: pd.DataFrame,
        initial_active_gene_ids: set[int],
        initial_symbol: str,
        initial_visible: bool,
        on_selection_changed: Callable[[str, set[int]], None],
        on_symbol_changed: Callable[[str, str], None],
        on_visible_changed: Callable[[str, bool], None],
        parent=None,
    ) -> None:
        super().__init__(f"Codebook {codebook_id}", parent)
        self.codebook_id = codebook_id
        self._on_selection_changed = on_selection_changed
        self._suspend_signal = False

        layout = QVBoxLayout()

        top_row = QHBoxLayout()
        self.visible_checkbox = QCheckBox("Visible")
        self.visible_checkbox.setChecked(initial_visible)
        self.visible_checkbox.toggled.connect(lambda v: on_visible_changed(codebook_id, v))
        top_row.addWidget(self.visible_checkbox)

        top_row.addWidget(QLabel("Symbol"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(AVAILABLE_SYMBOLS)
        self.symbol_combo.setCurrentText(initial_symbol)
        self.symbol_combo.currentTextChanged.connect(lambda s: on_symbol_changed(codebook_id, s))
        top_row.addWidget(self.symbol_combo)
        layout.addLayout(top_row)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search genes...")
        self.search_box.textChanged.connect(self._apply_filter)
        layout.addWidget(self.search_box)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.list_widget.itemChanged.connect(self._handle_item_changed)
        for row in genes.itertuples(index=False):
            label = f"{row.gene_name} ({row.spot_count})"
            item = QListWidgetItem(_swatch_icon(row.color_hex), label)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if int(row.gene_id) in initial_active_gene_ids else Qt.CheckState.Unchecked
            )
            item.setData(GENE_ID_ROLE, int(row.gene_id))
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        button_row = QHBoxLayout()
        all_btn = QPushButton("All")
        all_btn.clicked.connect(lambda: self._bulk_set(True))
        none_btn = QPushButton("None")
        none_btn.clicked.connect(lambda: self._bulk_set(False))
        invert_btn = QPushButton("Invert")
        invert_btn.clicked.connect(self._invert)
        for b in (all_btn, none_btn, invert_btn):
            button_row.addWidget(b)
        layout.addLayout(button_row)

        self.count_label = QLabel("Visible: 0 / 0")
        layout.addWidget(self.count_label)

        self.setLayout(layout)

    def _apply_filter(self, text: str) -> None:
        text = text.strip().lower()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(bool(text) and text not in item.text().lower())

    def active_gene_ids(self) -> set[int]:
        ids = set()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                ids.add(item.data(GENE_ID_ROLE))
        return ids

    def set_counts(self, shown: int) -> None:
        self.count_label.setText(f"Visible: {shown}")

    def _handle_item_changed(self, _item) -> None:
        if not self._suspend_signal:
            self._on_selection_changed(self.codebook_id, self.active_gene_ids())

    def _bulk_set(self, checked: bool) -> None:
        self._suspend_signal = True
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)
        self._suspend_signal = False
        self._on_selection_changed(self.codebook_id, self.active_gene_ids())

    def _invert(self) -> None:
        self._suspend_signal = True
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(
                Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked
            )
        self._suspend_signal = False
        self._on_selection_changed(self.codebook_id, self.active_gene_ids())
