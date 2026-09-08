from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QIcon, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

GENE_ID_ROLE = Qt.ItemDataRole.UserRole


def _swatch_icon(color_hex: str) -> QIcon:
    pixmap = QPixmap(12, 12)
    pixmap.fill(QColor(color_hex))
    return QIcon(pixmap)


class GenePanel(QGroupBox):
    def __init__(
        self,
        *,
        genes: pd.DataFrame,
        initial_active_gene_ids: set[int],
        on_selection_changed: Callable[[set[int]], None],
        parent=None,
    ) -> None:
        super().__init__("Genes", parent)
        self._on_selection_changed = on_selection_changed
        self._suspend_signal = False

        layout = QVBoxLayout()

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

    def _handle_item_changed(self, _item) -> None:
        if not self._suspend_signal:
            self._on_selection_changed(self.active_gene_ids())

    def _bulk_set(self, checked: bool) -> None:
        self._suspend_signal = True
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)
        self._suspend_signal = False
        self._on_selection_changed(self.active_gene_ids())

    def _invert(self) -> None:
        self._suspend_signal = True
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(
                Qt.CheckState.Unchecked if item.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked
            )
        self._suspend_signal = False
        self._on_selection_changed(self.active_gene_ids())
