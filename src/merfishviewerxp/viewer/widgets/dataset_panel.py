from __future__ import annotations

from collections.abc import Callable

from qtpy.QtWidgets import QGroupBox, QLabel, QPushButton, QVBoxLayout


class DatasetPanel(QGroupBox):
    def __init__(
        self,
        *,
        experiment_path: str,
        cache_status: str,
        on_rebuild_cache: Callable[[], None],
        on_diagnostics: Callable[[], None],
        parent=None,
    ) -> None:
        super().__init__("Dataset", parent)
        layout = QVBoxLayout()

        self.path_label = QLabel(f"Experiment: {experiment_path}")
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

        self.status_label = QLabel(f"Cache status: {cache_status}")
        layout.addWidget(self.status_label)

        rebuild_btn = QPushButton("Rebuild cache")
        rebuild_btn.clicked.connect(on_rebuild_cache)
        layout.addWidget(rebuild_btn)

        diagnostics_btn = QPushButton("Diagnostics...")
        diagnostics_btn.clicked.connect(on_diagnostics)
        layout.addWidget(diagnostics_btn)

        self.setLayout(layout)

    def set_cache_status(self, status: str) -> None:
        self.status_label.setText(f"Cache status: {status}")
