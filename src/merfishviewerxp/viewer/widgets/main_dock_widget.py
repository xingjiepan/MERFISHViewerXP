"""Assembles the MERFISHViewerXP dock widget from its logical sections (spec 13.2)."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from qtpy.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from .dataset_panel import DatasetPanel
from .gene_panel import GenePanel
from .image_panel import ImagePanel
from .qc_panel import QCPanel
from .transcript_panel import TranscriptPanel


class MainDockWidget(QScrollArea):
    def __init__(
        self,
        *,
        experiment_path: str,
        cache_status: str,
        channels: list[str],
        z_count: int,
        genes_by_codebook: dict[str, pd.DataFrame],
        initial_active_gene_ids_by_codebook: dict[str, set[int]],
        initial_symbols_by_codebook: dict[str, str],
        initial_visible_by_codebook: dict[str, bool],
        initial_point_size: float,
        initial_point_opacity: float,
        initial_include_blanks: bool,
        initial_display_mode: str,
        initial_transcripts_visible: bool,
        max_fov_id: int,
        callbacks: dict[str, Callable],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)

        content = QWidget()
        layout = QVBoxLayout()

        self.dataset_panel = DatasetPanel(
            experiment_path=experiment_path,
            cache_status=cache_status,
            on_rebuild_cache=callbacks["on_rebuild_cache"],
            on_diagnostics=callbacks["on_diagnostics"],
        )
        layout.addWidget(self.dataset_panel)

        self.image_panel = ImagePanel(
            channels=channels,
            z_count=z_count,
            on_visible_changed=callbacks["on_image_visible_changed"],
            on_opacity_changed=callbacks["on_image_opacity_changed"],
            on_contrast_changed=callbacks["on_image_contrast_changed"],
            on_z_mode_changed=callbacks["on_z_mode_changed"],
            on_z_index_changed=callbacks["on_z_index_changed"],
            on_z_range_changed=callbacks["on_z_range_changed"],
        )
        layout.addWidget(self.image_panel)

        self.transcript_panel = TranscriptPanel(
            initial_point_size=initial_point_size,
            initial_point_opacity=initial_point_opacity,
            initial_include_blanks=initial_include_blanks,
            initial_display_mode=initial_display_mode,
            initial_visible=initial_transcripts_visible,
            on_visible_changed=callbacks["on_transcripts_visible_changed"],
            on_point_size_changed=callbacks["on_point_size_changed"],
            on_point_opacity_changed=callbacks["on_point_opacity_changed"],
            on_include_blanks_changed=callbacks["on_include_blanks_changed"],
            on_display_mode_changed=callbacks["on_display_mode_changed"],
        )
        layout.addWidget(self.transcript_panel)

        self.codebook_panels: dict[str, GenePanel] = {}
        for codebook_id in sorted(genes_by_codebook):
            panel = GenePanel(
                codebook_id=codebook_id,
                genes=genes_by_codebook[codebook_id],
                initial_active_gene_ids=initial_active_gene_ids_by_codebook.get(codebook_id, set()),
                initial_symbol=initial_symbols_by_codebook.get(codebook_id, "disc"),
                initial_visible=initial_visible_by_codebook.get(codebook_id, False),
                on_selection_changed=callbacks["on_gene_selection_changed"],
                on_symbol_changed=callbacks["on_codebook_symbol_changed"],
                on_visible_changed=callbacks["on_codebook_visible_changed"],
            )
            self.codebook_panels[codebook_id] = panel
            layout.addWidget(panel)

        self.qc_panel = QCPanel(
            max_fov_id=max_fov_id,
            on_show_boundaries_changed=callbacks["on_show_fov_boundaries_changed"],
            on_show_ids_changed=callbacks["on_show_fov_ids_changed"],
            on_jump_to_fov=callbacks["on_jump_to_fov"],
            on_jump_to_xy=callbacks["on_jump_to_xy"],
        )
        layout.addWidget(self.qc_panel)

        layout.addStretch(1)
        content.setLayout(layout)
        self.setWidget(content)
