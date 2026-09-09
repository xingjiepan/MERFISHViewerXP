"""Wires the napari canvas, layers, dock widget, and background queries together."""

from __future__ import annotations

import logging

import napari
import numpy as np
from napari.utils.notifications import show_info
from qtpy.QtCore import QTimer
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QToolTip

from ..config import AppConfig
from ..indexed_dataset import IndexedDataset
from . import layers as layer_builders
from .state import ViewerState
from .widgets.gene_panel import AVAILABLE_SYMBOLS
from .widgets.main_dock_widget import MainDockWidget
from .workers import ViewportQueryRunner

logger = logging.getLogger(__name__)


class MerfishViewerXPApp:
    def __init__(self, indexed: IndexedDataset, config: AppConfig, *, viewer: napari.Viewer | None = None) -> None:
        self.indexed = indexed
        self.config = config
        self.viewer = viewer or napari.Viewer(title=f"MERFISHViewerXP - {indexed.dataset_id}")

        genes_df = indexed.genes()
        self._genes_by_codebook = layer_builders.genes_by_codebook(genes_df)
        gene_ids_by_codebook = {cb: [int(g) for g in df["gene_id"]] for cb, df in self._genes_by_codebook.items()}
        self._all_gene_ids = {int(g) for g in genes_df["gene_id"]}

        self.state = ViewerState.load_or_default(
            indexed.cache.settings_path,
            dataset_id=indexed.dataset_id,
            cache_path=indexed.cache.cache_dir,
            gene_ids_by_codebook=gene_ids_by_codebook,
        )
        self.state.max_visible_points = config.spots.max_visible_points
        if not self.state.include_blanks:
            self.state.include_blanks = config.spots.include_blanks_default
        # Fill in any codebook missing from a previously-saved settings.json
        # (e.g. the dataset gained a codebook since the state was last saved).
        for i, codebook_id in enumerate(sorted(gene_ids_by_codebook)):
            self.state.active_gene_ids_by_codebook.setdefault(codebook_id, list(gene_ids_by_codebook[codebook_id]))
            self.state.codebook_symbols.setdefault(codebook_id, AVAILABLE_SYMBOLS[i % len(AVAILABLE_SYMBOLS)])
            self.state.codebook_visible.setdefault(codebook_id, True)

        self.image_layers: dict[str, napari.layers.Image] = {}
        self.transcript_layers: dict[str, napari.layers.Points] = {}
        self._applying_query_result = False
        self._pending_query_result: tuple[object, dict] | None = None
        self._last_hover_gene: str | None = None

        self._build_image_layers()
        self._build_fov_layers()
        self._build_transcript_layers()

        self.query_runner = ViewportQueryRunner(indexed, on_result=self._on_query_result, on_error=self._on_query_error)

        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(config.viewer.viewport_debounce_ms)
        self._debounce_timer.timeout.connect(self._request_viewport_query)

        self.viewer.scene.camera.events.center.connect(self._on_camera_changed)
        self.viewer.scene.camera.events.zoom.connect(self._on_camera_changed)
        self.viewer.mouse_move_callbacks.append(self._on_mouse_move)

        self.dock_widget = MainDockWidget(
            experiment_path=str(indexed.cache.dataset_root),
            cache_status="valid",
            channels=indexed.channels(),
            z_count=self._z_count(),
            genes_by_codebook=self._genes_by_codebook,
            initial_active_gene_ids_by_codebook={
                cb: set(ids) for cb, ids in self.state.active_gene_ids_by_codebook.items()
            },
            initial_symbols_by_codebook=dict(self.state.codebook_symbols),
            initial_visible_by_codebook=dict(self.state.codebook_visible),
            initial_point_size=self.state.point_size,
            initial_point_opacity=self.state.point_opacity,
            initial_include_blanks=self.state.include_blanks,
            initial_display_mode=self.state.lod_mode,
            max_fov_id=int(indexed.fovs()["fov_id"].max()) if len(indexed.fovs()) else 0,
            callbacks=self._build_callbacks(),
        )
        self.viewer.window.add_dock_widget(self.dock_widget, area="right", name="MERFISHViewerXP")

        self._wire_point_inspection()
        self._request_viewport_query()

    # -- layer construction -------------------------------------------------

    def _z_count(self) -> int:
        for meta in self.indexed.mosaic_metadata().values():
            return int(meta["z_count"])
        return 1

    def _build_image_layers(self) -> None:
        for channel_id in self.indexed.channels():
            kwargs = layer_builders.image_layer_kwargs(
                self.indexed, channel_id, z_mode=self.state.z_mode, z_index=self.state.z_index, z_range=self.state.z_range
            )
            kwargs["visible"] = self.state.image_visibility.get(channel_id, True)
            kwargs["opacity"] = self.state.image_opacity.get(channel_id, 1.0)
            layer = self.viewer.add_image(**kwargs)
            self.image_layers[channel_id] = layer

    def _build_fov_layers(self) -> None:
        polygons, labels, centers = layer_builders.fov_boundary_polygons(self.indexed)
        self.fov_shapes_layer = self.viewer.add_shapes(
            polygons if polygons else None,
            shape_type="polygon",
            edge_color="yellow",
            face_color="transparent",
            edge_width=2,
            name="FOV boundaries",
            visible=self.state.show_fov_boundaries,
        )
        self.fov_labels_layer = self.viewer.add_points(
            centers if len(centers) else np.empty((0, 2)),
            features={"label": labels} if labels else {"label": []},
            text="label" if labels else None,
            size=0,
            face_color="transparent",
            name="FOV IDs",
            visible=self.state.show_fov_ids,
        )

    def _build_transcript_layers(self) -> None:
        for codebook_id in sorted(self._genes_by_codebook):
            layer = self.viewer.add_points(
                np.empty((0, 2)),
                name=f"Decoded transcripts ({codebook_id})",
                size=self.state.point_size,
                opacity=self.state.point_opacity,
                visible=self.state.codebook_visible.get(codebook_id, True),
                symbol=self.state.codebook_symbols.get(codebook_id, "disc"),
                border_width=0,
            )
            self.transcript_layers[codebook_id] = layer

    # -- viewport querying ----------------------------------------------------

    def _current_world_bounds(self) -> tuple[float, float, float, float]:
        return layer_builders.visible_world_bounds(self.viewer)

    def _on_camera_changed(self, _event=None) -> None:
        self._debounce_timer.start()

    def _request_viewport_query(self) -> None:
        active = self.state.all_active_gene_ids()
        gene_ids = None if active == self._all_gene_ids else list(active)
        self.query_runner.request(
            bounds_um=self._current_world_bounds(),
            gene_ids=gene_ids,
            include_blanks=self.state.include_blanks,
            max_visible_points=self.state.max_visible_points,
            apply_lod_sampling=self.state.lod_mode != "show_all",
        )

    def _on_query_result(self, table, lod_info: dict) -> None:
        # napari's Points layer is not safe against being mutated again while
        # a previous mutation is still being applied (its own internals
        # acknowledge this -- see the StatusChecker's docstring). Something
        # in the data/color/symbol update sequence below can apparently pump
        # the Qt event loop and let a second, newer query result be
        # delivered before this call returns. Rather than let two calls
        # interleave their writes to the same layers, we serialize: a
        # reentrant call just records itself as "pending" and returns; the
        # outer call keeps applying pending results until none remain, so
        # the layers always end up reflecting the latest query.
        if self._applying_query_result:
            self._pending_query_result = (table, lod_info)
            return
        self._applying_query_result = True
        try:
            self._apply_query_result(table, lod_info)
            while self._pending_query_result is not None:
                pending_table, pending_lod_info = self._pending_query_result
                self._pending_query_result = None
                self._apply_query_result(pending_table, pending_lod_info)
        finally:
            self._applying_query_result = False

    def _apply_query_result(self, table, lod_info: dict) -> None:
        per_codebook = layer_builders.split_table_by_codebook(table, list(self.transcript_layers.keys()))
        for codebook_id, sub_table in per_codebook.items():
            coords, colors, features = layer_builders.spot_table_to_points(sub_table)
            layer_builders.apply_points_update(
                self.transcript_layers[codebook_id],
                coords=coords,
                colors=colors,
                symbol=self.state.codebook_symbols.get(codebook_id, "disc"),
                features=features,
            )
            panel = self.dock_widget.codebook_panels.get(codebook_id)
            if panel is not None:
                panel.set_counts(sub_table.num_rows)
        self.dock_widget.transcript_panel.set_counts(lod_info["shown"], lod_info["total_in_view"], lod_info["sampled"])
        self.viewer.status = f"Visible transcripts: {lod_info['shown']} / {lod_info['total_in_view']}"

    def _on_query_error(self, exc: Exception) -> None:
        logger.exception("Viewport query failed", exc_info=exc)
        show_info(f"Transcript query failed: {exc}")

    # -- point inspection / cursor probe --------------------------------------

    @staticmethod
    def _feature_row(layer, index: int) -> dict:
        if hasattr(layer.features, "items"):
            return {k: v[index] for k, v in layer.features.items()}
        return layer.features.iloc[index].to_dict()

    def _find_hovered_spot(self, position) -> tuple[napari.layers.Points, int] | None:
        """The topmost visible transcript layer's point under `position`, if any."""
        transcript_layer_set = set(self.transcript_layers.values())
        for layer in reversed(self.viewer.layers):
            if layer not in transcript_layer_set or not layer.visible:
                continue
            index = layer.get_value(position, world=True)
            if index is not None:
                return layer, index
        return None

    def _on_transcript_click(self, layer, event) -> None:
        if event.type != "mouse_press":
            return
        index = layer.get_value(event.position, world=True)
        if index is None:
            return
        row = self._feature_row(layer, index)
        y, x = layer.data[index]
        info = (
            f"gene={row.get('gene_name')} barcode_id={row.get('barcode_id')} "
            f"codebook={row.get('codebook_id')} fov={row.get('fov_id')} "
            f"world=({x:.2f}, {y:.2f}) spot_id={row.get('spot_id')}"
        )
        show_info(info)

    def _wire_point_inspection(self) -> None:
        for layer in self.transcript_layers.values():
            layer.mouse_drag_callbacks.append(self._on_transcript_click)

    def _on_mouse_move(self, _viewer, event) -> None:
        if len(event.position) >= 2:
            y, x = event.position[-2], event.position[-1]
            self.dock_widget.qc_panel.set_cursor_world(x, y)

        hit = self._find_hovered_spot(event.position)
        gene_name = self._feature_row(*hit).get("gene_name") if hit is not None else None
        if gene_name != self._last_hover_gene:
            self._last_hover_gene = gene_name
            if gene_name is None:
                QToolTip.hideText()
            else:
                QToolTip.showText(QCursor.pos(), str(gene_name))

    # -- callbacks from the dock widget ---------------------------------------

    def _build_callbacks(self) -> dict:
        return {
            "on_rebuild_cache": self._on_rebuild_cache,
            "on_diagnostics": self._on_diagnostics,
            "on_image_visible_changed": self._on_image_visible_changed,
            "on_image_opacity_changed": self._on_image_opacity_changed,
            "on_image_contrast_changed": self._on_image_contrast_changed,
            "on_z_mode_changed": self._on_z_mode_changed,
            "on_z_index_changed": self._on_z_index_changed,
            "on_z_range_changed": self._on_z_range_changed,
            "on_transcripts_visible_changed": self._on_transcripts_visible_changed,
            "on_point_size_changed": self._on_point_size_changed,
            "on_point_opacity_changed": self._on_point_opacity_changed,
            "on_include_blanks_changed": self._on_include_blanks_changed,
            "on_display_mode_changed": self._on_display_mode_changed,
            "on_gene_selection_changed": self._on_gene_selection_changed,
            "on_codebook_symbol_changed": self._on_codebook_symbol_changed,
            "on_codebook_visible_changed": self._on_codebook_visible_changed,
            "on_show_fov_boundaries_changed": self._on_show_fov_boundaries_changed,
            "on_show_fov_ids_changed": self._on_show_fov_ids_changed,
            "on_jump_to_fov": self._on_jump_to_fov,
            "on_jump_to_xy": self._on_jump_to_xy,
        }

    def _on_rebuild_cache(self) -> None:
        show_info("Rebuild cache from the CLI: `merfishviewerxp index <experiment> --force`")

    def _on_diagnostics(self) -> None:
        show_info(f"Run `merfishviewerxp diagnose {self.indexed.cache.dataset_root}` for a full report.")

    def _on_image_visible_changed(self, channel_id: str, visible: bool) -> None:
        self.state.image_visibility[channel_id] = visible
        self.image_layers[channel_id].visible = visible
        self.state.save(self.indexed.cache.settings_path)

    def _on_image_opacity_changed(self, channel_id: str, opacity: float) -> None:
        self.state.image_opacity[channel_id] = opacity
        self.image_layers[channel_id].opacity = opacity

    def _on_image_contrast_changed(self, channel_id: str, lo: float, hi: float) -> None:
        if hi <= lo:
            return
        self.state.image_contrast[channel_id] = (lo, hi)
        self.image_layers[channel_id].contrast_limits = (lo, hi)

    def _refresh_all_image_layers(self) -> None:
        for channel_id, layer in self.image_layers.items():
            data = layer_builders.project_z(
                layer_builders.image_pyramid_as_dask(self.indexed, channel_id),
                z_mode=self.state.z_mode,
                z_index=self.state.z_index,
                z_range=self.state.z_range,
            )
            layer.data = data

    def _on_z_mode_changed(self, mode: str) -> None:
        self.state.z_mode = mode
        self._refresh_all_image_layers()

    def _on_z_index_changed(self, index: int) -> None:
        self.state.z_index = index
        if self.state.z_mode == "single":
            self._refresh_all_image_layers()

    def _on_z_range_changed(self, lo: int, hi: int) -> None:
        if hi < lo:
            return
        self.state.z_range = (lo, hi)
        if self.state.z_mode == "max_projection_range":
            self._refresh_all_image_layers()

    def _on_transcripts_visible_changed(self, visible: bool) -> None:
        # Master toggle: cascades into each codebook panel's own visible
        # checkbox, which is the single source of truth for layer visibility.
        self.state.transcripts_visible = visible
        for panel in self.dock_widget.codebook_panels.values():
            panel.visible_checkbox.setChecked(visible)

    def _on_point_size_changed(self, size: float) -> None:
        self.state.point_size = size
        for layer in self.transcript_layers.values():
            layer.size = size

    def _on_point_opacity_changed(self, opacity: float) -> None:
        self.state.point_opacity = opacity
        for layer in self.transcript_layers.values():
            layer.opacity = opacity

    def _on_include_blanks_changed(self, include: bool) -> None:
        self.state.include_blanks = include
        self._request_viewport_query()

    def _on_display_mode_changed(self, mode: str) -> None:
        self.state.lod_mode = mode
        self.state.save(self.indexed.cache.settings_path)
        self._request_viewport_query()

    def _on_gene_selection_changed(self, codebook_id: str, gene_ids: set[int]) -> None:
        self.state.active_gene_ids_by_codebook[codebook_id] = list(gene_ids)
        self.state.save(self.indexed.cache.settings_path)
        self._request_viewport_query()

    def _on_codebook_symbol_changed(self, codebook_id: str, symbol: str) -> None:
        self.state.codebook_symbols[codebook_id] = symbol
        self.transcript_layers[codebook_id].symbol = symbol
        self.state.save(self.indexed.cache.settings_path)

    def _on_codebook_visible_changed(self, codebook_id: str, visible: bool) -> None:
        self.state.codebook_visible[codebook_id] = visible
        self.transcript_layers[codebook_id].visible = visible
        self.state.save(self.indexed.cache.settings_path)

    def _on_show_fov_boundaries_changed(self, visible: bool) -> None:
        self.state.show_fov_boundaries = visible
        self.fov_shapes_layer.visible = visible

    def _on_show_fov_ids_changed(self, visible: bool) -> None:
        self.state.show_fov_ids = visible
        self.fov_labels_layer.visible = visible

    def _on_jump_to_fov(self, fov_id: int) -> None:
        fovs = self.indexed.fovs()
        match = fovs[fovs.fov_id == fov_id]
        if match.empty:
            show_info(f"FOV {fov_id} not found")
            return
        row = match.iloc[0]
        cy = (row.ymin_um + row.ymax_um) / 2
        cx = (row.xmin_um + row.xmax_um) / 2
        self.viewer.scene.camera.center = (cy, cx)

    def _on_jump_to_xy(self, x_um: float, y_um: float) -> None:
        self.viewer.scene.camera.center = (y_um, x_um)


def launch_viewer(indexed: IndexedDataset, config: AppConfig, *, run_event_loop: bool = True) -> MerfishViewerXPApp:
    app = MerfishViewerXPApp(indexed, config)
    if run_event_loop:
        napari.run()
    return app
