"""Wires the napari canvas, layers, dock widget, and background queries together."""

from __future__ import annotations

import logging

import napari
import numpy as np
from napari.utils.notifications import show_info
from qtpy.QtCore import QTimer

from ..config import AppConfig
from ..indexed_dataset import IndexedDataset
from . import layers as layer_builders
from .state import ViewerState
from .widgets.main_dock_widget import MainDockWidget
from .workers import ViewportQueryRunner

logger = logging.getLogger(__name__)


class MerfishViewerXPApp:
    def __init__(self, indexed: IndexedDataset, config: AppConfig, *, viewer: napari.Viewer | None = None) -> None:
        self.indexed = indexed
        self.config = config
        self.viewer = viewer or napari.Viewer(title=f"MERFISHViewerXP - {indexed.dataset_id}")

        genes_df = indexed.genes()
        all_gene_ids = [int(g) for g in genes_df["gene_id"]]
        self.state = ViewerState.load_or_default(
            indexed.cache.settings_path,
            dataset_id=indexed.dataset_id,
            cache_path=indexed.cache.cache_dir,
            all_gene_ids=all_gene_ids,
        )
        self.state.max_visible_points = config.spots.max_visible_points
        if not self.state.include_blanks:
            self.state.include_blanks = config.spots.include_blanks_default
        self._all_gene_ids = set(all_gene_ids)

        self.image_layers: dict[str, napari.layers.Image] = {}

        self._build_image_layers()
        self._build_fov_layers()
        self._build_transcript_layer()

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
            genes=genes_df,
            initial_active_gene_ids=set(self.state.active_gene_ids),
            initial_point_size=self.state.point_size,
            initial_point_opacity=self.state.point_opacity,
            initial_include_blanks=self.state.include_blanks,
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

    def _build_transcript_layer(self) -> None:
        self.transcript_layer = self.viewer.add_points(
            np.empty((0, 2)),
            name="Decoded transcripts",
            size=self.state.point_size,
            opacity=self.state.point_opacity,
            visible=self.state.transcripts_visible,
        )

    # -- viewport querying ----------------------------------------------------

    def _current_world_bounds(self) -> tuple[float, float, float, float]:
        for layer in self.image_layers.values():
            corners = layer.data_to_world(layer.corner_pixels)
            (y0, x0), (y1, x1) = corners
            return float(x0), float(y0), float(x1), float(y1)
        fovs = self.indexed.fovs()
        return float(fovs.xmin_um.min()), float(fovs.ymin_um.min()), float(fovs.xmax_um.max()), float(fovs.ymax_um.max())

    def _on_camera_changed(self, _event=None) -> None:
        self._debounce_timer.start()

    def _request_viewport_query(self) -> None:
        gene_ids = None if set(self.state.active_gene_ids) == self._all_gene_ids else list(self.state.active_gene_ids)
        self.query_runner.request(
            bounds_um=self._current_world_bounds(),
            gene_ids=gene_ids,
            include_blanks=self.state.include_blanks,
            max_visible_points=self.state.max_visible_points,
        )

    def _on_query_result(self, table, lod_info: dict) -> None:
        coords, colors, features = layer_builders.spot_table_to_points(table)
        self.transcript_layer.data = coords
        if len(coords):
            self.transcript_layer.face_color = colors
        self.transcript_layer.features = features
        self.dock_widget.transcript_panel.set_counts(lod_info["shown"], lod_info["total_in_view"], lod_info["sampled"])
        self.viewer.status = f"Visible transcripts: {lod_info['shown']} / {lod_info['total_in_view']}"

    def _on_query_error(self, exc: Exception) -> None:
        logger.exception("Viewport query failed", exc_info=exc)
        show_info(f"Transcript query failed: {exc}")

    # -- point inspection / cursor probe --------------------------------------

    def _wire_point_inspection(self) -> None:
        @self.transcript_layer.mouse_drag_callbacks.append
        def _on_click(layer, event):
            if event.type != "mouse_press":
                return
            index = layer.get_value(event.position, world=True)
            if index is None:
                return
            if hasattr(layer.features, "items"):
                row = {k: v[index] for k, v in layer.features.items()}
            else:
                row = layer.features.iloc[index].to_dict()
            y, x = layer.data[index]
            info = (
                f"gene={row.get('gene_name')} barcode_id={row.get('barcode_id')} "
                f"codebook={row.get('codebook_id')} fov={row.get('fov_id')} "
                f"world=({x:.2f}, {y:.2f}) spot_id={row.get('spot_id')}"
            )
            show_info(info)

    def _on_mouse_move(self, _viewer, event) -> None:
        if len(event.position) >= 2:
            y, x = event.position[-2], event.position[-1]
            self.dock_widget.qc_panel.set_cursor_world(x, y)

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
            "on_display_mode_changed": lambda _mode: None,
            "on_gene_selection_changed": self._on_gene_selection_changed,
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
        self.state.transcripts_visible = visible
        self.transcript_layer.visible = visible

    def _on_point_size_changed(self, size: float) -> None:
        self.state.point_size = size
        self.transcript_layer.size = size

    def _on_point_opacity_changed(self, opacity: float) -> None:
        self.state.point_opacity = opacity
        self.transcript_layer.opacity = opacity

    def _on_include_blanks_changed(self, include: bool) -> None:
        self.state.include_blanks = include
        self._request_viewport_query()

    def _on_gene_selection_changed(self, gene_ids: set[int]) -> None:
        self.state.active_gene_ids = list(gene_ids)
        self.state.save(self.indexed.cache.settings_path)
        self._request_viewport_query()

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
