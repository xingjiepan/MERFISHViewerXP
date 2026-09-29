"""GUI coverage for the cell-boundary overlay (offscreen Qt)."""

from __future__ import annotations

import json

import napari
import numpy as np
import pytest
from fixtures.synthetic_dataset import build_synthetic_dataset

from merfishviewerxp.api import MerfishDataset, index_dataset
from merfishviewerxp.config import AppConfig
from merfishviewerxp.indexed_dataset import IndexedDataset
from merfishviewerxp.viewer.app import MerfishViewerXPApp
from merfishviewerxp.viewer.layers import CELL_BOUNDARY_LAYER_NAME
from merfishviewerxp.viewer.state import ViewerState


def _make_app(tmp_path, qtbot, *, with_masks: bool, settings: dict | None = None):
    root = tmp_path / "experiment"
    info = build_synthetic_dataset(root, with_masks=with_masks)
    cache_dir = root / "merfishviewerxp_cache"
    index_dataset(MerfishDataset.open(root), cache_dir=cache_dir, config=AppConfig())
    indexed = IndexedDataset.open(cache_dir)
    if settings is not None:
        indexed.cache.settings_path.write_text(json.dumps(settings))
    viewer = napari.Viewer(show=False)
    app = MerfishViewerXPApp(indexed, AppConfig(), viewer=viewer)
    qtbot.addWidget(app.dock_widget)
    app.fixture_info = info
    return app, viewer


@pytest.fixture
def app_with_masks(tmp_path, qtbot):
    app, viewer = _make_app(tmp_path, qtbot, with_masks=True)
    yield app
    viewer.close()


def _cell_interior(app, fov_id: int) -> tuple[float, float]:
    return app.fixture_info["expected_cell_interior_world_um"][fov_id]


def _displayed_value_at(app, x_um: float, y_um: float) -> int:
    """Display value (0 none, 1 default, 2+ recolored) of the nearest boundary pixel's cell,
    read from level 0 of the boundary layer at the cell's marker (corner) pixel."""
    meta = app.indexed.cell_boundary_metadata()
    ox, oy = meta["origin_world_um"]
    ps = meta["pixel_size_um"][0]
    level0 = np.asarray(app.cell_boundary_layer.data[0])
    return int(level0[round((y_um - oy) / ps), round((x_um - ox) / ps)])


def _reload_state(app) -> ViewerState:
    return ViewerState.load_or_default(
        app.indexed.cache.settings_path,
        dataset_id=app.indexed.dataset_id,
        cache_path=app.indexed.cache.cache_dir,
        gene_ids_by_codebook={},
    )


class _FakeMouseEvent:
    def __init__(self, *, position, pos=(100.0, 100.0)):
        self.type = "mouse_press"
        self.position = position
        self.pos = pos


def _simulate_mouse(app, *, position, move_px: float = 0.0) -> None:
    event = _FakeMouseEvent(position=position)
    gen = app._on_canvas_click_for_cell_coloring(app.viewer, event)
    if gen is None:
        return
    try:
        next(gen)
        if move_px:
            event.type = "mouse_move"
            event.pos = (100.0 + move_px, 100.0)
            next(gen)
        event.type = "mouse_release"
        next(gen)
    except StopIteration:
        pass


def test_boundary_layer_exists_and_starts_hidden(app_with_masks):
    app = app_with_masks
    layer = app.cell_boundary_layer
    assert layer is not None
    assert layer.name == CELL_BOUNDARY_LAYER_NAME
    assert layer.visible is False
    assert tuple(layer.contrast_limits) == (0, 1)
    controls = app.dock_widget.image_panel.cell_boundary_controls
    assert controls is not None
    assert controls.visible_checkbox.isChecked() is False


def test_boundary_layer_is_drawn_above_images_and_below_transcripts(app_with_masks):
    app = app_with_masks
    names = [layer.name for layer in app.viewer.layers]
    boundary_idx = names.index(CELL_BOUNDARY_LAYER_NAME)
    assert all(names.index(ch) < boundary_idx for ch in app.image_layers)
    assert all(names.index(layer.name) > boundary_idx for layer in app.transcript_layers.values())


def test_checkbox_toggles_boundaries_and_persists(app_with_masks):
    app = app_with_masks
    controls = app.dock_widget.image_panel.cell_boundary_controls
    controls.visible_checkbox.setChecked(True)
    assert app.cell_boundary_layer.visible is True
    assert app.state.show_cell_boundaries is True

    reloaded = ViewerState.load_or_default(
        app.indexed.cache.settings_path,
        dataset_id=app.indexed.dataset_id,
        cache_path=app.indexed.cache.cache_dir,
        gene_ids_by_codebook={},
    )
    assert reloaded.show_cell_boundaries is True

    controls.visible_checkbox.setChecked(False)
    assert app.cell_boundary_layer.visible is False


def test_opacity_slider_updates_boundary_layer(app_with_masks):
    app = app_with_masks
    app.dock_widget.image_panel.cell_boundary_controls.opacity_slider.setValue(40)
    assert app.cell_boundary_layer.opacity == pytest.approx(0.4)
    assert app.state.cell_boundary_opacity == pytest.approx(0.4)


def test_z_mode_change_and_recoloring_update_layers_in_place(app_with_masks):
    """Replacing a multiscale layer's data makes napari slice an empty tile while
    zoomed in (see layers.DisplayView), so these updates must never replace it."""
    app = app_with_masks
    boundary_levels = list(app.cell_boundary_layer.data)
    nucleus_levels = list(app.image_layers["nucleus"].data)

    app.dock_widget.image_panel.z_mode_combo.setCurrentText("single")
    app.toggle_cell_color_at(*_cell_interior(app, 0))

    assert app._view.z_mode == "single"
    assert all(a is b for a, b in zip(app.cell_boundary_layer.data, boundary_levels, strict=True))
    assert all(a is b for a, b in zip(app.image_layers["nucleus"].data, nucleus_levels, strict=True))
    assert app.cell_boundary_layer.data[0].ndim == 2


@pytest.fixture
def zoomed_in_multiscale_boundaries(qtbot):
    """A 2-level boundary layer showing a region of level 0 that lies beyond level 1's extent."""
    from merfishviewerxp.viewer.layers import DisplayView, ProjectedLevel, cell_color_table

    ids = np.zeros((1, 2048, 2048), dtype=np.uint32)
    ids[0, 1500:1600, 1500] = 7
    view = DisplayView(z_mode="max_projection", cell_colors=cell_color_table({}))
    levels = [ProjectedLevel(ids, view, color_cells=True), ProjectedLevel(ids[:, ::2, ::2], view, color_cells=True)]
    viewer = napari.Viewer(show=False)
    layer = viewer.add_image(levels, multiscale=True, contrast_limits=(0, 2))
    layer._data_level = 0
    layer.corner_pixels = np.array([[1400, 1400], [1700, 1700]])
    layer.refresh()
    yield layer, view
    viewer.close()


def test_view_change_keeps_zoomed_in_level_and_region(zoomed_in_multiscale_boundaries):
    from merfishviewerxp.viewer.layers import cell_color_table

    layer, view = zoomed_in_multiscale_boundaries
    assert layer._slice.image.raw.shape == (301, 301)

    view.cell_colors = cell_color_table({7: "#00ff00"})
    layer.refresh()

    assert layer.data_level == 0
    assert layer._slice.image.raw.shape == (301, 301)  # never an empty tile
    assert layer._slice.image.raw.max() == 2  # cell 7 now drawn in its override color


def test_default_boundary_color_can_be_changed_and_persists(app_with_masks, monkeypatch):
    from qtpy.QtGui import QColor

    app = app_with_masks
    monkeypatch.setattr(
        "merfishviewerxp.viewer.widgets.image_panel.QColorDialog.getColor", lambda *a, **k: QColor("#00ffff")
    )
    app.dock_widget.image_panel.cell_boundary_controls.default_color_button.click()

    assert app.state.cell_boundary_default_color == "#00ffff"
    assert app.cell_boundary_layer.colormap.map(np.array([1.0]))[0][:3] == pytest.approx([0.0, 1.0, 1.0])
    assert _reload_state(app).cell_boundary_default_color == "#00ffff"


def test_clicking_a_cell_toggles_its_highlight_color(app_with_masks):
    app = app_with_masks
    controls = app.dock_widget.image_panel.cell_boundary_controls
    x, y = _cell_interior(app, 2)
    marker_x, marker_y = app.fixture_info["expected_marker_world_um"][2]
    other_x, other_y = app.fixture_info["expected_marker_world_um"][1]

    cell_id = app.toggle_cell_color_at(x, y)
    assert cell_id is not None
    assert app.indexed.cell_id_source(cell_id) == (2, 1)
    assert app.state.cell_boundary_colors == {cell_id: app.state.cell_highlight_color}
    assert tuple(app.cell_boundary_layer.contrast_limits) == (0, 2)
    assert _displayed_value_at(app, marker_x, marker_y) == 2  # the clicked cell: highlight color
    assert _displayed_value_at(app, other_x, other_y) == 1  # other cells keep the boundary color
    assert "1 cell(s)" in controls.colored_count_label.text()
    assert _reload_state(app).cell_boundary_colors == {cell_id: app.state.cell_highlight_color}

    assert app.toggle_cell_color_at(x, y) == cell_id  # clicking again restores it
    assert app.state.cell_boundary_colors == {}
    assert _displayed_value_at(app, marker_x, marker_y) == 1
    assert tuple(app.cell_boundary_layer.contrast_limits) == (0, 1)


def test_highlight_color_choice_applies_to_the_next_clicked_cell(app_with_masks, monkeypatch):
    from qtpy.QtGui import QColor

    app = app_with_masks
    monkeypatch.setattr(
        "merfishviewerxp.viewer.widgets.image_panel.QColorDialog.getColor", lambda *a, **k: QColor("#123456")
    )
    app.dock_widget.image_panel.cell_boundary_controls.highlight_color_button.click()
    cell_id = app.toggle_cell_color_at(*_cell_interior(app, 0))
    assert app.state.cell_boundary_colors[cell_id] == "#123456"


def test_mouse_click_colors_a_cell_only_in_coloring_mode_and_not_when_panning(app_with_masks):
    app = app_with_masks
    controls = app.dock_widget.image_panel.cell_boundary_controls
    x, y = _cell_interior(app, 3)

    _simulate_mouse(app, position=(y, x))
    assert app.state.cell_boundary_colors == {}  # coloring mode is off

    controls.color_cells_button.setChecked(True)
    assert controls.visible_checkbox.isChecked()  # turning coloring on shows the boundaries
    _simulate_mouse(app, position=(y, x), move_px=20)
    assert app.state.cell_boundary_colors == {}  # a drag is a pan

    _simulate_mouse(app, position=(y, x))
    assert len(app.state.cell_boundary_colors) == 1


def test_clear_button_restores_all_cells(app_with_masks):
    app = app_with_masks
    for fov_id in (0, 1):
        app.toggle_cell_color_at(*_cell_interior(app, fov_id))
    controls = app.dock_widget.image_panel.cell_boundary_controls
    assert controls.clear_button.isEnabled()

    controls.clear_button.click()
    assert app.state.cell_boundary_colors == {}
    assert not controls.clear_button.isEnabled()
    assert _reload_state(app).cell_boundary_colors == {}


def test_clicking_away_from_cells_changes_nothing(app_with_masks, monkeypatch):
    shown = []
    monkeypatch.setattr("merfishviewerxp.viewer.app.show_info", shown.append)
    app = app_with_masks
    assert app.toggle_cell_color_at(10_000.0, 10_000.0) is None
    assert app.state.cell_boundary_colors == {}
    assert shown == ["No segmented cell here"]


def test_coloring_a_cell_reports_which_cell_it_was(app_with_masks, monkeypatch):
    shown = []
    monkeypatch.setattr("merfishviewerxp.viewer.app.show_info", shown.append)
    app = app_with_masks
    cell_id = app.toggle_cell_color_at(*_cell_interior(app, 2))
    assert shown == [f"Cell {cell_id} (FOV 2, mask label 1) -> {app.state.cell_highlight_color}"]


@pytest.fixture
def reopened_after_coloring(tmp_path, qtbot):
    """A first session that recolors one cell, then a second session on the same cache."""
    app, viewer = _make_app(tmp_path, qtbot, with_masks=True)
    cell_id = app.toggle_cell_color_at(*_cell_interior(app, 1))
    viewer2 = napari.Viewer(show=False)
    app2 = MerfishViewerXPApp(IndexedDataset.open(app.indexed.cache.cache_dir), AppConfig(), viewer=viewer2)
    qtbot.addWidget(app2.dock_widget)
    yield app, app2, cell_id
    viewer2.close()
    viewer.close()


def test_saved_cell_colors_are_restored_when_masks_are_unchanged(reopened_after_coloring):
    app, app2, cell_id = reopened_after_coloring
    assert app2.state.cell_boundary_colors == {cell_id: app.state.cell_highlight_color}
    assert "1 cell(s)" in app2.dock_widget.image_panel.cell_boundary_controls.colored_count_label.text()
    marker_x, marker_y = app.fixture_info["expected_marker_world_um"][1]
    assert _displayed_value_at(app2, marker_x, marker_y) == 2


@pytest.fixture
def app_with_stale_cell_colors(tmp_path, qtbot):
    stale = {"cell_boundary_colors": {"1": "#ffff00"}, "cell_boundary_colors_source": "masks-from-an-older-run"}
    app, viewer = _make_app(tmp_path, qtbot, with_masks=True, settings=stale)
    yield app
    viewer.close()


def test_saved_cell_colors_are_dropped_when_masks_changed(app_with_stale_cell_colors):
    app = app_with_stale_cell_colors
    assert app.state.cell_boundary_colors == {}
    assert app.state.cell_boundary_colors_source == app.indexed.manifest.fingerprint.mask_inventory_hash


@pytest.fixture
def app_without_masks(tmp_path, qtbot):
    app, viewer = _make_app(tmp_path, qtbot, with_masks=False)
    yield app
    viewer.close()


def test_no_boundary_layer_or_controls_without_masks(app_without_masks):
    app = app_without_masks
    assert app.cell_boundary_layer is None
    assert app.dock_widget.image_panel.cell_boundary_controls is None
    assert CELL_BOUNDARY_LAYER_NAME not in [layer.name for layer in app.viewer.layers]
