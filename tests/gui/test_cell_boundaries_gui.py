"""GUI coverage for the cell-boundary overlay (offscreen Qt)."""

from __future__ import annotations

import napari
import pytest
from fixtures.synthetic_dataset import build_synthetic_dataset

from merfishviewerxp.api import MerfishDataset, index_dataset
from merfishviewerxp.config import AppConfig
from merfishviewerxp.indexed_dataset import IndexedDataset
from merfishviewerxp.viewer.app import MerfishViewerXPApp
from merfishviewerxp.viewer.layers import CELL_BOUNDARY_LAYER_NAME
from merfishviewerxp.viewer.state import ViewerState


def _make_app(tmp_path, qtbot, *, with_masks: bool):
    root = tmp_path / "experiment"
    build_synthetic_dataset(root, with_masks=with_masks)
    cache_dir = root / "merfishviewerxp_cache"
    index_dataset(MerfishDataset.open(root), cache_dir=cache_dir, config=AppConfig())
    viewer = napari.Viewer(show=False)
    app = MerfishViewerXPApp(IndexedDataset.open(cache_dir), AppConfig(), viewer=viewer)
    qtbot.addWidget(app.dock_widget)
    return app, viewer


@pytest.fixture
def app_with_masks(tmp_path, qtbot):
    app, viewer = _make_app(tmp_path, qtbot, with_masks=True)
    yield app
    viewer.close()


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


def test_z_mode_change_reprojects_boundaries(app_with_masks):
    app = app_with_masks
    before = app.cell_boundary_layer.data
    app.dock_widget.image_panel.z_mode_combo.setCurrentText("single")
    after = app.cell_boundary_layer.data
    assert len(after) == len(before)
    assert after[0] is not before[0]
    assert after[0].ndim == 2


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
