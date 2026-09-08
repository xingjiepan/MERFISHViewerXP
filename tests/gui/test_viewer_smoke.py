"""GUI smoke tests (spec section 22.4), run with an offscreen Qt platform."""

from __future__ import annotations

import napari
import pytest
from fixtures.synthetic_dataset import build_synthetic_dataset

from merfishviewerxp.api import MerfishDataset, index_dataset
from merfishviewerxp.config import AppConfig
from merfishviewerxp.indexed_dataset import IndexedDataset
from merfishviewerxp.viewer.app import MerfishViewerXPApp


@pytest.fixture
def app_instance(tmp_path, qtbot):
    root = tmp_path / "experiment"
    build_synthetic_dataset(root)
    dataset = MerfishDataset.open(root)
    cache_dir = root / "merfishviewerxp_cache"
    index_dataset(dataset, cache_dir=cache_dir, config=AppConfig())
    indexed = IndexedDataset.open(cache_dir)

    viewer = napari.Viewer(show=False)
    app = MerfishViewerXPApp(indexed, AppConfig(), viewer=viewer)
    qtbot.addWidget(app.dock_widget)
    yield app, qtbot
    viewer.close()


def _wait_for_query(app, qtbot):
    # `worker.is_running` is False both before Qt has dispatched the thread's
    # start and after it finishes, so poll the actual observable result
    # instead of the worker's internal state.
    qtbot.waitUntil(lambda: app.transcript_layer.data.shape[0] > 0, timeout=5000)


def test_app_opens_and_gene_list_loads(app_instance):
    app, qtbot = app_instance
    n_genes = len(app.indexed.genes())
    assert app.dock_widget.gene_panel.list_widget.count() == n_genes
    assert n_genes > 0


def test_toggle_nucleus_visibility(app_instance):
    app, qtbot = app_instance
    ctrl = app.dock_widget.image_panel.channel_controls["nucleus"]
    assert app.image_layers["nucleus"].visible is True
    ctrl.visible_checkbox.setChecked(False)
    assert app.image_layers["nucleus"].visible is False
    ctrl.visible_checkbox.setChecked(True)
    assert app.image_layers["nucleus"].visible is True


def test_toggle_membrane_visibility(app_instance):
    app, qtbot = app_instance
    ctrl = app.dock_widget.image_panel.channel_controls["membrane"]
    ctrl.visible_checkbox.setChecked(False)
    assert app.image_layers["membrane"].visible is False


def test_gene_checkbox_changes_transcript_query(app_instance):
    app, qtbot = app_instance
    _wait_for_query(app, qtbot)
    initial_count = app.transcript_layer.data.shape[0]
    assert initial_count > 0

    # uncheck everything -> transcript layer should end up empty after the requery
    app.dock_widget.gene_panel._bulk_set(False)
    qtbot.waitUntil(lambda: app.transcript_layer.data.shape[0] == 0, timeout=5000)


def test_z_control_updates_image(app_instance):
    app, qtbot = app_instance
    layer = app.image_layers["nucleus"]
    before = [d.copy() if hasattr(d, "copy") else d for d in layer.data]
    app.dock_widget.image_panel.z_mode_combo.setCurrentText("single")
    app.dock_widget.image_panel.z_index_spin.setValue(0)
    after = layer.data
    assert len(after) == len(before)  # still one array per pyramid level


def test_fov_boundary_toggle(app_instance):
    app, qtbot = app_instance
    assert app.fov_shapes_layer.visible is False
    app.dock_widget.qc_panel.boundaries_checkbox.setChecked(True)
    assert app.fov_shapes_layer.visible is True


def test_point_click_returns_metadata(app_instance, monkeypatch):
    app, qtbot = app_instance
    _wait_for_query(app, qtbot)
    assert app.transcript_layer.data.shape[0] > 0

    captured = {}
    monkeypatch.setattr("merfishviewerxp.viewer.app.show_info", lambda msg: captured.setdefault("msg", msg))

    class FakeEvent:
        type = "mouse_press"
        position = tuple(app.transcript_layer.data[0])

    callback = app.transcript_layer.mouse_drag_callbacks[-1]
    callback(app.transcript_layer, FakeEvent())

    assert "msg" in captured
    assert "gene=" in captured["msg"]
