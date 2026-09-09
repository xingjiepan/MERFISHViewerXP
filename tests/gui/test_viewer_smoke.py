"""GUI smoke tests (spec section 22.4), run with an offscreen Qt platform."""

from __future__ import annotations

import napari
import pytest
from fixtures.synthetic_dataset import build_synthetic_dataset

from merfishviewerxp.api import MerfishDataset, index_dataset
from merfishviewerxp.config import AppConfig
from merfishviewerxp.indexed_dataset import IndexedDataset
from merfishviewerxp.viewer.app import MerfishViewerXPApp

# The synthetic fixture has two codebooks (CB0, CB1), so these smoke tests
# double as regression coverage for per-codebook panels/layers/symbols.


def _build_app(tmp_path, qtbot):
    root = tmp_path / "experiment"
    build_synthetic_dataset(root)
    dataset = MerfishDataset.open(root)
    cache_dir = root / "merfishviewerxp_cache"
    index_dataset(dataset, cache_dir=cache_dir, config=AppConfig())
    indexed = IndexedDataset.open(cache_dir)

    viewer = napari.Viewer(show=False)
    app = MerfishViewerXPApp(indexed, AppConfig(), viewer=viewer)
    qtbot.addWidget(app.dock_widget)
    return app, viewer


def _silence_late_query_results(app) -> None:
    # A background query worker started just before a test ends can still be
    # running when the test's widgets get torn down; if its result callback
    # fires afterwards it mutates deleted Qt objects and crashes teardown (or
    # even a later test's setup). Swallow anything that arrives after we're
    # done rather than trying to guarantee every test waits it out.
    app.query_runner.on_result = lambda *a, **k: None
    app.query_runner.on_error = lambda *a, **k: None


@pytest.fixture
def app_instance(tmp_path, qtbot):
    app, viewer = _build_app(tmp_path, qtbot)
    # Transcripts start hidden on every launch (see
    # test_transcripts_hidden_by_default_at_launch); most of these smoke
    # tests exercise transcript behavior, so opt back in here rather than
    # in every individual test.
    app.dock_widget.transcript_panel.visible_checkbox.setChecked(True)
    qtbot.waitUntil(lambda: sum(layer.data.shape[0] for layer in app.transcript_layers.values()) > 0, timeout=5000)
    yield app, qtbot
    _silence_late_query_results(app)
    viewer.close()


@pytest.fixture
def hidden_app_instance(tmp_path, qtbot):
    """An app instance left in its real post-launch state: nothing visible."""
    app, viewer = _build_app(tmp_path, qtbot)
    yield app, qtbot
    _silence_late_query_results(app)
    viewer.close()


def _total_points(app) -> int:
    return sum(layer.data.shape[0] for layer in app.transcript_layers.values())


def _wait_for_query(app, qtbot):
    # `worker.is_running` is False both before Qt has dispatched the thread's
    # start and after it finishes, so poll the actual observable result
    # instead of the worker's internal state.
    qtbot.waitUntil(lambda: _total_points(app) > 0, timeout=5000)


def test_app_opens_and_gene_list_loads(app_instance):
    app, qtbot = app_instance
    assert set(app.dock_widget.codebook_panels.keys()) == {"CB0", "CB1"}
    total_genes_in_panels = sum(p.list_widget.count() for p in app.dock_widget.codebook_panels.values())
    assert total_genes_in_panels == len(app.indexed.genes())
    assert total_genes_in_panels > 0


def test_codebook_panels_get_distinct_default_symbols(app_instance):
    app, qtbot = app_instance
    cb0_symbol = app.dock_widget.codebook_panels["CB0"].symbol_combo.currentText()
    cb1_symbol = app.dock_widget.codebook_panels["CB1"].symbol_combo.currentText()
    assert cb0_symbol != cb1_symbol


def test_changing_codebook_symbol_updates_its_layer_only(app_instance):
    app, qtbot = app_instance
    _wait_for_query(app, qtbot)
    qtbot.waitUntil(lambda: app.transcript_layers["CB0"].data.shape[0] > 0, timeout=5000)
    cb1_symbols_before = [str(s) for s in app.transcript_layers["CB1"].symbol]

    app.dock_widget.codebook_panels["CB0"].symbol_combo.setCurrentText("star")

    assert all(str(s) == "star" for s in app.transcript_layers["CB0"].symbol)
    assert [str(s) for s in app.transcript_layers["CB1"].symbol] == cb1_symbols_before


def test_symbol_survives_a_fresh_data_replacement(app_instance):
    """A scalar `.symbol` assignment on napari's Points layer only paints
    points that exist at assignment time; points arriving later via a fresh
    `.data` replacement must not silently fall back to the default symbol."""
    app, qtbot = app_instance
    _wait_for_query(app, qtbot)
    qtbot.waitUntil(lambda: app.transcript_layers["CB0"].data.shape[0] > 0, timeout=5000)

    app.dock_widget.codebook_panels["CB0"].symbol_combo.setCurrentText("square")
    assert all(str(s) == "square" for s in app.transcript_layers["CB0"].symbol)

    # simulate a fresh viewport query landing (as if the user panned/zoomed)
    table, _ = app.indexed.query_spots(bounds_um=(-1000, -1000, 1000, 1000))
    app._on_query_result(table, {"shown": table.num_rows, "total_in_view": table.num_rows, "sampled": False})

    assert app.transcript_layers["CB0"].data.shape[0] > 0
    assert all(str(s) == "square" for s in app.transcript_layers["CB0"].symbol)


def test_reentrant_query_result_is_serialized_not_interleaved(app_instance):
    """If applying one query result somehow triggers delivery of a second,
    newer one before the first call returns (napari's Points layer is not
    documented as safe against concurrent mutation), the layers must end up
    reflecting the newer result -- never a mix of both, and never crash."""
    app, qtbot = app_instance
    _wait_for_query(app, qtbot)

    table_a, _ = app.indexed.query_spots(bounds_um=(-1000, -1000, 1000, 1000))
    table_b, _ = app.indexed.query_spots(bounds_um=(-1000, -1000, 1000, 1000), gene_ids=[])
    lod_a = {"shown": table_a.num_rows, "total_in_view": table_a.num_rows, "sampled": False}
    lod_b = {"shown": table_b.num_rows, "total_in_view": table_b.num_rows, "sampled": False}
    assert table_a.num_rows > table_b.num_rows == 0  # table_b (empty gene filter) is the distinguishable "newer" state

    calls = []
    real_apply = app._apply_query_result

    def spy_apply(table, lod_info):
        calls.append(table.num_rows)
        if len(calls) == 1:
            # reenter exactly as a stale/interleaved delivery would
            app._on_query_result(table_b, lod_b)
        real_apply(table, lod_info)

    app._apply_query_result = spy_apply
    try:
        app._on_query_result(table_a, lod_a)
    finally:
        app._apply_query_result = real_apply

    assert not app._applying_query_result
    assert app._pending_query_result is None
    assert app.transcript_layers["CB0"].data.shape[0] == 0
    assert app.transcript_layers["CB1"].data.shape[0] == 0


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


def test_gene_checkbox_changes_transcript_query_for_its_own_codebook_only(app_instance):
    app, qtbot = app_instance
    _wait_for_query(app, qtbot)
    qtbot.waitUntil(lambda: app.transcript_layers["CB0"].data.shape[0] > 0, timeout=5000)

    cb1_count_before = app.transcript_layers["CB1"].data.shape[0]

    # uncheck everything in CB0 only -> CB0's layer empties, CB1 is untouched
    app.dock_widget.codebook_panels["CB0"]._bulk_set(False)
    qtbot.waitUntil(lambda: app.transcript_layers["CB0"].data.shape[0] == 0, timeout=5000)
    assert app.transcript_layers["CB1"].data.shape[0] == cb1_count_before


def test_codebook_visibility_toggle_is_independent(app_instance):
    app, qtbot = app_instance
    panel_cb0 = app.dock_widget.codebook_panels["CB0"]
    assert app.transcript_layers["CB0"].visible is True
    assert app.transcript_layers["CB1"].visible is True

    panel_cb0.visible_checkbox.setChecked(False)
    assert app.transcript_layers["CB0"].visible is False
    assert app.transcript_layers["CB1"].visible is True


def test_master_visible_checkbox_cascades_to_all_codebooks(app_instance):
    app, qtbot = app_instance
    app.dock_widget.transcript_panel.visible_checkbox.setChecked(False)
    assert all(not layer.visible for layer in app.transcript_layers.values())
    app.dock_widget.transcript_panel.visible_checkbox.setChecked(True)
    assert all(layer.visible for layer in app.transcript_layers.values())


def test_display_mode_defaults_to_auto_lod_sampling(app_instance):
    app, qtbot = app_instance
    # let the construction-time initial query settle before replacing
    # `query_runner.request` and tearing down, or a late-arriving real
    # worker result can crash trying to update an already-destroyed widget.
    _wait_for_query(app, qtbot)
    assert app.dock_widget.transcript_panel.display_mode_combo.currentText() == "auto"
    assert app.state.lod_mode == "auto"

    captured = {}
    app.query_runner.request = lambda **kwargs: captured.update(kwargs)
    app._request_viewport_query()
    assert captured["apply_lod_sampling"] is True


def test_show_all_display_mode_disables_lod_sampling(app_instance):
    """The 'show_all' display mode is how the user forces every decoded
    transcript in view to be drawn -- e.g. when zoomed out over a large area
    -- instead of the default performance-capped sample."""
    app, qtbot = app_instance
    _wait_for_query(app, qtbot)

    captured = {}
    app.query_runner.request = lambda **kwargs: captured.update(kwargs)
    app.dock_widget.transcript_panel.display_mode_combo.setCurrentText("show_all")
    assert app.state.lod_mode == "show_all"
    assert captured["apply_lod_sampling"] is False


def test_display_mode_persists_across_reload(app_instance):
    app, qtbot = app_instance
    _wait_for_query(app, qtbot)
    app.query_runner.request = lambda **kwargs: None  # avoid spawning a real worker past teardown
    app.dock_widget.transcript_panel.display_mode_combo.setCurrentText("show_all")

    from merfishviewerxp.viewer.state import ViewerState

    reloaded = ViewerState.load_or_default(
        app.indexed.cache.settings_path,
        dataset_id=app.indexed.dataset_id,
        cache_path=app.indexed.cache.cache_dir,
        gene_ids_by_codebook={cb: [1] for cb in app.transcript_layers},
    )
    assert reloaded.lod_mode == "show_all"


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

    captured = {}
    monkeypatch.setattr("merfishviewerxp.viewer.app.show_info", lambda msg: captured.setdefault("msg", msg))

    layer = next(layer for layer in app.transcript_layers.values() if layer.data.shape[0] > 0)

    class FakeEvent:
        type = "mouse_press"
        position = tuple(layer.data[0])

    callback = layer.mouse_drag_callbacks[-1]
    callback(layer, FakeEvent())

    assert "msg" in captured
    assert "gene=" in captured["msg"]


def _find_cb0_only_point(app):
    """A CB0 point whose world position isn't also occupied by a CB1 point.

    The synthetic fixture deliberately places a CB0 and a CB1 spot at the
    same marker pixel in FOV 0 (to test the transform), so naively picking
    `data[0]` is unreliable for tests that need an unambiguous, single-layer
    hit.
    """
    cb1_positions = {tuple(p) for p in app.transcript_layers["CB1"].data}
    for i, p in enumerate(app.transcript_layers["CB0"].data):
        if tuple(p) not in cb1_positions:
            return i, tuple(p)
    raise AssertionError("no CB0-only point found in the synthetic fixture")


def test_hover_over_spot_shows_gene_name_tooltip(app_instance, monkeypatch):
    app, qtbot = app_instance
    _wait_for_query(app, qtbot)
    qtbot.waitUntil(lambda: app.transcript_layers["CB0"].data.shape[0] > 0, timeout=5000)

    index, (y, x) = _find_cb0_only_point(app)
    expected_gene = app._feature_row(app.transcript_layers["CB0"], index)["gene_name"]

    shown = {}
    monkeypatch.setattr("merfishviewerxp.viewer.app.QToolTip.showText", lambda _pos, text: shown.__setitem__("text", text))
    monkeypatch.setattr("merfishviewerxp.viewer.app.QToolTip.hideText", lambda: shown.__setitem__("hidden", True))

    class FakeEventOnSpot:
        position = (y, x)

    app._on_mouse_move(app.viewer, FakeEventOnSpot())
    assert shown.get("text") == expected_gene

    class FakeEventAway:
        position = (-999_999.0, -999_999.0)

    app._on_mouse_move(app.viewer, FakeEventAway())
    assert shown.get("hidden") is True


def test_transcripts_hidden_by_default_at_launch(hidden_app_instance):
    app, qtbot = hidden_app_instance
    assert app.state.transcripts_visible is False
    assert all(v is False for v in app.state.codebook_visible.values())
    assert app.dock_widget.transcript_panel.visible_checkbox.isChecked() is False
    for panel in app.dock_widget.codebook_panels.values():
        assert panel.visible_checkbox.isChecked() is False
    assert all(not layer.visible for layer in app.transcript_layers.values())
    assert _total_points(app) == 0


def test_transcripts_hidden_by_default_even_if_previously_persisted_visible(tmp_path, qtbot):
    """A stale settings.json from an older session (or a killed test run)
    must never be able to make transcripts start visible -- that's exactly
    the "stuck displaying too many points" scenario this default guards
    against."""
    import json

    root = tmp_path / "experiment"
    build_synthetic_dataset(root)
    dataset = MerfishDataset.open(root)
    cache_dir = root / "merfishviewerxp_cache"
    index_dataset(dataset, cache_dir=cache_dir, config=AppConfig())
    indexed = IndexedDataset.open(cache_dir)

    settings_path = indexed.cache.settings_path
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps({"transcripts_visible": True, "codebook_visible": {"CB0": True, "CB1": True}})
    )

    viewer = napari.Viewer(show=False)
    app = MerfishViewerXPApp(indexed, AppConfig(), viewer=viewer)
    qtbot.addWidget(app.dock_widget)

    assert app.state.transcripts_visible is False
    assert all(v is False for v in app.state.codebook_visible.values())
    assert all(not layer.visible for layer in app.transcript_layers.values())
    assert _total_points(app) == 0


def test_no_query_fires_at_startup_while_transcripts_hidden(tmp_path, qtbot, monkeypatch):
    from merfishviewerxp.viewer.workers import ViewportQueryRunner

    calls = []
    monkeypatch.setattr(ViewportQueryRunner, "request", lambda self, **kwargs: calls.append(kwargs))

    app, viewer = _build_app(tmp_path, qtbot)

    assert calls == []


def test_turning_on_codebook_visibility_triggers_a_query(hidden_app_instance):
    """A codebook layer may never have been populated (the initial query is
    skipped entirely while nothing is visible), so making it visible must
    fetch fresh data -- even though the query itself covers every active
    codebook's genes, not just the one being toggled (see
    `_request_viewport_query`/`_apply_query_result`), so CB1's layer also
    ends up populated. Visibility, not data, is what stays independent."""
    app, qtbot = hidden_app_instance
    assert _total_points(app) == 0

    app.dock_widget.codebook_panels["CB0"].visible_checkbox.setChecked(True)

    qtbot.waitUntil(lambda: app.transcript_layers["CB0"].data.shape[0] > 0, timeout=5000)
    assert app.transcript_layers["CB0"].visible is True
    assert app.transcript_layers["CB1"].visible is False


def test_hover_ignores_hidden_codebook_layer(app_instance, monkeypatch):
    app, qtbot = app_instance
    _wait_for_query(app, qtbot)
    qtbot.waitUntil(lambda: app.transcript_layers["CB0"].data.shape[0] > 0, timeout=5000)

    _, (y, x) = _find_cb0_only_point(app)
    app.dock_widget.codebook_panels["CB0"].visible_checkbox.setChecked(False)

    shown = {}
    monkeypatch.setattr("merfishviewerxp.viewer.app.QToolTip.showText", lambda _pos, text: shown.__setitem__("text", text))
    monkeypatch.setattr("merfishviewerxp.viewer.app.QToolTip.hideText", lambda: shown.__setitem__("hidden", True))

    class FakeEvent:
        position = (y, x)

    app._on_mouse_move(app.viewer, FakeEvent())
    assert "text" not in shown
