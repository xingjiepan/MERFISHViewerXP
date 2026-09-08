import json
from pathlib import Path

from merfishviewerxp.viewer.state import ViewerState


def test_default_state_activates_all_genes(tmp_path):
    state = ViewerState.load_or_default(
        tmp_path / "settings.json", dataset_id="ds", cache_path=tmp_path, all_gene_ids=[1, 2, 3]
    )
    assert set(state.active_gene_ids) == {1, 2, 3}


def test_save_and_reload_round_trips_gene_selection(tmp_path):
    settings_path = tmp_path / "settings.json"
    state = ViewerState.load_or_default(settings_path, dataset_id="ds", cache_path=tmp_path, all_gene_ids=[1, 2, 3])
    state.active_gene_ids = [2]
    state.include_blanks = True
    state.point_size = 7.5
    state.save(settings_path)

    reloaded = ViewerState.load_or_default(settings_path, dataset_id="ds", cache_path=tmp_path, all_gene_ids=[1, 2, 3])
    assert reloaded.active_gene_ids == [2]
    assert reloaded.include_blanks is True
    assert reloaded.point_size == 7.5


def test_corrupt_settings_file_falls_back_to_default(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text("not valid json{{{")
    state = ViewerState.load_or_default(settings_path, dataset_id="ds", cache_path=tmp_path, all_gene_ids=[1, 2])
    assert set(state.active_gene_ids) == {1, 2}


def test_save_is_atomic_no_partial_file_left(tmp_path):
    settings_path = tmp_path / "settings.json"
    state = ViewerState.load_or_default(settings_path, dataset_id="ds", cache_path=tmp_path, all_gene_ids=[1])
    state.save(settings_path)
    assert settings_path.is_file()
    assert not Path(str(settings_path) + ".tmp").exists()


def test_save_preserves_unrelated_top_level_keys(tmp_path):
    """settings.json is shared with AppConfig's dataset-local overrides
    (spec section 18) -- saving viewer state must not clobber them."""
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"images": {"chunk_size": 1024}, "spots": {"max_visible_points": 42}}))

    state = ViewerState.load_or_default(settings_path, dataset_id="ds", cache_path=tmp_path, all_gene_ids=[1, 2])
    state.point_size = 9.0
    state.save(settings_path)

    on_disk = json.loads(settings_path.read_text())
    assert on_disk["images"] == {"chunk_size": 1024}
    assert on_disk["spots"] == {"max_visible_points": 42}
    assert on_disk["point_size"] == 9.0
