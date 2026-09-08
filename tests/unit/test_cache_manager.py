import json

from merfishviewerxp.config import load_config
from merfishviewerxp.storage.cache import CacheManager


def test_default_cache_dir_is_visible_not_hidden(tmp_path):
    cache = CacheManager(dataset_root=tmp_path)
    assert cache.cache_dir == tmp_path / "merfishviewerxp_cache"
    assert not cache.cache_dir.name.startswith(".")


def test_explicit_cache_dir_overrides_default(tmp_path):
    custom = tmp_path / "somewhere_else"
    cache = CacheManager(dataset_root=tmp_path, cache_dir=custom)
    assert cache.cache_dir == custom


def test_load_config_reads_settings_from_custom_cache_dir(tmp_path):
    custom_cache_dir = tmp_path / "custom_cache"
    custom_cache_dir.mkdir()
    (custom_cache_dir / "settings.json").write_text(json.dumps({"spots": {"max_visible_points": 42}}))

    config = load_config(dataset_root=tmp_path, cache_dir=custom_cache_dir)
    assert config.spots.max_visible_points == 42

    # a settings.json at the *default* location must NOT be picked up when
    # an explicit cache_dir is given
    default_cache_dir = tmp_path / "merfishviewerxp_cache"
    default_cache_dir.mkdir()
    (default_cache_dir / "settings.json").write_text(json.dumps({"spots": {"max_visible_points": 999}}))
    config2 = load_config(dataset_root=tmp_path, cache_dir=custom_cache_dir)
    assert config2.spots.max_visible_points == 42
