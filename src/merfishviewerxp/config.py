"""Configuration model and precedence loader (spec section 18).

Precedence (highest wins): CLI argument > dataset-local viewer settings
(``<cache_dir>/settings.json``) > user-global config
(``~/.config/merfishviewerxp/config.yaml``) > application default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

USER_CONFIG_PATH = Path.home() / ".config" / "merfishviewerxp" / "config.yaml"


class ImagesConfig(BaseModel):
    nucleus_pattern: str | None = None
    membrane_pattern: str | None = None
    chunk_size: int = 512
    overlap_mode: str = "feather"
    fov_crop_px: int = 0
    pyramid_downsample: int = 2


class SpotsConfig(BaseModel):
    spatial_tile_size_um: float = 1000.0
    max_visible_points: int = 250_000
    include_blanks_default: bool = False


class ViewerConfig(BaseModel):
    viewport_debounce_ms: int = 200
    default_projection: str = "max"


class AppConfig(BaseModel):
    images: ImagesConfig = Field(default_factory=ImagesConfig)
    spots: SpotsConfig = Field(default_factory=SpotsConfig)
    viewer: ViewerConfig = Field(default_factory=ViewerConfig)

    def channel_patterns(self) -> dict[str, str] | None:
        overrides = {}
        if self.images.nucleus_pattern:
            overrides["nucleus"] = self.images.nucleus_pattern
        if self.images.membrane_pattern:
            overrides["membrane"] = self.images.membrane_pattern
        return overrides or None


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text) or {}
    return json.loads(text) or {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    *,
    dataset_root: Path | None = None,
    cache_dir: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    merged: dict[str, Any] = {}
    merged = _deep_merge(merged, _read_config_file(USER_CONFIG_PATH))
    if dataset_root is not None:
        from .storage.cache import CacheManager  # local import: storage doesn't depend on config

        settings_path = CacheManager(dataset_root=dataset_root, cache_dir=cache_dir).settings_path
        merged = _deep_merge(merged, _read_config_file(settings_path))
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)
    return AppConfig.model_validate(merged)
