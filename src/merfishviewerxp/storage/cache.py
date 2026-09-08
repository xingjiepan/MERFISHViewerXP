"""Cache directory layout (spec section 15.1)."""

from __future__ import annotations

from pathlib import Path

DEFAULT_CACHE_DIRNAME = "merfishviewerxp_cache"


class CacheManager:
    def __init__(self, dataset_root: Path, cache_dir: Path | None = None) -> None:
        self.dataset_root = Path(dataset_root)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.dataset_root / DEFAULT_CACHE_DIRNAME

    @property
    def manifest_path(self) -> Path:
        return self.cache_dir / "manifest.json"

    @property
    def dataset_json_path(self) -> Path:
        return self.cache_dir / "dataset.json"

    @property
    def genes_path(self) -> Path:
        return self.cache_dir / "genes.parquet"

    @property
    def fovs_path(self) -> Path:
        return self.cache_dir / "fovs.parquet"

    @property
    def spots_dir(self) -> Path:
        return self.cache_dir / "spots"

    @property
    def images_dir(self) -> Path:
        return self.cache_dir / "images.zarr"

    @property
    def mosaic_metadata_path(self) -> Path:
        return self.images_dir / "mosaic_metadata.json"

    @property
    def logs_dir(self) -> Path:
        return self.cache_dir / "logs"

    @property
    def settings_path(self) -> Path:
        return self.cache_dir / "settings.json"

    @property
    def tmp_dir(self) -> Path:
        return self.cache_dir / ".build_tmp"

    def exists(self) -> bool:
        return self.manifest_path.is_file()
