"""Small, stable public entry points (spec section 25)."""

from __future__ import annotations

from pathlib import Path

from .adapters.merlin.dataset import MerlinDatasetAdapter
from .adapters.merlin.dataset import validate_dataset as _validate_dataset
from .config import AppConfig, load_config
from .indexed_dataset import IndexedDataset
from .indexing.indexer import index_dataset as _index_dataset_impl
from .model.dataset import DatasetDescriptor, ValidationReport
from .storage.cache import CacheManager

__all__ = ["IndexedDataset", "MerfishDataset", "index_dataset"]


class MerfishDataset:
    """``dataset = MerfishDataset.open(path)``."""

    def __init__(self, descriptor: DatasetDescriptor) -> None:
        self.descriptor = descriptor

    @classmethod
    def open(cls, path: Path, *, channel_patterns: dict[str, str] | None = None) -> MerfishDataset:
        adapter = MerlinDatasetAdapter(Path(path), channel_patterns=channel_patterns)
        return cls(adapter.discover())

    def validate(self) -> ValidationReport:
        return _validate_dataset(self.descriptor)

    @property
    def root_path(self) -> Path:
        return self.descriptor.root_path


def index_dataset(
    dataset: MerfishDataset | DatasetDescriptor,
    *,
    cache_dir: Path | None = None,
    config: AppConfig | None = None,
    force: bool = False,
    **kwargs,
):
    descriptor = dataset.descriptor if isinstance(dataset, MerfishDataset) else dataset
    cache = CacheManager(dataset_root=descriptor.root_path, cache_dir=cache_dir)
    cfg = config or load_config(dataset_root=descriptor.root_path, cache_dir=cache_dir)
    return _index_dataset_impl(descriptor, cache=cache, config=cfg, force=force, **kwargs)
