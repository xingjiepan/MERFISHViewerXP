"""Public read API over a built cache (spec section 25).

Deliberately independent of napari so it can be reused by other frontends
(spec section 29.6) and by CLI diagnostics.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_dataset
import zarr

from .errors import MerfishViewerXPError
from .indexing.manifest import Manifest, load_manifest
from .model.spots import DEFAULT_SPATIAL_TILE_SIZE_UM, MAX_VISIBLE_POINTS_DEFAULT
from .model.transforms import MicroscopeTransformParameters
from .query.lod import apply_lod
from .query.viewport import query_spots as _query_spots
from .storage.cache import CacheManager
from .storage.dataset_json import load_dataset_summary, microscope_from_summary
from .storage.parquet_store import open_dataset


class IndexedDataset:
    """Read-only handle to a built MERFISHViewerXP cache."""

    def __init__(self, cache: CacheManager, manifest: Manifest, dataset_summary: dict) -> None:
        self.cache = cache
        self.manifest = manifest
        self.dataset_summary = dataset_summary
        self._spots_dataset: pa_dataset.Dataset | None = None
        self._genes_df: pd.DataFrame | None = None
        self._fovs_df: pd.DataFrame | None = None
        self._mosaic_metadata: dict | None = None

    @classmethod
    def open(cls, cache_dir: Path) -> IndexedDataset:
        cache_dir = Path(cache_dir)
        cache = CacheManager(dataset_root=cache_dir.parent, cache_dir=cache_dir)
        manifest = load_manifest(cache.cache_dir)
        if manifest is None:
            raise MerfishViewerXPError(
                f"No valid cache manifest found at {cache.manifest_path}.\n"
                "Run `merfishviewerxp index <experiment>` first."
            )
        if not cache.dataset_json_path.is_file():
            raise MerfishViewerXPError(f"Cache at {cache.cache_dir} is missing dataset.json; rebuild the cache.")
        summary = load_dataset_summary(cache.dataset_json_path)
        return cls(cache, manifest, summary)

    @property
    def dataset_id(self) -> str:
        return self.dataset_summary["dataset_id"]

    @property
    def microscope(self) -> MicroscopeTransformParameters:
        return microscope_from_summary(self.dataset_summary)

    @property
    def image_orientation_apply(self) -> bool:
        return self.dataset_summary["image_orientation_apply"]

    def genes(self) -> pd.DataFrame:
        if self._genes_df is None:
            self._genes_df = pd.read_parquet(self.cache.genes_path)
        return self._genes_df

    def fovs(self) -> pd.DataFrame:
        if self._fovs_df is None:
            self._fovs_df = pd.read_parquet(self.cache.fovs_path)
        return self._fovs_df

    def channels(self) -> list[str]:
        return list(self.dataset_summary["channels"])

    def _spots(self) -> pa_dataset.Dataset:
        if self._spots_dataset is None:
            self._spots_dataset = open_dataset(self.cache.spots_dir)
        return self._spots_dataset

    def _tile_size_um(self) -> float:
        return self.manifest.indexing_config.get("spots", {}).get("spatial_tile_size_um", DEFAULT_SPATIAL_TILE_SIZE_UM)

    def query_spots(
        self,
        *,
        bounds_um: tuple[float, float, float, float],
        gene_ids: Iterable[int] | None = None,
        z_range_um: tuple[float, float] | None = None,
        include_blanks: bool = False,
        max_visible_points: int = MAX_VISIBLE_POINTS_DEFAULT,
        apply_lod_sampling: bool = True,
    ) -> tuple[pa.Table, dict]:
        xmin, ymin, xmax, ymax = bounds_um
        table = _query_spots(
            self._spots(),
            xmin_um=xmin,
            ymin_um=ymin,
            xmax_um=xmax,
            ymax_um=ymax,
            gene_ids=gene_ids,
            z_range_um=z_range_um,
            include_blanks=include_blanks,
            tile_size_um=self._tile_size_um(),
        )
        if apply_lod_sampling:
            return apply_lod(table, max_visible_points)
        return table, {"sampled": False, "total_in_view": table.num_rows, "shown": table.num_rows}

    def mosaic_metadata(self) -> dict:
        if self._mosaic_metadata is None:
            self._mosaic_metadata = json.loads(self.cache.mosaic_metadata_path.read_text())
        return self._mosaic_metadata

    def image_pyramid(self, channel_id: str) -> list[zarr.Array]:
        channel_dir = self.cache.images_dir / channel_id
        if not channel_dir.is_dir():
            raise MerfishViewerXPError(
                f"No image mosaic cached for channel {channel_id!r} under {channel_dir}.\n"
                f"Available channels: {self.channels()}."
            )
        levels = sorted(
            (p for p in channel_dir.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        )
        if not levels:
            raise MerfishViewerXPError(f"No pyramid levels found under {channel_dir}.")
        return [zarr.open_array(str(p), mode="r") for p in levels]
