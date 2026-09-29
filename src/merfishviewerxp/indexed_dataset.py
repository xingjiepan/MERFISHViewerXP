"""Public read API over a built cache (spec section 25).

Deliberately independent of napari so it can be reused by other frontends
(spec section 29.6) and by CLI diagnostics.
"""

from __future__ import annotations

import bisect
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_dataset
import zarr

from .errors import MerfishViewerXPError
from .indexing.manifest import Manifest, load_manifest
from .indexing.segmentation import BOUNDARY_ARRAY_NAME, SEGMENTATION_FORMAT_VERSION
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
        self._segmentation_metadata: dict | None = None
        self._cell_id_offsets: tuple[list[int], list[dict]] | None = None

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
        return _open_pyramid_levels(channel_dir)

    def has_cell_boundaries(self) -> bool:
        if not (self.manifest.components_built.get("segmentation") and self.cache.segmentation_metadata_path.is_file()):
            return False
        # A boundary cache in an older format (e.g. kept via --no-segmentation) is not usable.
        return self.cell_boundary_metadata().get("format_version") == SEGMENTATION_FORMAT_VERSION

    def cell_boundary_metadata(self) -> dict:
        """Mosaic geometry of the cell-boundary overlay (same keys as `mosaic_metadata()` entries),
        plus ``cell_id_ranges`` for mapping cell ids back to FOVs."""
        if self._segmentation_metadata is None:
            self._segmentation_metadata = json.loads(self.cache.segmentation_metadata_path.read_text())
        return self._segmentation_metadata[BOUNDARY_ARRAY_NAME]

    def cell_id_source(self, cell_id: int) -> tuple[int, int] | None:
        """``(fov_id, label within that FOV's mask)`` for a dataset-wide cell id."""
        if self._cell_id_offsets is None:
            ranges = sorted(
                (r for r in self.cell_boundary_metadata()["cell_id_ranges"] if r["max_label"] > 0),
                key=lambda r: r["offset"],
            )
            self._cell_id_offsets = ([r["offset"] for r in ranges], ranges)
        offsets, ranges = self._cell_id_offsets
        i = bisect.bisect_left(offsets, cell_id) - 1
        if i < 0 or cell_id - offsets[i] > ranges[i]["max_label"]:
            return None
        return int(ranges[i]["fov_id"]), int(cell_id - offsets[i])

    def cell_boundary_pyramid(self) -> list[zarr.Array]:
        """uint32 (z, y, x) pyramid levels: the dataset-wide cell id on segmented cell
        boundaries, 0 elsewhere (see indexing.segmentation)."""
        if not self.has_cell_boundaries():
            raise MerfishViewerXPError(
                f"No cell-boundary cache under {self.cache.segmentation_dir}.\n"
                "Segmentation masks (CellPoseSegment/images/segmented_mask<fov>.tif) were not found "
                "or not indexed; run `merfishviewerxp index <experiment>`."
            )
        return _open_pyramid_levels(self.cache.segmentation_dir / BOUNDARY_ARRAY_NAME)


def _open_pyramid_levels(pyramid_dir: Path) -> list[zarr.Array]:
    levels = sorted(
        (p for p in pyramid_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    if not levels:
        raise MerfishViewerXPError(f"No pyramid levels found under {pyramid_dir}.")
    return [zarr.open_array(str(p), mode="r") for p in levels]
