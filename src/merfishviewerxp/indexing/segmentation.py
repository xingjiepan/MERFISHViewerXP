"""Build a global cell-boundary mosaic from per-FOV segmentation label masks.

Each FOV's mask is a (z, y, x) label image (0 = background, one integer per
cell, numbered independently in every FOV). Neither blending nor
mean-downsampling preserves labels, and full label images would be large, so
the cache stores *boundaries*: a uint32 mosaic that holds a dataset-wide cell
id on the inner edge of every cell in each z-plane and 0 elsewhere. A cell's
id is its FOV's label offset plus its label within that FOV, so ids are unique
across FOVs and stable as long as the masks don't change. The mosaic uses
exactly the same global grid as the stain-image mosaic, so it overlays the
images with no extra transform, and the viewer's z modes (single plane, max
projection) apply to it unchanged.

Where FOVs overlap, boundaries from every FOV are kept (union), so both
segmentations of a cell in an overlap band remain visible (as two ids).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile

from ..errors import DatasetValidationError
from ..model.dataset import DatasetDescriptor
from ..storage.zarr_store import create_array
from .image_mosaic import (
    MosaicGeometry,
    compute_global_mosaic_geometry,
    fov_pixel_extent,
    orient_image_array,
    touched_chunks_for_fovs,
)

logger = logging.getLogger(__name__)

BOUNDARY_ARRAY_NAME = "boundaries"
# Bump whenever the cached boundary representation changes, so existing caches rebuild it.
# 1: uint8 0/1 boundaries. 2: uint32 dataset-wide cell ids on boundaries.
SEGMENTATION_FORMAT_VERSION = 2


@dataclass
class BoundaryMosaicResult:
    geometry: MosaicGeometry
    touched_chunks: set[tuple[int, int]]
    n_fovs: int
    # (fov_id, id offset, max local label): cell id = offset + local label
    cell_id_ranges: list[tuple[int, int, int]] = field(default_factory=list)

    @property
    def n_cell_ids(self) -> int:
        return sum(n for _, _, n in self.cell_id_ranges)


def label_boundaries(labels: np.ndarray) -> np.ndarray:
    """Boolean mask of inner cell boundaries, computed independently in each z-plane.

    A pixel is a boundary pixel if it belongs to a cell (label != 0) and at
    least one of its 4-connected in-plane neighbors has a different label
    (another cell or background). Neighbors across z are ignored, so the top
    and bottom planes of a cell are not drawn as solid boundaries.
    """
    boundary = np.zeros(labels.shape, dtype=bool)
    diff_y = labels[..., 1:, :] != labels[..., :-1, :]
    boundary[..., 1:, :] |= diff_y
    boundary[..., :-1, :] |= diff_y
    diff_x = labels[..., :, 1:] != labels[..., :, :-1]
    boundary[..., :, 1:] |= diff_x
    boundary[..., :, :-1] |= diff_x
    boundary &= labels != 0
    return boundary


def _read_mask_labels(path: Path) -> np.ndarray:
    mask = tifffile.imread(path)
    if mask.ndim == 2:
        mask = mask[None, :, :]
    if np.issubdtype(mask.dtype, np.floating):
        # CellPose exports label ids as float32; round so float noise can't split a cell.
        mask = np.rint(mask).astype(np.int32)
    return mask


def build_boundary_mosaic(
    dataset: DatasetDescriptor,
    *,
    level0_out_path: Path,
    chunk_size: int = 512,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BoundaryMosaicResult:
    """Write the level-0 cell-id boundary mosaic."""
    geometry = compute_global_mosaic_geometry(dataset)
    fovs = [f for f in dataset.fovs if f.mask_path is not None and f.image_shape_zyx is not None]
    out = create_array(
        level0_out_path,
        shape=(geometry.z_count, geometry.height_px, geometry.width_px),
        chunks=(1, chunk_size, chunk_size),
        dtype=np.uint32,
    )

    cell_id_ranges: list[tuple[int, int, int]] = []
    next_offset = 0
    for i, fov in enumerate(fovs):
        extent = fov_pixel_extent(fov, dataset, geometry)
        if extent is not None:
            row0, col0, row0c, col0c, row1c, col1c = extent
            labels = _read_mask_labels(fov.mask_path)
            if tuple(labels.shape) != tuple(fov.image_shape_zyx):
                raise DatasetValidationError(
                    f"Segmentation mask {fov.mask_path} has shape {labels.shape}, but FOV {fov.fov_id}'s "
                    f"stain images have shape {fov.image_shape_zyx}.\n"
                    "Cell boundaries can only be overlaid when each mask has the same (z, y, x) shape "
                    "as its FOV's images. Pass --no-segmentation to index/view without cell boundaries."
                )
            if dataset.image_orientation_apply:
                labels = orient_image_array(
                    labels,
                    flip_horizontal=dataset.microscope.flip_horizontal,
                    flip_vertical=dataset.microscope.flip_vertical,
                    transpose=dataset.microscope.transpose,
                )
            max_label = int(labels.max()) if labels.size else 0
            if next_offset + max_label > np.iinfo(np.uint32).max:
                raise DatasetValidationError("Too many segmented cells for uint32 cell ids.")
            cell_ids = np.where(label_boundaries(labels), labels.astype(np.uint32) + np.uint32(next_offset), 0)
            crop = cell_ids[:, row0c - row0 : row1c - row0, col0c - col0 : col1c - col0].astype(np.uint32)
            existing = out[:, row0c:row1c, col0c:col1c]
            out[:, row0c:row1c, col0c:col1c] = np.maximum(existing, crop)
            cell_id_ranges.append((fov.fov_id, next_offset, max_label))
            next_offset += max_label
        if progress_callback:
            progress_callback(i + 1, len(fovs))

    touched = touched_chunks_for_fovs(fovs, dataset, geometry, chunk_size=chunk_size)
    logger.info(
        "Built cell-boundary mosaic from %d FOV masks (%d cell ids), geometry=%s",
        len(cell_id_ranges),
        next_offset,
        geometry,
    )
    return BoundaryMosaicResult(
        geometry=geometry, touched_chunks=touched, n_fovs=len(cell_id_ranges), cell_id_ranges=cell_id_ranges
    )
