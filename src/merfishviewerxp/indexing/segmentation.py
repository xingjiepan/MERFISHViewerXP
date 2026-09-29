"""Build a global cell-boundary mosaic from per-FOV segmentation label masks.

Each FOV's mask is a (z, y, x) label image (0 = background, one integer per
cell, numbered independently in every FOV). Label ids are therefore not
comparable across FOVs, and neither blending nor mean-downsampling preserves
them, so the cache stores *boundaries* rather than labels: a uint8 mosaic that
is 1 on the inner edge of every cell in each z-plane and 0 elsewhere. It uses
exactly the same global grid as the stain-image mosaic, so it overlays the
images with no extra transform, and the viewer's z modes (single plane, max
projection) apply to it unchanged.

Where FOVs overlap, boundaries from every FOV are kept (union), so both
segmentations of a cell in an overlap band remain visible.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
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
) -> tuple[MosaicGeometry, set[tuple[int, int]], int]:
    """Write the level-0 boundary mosaic. Returns (geometry, touched chunks, FOVs placed)."""
    geometry = compute_global_mosaic_geometry(dataset)
    fovs = [f for f in dataset.fovs if f.mask_path is not None and f.image_shape_zyx is not None]
    out = create_array(
        level0_out_path,
        shape=(geometry.z_count, geometry.height_px, geometry.width_px),
        chunks=(1, chunk_size, chunk_size),
        dtype=np.uint8,
    )

    n_placed = 0
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
            boundary = label_boundaries(labels).astype(np.uint8)
            crop = boundary[:, row0c - row0 : row1c - row0, col0c - col0 : col1c - col0]
            existing = out[:, row0c:row1c, col0c:col1c]
            out[:, row0c:row1c, col0c:col1c] = np.maximum(existing, crop)
            n_placed += 1
        if progress_callback:
            progress_callback(i + 1, len(fovs))

    touched = touched_chunks_for_fovs(fovs, dataset, geometry, chunk_size=chunk_size)
    logger.info("Built cell-boundary mosaic from %d FOV masks, geometry=%s", n_placed, geometry)
    return geometry, touched, n_placed
