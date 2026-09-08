"""Build the global per-channel image mosaic at native resolution (spec 8.3-8.6).

FOV overlaps are handled per an explicit, configurable policy: ``feather``
(default, edge-weighted blend), ``mean`` (uniform-weight blend), ``max``, or
``first``. The mosaic is a visualization product, not quantitative
fluorescence data (spec 8.6).
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from ..errors import DatasetValidationError
from ..model.dataset import DatasetDescriptor
from ..model.transforms import fov_bounding_box_um
from ..storage.zarr_store import create_array

logger = logging.getLogger(__name__)

OVERLAP_MODES = ("feather", "mean", "max", "first")


@dataclass(frozen=True)
class MosaicGeometry:
    origin_x_um: float
    origin_y_um: float
    width_px: int
    height_px: int
    z_count: int
    pixel_size_um: float


def compute_global_mosaic_geometry(dataset: DatasetDescriptor) -> MosaicGeometry:
    xs_min, ys_min, xs_max, ys_max = [], [], [], []
    z_counts: set[int] = set()
    for fov in dataset.fovs:
        if fov.image_shape_zyx is None:
            continue
        z, h, w = fov.image_shape_zyx
        z_counts.add(z)
        xmin, ymin, xmax, ymax = fov_bounding_box_um(
            fov_origin_x_um=fov.position_x_um,
            fov_origin_y_um=fov.position_y_um,
            image_height=h,
            image_width=w,
            microscope=dataset.microscope,
            apply_orientation=dataset.image_orientation_apply,
        )
        xs_min.append(xmin)
        ys_min.append(ymin)
        xs_max.append(xmax)
        ys_max.append(ymax)

    if not xs_min:
        raise DatasetValidationError("No FOVs with discovered images; cannot build an image mosaic.")
    if len(z_counts) > 1:
        raise DatasetValidationError(
            f"FOVs have inconsistent z-stack depths: {sorted(z_counts)}.\n"
            "MERFISHViewerXP's MVP mosaic builder requires a uniform z depth across all FOVs."
        )

    origin_x, origin_y = min(xs_min), min(ys_min)
    pixel_size = dataset.microscope.pixel_size_um
    width_px = int(np.ceil((max(xs_max) - origin_x) / pixel_size)) + 1
    height_px = int(np.ceil((max(ys_max) - origin_y) / pixel_size)) + 1
    return MosaicGeometry(
        origin_x_um=origin_x,
        origin_y_um=origin_y,
        width_px=width_px,
        height_px=height_px,
        z_count=z_counts.pop(),
        pixel_size_um=pixel_size,
    )


def orient_image_array(arr: np.ndarray, *, flip_horizontal: bool, flip_vertical: bool, transpose: bool) -> np.ndarray:
    """Apply the same orientation normalization as `model.transforms` to a (z, y, x) array."""
    if transpose:
        arr = arr.transpose(0, 2, 1)
    if flip_horizontal:
        arr = arr[:, :, ::-1]
    if flip_vertical:
        arr = arr[:, ::-1, :]
    return arr


def _feather_alpha(height: int, width: int, crop_px: int) -> np.ndarray:
    if crop_px <= 0:
        return np.ones((height, width), dtype=np.float32)
    ramp_y = np.minimum(np.arange(height), height - 1 - np.arange(height)).astype(np.float32)
    ramp_x = np.minimum(np.arange(width), width - 1 - np.arange(width)).astype(np.float32)
    alpha_y = np.clip(ramp_y / crop_px, 0.0, 1.0)
    alpha_x = np.clip(ramp_x / crop_px, 0.0, 1.0)
    return np.minimum(alpha_y[:, None], alpha_x[None, :])


def build_channel_mosaic(
    dataset: DatasetDescriptor,
    channel_id: str,
    *,
    level0_out_path: Path,
    tmp_dir: Path,
    chunk_size: int = 512,
    overlap_mode: str = "feather",
    fov_crop_px: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[MosaicGeometry, set[tuple[int, int]]]:
    if overlap_mode not in OVERLAP_MODES:
        raise ValueError(f"Unknown overlap_mode {overlap_mode!r}; expected one of {OVERLAP_MODES}")

    geometry = compute_global_mosaic_geometry(dataset)
    value_chunks = (1, chunk_size, chunk_size)
    weight_chunks = (chunk_size, chunk_size)

    tmp_dir.mkdir(parents=True, exist_ok=True)
    value_path = tmp_dir / f"{channel_id}_value.zarr"
    weight_path = tmp_dir / f"{channel_id}_weight.zarr"
    weighted = overlap_mode in ("feather", "mean")
    weight_dtype = np.float32 if weighted else np.uint8

    value_arr = create_array(
        value_path, shape=(geometry.z_count, geometry.height_px, geometry.width_px), chunks=value_chunks, dtype=np.float32
    )
    weight_arr = create_array(
        weight_path, shape=(geometry.height_px, geometry.width_px), chunks=weight_chunks, dtype=weight_dtype
    )

    # FOVs are often scattered (e.g. multiple disjoint organoids on one slide)
    # rather than tiling the full bounding box, so the finalize pass below
    # only visits chunks that actually received data instead of the entire
    # (potentially mostly-empty) bounding-box grid.
    touched_chunks: set[tuple[int, int]] = set()

    fovs_with_images = [f for f in dataset.fovs if channel_id in f.image_paths and f.image_shape_zyx is not None]
    for i, fov in enumerate(fovs_with_images):
        arr = tifffile.imread(fov.image_paths[channel_id]).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        if dataset.image_orientation_apply:
            arr = orient_image_array(
                arr,
                flip_horizontal=dataset.microscope.flip_horizontal,
                flip_vertical=dataset.microscope.flip_vertical,
                transpose=dataset.microscope.transpose,
            )

        _, h, w = arr.shape
        col0 = round((fov.position_x_um - geometry.origin_x_um) / geometry.pixel_size_um)
        row0 = round((fov.position_y_um - geometry.origin_y_um) / geometry.pixel_size_um)
        row0c, col0c = max(row0, 0), max(col0, 0)
        row1c, col1c = min(row0 + h, geometry.height_px), min(col0 + w, geometry.width_px)
        if row0c >= row1c or col0c >= col1c:
            continue
        arr_crop = arr[:, row0c - row0 : row1c - row0, col0c - col0 : col1c - col0]

        if weighted:
            alpha_full = _feather_alpha(h, w, fov_crop_px) if overlap_mode == "feather" else np.ones((h, w), dtype=np.float32)
            alpha_crop = alpha_full[row0c - row0 : row1c - row0, col0c - col0 : col1c - col0]
            value_arr[:, row0c:row1c, col0c:col1c] = value_arr[:, row0c:row1c, col0c:col1c] + arr_crop * alpha_crop[None, :, :]
            weight_arr[row0c:row1c, col0c:col1c] = weight_arr[row0c:row1c, col0c:col1c] + alpha_crop
        elif overlap_mode == "max":
            existing_covered = weight_arr[row0c:row1c, col0c:col1c]
            existing_value = value_arr[:, row0c:row1c, col0c:col1c]
            new_value = np.where(existing_covered[None, :, :] > 0, np.maximum(existing_value, arr_crop), arr_crop)
            value_arr[:, row0c:row1c, col0c:col1c] = new_value
            weight_arr[row0c:row1c, col0c:col1c] = 1
        else:  # first
            existing_covered = weight_arr[row0c:row1c, col0c:col1c]
            write_mask = existing_covered == 0
            existing_value = value_arr[:, row0c:row1c, col0c:col1c]
            new_value = np.where(write_mask[None, :, :], arr_crop, existing_value)
            value_arr[:, row0c:row1c, col0c:col1c] = new_value
            weight_arr[row0c:row1c, col0c:col1c] = np.maximum(existing_covered, write_mask.astype(np.uint8))

        for cr in range(row0c // chunk_size, (row1c - 1) // chunk_size + 1):
            for cc in range(col0c // chunk_size, (col1c - 1) // chunk_size + 1):
                touched_chunks.add((cr, cc))

        if progress_callback:
            progress_callback(i + 1, len(fovs_with_images))

    out_arr = create_array(
        level0_out_path, shape=value_arr.shape, chunks=value_chunks, dtype=np.float32
    )
    for cr, cc in sorted(touched_chunks):
        row0, row1 = cr * chunk_size, min((cr + 1) * chunk_size, geometry.height_px)
        col0, col1 = cc * chunk_size, min((cc + 1) * chunk_size, geometry.width_px)
        v = value_arr[:, row0:row1, col0:col1]
        if weighted:
            w_block = weight_arr[row0:row1, col0:col1]
            with np.errstate(invalid="ignore", divide="ignore"):
                result = np.where(w_block[None, :, :] > 0, v / np.maximum(w_block, 1e-6)[None, :, :], 0.0)
        else:
            result = v
        out_arr[:, row0:row1, col0:col1] = result.astype(np.float32)
    n_chunk_rows = -(-geometry.height_px // chunk_size)
    n_chunk_cols = -(-geometry.width_px // chunk_size)
    logger.info("Finalized %d of %d possible chunks (sparse mosaic)", len(touched_chunks), n_chunk_rows * n_chunk_cols)

    shutil.rmtree(value_path, ignore_errors=True)
    shutil.rmtree(weight_path, ignore_errors=True)
    logger.info(
        "Built %s mosaic: %d FOVs, geometry=%s, overlap_mode=%s", channel_id, len(fovs_with_images), geometry, overlap_mode
    )
    return geometry, touched_chunks
