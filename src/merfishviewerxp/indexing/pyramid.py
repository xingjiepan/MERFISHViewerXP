"""Build multiscale pyramid levels from a level-0 mosaic (spec 8.7)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import zarr

from ..storage.zarr_store import create_array

logger = logging.getLogger(__name__)


def _downsample_block(block: np.ndarray, factor: int) -> np.ndarray:
    """Area/mean downsample a (z, y, x) block by `factor` in y and x."""
    _z, h, w = block.shape
    pad_h = (-h) % factor
    pad_w = (-w) % factor
    if pad_h or pad_w:
        block = np.pad(block, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
    zz, bh, bw = block.shape
    reshaped = block.reshape(zz, bh // factor, factor, bw // factor, factor)
    return reshaped.mean(axis=(2, 4))


def build_pyramid(
    level0_path: Path,
    out_dir: Path,
    *,
    downsample_factor: int = 2,
    chunk_size: int = 512,
    min_dim_px: int = 1024,
    max_levels: int = 12,
    active_chunks: set[tuple[int, int]] | None = None,
) -> list[int]:
    """Create levels 1..N under ``out_dir`` from the level-0 array. Returns level indices created.

    ``active_chunks`` (row_chunk, col_chunk) at level 0's chunk size lets a
    sparse mosaic (e.g. several disjoint organoids on one slide, far apart
    relative to their own size) skip empty regions at every level instead of
    scanning the full -- potentially mostly-empty -- bounding-box grid.
    Remapped level-to-level assuming a fixed chunk size and downsample_factor.
    """
    current = zarr.open_array(str(level0_path), mode="r")
    z, h, w = current.shape
    created: list[int] = []
    level_idx = 1
    chunks_this_level = active_chunks
    while (h > min_dim_px or w > min_dim_px) and level_idx < max_levels:
        new_h = -(-h // downsample_factor)
        new_w = -(-w // downsample_factor)
        out_path = out_dir / str(level_idx)
        out_arr = create_array(
            out_path,
            shape=(z, new_h, new_w),
            chunks=(1, min(chunk_size, new_h), min(chunk_size, new_w)),
            dtype=current.dtype,
        )

        n_chunk_rows = -(-new_h // chunk_size)
        n_chunk_cols = -(-new_w // chunk_size)
        if chunks_this_level is None:
            targets = [(cr, cc) for cr in range(n_chunk_rows) for cc in range(n_chunk_cols)]
        else:
            # `chunks_this_level` is expressed in the *source* (current) level's
            # chunk-index space. It must be rescaled into the destination
            # level's chunk-index space -- via the same halving relationship
            # as the pixel dimensions -- before use, not just clamped into
            # range. Clamping alone (the original bug here) silently computes
            # the wrong destination position for nearly every chunk.
            rescaled = {(cr // downsample_factor, cc // downsample_factor) for cr, cc in chunks_this_level}
            targets = sorted({(min(cr, n_chunk_rows - 1), min(cc, n_chunk_cols - 1)) for cr, cc in rescaled})

        for cr, cc in targets:
            row0, row1 = cr * chunk_size, min((cr + 1) * chunk_size, new_h)
            col0, col1 = cc * chunk_size, min((cc + 1) * chunk_size, new_w)
            src_row0, src_row1 = row0 * downsample_factor, min(row1 * downsample_factor, h)
            src_col0, src_col1 = col0 * downsample_factor, min(col1 * downsample_factor, w)
            block = current[:, src_row0:src_row1, src_col0:src_col1]
            out_arr[:, row0:row1, col0:col1] = _downsample_block(block, downsample_factor).astype(current.dtype)

        created.append(level_idx)
        current = out_arr
        h, w = new_h, new_w
        level_idx += 1
        if chunks_this_level is not None:
            chunks_this_level = set(targets)  # already rescaled into this (now-current) level's own index space

    logger.info("Built %d pyramid level(s) under %s (final shape %s)", len(created), out_dir, (z, h, w))
    return created
