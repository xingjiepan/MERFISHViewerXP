"""Regression coverage for pyramid level-to-level chunk remapping.

The synthetic GUI/integration fixtures are small enough to fit in a single
chunk at every level, so a bug in how `active_chunks` gets rescaled between
levels would never show up there. These tests use a chunk_size small enough,
relative to the array, that multiple distinct chunks exist -- and place data
far from the origin chunk, where an off-by-rescale bug would put it in the
wrong place or drop it entirely.
"""

from __future__ import annotations

import numpy as np
import zarr

from merfishviewerxp.indexing.pyramid import build_pyramid
from merfishviewerxp.storage.zarr_store import create_array


def _make_level0(path, *, shape, chunk_size, marker_row_chunk, marker_col_chunk, marker_value=1000.0):
    arr = create_array(path, shape=shape, chunks=(1, chunk_size, chunk_size), dtype=np.float32)
    r0, r1 = marker_row_chunk * chunk_size, (marker_row_chunk + 1) * chunk_size
    c0, c1 = marker_col_chunk * chunk_size, (marker_col_chunk + 1) * chunk_size
    arr[:, r0:r1, c0:c1] = marker_value
    return arr


def test_pyramid_preserves_far_from_origin_marker(tmp_path):
    chunk_size = 64
    shape = (1, 2048, 2048)
    marker_row_chunk, marker_col_chunk = 20, 25  # far from (0, 0); (2048/64=32 chunks per axis)

    level0_path = tmp_path / "0"
    _make_level0(
        level0_path, shape=shape, chunk_size=chunk_size, marker_row_chunk=marker_row_chunk, marker_col_chunk=marker_col_chunk
    )
    active_chunks = {(marker_row_chunk, marker_col_chunk)}

    created = build_pyramid(
        level0_path, tmp_path, chunk_size=chunk_size, min_dim_px=128, active_chunks=active_chunks
    )
    assert created, "expected at least one pyramid level to be built"

    expected_row_chunk, expected_col_chunk = marker_row_chunk, marker_col_chunk
    for level_idx in created:
        expected_row_chunk //= 2
        expected_col_chunk //= 2
        level_arr = zarr.open_array(str(tmp_path / str(level_idx)), mode="r")

        r0, r1 = expected_row_chunk * chunk_size, (expected_row_chunk + 1) * chunk_size
        c0, c1 = expected_col_chunk * chunk_size, (expected_col_chunk + 1) * chunk_size
        r1, c1 = min(r1, level_arr.shape[1]), min(c1, level_arr.shape[2])
        marker_region = level_arr[:, r0:r1, c0:c1]
        assert marker_region.max() > 500.0, (
            f"level {level_idx}: expected marker at chunk ({expected_row_chunk},{expected_col_chunk}), "
            f"found max={marker_region.max()}"
        )

        # a region far from the marker, well within the array, must remain empty
        far_region = level_arr[:, 0:chunk_size, 0:chunk_size]
        assert far_region.max() == 0.0, f"level {level_idx}: expected empty region at origin chunk, found data"


def test_pyramid_matches_full_grid_result_for_touched_chunks(tmp_path):
    """active_chunks must produce identical output to the exhaustive (None) path
    for the chunks it claims to cover."""
    chunk_size = 32
    shape = (1, 512, 512)
    marker_row_chunk, marker_col_chunk = 10, 12

    exhaustive_path = tmp_path / "exhaustive" / "0"
    sparse_path = tmp_path / "sparse" / "0"
    for path in (exhaustive_path, sparse_path):
        _make_level0(
            path, shape=shape, chunk_size=chunk_size, marker_row_chunk=marker_row_chunk, marker_col_chunk=marker_col_chunk
        )

    build_pyramid(exhaustive_path, tmp_path / "exhaustive", chunk_size=chunk_size, min_dim_px=64, active_chunks=None)
    build_pyramid(
        sparse_path,
        tmp_path / "sparse",
        chunk_size=chunk_size,
        min_dim_px=64,
        active_chunks={(marker_row_chunk, marker_col_chunk)},
    )

    level1_exhaustive = zarr.open_array(str(tmp_path / "exhaustive" / "1"), mode="r")[:]
    level1_sparse = zarr.open_array(str(tmp_path / "sparse" / "1"), mode="r")[:]
    np.testing.assert_array_equal(level1_exhaustive, level1_sparse)
