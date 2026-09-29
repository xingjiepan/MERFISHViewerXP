from __future__ import annotations

import numpy as np
import zarr

from merfishviewerxp.indexing.pyramid import build_pyramid
from merfishviewerxp.indexing.segmentation import label_boundaries
from merfishviewerxp.storage.zarr_store import create_array


def test_boundaries_are_the_inner_ring_of_a_cell():
    labels = np.zeros((1, 6, 6), dtype=np.int32)
    labels[0, 1:5, 1:5] = 7

    boundary = label_boundaries(labels)[0]

    expected = np.zeros((6, 6), dtype=bool)
    expected[1:5, 1:5] = True
    expected[2:4, 2:4] = False  # interior
    np.testing.assert_array_equal(boundary, expected)


def test_touching_cells_each_get_their_own_edge():
    labels = np.array([[[1, 1, 2, 2]]], dtype=np.int32)
    boundary = label_boundaries(labels)[0, 0]
    # both sides of the 1|2 interface are boundary pixels; pixels touching only their own label are not
    np.testing.assert_array_equal(boundary, [False, True, True, False])


def test_boundaries_are_computed_per_z_plane():
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    labels[:, 1:4, 1:4] = 3  # same cell spanning both planes

    boundary = label_boundaries(labels)

    # the interior pixel is not a boundary in either plane: differences across z are ignored
    assert not boundary[0, 2, 2]
    assert not boundary[1, 2, 2]
    np.testing.assert_array_equal(boundary[0], boundary[1])


def test_background_is_never_a_boundary():
    labels = np.zeros((1, 4, 4), dtype=np.int32)
    labels[0, 0, 0] = 1
    boundary = label_boundaries(labels)
    assert boundary.sum() == 1 and boundary[0, 0, 0]


def test_max_reduction_pyramid_keeps_one_pixel_lines(tmp_path):
    chunk_size = 32
    level0 = create_array(tmp_path / "0", shape=(1, 256, 256), chunks=(1, chunk_size, chunk_size), dtype=np.uint8)
    level0[0, 101, :] = 1  # a 1-pixel horizontal boundary line

    created = build_pyramid(tmp_path / "0", tmp_path, chunk_size=chunk_size, min_dim_px=32, reduction="max")

    assert created
    for level_idx in created:
        arr = zarr.open_array(str(tmp_path / str(level_idx)), mode="r")[:]
        assert arr.dtype == np.uint8
        row = 101 // (2**level_idx)
        assert arr[0, row, :].min() == 1, f"level {level_idx}: line lost"
        assert arr.sum() == arr.shape[2], f"level {level_idx}: line should stay one pixel thick"


def test_mean_reduction_is_still_the_default(tmp_path):
    level0 = create_array(tmp_path / "0", shape=(1, 64, 64), chunks=(1, 32, 32), dtype=np.float32)
    level0[0, 0:2, 0:2] = [[4.0, 0.0], [0.0, 0.0]]
    build_pyramid(tmp_path / "0", tmp_path, chunk_size=32, min_dim_px=32)
    assert zarr.open_array(str(tmp_path / "1"), mode="r")[0, 0, 0] == 1.0
