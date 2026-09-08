from merfishviewerxp.model.spots import spatial_tile_index, tiles_overlapping_bounds


def test_spatial_tile_index_basic():
    assert spatial_tile_index(0, 0, 1000) == (0, 0)
    assert spatial_tile_index(999.9, 0, 1000) == (0, 0)
    assert spatial_tile_index(1000.0, 0, 1000) == (1, 0)
    assert spatial_tile_index(-1.0, 0, 1000) == (-1, 0)
    assert spatial_tile_index(-1000.0, -1000.0, 1000) == (-1, -1)


def test_tiles_overlapping_bounds_single_tile():
    tiles = tiles_overlapping_bounds(10, 10, 20, 20, tile_size_um=1000)
    assert tiles == [(0, 0)]


def test_tiles_overlapping_bounds_spans_multiple():
    tiles = tiles_overlapping_bounds(-50, -50, 1050, 1050, tile_size_um=1000)
    assert set(tiles) == {(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)}
