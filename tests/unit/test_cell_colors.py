from __future__ import annotations

import numpy as np
import pytest

from merfishviewerxp.indexing.segmentation import label_boundaries
from merfishviewerxp.model.genes import hex_to_rgb
from merfishviewerxp.viewer.layers import (
    cell_boundary_colormap,
    cell_boundary_contrast_limits,
    cell_color_table,
    color_cell_boundary_block,
    nearest_cell_id,
)


def test_color_table_gives_each_distinct_color_one_display_value():
    table = cell_color_table({30: "#00FF00", 7: "#ff0000", 12: "#00ff00"})
    assert table.palette == ("#00ff00", "#ff0000")
    assert table.cell_ids.tolist() == [7, 12, 30]
    assert table.display_index.tolist() == [3, 2, 2]  # 2 -> "#00ff00", 3 -> "#ff0000"
    assert cell_boundary_contrast_limits(table) == (0, 3)


def test_block_coloring_maps_background_default_and_overridden_cells():
    block = np.array([[[0, 5, 7], [12, 0, 99]]], dtype=np.uint32)
    table = cell_color_table({7: "#ff0000", 12: "#00ff00"})
    np.testing.assert_array_equal(color_cell_boundary_block(block, table), [[[0, 1, 3], [2, 0, 1]]])


def test_block_coloring_without_overrides_is_just_boundary_presence():
    block = np.array([[[0, 5], [4_000_000_000, 0]]], dtype=np.uint32)
    table = cell_color_table({})
    assert cell_boundary_contrast_limits(table) == (0, 1)
    np.testing.assert_array_equal(color_cell_boundary_block(block, table), [[[0, 1], [1, 0]]])


def test_colormap_bins_map_each_display_value_to_its_color():
    table = cell_color_table({1: "#00ff00", 2: "#0000ff"})
    cmap = cell_boundary_colormap("#ff00ff", table.palette)
    _, hi = cell_boundary_contrast_limits(table)
    rgba = cmap.map(np.arange(hi + 1) / hi)
    assert rgba[0][3] == 0  # not a boundary: transparent
    for value, color in [(1, "#ff00ff"), (2, "#0000ff"), (3, "#00ff00")]:
        assert tuple(rgba[value][:3]) == pytest.approx(hex_to_rgb(color), abs=1e-6)
        assert rgba[value][3] == pytest.approx(1.0)


def _two_cells() -> np.ndarray:
    labels = np.zeros((40, 60), dtype=np.uint32)
    labels[5:25, 5:25] = 101  # cell A
    labels[5:25, 25:45] = 202  # cell B, touching A
    return np.where(label_boundaries(labels), labels, 0)


def test_clicking_inside_a_cell_picks_that_cell_even_next_to_a_neighbor():
    ids = _two_cells()
    assert nearest_cell_id(ids, 15, 12, max_distance_px=30) == 101
    assert nearest_cell_id(ids, 15, 23, max_distance_px=30) == 101  # one pixel from the shared edge
    assert nearest_cell_id(ids, 15, 26, max_distance_px=30) == 202
    assert nearest_cell_id(ids, 15, 35, max_distance_px=30) == 202


def test_clicks_far_from_any_cell_or_outside_the_window_pick_nothing():
    ids = _two_cells()
    assert nearest_cell_id(ids, 38, 58, max_distance_px=5) is None
    assert nearest_cell_id(ids, -1, 10, max_distance_px=30) is None
    assert nearest_cell_id(np.zeros((10, 10), dtype=np.uint32), 5, 5, max_distance_px=30) is None
