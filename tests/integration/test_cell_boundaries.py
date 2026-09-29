"""End-to-end: segmentation masks -> cell-boundary mosaic aligned with the stain images."""

from __future__ import annotations

import numpy as np
import pytest
from fixtures.synthetic_dataset import build_synthetic_dataset, write_synthetic_masks

from merfishviewerxp.api import MerfishDataset, index_dataset
from merfishviewerxp.config import AppConfig
from merfishviewerxp.indexed_dataset import IndexedDataset
from merfishviewerxp.model.transforms import world_um_to_local_pixel


def _mosaic_pixel(indexed: IndexedDataset, meta: dict, x_um: float, y_um: float) -> tuple[int, int]:
    origin_x, origin_y = meta["origin_world_um"]
    row, col, _ = world_um_to_local_pixel(
        x_um,
        y_um,
        fov_origin_x_um=origin_x,
        fov_origin_y_um=origin_y,
        image_height=meta["shape_yx"][0],
        image_width=meta["shape_yx"][1],
        microscope=indexed.microscope,
        apply_orientation=False,
    )
    return round(row), round(col)


@pytest.fixture
def indexed_with_masks(tmp_path):
    root = tmp_path / "experiment"
    info = build_synthetic_dataset(root, with_masks=True)
    cache_dir = root / "merfishviewerxp_cache"
    manifest = index_dataset(MerfishDataset.open(root), cache_dir=cache_dir, config=AppConfig())
    return IndexedDataset.open(cache_dir), manifest, info


def test_masks_are_discovered_per_fov_but_are_not_a_stain_channel(tmp_path):
    root = tmp_path / "experiment"
    build_synthetic_dataset(root, with_masks=True)
    descriptor = MerfishDataset.open(root).descriptor
    assert all(fov.mask_path is not None and fov.mask_path.name.startswith("segmented_mask") for fov in descriptor.fovs)
    assert {c.channel_id for c in descriptor.channels} == {"nucleus", "membrane"}


def test_boundary_mosaic_aligns_with_the_image_mosaic(indexed_with_masks):
    indexed, manifest, info = indexed_with_masks
    assert manifest.components_built["segmentation"] is True
    assert indexed.has_cell_boundaries()

    meta = indexed.cell_boundary_metadata()
    image_meta = indexed.mosaic_metadata()["nucleus"]
    for key in ("origin_world_um", "pixel_size_um", "shape_yx", "z_count"):
        assert meta[key] == image_meta[key]
    assert meta["n_fovs"] == 4

    level0 = indexed.cell_boundary_pyramid()[0]
    assert level0.dtype == np.uint8
    image_level0 = indexed.image_pyramid("nucleus")[0]
    for fov_id, (x_um, y_um) in info["expected_marker_world_um"].items():
        row, col = _mosaic_pixel(indexed, meta, x_um, y_um)
        # the marker pixel is the cell's corner: a boundary, at the same place as the bright image marker
        assert level0[0, row, col] == 1, f"FOV {fov_id}: expected a boundary at mosaic pixel ({row},{col})"
        assert image_level0[0, row, col] > 500.0
    for fov_id, (x_um, y_um) in info["expected_cell_interior_world_um"].items():
        row, col = _mosaic_pixel(indexed, meta, x_um, y_um)
        assert level0[0, row, col] == 0, f"FOV {fov_id}: cell interior at ({row},{col}) must not be a boundary"


def test_boundary_pyramid_levels_are_readable(indexed_with_masks):
    indexed, _, _ = indexed_with_masks
    levels = indexed.cell_boundary_pyramid()
    assert levels
    assert all(level.dtype == np.uint8 and level[:].max() <= 1 for level in levels)


def test_adding_masks_later_builds_only_the_boundaries(tmp_path):
    root = tmp_path / "experiment"
    build_synthetic_dataset(root)
    cache_dir = root / "merfishviewerxp_cache"
    first = index_dataset(MerfishDataset.open(root), cache_dir=cache_dir, config=AppConfig())
    assert not first.components_built.get("segmentation", False)
    assert not IndexedDataset.open(cache_dir).has_cell_boundaries()
    images_zarray = cache_dir / "images.zarr" / "nucleus" / "0" / ".zarray"
    images_mtime = images_zarray.stat().st_mtime_ns

    write_synthetic_masks(root)
    second = index_dataset(MerfishDataset.open(root), cache_dir=cache_dir, config=AppConfig())

    assert second.components_built["segmentation"] is True
    assert IndexedDataset.open(cache_dir).has_cell_boundaries()
    assert images_zarray.stat().st_mtime_ns == images_mtime, "stain mosaics must not be rebuilt"


def test_removing_masks_removes_stale_boundaries(tmp_path):
    root = tmp_path / "experiment"
    build_synthetic_dataset(root, with_masks=True)
    cache_dir = root / "merfishviewerxp_cache"
    index_dataset(MerfishDataset.open(root), cache_dir=cache_dir, config=AppConfig())

    for mask in (root / "CellPoseSegment" / "images").glob("segmented_mask*.tif"):
        mask.unlink()
    manifest = index_dataset(MerfishDataset.open(root), cache_dir=cache_dir, config=AppConfig())

    assert manifest.components_built["segmentation"] is False
    assert not IndexedDataset.open(cache_dir).has_cell_boundaries()


def test_no_segmentation_flag_skips_the_boundaries(tmp_path):
    root = tmp_path / "experiment"
    build_synthetic_dataset(root, with_masks=True)
    cache_dir = root / "merfishviewerxp_cache"
    manifest = index_dataset(MerfishDataset.open(root), cache_dir=cache_dir, config=AppConfig(), build_segmentation=False)
    assert not manifest.components_built.get("segmentation", False)
    assert not IndexedDataset.open(cache_dir).has_cell_boundaries()


def test_mask_with_wrong_shape_is_rejected_clearly(tmp_path):
    import tifffile

    from merfishviewerxp.errors import DatasetValidationError

    root = tmp_path / "experiment"
    build_synthetic_dataset(root, with_masks=True)
    tifffile.imwrite(root / "CellPoseSegment" / "images" / "segmented_mask2.tif", np.zeros((1, 5, 5), dtype=np.float32))

    with pytest.raises(DatasetValidationError, match="--no-segmentation"):
        index_dataset(MerfishDataset.open(root), cache_dir=root / "merfishviewerxp_cache", config=AppConfig())
