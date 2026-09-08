"""End-to-end: source folder -> validation -> indexing -> cache -> viewport query -> layer data.

This is the acceptance gate from spec section 32: transcript overlays must
be validated *quantitatively* against known fixture coordinates, not just
be "visually plausible".
"""

from __future__ import annotations

import pytest
import zarr
from fixtures.synthetic_dataset import build_synthetic_dataset

from merfishviewerxp.api import MerfishDataset, index_dataset
from merfishviewerxp.config import AppConfig
from merfishviewerxp.indexed_dataset import IndexedDataset
from merfishviewerxp.model.transforms import world_um_to_local_pixel

TOLERANCE_UM = 1e-6


@pytest.fixture
def synthetic_root(tmp_path):
    root = tmp_path / "experiment"
    info = build_synthetic_dataset(root)
    return root, info


def test_discovery_and_validation(synthetic_root):
    root, _ = synthetic_root
    ds = MerfishDataset.open(root)
    assert len(ds.descriptor.fovs) == 4
    assert {c.channel_id for c in ds.descriptor.channels} == {"nucleus", "membrane"}
    assert {cb.codebook_id for cb in ds.descriptor.codebooks} == {"CB0", "CB1"}
    assert {b.export_id for b in ds.descriptor.barcode_exports} == {"CB0", "CB1"}
    # no global coords in the synthetic barcodes -> orientation defaults to
    # applying the documented (non-trivial) flip_horizontal flag, unvalidated
    assert ds.descriptor.image_orientation_apply is True

    report = ds.validate()
    assert report.is_valid, report.issues


def test_full_pipeline_transcript_overlay(synthetic_root):
    root, info = synthetic_root
    ds = MerfishDataset.open(root)
    cache_dir = root / "merfishviewerxp_cache"
    index_dataset(ds, cache_dir=cache_dir, config=AppConfig())

    indexed = IndexedDataset.open(cache_dir)

    genes = indexed.genes()
    gene_names = set(genes["gene_name"])
    assert {"GENEA", "GENEB", "exo_GENEC"}.issubset(gene_names)
    # blanks from different codebooks must never be merged
    assert "Blank-1 [CB0]" in gene_names
    assert "Blank-1 [CB1]" in gene_names

    genea_id = int(genes.loc[genes.gene_name == "GENEA", "gene_id"].iloc[0])
    table, lod = indexed.query_spots(bounds_um=(-100, -100, 200, 200), gene_ids=[genea_id])
    assert not lod["sampled"]
    df = table.to_pandas().sort_values("fov_id").reset_index(drop=True)
    assert len(df) == 4  # one GENEA transcript per FOV

    for _, row in df.iterrows():
        expected_x, expected_y = info["expected_marker_world_um"][int(row.fov_id)]
        assert row.world_x_um == pytest.approx(expected_x, abs=TOLERANCE_UM)
        assert row.world_y_um == pytest.approx(expected_y, abs=TOLERANCE_UM)
        assert row.gene_name == "GENEA"
        assert not row.is_blank
        assert row.codebook_id == "CB0"


def test_full_pipeline_image_mosaic_alignment(synthetic_root):
    root, info = synthetic_root
    ds = MerfishDataset.open(root)
    cache_dir = root / "merfishviewerxp_cache"
    index_dataset(ds, cache_dir=cache_dir, config=AppConfig())

    indexed = IndexedDataset.open(cache_dir)
    mosaic_meta = indexed.mosaic_metadata()["nucleus"]
    origin_x, origin_y = mosaic_meta["origin_world_um"]

    level0 = indexed.image_pyramid("nucleus")[0]
    assert isinstance(level0, zarr.Array)

    # For each FOV, the marker's world coordinate must map back onto the
    # mosaic pixel that actually holds the bright marker value.
    for fov_id, (x_um, y_um) in info["expected_marker_world_um"].items():
        row, col, _ = world_um_to_local_pixel(
            x_um,
            y_um,
            fov_origin_x_um=origin_x,
            fov_origin_y_um=origin_y,
            image_height=mosaic_meta["shape_yx"][0],
            image_width=mosaic_meta["shape_yx"][1],
            microscope=indexed.microscope,
            apply_orientation=False,  # mosaic pixel space is already global, no further orientation
        )
        value = level0[0, round(row), round(col)]
        assert value > 500.0, f"FOV {fov_id}: expected bright marker at mosaic pixel ({row},{col}), got {value}"


def test_query_spots_prunes_by_viewport(synthetic_root):
    root, info = synthetic_root
    ds = MerfishDataset.open(root)
    cache_dir = root / "merfishviewerxp_cache"
    index_dataset(ds, cache_dir=cache_dir, config=AppConfig())
    indexed = IndexedDataset.open(cache_dir)

    # A tiny viewport far away from every FOV should return zero spots.
    table, _ = indexed.query_spots(bounds_um=(10_000, 10_000, 10_010, 10_010))
    assert table.num_rows == 0


def test_cache_reopen_is_a_hit_without_rebuild(synthetic_root):
    root, info = synthetic_root
    ds = MerfishDataset.open(root)
    cache_dir = root / "merfishviewerxp_cache"
    m1 = index_dataset(ds, cache_dir=cache_dir, config=AppConfig())
    m2 = index_dataset(MerfishDataset.open(root), cache_dir=cache_dir, config=AppConfig())
    assert m1.created_at == m2.created_at  # untouched manifest -> no rebuild happened
