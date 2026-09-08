import pandas as pd
import pyarrow as pa
import pytest

from merfishviewerxp.model.spots import SPOT_SCHEMA
from merfishviewerxp.query.lod import apply_lod
from merfishviewerxp.query.viewport import query_spots
from merfishviewerxp.storage.parquet_store import open_dataset, write_partitioned_chunks


def _make_spot_row(**overrides):
    row = {
        "spot_id": 0,
        "codebook_id": "CB0",
        "barcode_id": 0,
        "gene_id": 1,
        "gene_name": "GENEA",
        "is_blank": False,
        "mapping_status": "mapped",
        "fov_id": 0,
        "world_x_um": 0.0,
        "world_y_um": 0.0,
        "world_z_um": 0.0,
        "source_x": 0.0,
        "source_y": 0.0,
        "source_z": 0.0,
        "cell_index": -1,
        "mean_intensity": None,
        "area": None,
        "distance": None,
        "source_file": "x.csv",
        "source_row": 0,
        "spatial_tile_x": 0,
        "spatial_tile_y": 0,
    }
    row.update(overrides)
    return row


@pytest.fixture
def spots_dir(tmp_path):
    rows = [
        _make_spot_row(spot_id=0, gene_id=1, gene_name="GENEA", world_x_um=5, world_y_um=5, spatial_tile_x=0, spatial_tile_y=0),
        _make_spot_row(spot_id=1, gene_id=2, gene_name="GENEB", world_x_um=5, world_y_um=5, spatial_tile_x=0, spatial_tile_y=0),
        _make_spot_row(
            spot_id=2, gene_id=1, gene_name="GENEA", is_blank=False, world_x_um=1500, world_y_um=5,
            spatial_tile_x=1, spatial_tile_y=0,
        ),
        _make_spot_row(
            spot_id=3, gene_id=3, gene_name="Blank-1 [CB0]", is_blank=True, world_x_um=5, world_y_um=5,
            spatial_tile_x=0, spatial_tile_y=0,
        ),
    ]
    df = pd.DataFrame.from_records(rows)[[f.name for f in SPOT_SCHEMA]]
    out_dir = tmp_path / "spots"
    write_partitioned_chunks([df], schema=SPOT_SCHEMA, out_dir=out_dir, partition_cols=["spatial_tile_x", "spatial_tile_y"])
    return out_dir


def test_query_spots_filters_by_viewport(spots_dir):
    dataset = open_dataset(spots_dir)
    table = query_spots(dataset, xmin_um=0, ymin_um=0, xmax_um=10, ymax_um=10, tile_size_um=1000)
    assert set(table.column("spot_id").to_pylist()) == {0, 1}


def test_query_spots_excludes_blanks_by_default(spots_dir):
    dataset = open_dataset(spots_dir)
    table = query_spots(dataset, xmin_um=0, ymin_um=0, xmax_um=10, ymax_um=10, tile_size_um=1000)
    assert 3 not in table.column("spot_id").to_pylist()


def test_query_spots_includes_blanks_when_requested(spots_dir):
    dataset = open_dataset(spots_dir)
    table = query_spots(dataset, xmin_um=0, ymin_um=0, xmax_um=10, ymax_um=10, include_blanks=True, tile_size_um=1000)
    assert 3 in table.column("spot_id").to_pylist()


def test_query_spots_filters_by_gene_id(spots_dir):
    dataset = open_dataset(spots_dir)
    table = query_spots(dataset, xmin_um=0, ymin_um=0, xmax_um=2000, ymax_um=10, gene_ids=[1], tile_size_um=1000)
    assert set(table.column("spot_id").to_pylist()) == {0, 2}


def test_query_spots_prunes_distant_tile(spots_dir):
    dataset = open_dataset(spots_dir)
    table = query_spots(dataset, xmin_um=1400, ymin_um=0, xmax_um=1600, ymax_um=10, tile_size_um=1000)
    assert table.column("spot_id").to_pylist() == [2]


def test_apply_lod_no_sampling_below_max():
    table = pa.table({"spot_id": list(range(10))})
    sampled, info = apply_lod(table, max_points=100)
    assert not info["sampled"]
    assert sampled.num_rows == 10


def test_apply_lod_deterministic_and_bounded():
    table = pa.table({"spot_id": list(range(10_000))})
    sampled1, info1 = apply_lod(table, max_points=1000)
    sampled2, info2 = apply_lod(table, max_points=1000)
    assert info1["sampled"] and info1["total_in_view"] == 10_000
    assert sampled1.num_rows <= 1000
    assert sampled1.column("spot_id").to_pylist() == sampled2.column("spot_id").to_pylist()
