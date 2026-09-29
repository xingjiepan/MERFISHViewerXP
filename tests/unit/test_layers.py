import pyarrow as pa

from merfishviewerxp.model.genes import gene_color_rgb, hex_to_rgb
from merfishviewerxp.viewer.layers import spot_table_to_points


def _spot_table(rows: list[dict]) -> pa.Table:
    columns = {
        "spot_id": [r["spot_id"] for r in rows],
        "codebook_id": [r.get("codebook_id", "CB0") for r in rows],
        "barcode_id": [r.get("barcode_id", 0) for r in rows],
        "gene_id": [r["gene_id"] for r in rows],
        "gene_name": [r["gene_name"] for r in rows],
        "fov_id": [r.get("fov_id", 0) for r in rows],
        "world_x_um": [r.get("world_x_um", 0.0) for r in rows],
        "world_y_um": [r.get("world_y_um", 0.0) for r in rows],
        "world_z_um": [r.get("world_z_um", 0.0) for r in rows],
    }
    return pa.table(columns)


def test_spot_table_to_points_uses_deterministic_color_by_default():
    table = _spot_table([{"spot_id": 1, "gene_id": 100, "gene_name": "GENEA"}])
    _, colors, _ = spot_table_to_points(table)
    assert tuple(colors[0]) == (*gene_color_rgb("GENEA"), 1.0)


def test_spot_table_to_points_applies_color_override_by_gene_id():
    table = _spot_table(
        [
            {"spot_id": 1, "gene_id": 100, "gene_name": "GENEA"},
            {"spot_id": 2, "gene_id": 200, "gene_name": "GENEB"},
        ]
    )
    _, colors, features = spot_table_to_points(table, color_overrides={100: "#ff0000"})

    assert tuple(colors[0]) == (*hex_to_rgb("#ff0000"), 1.0)
    # gene 200 has no override -- keeps its deterministic default
    assert tuple(colors[1]) == (*gene_color_rgb("GENEB"), 1.0)
    assert features["gene_id"] == [100, 200]


def test_spot_table_to_points_empty_table_still_reports_gene_id_feature():
    table = _spot_table([])
    coords, colors, features = spot_table_to_points(table)
    assert coords.shape == (0, 2)
    assert colors.shape == (0, 4)
    assert features["gene_id"] == []
