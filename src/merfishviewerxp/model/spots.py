"""Canonical normalized transcript schema (spec section 10.2)."""

from __future__ import annotations

import pyarrow as pa

DEFAULT_SPATIAL_TILE_SIZE_UM = 1000.0
MAX_VISIBLE_POINTS_DEFAULT = 250_000

SPOT_SCHEMA = pa.schema(
    [
        ("spot_id", pa.int64()),
        ("codebook_id", pa.string()),
        ("barcode_id", pa.int64()),
        ("gene_id", pa.int64()),
        ("gene_name", pa.string()),
        ("is_blank", pa.bool_()),
        ("mapping_status", pa.string()),
        ("fov_id", pa.int64()),
        ("world_x_um", pa.float64()),
        ("world_y_um", pa.float64()),
        ("world_z_um", pa.float64()),
        ("source_x", pa.float64()),
        ("source_y", pa.float64()),
        ("source_z", pa.float64()),
        ("mean_intensity", pa.float64()),
        ("area", pa.float64()),
        ("distance", pa.float64()),
        ("cell_index", pa.int64()),
        ("source_file", pa.string()),
        ("source_row", pa.int64()),
        ("spatial_tile_x", pa.int32()),
        ("spatial_tile_y", pa.int32()),
    ]
)

GENE_SCHEMA = pa.schema(
    [
        ("gene_id", pa.int64()),
        ("gene_name", pa.string()),
        ("is_blank", pa.bool_()),
        ("color_hex", pa.string()),
        ("codebook_ids", pa.string()),
        ("spot_count", pa.int64()),
    ]
)


def spatial_tile_index(x_um: float, y_um: float, tile_size_um: float = DEFAULT_SPATIAL_TILE_SIZE_UM) -> tuple[int, int]:
    """Deterministic integer tile id for a world coordinate (spec 10.3)."""
    import math

    return math.floor(x_um / tile_size_um), math.floor(y_um / tile_size_um)


def tiles_overlapping_bounds(
    xmin_um: float, ymin_um: float, xmax_um: float, ymax_um: float, tile_size_um: float = DEFAULT_SPATIAL_TILE_SIZE_UM
) -> list[tuple[int, int]]:
    """All ``(spatial_tile_x, spatial_tile_y)`` pairs overlapping a viewport."""
    x0, y0 = spatial_tile_index(xmin_um, ymin_um, tile_size_um)
    x1, y1 = spatial_tile_index(xmax_um, ymax_um, tile_size_um)
    return [(tx, ty) for tx in range(x0, x1 + 1) for ty in range(y0, y1 + 1)]
