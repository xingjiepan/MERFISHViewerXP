"""Viewport-bounded transcript queries with spatial partition pruning (spec 10.4)."""

from __future__ import annotations

from collections.abc import Iterable

import pyarrow as pa
import pyarrow.dataset as pa_dataset

from ..model.spots import DEFAULT_SPATIAL_TILE_SIZE_UM, tiles_overlapping_bounds


def query_spots(
    dataset: pa_dataset.Dataset,
    *,
    xmin_um: float,
    ymin_um: float,
    xmax_um: float,
    ymax_um: float,
    gene_ids: Iterable[int] | None = None,
    z_range_um: tuple[float, float] | None = None,
    include_blanks: bool = False,
    limit: int | None = None,
    tile_size_um: float = DEFAULT_SPATIAL_TILE_SIZE_UM,
) -> pa.Table:
    tiles = tiles_overlapping_bounds(xmin_um, ymin_um, xmax_um, ymax_um, tile_size_um)
    tile_x = [t[0] for t in tiles]
    tile_y = [t[1] for t in tiles]

    expr = (
        (pa_dataset.field("spatial_tile_x") >= min(tile_x))
        & (pa_dataset.field("spatial_tile_x") <= max(tile_x))
        & (pa_dataset.field("spatial_tile_y") >= min(tile_y))
        & (pa_dataset.field("spatial_tile_y") <= max(tile_y))
        & (pa_dataset.field("world_x_um") >= xmin_um)
        & (pa_dataset.field("world_x_um") <= xmax_um)
        & (pa_dataset.field("world_y_um") >= ymin_um)
        & (pa_dataset.field("world_y_um") <= ymax_um)
        & (pa_dataset.field("mapping_status") == "mapped")
    )
    if not include_blanks:
        expr = expr & ~pa_dataset.field("is_blank")
    if gene_ids is not None:
        expr = expr & pa_dataset.field("gene_id").isin(list(gene_ids))
    if z_range_um is not None:
        zmin, zmax = z_range_um
        expr = expr & (pa_dataset.field("world_z_um") >= zmin) & (pa_dataset.field("world_z_um") <= zmax)

    table = dataset.to_table(filter=expr)
    if limit is not None and table.num_rows > limit:
        table = table.slice(0, limit)
    return table
