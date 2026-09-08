"""Stream-normalize barcode exports into a partitioned transcript Parquet cache."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from ..adapters.merlin import barcodes as barcodes_mod
from ..adapters.merlin.codebooks import ParsedCodebook
from ..model.dataset import DatasetDescriptor
from ..model.genes import deterministic_gene_id
from ..model.spots import DEFAULT_SPATIAL_TILE_SIZE_UM, SPOT_SCHEMA, spatial_tile_index
from ..storage.parquet_store import write_partitioned_chunks

logger = logging.getLogger(__name__)


def _finalize_chunk(chunk: pd.DataFrame, *, spot_id_start: int, tile_size_um: float) -> pd.DataFrame:
    n = len(chunk)
    chunk = chunk.copy()
    chunk["spot_id"] = range(spot_id_start, spot_id_start + n)
    chunk["gene_id"] = chunk["gene_name"].map(lambda g: deterministic_gene_id(g) if g is not None else -1)
    tile_xy = [spatial_tile_index(x, y, tile_size_um) for x, y in zip(chunk["world_x_um"], chunk["world_y_um"], strict=True)]
    chunk["spatial_tile_x"] = [t[0] for t in tile_xy]
    chunk["spatial_tile_y"] = [t[1] for t in tile_xy]
    return chunk[[f.name for f in SPOT_SCHEMA]]


def build_spot_index(
    dataset: DatasetDescriptor,
    parsed_codebooks: dict[str, ParsedCodebook],
    *,
    out_dir: Path,
    tile_size_um: float = DEFAULT_SPATIAL_TILE_SIZE_UM,
    chunk_size: int = 500_000,
    progress_callback: Callable[[int], None] | None = None,
) -> Counter:
    """Normalize all barcode exports into a hive-partitioned Parquet cache.

    Returns a ``Counter`` of ``gene_name -> spot_count`` (including blanks)
    accumulated across all exports, for use by :mod:`gene_index`.
    """
    exports = barcodes_mod.discover_barcode_exports(dataset.root_path)
    fov_positions = pd.DataFrame(
        {"x_um": [f.position_x_um for f in dataset.fovs], "y_um": [f.position_y_um for f in dataset.fovs]},
        index=[f.fov_id for f in dataset.fovs],
    )
    fovs_by_id = {f.fov_id: f for f in dataset.fovs}
    export_to_codebook_id = {be.export_id: be.codebook_id for be in dataset.barcode_exports}

    gene_counts: Counter[str] = Counter()
    counters = {"spot_id": 0, "total_rows": 0}

    def chunk_stream():
        for export in exports:
            codebook_id = export_to_codebook_id[export.export_id]
            codebook_table = parsed_codebooks[codebook_id].table
            for raw_chunk in barcodes_mod.iter_normalized_barcode_chunks(
                export,
                codebook_id=codebook_id,
                codebook_table=codebook_table,
                fov_positions=fov_positions,
                fovs_by_id=fovs_by_id,
                microscope=dataset.microscope,
                apply_orientation=dataset.image_orientation_apply,
                chunk_size=chunk_size,
            ):
                normalized = _finalize_chunk(raw_chunk, spot_id_start=counters["spot_id"], tile_size_um=tile_size_um)
                counters["spot_id"] += len(normalized)
                counters["total_rows"] += len(normalized)
                gene_counts.update(raw_chunk["gene_name"].dropna())
                if progress_callback:
                    progress_callback(counters["total_rows"])
                yield normalized

    write_partitioned_chunks(
        chunk_stream(), schema=SPOT_SCHEMA, out_dir=out_dir, partition_cols=["spatial_tile_x", "spatial_tile_y"]
    )
    logger.info("Indexed %d spots across %d barcode export(s) into %s", counters["total_rows"], len(exports), out_dir)
    return gene_counts
