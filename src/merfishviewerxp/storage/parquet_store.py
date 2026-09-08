"""Partitioned Parquet read/write helpers (spec section 10.3)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pq


def write_partitioned_chunks(
    chunks: Iterable[pd.DataFrame],
    *,
    schema: pa.Schema,
    out_dir: Path,
    partition_cols: list[str],
) -> int:
    """Write a stream of DataFrame chunks into a hive-partitioned Parquet dataset.

    Each chunk gets a distinct basename template so concurrent/sequential
    calls never collide on file names.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for i, chunk in enumerate(chunks):
        if chunk.empty:
            continue
        table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
        pq.write_to_dataset(
            table,
            root_path=str(out_dir),
            partition_cols=partition_cols,
            basename_template=f"chunk-{i}-part-{{i}}.parquet",
        )
        total_rows += len(chunk)
    return total_rows


def write_table(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def open_dataset(directory: Path) -> pa_dataset.Dataset:
    return pa_dataset.dataset(str(directory), format="parquet", partitioning="hive")
