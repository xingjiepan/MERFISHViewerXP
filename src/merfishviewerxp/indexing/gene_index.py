"""Build the normalized gene/codebook index (spec section 9.3, 12)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pa_dataset

from ..adapters.merlin.codebooks import ParsedCodebook
from ..model.genes import deterministic_gene_id, gene_color_hex
from ..model.spots import GENE_SCHEMA
from ..storage.parquet_store import open_dataset, write_table


def build_gene_table(parsed_codebooks: dict[str, ParsedCodebook], gene_counts: Counter[str]) -> pd.DataFrame:
    genes: dict[str, dict] = {}
    for codebook_id, parsed in parsed_codebooks.items():
        for row in parsed.table.itertuples(index=False):
            entry = genes.setdefault(row.gene_name, {"is_blank": row.is_blank, "codebook_ids": set()})
            entry["codebook_ids"].add(codebook_id)

    records = []
    for gene_name, info in genes.items():
        records.append(
            {
                "gene_id": deterministic_gene_id(gene_name),
                "gene_name": gene_name,
                "is_blank": bool(info["is_blank"]),
                "color_hex": gene_color_hex(gene_name),
                "codebook_ids": ",".join(sorted(info["codebook_ids"])),
                "spot_count": int(gene_counts.get(gene_name, 0)),
            }
        )
    df = pd.DataFrame.from_records(records).sort_values("gene_name").reset_index(drop=True)
    return df


def write_gene_index(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, schema=GENE_SCHEMA, preserve_index=False)
    write_table(table, path)


def compute_gene_counts_from_spots_dataset(spots_dir: Path) -> Counter[str]:
    """Fallback: recompute gene counts from an already-built spots cache.

    Used when only the gene index needs rebuilding (e.g. after an
    interrupted build) without re-normalizing the transcript table.
    """
    dataset = open_dataset(spots_dir)
    table = dataset.to_table(columns=["gene_name"], filter=pa_dataset.field("mapping_status") == "mapped")
    names = table.column("gene_name").to_pylist()
    return Counter(names)
