"""Discover and parse ``codebook_*.csv`` files (spec section 9).

Do not assume there is only one codebook. Barcode ids are the codebook-local,
zero-based row order of the codebook table (excluding the header) -- this
matches MERlin's own barcode-id convention and is validated against real
decoded barcodes in tests. Barcode ids are never assumed globally unique
across codebooks.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ...errors import DatasetValidationError
from ...model.genes import is_blank_name
from .aliases import CODEBOOK_NAME_ALIASES, resolve_column

logger = logging.getLogger(__name__)

_CODEBOOK_FILENAME_RE = re.compile(r"^codebook_(?P<index>\d+)", re.IGNORECASE)


@dataclass
class ParsedCodebook:
    codebook_id: str
    codebook_index: int | None
    source_file: Path
    content_hash: str
    table: pd.DataFrame  # columns: barcode_id, gene_name, is_blank


def discover_codebook_files(root: Path) -> list[Path]:
    files = sorted(root.glob("codebook_*.csv"))
    if not files:
        raise DatasetValidationError(
            f"No files matching 'codebook_*.csv' were found under {root}.\n"
            "MERFISHViewerXP requires at least one codebook to map barcodes to genes."
        )
    return files


def _content_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _codebook_id_from_filename(path: Path) -> tuple[str, int | None]:
    match = _CODEBOOK_FILENAME_RE.match(path.stem)
    index = int(match.group("index")) if match else None
    codebook_id = f"CB{index}" if index is not None else path.stem
    return codebook_id, index


def parse_codebook(path: Path) -> ParsedCodebook:
    df = pd.read_csv(path)
    if df.empty:
        raise DatasetValidationError(f"Codebook {path} contains no rows.")

    name_col = resolve_column(
        columns=list(df.columns), canonical_field="name", aliases=CODEBOOK_NAME_ALIASES, source_file=str(path)
    )

    codebook_id, codebook_index = _codebook_id_from_filename(path)
    raw_names = df[name_col].astype(str)
    is_blank = raw_names.map(is_blank_name)

    # Blank/control names (e.g. "Blank-23") are only unique *within* one
    # codebook -- different codebooks routinely reuse the same blank names
    # for unrelated negative-control barcodes. Real gene names are assumed
    # globally meaningful across codebooks, but blanks are disambiguated by
    # codebook here so they are never silently merged into one shared
    # gene_id/color/count (spec 9.4, 27.2).
    gene_names = raw_names.where(~is_blank, raw_names + f" [{codebook_id}]")

    table = pd.DataFrame(
        {
            "barcode_id": range(len(df)),
            "gene_name": gene_names,
            "is_blank": is_blank,
        }
    )

    logger.info(
        "Parsed codebook %s from %s: %d barcodes (%d blank)",
        codebook_id,
        path,
        len(table),
        int(table["is_blank"].sum()),
    )

    return ParsedCodebook(
        codebook_id=codebook_id,
        codebook_index=codebook_index,
        source_file=path,
        content_hash=_content_hash(path),
        table=table,
    )


def load_all_codebooks(root: Path) -> list[ParsedCodebook]:
    return [parse_codebook(p) for p in discover_codebook_files(root)]
