"""Discover and stream ``*Barcodes_CB*/barcodes.csv`` exports (spec section 10).

The MERFISHViewerXP spec's dataset contract names the folder pattern
``ExprtBarcodes_CB*``; real MERlin output instead names it
``ExportBarcodes_CB*``. Discovery below accepts any ``*Barcodes_CB<n>``
directory that directly contains ``barcodes.csv``, which matches both
spellings, and logs exactly which directory was resolved for each codebook
index so this is never a silent guess.

When the source CSV already carries global (world) coordinates -- as MERlin
does today -- those are used directly as the most authoritative source of
truth, per the "never invent MERlin's transform semantics" rule. The
per-pixel transform in ``model.transforms`` is used only as a fallback when
a MERlin variant omits global coordinates.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ...errors import DatasetValidationError
from ...model.fov import FOVDescriptor
from ...model.transforms import MicroscopeTransformParameters, local_pixel_to_world_um
from .aliases import BARCODE_ALIASES, resolve_column

logger = logging.getLogger(__name__)

_EXPORT_DIR_RE = re.compile(r".*Barcodes_CB(?P<index>\d+)$", re.IGNORECASE)


@dataclass
class BarcodeExportInfo:
    export_id: str
    codebook_index: int
    source_file: Path


def discover_barcode_exports(root: Path) -> list[BarcodeExportInfo]:
    by_index: dict[int, list[tuple[str, Path]]] = {}
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        match = _EXPORT_DIR_RE.match(entry.name)
        if not match:
            continue
        csv_path = entry / "barcodes.csv"
        if csv_path.is_file():
            by_index.setdefault(int(match.group("index")), []).append((entry.name, csv_path))

    if not by_index:
        raise DatasetValidationError(
            f"No '*Barcodes_CB<n>/barcodes.csv' export folders were found under {root}.\n"
            "MERFISHViewerXP requires at least one decoded-barcode export."
        )

    exports: list[BarcodeExportInfo] = []
    for index, entries in sorted(by_index.items()):
        if len(entries) > 1:
            preferred = [e for e in entries if e[0].lower().startswith("export")]
            if len(preferred) != 1:
                raise DatasetValidationError(
                    f"Ambiguous barcode export folders for codebook index {index}: "
                    f"{[e[0] for e in entries]}.\n"
                    "Expected exactly one 'Export*Barcodes_CB<n>' folder. Remove or rename "
                    "the extra folder, or provide a dataset config override."
                )
            chosen = preferred[0]
            logger.warning(
                "Multiple barcode export folders for CB%d found (%s); using %r",
                index,
                [e[0] for e in entries],
                chosen[0],
            )
        else:
            chosen = entries[0]
        logger.info("Resolved barcode export for codebook index %d -> %s", index, chosen[1])
        exports.append(BarcodeExportInfo(export_id=f"CB{index}", codebook_index=index, source_file=chosen[1]))
    return exports


def _resolve_barcode_columns(columns: list[str], source_file: str) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {}
    resolved["barcode_id"] = resolve_column(
        columns=columns, canonical_field="barcode_id", aliases=BARCODE_ALIASES["barcode_id"], source_file=source_file
    )
    resolved["fov_id"] = resolve_column(
        columns=columns, canonical_field="fov_id", aliases=BARCODE_ALIASES["fov_id"], source_file=source_file
    )
    optional_fields = (
        "world_x", "world_y", "world_z", "source_x", "source_y", "source_z", "cell_index", "mean_intensity", "area", "distance",
    )
    for field in optional_fields:
        resolved[field] = resolve_column(
            columns=columns, canonical_field=field, aliases=BARCODE_ALIASES[field], source_file=source_file, required=False
        )
    return resolved


def load_barcode_sample(export: BarcodeExportInfo, *, n: int = 2000) -> pd.DataFrame | None:
    """Read a small sample with world+local coordinates, for orientation resolution.

    Returns ``None`` if this export has no global/world coordinate columns.
    """
    header_cols = list(pd.read_csv(export.source_file, nrows=0).columns)
    cols = _resolve_barcode_columns(header_cols, str(export.source_file))
    if cols["world_x"] is None or cols["world_y"] is None:
        return None

    usecols = [c for c in (cols["fov_id"], cols["source_x"], cols["source_y"], cols["world_x"], cols["world_y"]) if c]
    df = pd.read_csv(export.source_file, nrows=n, usecols=usecols)
    return pd.DataFrame(
        {
            "fov_id": df[cols["fov_id"]].astype("int64"),
            "source_x": df[cols["source_x"]].astype("float64") if cols["source_x"] else np.nan,
            "source_y": df[cols["source_y"]].astype("float64") if cols["source_y"] else np.nan,
            "world_x_um": df[cols["world_x"]].astype("float64"),
            "world_y_um": df[cols["world_y"]].astype("float64"),
        }
    )


def iter_normalized_barcode_chunks(
    export: BarcodeExportInfo,
    *,
    codebook_id: str,
    codebook_table: pd.DataFrame,
    fov_positions: pd.DataFrame,
    fovs_by_id: dict[int, FOVDescriptor],
    microscope: MicroscopeTransformParameters,
    apply_orientation: bool,
    chunk_size: int = 500_000,
) -> Iterator[pd.DataFrame]:
    """Yield normalized transcript chunks (spec 10.2 columns, minus spatial tile)."""
    header_cols = list(pd.read_csv(export.source_file, nrows=0).columns)
    cols = _resolve_barcode_columns(header_cols, str(export.source_file))
    has_world = cols["world_x"] is not None and cols["world_y"] is not None
    if not has_world:
        logger.warning(
            "%s has no global/world coordinate columns; falling back to the "
            "local-pixel-to-world transform (unvalidated against MERlin ground truth "
            "for this export).",
            export.source_file,
        )

    gene_lookup = codebook_table.set_index("barcode_id")

    row_offset = 0
    for chunk in pd.read_csv(export.source_file, chunksize=chunk_size):
        n = len(chunk)
        barcode_ids = chunk[cols["barcode_id"]].astype("int64")
        joined = gene_lookup.reindex(barcode_ids.to_numpy())
        gene_name = joined["gene_name"].to_numpy()
        is_blank = joined["is_blank"].to_numpy()
        mapping_status = np.where(pd.isna(gene_name), "unmapped", "mapped")
        gene_name = np.where(pd.isna(gene_name), None, gene_name)
        is_blank = np.where(pd.isna(is_blank), False, is_blank).astype(bool)

        fov_ids = chunk[cols["fov_id"]].astype("int64")

        if has_world:
            world_x = chunk[cols["world_x"]].astype("float64").to_numpy()
            world_y = chunk[cols["world_y"]].astype("float64").to_numpy()
            world_z = (
                chunk[cols["world_z"]].astype("float64").to_numpy()
                if cols["world_z"] is not None
                else np.full(n, np.nan)
            )
        else:
            positions = fov_positions.reindex(fov_ids.to_numpy())
            src_x = chunk[cols["source_x"]].astype("float64").to_numpy() if cols["source_x"] else np.full(n, np.nan)
            src_y = chunk[cols["source_y"]].astype("float64").to_numpy() if cols["source_y"] else np.full(n, np.nan)
            world_x = np.empty(n)
            world_y = np.empty(n)
            world_z = np.full(n, np.nan)
            for i in range(n):
                fov = fovs_by_id.get(int(fov_ids.iat[i]))
                shape = fov.image_shape_zyx if fov else None
                h, w = (shape[1], shape[2]) if shape else (0, 0)
                x_um, y_um, _ = local_pixel_to_world_um(
                    src_y[i],
                    src_x[i],
                    fov_origin_x_um=positions["x_um"].iat[i],
                    fov_origin_y_um=positions["y_um"].iat[i],
                    image_height=h,
                    image_width=w,
                    microscope=microscope,
                    apply_orientation=apply_orientation,
                )
                world_x[i] = x_um
                world_y[i] = y_um

        source_x = chunk[cols["source_x"]].astype("float64").to_numpy() if cols["source_x"] else np.full(n, np.nan)
        source_y = chunk[cols["source_y"]].astype("float64").to_numpy() if cols["source_y"] else np.full(n, np.nan)
        source_z = chunk[cols["source_z"]].astype("float64").to_numpy() if cols["source_z"] else np.full(n, np.nan)
        cell_index = (
            chunk[cols["cell_index"]].astype("int64").to_numpy() if cols["cell_index"] else np.full(n, -1, dtype="int64")
        )
        mean_intensity = (
            chunk[cols["mean_intensity"]].astype("float64").to_numpy() if cols["mean_intensity"] else np.full(n, np.nan)
        )
        area = chunk[cols["area"]].astype("float64").to_numpy() if cols["area"] else np.full(n, np.nan)
        distance = chunk[cols["distance"]].astype("float64").to_numpy() if cols["distance"] else np.full(n, np.nan)

        out = pd.DataFrame(
            {
                "codebook_id": codebook_id,
                "barcode_id": barcode_ids.to_numpy(),
                "gene_name": gene_name,
                "is_blank": is_blank,
                "mapping_status": mapping_status,
                "fov_id": fov_ids.to_numpy(),
                "world_x_um": world_x,
                "world_y_um": world_y,
                "world_z_um": world_z,
                "source_x": source_x,
                "source_y": source_y,
                "source_z": source_z,
                "cell_index": cell_index,
                "mean_intensity": mean_intensity,
                "area": area,
                "distance": distance,
                "source_file": str(export.source_file),
                "source_row": np.arange(row_offset, row_offset + n, dtype="int64"),
            }
        )
        row_offset += n
        yield out
