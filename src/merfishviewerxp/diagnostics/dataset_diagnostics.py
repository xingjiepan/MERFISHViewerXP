"""Structured `merfishviewerxp diagnose` output (spec section 7.4)."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from ..adapters.merlin import barcodes as barcodes_mod
from ..model.dataset import DatasetDescriptor
from . import transform_diagnostics


def _tail_csv_rows(path: Path, n: int, *, chunk_size: int = 65536, max_chunks: int = 50) -> pd.DataFrame:
    """Read the last `n` data rows of a CSV without loading the whole file."""
    with path.open("rb") as fh:
        header = fh.readline()
        fh.seek(0, io.SEEK_END)
        file_size = fh.tell()
        block = b""
        pos = file_size
        for _ in range(max_chunks):
            if pos <= len(header):
                break
            read_size = min(chunk_size, pos - len(header))
            pos -= read_size
            fh.seek(pos)
            block = fh.read(read_size) + block
            if block.count(b"\n") > n:
                break
    lines = [line for line in block.split(b"\n") if line.strip()]
    tail_lines = lines[-n:]
    csv_bytes = header + b"\n".join(tail_lines) + b"\n"
    return pd.read_csv(io.BytesIO(csv_bytes))


def sample_transcripts(dataset: DatasetDescriptor, n: int = 5) -> dict[str, dict]:
    samples: dict[str, dict] = {}
    for export_desc in dataset.barcode_exports:
        export = barcodes_mod.BarcodeExportInfo(
            export_id=export_desc.export_id, codebook_index=-1, source_file=export_desc.source_file
        )
        head = pd.read_csv(export.source_file, nrows=n)
        try:
            tail = _tail_csv_rows(export.source_file, n)
        except Exception:  # pragma: no cover - diagnostics must never crash on a malformed tail
            tail = pd.DataFrame()
        samples[export_desc.export_id] = {
            "head": head.to_dict(orient="records"),
            "tail": tail.to_dict(orient="records"),
        }
    return samples


def diagnose_dataset(dataset: DatasetDescriptor) -> dict:
    boxes = transform_diagnostics.fov_bounding_boxes(dataset)
    overall = transform_diagnostics.overall_bounding_box(boxes)
    warnings = transform_diagnostics.transform_warnings(dataset, boxes)

    return {
        "dataset_id": dataset.dataset_id,
        "root_path": str(dataset.root_path),
        "microscope_parameters": {
            "flip_horizontal": dataset.microscope.flip_horizontal,
            "flip_vertical": dataset.microscope.flip_vertical,
            "transpose": dataset.microscope.transpose,
            "pixel_size_um": dataset.microscope.pixel_size_um,
        },
        "resolved_image_orientation_apply": dataset.image_orientation_apply,
        "resolved_image_orientation_residual_um": dataset.image_orientation_residual_um,
        "position_units_assumed": "micrometers",
        "n_fovs": len(dataset.fovs),
        "n_fovs_with_images": sum(1 for f in dataset.fovs if f.image_paths),
        "channels": [c.channel_id for c in dataset.channels],
        "codebooks": [
            {"codebook_id": cb.codebook_id, "n_barcodes": cb.n_barcodes, "source_file": str(cb.source_file)}
            for cb in dataset.codebooks
        ],
        "barcode_exports": [
            {"export_id": b.export_id, "codebook_id": b.codebook_id, "source_file": str(b.source_file)}
            for b in dataset.barcode_exports
        ],
        "fov_bounding_boxes_um": {str(k): v for k, v in boxes.items()},
        "overall_bounding_box_um": overall,
        "transcript_samples": sample_transcripts(dataset),
        "warnings": warnings,
    }
