"""Serialize the parts of a DatasetDescriptor an IndexedDataset needs, without re-parsing MERlin files."""

from __future__ import annotations

import json
from pathlib import Path

from ..model.dataset import DatasetDescriptor
from ..model.fov import FOVDescriptor
from ..model.transforms import MicroscopeTransformParameters


def dataset_to_dict(dataset: DatasetDescriptor) -> dict:
    return {
        "dataset_id": dataset.dataset_id,
        "root_path": str(dataset.root_path),
        "microscope": {
            "flip_horizontal": dataset.microscope.flip_horizontal,
            "flip_vertical": dataset.microscope.flip_vertical,
            "transpose": dataset.microscope.transpose,
            "pixel_size_um": dataset.microscope.pixel_size_um,
            "z_step_um": dataset.microscope.z_step_um,
        },
        "image_orientation_apply": dataset.image_orientation_apply,
        "image_orientation_residual_um": dataset.image_orientation_residual_um,
        "channels": [c.channel_id for c in dataset.channels],
        "codebooks": [
            {"codebook_id": cb.codebook_id, "codebook_index": cb.codebook_index, "n_barcodes": cb.n_barcodes}
            for cb in dataset.codebooks
        ],
        "fovs": [
            {
                "fov_id": f.fov_id,
                "position_x_um": f.position_x_um,
                "position_y_um": f.position_y_um,
                "image_shape_zyx": list(f.image_shape_zyx) if f.image_shape_zyx else None,
                "image_dtype": f.image_dtype,
                "channels": sorted(f.image_paths.keys()),
            }
            for f in dataset.fovs
        ],
    }


def save_dataset_json(dataset: DatasetDescriptor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(dataset_to_dict(dataset), indent=2))
    tmp_path.replace(path)


def load_dataset_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def microscope_from_summary(summary: dict) -> MicroscopeTransformParameters:
    m = summary["microscope"]
    return MicroscopeTransformParameters(
        flip_horizontal=m["flip_horizontal"],
        flip_vertical=m["flip_vertical"],
        transpose=m["transpose"],
        pixel_size_um=m["pixel_size_um"],
        z_step_um=m.get("z_step_um"),
    )


def fovs_from_summary(summary: dict) -> list[FOVDescriptor]:
    fovs = []
    for f in summary["fovs"]:
        shape = tuple(f["image_shape_zyx"]) if f["image_shape_zyx"] else None
        fovs.append(
            FOVDescriptor(
                fov_id=f["fov_id"],
                position_x_um=f["position_x_um"],
                position_y_um=f["position_y_um"],
                image_shape_zyx=shape,
                image_dtype=f["image_dtype"],
            )
        )
    return fovs
