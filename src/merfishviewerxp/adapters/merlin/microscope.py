"""Parse MERlin's ``microscope_parameters.json`` (spec section 5, 7)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ...errors import DatasetValidationError
from ...model.transforms import MicroscopeTransformParameters

logger = logging.getLogger(__name__)

_PIXEL_SIZE_ALIASES = ["microns_per_pixel", "pixel_size_um", "micron_per_pixel", "pixel_size"]
_FLIP_H_ALIASES = ["flip_horizontal", "flip_x", "fliplr"]
_FLIP_V_ALIASES = ["flip_vertical", "flip_y", "flipud"]
_TRANSPOSE_ALIASES = ["transpose", "swap_xy"]


def _find_key(payload: dict, aliases: list[str], *, source_file: str, required: bool = True):
    normalized = {k.strip().lower(): k for k in payload}
    for alias in aliases:
        key = normalized.get(alias.strip().lower())
        if key is not None:
            logger.info("Resolved microscope field %s -> key %r in %s", aliases[0], key, source_file)
            return payload[key]
    if required:
        raise DatasetValidationError(
            f"Could not find any of {aliases} in {source_file}.\n"
            f"Found keys: {list(payload.keys())}.\n"
            "Provide a dataset config override to specify the mapping."
        )
    return None


def load_microscope_parameters(path: Path, *, z_step_um: float | None = None) -> MicroscopeTransformParameters:
    if not path.is_file():
        raise DatasetValidationError(
            f"Expected microscope parameters file at {path} but it does not exist.\n"
            "MERFISHViewerXP requires microscope_parameters.json to resolve pixel size "
            "and orientation flags."
        )
    payload = json.loads(path.read_text())

    pixel_size = _find_key(payload, _PIXEL_SIZE_ALIASES, source_file=str(path))
    flip_h = _find_key(payload, _FLIP_H_ALIASES, source_file=str(path), required=False)
    flip_v = _find_key(payload, _FLIP_V_ALIASES, source_file=str(path), required=False)
    transpose = _find_key(payload, _TRANSPOSE_ALIASES, source_file=str(path), required=False)

    params = MicroscopeTransformParameters(
        flip_horizontal=bool(flip_h) if flip_h is not None else False,
        flip_vertical=bool(flip_v) if flip_v is not None else False,
        transpose=bool(transpose) if transpose is not None else False,
        pixel_size_um=float(pixel_size),
        z_step_um=z_step_um,
    )
    logger.info(
        "Loaded microscope parameters from %s: pixel_size_um=%.6f flip_h=%s flip_v=%s transpose=%s",
        path,
        params.pixel_size_um,
        params.flip_horizontal,
        params.flip_vertical,
        params.transpose,
    )
    return params
