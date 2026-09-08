"""Discover per-FOV image stacks and resolve the image orientation transform.

Filenames under ``CellPoseSegment/images`` vary across MERlin runs, so
discovery is pattern-based and configurable (spec section 5). Exact
filenames observed in practice: ``raw_nuclear_images<fov>.tif``,
``raw_membrane_images<fov>.tif`` (plus ``segmented_mask<fov>.tif``, which is
a cell-segmentation product out of MVP scope, not a stain channel).

Whether the ``microscope_parameters.json`` flip/transpose flags must be
reapplied to these already-exported per-FOV arrays is resolved empirically
in :func:`resolve_image_orientation` by cross-checking against MERlin's own
decoded barcode global coordinates -- never assumed (spec section 7.3).
"""

from __future__ import annotations

import logging
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tifffile

from ...errors import AmbiguousDiscoveryError, DatasetValidationError
from ...model.fov import FOVDescriptor
from ...model.transforms import MicroscopeTransformParameters, local_pixel_to_world_um

logger = logging.getLogger(__name__)

DEFAULT_CHANNEL_PATTERNS: dict[str, str] = {
    "nucleus": r"^raw_nuclear_images(?P<fov>\d+)\.tiff?$",
    "membrane": r"^raw_membrane_images(?P<fov>\d+)\.tiff?$",
}


def discover_fov_images(
    images_dir: Path, channel_patterns: dict[str, str] | None = None
) -> dict[int, dict[str, Path]]:
    """Return ``{fov_id: {channel_id: path}}`` discovered under ``images_dir``."""
    if not images_dir.is_dir():
        raise DatasetValidationError(
            f"Expected FOV image directory at {images_dir} but it does not exist.\n"
            "MERFISHViewerXP requires nucleus/membrane image stacks under CellPoseSegment/images."
        )

    patterns = {ch: re.compile(pat, re.IGNORECASE) for ch, pat in (channel_patterns or DEFAULT_CHANNEL_PATTERNS).items()}

    result: dict[int, dict[str, Path]] = {}
    ambiguous: list[str] = []
    for entry in sorted(images_dir.iterdir()):
        if not entry.is_file():
            continue
        hits = []
        for channel_id, pattern in patterns.items():
            match = pattern.match(entry.name)
            if match:
                hits.append((channel_id, int(match.group("fov"))))
        if len(hits) > 1:
            ambiguous.append(entry.name)
            continue
        if len(hits) == 1:
            channel_id, fov_id = hits[0]
            result.setdefault(fov_id, {})[channel_id] = entry

    if ambiguous:
        raise AmbiguousDiscoveryError(
            directory=str(images_dir),
            ambiguous_files=ambiguous,
            suggestion="These files matched more than one channel pattern. Provide explicit "
            "nucleus_pattern/membrane_pattern config overrides to disambiguate.",
        )

    if not result:
        raise DatasetValidationError(
            f"No FOV images matched known filename patterns under {images_dir}.\n"
            f"Tried patterns: {patterns}.\n"
            "Provide --nucleus-pattern/--membrane-pattern config overrides for this MERlin variant."
        )

    logger.info("Discovered images for %d FOVs under %s", len(result), images_dir)
    return result


def read_image_shape_dtype(path: Path) -> tuple[tuple[int, int, int], str]:
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        shape = tuple(series.shape)
        dtype = str(series.dtype)
    if len(shape) == 2:
        shape = (1, shape[0], shape[1])
    elif len(shape) != 3:
        raise DatasetValidationError(
            f"Unexpected image array shape {shape} for {path}; expected (z, y, x) or (y, x).\n"
            "MERFISHViewerXP assumes a single-series z-stack per channel per FOV."
        )
    return shape, dtype


@dataclass
class ResolvedOrientation:
    apply_orientation: bool
    residual_identity_um: float
    residual_with_flags_um: float

    @property
    def residual_um(self) -> float:
        return self.residual_with_flags_um if self.apply_orientation else self.residual_identity_um


def resolve_image_orientation(
    *,
    barcode_sample: pd.DataFrame,
    fov_positions: pd.DataFrame,
    fovs_by_id: dict[int, FOVDescriptor],
    microscope: MicroscopeTransformParameters,
    max_plausible_residual_um: float | None = None,
) -> ResolvedOrientation:
    """Empirically resolve whether flip/transpose flags apply to stored FOV images.

    ``barcode_sample`` must have columns ``fov_id``, ``source_x``, ``source_y``,
    ``world_x_um``, ``world_y_um`` taken from MERlin's own decoded barcodes
    (which already carry authoritative global coordinates). We recompute the
    world position from the local pixel coordinate under both hypotheses
    (orientation flags applied vs. identity) and pick whichever matches
    MERlin's own numbers.
    """
    residuals: dict[bool, list[float]] = {True: [], False: []}
    for row in barcode_sample.itertuples(index=False):
        fov = fovs_by_id.get(int(row.fov_id))
        if fov is None or fov.image_shape_zyx is None:
            continue
        if row.source_x != row.source_x or row.source_y != row.source_y:  # NaN check
            continue
        _, height, width = fov.image_shape_zyx
        pos = fov_positions.loc[int(row.fov_id)]
        for apply_flag in (True, False):
            x_um, y_um, _ = local_pixel_to_world_um(
                row.source_y,
                row.source_x,
                fov_origin_x_um=float(pos["x_um"]),
                fov_origin_y_um=float(pos["y_um"]),
                image_height=height,
                image_width=width,
                microscope=microscope,
                apply_orientation=apply_flag,
            )
            residuals[apply_flag].append(math.hypot(x_um - row.world_x_um, y_um - row.world_y_um))

    if not residuals[True] and not residuals[False]:
        raise DatasetValidationError(
            "Could not resolve image orientation: no barcode sample rows had both a "
            "matching FOV image and local pixel coordinates. Provide --nucleus-pattern/"
            "--membrane-pattern overrides or check positions.csv/barcodes.csv alignment."
        )

    mean_with_flags = statistics.fmean(residuals[True]) if residuals[True] else math.inf
    mean_identity = statistics.fmean(residuals[False]) if residuals[False] else math.inf
    apply_orientation = mean_with_flags <= mean_identity
    chosen_residual = min(mean_with_flags, mean_identity)

    tolerance = max_plausible_residual_um if max_plausible_residual_um is not None else microscope.pixel_size_um * 5
    if chosen_residual > tolerance:
        raise DatasetValidationError(
            "Image orientation could not be validated against decoded barcode coordinates: "
            f"best mean residual {chosen_residual:.3f} um exceeds tolerance {tolerance:.3f} um "
            f"(with flip/transpose flags applied: {mean_with_flags:.3f} um; identity: "
            f"{mean_identity:.3f} um).\n"
            "This usually means positions.csv row order does not match fov ids, the pixel "
            "size is wrong, or the local x/y column mapping is swapped. Inspect with "
            "`merfishviewerxp diagnose` and consider a dataset config override."
        )

    logger.info(
        "Resolved image orientation: apply_orientation=%s (residual %.4f um; "
        "with_flags=%.4f um, identity=%.4f um)",
        apply_orientation,
        chosen_residual,
        mean_with_flags,
        mean_identity,
    )
    return ResolvedOrientation(
        apply_orientation=apply_orientation,
        residual_identity_um=mean_identity,
        residual_with_flags_um=mean_with_flags,
    )
