"""Canonical column names and documented alias maps (spec section 6.2).

MERlin versions or laboratory pipelines may use slightly different CSV
column names. Each canonical field below lists every accepted alias
(case-insensitive, whitespace-stripped). Resolution never silently guesses
between two equally plausible columns -- see :func:`resolve_column`.
"""

from __future__ import annotations

import logging

from ...errors import ColumnResolutionError

logger = logging.getLogger(__name__)

POSITIONS_ALIASES: dict[str, list[str]] = {
    "x": ["x", "xpos", "x_um", "stage_x", "position_x", "pos_x", "global_x"],
    "y": ["y", "ypos", "y_um", "stage_y", "position_y", "pos_y", "global_y"],
}

BARCODE_ALIASES: dict[str, list[str]] = {
    "barcode_id": ["barcode_id", "barcodeid", "barcode_index", "id"],
    "world_x": ["global_x", "global_x_um", "x_global"],
    "world_y": ["global_y", "global_y_um", "y_global"],
    "world_z": ["global_z", "global_z_um", "z_global"],
    "source_x": ["x", "local_x", "x_pixel", "xpix"],
    "source_y": ["y", "local_y", "y_pixel", "ypix"],
    "source_z": ["z", "local_z", "z_pixel", "zpix"],
    "fov_id": ["fov", "fov_id", "fovid", "field_of_view"],
    "cell_index": ["cell_index", "cellindex", "cell_id"],
    "mean_intensity": ["mean_intensity", "intensity", "meanintensity", "average_intensity"],
    "area": ["area", "pixel_area", "size"],
    "distance": ["distance", "mean_distance", "error", "decode_distance"],
}

CODEBOOK_NAME_ALIASES = ["name", "gene_name", "target", "genename"]


def resolve_column(
    *,
    columns: list[str],
    canonical_field: str,
    aliases: list[str],
    source_file: str,
    required: bool = True,
) -> str | None:
    """Find exactly one column in ``columns`` matching one of ``aliases``.

    Matching is case-insensitive and whitespace-insensitive. Raises
    :class:`ColumnResolutionError` if zero (and required) or more than one
    alias-distinct column matches -- ambiguity is never silently resolved.
    """
    normalized = {c.strip().lower(): c for c in columns}
    alias_set = {a.strip().lower() for a in aliases}
    matches = [normalized[key] for key in normalized if key in alias_set]

    if len(matches) == 1:
        logger.info("Resolved %s -> column %r in %s", canonical_field, matches[0], source_file)
        return matches[0]

    if not matches and not required:
        return None

    raise ColumnResolutionError(
        source_file=source_file,
        canonical_field=canonical_field,
        expected_aliases=aliases,
        found_columns=columns,
        matched_columns=matches if len(matches) > 1 else None,
    )
