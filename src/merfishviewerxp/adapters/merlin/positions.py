"""Parse MERlin's ``positions.csv`` (global FOV stage positions).

MERlin's canonical ``positions.csv`` has no header and two columns
(x, y) in stage micrometers, with the FOV id given implicitly by the
(zero-based) row order. Some pipelines instead write a header with named
columns. Both forms are supported; the resolved convention is always
logged so it is never silently guessed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ...errors import DatasetValidationError
from .aliases import POSITIONS_ALIASES, resolve_column

logger = logging.getLogger(__name__)


def _looks_like_header(first_row: list[str]) -> bool:
    for value in first_row:
        try:
            float(value)
        except ValueError:
            return True
    return False


def load_positions(path: Path) -> pd.DataFrame:
    """Return a DataFrame indexed by ``fov_id`` with columns ``x_um``, ``y_um``."""
    if not path.is_file():
        raise DatasetValidationError(
            f"Expected FOV positions file at {path} but it does not exist.\n"
            "MERFISHViewerXP requires positions.csv to place FOVs in world coordinates."
        )

    with path.open() as fh:
        first_line = fh.readline().strip()
    first_row = [c.strip() for c in first_line.split(",")]

    if _looks_like_header(first_row):
        df = pd.read_csv(path)
        x_col = resolve_column(
            columns=list(df.columns), canonical_field="x", aliases=POSITIONS_ALIASES["x"], source_file=str(path)
        )
        y_col = resolve_column(
            columns=list(df.columns), canonical_field="y", aliases=POSITIONS_ALIASES["y"], source_file=str(path)
        )
        result = pd.DataFrame({"x_um": df[x_col].astype(float), "y_um": df[y_col].astype(float)})
        result.index.name = "fov_id"
        logger.info("Loaded %d FOV positions from %s (headered: x=%r, y=%r)", len(result), path, x_col, y_col)
        return result

    if len(first_row) != 2:
        raise DatasetValidationError(
            f"positions.csv at {path} has no header and {len(first_row)} columns; expected exactly 2 (x, y).\n"
            "MERFISHViewerXP cannot infer FOV position columns for a headerless file with a "
            "different column count. Provide a dataset config override."
        )

    df = pd.read_csv(path, header=None, names=["x_um", "y_um"])
    df.index.name = "fov_id"
    logger.info(
        "Loaded %d FOV positions from %s (headerless: assumed column order x,y; fov_id = row order)",
        len(df),
        path,
    )
    return df
