"""Level-of-detail: bound the number of points ever handed to the GUI (spec 10.5, 11)."""

from __future__ import annotations

import numpy as np
import pyarrow as pa


def apply_lod(table: pa.Table, max_points: int) -> tuple[pa.Table, dict]:
    """Deterministically subsample `table` to at most `max_points` rows.

    Sampling keys off `spot_id`, which is assigned once during indexing and
    is stable across viewport changes, so the same molecules are shown for
    the same viewport on repeated queries (spec 12.1-style determinism).
    """
    n = table.num_rows
    if n <= max_points or n == 0:
        return table, {"sampled": False, "total_in_view": n, "shown": n}

    step = int(np.ceil(n / max_points))
    spot_ids = table.column("spot_id").to_numpy(zero_copy_only=False)
    keep_mask = (spot_ids % step) == 0
    sampled = table.filter(pa.array(keep_mask))
    return sampled, {"sampled": True, "total_in_view": n, "shown": sampled.num_rows}
