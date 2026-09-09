"""Background viewport queries with stale-result cancellation (spec 14.2, 16)."""

from __future__ import annotations

from collections.abc import Callable

from napari.qt.threading import create_worker

from ..indexed_dataset import IndexedDataset


class ViewportQueryRunner:
    """Runs `IndexedDataset.query_spots` off the GUI thread.

    Each call bumps a generation counter; a result is only delivered if it
    is still the most recently requested one, so a fast pan/zoom sequence
    never overwrites a newer viewport with a stale, slower one.
    """

    def __init__(
        self,
        indexed: IndexedDataset,
        *,
        on_result: Callable[[object, dict], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.indexed = indexed
        self.on_result = on_result
        self.on_error = on_error
        self._generation = 0
        self._worker = None

    def request(
        self,
        *,
        bounds_um: tuple[float, float, float, float],
        gene_ids,
        include_blanks: bool,
        max_visible_points: int,
        z_range_um: tuple[float, float] | None = None,
        apply_lod_sampling: bool = True,
    ) -> None:
        self._generation += 1
        generation = self._generation
        if self._worker is not None:
            self._worker.quit()

        def _do_query():
            table, lod_info = self.indexed.query_spots(
                bounds_um=bounds_um,
                gene_ids=gene_ids,
                include_blanks=include_blanks,
                max_visible_points=max_visible_points,
                z_range_um=z_range_um,
                apply_lod_sampling=apply_lod_sampling,
            )
            return table, lod_info, generation

        worker = create_worker(_do_query)
        worker.returned.connect(self._handle_result)
        if self.on_error is not None:
            worker.errored.connect(self.on_error)
        worker.start()
        self._worker = worker

    def _handle_result(self, result) -> None:
        table, lod_info, generation = result
        if generation != self._generation:
            return
        self.on_result(table, lod_info)
