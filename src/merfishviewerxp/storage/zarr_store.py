"""Zarr array helpers for the multiscale image mosaic cache (spec section 8)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy.typing as npt
import zarr


def create_array(
    store_path: Path,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: npt.DTypeLike,
    fill_value: float = 0,
) -> zarr.Array:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    return zarr.open_array(
        str(store_path),
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        fill_value=fill_value,
    )


def open_array_readonly(store_path: Path) -> zarr.Array:
    return zarr.open_array(str(store_path), mode="r")


def atomic_replace_dir(tmp_path: Path, final_path: Path) -> None:
    """Atomically move a completed zarr array/group directory into place."""
    if final_path.exists():
        backup = final_path.with_name(final_path.name + ".old")
        if backup.exists():
            shutil.rmtree(backup)
        final_path.rename(backup)
        tmp_path.rename(final_path)
        shutil.rmtree(backup)
    else:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.rename(final_path)
