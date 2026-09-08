"""Orchestrates cache (re)building with atomic component swaps (spec 15.4)."""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa

from ..adapters.merlin.codebooks import load_all_codebooks
from ..config import AppConfig
from ..model.dataset import DatasetDescriptor
from ..storage.cache import CacheManager
from ..storage.dataset_json import save_dataset_json
from ..storage.parquet_store import write_table
from ..storage.zarr_store import atomic_replace_dir
from . import gene_index as gene_index_mod
from . import image_mosaic
from . import pyramid as pyramid_mod
from . import spot_index as spot_index_mod
from .manifest import (
    InvalidationDecision,
    Manifest,
    build_manifest,
    compute_source_fingerprint,
    decide_invalidation,
    load_manifest,
    save_manifest,
)

logger = logging.getLogger(__name__)

FOV_SCHEMA = pa.schema(
    [
        ("fov_id", pa.int64()),
        ("position_x_um", pa.float64()),
        ("position_y_um", pa.float64()),
        ("xmin_um", pa.float64()),
        ("ymin_um", pa.float64()),
        ("xmax_um", pa.float64()),
        ("ymax_um", pa.float64()),
        ("z_count", pa.int32()),
        ("has_images", pa.bool_()),
    ]
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _write_fov_table(dataset: DatasetDescriptor, path: Path) -> None:
    from ..model.transforms import fov_bounding_box_um

    rows = []
    for fov in dataset.fovs:
        if fov.image_shape_zyx is not None:
            z, h, w = fov.image_shape_zyx
            xmin, ymin, xmax, ymax = fov_bounding_box_um(
                fov_origin_x_um=fov.position_x_um,
                fov_origin_y_um=fov.position_y_um,
                image_height=h,
                image_width=w,
                microscope=dataset.microscope,
                apply_orientation=dataset.image_orientation_apply,
            )
        else:
            z = 0
            xmin = ymin = xmax = ymax = float("nan")
        rows.append(
            {
                "fov_id": fov.fov_id,
                "position_x_um": fov.position_x_um,
                "position_y_um": fov.position_y_um,
                "xmin_um": xmin,
                "ymin_um": ymin,
                "xmax_um": xmax,
                "ymax_um": ymax,
                "z_count": z,
                "has_images": bool(fov.image_paths),
            }
        )
    df = pd.DataFrame.from_records(rows)
    table = pa.Table.from_pandas(df, schema=FOV_SCHEMA, preserve_index=False)
    write_table(table, path)


def index_dataset(
    dataset: DatasetDescriptor,
    *,
    cache: CacheManager,
    config: AppConfig,
    force: bool = False,
    build_images: bool = True,
    build_transcripts: bool = True,
    build_pyramid_levels: bool = True,
    progress_callback: Callable[[str, str], None] | None = None,
) -> Manifest:
    """Build or refresh the on-disk cache. Returns the resulting manifest.

    A crash or exception here never leaves a cache falsely marked valid:
    every rebuilt component is assembled under a temp directory and only
    swapped into place after it succeeds; ``manifest.json`` is written last.
    """

    def report(stage: str, message: str) -> None:
        logger.info("[%s] %s", stage, message)
        if progress_callback:
            progress_callback(stage, message)

    old_manifest = None if force else load_manifest(cache.cache_dir)
    current_fp = compute_source_fingerprint(dataset)
    decision = (
        InvalidationDecision(rebuild_images=True, rebuild_spots=True, rebuild_genes=True, reasons=["--force"])
        if force
        else decide_invalidation(old_manifest, current_fp)
    )

    report("plan", f"reasons={decision.reasons} rebuild_images={decision.rebuild_images} "
                   f"rebuild_spots={decision.rebuild_spots} rebuild_genes={decision.rebuild_genes}")

    if not decision.rebuild_anything and old_manifest is not None:
        report("cache", "cache is valid; nothing to rebuild")
        return old_manifest

    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    if cache.tmp_dir.exists():
        shutil.rmtree(cache.tmp_dir)
    cache.tmp_dir.mkdir(parents=True)

    components_built = dict(old_manifest.components_built) if old_manifest else {"images": False, "spots": False, "genes": False}

    parsed_codebooks = None
    if decision.rebuild_spots or decision.rebuild_genes:
        parsed = load_all_codebooks(dataset.root_path)
        parsed_codebooks = {p.codebook_id: p for p in parsed}

    gene_counts = None

    if build_transcripts and decision.rebuild_spots:
        assert parsed_codebooks is not None  # guaranteed by the `decision.rebuild_spots` branch above
        report("spots", "normalizing barcode exports into partitioned Parquet")
        tmp_spots = cache.tmp_dir / "spots"

        def spot_progress(n: int) -> None:
            report("spots", f"{n} spots written")

        gene_counts = spot_index_mod.build_spot_index(
            dataset,
            parsed_codebooks,
            out_dir=tmp_spots,
            tile_size_um=config.spots.spatial_tile_size_um,
            progress_callback=spot_progress,
        )
        atomic_replace_dir(tmp_spots, cache.spots_dir)
        components_built["spots"] = True
        report("spots", f"done: {len(gene_counts)} distinct genes observed")

    if build_transcripts and decision.rebuild_genes:
        assert parsed_codebooks is not None  # guaranteed by the `decision.rebuild_genes` branch above
        if gene_counts is None:
            gene_counts = gene_index_mod.compute_gene_counts_from_spots_dataset(cache.spots_dir)
        gene_df = gene_index_mod.build_gene_table(parsed_codebooks, gene_counts)
        tmp_genes = cache.tmp_dir / "genes.parquet"
        gene_index_mod.write_gene_index(gene_df, tmp_genes)
        tmp_genes.replace(cache.genes_path)
        components_built["genes"] = True
        report("genes", f"done: {len(gene_df)} genes")

    if build_images and decision.rebuild_images:
        tmp_images = cache.tmp_dir / "images.zarr"
        mosaic_metadata: dict[str, dict] = {}
        for channel in dataset.channels:
            report("images", f"building mosaic for channel {channel.channel_id!r}")
            level0_path = tmp_images / channel.channel_id / "0"

            def image_progress(done: int, total: int, _channel=channel.channel_id) -> None:
                if done % 50 == 0 or done == total:
                    report("images", f"{_channel}: {done}/{total} FOVs placed")

            geometry, touched_chunks = image_mosaic.build_channel_mosaic(
                dataset,
                channel.channel_id,
                level0_out_path=level0_path,
                tmp_dir=cache.tmp_dir / "mosaic_build",
                chunk_size=config.images.chunk_size,
                overlap_mode=config.images.overlap_mode,
                fov_crop_px=config.images.fov_crop_px,
                progress_callback=image_progress,
            )
            if build_pyramid_levels:
                report("images", f"building pyramid for channel {channel.channel_id!r}")
                pyramid_mod.build_pyramid(
                    level0_path,
                    tmp_images / channel.channel_id,
                    chunk_size=config.images.chunk_size,
                    active_chunks=touched_chunks,
                )
            mosaic_metadata[channel.channel_id] = {
                "origin_world_um": [geometry.origin_x_um, geometry.origin_y_um],
                "pixel_size_um": [geometry.pixel_size_um, geometry.pixel_size_um],
                "shape_yx": [geometry.height_px, geometry.width_px],
                "z_count": geometry.z_count,
            }
        (tmp_images / "mosaic_metadata.json").write_text(json.dumps(mosaic_metadata, indent=2))
        atomic_replace_dir(tmp_images, cache.images_dir)
        components_built["images"] = True
        report("images", "done")

    _write_fov_table(dataset, cache.fovs_path)
    save_dataset_json(dataset, cache.dataset_json_path)

    now = _now_iso()
    manifest = build_manifest(
        dataset,
        indexing_config=config.model_dump(),
        now_iso=now,
        components_built=components_built,
    )
    if old_manifest is not None:
        manifest.created_at = old_manifest.created_at
    save_manifest(cache.cache_dir, manifest)

    shutil.rmtree(cache.tmp_dir, ignore_errors=True)
    report("cache", "manifest written; cache is now valid")
    return manifest
