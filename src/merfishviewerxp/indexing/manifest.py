"""Cache manifest: fingerprinting and invalidation decisions (spec section 15)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .. import __version__
from ..model.dataset import DatasetDescriptor
from .segmentation import SEGMENTATION_FORMAT_VERSION

CACHE_SCHEMA_VERSION = 1

MANIFEST_FILENAME = "manifest.json"


class SourceFingerprint(BaseModel):
    positions: str
    microscope: str
    codebooks: dict[str, str]
    barcode_exports: dict[str, str]
    image_inventory_hash: str
    image_file_count: int
    # Defaults keep manifests written before segmentation support loadable;
    # "" is also what a dataset with no masks hashes to.
    mask_inventory_hash: str = ""
    mask_file_count: int = 0


class Manifest(BaseModel):
    cache_schema_version: int
    merfishviewerxp_version: str
    created_at: str
    updated_at: str
    source_root: str
    fingerprint: SourceFingerprint
    indexing_config: dict[str, Any]
    image_orientation_apply: bool
    components_built: dict[str, bool] = {"images": False, "spots": False, "genes": False}
    # Format of the cached cell boundaries (see indexing.segmentation). Manifests
    # written before this field existed held format-1 boundaries, if any.
    segmentation_format_version: int = 1


def _file_fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{int(stat.st_mtime_ns)}"


def compute_image_inventory_hash(dataset: DatasetDescriptor) -> tuple[str, int]:
    entries = []
    for fov in dataset.fovs:
        for channel_id, path in sorted(fov.image_paths.items()):
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            entries.append(f"{fov.fov_id}:{channel_id}:{path.name}:{stat.st_size}:{int(stat.st_mtime_ns)}")
    entries.sort()
    digest = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
    return digest, len(entries)


def compute_mask_inventory_hash(dataset: DatasetDescriptor) -> tuple[str, int]:
    entries = []
    for fov in dataset.fovs:
        if fov.mask_path is None:
            continue
        try:
            stat = fov.mask_path.stat()
        except FileNotFoundError:
            continue
        entries.append(f"{fov.fov_id}:{fov.mask_path.name}:{stat.st_size}:{int(stat.st_mtime_ns)}")
    if not entries:
        return "", 0
    entries.sort()
    return hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest(), len(entries)


def compute_source_fingerprint(dataset: DatasetDescriptor) -> SourceFingerprint:
    image_hash, image_count = compute_image_inventory_hash(dataset)
    mask_hash, mask_count = compute_mask_inventory_hash(dataset)
    return SourceFingerprint(
        positions=_file_fingerprint(dataset.positions_file),
        microscope=_file_fingerprint(dataset.microscope_file),
        codebooks={cb.codebook_id: cb.content_hash for cb in dataset.codebooks},
        barcode_exports={be.export_id: be.fingerprint for be in dataset.barcode_exports},
        image_inventory_hash=image_hash,
        image_file_count=image_count,
        mask_inventory_hash=mask_hash,
        mask_file_count=mask_count,
    )


def build_manifest(
    dataset: DatasetDescriptor,
    *,
    indexing_config: dict[str, Any],
    now_iso: str,
    components_built: dict[str, bool],
    segmentation_format_version: int = SEGMENTATION_FORMAT_VERSION,
) -> Manifest:
    return Manifest(
        segmentation_format_version=segmentation_format_version,
        cache_schema_version=CACHE_SCHEMA_VERSION,
        merfishviewerxp_version=__version__,
        created_at=now_iso,
        updated_at=now_iso,
        source_root=str(dataset.root_path),
        fingerprint=compute_source_fingerprint(dataset),
        indexing_config=indexing_config,
        image_orientation_apply=dataset.image_orientation_apply,
        components_built=components_built,
    )


def load_manifest(cache_dir: Path) -> Manifest | None:
    path = cache_dir / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        return Manifest.model_validate(json.loads(path.read_text()))
    except (json.JSONDecodeError, ValueError):
        return None


def save_manifest(cache_dir: Path, manifest: Manifest) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / MANIFEST_FILENAME
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(manifest.model_dump_json(indent=2))
    tmp_path.replace(path)


class InvalidationDecision(BaseModel):
    rebuild_images: bool
    rebuild_spots: bool
    rebuild_genes: bool
    # Also set when masks disappear, so the indexer removes stale boundaries.
    rebuild_segmentation: bool = False
    reasons: list[str]

    @property
    def rebuild_anything(self) -> bool:
        return self.rebuild_images or self.rebuild_spots or self.rebuild_genes or self.rebuild_segmentation


def decide_invalidation(old: Manifest | None, current_fp: SourceFingerprint) -> InvalidationDecision:
    """Decide which cache components must be rebuilt (spec 15.3)."""
    has_masks = current_fp.mask_file_count > 0
    if old is None:
        return InvalidationDecision(
            rebuild_images=True,
            rebuild_spots=True,
            rebuild_genes=True,
            rebuild_segmentation=has_masks,
            reasons=["no existing cache"],
        )

    reasons: list[str] = []
    if old.cache_schema_version != CACHE_SCHEMA_VERSION:
        return InvalidationDecision(
            rebuild_images=True,
            rebuild_spots=True,
            rebuild_genes=True,
            rebuild_segmentation=has_masks,
            reasons=[f"cache schema version changed ({old.cache_schema_version} -> {CACHE_SCHEMA_VERSION})"],
        )

    coords_changed = old.fingerprint.positions != current_fp.positions or old.fingerprint.microscope != current_fp.microscope
    if old.fingerprint.positions != current_fp.positions:
        reasons.append("positions.csv changed")
    if old.fingerprint.microscope != current_fp.microscope:
        reasons.append("microscope_parameters.json changed")

    images_changed = coords_changed or old.fingerprint.image_inventory_hash != current_fp.image_inventory_hash
    if not coords_changed and old.fingerprint.image_inventory_hash != current_fp.image_inventory_hash:
        reasons.append("image file inventory changed")

    codebooks_changed = old.fingerprint.codebooks != current_fp.codebooks
    barcodes_changed = old.fingerprint.barcode_exports != current_fp.barcode_exports
    if codebooks_changed:
        reasons.append("codebook(s) changed")
    if barcodes_changed:
        reasons.append("barcode export(s) changed")

    spots_changed = coords_changed or codebooks_changed or barcodes_changed
    genes_changed = codebooks_changed or barcodes_changed

    masks_changed = old.fingerprint.mask_inventory_hash != current_fp.mask_inventory_hash
    if masks_changed:
        reasons.append("segmentation mask inventory changed")
    segmentation_missing = has_masks and not old.components_built.get("segmentation", False)
    segmentation_outdated = (
        has_masks
        and not segmentation_missing
        and old.segmentation_format_version != SEGMENTATION_FORMAT_VERSION
    )
    if segmentation_outdated:
        reasons.append(
            f"cell-boundary cache format changed ({old.segmentation_format_version} -> {SEGMENTATION_FORMAT_VERSION})"
        )

    missing_components = [k for k, v in old.components_built.items() if not v and k != "segmentation"]
    if segmentation_missing and not masks_changed:
        missing_components.append("segmentation")
    if missing_components:
        reasons.append(f"previously incomplete components: {missing_components}")

    return InvalidationDecision(
        rebuild_images=images_changed or not old.components_built.get("images", False),
        rebuild_spots=spots_changed or not old.components_built.get("spots", False),
        rebuild_genes=genes_changed or not old.components_built.get("genes", False),
        # The boundary mosaic shares the stain mosaic's grid, which depends on
        # positions and image shapes, so it follows image changes too.
        rebuild_segmentation=masks_changed
        or segmentation_outdated
        or (has_masks and (images_changed or segmentation_missing)),
        reasons=reasons or ["unchanged"],
    )
