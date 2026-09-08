from fixtures.synthetic_dataset import build_synthetic_dataset

from merfishviewerxp.adapters.merlin.dataset import MerlinDatasetAdapter
from merfishviewerxp.indexing.manifest import (
    build_manifest,
    compute_source_fingerprint,
    decide_invalidation,
)


def _discover(root):
    return MerlinDatasetAdapter(root).discover()


def test_no_existing_cache_rebuilds_everything(tmp_path):
    root = tmp_path / "exp"
    build_synthetic_dataset(root)
    dataset = _discover(root)
    fp = compute_source_fingerprint(dataset)
    decision = decide_invalidation(None, fp)
    assert decision.rebuild_images and decision.rebuild_spots and decision.rebuild_genes


def test_unchanged_dataset_rebuilds_nothing(tmp_path):
    root = tmp_path / "exp"
    build_synthetic_dataset(root)
    dataset = _discover(root)
    manifest = build_manifest(
        dataset,
        indexing_config={},
        now_iso="2026-01-01T00:00:00+00:00",
        components_built={"images": True, "spots": True, "genes": True},
    )
    fp_again = compute_source_fingerprint(_discover(root))
    decision = decide_invalidation(manifest, fp_again)
    assert not decision.rebuild_anything


def test_positions_change_invalidates_spots_and_images_not_genes(tmp_path):
    root = tmp_path / "exp"
    build_synthetic_dataset(root)
    dataset = _discover(root)
    manifest = build_manifest(
        dataset, indexing_config={}, now_iso="t", components_built={"images": True, "spots": True, "genes": True}
    )

    (root / "positions.csv").write_text((root / "positions.csv").read_text().replace("0.0", "1.0"))
    dataset2 = _discover(root)
    fp2 = compute_source_fingerprint(dataset2)
    decision = decide_invalidation(manifest, fp2)
    assert decision.rebuild_images
    assert decision.rebuild_spots
    assert not decision.rebuild_genes
    assert any("positions.csv" in r for r in decision.reasons)


def test_codebook_change_invalidates_spots_and_genes_not_images(tmp_path):
    root = tmp_path / "exp"
    build_synthetic_dataset(root)
    dataset = _discover(root)
    manifest = build_manifest(
        dataset, indexing_config={}, now_iso="t", components_built={"images": True, "spots": True, "genes": True}
    )

    codebook_path = next(root.glob("codebook_0_*.csv"))
    codebook_path.write_text(codebook_path.read_text().replace("GENEA", "GENEZ"))
    dataset2 = _discover(root)
    fp2 = compute_source_fingerprint(dataset2)
    decision = decide_invalidation(manifest, fp2)
    assert not decision.rebuild_images
    assert decision.rebuild_spots
    assert decision.rebuild_genes


def test_incomplete_previous_build_forces_rebuild_of_missing_component(tmp_path):
    root = tmp_path / "exp"
    build_synthetic_dataset(root)
    dataset = _discover(root)
    manifest = build_manifest(
        dataset, indexing_config={}, now_iso="t", components_built={"images": False, "spots": True, "genes": True}
    )
    fp_again = compute_source_fingerprint(_discover(root))
    decision = decide_invalidation(manifest, fp_again)
    assert decision.rebuild_images
    assert not decision.rebuild_spots
    assert not decision.rebuild_genes
