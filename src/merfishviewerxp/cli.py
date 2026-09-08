"""``merfishviewerxp`` console entry point (spec section 17)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import typer

from .api import MerfishDataset, index_dataset
from .config import load_config
from .diagnostics.dataset_diagnostics import diagnose_dataset
from .errors import MerfishViewerXPError
from .logging_config import configure_logging
from .storage.cache import CacheManager

app = typer.Typer(add_completion=False, help="Interactive viewer for MERlin MERFISH decoding results.")


def _cli_overrides(
    *,
    tile_size_um: float | None,
    image_chunk_size: int | None,
    overlap_mode: str | None,
    fov_crop_px: int | None,
) -> dict:
    overrides: dict = {"images": {}, "spots": {}}
    if tile_size_um is not None:
        overrides["spots"]["spatial_tile_size_um"] = tile_size_um
    if image_chunk_size is not None:
        overrides["images"]["chunk_size"] = image_chunk_size
    if overlap_mode is not None:
        overrides["images"]["overlap_mode"] = overlap_mode
    if fov_crop_px is not None:
        overrides["images"]["fov_crop_px"] = fov_crop_px
    return overrides


@app.command()
def index(
    experiment: Path = typer.Argument(..., exists=True, file_okay=False, help="MERlin experiment root folder."),
    force: bool = typer.Option(False, "--force", help="Rebuild the cache from scratch."),
    cache_dir: Path | None = typer.Option(None, "--cache-dir", help="Alternate cache location."),
    tile_size_um: float | None = typer.Option(None, "--tile-size-um"),
    image_chunk_size: int | None = typer.Option(None, "--image-chunk-size"),
    overlap_mode: str | None = typer.Option(None, "--overlap-mode", help="feather|mean|max|first"),
    fov_crop_px: int | None = typer.Option(None, "--fov-crop-px"),
    no_image_pyramid: bool = typer.Option(False, "--no-image-pyramid"),
    no_transcripts: bool = typer.Option(False, "--no-transcripts"),
    no_images: bool = typer.Option(False, "--no-images"),
) -> None:
    """Build or refresh the on-disk cache without opening the viewer."""
    configure_logging()
    try:
        dataset = MerfishDataset.open(experiment)
        report = dataset.validate()
        for issue in report.issues:
            typer.echo(f"[{issue.level}] {issue.message}")
        if not report.is_valid:
            typer.echo("Dataset failed validation; aborting.", err=True)
            raise typer.Exit(code=1)

        config = load_config(
            dataset_root=experiment,
            cache_dir=cache_dir,
            cli_overrides=_cli_overrides(
                tile_size_um=tile_size_um, image_chunk_size=image_chunk_size, overlap_mode=overlap_mode, fov_crop_px=fov_crop_px
            ),
        )
        manifest = index_dataset(
            dataset,
            cache_dir=cache_dir,
            config=config,
            force=force,
            build_images=not no_images,
            build_transcripts=not no_transcripts,
            build_pyramid_levels=not no_image_pyramid,
        )
        resolved_cache_dir = CacheManager(dataset_root=experiment, cache_dir=cache_dir).cache_dir
        typer.echo(f"Cache ready at {resolved_cache_dir}")
        typer.echo(f"Components built: {manifest.components_built}")
    except MerfishViewerXPError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def view(
    experiment: Path = typer.Argument(..., exists=True, file_okay=False),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
) -> None:
    """Validate, build any missing/stale cache, then open the interactive viewer."""
    configure_logging()
    try:
        dataset = MerfishDataset.open(experiment)
        report = dataset.validate()
        for issue in report.issues:
            typer.echo(f"[{issue.level}] {issue.message}")
        if not report.is_valid:
            typer.echo("Dataset failed validation; aborting.", err=True)
            raise typer.Exit(code=1)

        config = load_config(dataset_root=experiment, cache_dir=cache_dir)
        index_dataset(dataset, cache_dir=cache_dir, config=config)

        from .indexed_dataset import IndexedDataset
        from .viewer.app import launch_viewer

        cache = CacheManager(dataset_root=experiment, cache_dir=cache_dir)
        indexed = IndexedDataset.open(cache.cache_dir)
        launch_viewer(indexed, config=config)
    except MerfishViewerXPError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def diagnose(
    experiment: Path = typer.Argument(..., exists=True, file_okay=False),
    json_out: Path | None = typer.Option(None, "--json", help="Also write the diagnosis as JSON."),
) -> None:
    """Print resolved coordinate-transform parameters and dataset structure."""
    configure_logging(level=30)  # WARNING; keep diagnose output focused on the report itself
    try:
        dataset = MerfishDataset.open(experiment)
        result = diagnose_dataset(dataset.descriptor)
    except MerfishViewerXPError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    summary = {k: v for k, v in result.items() if k not in ("fov_bounding_boxes_um", "transcript_samples")}
    typer.echo(json.dumps(summary, indent=2, default=str))
    typer.echo(f"(fov_bounding_boxes_um omitted from console output: {len(result['fov_bounding_boxes_um'])} entries)")
    if result["warnings"]:
        typer.echo("\nWarnings:")
        for w in result["warnings"]:
            typer.echo(f"  - {w}")

    if json_out is not None:
        json_out.write_text(json.dumps(result, indent=2, default=str))
        typer.echo(f"\nFull diagnosis written to {json_out}")


@app.command("clear-cache")
def clear_cache(
    experiment: Path = typer.Argument(..., exists=True, file_okay=False),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    yes: bool = typer.Option(False, "--yes", help="Do not prompt for confirmation."),
) -> None:
    """Delete the on-disk cache. The original MERlin output is never touched."""
    cache = CacheManager(dataset_root=experiment, cache_dir=cache_dir)
    if not cache.cache_dir.exists():
        typer.echo(f"No cache found at {cache.cache_dir}.")
        return
    if not yes and not typer.confirm(f"Delete cache at {cache.cache_dir}?"):
        typer.echo("Aborted.")
        raise typer.Exit(code=1)
    shutil.rmtree(cache.cache_dir)
    typer.echo(f"Deleted {cache.cache_dir}.")


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
