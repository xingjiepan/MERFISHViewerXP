"""Deterministic gene identity and color assignment (spec section 12).

The same gene name must always receive the same ``gene_id`` and the same
display color, regardless of list order, dataset re-indexing, or which
codebook it came from. We derive both from a stable hash of the gene name.
"""

from __future__ import annotations

import colorsys
import hashlib
import re

DEFAULT_PALETTE_SIZE = 512

_BLANK_PATTERN = re.compile(r"blank", re.IGNORECASE)


def is_blank_name(gene_name: str) -> bool:
    """Default blank/control matching: case-insensitive names containing ``blank``."""
    return bool(_BLANK_PATTERN.search(gene_name))


def _stable_hash(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def deterministic_gene_id(gene_name: str) -> int:
    """A stable, order-independent integer id for a gene name."""
    return _stable_hash(f"gene:{gene_name}") % (2**31 - 1)


def gene_color_index(gene_name: str, palette_size: int = DEFAULT_PALETTE_SIZE) -> int:
    return _stable_hash(f"color:{gene_name}") % palette_size


def gene_color_rgb(gene_name: str, palette_size: int = DEFAULT_PALETTE_SIZE) -> tuple[float, float, float]:
    """A deterministic RGB triple (0-1 floats) for a gene name.

    Uses the golden-angle-like spread over hue derived from a stable hash
    index so that colors for "adjacent" hash buckets remain visually
    distinct even for large palettes.
    """
    idx = gene_color_index(gene_name, palette_size)
    hue = (idx * 0.6180339887498949) % 1.0
    saturation = 0.55 + 0.35 * ((idx // 7) % 3) / 2
    value = 0.85 + 0.15 * (idx % 2)
    return colorsys.hsv_to_rgb(hue, saturation, value)


def gene_color_hex(gene_name: str, palette_size: int = DEFAULT_PALETTE_SIZE) -> str:
    r, g, b = gene_color_rgb(gene_name, palette_size)
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"
