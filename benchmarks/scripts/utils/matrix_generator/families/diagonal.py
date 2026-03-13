"""F2: Pure diagonal and narrow-banded matrices."""

from __future__ import annotations

import random
from typing import List, Tuple

from ..registry import register
from ..types import COOEntries, GeneratedMatrix, MatrixSpec


def _sample_value(rng: random.Random, value_mode: str) -> float:
    if value_mode == "ones":
        return 1.0
    return rng.uniform(-1.0, 1.0)


def _place_diagonals(rows: int, cols: int, offsets: List[int], rng: random.Random, value_mode: str) -> COOEntries:
    """Place entries on specified diagonals. offset=0 is main, +k is upper, -k is lower."""
    entries: COOEntries = []
    seen = set()
    for k in offsets:
        if k >= 0:
            r_start, c_start = 0, k
        else:
            r_start, c_start = -k, 0
        r, c = r_start, c_start
        while r < rows and c < cols:
            if (r, c) not in seen:
                entries.append((r, c, _sample_value(rng, value_mode)))
                seen.add((r, c))
            r += 1
            c += 1
    entries.sort()
    return entries


def _place_block_diagonal(rows: int, cols: int, block_size: int, block_count: int,
                           rng: random.Random, value_mode: str, density: float = 1.0) -> COOEntries:
    """Place dense (or partially dense) blocks along the diagonal."""
    entries: COOEntries = []
    seen = set()
    for b in range(block_count):
        r_start = b * block_size
        c_start = b * block_size
        for r in range(r_start, min(r_start + block_size, rows)):
            for c in range(c_start, min(c_start + block_size, cols)):
                if density < 1.0 and rng.random() > density:
                    continue
                if (r, c) not in seen:
                    entries.append((r, c, _sample_value(rng, value_mode)))
                    seen.add((r, c))
    entries.sort()
    return entries


@register("diagonal")
def generate_diagonal(spec: MatrixSpec, rng: random.Random) -> GeneratedMatrix:
    fp = spec.family_params
    rows, cols = spec.rows, spec.cols
    vm = spec.value_mode
    v = spec.variant

    if v == "strict_diagonal":
        entries = _place_diagonals(rows, cols, [0], rng, vm)

    elif v == "upper_plus_main":
        entries = _place_diagonals(rows, cols, [0, 1], rng, vm)

    elif v == "tridiagonal":
        entries = _place_diagonals(rows, cols, [-1, 0, 1], rng, vm)

    elif v == "k_diagonal":
        k = int(fp.get("k", 10))
        offsets = list(range(-(k // 2), k // 2 + 1))
        entries = _place_diagonals(rows, cols, offsets, rng, vm)

    elif v == "wide_band":
        k = int(fp.get("k", 100))
        offsets = list(range(-(k // 2), k // 2 + 1))
        entries = _place_diagonals(rows, cols, offsets, rng, vm)

    elif v == "band_with_spikes":
        k = int(fp.get("k", 5))
        offsets = list(range(-(k // 2), k // 2 + 1))
        entries = _place_diagonals(rows, cols, offsets, rng, vm)
        # Spikes handled via postprocess.spike_count or inline
        spike_count = int(fp.get("spike_count", 50))
        existing = set((r, c) for r, c, _ in entries)
        added = 0
        attempts = 0
        while added < spike_count and attempts < spike_count * 20:
            r = rng.randrange(rows)
            c = rng.randrange(cols)
            if abs(r - c) > k and (r, c) not in existing:
                entries.append((r, c, _sample_value(rng, vm)))
                existing.add((r, c))
                added += 1
            attempts += 1
        entries.sort()

    elif v == "block_diagonal_dense":
        block_size = int(fp.get("block_size", 100))
        block_count = int(fp.get("block_count", min(rows, cols) // max(block_size, 1)))
        entries = _place_block_diagonal(rows, cols, block_size, block_count, rng, vm)

    elif v == "block_diagonal_with_offdiag":
        block_size = int(fp.get("block_size", 100))
        block_count = int(fp.get("block_count", min(rows, cols) // max(block_size, 1)))
        entries = _place_block_diagonal(rows, cols, block_size, block_count, rng, vm)
        # Add small off-diagonal blocks
        offdiag_count = int(fp.get("offdiag_count", max(1, block_count // 3)))
        offdiag_size = int(fp.get("offdiag_size", max(1, block_size // 4)))
        offdiag_density = float(fp.get("offdiag_density", 0.5))
        existing = set((r, c) for r, c, _ in entries)
        for _ in range(offdiag_count):
            bi = rng.randrange(block_count)
            bj = rng.randrange(block_count)
            if bi == bj:
                bj = (bi + 1) % block_count
            r_start = bi * block_size
            c_start = bj * block_size
            for r in range(r_start, min(r_start + offdiag_size, rows)):
                for c in range(c_start, min(c_start + offdiag_size, cols)):
                    if rng.random() <= offdiag_density and (r, c) not in existing:
                        entries.append((r, c, _sample_value(rng, vm)))
                        existing.add((r, c))
        entries.sort()

    elif v == "narrow_band_tall":
        k = int(fp.get("k", 3))
        offsets = list(range(-(k // 2), k // 2 + 1))
        entries = _place_diagonals(rows, cols, offsets, rng, vm)

    elif v == "narrow_band_wide":
        k = int(fp.get("k", 3))
        offsets = list(range(-(k // 2), k // 2 + 1))
        entries = _place_diagonals(rows, cols, offsets, rng, vm)

    else:
        raise ValueError(f"Unknown diagonal variant '{v}'")

    return GeneratedMatrix(rows=rows, cols=cols, entries=entries, params={"variant": v, **fp})
