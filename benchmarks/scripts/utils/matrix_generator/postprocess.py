"""Shared postprocessing transforms for generated matrices."""

from __future__ import annotations

import random
from typing import Dict

from .types import COOEntries, GeneratedMatrix


def transpose(matrix: GeneratedMatrix) -> GeneratedMatrix:
    """Swap rows and cols of all entries."""
    new_entries: COOEntries = [
        (col, row, val) for row, col, val in matrix.entries
    ]
    new_entries.sort()
    return GeneratedMatrix(
        rows=matrix.cols,
        cols=matrix.rows,
        entries=new_entries,
        params=matrix.params,
    )


def symmetrize(matrix: GeneratedMatrix) -> GeneratedMatrix:
    """Add (j, i) for every (i, j) where i != j, deduplicating."""
    seen: Dict[tuple, float] = {}
    for r, c, v in matrix.entries:
        seen[(r, c)] = v
        if r != c:
            seen.setdefault((c, r), v)
    entries = sorted((r, c, v) for (r, c), v in seen.items())
    n = max(matrix.rows, matrix.cols)
    return GeneratedMatrix(rows=n, cols=n, entries=entries, params=matrix.params)


def random_delete(matrix: GeneratedMatrix, fraction: float, rng: random.Random) -> GeneratedMatrix:
    """Randomly remove a fraction of entries."""
    if fraction <= 0:
        return matrix
    entries = [e for e in matrix.entries if rng.random() >= fraction]
    return GeneratedMatrix(
        rows=matrix.rows, cols=matrix.cols, entries=entries, params=matrix.params
    )


def add_noise(matrix: GeneratedMatrix, count: int, rng: random.Random, value_mode: str = "ones") -> GeneratedMatrix:
    """Add random scattered entries outside existing positions."""
    existing = set((r, c) for r, c, _ in matrix.entries)
    entries = list(matrix.entries)
    added = 0
    attempts = 0
    max_attempts = count * 10
    while added < count and attempts < max_attempts:
        r = rng.randrange(matrix.rows)
        c = rng.randrange(matrix.cols)
        if (r, c) not in existing:
            v = 1.0 if value_mode == "ones" else rng.uniform(-1.0, 1.0)
            entries.append((r, c, v))
            existing.add((r, c))
            added += 1
        attempts += 1
    entries.sort()
    return GeneratedMatrix(
        rows=matrix.rows, cols=matrix.cols, entries=entries, params=matrix.params
    )


def add_spikes(matrix: GeneratedMatrix, count: int, rng: random.Random, value_mode: str = "ones") -> GeneratedMatrix:
    """Add random entries far from the diagonal."""
    existing = set((r, c) for r, c, _ in matrix.entries)
    entries = list(matrix.entries)
    added = 0
    attempts = 0
    max_attempts = count * 20
    while added < count and attempts < max_attempts:
        r = rng.randrange(matrix.rows)
        # pick column far from diagonal
        if rng.random() < 0.5:
            c = rng.randrange(0, max(1, matrix.cols // 4))
        else:
            c = rng.randrange(max(0, matrix.cols * 3 // 4), matrix.cols)
        if abs(r - c) > min(matrix.rows, matrix.cols) // 4 and (r, c) not in existing:
            v = 1.0 if value_mode == "ones" else rng.uniform(-1.0, 1.0)
            entries.append((r, c, v))
            existing.add((r, c))
            added += 1
        attempts += 1
    entries.sort()
    return GeneratedMatrix(
        rows=matrix.rows, cols=matrix.cols, entries=entries, params=matrix.params
    )


def apply_postprocess(matrix: GeneratedMatrix, postprocess: Dict[str, object], rng: random.Random, value_mode: str = "ones") -> GeneratedMatrix:
    """Apply postprocessing steps from a spec's postprocess dict."""
    if not postprocess:
        return matrix

    if postprocess.get("transpose", False):
        matrix = transpose(matrix)

    if postprocess.get("symmetrize", False):
        matrix = symmetrize(matrix)

    delete_frac = float(postprocess.get("random_delete", 0))
    if delete_frac > 0:
        matrix = random_delete(matrix, delete_frac, rng)

    noise_count = int(postprocess.get("noise_count", 0))
    if noise_count > 0:
        matrix = add_noise(matrix, noise_count, rng, value_mode)

    spike_count = int(postprocess.get("spike_count", 0))
    if spike_count > 0:
        matrix = add_spikes(matrix, spike_count, rng, value_mode)

    return matrix
