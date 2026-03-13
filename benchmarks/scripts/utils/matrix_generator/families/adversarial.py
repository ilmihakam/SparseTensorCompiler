"""F10: Adversarial / edge cases for heuristics (Phase 4)."""

from __future__ import annotations

import random
from typing import Set

from ..registry import register
from ..types import COOEntries, GeneratedMatrix, MatrixSpec


def _sample_value(rng: random.Random, value_mode: str) -> float:
    if value_mode == "ones":
        return 1.0
    return rng.uniform(-1.0, 1.0)


@register("adversarial")
def generate_adversarial(spec: MatrixSpec, rng: random.Random) -> GeneratedMatrix:
    fp = spec.family_params
    rows, cols = spec.rows, spec.cols
    vm = spec.value_mode
    v = spec.variant

    tile_size = int(fp.get("tile_size", 32))

    if v == "borderline_tile_useful":
        # Each tile has nnz just below a "useful" threshold
        threshold_nnz = int(fp.get("threshold_nnz", tile_size // 2))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for ti in range(0, rows, tile_size):
            for tj in range(0, cols, tile_size):
                count = threshold_nnz - 1
                for _ in range(max(0, count)):
                    r = rng.randint(ti, min(ti + tile_size - 1, rows - 1))
                    c = rng.randint(tj, min(tj + tile_size - 1, cols - 1))
                    if (r, c) not in seen:
                        entries.append((r, c, _sample_value(rng, vm)))
                        seen.add((r, c))

    elif v == "high_tile_span":
        # High nnz per tile but spread over whole tile (span_frac ~ 1)
        nnz_per_tile = int(fp.get("nnz_per_tile", tile_size))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for ti in range(0, rows, tile_size):
            for tj in range(0, cols, tile_size):
                tile_h = min(tile_size, rows - ti)
                tile_w = min(tile_size, cols - tj)
                for _ in range(min(nnz_per_tile, tile_h * tile_w)):
                    r = rng.randint(ti, ti + tile_h - 1)
                    c = rng.randint(tj, tj + tile_w - 1)
                    if (r, c) not in seen:
                        entries.append((r, c, _sample_value(rng, vm)))
                        seen.add((r, c))

    elif v == "high_tile_row_cv":
        # Most nnz in single row of each tile
        nnz_per_tile = int(fp.get("nnz_per_tile", tile_size))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for ti in range(0, rows, tile_size):
            for tj in range(0, cols, tile_size):
                tile_h = min(tile_size, rows - ti)
                tile_w = min(tile_size, cols - tj)
                hot_row = ti + rng.randrange(tile_h)
                for _ in range(min(nnz_per_tile, tile_w)):
                    c = rng.randint(tj, tj + tile_w - 1)
                    if (hot_row, c) not in seen:
                        entries.append((hot_row, c, _sample_value(rng, vm)))
                        seen.add((hot_row, c))

    elif v == "mixed_narrow_wide":
        # Many narrow rows interleaved with single extremely wide row
        narrow_nnz = int(fp.get("narrow_nnz", 2))
        wide_row_freq = int(fp.get("wide_row_freq", 50))
        wide_nnz = int(fp.get("wide_nnz", cols // 2))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for r in range(rows):
            if r % wide_row_freq == 0:
                nnz = min(wide_nnz, cols)
            else:
                nnz = min(narrow_nnz, cols)
            if nnz > 0:
                chosen = rng.sample(range(cols), nnz)
                for c in chosen:
                    if (r, c) not in seen:
                        entries.append((r, c, _sample_value(rng, vm)))
                        seen.add((r, c))

    elif v == "tile_boundary_runs":
        # Runs that straddle tile boundaries
        run_length = int(fp.get("run_length", tile_size))
        runs_per_row = int(fp.get("runs_per_row", 2))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for r in range(rows):
            for _ in range(runs_per_row):
                # Start near a tile boundary
                tile_idx = rng.randrange(max(1, cols // tile_size))
                start = tile_idx * tile_size - run_length // 2
                start = max(0, min(start, cols - run_length))
                for c in range(start, min(start + run_length, cols)):
                    if (r, c) not in seen:
                        entries.append((r, c, _sample_value(rng, vm)))
                        seen.add((r, c))

    elif v == "tile_boundary_cols":
        # Column variant of tile_boundary_runs (transpose-like stress)
        run_length = int(fp.get("run_length", tile_size))
        runs_per_col = int(fp.get("runs_per_col", 2))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for c in range(cols):
            for _ in range(runs_per_col):
                tile_idx = rng.randrange(max(1, rows // tile_size))
                start = tile_idx * tile_size - run_length // 2
                start = max(0, min(start, rows - run_length))
                for r in range(start, min(start + run_length, rows)):
                    if (r, c) not in seen:
                        entries.append((r, c, _sample_value(rng, vm)))
                        seen.add((r, c))

    elif v == "high_entropy":
        # Extremely high normalized entropy of row-length distribution
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for r in range(rows):
            nnz = rng.randint(1, min(cols, max(2, cols // 5)))
            chosen = rng.sample(range(cols), nnz)
            for c in chosen:
                if (r, c) not in seen:
                    entries.append((r, c, _sample_value(rng, vm)))
                    seen.add((r, c))

    elif v == "fragmented_runs":
        # High avg_runs_per_row with very short runs and big gaps
        runs_per_row = int(fp.get("runs_per_row", 20))
        run_length = int(fp.get("run_length", 2))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for r in range(rows):
            for _ in range(runs_per_row):
                start = rng.randrange(cols)
                for c in range(start, min(start + run_length, cols)):
                    if (r, c) not in seen:
                        entries.append((r, c, _sample_value(rng, vm)))
                        seen.add((r, c))

    elif v == "extreme_aspect":
        # Extreme aspect ratio with scattered nnz
        avg_nnz = int(fp.get("avg_nnz_row", 5))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for r in range(rows):
            nnz = min(avg_nnz, cols)
            if nnz > 0:
                chosen = rng.sample(range(cols), nnz)
                for c in chosen:
                    if (r, c) not in seen:
                        entries.append((r, c, _sample_value(rng, vm)))
                        seen.add((r, c))

    elif v == "reorder_degrades":
        # Current order matches CSR stride-1, reordering would hurt
        nnz_per_row = int(fp.get("nnz_per_row", 10))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for r in range(rows):
            # Place contiguous cols starting at r (diagonal-ish runs)
            start = r % max(1, cols - nnz_per_row)
            for c in range(start, min(start + nnz_per_row, cols)):
                if (r, c) not in seen:
                    entries.append((r, c, _sample_value(rng, vm)))
                    seen.add((r, c))

    else:
        raise ValueError(f"Unknown adversarial variant '{v}'")

    entries.sort()
    return GeneratedMatrix(rows=rows, cols=cols, entries=entries, params={"variant": v, **fp})
