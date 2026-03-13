"""F3: Block-sparse with dense tiles."""

from __future__ import annotations

import math
import random
from typing import List, Set, Tuple

from ..registry import register
from ..types import COOEntries, GeneratedMatrix, MatrixSpec


def _sample_value(rng: random.Random, value_mode: str) -> float:
    if value_mode == "ones":
        return 1.0
    return rng.uniform(-1.0, 1.0)


def _generate_block_grid(
    rows: int, cols: int,
    grid_r: int, grid_c: int,
    active_blocks: Set[Tuple[int, int]],
    internal_density: float,
    rng: random.Random,
    value_mode: str,
) -> COOEntries:
    """Generate entries for a block-sparse matrix.

    The matrix is divided into grid_r x grid_c blocks.
    Only blocks in active_blocks set are populated.
    """
    block_h = rows // grid_r
    block_w = cols // grid_c
    entries: COOEntries = []

    for bi, bj in sorted(active_blocks):
        r_start = bi * block_h
        c_start = bj * block_w
        r_end = min(r_start + block_h, rows)
        c_end = min(c_start + block_w, cols)
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                if rng.random() <= internal_density:
                    entries.append((r, c, _sample_value(rng, value_mode)))

    entries.sort()
    return entries


@register("block_sparse")
def generate_block_sparse(spec: MatrixSpec, rng: random.Random) -> GeneratedMatrix:
    fp = spec.family_params
    rows, cols = spec.rows, spec.cols
    vm = spec.value_mode
    v = spec.variant

    grid_r = int(fp.get("grid_r", 10))
    grid_c = int(fp.get("grid_c", grid_r))
    internal_density = float(fp.get("internal_density", 1.0))

    if v == "explicit_blocks":
        # Active blocks specified as list of [bi, bj] pairs
        block_list = fp.get("active_blocks", [[0, 0]])
        active = {(int(b[0]), int(b[1])) for b in block_list}

    elif v == "random_block_occupancy":
        num_active = int(fp.get("num_active", 20))
        all_blocks = [(i, j) for i in range(grid_r) for j in range(grid_c)]
        num_active = min(num_active, len(all_blocks))
        active = set(rng.sample(all_blocks, num_active))

    elif v == "banded_block_diagonal":
        band_width = int(fp.get("block_band_width", 1))
        active = set()
        for i in range(min(grid_r, grid_c)):
            for offset in range(-band_width, band_width + 1):
                j = i + offset
                if 0 <= j < grid_c:
                    active.add((i, j))

    elif v == "hierarchical_blocks":
        # Big blocks, each containing sub-blocks
        sub_grid = int(fp.get("sub_grid", 4))
        big_active = int(fp.get("big_active", 4))
        sub_active_per_big = int(fp.get("sub_active", 4))

        big_blocks = [(i, j) for i in range(grid_r) for j in range(grid_c)]
        big_blocks = rng.sample(big_blocks, min(big_active, len(big_blocks)))

        block_h = rows // grid_r
        block_w = cols // grid_c
        sub_h = block_h // sub_grid
        sub_w = block_w // sub_grid
        entries: COOEntries = []

        for bi, bj in big_blocks:
            all_subs = [(si, sj) for si in range(sub_grid) for sj in range(sub_grid)]
            chosen_subs = rng.sample(all_subs, min(sub_active_per_big, len(all_subs)))
            for si, sj in chosen_subs:
                r_start = bi * block_h + si * sub_h
                c_start = bj * block_w + sj * sub_w
                for r in range(r_start, min(r_start + sub_h, rows)):
                    for c in range(c_start, min(c_start + sub_w, cols)):
                        if rng.random() <= internal_density:
                            entries.append((r, c, _sample_value(rng, vm)))

        entries.sort()
        return GeneratedMatrix(rows=rows, cols=cols, entries=entries, params={"variant": v, **fp})

    elif v == "rectangular_blocks":
        # Blocks with different aspect ratios
        num_active = int(fp.get("num_active", 10))
        block_h_mult = float(fp.get("block_h_mult", 2.0))
        block_w_mult = float(fp.get("block_w_mult", 0.5))
        block_h = int((rows // grid_r) * block_h_mult)
        block_w = int((cols // grid_c) * block_w_mult)
        entries: COOEntries = []
        for _ in range(num_active):
            r_start = rng.randrange(max(1, rows - block_h))
            c_start = rng.randrange(max(1, cols - block_w))
            for r in range(r_start, min(r_start + block_h, rows)):
                for c in range(c_start, min(c_start + block_w, cols)):
                    if rng.random() <= internal_density:
                        entries.append((r, c, _sample_value(rng, vm)))
        # Deduplicate
        entries = sorted(set(entries))
        return GeneratedMatrix(rows=rows, cols=cols, entries=entries, params={"variant": v, **fp})

    elif v == "sparse_internal":
        # Blocks have low internal density
        internal_density = float(fp.get("internal_density", 0.1))
        num_active = int(fp.get("num_active", 20))
        all_blocks = [(i, j) for i in range(grid_r) for j in range(grid_c)]
        num_active = min(num_active, len(all_blocks))
        active = set(rng.sample(all_blocks, num_active))

    elif v == "uniform_blocks_per_row":
        blocks_per_row = int(fp.get("blocks_per_row", 3))
        active = set()
        for i in range(grid_r):
            cols_avail = list(range(grid_c))
            chosen = rng.sample(cols_avail, min(blocks_per_row, len(cols_avail)))
            for j in chosen:
                active.add((i, j))

    elif v == "powerlaw_blocks":
        # Power-law over block-rows: few rows heavily populated
        alpha = float(fp.get("alpha", 1.5))
        total_active = int(fp.get("total_active", grid_r * 2))
        active = set()
        for _ in range(total_active):
            # Zipf-like selection for row
            bi = int(rng.paretovariate(alpha)) % grid_r
            bj = rng.randrange(grid_c)
            active.add((bi, bj))

    elif v == "coarse_blocks":
        num_active = int(fp.get("num_active", 3))
        all_blocks = [(i, j) for i in range(grid_r) for j in range(grid_c)]
        num_active = min(num_active, len(all_blocks))
        active = set(rng.sample(all_blocks, num_active))

    else:
        raise ValueError(f"Unknown block_sparse variant '{v}'")

    entries = _generate_block_grid(rows, cols, grid_r, grid_c, active, internal_density, rng, vm)
    return GeneratedMatrix(rows=rows, cols=cols, entries=entries, params={"variant": v, **fp})
