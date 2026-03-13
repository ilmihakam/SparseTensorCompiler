#!/usr/bin/env python3
"""Unit tests for F3: Block-sparse family."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_UTILS_DIR = str(Path(__file__).resolve().parent.parent / "utils")
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from matrix_generator.types import MatrixSpec
from matrix_generator import generate_one


def _make_spec(variant: str, rows: int, cols: int, seed: int, **fp) -> MatrixSpec:
    return MatrixSpec(
        name=f"test_{variant}",
        family="block_sparse",
        variant=variant,
        family_number="F3.x",
        rows=rows, cols=cols, seed=seed,
        tags=["test"],
        family_params=fp,
        postprocess={},
        value_mode="ones",
    )


class TestF03BlockSparse(unittest.TestCase):
    def test_explicit_blocks_positions(self):
        spec = _make_spec("explicit_blocks", 400, 400, 1,
                          grid_r=4, grid_c=4, active_blocks=[[0, 0], [1, 1]])
        mat = generate_one(spec)
        block_h = 100
        for r, c, _ in mat.entries:
            br, bc = r // block_h, c // block_h
            self.assertIn((br, bc), {(0, 0), (1, 1)},
                          f"Entry ({r},{c}) in block ({br},{bc}) not in active set")

    def test_random_block_occupancy(self):
        spec = _make_spec("random_block_occupancy", 400, 400, 42,
                          grid_r=4, grid_c=4, num_active=4)
        mat = generate_one(spec)
        self.assertGreater(len(mat.entries), 0)
        # Count occupied blocks
        blocks = set()
        for r, c, _ in mat.entries:
            blocks.add((r // 100, c // 100))
        self.assertEqual(len(blocks), 4)

    def test_banded_block_diagonal(self):
        spec = _make_spec("banded_block_diagonal", 500, 500, 1,
                          grid_r=5, grid_c=5, block_band_width=1)
        mat = generate_one(spec)
        for r, c, _ in mat.entries:
            br, bc = r // 100, c // 100
            self.assertLessEqual(abs(br - bc), 1,
                                 f"Block ({br},{bc}) outside band width 1")

    def test_sparse_internal_density(self):
        spec = _make_spec("sparse_internal", 400, 400, 42,
                          grid_r=4, grid_c=4, internal_density=0.1, num_active=4)
        mat = generate_one(spec)
        # With 4 blocks of 100x100 and 10% density, expect ~4000 entries
        self.assertGreater(len(mat.entries), 1000)
        self.assertLess(len(mat.entries), 8000)

    def test_seed_reproducibility(self):
        spec1 = _make_spec("random_block_occupancy", 200, 200, 99,
                           grid_r=4, grid_c=4, num_active=3)
        spec2 = _make_spec("random_block_occupancy", 200, 200, 99,
                           grid_r=4, grid_c=4, num_active=3)
        self.assertEqual(generate_one(spec1).entries, generate_one(spec2).entries)

    def test_bounds(self):
        spec = _make_spec("explicit_blocks", 300, 400, 1,
                          grid_r=3, grid_c=4, active_blocks=[[0, 0]])
        mat = generate_one(spec)
        for r, c, v in mat.entries:
            self.assertGreaterEqual(r, 0)
            self.assertLess(r, 300)
            self.assertGreaterEqual(c, 0)
            self.assertLess(c, 400)


if __name__ == "__main__":
    unittest.main()
