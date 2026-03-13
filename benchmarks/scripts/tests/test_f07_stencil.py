#!/usr/bin/env python3
"""Unit tests for F7: Stencil / grid-like family."""

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
        family="stencil",
        variant=variant,
        family_number="F7.x",
        rows=rows, cols=cols, seed=seed,
        tags=["test"],
        family_params=fp,
        postprocess={},
        value_mode="ones",
    )


class TestF07Stencil(unittest.TestCase):
    def test_2d_5pt_size(self):
        spec = _make_spec("2d_5pt", 400, 400, 1, grid_n=20, grid_m=20)
        mat = generate_one(spec)
        # 20x20 grid = 400 nodes, each interior node has 5 entries
        self.assertEqual(mat.rows, 400)
        self.assertEqual(mat.cols, 400)
        # Interior: (18*18)*5 = 1620, boundary adds less
        # Total should be near 5*400 - boundary corrections ~ 1880
        self.assertGreater(len(mat.entries), 1500)

    def test_2d_5pt_degree(self):
        spec = _make_spec("2d_5pt", 100, 100, 1, grid_n=10, grid_m=10)
        mat = generate_one(spec)
        row_nnz = {}
        for r, c, _ in mat.entries:
            row_nnz[r] = row_nnz.get(r, 0) + 1
        # Interior nodes should have exactly 5 neighbors
        interior_count = sum(1 for n in row_nnz.values() if n == 5)
        # 8*8 = 64 interior nodes
        self.assertEqual(interior_count, 64)

    def test_2d_9pt_denser(self):
        spec5 = _make_spec("2d_5pt", 100, 100, 1, grid_n=10, grid_m=10)
        spec9 = _make_spec("2d_9pt", 100, 100, 1, grid_n=10, grid_m=10)
        mat5 = generate_one(spec5)
        mat9 = generate_one(spec9)
        self.assertGreater(len(mat9.entries), len(mat5.entries))

    def test_3d_7pt(self):
        spec = _make_spec("3d_7pt", 1000, 1000, 1, grid_x=10, grid_y=10, grid_z=10)
        mat = generate_one(spec)
        self.assertEqual(mat.rows, 1000)
        self.assertGreater(len(mat.entries), 5000)

    def test_2d_with_holes_fewer_entries(self):
        spec_full = _make_spec("2d_5pt", 100, 100, 1, grid_n=10, grid_m=10)
        spec_holes = _make_spec("2d_with_holes", 100, 100, 1, grid_n=10, grid_m=10, hole_fraction=0.2)
        mat_full = generate_one(spec_full)
        mat_holes = generate_one(spec_holes)
        self.assertLess(len(mat_holes.entries), len(mat_full.entries))

    def test_seed_reproducibility(self):
        spec1 = _make_spec("2d_5pt", 100, 100, 42, grid_n=10, grid_m=10)
        spec2 = _make_spec("2d_5pt", 100, 100, 42, grid_n=10, grid_m=10)
        self.assertEqual(generate_one(spec1).entries, generate_one(spec2).entries)

    def test_locally_refined_denser(self):
        spec_base = _make_spec("2d_5pt", 100, 100, 1, grid_n=10, grid_m=10)
        spec_refined = _make_spec("locally_refined", 100, 100, 1, grid_n=10, grid_m=10)
        mat_base = generate_one(spec_base)
        mat_refined = generate_one(spec_refined)
        self.assertGreater(len(mat_refined.entries), len(mat_base.entries))


if __name__ == "__main__":
    unittest.main()
