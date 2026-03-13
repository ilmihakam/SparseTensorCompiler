#!/usr/bin/env python3
"""Unit tests for F1: Uniform random family."""

from __future__ import annotations

import random
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
        family="uniform_random",
        variant=variant,
        family_number="F1.x",
        rows=rows, cols=cols, seed=seed,
        tags=["test"],
        family_params=fp,
        postprocess={},
        value_mode="ones",
    )


class TestF01UniformRandom(unittest.TestCase):
    def test_full_random_produces_entries(self):
        spec = _make_spec("full_random", 100, 100, 42, density=0.01)
        mat = generate_one(spec)
        self.assertEqual(mat.rows, 100)
        self.assertEqual(mat.cols, 100)
        self.assertGreater(len(mat.entries), 0)

    def test_banded_random_within_band(self):
        spec = _make_spec("banded_random", 200, 200, 42, density=0.05, bandwidth=20)
        mat = generate_one(spec)
        self.assertGreater(len(mat.entries), 0)
        # With banded support, most entries should be near diagonal
        near_diag = sum(1 for r, c, _ in mat.entries if abs(r - c) <= 25)
        self.assertGreater(near_diag, len(mat.entries) * 0.5)

    def test_tall_skinny(self):
        spec = _make_spec("tall_skinny", 1000, 100, 42, avg_nnz_row=5)
        mat = generate_one(spec)
        self.assertEqual(mat.rows, 1000)
        self.assertEqual(mat.cols, 100)
        self.assertGreater(len(mat.entries), 0)

    def test_seed_reproducibility(self):
        spec1 = _make_spec("full_random", 100, 100, 99, density=0.02)
        spec2 = _make_spec("full_random", 100, 100, 99, density=0.02)
        mat1 = generate_one(spec1)
        mat2 = generate_one(spec2)
        self.assertEqual(mat1.entries, mat2.entries)

    def test_different_seeds_differ(self):
        spec1 = _make_spec("full_random", 100, 100, 1, density=0.02)
        spec2 = _make_spec("full_random", 100, 100, 2, density=0.02)
        mat1 = generate_one(spec1)
        mat2 = generate_one(spec2)
        self.assertNotEqual(mat1.entries, mat2.entries)

    def test_bounds_correctness(self):
        spec = _make_spec("full_random", 50, 80, 42, density=0.05)
        mat = generate_one(spec)
        for r, c, v in mat.entries:
            self.assertGreaterEqual(r, 0)
            self.assertLess(r, 50)
            self.assertGreaterEqual(c, 0)
            self.assertLess(c, 80)


if __name__ == "__main__":
    unittest.main()
