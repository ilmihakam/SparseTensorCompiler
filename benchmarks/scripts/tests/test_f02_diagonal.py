#!/usr/bin/env python3
"""Unit tests for F2: Diagonal and narrow-banded family."""

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
        family="diagonal",
        variant=variant,
        family_number="F2.x",
        rows=rows, cols=cols, seed=seed,
        tags=["test"],
        family_params=fp,
        postprocess={},
        value_mode="ones",
    )


class TestF02Diagonal(unittest.TestCase):
    def test_strict_diagonal(self):
        spec = _make_spec("strict_diagonal", 100, 100, 1)
        mat = generate_one(spec)
        self.assertEqual(len(mat.entries), 100)
        for r, c, v in mat.entries:
            self.assertEqual(r, c, "Strict diagonal entry should be on main diagonal")

    def test_tridiagonal(self):
        spec = _make_spec("tridiagonal", 100, 100, 1)
        mat = generate_one(spec)
        for r, c, v in mat.entries:
            self.assertLessEqual(abs(r - c), 1, "Tridiagonal entry should be within 1 of diagonal")
        # Should have ~3 entries per interior row, 2 for boundary
        self.assertGreater(len(mat.entries), 200)

    def test_k_diagonal(self):
        spec = _make_spec("k_diagonal", 200, 200, 1, k=10)
        mat = generate_one(spec)
        for r, c, v in mat.entries:
            self.assertLessEqual(abs(r - c), 5, "k=10 diagonal entries within offset 5")

    def test_band_with_spikes(self):
        spec = _make_spec("band_with_spikes", 200, 200, 1, k=5, spike_count=20)
        mat = generate_one(spec)
        spike_count = sum(1 for r, c, _ in mat.entries if abs(r - c) > 5)
        self.assertGreater(spike_count, 0, "Should have spikes far from band")

    def test_block_diagonal_dense(self):
        spec = _make_spec("block_diagonal_dense", 400, 400, 1, block_size=100)
        mat = generate_one(spec)
        # 4 blocks of 100x100 = up to 40000 entries
        self.assertGreater(len(mat.entries), 10000)
        # All entries should be within diagonal blocks
        for r, c, _ in mat.entries:
            block_r = r // 100
            block_c = c // 100
            self.assertEqual(block_r, block_c, "Block diagonal entry should be in diagonal block")

    def test_seed_reproducibility(self):
        spec1 = _make_spec("strict_diagonal", 50, 50, 99)
        spec2 = _make_spec("strict_diagonal", 50, 50, 99)
        self.assertEqual(generate_one(spec1).entries, generate_one(spec2).entries)

    def test_narrow_band_tall(self):
        spec = _make_spec("narrow_band_tall", 1000, 100, 1, k=3)
        mat = generate_one(spec)
        self.assertGreater(len(mat.entries), 0)
        for r, c, _ in mat.entries:
            self.assertGreaterEqual(r, 0)
            self.assertLess(r, 1000)
            self.assertGreaterEqual(c, 0)
            self.assertLess(c, 100)

    def test_block_diagonal_with_offdiag(self):
        spec = _make_spec("block_diagonal_with_offdiag", 400, 400, 1,
                          block_size=100, offdiag_count=2, offdiag_size=20)
        mat = generate_one(spec)
        # Should have entries in off-diagonal regions
        offdiag = sum(1 for r, c, _ in mat.entries if r // 100 != c // 100)
        self.assertGreater(offdiag, 0, "Should have off-diagonal block entries")


if __name__ == "__main__":
    unittest.main()
