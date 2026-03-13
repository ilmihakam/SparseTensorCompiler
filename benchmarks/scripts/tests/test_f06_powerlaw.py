#!/usr/bin/env python3
"""Unit tests for F6: Power-law family."""

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
        family="powerlaw",
        variant=variant,
        family_number="F6.x",
        rows=rows, cols=cols, seed=seed,
        tags=["test"],
        family_params=fp,
        postprocess={},
        value_mode="ones",
    )


class TestF06Powerlaw(unittest.TestCase):
    def test_row_zipf_produces_entries(self):
        spec = _make_spec("row_zipf", 500, 500, 42, alpha=1.2, avg_nnz_row=10)
        mat = generate_one(spec)
        self.assertGreater(len(mat.entries), 0)

    def test_row_zipf_has_high_row_cv(self):
        spec = _make_spec("row_zipf", 1000, 1000, 42, alpha=1.2, avg_nnz_row=10)
        mat = generate_one(spec)
        row_counts = {}
        for r, c, _ in mat.entries:
            row_counts[r] = row_counts.get(r, 0) + 1
        counts = list(row_counts.values())
        if counts:
            import math
            mean = sum(counts) / len(counts)
            var = sum((x - mean)**2 for x in counts) / len(counts)
            cv = math.sqrt(var) / mean if mean > 0 else 0
            # Power-law should produce high CV
            self.assertGreater(cv, 0.5)

    def test_super_rows(self):
        spec = _make_spec("super_rows", 200, 200, 42,
                          num_super=3, super_nnz=100, sparse_nnz=2)
        mat = generate_one(spec)
        row_counts = {}
        for r, c, _ in mat.entries:
            row_counts[r] = row_counts.get(r, 0) + 1
        # Should have exactly 3 rows with ~100 entries
        heavy_rows = sum(1 for n in row_counts.values() if n > 50)
        self.assertEqual(heavy_rows, 3)

    def test_col_zipf(self):
        spec = _make_spec("col_zipf", 500, 500, 42, alpha=1.2, avg_nnz_row=10)
        mat = generate_one(spec)
        self.assertGreater(len(mat.entries), 0)

    def test_banded_powerlaw(self):
        spec = _make_spec("banded_powerlaw", 500, 500, 42,
                          alpha=1.2, avg_nnz_row=10, bandwidth=50)
        mat = generate_one(spec)
        # Most entries should be near diagonal
        near_diag = sum(1 for r, c, _ in mat.entries if abs(r - c) <= 55)
        self.assertGreater(near_diag, len(mat.entries) * 0.8)

    def test_seed_reproducibility(self):
        spec1 = _make_spec("row_zipf", 200, 200, 99, alpha=1.2, avg_nnz_row=10)
        spec2 = _make_spec("row_zipf", 200, 200, 99, alpha=1.2, avg_nnz_row=10)
        self.assertEqual(generate_one(spec1).entries, generate_one(spec2).entries)

    def test_bounds(self):
        spec = _make_spec("row_zipf", 100, 200, 42, alpha=1.2, avg_nnz_row=5)
        mat = generate_one(spec)
        for r, c, v in mat.entries:
            self.assertGreaterEqual(r, 0)
            self.assertLess(r, 100)
            self.assertGreaterEqual(c, 0)
            self.assertLess(c, 200)


if __name__ == "__main__":
    unittest.main()
