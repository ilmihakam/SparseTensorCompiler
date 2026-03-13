#!/usr/bin/env python3
"""Unit tests for F8: Template reuse / low-rank-ish family."""

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
        family="template_reuse",
        variant=variant,
        family_number="F8.x",
        rows=rows, cols=cols, seed=seed,
        tags=["test"],
        family_params=fp,
        postprocess={},
        value_mode="ones",
    )


class TestF08TemplateReuse(unittest.TestCase):
    def test_dense_columns(self):
        spec = _make_spec("dense_columns", 200, 200, 42,
                          num_dense=3, dense_density=0.5, sparse_nnz=1)
        mat = generate_one(spec)
        col_counts = {}
        for r, c, _ in mat.entries:
            col_counts[c] = col_counts.get(c, 0) + 1
        # 3 dense columns should have many more entries
        heavy_cols = sum(1 for n in col_counts.values() if n > 50)
        self.assertEqual(heavy_cols, 3)

    def test_dense_rows(self):
        spec = _make_spec("dense_rows", 200, 200, 42,
                          num_dense=3, dense_density=0.5, sparse_nnz=1)
        mat = generate_one(spec)
        row_counts = {}
        for r, c, _ in mat.entries:
            row_counts[r] = row_counts.get(r, 0) + 1
        heavy_rows = sum(1 for n in row_counts.values() if n > 50)
        self.assertEqual(heavy_rows, 3)

    def test_row_templates_limited_patterns(self):
        spec = _make_spec("row_templates", 200, 200, 42,
                          num_templates=4, template_nnz=15, noise_prob=0.0)
        mat = generate_one(spec)
        # Extract row patterns (ignoring noise)
        row_patterns = {}
        for r, c, _ in mat.entries:
            row_patterns.setdefault(r, set()).add(c)
        unique_patterns = set()
        for cols_set in row_patterns.values():
            unique_patterns.add(frozenset(cols_set))
        # With 4 templates and no noise, should have <=4 unique patterns
        self.assertLessEqual(len(unique_patterns), 4)

    def test_disjoint_row_groups(self):
        spec = _make_spec("disjoint_row_groups", 200, 200, 42,
                          num_groups=2, nnz_per_row=10)
        mat = generate_one(spec)
        # Group 0 (even rows) should use cols [0, 100), group 1 (odd) [100, 200)
        for r, c, _ in mat.entries:
            group = r % 2
            expected_left = group * 100
            expected_right = expected_left + 99
            self.assertGreaterEqual(c, expected_left,
                                    f"Row {r} group {group} col {c} below range")
            self.assertLessEqual(c, expected_right,
                                 f"Row {r} group {group} col {c} above range")

    def test_lowrank_plus_sparse(self):
        spec = _make_spec("lowrank_plus_sparse", 200, 200, 42,
                          k=3, dense_density=0.3, sparse_nnz_per_row=2)
        mat = generate_one(spec)
        self.assertGreater(len(mat.entries), 0)

    def test_seed_reproducibility(self):
        spec1 = _make_spec("dense_columns", 100, 100, 99,
                           num_dense=2, dense_density=0.5, sparse_nnz=1)
        spec2 = _make_spec("dense_columns", 100, 100, 99,
                           num_dense=2, dense_density=0.5, sparse_nnz=1)
        self.assertEqual(generate_one(spec1).entries, generate_one(spec2).entries)

    def test_bounds(self):
        spec = _make_spec("dense_columns", 100, 200, 42,
                          num_dense=2, dense_density=0.5, sparse_nnz=1)
        mat = generate_one(spec)
        for r, c, v in mat.entries:
            self.assertGreaterEqual(r, 0)
            self.assertLess(r, 100)
            self.assertGreaterEqual(c, 0)
            self.assertLess(c, 200)


if __name__ == "__main__":
    unittest.main()
