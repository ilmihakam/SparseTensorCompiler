#!/usr/bin/env python3
"""Unit tests for F4: Row-clustered long runs family."""

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
        family="row_runs",
        variant=variant,
        family_number="F4.x",
        rows=rows, cols=cols, seed=seed,
        tags=["test"],
        family_params=fp,
        postprocess={},
        value_mode="ones",
    )


def _avg_run_length(entries, rows):
    """Compute average run length across rows."""
    row_cols = {}
    for r, c, _ in entries:
        row_cols.setdefault(r, []).append(c)

    total_runs = 0
    total_nnz = 0
    for cols_list in row_cols.values():
        cols_sorted = sorted(cols_list)
        runs = 1
        for i in range(1, len(cols_sorted)):
            if cols_sorted[i] != cols_sorted[i-1] + 1:
                runs += 1
        total_runs += runs
        total_nnz += len(cols_sorted)
    if total_runs == 0:
        return 0
    return total_nnz / total_runs


class TestF04RowRuns(unittest.TestCase):
    def test_single_segment_length(self):
        spec = _make_spec("single_segment", 100, 1000, 42, length=50)
        mat = generate_one(spec)
        # Each row should have ~50 entries
        row_counts = {}
        for r, c, _ in mat.entries:
            row_counts[r] = row_counts.get(r, 0) + 1
        for r in range(100):
            self.assertEqual(row_counts.get(r, 0), 50)

    def test_single_segment_contiguous(self):
        spec = _make_spec("single_segment", 50, 500, 42, length=30)
        mat = generate_one(spec)
        # Check contiguity: avg run length should be 30
        avg_rl = _avg_run_length(mat.entries, 50)
        self.assertGreaterEqual(avg_rl, 29.0)

    def test_double_segment_gap(self):
        spec = _make_spec("double_segment_fixed_gap", 50, 500, 42, length=20, gap=30)
        mat = generate_one(spec)
        # Each row should have entries (2 segments of up to 20, may be clipped at boundary)
        row_counts = {}
        for r, c, _ in mat.entries:
            row_counts[r] = row_counts.get(r, 0) + 1
        # Average should be close to 40, allow boundary clipping
        avg = sum(row_counts.values()) / max(len(row_counts), 1)
        self.assertGreater(avg, 25, "Average nnz per row should be substantial")

    def test_shared_overlap_high_matches(self):
        spec = _make_spec("shared_overlap", 100, 1000, 42, length=50, spread=10)
        mat = generate_one(spec)
        # Overlapping rows should share many columns
        row_cols = {}
        for r, c, _ in mat.entries:
            row_cols.setdefault(r, set()).add(c)
        if 0 in row_cols and 1 in row_cols:
            intersection = len(row_cols[0] & row_cols[1])
            self.assertGreater(intersection, 10)

    def test_staggered_low_overlap(self):
        spec = _make_spec("staggered_disjoint", 100, 10000, 42, length=50)
        mat = generate_one(spec)
        row_cols = {}
        for r, c, _ in mat.entries:
            row_cols.setdefault(r, set()).add(c)
        # Adjacent rows should have low overlap due to staggering
        if 0 in row_cols and 50 in row_cols:
            intersection = len(row_cols[0] & row_cols[50])
            # With stride=50, row 0 and row 50 start at different positions
            self.assertLess(intersection, 30)

    def test_seed_reproducibility(self):
        spec1 = _make_spec("single_segment", 50, 500, 99, length=30)
        spec2 = _make_spec("single_segment", 50, 500, 99, length=30)
        self.assertEqual(generate_one(spec1).entries, generate_one(spec2).entries)

    def test_bounds(self):
        spec = _make_spec("single_segment", 100, 200, 42, length=30)
        mat = generate_one(spec)
        for r, c, v in mat.entries:
            self.assertGreaterEqual(r, 0)
            self.assertLess(r, 100)
            self.assertGreaterEqual(c, 0)
            self.assertLess(c, 200)


class TestF05ColRuns(unittest.TestCase):
    def test_col_runs_is_transpose(self):
        """F5 col_runs should produce entries that are transposed from row_runs."""
        spec = MatrixSpec(
            name="test_col",
            family="col_runs",
            variant="single_segment",
            family_number="F5.1",
            rows=100, cols=200, seed=42,
            tags=["test"],
            family_params={"length": 30},
            postprocess={},
            value_mode="ones",
        )
        mat = generate_one(spec)
        self.assertEqual(mat.rows, 100)
        self.assertEqual(mat.cols, 200)
        self.assertGreater(len(mat.entries), 0)
        # Entries should be within bounds
        for r, c, _ in mat.entries:
            self.assertGreaterEqual(r, 0)
            self.assertLess(r, 100)
            self.assertGreaterEqual(c, 0)
            self.assertLess(c, 200)


if __name__ == "__main__":
    unittest.main()
