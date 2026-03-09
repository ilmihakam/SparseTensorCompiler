#!/usr/bin/env python3
"""Unit tests for analyze_loop_interchange and analyze_1d_blocking."""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


_ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "analysis"
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from analyze_loop_interchange import (
    MatrixData as IntMatrixData,
    ThresholdProfile as IntThresholdProfile,
    build_rules as int_build_rules,
    compute_row_analysis,
    parse_interchange_config,
)
from analyze_1d_blocking import (
    MatrixData as BlkMatrixData,
    ThresholdProfile as BlkThresholdProfile,
    build_rules as blk_build_rules,
    compute_block_analysis,
    parse_1d_block_config,
    DEFAULT_BLOCK_SIZE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_mtx(path: Path, rows: int, cols: int, coords: list) -> None:
    """Write a minimal MatrixMarket file (1-indexed coords)."""
    lines = [
        "%%MatrixMarket matrix coordinate real general",
        f"{rows} {cols} {len(coords)}",
    ]
    for r, c in coords:
        lines.append(f"{r + 1} {c + 1} 1.0")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_benchmark_csv(path: Path, rows: list) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_benchmark_row(
    matrix_name: str,
    config: str,
    avg_time_ms: str,
    format: str = "csr",
    kernel: str = "spmv",
) -> dict:
    return {
        "matrix_name": matrix_name,
        "format": format,
        "kernel": kernel,
        "impl": "ours",
        "config": config,
        "rows": "4",
        "cols": "8",
        "nnz": "6",
        "iterations": "5",
        "total_time_ms": str(float(avg_time_ms) * 5),
        "avg_time_ms": avg_time_ms,
        "min_time_ms": avg_time_ms,
        "max_time_ms": avg_time_ms,
        "stddev_ms": "0.0",
        "variance_pct": "0.0",
        "max_error": "0.0",
    }


# ---------------------------------------------------------------------------
# parse_interchange_config tests
# ---------------------------------------------------------------------------

class ParseInterchangeConfigTest(unittest.TestCase):
    def test_interchange_only_returns_true(self) -> None:
        self.assertTrue(parse_interchange_config("interchange_only"))

    def test_other_configs_return_false(self) -> None:
        for name in ("baseline", "block_only", "block_b32", "i_then_b", "all_I_THEN_B_b32",
                     "block2d_b16x16", "block_b16"):
            with self.subTest(config=name):
                self.assertFalse(parse_interchange_config(name))


# ---------------------------------------------------------------------------
# parse_1d_block_config tests
# ---------------------------------------------------------------------------

class Parse1dBlockConfigTest(unittest.TestCase):
    def test_block_only_returns_default(self) -> None:
        self.assertEqual(parse_1d_block_config("block_only"), DEFAULT_BLOCK_SIZE)

    def test_block_b_variants(self) -> None:
        self.assertEqual(parse_1d_block_config("block_b16"), 16)
        self.assertEqual(parse_1d_block_config("block_b32"), 32)
        self.assertEqual(parse_1d_block_config("block_b64"), 64)

    def test_other_configs_return_none(self) -> None:
        for name in ("baseline", "interchange_only", "block2d_b16x16", "i_then_b",
                     "all_I_THEN_B_b32", "block_b16x16"):
            with self.subTest(config=name):
                self.assertIsNone(parse_1d_block_config(name))


# ---------------------------------------------------------------------------
# compute_row_analysis tests
# ---------------------------------------------------------------------------

class ComputeRowAnalysisTest(unittest.TestCase):
    def _make_matrix(self) -> IntMatrixData:
        # 4 rows, 8 cols
        # Row 0: cols 0,1,2   (span=3, span_frac=3/8, density=1.0, runs=1)
        # Row 1: cols 0,4,7   (span=8, span_frac=1.0, density=3/8, runs=3)
        # Row 2: cols 3,4     (span=2, span_frac=0.25, density=1.0, runs=1)
        # Row 3: empty
        coords = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 4), (1, 7), (2, 3), (2, 4)]
        return IntMatrixData(rows=4, cols=8, coords=coords)

    def test_as_is_row_count(self) -> None:
        matrix = self._make_matrix()
        thresholds = IntThresholdProfile(
            col_span_frac_min=0.1, row_cv_max=3.0,
            use_col_span_frac=True, use_row_cv=True,
        )
        rules = int_build_rules(thresholds)
        per_row, agg = compute_row_analysis(matrix, "as_is", thresholds, rules)
        self.assertEqual(len(per_row), 4)

    def test_col_span_frac_values(self) -> None:
        matrix = self._make_matrix()
        thresholds = IntThresholdProfile(
            col_span_frac_min=0.5, row_cv_max=10.0,
            use_col_span_frac=True, use_row_cv=False,
        )
        rules = int_build_rules(thresholds)
        per_row, agg = compute_row_analysis(matrix, "as_is", thresholds, rules)
        # Row 1 has col_span_frac = 1.0 >= 0.5 → passes
        self.assertTrue(per_row[1]["passes_col_span_frac"])
        # Row 0 has col_span_frac = 3/8 = 0.375 < 0.5 → fails
        self.assertFalse(per_row[0]["passes_col_span_frac"])
        # Row 3 (empty) has col_span_frac = 0.0 → fails
        self.assertFalse(per_row[3]["passes_col_span_frac"])

    def test_transpose_swaps_dimensions(self) -> None:
        matrix = self._make_matrix()
        thresholds = IntThresholdProfile(
            col_span_frac_min=0.1, row_cv_max=10.0,
            use_col_span_frac=True, use_row_cv=False,
        )
        rules = int_build_rules(thresholds)
        per_row_t, agg_t = compute_row_analysis(matrix, "transpose", thresholds, rules)
        # Transpose: 8 rows (cols become rows), 4 cols (rows become cols)
        self.assertEqual(len(per_row_t), 8)

    def test_aggregate_has_expected_keys(self) -> None:
        matrix = self._make_matrix()
        thresholds = IntThresholdProfile(
            col_span_frac_min=0.1, row_cv_max=3.0,
            use_col_span_frac=True, use_row_cv=True,
        )
        rules = int_build_rules(thresholds)
        _, agg = compute_row_analysis(matrix, "as_is", thresholds, rules)
        for key in ("row_nnz_cv", "col_span_frac_mean", "col_span_frac_p50",
                    "row_pass_frac", "interchange_beneficial_by_cv"):
            self.assertIn(key, agg, f"Missing aggregate key: {key}")


# ---------------------------------------------------------------------------
# compute_block_analysis tests
# ---------------------------------------------------------------------------

class ComputeBlockAnalysisTest(unittest.TestCase):
    def _make_matrix(self) -> BlkMatrixData:
        # 6 rows, 8 cols, block_size=3
        # Block 0 (rows 0-2): cols accessed = {0,1,2,1,3}
        # Block 1 (rows 3-5): cols accessed = {5,6,7}
        coords = [
            (0, 0), (0, 1), (1, 2), (1, 1), (2, 3),  # block 0: 5 nnz, 4 unique cols
            (3, 5), (4, 6), (5, 7),                    # block 1: 3 nnz, 3 unique cols
        ]
        return BlkMatrixData(rows=6, cols=8, coords=coords)

    def test_block_count(self) -> None:
        matrix = self._make_matrix()
        thresholds = BlkThresholdProfile(
            col_reuse_min=1.0, col_coverage_max=0.9,
            use_col_reuse=True, use_col_coverage=True,
        )
        rules = blk_build_rules(thresholds)
        per_block, agg = compute_block_analysis(matrix, 3, "as_is", thresholds, rules)
        self.assertEqual(len(per_block), 2)

    def test_col_reuse_values(self) -> None:
        matrix = self._make_matrix()
        thresholds = BlkThresholdProfile(
            col_reuse_min=1.0, col_coverage_max=1.0,
            use_col_reuse=True, use_col_coverage=False,
        )
        rules = blk_build_rules(thresholds)
        per_block, agg = compute_block_analysis(matrix, 3, "as_is", thresholds, rules)
        # Block 0: 5 nnz / 4 unique = 1.25
        self.assertAlmostEqual(per_block[0]["col_reuse"], 5 / 4, places=5)
        # Block 1: 3 nnz / 3 unique = 1.0
        self.assertAlmostEqual(per_block[1]["col_reuse"], 1.0, places=5)
        # Both pass col_reuse >= 1.0
        self.assertTrue(per_block[0]["passes_col_reuse"])
        self.assertTrue(per_block[1]["passes_col_reuse"])

    def test_col_coverage_threshold(self) -> None:
        matrix = self._make_matrix()
        thresholds = BlkThresholdProfile(
            col_reuse_min=0.0, col_coverage_max=0.4,
            use_col_reuse=False, use_col_coverage=True,
        )
        rules = blk_build_rules(thresholds)
        per_block, agg = compute_block_analysis(matrix, 3, "as_is", thresholds, rules)
        # Block 0: 4/8 = 0.5 > 0.4 → fails
        self.assertFalse(per_block[0]["passes_col_coverage"])
        # Block 1: 3/8 = 0.375 <= 0.4 → passes
        self.assertTrue(per_block[1]["passes_col_coverage"])

    def test_aggregate_has_expected_keys(self) -> None:
        matrix = self._make_matrix()
        thresholds = BlkThresholdProfile(
            col_reuse_min=1.5, col_coverage_max=0.8,
            use_col_reuse=True, use_col_coverage=True,
        )
        rules = blk_build_rules(thresholds)
        _, agg = compute_block_analysis(matrix, 3, "as_is", thresholds, rules)
        for key in ("col_reuse_mean", "col_reuse_p50", "col_coverage_mean",
                    "blocks_total", "blocks_good", "r_block_all", "row_nnz_cv"):
            self.assertIn(key, agg, f"Missing aggregate key: {key}")

    def test_transpose_swaps_dimensions(self) -> None:
        matrix = self._make_matrix()
        thresholds = BlkThresholdProfile(
            col_reuse_min=1.0, col_coverage_max=1.0,
            use_col_reuse=True, use_col_coverage=True,
        )
        rules = blk_build_rules(thresholds)
        per_block_t, _ = compute_block_analysis(matrix, 3, "transpose", thresholds, rules)
        # 8 cols become rows → ceil(8/3) = 3 blocks
        self.assertEqual(len(per_block_t), 3)


# ---------------------------------------------------------------------------
# End-to-end script tests
# ---------------------------------------------------------------------------

class InterchangeScriptTest(unittest.TestCase):
    def _make_toy_matrix(self, tmp: Path) -> Path:
        canonical = tmp / "canonical"
        canonical.mkdir(parents=True, exist_ok=True)
        mtx = canonical / "toy.mtx"
        coords = [(0, 0), (0, 4), (0, 7), (1, 1), (2, 2), (3, 6)]
        write_mtx(mtx, 4, 8, coords)
        return canonical

    def test_produces_three_csvs(self) -> None:
        import subprocess
        script = _ANALYSIS_DIR / "analyze_loop_interchange.py"
        with tempfile.TemporaryDirectory(prefix="test_interchange_") as tmp_dir:
            tmp = Path(tmp_dir)
            canonical_dir = self._make_toy_matrix(tmp)
            bench_csv = tmp / "bench.csv"
            write_benchmark_csv(bench_csv, [
                make_benchmark_row("toy", "baseline", "2.0"),
                make_benchmark_row("toy", "interchange_only", "1.0"),
            ])

            dataset = tmp / "dataset.csv"
            rows_out = tmp / "rows.csv"
            thresholds_out = tmp / "thresholds.csv"
            result = subprocess.run(
                [
                    sys.executable, str(script),
                    "--csv", str(bench_csv),
                    "--canonical-dir", str(canonical_dir),
                    "--output", str(dataset),
                    "--rows-output", str(rows_out),
                    "--thresholds-output", str(thresholds_out),
                ],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            with dataset.open(encoding="utf-8", newline="") as f:
                ds_rows = list(csv.DictReader(f))
            self.assertEqual(len(ds_rows), 1)
            self.assertEqual(ds_rows[0]["config_kind"], "interchange")
            self.assertAlmostEqual(float(ds_rows[0]["speedup_vs_baseline"]), 2.0, places=5)

            with rows_out.open(encoding="utf-8", newline="") as f:
                row_rows = list(csv.DictReader(f))
            # as_is: 4 rows + transpose: 8 rows (cols become rows) = 12
            self.assertEqual(len(row_rows), 12)

            with thresholds_out.open(encoding="utf-8", newline="") as f:
                thresh_rows = list(csv.DictReader(f))
            self.assertEqual(len(thresh_rows), 1)


class Blocking1dScriptTest(unittest.TestCase):
    def _make_toy_matrix(self, tmp: Path) -> Path:
        canonical = tmp / "canonical"
        canonical.mkdir(parents=True, exist_ok=True)
        mtx = canonical / "toy.mtx"
        coords = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 3), (3, 4)]
        write_mtx(mtx, 4, 8, coords)
        return canonical

    def test_block_only_and_block_b_configs(self) -> None:
        import subprocess
        script = _ANALYSIS_DIR / "analyze_1d_blocking.py"
        with tempfile.TemporaryDirectory(prefix="test_1d_block_") as tmp_dir:
            tmp = Path(tmp_dir)
            canonical_dir = self._make_toy_matrix(tmp)
            bench_csv = tmp / "bench.csv"
            write_benchmark_csv(bench_csv, [
                make_benchmark_row("toy", "baseline", "4.0"),
                make_benchmark_row("toy", "block_only", "2.0"),
                make_benchmark_row("toy", "block_b16", "2.5"),
                make_benchmark_row("toy", "block_b64", "1.5"),
                # should be filtered out
                make_benchmark_row("toy", "interchange_only", "1.8"),
            ])

            dataset = tmp / "dataset.csv"
            blocks_out = tmp / "blocks.csv"
            thresholds_out = tmp / "thresholds.csv"
            result = subprocess.run(
                [
                    sys.executable, str(script),
                    "--csv", str(bench_csv),
                    "--canonical-dir", str(canonical_dir),
                    "--output", str(dataset),
                    "--blocks-output", str(blocks_out),
                    "--thresholds-output", str(thresholds_out),
                ],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)

            with dataset.open(encoding="utf-8", newline="") as f:
                ds_rows = list(csv.DictReader(f))
            # 3 1d-blocking configs × 1 matrix = 3 rows
            self.assertEqual(len(ds_rows), 3)
            config_kinds = {r["config_kind"] for r in ds_rows}
            self.assertEqual(config_kinds, {"block_1d"})

            block_sizes = {r["block_size"] for r in ds_rows}
            self.assertIn("32", block_sizes)  # block_only
            self.assertIn("16", block_sizes)
            self.assertIn("64", block_sizes)

            with thresholds_out.open(encoding="utf-8", newline="") as f:
                thresh_rows = list(csv.DictReader(f))
            self.assertEqual(len(thresh_rows), 1)
            self.assertEqual(thresh_rows[0]["col_reuse_min"], "1.5")

    def test_help_flag(self) -> None:
        import subprocess
        script = _ANALYSIS_DIR / "analyze_1d_blocking.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("1D blocking", result.stdout)


if __name__ == "__main__":
    unittest.main()
