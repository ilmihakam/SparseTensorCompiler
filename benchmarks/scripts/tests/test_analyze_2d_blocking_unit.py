#!/usr/bin/env python3

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
_ANALYSIS_DIR = _SCRIPTS_DIR / "analysis"
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from analyze_2d_blocking import parse_2d_config


SCRIPT_DIR = _ANALYSIS_DIR


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Analyze2DBlockingUnitTest(unittest.TestCase):
    def test_parse_2d_config(self) -> None:
        self.assertEqual(
            parse_2d_config("block2d_b16x32"),
            {
                "is_2d": True,
                "config_kind": "block2d",
                "tile_i": 16,
                "tile_j": 32,
                "order": "",
            },
        )
        self.assertEqual(
            parse_2d_config("all2d_I_THEN_B_b32x64"),
            {
                "is_2d": True,
                "config_kind": "all2d",
                "tile_i": 32,
                "tile_j": 64,
                "order": "I_THEN_B",
            },
        )
        self.assertIsNone(parse_2d_config("baseline"))

    def test_analysis_script_outputs_dataset_tiles_and_threshold_flip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="analyze_2d_blocking_") as tmp_dir:
            tmp = Path(tmp_dir)
            canonical_dir = tmp / "canonical"
            canonical_dir.mkdir(parents=True, exist_ok=True)
            matrix_path = canonical_dir / "toy.mtx"
            matrix_path.write_text(
                "\n".join(
                    [
                        "%%MatrixMarket matrix coordinate real general",
                        "% toy matrix",
                        "2 20 9",
                        "1 1 1.0",
                        "1 2 1.0",
                        "1 4 1.0",
                        "1 5 1.0",
                        "2 1 1.0",
                        "2 2 1.0",
                        "2 4 1.0",
                        "2 5 1.0",
                        "1 20 1.0",
                    ]
                ),
                encoding="utf-8",
            )

            spmm_csv = tmp / "benchmark_spmm.csv"
            with spmm_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "matrix_name",
                        "format",
                        "kernel",
                        "impl",
                        "config",
                        "rows",
                        "cols",
                        "nnz",
                        "N",
                        "iterations",
                        "total_time_ms",
                        "avg_time_ms",
                        "min_time_ms",
                        "max_time_ms",
                        "stddev_ms",
                        "variance_pct",
                        "max_error",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "matrix_name": "toy",
                        "format": "csr",
                        "kernel": "spmm",
                        "impl": "ours",
                        "config": "baseline",
                        "rows": "2",
                        "cols": "20",
                        "nnz": "9",
                        "N": "32",
                        "iterations": "10",
                        "total_time_ms": "10.0",
                        "avg_time_ms": "1.0",
                        "min_time_ms": "0.9",
                        "max_time_ms": "1.1",
                        "stddev_ms": "0.05",
                        "variance_pct": "5.0",
                        "max_error": "0.0",
                    }
                )
                writer.writerow(
                    {
                        "matrix_name": "toy",
                        "format": "csr",
                        "kernel": "spmm",
                        "impl": "ours",
                        "config": "block2d_b2x10",
                        "rows": "2",
                        "cols": "20",
                        "nnz": "9",
                        "N": "32",
                        "iterations": "10",
                        "total_time_ms": "5.0",
                        "avg_time_ms": "0.5",
                        "min_time_ms": "0.45",
                        "max_time_ms": "0.55",
                        "stddev_ms": "0.03",
                        "variance_pct": "6.0",
                        "max_error": "0.0",
                    }
                )
                writer.writerow(
                    {
                        "matrix_name": "toy",
                        "format": "csc",
                        "kernel": "spmm",
                        "impl": "ours",
                        "config": "baseline",
                        "rows": "2",
                        "cols": "20",
                        "nnz": "9",
                        "N": "32",
                        "iterations": "10",
                        "total_time_ms": "12.0",
                        "avg_time_ms": "1.2",
                        "min_time_ms": "1.1",
                        "max_time_ms": "1.3",
                        "stddev_ms": "0.05",
                        "variance_pct": "4.5",
                        "max_error": "0.0",
                    }
                )
                writer.writerow(
                    {
                        "matrix_name": "toy",
                        "format": "csc",
                        "kernel": "spmm",
                        "impl": "ours",
                        "config": "block2d_b2x10",
                        "rows": "2",
                        "cols": "20",
                        "nnz": "9",
                        "N": "32",
                        "iterations": "10",
                        "total_time_ms": "6.0",
                        "avg_time_ms": "0.6",
                        "min_time_ms": "0.55",
                        "max_time_ms": "0.65",
                        "stddev_ms": "0.03",
                        "variance_pct": "5.5",
                        "max_error": "0.0",
                    }
                )

            dataset_path = tmp / "dataset.csv"
            tiles_path = tmp / "tiles.csv"
            thresholds_path = tmp / "thresholds.csv"
            base_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "analyze_2d_blocking.py"),
                "--spmm-csv",
                str(spmm_csv),
                "--sddmm-csv",
                str(tmp / "missing_sddmm.csv"),
                "--canonical-dir",
                str(canonical_dir),
            ]
            cmd = [
                *base_cmd,
                "--output",
                str(dataset_path),
                "--tiles-output",
                str(tiles_path),
                "--thresholds-output",
                str(thresholds_path),
            ]
            subprocess.run(cmd, cwd=SCRIPT_DIR, check=True, capture_output=True, text=True)

            dataset_rows = read_rows(dataset_path)
            self.assertEqual(len(dataset_rows), 2)
            rows_by_format = {row["format"]: row for row in dataset_rows}
            self.assertEqual(rows_by_format["csr"]["speedup_vs_baseline"], "2.0")
            self.assertEqual(rows_by_format["csr"]["r_tile_nonempty_selected"], "0.5")
            self.assertEqual(rows_by_format["csr"]["selected_metric_mode"], "as_is")
            self.assertEqual(rows_by_format["csr"]["tiles_empty_selected"], "0.0")
            self.assertEqual(rows_by_format["csr"]["empty_tile_frac_selected"], "0.0")
            self.assertEqual(rows_by_format["csc"]["speedup_vs_baseline"], "2.0")
            self.assertEqual(rows_by_format["csc"]["selected_metric_mode"], "transpose")
            self.assertEqual(
                rows_by_format["csc"]["r_tile_nonempty_selected"],
                rows_by_format["csc"]["r_tile_nonempty_transpose"],
            )
            self.assertEqual(
                rows_by_format["csc"]["empty_tile_frac_selected"],
                rows_by_format["csc"]["empty_tile_frac_transpose"],
            )

            tile_rows = read_rows(tiles_path)
            as_is_rows = [row for row in tile_rows if row["mode"] == "as_is"]
            self.assertEqual(len(as_is_rows), 2)
            self.assertEqual({row["is_good"] for row in as_is_rows}, {"0", "1"})

            threshold_rows = read_rows(thresholds_path)
            self.assertEqual(len(threshold_rows), 1)
            self.assertEqual(threshold_rows[0]["density_min"], "0.05")

            high_density_dataset = tmp / "dataset_high_density.csv"
            subprocess.run(
                [
                    *base_cmd,
                    "--output",
                    str(high_density_dataset),
                    "--tiles-output",
                    str(tmp / "tiles_high_density.csv"),
                    "--thresholds-output",
                    str(tmp / "thresholds_high_density.csv"),
                    "--density-min",
                    "0.6",
                ],
                cwd=SCRIPT_DIR,
                check=True,
                capture_output=True,
                text=True,
            )
            high_density_rows = read_rows(high_density_dataset)
            high_density_by_format = {row["format"]: row for row in high_density_rows}
            self.assertEqual(high_density_by_format["csr"]["r_tile_nonempty_selected"], "0.0")


if __name__ == "__main__":
    unittest.main()
