#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
_UTILS_DIR = SCRIPT_DIR.parent / "utils"
GENERATOR = _UTILS_DIR / "generate_matrices.py"


class MatrixGeneratorUnitTest(unittest.TestCase):
    def test_generate_manifest_and_pairs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="matrix_gen_unit_") as tmp_dir:
            tmp = Path(tmp_dir)
            spec = tmp / "spec.json"
            out_root = tmp / "custom"
            spec.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "matrices": [
                            {
                                "name": "m_a",
                                "generator": "structured_random_v1",
                                "rows": 64,
                                "cols": 64,
                                "seed": 1,
                                "nnz": {
                                    "mode": "density",
                                    "density": 0.02,
                                    "row_distribution": "lognormal",
                                    "row_cv": 1.2,
                                },
                                "support": {"mode": "banded", "bandwidth": 8},
                                "clustering": {"mode": "runs", "avg_run_length": 3.0, "avg_gap": 4.0},
                                "columns": {"mode": "uniform"},
                                "inter_row_similarity": {"mode": "window_share", "window": 4, "share_prob": 0.2},
                                "block_structure": {"enabled": False},
                            },
                            {
                                "name": "m_b",
                                "generator": "structured_random_v1",
                                "rows": 64,
                                "cols": 64,
                                "seed": 2,
                                "nnz": {"mode": "density", "density": 0.03},
                                "support": {"mode": "global"},
                                "clustering": {"mode": "runs", "avg_run_length": 5.0, "avg_gap": 2.0},
                                "columns": {
                                    "mode": "hotspots",
                                    "hotspot_count": 8,
                                    "hotspot_prob": 0.7,
                                    "hotspot_spread": 1,
                                },
                                "inter_row_similarity": {"mode": "window_share", "window": 6, "share_prob": 0.4},
                                "block_structure": {
                                    "enabled": True,
                                    "block_rows": 4,
                                    "block_cols": 4,
                                    "block_density": 0.5,
                                    "block_prob": 0.1,
                                },
                            },
                        ],
                        "pairs": [
                            {
                                "kernel": "spadd",
                                "pairs": [{"a": "m_a", "b": "m_b"}],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            cmd = [sys.executable, str(GENERATOR), "--spec", str(spec), "--out", str(out_root), "--force"]
            result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)
            if result.returncode != 0:
                raise AssertionError(f"Generator failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")

            manifest = out_root / "manifest.csv"
            pairs = out_root / "pairs.csv"
            canonical_a = out_root / "canonical" / "m_a.mtx"
            canonical_b = out_root / "canonical" / "m_b.mtx"

            self.assertTrue(manifest.exists())
            self.assertTrue(pairs.exists())
            self.assertTrue(canonical_a.exists())
            self.assertTrue(canonical_b.exists())

            with manifest.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            by_name = {row["name"]: row for row in rows}
            self.assertIn("m_a", by_name)
            self.assertIn("m_b", by_name)

            self.assertEqual(by_name["m_a"]["source"], "generated")
            self.assertEqual(by_name["m_a"]["generator"], "structured_random_v1")
            self.assertGreater(float(by_name["m_a"]["char_nnz"]), 0.0)
            self.assertGreater(float(by_name["m_a"]["char_density"]), 0.0)
            self.assertGreaterEqual(float(by_name["m_a"]["char_bandwidth"]), 0.0)
            self.assertIn("nnz", by_name["m_a"]["params_json"])

            with pairs.open("r", encoding="utf-8", newline="") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual(len(pair_rows), 1)
            self.assertEqual(pair_rows[0]["kernel"], "spadd")
            self.assertEqual(pair_rows[0]["matrix_a"], "m_a")
            self.assertEqual(pair_rows[0]["matrix_b"], "m_b")


if __name__ == "__main__":
    unittest.main()
