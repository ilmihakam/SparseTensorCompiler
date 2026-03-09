#!/usr/bin/env python3
"""Smoke tests for custom generator + manifest + pair-file runner integration."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
_RUN_DIR = SCRIPT_DIR.parent / "run"
_UTILS_DIR = SCRIPT_DIR.parent / "utils"
RUNNERS = {
    "spmv": _RUN_DIR / "benchmark_spmv.py",
    "spadd": _RUN_DIR / "benchmark_spadd.py",
}
GENERATOR = _UTILS_DIR / "generate_matrices.py"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_cmd(cmd: list[str]) -> None:
    result = subprocess.run(cmd, cwd=_RUN_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed\nCMD: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def write_spec(path: Path) -> None:
    payload = {
        "version": 1,
        "matrices": [
            {
                "name": "smoke_a",
                "generator": "structured_random_v1",
                "rows": 96,
                "cols": 96,
                "seed": 7,
                "nnz": {"mode": "density", "density": 0.025, "row_distribution": "lognormal", "row_cv": 0.9},
                "support": {"mode": "banded", "bandwidth": 12},
                "clustering": {"mode": "runs", "avg_run_length": 3.0, "avg_gap": 3.0},
                "columns": {"mode": "uniform"},
                "inter_row_similarity": {"mode": "window_share", "window": 4, "share_prob": 0.2},
                "block_structure": {"enabled": False},
            },
            {
                "name": "smoke_b",
                "generator": "structured_random_v1",
                "rows": 96,
                "cols": 96,
                "seed": 8,
                "nnz": {"mode": "density", "density": 0.03, "row_distribution": "lognormal", "row_cv": 1.3},
                "support": {"mode": "global"},
                "clustering": {"mode": "runs", "avg_run_length": 5.0, "avg_gap": 2.0},
                "columns": {"mode": "hotspots", "hotspot_count": 16, "hotspot_prob": 0.7, "hotspot_spread": 1},
                "inter_row_similarity": {"mode": "window_share", "window": 6, "share_prob": 0.45},
                "block_structure": {
                    "enabled": True,
                    "block_rows": 4,
                    "block_cols": 4,
                    "block_density": 0.5,
                    "block_prob": 0.08,
                },
            },
        ],
        "pairs": [
            {
                "kernel": "spadd",
                "pairs": [{"a": "smoke_a", "b": "smoke_b"}],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def run_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="custom_bench_smoke_") as tmp_dir:
        tmp = Path(tmp_dir)
        spec = tmp / "spec.json"
        custom = tmp / "custom"
        write_spec(spec)

        run_cmd([sys.executable, str(GENERATOR), "--spec", str(spec), "--out", str(custom), "--force"])

        manifest = custom / "manifest.csv"
        pairs_file = custom / "pairs.csv"
        matrices_dir = custom / "raw"
        canonical_dir = custom / "canonical"

        spmv_out = tmp / "spmv.csv"
        spmv_fail = tmp / "spmv_fail.csv"
        run_cmd(
            [
                sys.executable,
                str(RUNNERS["spmv"]),
                "--iterations",
                "1",
                "--formats",
                "csr",
                "--configs",
                "baseline",
                "--max-matrices",
                "1",
                "--matrices",
                "smoke_a",
                "--matrices-dir",
                str(matrices_dir),
                "--canonical-dir",
                str(canonical_dir),
                "--matrix-manifest",
                str(manifest),
                "--output",
                str(spmv_out),
                "--failed-output",
                str(spmv_fail),
                "--no-taco",
            ]
        )
        spmv_rows = read_rows(spmv_out)
        if len(spmv_rows) != 1:
            raise AssertionError(f"Expected one spmv row, got {len(spmv_rows)}")
        row = spmv_rows[0]
        if row.get("matrix_generator", "") != "structured_random_v1":
            raise AssertionError(f"Missing matrix_generator metadata in spmv row: {row}")
        if float(row.get("char_density", "0")) <= 0.0:
            raise AssertionError(f"Expected char_density in spmv row: {row}")

        spadd_out = tmp / "spadd.csv"
        spadd_fail = tmp / "spadd_fail.csv"
        run_cmd(
            [
                sys.executable,
                str(RUNNERS["spadd"]),
                "--iterations",
                "1",
                "--formats",
                "csr",
                "--configs",
                "baseline",
                "--max-pairs",
                "1",
                "--matrices-dir",
                str(matrices_dir),
                "--canonical-dir",
                str(canonical_dir),
                "--matrix-manifest",
                str(manifest),
                "--pairs-file",
                str(pairs_file),
                "--pairs-output",
                str(tmp / "pairs_used.csv"),
                "--output",
                str(spadd_out),
                "--failed-output",
                str(spadd_fail),
                "--no-taco",
            ]
        )
        spadd_rows = read_rows(spadd_out)
        if len(spadd_rows) != 1:
            raise AssertionError(f"Expected one spadd row, got {len(spadd_rows)}")
        row = spadd_rows[0]
        if row.get("matrix_a_generator", "") != "structured_random_v1":
            raise AssertionError(f"Missing matrix_a_generator metadata in spadd row: {row}")
        if row.get("matrix_b_generator", "") != "structured_random_v1":
            raise AssertionError(f"Missing matrix_b_generator metadata in spadd row: {row}")
        if float(row.get("char_a_density", "0")) <= 0.0 or float(row.get("char_b_density", "0")) <= 0.0:
            raise AssertionError(f"Missing pair characteristics in spadd row: {row}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Custom matrix generation benchmark smoke test")
    parser.parse_args()
    run_smoke()
    print("PASS: custom matrix pipeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
