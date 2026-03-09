#!/usr/bin/env python3
"""Smoke tests for unified benchmark runners."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
_RUN_DIR = SCRIPT_DIR.parent / "run"
BENCHMARKS_DIR = SCRIPT_DIR.parent.parent

RUNNERS = {
    "spmv": _RUN_DIR / "benchmark_spmv.py",
    "spmm": _RUN_DIR / "benchmark_spmm.py",
    "spadd": _RUN_DIR / "benchmark_spadd.py",
    "spelmul": _RUN_DIR / "benchmark_spelmul.py",
    "spgemm": _RUN_DIR / "benchmark_spgemm.py",
    "sddmm": _RUN_DIR / "benchmark_sddmm.py",
}

TACO_EXES = {
    "spmv": BENCHMARKS_DIR / "taco" / "build" / "spmv_taco",
    "spmm": BENCHMARKS_DIR / "taco" / "build" / "spmm_taco",
    "spadd": BENCHMARKS_DIR / "taco" / "build" / "spadd_taco",
    "spelmul": BENCHMARKS_DIR / "taco" / "build" / "spelmul_taco",
    "spgemm": BENCHMARKS_DIR / "taco" / "build" / "spgemm_taco",
    "sddmm": BENCHMARKS_DIR / "taco" / "build" / "sddmm_taco",
}

PAIR_KERNELS = {"spadd", "spelmul", "spgemm"}
MATRIX_KERNELS = {"spmv", "spmm", "sddmm"}
SMALL_MATRICES_DIR = BENCHMARKS_DIR / "matrices" / "suitesparse" / "raw"


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_no_failure_rows(path: Path) -> None:
    if not path.exists():
        return
    rows = read_rows(path)
    if rows:
        raise AssertionError(f"Expected empty failures CSV: {path}")


def read_matrix_header(path: Path) -> tuple[int, int, int]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid matrix header: {path}")
            return int(parts[0]), int(parts[1]), int(parts[2])
    raise ValueError(f"Missing matrix header: {path}")


def select_source_matrices(kernel: str) -> list[Path]:
    candidates = sorted(SMALL_MATRICES_DIR.glob("*.mtx"))
    valid = []
    for candidate in candidates:
        try:
            rows, cols, _ = read_matrix_header(candidate)
            valid.append((candidate, rows, cols))
        except Exception:  # noqa: BLE001
            continue
    if not valid:
        raise RuntimeError("No valid matrices available for smoke tests")

    if kernel == "spgemm":
        for candidate, rows, cols in valid:
            if rows == cols:
                return [candidate]
        raise RuntimeError("No square matrix found for spgemm smoke test")

    return [valid[0][0]]


def run_kernel(kernel: str, with_taco: bool) -> None:
    runner = RUNNERS[kernel]
    with tempfile.TemporaryDirectory(prefix=f"bench_smoke_{kernel}_") as tmp_dir:
        tmp = Path(tmp_dir)
        matrices_dir = tmp / "matrices"
        matrices_dir.mkdir(parents=True, exist_ok=True)
        for source in select_source_matrices(kernel):
            shutil.copy2(source, matrices_dir / source.name)

        output_csv = tmp / "results.csv"
        failed_csv = tmp / "failed.csv"
        cmd = [
            sys.executable,
            str(runner),
            "--iterations",
            "1",
            "--formats",
            "csr,csc",
            "--configs",
            "baseline",
            "--output",
            str(output_csv),
            "--failed-output",
            str(failed_csv),
            "--matrices-dir",
            str(matrices_dir),
            "--canonical-dir",
            str(tmp / "canonical"),
        ]

        if kernel in PAIR_KERNELS:
            cmd.extend(["--max-pairs", "1", "--pairs-output", str(tmp / "pairs.csv")])
        if kernel in MATRIX_KERNELS:
            cmd.extend(["--max-matrices", "1"])
        if kernel == "sddmm":
            cmd.extend(["--K", "8"])
        if kernel == "spmm":
            cmd.extend(["--output-cols", "32"])

        cmd.append("--with-taco" if with_taco else "--no-taco")
        result = subprocess.run(cmd, cwd=_RUN_DIR, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Runner failed for {kernel}\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )

        if not output_csv.exists():
            raise AssertionError(f"Missing output CSV for {kernel}: {output_csv}")

        assert_no_failure_rows(failed_csv)
        rows = read_rows(output_csv)
        expected_rows = 2 if not with_taco else 4
        if len(rows) != expected_rows:
            raise AssertionError(f"Expected {expected_rows} rows for {kernel}, got {len(rows)}")

        formats = {row["format"] for row in rows}
        if formats != {"csr", "csc"}:
            raise AssertionError(f"Expected csr/csc rows for {kernel}, got {formats}")

        impls = {row["impl"] for row in rows}
        if with_taco:
            if impls != {"ours", "taco"}:
                raise AssertionError(f"Expected ours+taco rows for {kernel}, got {impls}")
        else:
            if impls != {"ours"}:
                raise AssertionError(f"Expected ours-only rows for {kernel}, got {impls}")

        for row in rows:
            if float(row["max_error"]) > 1e-5:
                raise AssertionError(f"max_error too large for {kernel}: {row}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified benchmark smoke tests")
    parser.add_argument("--kernel", choices=["all", *RUNNERS.keys()], default="all")
    parser.add_argument("--mode", choices=["ours", "taco"], default="ours")
    args = parser.parse_args()

    kernels = list(RUNNERS.keys()) if args.kernel == "all" else [args.kernel]
    with_taco = args.mode == "taco"

    for kernel in kernels:
        if with_taco and not TACO_EXES[kernel].exists():
            print(f"SKIP: TACO executable missing for {kernel} at {TACO_EXES[kernel]}")
            continue
        run_kernel(kernel, with_taco=with_taco)
        print(f"PASS: {kernel} ({args.mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
