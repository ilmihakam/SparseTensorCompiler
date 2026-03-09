#!/usr/bin/env python3
"""Run unified benchmarks for multiple kernels with one command."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from benchmark_common import DEFAULT_CONFIG_ORDER, add_hwc_args


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = SCRIPT_DIR.parent.parent

RUNNERS = {
    "spmv": SCRIPT_DIR / "benchmark_spmv.py",
    "spmm": SCRIPT_DIR / "benchmark_spmm.py",
    "spadd": SCRIPT_DIR / "benchmark_spadd.py",
    "spelmul": SCRIPT_DIR / "benchmark_spelmul.py",
    "spgemm": SCRIPT_DIR / "benchmark_spgemm.py",
    "sddmm": SCRIPT_DIR / "benchmark_sddmm.py",
}

PAIR_KERNELS = {"spadd", "spelmul", "spgemm"}
MATRIX_KERNELS = {"spmv", "spmm", "sddmm"}


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified benchmark runners for selected kernels")
    parser.add_argument("--kernels", default="spmv,spmm,spadd,spelmul,spgemm,sddmm")
    parser.add_argument("--matrices-dir", type=Path, default=BENCHMARKS_DIR / "matrices" / "suitesparse" / "raw")
    parser.add_argument("--canonical-dir", type=Path, default=BENCHMARKS_DIR / "matrices" / "suitesparse" / "canonical")
    parser.add_argument("--matrix-manifest", type=Path, default=None)
    parser.add_argument("--matrices", default="")
    parser.add_argument("--pairs-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=BENCHMARKS_DIR / "results" / "csv")
    parser.add_argument("--formats", default="csr,csc")
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIG_ORDER))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--sweep-block", action="store_true", help="Enable default 1D block sweep")
    parser.add_argument("--sweep-block-sizes", default="", help="Comma-separated 1D block sizes (implies sweep)")
    parser.add_argument("--sweep-block-2d", action="store_true", help="Enable default 2D block sweep (spmm/sddmm)")
    parser.add_argument("--sweep-block-2d-sizes", default="", help="Comma-separated 2D block sizes (spmm/sddmm)")
    parser.add_argument("--sweep-orders", default="", help="Comma-separated opt orders for sweep 'all' configs")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-matrices", type=int, default=10)
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--K", dest="k_dim", type=int, default=64)
    parser.add_argument("--output-cols", type=int, default=32)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--taco-tol", type=float, default=1e-6)
    parser.add_argument("--with-taco", dest="with_taco", action="store_true", default=True)
    parser.add_argument("--no-taco", dest="with_taco", action="store_false")
    parser.add_argument("--fail-fast", action="store_true")
    add_hwc_args(parser)
    return parser.parse_args()


def build_command(args: argparse.Namespace, kernel: str) -> list[str]:
    output = args.output_dir / f"benchmark_{kernel}.csv"
    failed = args.output_dir / f"benchmark_{kernel}_failed.csv"

    cmd = [
        sys.executable,
        str(RUNNERS[kernel]),
        "--matrices-dir",
        str(args.matrices_dir),
        "--canonical-dir",
        str(args.canonical_dir),
        "--formats",
        args.formats,
        "--configs",
        args.configs,
        "--trials",
        str(args.trials),
        "--iterations",
        str(args.iterations),
        "--tol",
        str(args.tol),
        "--taco-tol",
        str(args.taco_tol),
        "--output",
        str(output),
        "--failed-output",
        str(failed),
    ]
    if args.matrix_manifest:
        cmd.extend(["--matrix-manifest", str(args.matrix_manifest)])
    if args.matrices:
        cmd.extend(["--matrices", args.matrices])
    cmd.append("--with-taco" if args.with_taco else "--no-taco")
    if args.hwc_mode != "off":
        cmd.extend(["--hwc-mode", args.hwc_mode, "--hwc-events", args.hwc_events])
    if args.hwc_strict:
        cmd.append("--hwc-strict")
    if args.hwc_mode == "lauka":
        cmd.extend([
            "--hwc-lauka-bin", args.hwc_lauka_bin,
            "--hwc-lauka-runs", str(args.hwc_lauka_runs),
            "--hwc-lauka-warmup", str(args.hwc_lauka_warmup),
        ])

    if args.sweep_block:
        cmd.append("--sweep-block")
    if args.sweep_block_sizes:
        cmd.extend(["--sweep-block-sizes", args.sweep_block_sizes])
    if args.sweep_orders:
        cmd.extend(["--sweep-orders", args.sweep_orders])

    if kernel in MATRIX_KERNELS:
        cmd.extend(["--max-matrices", str(args.max_matrices)])
    if kernel in PAIR_KERNELS:
        pairs = args.output_dir / f"{kernel}_pairs_used.csv"
        cmd.extend(["--max-pairs", str(args.max_pairs), "--pairs-output", str(pairs)])
        if args.pairs_file:
            cmd.extend(["--pairs-file", str(args.pairs_file)])
    if kernel == "spmm":
        if args.sweep_block_2d:
            cmd.append("--sweep-block-2d")
        if args.sweep_block_2d_sizes:
            cmd.extend(["--sweep-block-2d-sizes", args.sweep_block_2d_sizes])
        cmd.extend(["--output-cols", str(args.output_cols)])
    if kernel == "sddmm":
        if args.sweep_block_2d:
            cmd.append("--sweep-block-2d")
        if args.sweep_block_2d_sizes:
            cmd.extend(["--sweep-block-2d-sizes", args.sweep_block_2d_sizes])
        cmd.extend(["--K", str(args.k_dim)])

    return cmd


def main() -> int:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if args.trials <= 0:
        raise ValueError("--trials must be > 0")
    if args.max_matrices <= 0:
        raise ValueError("--max-matrices must be > 0")
    if args.max_pairs <= 0:
        raise ValueError("--max-pairs must be > 0")
    if args.k_dim <= 0:
        raise ValueError("--K must be > 0")
    if args.output_cols <= 0:
        raise ValueError("--output-cols must be > 0")

    kernels = parse_csv_list(args.kernels)
    invalid = [kernel for kernel in kernels if kernel not in RUNNERS]
    if invalid:
        raise ValueError(f"Unknown kernels: {', '.join(invalid)}")
    if not kernels:
        raise ValueError("No kernels selected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    failed_kernels = []

    print(f"Running kernels: {', '.join(kernels)}")
    for kernel in kernels:
        cmd = build_command(args, kernel)
        print(f"\n[{kernel}] {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout.rstrip())
        if result.returncode != 0:
            failed_kernels.append(kernel)
            if result.stderr:
                print(result.stderr.rstrip(), file=sys.stderr)
            print(f"[{kernel}] FAILED")
            if args.fail_fast:
                break
        else:
            print(f"[{kernel}] OK")

    elapsed = time.time() - started
    print(f"\nCompleted in {elapsed:.1f}s")
    if failed_kernels:
        print(f"Failed kernels: {', '.join(failed_kernels)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
