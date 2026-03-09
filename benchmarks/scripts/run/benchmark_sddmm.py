#!/usr/bin/env python3
"""Unified benchmark runner for SDDMM."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from benchmark_common import (
    CONFIGS,
    DEFAULT_CONFIG_ORDER,
    DEFAULT_SWEEP_BLOCK_SIZES,
    DEFAULT_SWEEP_BLOCK2D_SIZES,
    DEFAULT_SWEEP_ORDERS,
    FailedRun,
    SDDMMBenchmarkResult,
    add_hwc_args,
    build_hwc_config,
    build_config_specs,
    build_single_matrix_metadata,
    collect_matrix_meta,
    compile_c_to_executable,
    ensure_compiler_exists,
    generate_kernel,
    load_matrix_manifest,
    manifest_lookup,
    default_hwc_fields,
    parse_block2d_csv,
    parse_int_csv,
    parse_orders_csv,
    parse_csv_list,
    parse_features,
    reset_output_csvs,
    run_trials,
    save_csv,
    select_matrices_custom,
    validate_config_names,
    validate_formats,
)
from canonicalize_mtx import canonicalize_directory


KERNEL = "sddmm"
SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = SCRIPT_DIR.parent.parent
KERNELS_DIR = BENCHMARKS_DIR / "kernels" / "benchmark" / "sddmm"
TACO_EXE = BENCHMARKS_DIR / "taco" / "build" / "sddmm_taco"
DEFAULT_OUTPUT = BENCHMARKS_DIR / "results" / "csv" / "benchmark_sddmm.csv"
DEFAULT_FAILED = BENCHMARKS_DIR / "results" / "csv" / "benchmark_sddmm_failed.csv"


def write_dsl_file(path: Path, rows: int, cols: int, k_dim: int, sparse_format: str) -> None:
    tensor_format = "CSR" if sparse_format == "csr" else "CSC"
    content = (
        f"tensor S : {tensor_format}<{rows}, {cols}>;\n"
        f"tensor D : Dense<{rows}, {k_dim}>;\n"
        f"tensor E : Dense<{k_dim}, {cols}>;\n"
        f"tensor C : {tensor_format}<{rows}, {cols}>;\n"
        "compute C[i, j] = S[i, j] * D[i, k] * E[k, j];\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified SDDMM benchmark")
    parser.add_argument("--max-matrices", type=int, default=10)
    parser.add_argument("--matrices-dir", type=Path, default=BENCHMARKS_DIR / "matrices" / "suitesparse" / "raw")
    parser.add_argument("--canonical-dir", type=Path, default=BENCHMARKS_DIR / "matrices" / "suitesparse" / "canonical")
    parser.add_argument("--matrix-manifest", type=Path, default=None)
    parser.add_argument("--matrices", default="")
    parser.add_argument("--formats", default="csr,csc")
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIG_ORDER))
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--sweep-block", action="store_true", help="Enable default 1D block sweep")
    parser.add_argument("--sweep-block-sizes", default="", help="Comma-separated 1D block sizes (implies sweep)")
    parser.add_argument("--sweep-block-2d", action="store_true", help="Enable default 2D block sweep")
    parser.add_argument("--sweep-block-2d-sizes", default="", help="Comma-separated 2D block sizes (e.g., 16x16)")
    parser.add_argument("--sweep-orders", default=",".join(DEFAULT_SWEEP_ORDERS), help="Comma-separated opt orders")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--taco-tol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failed-output", type=Path, default=DEFAULT_FAILED)
    parser.add_argument("--with-taco", dest="with_taco", action="store_true", default=True)
    parser.add_argument("--no-taco", dest="with_taco", action="store_false")
    add_hwc_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_matrices <= 0 or args.iterations <= 0 or args.K <= 0 or args.trials <= 0:
        raise ValueError("--max-matrices, --iterations, --K, and --trials must be > 0")

    ensure_compiler_exists()
    reset_output_csvs(args.output, args.failed_output)

    canonicalized = canonicalize_directory(args.matrices_dir, args.canonical_dir)
    stems = {item.path.stem for item in canonicalized}
    matrices = [meta for meta in collect_matrix_meta(args.canonical_dir) if meta.name in stems]
    manifest_entries = load_matrix_manifest(args.matrix_manifest) if args.matrix_manifest else []
    ordered_names = [entry.name for entry in manifest_entries] if manifest_entries else None
    requested_names = parse_csv_list(args.matrices) if args.matrices else None
    selected = select_matrices_custom(
        matrices,
        args.max_matrices,
        ordered_names=ordered_names,
        requested_names=requested_names,
    )
    manifest_by_name = manifest_lookup(manifest_entries) if manifest_entries else None
    if not selected:
        raise ValueError("No canonical matrices found for sddmm")

    formats = validate_formats(parse_csv_list(args.formats))
    static_names = validate_config_names(parse_csv_list(args.configs))

    sweep_block_sizes = []
    if args.sweep_block_sizes:
        sweep_block_sizes = parse_int_csv(args.sweep_block_sizes)
    elif args.sweep_block:
        sweep_block_sizes = list(DEFAULT_SWEEP_BLOCK_SIZES)

    sweep_block2d_sizes = []
    if args.sweep_block_2d_sizes:
        sweep_block2d_sizes = parse_block2d_csv(args.sweep_block_2d_sizes)
    elif args.sweep_block_2d:
        sweep_block2d_sizes = list(DEFAULT_SWEEP_BLOCK2D_SIZES)

    sweep_orders = parse_orders_csv(args.sweep_orders)
    hwc_config = build_hwc_config(args)
    config_specs = build_config_specs(
        KERNEL,
        static_names,
        sweep_block_sizes=sweep_block_sizes,
        sweep_orders=sweep_orders,
        sweep_block2d_sizes=sweep_block2d_sizes,
    )

    use_taco = args.with_taco and TACO_EXE.exists()
    if args.with_taco and not TACO_EXE.exists():
        print(f"Warning: TACO executable not found at {TACO_EXE}; skipping TACO")

    results = []
    failures = []
    start = time.time()
    print(f"Running {KERNEL}: matrices={len(selected)}, formats={formats}, configs={len(config_specs)}, trials={args.trials}")

    for fmt in formats:
        for matrix in selected:
            dsl_file = KERNELS_DIR / "_dsl_tmp" / f"{KERNEL}_{matrix.name}_{fmt}_k{args.K}.tc"
            write_dsl_file(dsl_file, matrix.rows, matrix.cols, args.K, fmt)

            for config_spec in config_specs:
                try:
                    kernel_dir = KERNELS_DIR / matrix.name / fmt
                    c_file = kernel_dir / f"{config_spec.name}.c"
                    exe_file = kernel_dir / config_spec.name
                    generate_kernel(dsl_file, config_spec.flags, c_file)
                    compile_c_to_executable(c_file, exe_file)
                    metrics, trial_summary, selected_stdout, hwc_fields = run_trials(
                        [str(exe_file), str(matrix.path), str(args.K), str(args.iterations)],
                        trials=args.trials,
                        timeout=1200,
                        hwc_config=hwc_config,
                    )
                    features = parse_features(selected_stdout)
                    if trial_summary.trial_max_error_max > args.tol:
                        failures.append(
                            FailedRun(
                                kernel=KERNEL,
                                impl="ours",
                                config=config_spec.name,
                                format=fmt,
                                item=matrix.name,
                                reason=f"max_error {trial_summary.trial_max_error_max:.3e} > tol {args.tol:.3e}",
                                **hwc_fields,
                                config_flags=" ".join(config_spec.flags),
                                **trial_summary.__dict__,
                            )
                        )
                    else:
                        results.append(
                            SDDMMBenchmarkResult(
                                matrix_name=matrix.name,
                                format=fmt,
                                kernel=KERNEL,
                                impl="ours",
                                config=config_spec.name,
                                rows=matrix.rows,
                                cols=matrix.cols,
                                nnz=matrix.nnz,
                                K=args.K,
                                iterations=metrics["iterations"],
                                total_time_ms=metrics["total_time_ms"],
                                avg_time_ms=metrics["avg_time_ms"],
                                min_time_ms=metrics["min_time_ms"],
                                max_time_ms=metrics["max_time_ms"],
                                stddev_ms=metrics["stddev_ms"],
                                variance_pct=metrics["variance_pct"],
                                max_error=metrics["max_error"],
                                **hwc_fields,
                                **build_single_matrix_metadata(matrix.name, manifest_by_name),
                                **features,
                                config_flags=" ".join(config_spec.flags),
                                **trial_summary.__dict__,
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        FailedRun(
                            kernel=KERNEL,
                            impl="ours",
                            config=config_spec.name,
                            format=fmt,
                            item=matrix.name,
                            reason=str(exc),
                            **default_hwc_fields(
                                status="off" if hwc_config.mode == "off" else "error",
                                tool="perf" if hwc_config.mode == "perf" else "",
                                events_requested=hwc_config.events,
                            ),
                            config_flags=" ".join(config_spec.flags),
                            trials=args.trials,
                        )
                    )

            if use_taco:
                try:
                    metrics, trial_summary, selected_stdout, hwc_fields = run_trials(
                        [str(TACO_EXE), str(matrix.path), str(args.K), str(args.iterations), fmt],
                        trials=args.trials,
                        timeout=1200,
                        hwc_config=hwc_config,
                    )
                    features = parse_features(selected_stdout)
                    if trial_summary.trial_max_error_max > args.taco_tol:
                        failures.append(
                            FailedRun(
                                kernel=KERNEL,
                                impl="taco",
                                config="taco",
                                format=fmt,
                                item=matrix.name,
                                reason=f"max_error {trial_summary.trial_max_error_max:.3e} > taco_tol {args.taco_tol:.3e}",
                                **hwc_fields,
                                config_flags="taco",
                                **trial_summary.__dict__,
                            )
                        )
                    else:
                        results.append(
                            SDDMMBenchmarkResult(
                                matrix_name=matrix.name,
                                format=fmt,
                                kernel=KERNEL,
                                impl="taco",
                                config="taco",
                                rows=matrix.rows,
                                cols=matrix.cols,
                                nnz=matrix.nnz,
                                K=args.K,
                                iterations=metrics["iterations"],
                                total_time_ms=metrics["total_time_ms"],
                                avg_time_ms=metrics["avg_time_ms"],
                                min_time_ms=metrics["min_time_ms"],
                                max_time_ms=metrics["max_time_ms"],
                                stddev_ms=metrics["stddev_ms"],
                                variance_pct=metrics["variance_pct"],
                                max_error=metrics["max_error"],
                                **hwc_fields,
                                **build_single_matrix_metadata(matrix.name, manifest_by_name),
                                **features,
                                config_flags="taco",
                                **trial_summary.__dict__,
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        FailedRun(
                            kernel=KERNEL,
                            impl="taco",
                            config="taco",
                            format=fmt,
                            item=matrix.name,
                            reason=str(exc),
                            **default_hwc_fields(
                                status="off" if hwc_config.mode == "off" else "error",
                                tool="perf" if hwc_config.mode == "perf" else "",
                                events_requested=hwc_config.events,
                            ),
                            config_flags="taco",
                            trials=args.trials,
                        )
                    )

    if results:
        save_csv(results, args.output, SDDMMBenchmarkResult)
    if failures:
        save_csv(failures, args.failed_output, FailedRun)

    elapsed = time.time() - start
    print(f"Completed {KERNEL} in {elapsed:.1f}s: rows={len(results)}, failures={len(failures)}")
    print(f"Results CSV: {args.output}")
    if failures:
        print(f"Failures CSV: {args.failed_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
