#!/usr/bin/env python3
"""Unified benchmark runner for SpGEMM."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from benchmark_common import (
    CONFIGS,
    DEFAULT_CONFIG_ORDER,
    DEFAULT_SWEEP_BLOCK_SIZES,
    DEFAULT_SWEEP_ORDERS,
    FailedRun,
    TwoSparseBenchmarkResult,
    add_hwc_args,
    build_hwc_config,
    build_config_specs,
    build_pair_matrix_metadata,
    collect_matrix_meta,
    compile_c_to_executable,
    ensure_compiler_exists,
    generate_kernel,
    load_matrix_manifest,
    load_pairs_file,
    manifest_lookup,
    default_hwc_fields,
    parse_int_csv,
    parse_orders_csv,
    parse_csv_list,
    parse_features,
    reset_output_csvs,
    run_trials,
    save_csv,
    select_pairs,
    validate_config_names,
    validate_formats,
    write_pairs_csv,
)
from canonicalize_mtx import canonicalize_directory


KERNEL = "spgemm"
SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = SCRIPT_DIR.parent.parent
KERNELS_DIR = BENCHMARKS_DIR / "kernels" / "benchmark" / "spgemm"
TACO_EXE = BENCHMARKS_DIR / "taco" / "build" / "spgemm_taco"
DEFAULT_OUTPUT = BENCHMARKS_DIR / "results" / "csv" / "benchmark_spgemm.csv"
DEFAULT_FAILED = BENCHMARKS_DIR / "results" / "csv" / "benchmark_spgemm_failed.csv"
DEFAULT_PAIRS = BENCHMARKS_DIR / "results" / "csv" / "spgemm_pairs_used.csv"


def write_dsl_file(path: Path, rows_a: int, cols_a: int, rows_b: int, cols_b: int, sparse_format: str) -> None:
    tensor_format = "CSR" if sparse_format == "csr" else "CSC"
    content = (
        f"tensor A : {tensor_format}<{rows_a}, {cols_a}>;\n"
        f"tensor B : {tensor_format}<{rows_b}, {cols_b}>;\n"
        f"tensor C : {tensor_format}<{rows_a}, {cols_b}>;\n"
        "compute C[i, j] = A[i, k] * B[k, j];\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified SpGEMM benchmark")
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--matrices-dir", type=Path, default=BENCHMARKS_DIR / "matrices" / "suitesparse" / "raw")
    parser.add_argument("--canonical-dir", type=Path, default=BENCHMARKS_DIR / "matrices" / "suitesparse" / "canonical")
    parser.add_argument("--matrix-manifest", type=Path, default=None)
    parser.add_argument("--matrices", default="")
    parser.add_argument("--formats", default="csr,csc")
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIG_ORDER))
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--sweep-block", action="store_true", help="Enable default 1D block sweep")
    parser.add_argument("--sweep-block-sizes", default="", help="Comma-separated 1D block sizes (implies sweep)")
    parser.add_argument("--sweep-orders", default=",".join(DEFAULT_SWEEP_ORDERS), help="Comma-separated opt orders")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--taco-tol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--failed-output", type=Path, default=DEFAULT_FAILED)
    parser.add_argument("--pairs-output", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--pairs-file", type=Path, default=None)
    parser.add_argument("--with-taco", dest="with_taco", action="store_true", default=True)
    parser.add_argument("--no-taco", dest="with_taco", action="store_false")
    add_hwc_args(parser)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_pairs <= 0 or args.iterations <= 0 or args.trials <= 0:
        raise ValueError("--max-pairs, --iterations, and --trials must be > 0")

    ensure_compiler_exists()
    reset_output_csvs(args.output, args.failed_output)

    canonicalized = canonicalize_directory(args.matrices_dir, args.canonical_dir)
    stems = {item.path.stem for item in canonicalized}
    matrices = [meta for meta in collect_matrix_meta(args.canonical_dir) if meta.name in stems]
    manifest_entries = load_matrix_manifest(args.matrix_manifest) if args.matrix_manifest else []
    manifest_by_name = manifest_lookup(manifest_entries) if manifest_entries else None
    requested_names = parse_csv_list(args.matrices) if args.matrices else []
    if requested_names:
        allowed = set(requested_names)
        matrices = [matrix for matrix in matrices if matrix.name in allowed]
        missing = [name for name in requested_names if name not in {m.name for m in matrices}]
        if missing:
            raise ValueError(f"Unknown matrix name(s): {', '.join(missing)}")
    explicit_pairs = load_pairs_file(args.pairs_file, kernel=KERNEL) if args.pairs_file else None
    pairs = select_pairs(KERNEL, matrices, args.max_pairs, explicit_pairs=explicit_pairs)
    write_pairs_csv(pairs, args.pairs_output)
    if not pairs:
        raise ValueError("No compatible matrix pairs found for spgemm")

    formats = validate_formats(parse_csv_list(args.formats))
    static_names = validate_config_names(parse_csv_list(args.configs))

    sweep_block_sizes = []
    if args.sweep_block_sizes:
        sweep_block_sizes = parse_int_csv(args.sweep_block_sizes)
    elif args.sweep_block:
        sweep_block_sizes = list(DEFAULT_SWEEP_BLOCK_SIZES)
    sweep_orders = parse_orders_csv(args.sweep_orders)
    hwc_config = build_hwc_config(args)
    config_specs = build_config_specs(
        KERNEL,
        static_names,
        sweep_block_sizes=sweep_block_sizes,
        sweep_orders=sweep_orders,
    )

    use_taco = args.with_taco and TACO_EXE.exists()
    if args.with_taco and not TACO_EXE.exists():
        print(f"Warning: TACO executable not found at {TACO_EXE}; skipping TACO")

    results = []
    failures = []
    start = time.time()
    print(f"Running {KERNEL}: pairs={len(pairs)}, formats={formats}, configs={len(config_specs)}, trials={args.trials}")

    for fmt in formats:
        for matrix_a, matrix_b in pairs:
            dsl_file = KERNELS_DIR / "_dsl_tmp" / f"{KERNEL}_{matrix_a.name}_{matrix_b.name}_{fmt}.tc"
            write_dsl_file(dsl_file, matrix_a.rows, matrix_a.cols, matrix_b.rows, matrix_b.cols, fmt)

            for config_spec in config_specs:
                try:
                    kernel_dir = KERNELS_DIR / f"{matrix_a.name}__{matrix_b.name}" / fmt
                    c_file = kernel_dir / f"{config_spec.name}.c"
                    exe_file = kernel_dir / config_spec.name
                    generate_kernel(dsl_file, config_spec.flags, c_file)
                    compile_c_to_executable(c_file, exe_file)
                    metrics, trial_summary, selected_stdout, hwc_fields = run_trials(
                        [str(exe_file), str(matrix_a.path), str(matrix_b.path), str(args.iterations)],
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
                                item=f"{matrix_a.name},{matrix_b.name}",
                                reason=f"max_error {trial_summary.trial_max_error_max:.3e} > tol {args.tol:.3e}",
                                **hwc_fields,
                                config_flags=" ".join(config_spec.flags),
                                **trial_summary.__dict__,
                            )
                        )
                    else:
                        results.append(
                            TwoSparseBenchmarkResult(
                                matrix_a=matrix_a.name,
                                matrix_b=matrix_b.name,
                                format=fmt,
                                kernel=KERNEL,
                                impl="ours",
                                config=config_spec.name,
                                rows_a=matrix_a.rows,
                                cols_a=matrix_a.cols,
                                nnz_a=matrix_a.nnz,
                                rows_b=matrix_b.rows,
                                cols_b=matrix_b.cols,
                                nnz_b=matrix_b.nnz,
                                iterations=metrics["iterations"],
                                total_time_ms=metrics["total_time_ms"],
                                avg_time_ms=metrics["avg_time_ms"],
                                min_time_ms=metrics["min_time_ms"],
                                max_time_ms=metrics["max_time_ms"],
                                stddev_ms=metrics["stddev_ms"],
                                variance_pct=metrics["variance_pct"],
                                max_error=metrics["max_error"],
                                **hwc_fields,
                                **build_pair_matrix_metadata(matrix_a.name, matrix_b.name, manifest_by_name),
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
                            item=f"{matrix_a.name},{matrix_b.name}",
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
                taco_config = "taco"
                try:
                    metrics, trial_summary, selected_stdout, hwc_fields = run_trials(
                        [str(TACO_EXE), str(matrix_a.path), str(matrix_b.path), str(args.iterations), fmt],
                        trials=args.trials,
                        timeout=1200,
                        hwc_config=hwc_config,
                    )
                    features = parse_features(selected_stdout)
                    if "Kernel mode: csc_safe_fallback_via_csr" in selected_stdout:
                        taco_config = "taco_csc_safe_fallback_via_csr"
                    if trial_summary.trial_max_error_max > args.taco_tol:
                        failures.append(
                            FailedRun(
                                kernel=KERNEL,
                                impl="taco",
                                config=taco_config,
                                format=fmt,
                                item=f"{matrix_a.name},{matrix_b.name}",
                                reason=f"max_error {trial_summary.trial_max_error_max:.3e} > taco_tol {args.taco_tol:.3e}",
                                **hwc_fields,
                                config_flags=taco_config,
                                **trial_summary.__dict__,
                            )
                        )
                    else:
                        results.append(
                            TwoSparseBenchmarkResult(
                                matrix_a=matrix_a.name,
                                matrix_b=matrix_b.name,
                                format=fmt,
                                kernel=KERNEL,
                                impl="taco",
                                config=taco_config,
                                rows_a=matrix_a.rows,
                                cols_a=matrix_a.cols,
                                nnz_a=matrix_a.nnz,
                                rows_b=matrix_b.rows,
                                cols_b=matrix_b.cols,
                                nnz_b=matrix_b.nnz,
                                iterations=metrics["iterations"],
                                total_time_ms=metrics["total_time_ms"],
                                avg_time_ms=metrics["avg_time_ms"],
                                min_time_ms=metrics["min_time_ms"],
                                max_time_ms=metrics["max_time_ms"],
                                stddev_ms=metrics["stddev_ms"],
                                variance_pct=metrics["variance_pct"],
                                max_error=metrics["max_error"],
                                **hwc_fields,
                                **build_pair_matrix_metadata(matrix_a.name, matrix_b.name, manifest_by_name),
                                **features,
                                config_flags=taco_config,
                                **trial_summary.__dict__,
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        FailedRun(
                            kernel=KERNEL,
                            impl="taco",
                            config=taco_config,
                            format=fmt,
                            item=f"{matrix_a.name},{matrix_b.name}",
                            reason=str(exc),
                            **default_hwc_fields(
                                status="off" if hwc_config.mode == "off" else "error",
                                tool="perf" if hwc_config.mode == "perf" else "",
                                events_requested=hwc_config.events,
                            ),
                            config_flags=taco_config,
                            trials=args.trials,
                        )
                    )

    if results:
        save_csv(results, args.output, TwoSparseBenchmarkResult)
    if failures:
        save_csv(failures, args.failed_output, FailedRun)

    elapsed = time.time() - start
    print(f"Completed {KERNEL} in {elapsed:.1f}s: rows={len(results)}, failures={len(failures)}")
    print(f"Pairs CSV: {args.pairs_output}")
    print(f"Results CSV: {args.output}")
    if failures:
        print(f"Failures CSV: {args.failed_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
