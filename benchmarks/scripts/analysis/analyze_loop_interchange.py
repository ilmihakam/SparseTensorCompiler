#!/usr/bin/env python3
"""Analyze loop interchange benchmark rows against sparse row-structure heuristics."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = SCRIPT_DIR.parent.parent
DEFAULT_ANALYSIS_DIR = BENCHMARKS_DIR / "results" / "analysis"

MODE_AS_IS = "as_is"
MODE_TRANSPOSE = "transpose"
MODES = (MODE_AS_IS, MODE_TRANSPOSE)


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatrixData:
    rows: int
    cols: int
    coords: List[tuple]  # list of (row, col) 0-indexed


@dataclass(frozen=True)
class RowMetrics:
    row_nnz: int
    col_span: int
    col_span_frac: float
    col_density_in_span: float
    num_runs: int


@dataclass(frozen=True)
class MetricRule:
    name: str
    enabled: bool
    predicate: Callable[["RowMetrics", "ThresholdProfile"], bool]


@dataclass(frozen=True)
class ThresholdProfile:
    col_span_frac_min: float
    row_cv_max: float
    use_col_span_frac: bool
    use_row_cv: bool

    @property
    def profile_id(self) -> str:
        tokens = [
            f"csf{self._fmt(self.col_span_frac_min)}",
            f"rcv{self._fmt(self.row_cv_max)}",
            f"u{int(self.use_col_span_frac)}{int(self.use_row_cv)}",
        ]
        return "_".join(tokens)

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{value:g}".replace(".", "p")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_interchange_config(config_name: str) -> bool:
    """Return True only for the 'interchange_only' config."""
    return config_name == "interchange_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze loop interchange benchmark rows")
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        dest="csv_files",
        default=None,
        help="Benchmark CSV file(s); may be repeated. Defaults to spmv + spmm CSVs.",
    )
    parser.add_argument("--canonical-dir", type=Path, default=BENCHMARKS_DIR / "matrices" / "suitesparse" / "canonical")
    parser.add_argument("--impls", default="ours", help="Comma-separated benchmark implementations to include")
    parser.add_argument("--output", type=Path, default=DEFAULT_ANALYSIS_DIR / "loop_interchange_dataset.csv")
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ANALYSIS_DIR / "loop_interchange_rows.csv")
    parser.add_argument(
        "--thresholds-output",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR / "loop_interchange_thresholds_used.csv",
    )
    parser.add_argument(
        "--col-span-frac-min",
        type=float,
        default=0.1,
        help="Minimum col_span_frac for a row to pass (wide scatter signals)",
    )
    parser.add_argument(
        "--row-cv-max",
        type=float,
        default=3.0,
        help="Maximum row-nnz CV for matrix to benefit from interchange",
    )
    parser.add_argument("--use-col-span-frac", dest="use_col_span_frac", action="store_true", default=True)
    parser.add_argument("--no-use-col-span-frac", dest="use_col_span_frac", action="store_false")
    parser.add_argument("--use-row-cv", dest="use_row_cv", action="store_true", default=True)
    parser.add_argument("--no-use-row-cv", dest="use_row_cv", action="store_false")
    return parser.parse_args()


def build_threshold_profile(args: argparse.Namespace) -> ThresholdProfile:
    if args.col_span_frac_min < 0:
        raise ValueError("--col-span-frac-min must be >= 0")
    if args.row_cv_max < 0:
        raise ValueError("--row-cv-max must be >= 0")
    return ThresholdProfile(
        col_span_frac_min=args.col_span_frac_min,
        row_cv_max=args.row_cv_max,
        use_col_span_frac=args.use_col_span_frac,
        use_row_cv=args.use_row_cv,
    )


def build_rules(thresholds: ThresholdProfile) -> List[MetricRule]:
    return [
        MetricRule(
            "col_span_frac",
            thresholds.use_col_span_frac,
            lambda row, cfg: row.col_span_frac >= cfg.col_span_frac_min,
        ),
    ]


# ---------------------------------------------------------------------------
# Matrix I/O
# ---------------------------------------------------------------------------

def load_matrix(path: Path) -> MatrixData:
    with path.open("r", encoding="utf-8") as handle:
        banner = handle.readline()
        if not banner.startswith("%%MatrixMarket"):
            raise ValueError(f"Invalid MatrixMarket banner in {path}")
        rows = cols = -1
        coords: List[tuple] = []
        for raw in handle:
            stripped = raw.strip()
            if not stripped or stripped.startswith("%"):
                continue
            if rows < 0:
                parts = stripped.split()
                if len(parts) < 3:
                    raise ValueError(f"Invalid header line in {path}")
                rows, cols = int(parts[0]), int(parts[1])
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid coordinate in {path}: {stripped}")
            coords.append((int(parts[0]) - 1, int(parts[1]) - 1))
    if rows < 0 or cols < 0:
        raise ValueError(f"Missing dimensions in {path}")
    return MatrixData(rows=rows, cols=cols, coords=coords)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_benchmark_rows(csv_paths: Sequence[Path], impls: Sequence[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        for row in read_csv_rows(csv_path):
            if row.get("impl", "") not in impls:
                continue
            rows.append(row)
    if not rows:
        raise FileNotFoundError("No matching benchmark rows found in the provided CSV files")
    return rows


# ---------------------------------------------------------------------------
# Row-level analysis
# ---------------------------------------------------------------------------

def count_runs(sorted_cols: Sequence[int]) -> int:
    if not sorted_cols:
        return 0
    runs = 1
    prev = sorted_cols[0]
    for col in sorted_cols[1:]:
        if col != prev + 1:
            runs += 1
        prev = col
    return runs


def aggregate_metric(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "p50": float(statistics.median(values)),
        "max": float(max(values)),
    }


def compute_row_analysis(
    matrix: MatrixData,
    mode: str,
    thresholds: ThresholdProfile,
    rules: Sequence[MetricRule],
) -> tuple:
    """Return (per_row_records, aggregate_stats).

    per_row_records: one dict per matrix row with metrics + pass/fail flags.
    aggregate_stats: matrix-level summary stats.
    """
    if mode == MODE_TRANSPOSE:
        num_rows = matrix.cols
        num_cols = matrix.rows
        coords = [(col, row) for row, col in matrix.coords]
    else:
        num_rows = matrix.rows
        num_cols = matrix.cols
        coords = list(matrix.coords)

    # Build per-row column lists
    row_cols: Dict[int, List[int]] = {i: [] for i in range(num_rows)}
    for row_idx, col_idx in coords:
        row_cols[row_idx].append(col_idx)

    per_row_records: List[Dict] = []
    col_span_frac_values: List[float] = []
    col_density_values: List[float] = []
    num_runs_values: List[float] = []
    row_nnz_values: List[float] = []

    for row_idx in range(num_rows):
        cols = row_cols[row_idx]
        nnz = len(cols)
        row_nnz_values.append(float(nnz))

        if nnz == 0:
            col_span = 0
            col_span_frac = 0.0
            col_density = 0.0
            n_runs = 0
        else:
            col_min = min(cols)
            col_max = max(cols)
            col_span = col_max - col_min + 1
            col_span_frac = col_span / num_cols if num_cols > 0 else 0.0
            col_density = nnz / col_span if col_span > 0 else 0.0
            n_runs = count_runs(sorted(cols))

        col_span_frac_values.append(col_span_frac)
        col_density_values.append(col_density)
        num_runs_values.append(float(n_runs))

        metrics = RowMetrics(
            row_nnz=nnz,
            col_span=col_span,
            col_span_frac=col_span_frac,
            col_density_in_span=col_density,
            num_runs=n_runs,
        )

        # Evaluate per-row rules
        row_status: Dict[str, bool] = {}
        for rule in rules:
            if not rule.enabled:
                continue
            row_status[f"passes_{rule.name}"] = rule.predicate(metrics, thresholds)

        per_row_records.append(
            {
                "mode": mode,
                "row_idx": row_idx,
                "row_nnz": nnz,
                "col_span": col_span,
                "col_span_frac": col_span_frac,
                "col_density_in_span": col_density,
                "num_runs": n_runs,
                **row_status,
            }
        )

    # Row-nnz CV (coefficient of variation)
    if len(row_nnz_values) > 1 and statistics.fmean(row_nnz_values) > 0:
        row_nnz_cv = statistics.pstdev(row_nnz_values) / statistics.fmean(row_nnz_values)
    else:
        row_nnz_cv = 0.0

    # Matrix-level aggregate
    aggregate: Dict[str, float] = {
        "row_nnz_cv": row_nnz_cv,
        "interchange_beneficial_by_cv": float(
            row_nnz_cv <= thresholds.row_cv_max if thresholds.use_row_cv else True
        ),
    }
    for metric_name, values in (
        ("col_span_frac", col_span_frac_values),
        ("col_density_in_span", col_density_values),
        ("num_runs", num_runs_values),
        ("row_nnz", row_nnz_values),
    ):
        stats = aggregate_metric(values)
        for suffix, value in stats.items():
            aggregate[f"{metric_name}_{suffix}"] = value

    # Fraction of rows passing all enabled rules
    if per_row_records:
        pass_keys = [k for k in per_row_records[0] if k.startswith("passes_")]
        if pass_keys:
            passing = sum(
                1 for r in per_row_records if all(r[k] for k in pass_keys)
            )
            aggregate["row_pass_frac"] = passing / len(per_row_records)
        else:
            aggregate["row_pass_frac"] = 1.0
    else:
        aggregate["row_pass_frac"] = 0.0

    return per_row_records, aggregate


# ---------------------------------------------------------------------------
# Baseline lookup
# ---------------------------------------------------------------------------

def benchmark_key(row: Dict[str, str]) -> tuple:
    kernel = row.get("kernel", "")
    return (kernel, row.get("impl", ""), row.get("format", ""), row.get("matrix_name", ""))


def build_baseline_map(rows: Sequence[Dict[str, str]]) -> Dict[tuple, float]:
    baseline_map: Dict[tuple, float] = {}
    for row in rows:
        if row.get("config") != "baseline":
            continue
        key = benchmark_key(row)
        baseline_map[key] = float(row["avg_time_ms"])
    return baseline_map


def safe_speedup(baseline_time: float, candidate_time: float) -> float:
    if baseline_time <= 0 or candidate_time <= 0:
        return math.nan
    return baseline_time / candidate_time


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def with_mode_prefix(values: Dict[str, float], mode: str) -> Dict[str, float]:
    return {f"{key}_{mode}": value for key, value in values.items()}


def selected_mode(format_name: str) -> str:
    if format_name == "csc":
        return MODE_TRANSPOSE
    return MODE_AS_IS


def write_csv(rows: Sequence[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen: set = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_threshold_row(thresholds: ThresholdProfile) -> Dict:
    return {
        "threshold_profile_id": thresholds.profile_id,
        "col_span_frac_min": thresholds.col_span_frac_min,
        "row_cv_max": thresholds.row_cv_max,
        "use_col_span_frac": int(thresholds.use_col_span_frac),
        "use_row_cv": int(thresholds.use_row_cv),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    thresholds = build_threshold_profile(args)
    rules = build_rules(thresholds)
    impls = parse_csv_list(args.impls)

    if args.csv_files is None:
        csv_files = [
            BENCHMARKS_DIR / "results" / "csv" / "benchmark_spmv.csv",
            BENCHMARKS_DIR / "results" / "csv" / "benchmark_spmm.csv",
        ]
    else:
        csv_files = args.csv_files

    benchmark_rows = load_benchmark_rows(csv_files, impls)
    baseline_map = build_baseline_map(benchmark_rows)

    matrices: Dict[str, MatrixData] = {}
    analysis_cache: Dict[tuple, tuple] = {}  # (matrix_name, mode) -> (per_row, aggregate)
    dataset_rows: List[Dict] = []
    row_output_rows: List[Dict] = []
    emitted_row_keys: set = set()
    threshold_row = build_threshold_row(thresholds)

    for row in benchmark_rows:
        if not parse_interchange_config(row.get("config", "")):
            continue

        matrix_name = row["matrix_name"]
        matrix_path = args.canonical_dir / f"{matrix_name}.mtx"
        if matrix_name not in matrices:
            matrices[matrix_name] = load_matrix(matrix_path)
        matrix = matrices[matrix_name]

        row_key = benchmark_key(row)
        baseline_time = baseline_map.get(row_key)
        if baseline_time is None:
            continue

        mode_aggregates: Dict[str, Dict[str, float]] = {}
        for mode in MODES:
            cache_key = (matrix_name, mode)
            if cache_key not in analysis_cache:
                analysis_cache[cache_key] = compute_row_analysis(matrix, mode, thresholds, rules)
            per_row, aggregate = analysis_cache[cache_key]
            mode_aggregates[mode] = aggregate

            emitted_key = (matrix_name, mode)
            if emitted_key not in emitted_row_keys:
                emitted_row_keys.add(emitted_key)
                for rec in per_row:
                    row_output_rows.append(
                        {
                            "threshold_profile_id": thresholds.profile_id,
                            "matrix_name": matrix_name,
                            **rec,
                        }
                    )

        chosen_mode = selected_mode(row.get("format", ""))
        dataset_row: Dict = dict(row)
        dataset_row.update(
            {
                "threshold_profile_id": thresholds.profile_id,
                "config_kind": "interchange",
                "baseline_avg_time_ms": baseline_time,
                "speedup_vs_baseline": safe_speedup(baseline_time, float(row["avg_time_ms"])),
                "selected_metric_mode": chosen_mode,
            }
        )
        for mode, aggregate in mode_aggregates.items():
            dataset_row.update(with_mode_prefix(aggregate, mode))
        for key, value in mode_aggregates[chosen_mode].items():
            dataset_row[f"{key}_selected"] = value
        dataset_rows.append(dataset_row)

    if not dataset_rows:
        raise ValueError(
            "No interchange_only benchmark rows found. Re-run benchmarks with interchange_only config "
            "or provide CSVs that contain interchange_only rows."
        )

    write_csv(dataset_rows, args.output)
    write_csv(row_output_rows, args.rows_output)
    write_csv([threshold_row], args.thresholds_output)

    print(f"Wrote dataset: {args.output}")
    print(f"Wrote rows: {args.rows_output}")
    print(f"Wrote thresholds: {args.thresholds_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
