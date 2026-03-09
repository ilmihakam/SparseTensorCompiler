#!/usr/bin/env python3
"""Analyze 1D blocking benchmark rows against sparse block-locality heuristics."""

from __future__ import annotations

import argparse
import csv
import math
import re
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

BLOCK_ONLY_RE = re.compile(r"^block_only$")
BLOCK_B_RE = re.compile(r"^block_b(?P<size>\d+)$")
DEFAULT_BLOCK_SIZE = 32


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
class BlockMetrics:
    block_nnz: int
    unique_cols: int
    col_reuse: float
    col_coverage: float
    intra_block_row_cv: float


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
    predicate: Callable[["BlockMetrics", "ThresholdProfile"], bool]


@dataclass(frozen=True)
class ThresholdProfile:
    col_reuse_min: float
    col_coverage_max: float
    use_col_reuse: bool
    use_col_coverage: bool

    @property
    def profile_id(self) -> str:
        tokens = [
            f"cr{self._fmt(self.col_reuse_min)}",
            f"cc{self._fmt(self.col_coverage_max)}",
            f"u{int(self.use_col_reuse)}{int(self.use_col_coverage)}",
        ]
        return "_".join(tokens)

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{value:g}".replace(".", "p")


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def parse_1d_block_config(config_name: str) -> int | None:
    """Return block size for 1D blocking configs, else None.

    "block_only" -> DEFAULT_BLOCK_SIZE (32)
    "block_b16"  -> 16
    "block_b64"  -> 64
    Anything else (interchange_only, block2d_*, i_then_b, all_*, ...) -> None
    """
    if BLOCK_ONLY_RE.match(config_name):
        return DEFAULT_BLOCK_SIZE
    match = BLOCK_B_RE.match(config_name)
    if match:
        return int(match.group("size"))
    return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze 1D blocking benchmark rows")
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        dest="csv_files",
        default=None,
        help="Benchmark CSV file(s); may be repeated. Defaults to spmv CSV.",
    )
    parser.add_argument("--canonical-dir", type=Path, default=BENCHMARKS_DIR / "matrices" / "suitesparse" / "canonical")
    parser.add_argument("--impls", default="ours", help="Comma-separated benchmark implementations to include")
    parser.add_argument("--output", type=Path, default=DEFAULT_ANALYSIS_DIR / "1d_blocking_dataset.csv")
    parser.add_argument("--blocks-output", type=Path, default=DEFAULT_ANALYSIS_DIR / "1d_blocking_blocks.csv")
    parser.add_argument(
        "--thresholds-output",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR / "1d_blocking_thresholds_used.csv",
    )
    parser.add_argument(
        "--col-reuse-min",
        type=float,
        default=1.5,
        help="Minimum col_reuse ratio for a block to be beneficial",
    )
    parser.add_argument(
        "--col-coverage-max",
        type=float,
        default=0.8,
        help="Maximum col_coverage; high coverage means no locality gain",
    )
    parser.add_argument("--use-col-reuse", dest="use_col_reuse", action="store_true", default=True)
    parser.add_argument("--no-use-col-reuse", dest="use_col_reuse", action="store_false")
    parser.add_argument("--use-col-coverage", dest="use_col_coverage", action="store_true", default=True)
    parser.add_argument("--no-use-col-coverage", dest="use_col_coverage", action="store_false")
    return parser.parse_args()


def build_threshold_profile(args: argparse.Namespace) -> ThresholdProfile:
    if args.col_reuse_min < 0:
        raise ValueError("--col-reuse-min must be >= 0")
    if args.col_coverage_max < 0:
        raise ValueError("--col-coverage-max must be >= 0")
    return ThresholdProfile(
        col_reuse_min=args.col_reuse_min,
        col_coverage_max=args.col_coverage_max,
        use_col_reuse=args.use_col_reuse,
        use_col_coverage=args.use_col_coverage,
    )


def build_rules(thresholds: ThresholdProfile) -> List[MetricRule]:
    return [
        MetricRule(
            "col_reuse",
            thresholds.use_col_reuse,
            lambda blk, cfg: blk.col_reuse >= cfg.col_reuse_min,
        ),
        MetricRule(
            "col_coverage",
            thresholds.use_col_coverage,
            lambda blk, cfg: blk.col_coverage <= cfg.col_coverage_max,
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
# Row-level helper (also included in dataset for completeness)
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


def compute_row_aggregates(
    coords: Sequence[tuple],
    num_rows: int,
    num_cols: int,
) -> Dict[str, float]:
    """Compute row-level aggregate stats (col_span_frac, col_density, num_runs, row_nnz_cv)."""
    row_cols: Dict[int, List[int]] = {i: [] for i in range(num_rows)}
    for row_idx, col_idx in coords:
        row_cols[row_idx].append(col_idx)

    col_span_frac_values: List[float] = []
    col_density_values: List[float] = []
    num_runs_values: List[float] = []
    row_nnz_values: List[float] = []

    for row_idx in range(num_rows):
        cols = row_cols[row_idx]
        nnz = len(cols)
        row_nnz_values.append(float(nnz))
        if nnz == 0:
            col_span_frac_values.append(0.0)
            col_density_values.append(0.0)
            num_runs_values.append(0.0)
        else:
            col_span = max(cols) - min(cols) + 1
            col_span_frac = col_span / num_cols if num_cols > 0 else 0.0
            col_density = nnz / col_span if col_span > 0 else 0.0
            col_span_frac_values.append(col_span_frac)
            col_density_values.append(col_density)
            num_runs_values.append(float(count_runs(sorted(cols))))

    if len(row_nnz_values) > 1 and statistics.fmean(row_nnz_values) > 0:
        row_nnz_cv = statistics.pstdev(row_nnz_values) / statistics.fmean(row_nnz_values)
    else:
        row_nnz_cv = 0.0

    agg: Dict[str, float] = {"row_nnz_cv": row_nnz_cv}
    for name, vals in (
        ("row_col_span_frac", col_span_frac_values),
        ("row_col_density_in_span", col_density_values),
        ("row_num_runs", num_runs_values),
        ("row_nnz", row_nnz_values),
    ):
        for suffix, v in aggregate_metric(vals).items():
            agg[f"{name}_{suffix}"] = v
    return agg


# ---------------------------------------------------------------------------
# Block-level analysis
# ---------------------------------------------------------------------------

def compute_block_analysis(
    matrix: MatrixData,
    block_size: int,
    mode: str,
    thresholds: ThresholdProfile,
    rules: Sequence[MetricRule],
) -> tuple:
    """Return (per_block_records, aggregate_stats).

    per_block_records: one dict per block of `block_size` rows.
    aggregate_stats: matrix-level summary.
    """
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    if mode == MODE_TRANSPOSE:
        num_rows = matrix.cols
        num_cols = matrix.rows
        coords = [(col, row) for row, col in matrix.coords]
    else:
        num_rows = matrix.rows
        num_cols = matrix.cols
        coords = list(matrix.coords)

    num_blocks = math.ceil(num_rows / block_size) if num_rows > 0 else 0

    # Build per-block col lists and per-row counts within each block
    block_data: Dict[int, Dict] = {}
    for row_idx, col_idx in coords:
        blk = row_idx // block_size
        if blk not in block_data:
            block_data[blk] = {"all_cols": [], "row_nnz": {}}
        block_data[blk]["all_cols"].append(col_idx)
        block_data[blk]["row_nnz"][row_idx] = block_data[blk]["row_nnz"].get(row_idx, 0) + 1

    per_block_records: List[Dict] = []
    col_reuse_values: List[float] = []
    col_coverage_values: List[float] = []
    intra_block_cv_values: List[float] = []
    block_nnz_values: List[float] = []
    good_blocks = 0

    for blk_idx in range(num_blocks):
        data = block_data.get(blk_idx, {"all_cols": [], "row_nnz": {}})
        all_cols = data["all_cols"]
        row_nnz_map = data["row_nnz"]

        block_nnz = len(all_cols)
        unique_cols = len(set(all_cols))
        col_reuse = block_nnz / unique_cols if unique_cols > 0 else 0.0
        col_coverage = unique_cols / num_cols if num_cols > 0 else 0.0

        # Intra-block row CV
        row_start = blk_idx * block_size
        row_end = min(row_start + block_size, num_rows)
        rows_in_block = row_end - row_start
        per_row_counts = [row_nnz_map.get(r, 0) for r in range(row_start, row_end)]
        if rows_in_block > 1:
            mean_nnz = statistics.fmean(per_row_counts)
            intra_cv = statistics.pstdev(per_row_counts) / mean_nnz if mean_nnz > 0 else 0.0
        else:
            intra_cv = 0.0

        metrics = BlockMetrics(
            block_nnz=block_nnz,
            unique_cols=unique_cols,
            col_reuse=col_reuse,
            col_coverage=col_coverage,
            intra_block_row_cv=intra_cv,
        )

        block_status: Dict[str, bool] = {}
        for rule in rules:
            if not rule.enabled:
                continue
            block_status[f"passes_{rule.name}"] = rule.predicate(metrics, thresholds)

        is_good = all(block_status.values()) if block_status else True
        if is_good:
            good_blocks += 1

        col_reuse_values.append(col_reuse)
        col_coverage_values.append(col_coverage)
        intra_block_cv_values.append(intra_cv)
        block_nnz_values.append(float(block_nnz))

        per_block_records.append(
            {
                "mode": mode,
                "block_size": block_size,
                "block_idx": blk_idx,
                "block_nnz": block_nnz,
                "unique_cols": unique_cols,
                "col_reuse": col_reuse,
                "col_coverage": col_coverage,
                "intra_block_row_cv": intra_cv,
                "is_good": int(is_good),
                **block_status,
            }
        )

    nonempty_blocks = len(block_data)
    bad_blocks = nonempty_blocks - good_blocks
    aggregate: Dict[str, float] = {
        "blocks_total": float(num_blocks),
        "blocks_nonempty": float(nonempty_blocks),
        "blocks_good": float(good_blocks),
        "blocks_bad": float(bad_blocks),
        "r_block_nonempty": (good_blocks / nonempty_blocks) if nonempty_blocks > 0 else 0.0,
        "r_block_all": (good_blocks / num_blocks) if num_blocks > 0 else 0.0,
        "good_to_bad_ratio": (good_blocks / bad_blocks) if bad_blocks > 0 else math.nan,
    }
    for metric_name, values in (
        ("col_reuse", col_reuse_values),
        ("col_coverage", col_coverage_values),
        ("intra_block_row_cv", intra_block_cv_values),
        ("block_nnz", block_nnz_values),
    ):
        stats = aggregate_metric(values)
        for suffix, value in stats.items():
            aggregate[f"{metric_name}_{suffix}"] = value

    # Also include row-level aggregates
    aggregate.update(compute_row_aggregates(coords, num_rows, num_cols))

    return per_block_records, aggregate


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
        "col_reuse_min": thresholds.col_reuse_min,
        "col_coverage_max": thresholds.col_coverage_max,
        "use_col_reuse": int(thresholds.use_col_reuse),
        "use_col_coverage": int(thresholds.use_col_coverage),
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
        ]
    else:
        csv_files = args.csv_files

    benchmark_rows = load_benchmark_rows(csv_files, impls)
    baseline_map = build_baseline_map(benchmark_rows)

    matrices: Dict[str, MatrixData] = {}
    analysis_cache: Dict[tuple, tuple] = {}  # (matrix_name, block_size, mode) -> (per_block, aggregate)
    dataset_rows: List[Dict] = []
    block_output_rows: List[Dict] = []
    emitted_block_keys: set = set()
    threshold_row = build_threshold_row(thresholds)

    for row in benchmark_rows:
        block_size = parse_1d_block_config(row.get("config", ""))
        if block_size is None:
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
            cache_key = (matrix_name, block_size, mode)
            if cache_key not in analysis_cache:
                analysis_cache[cache_key] = compute_block_analysis(matrix, block_size, mode, thresholds, rules)
            per_block, aggregate = analysis_cache[cache_key]
            mode_aggregates[mode] = aggregate

            emitted_key = (matrix_name, block_size, mode)
            if emitted_key not in emitted_block_keys:
                emitted_block_keys.add(emitted_key)
                for rec in per_block:
                    block_output_rows.append(
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
                "config_kind": "block_1d",
                "block_size": block_size,
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
            "No 1D blocking benchmark rows found. Re-run benchmarks with block_only/block_b<N> configs "
            "or provide CSVs that contain those configs."
        )

    write_csv(dataset_rows, args.output)
    write_csv(block_output_rows, args.blocks_output)
    write_csv([threshold_row], args.thresholds_output)

    print(f"Wrote dataset: {args.output}")
    print(f"Wrote blocks: {args.blocks_output}")
    print(f"Wrote thresholds: {args.thresholds_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
