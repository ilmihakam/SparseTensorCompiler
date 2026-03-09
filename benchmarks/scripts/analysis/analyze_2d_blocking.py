#!/usr/bin/env python3
"""Analyze 2D blocking benchmark rows against sparse tile-quality heuristics."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = SCRIPT_DIR.parent.parent
DEFAULT_ANALYSIS_DIR = BENCHMARKS_DIR / "results" / "analysis"

MODE_AS_IS = "as_is"
MODE_TRANSPOSE = "transpose"
MODES = (MODE_AS_IS, MODE_TRANSPOSE)

BLOCK2D_RE = re.compile(r"^block2d_b(?P<tile_i>\d+)x(?P<tile_j>\d+)$")
ALL2D_RE = re.compile(r"^all2d_(?P<order>[A-Z_]+)_b(?P<tile_i>\d+)x(?P<tile_j>\d+)$")


@dataclass(frozen=True)
class MatrixData:
    rows: int
    cols: int
    coords: List[tuple[int, int]]


@dataclass(frozen=True)
class TileMetrics:
    tile_nnz: int
    density: float
    span: int
    span_frac: float
    row_cv: float
    runs_avg: float


@dataclass(frozen=True)
class MetricRule:
    name: str
    enabled: bool
    predicate: Callable[[TileMetrics, "ThresholdProfile"], bool]


@dataclass(frozen=True)
class ThresholdProfile:
    nnz_min: int
    density_min: float
    span_frac_max: float
    cv_max: float
    runs_min: float
    use_nnz: bool
    use_density: bool
    use_span: bool
    use_cv: bool
    use_runs: bool

    @property
    def profile_id(self) -> str:
        tokens = [
            f"nnz{self.nnz_min}",
            f"d{self._fmt(self.density_min)}",
            f"s{self._fmt(self.span_frac_max)}",
            f"cv{self._fmt(self.cv_max)}",
            f"r{self._fmt(self.runs_min)}",
            f"u{int(self.use_nnz)}{int(self.use_density)}{int(self.use_span)}{int(self.use_cv)}{int(self.use_runs)}",
        ]
        return "_".join(tokens)

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{value:g}".replace(".", "p")


@dataclass
class TileState:
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    tile_nnz: int = 0
    min_col: int | None = None
    max_col: int | None = None
    row_counts: List[int] | None = None
    row_cols: Dict[int, List[int]] | None = None

    def __post_init__(self) -> None:
        rows_in_tile = self.row_end - self.row_start
        self.row_counts = [0 for _ in range(rows_in_tile)]
        self.row_cols = defaultdict(list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze 2D blocking benchmark rows")
    parser.add_argument("--spmm-csv", type=Path, default=BENCHMARKS_DIR / "results" / "csv" / "benchmark_spmm.csv")
    parser.add_argument("--sddmm-csv", type=Path, default=BENCHMARKS_DIR / "results" / "csv" / "benchmark_sddmm.csv")
    parser.add_argument("--canonical-dir", type=Path, default=BENCHMARKS_DIR / "matrices" / "suitesparse" / "canonical")
    parser.add_argument("--impls", default="ours", help="Comma-separated benchmark implementations to include")
    parser.add_argument("--output", type=Path, default=DEFAULT_ANALYSIS_DIR / "2d_blocking_dataset.csv")
    parser.add_argument("--tiles-output", type=Path, default=DEFAULT_ANALYSIS_DIR / "2d_blocking_tiles.csv")
    parser.add_argument(
        "--thresholds-output",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR / "2d_blocking_thresholds_used.csv",
    )
    parser.add_argument("--nnz-min", type=int, default=8)
    parser.add_argument("--density-min", type=float, default=0.05)
    parser.add_argument("--span-frac-max", type=float, default=0.5)
    parser.add_argument("--cv-max", type=float, default=2.0)
    parser.add_argument("--runs-min", type=float, default=1.5)
    parser.add_argument("--use-nnz", dest="use_nnz", action="store_true", default=True)
    parser.add_argument("--no-use-nnz", dest="use_nnz", action="store_false")
    parser.add_argument("--use-density", dest="use_density", action="store_true", default=True)
    parser.add_argument("--no-use-density", dest="use_density", action="store_false")
    parser.add_argument("--use-span", dest="use_span", action="store_true", default=True)
    parser.add_argument("--no-use-span", dest="use_span", action="store_false")
    parser.add_argument("--use-cv", dest="use_cv", action="store_true", default=True)
    parser.add_argument("--no-use-cv", dest="use_cv", action="store_false")
    parser.add_argument("--use-runs", dest="use_runs", action="store_true", default=True)
    parser.add_argument("--no-use-runs", dest="use_runs", action="store_false")
    return parser.parse_args()


def build_threshold_profile(args: argparse.Namespace) -> ThresholdProfile:
    if args.nnz_min < 0:
        raise ValueError("--nnz-min must be >= 0")
    if args.density_min < 0:
        raise ValueError("--density-min must be >= 0")
    if args.span_frac_max < 0:
        raise ValueError("--span-frac-max must be >= 0")
    if args.cv_max < 0:
        raise ValueError("--cv-max must be >= 0")
    if args.runs_min < 0:
        raise ValueError("--runs-min must be >= 0")
    return ThresholdProfile(
        nnz_min=args.nnz_min,
        density_min=args.density_min,
        span_frac_max=args.span_frac_max,
        cv_max=args.cv_max,
        runs_min=args.runs_min,
        use_nnz=args.use_nnz,
        use_density=args.use_density,
        use_span=args.use_span,
        use_cv=args.use_cv,
        use_runs=args.use_runs,
    )


def build_rules(thresholds: ThresholdProfile) -> List[MetricRule]:
    return [
        MetricRule("nnz", thresholds.use_nnz, lambda tile, cfg: tile.tile_nnz >= cfg.nnz_min),
        MetricRule("density", thresholds.use_density, lambda tile, cfg: tile.density >= cfg.density_min),
        MetricRule("span", thresholds.use_span, lambda tile, cfg: tile.span_frac <= cfg.span_frac_max),
        MetricRule("cv", thresholds.use_cv, lambda tile, cfg: tile.row_cv <= cfg.cv_max),
        MetricRule("runs", thresholds.use_runs, lambda tile, cfg: tile.runs_avg >= cfg.runs_min),
    ]


def parse_2d_config(config_name: str) -> Dict[str, object] | None:
    for regex, kind in ((BLOCK2D_RE, "block2d"), (ALL2D_RE, "all2d")):
        match = regex.match(config_name)
        if match:
            info: Dict[str, object] = {
                "is_2d": True,
                "config_kind": kind,
                "tile_i": int(match.group("tile_i")),
                "tile_j": int(match.group("tile_j")),
                "order": match.groupdict().get("order", ""),
            }
            return info
    return None


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


def build_baseline_map(rows: Sequence[Dict[str, str]]) -> Dict[tuple[str, ...], float]:
    baseline_map: Dict[tuple[str, ...], float] = {}
    for row in rows:
        if row.get("config") != "baseline":
            continue
        key = benchmark_key(row)
        baseline_map[key] = float(row["avg_time_ms"])
    return baseline_map


def benchmark_key(row: Dict[str, str]) -> tuple[str, ...]:
    kernel = row["kernel"]
    if kernel == "spmm":
        return (kernel, row["impl"], row["format"], row["matrix_name"], row.get("N", ""))
    if kernel == "sddmm":
        return (kernel, row["impl"], row["format"], row["matrix_name"], row.get("K", ""))
    raise ValueError(f"Unsupported kernel '{kernel}' for 2D analysis")


def load_matrix(path: Path) -> MatrixData:
    with path.open("r", encoding="utf-8") as handle:
        banner = handle.readline()
        if not banner.startswith("%%MatrixMarket"):
            raise ValueError(f"Invalid MatrixMarket banner in {path}")
        rows = cols = -1
        coords: List[tuple[int, int]] = []
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


def aggregate_metric(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.fmean(values)),
        "p50": float(statistics.median(values)),
        "max": float(max(values)),
    }


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


def evaluate_tile(metrics: TileMetrics, thresholds: ThresholdProfile, rules: Sequence[MetricRule]) -> Dict[str, bool]:
    status = {f"passes_{rule.name}": True for rule in rules if rule.enabled}
    for rule in rules:
        if not rule.enabled:
            continue
        status[f"passes_{rule.name}"] = rule.predicate(metrics, thresholds)
    return status


def compute_tile_analysis(
    matrix: MatrixData,
    tile_i: int,
    tile_j: int,
    mode: str,
    thresholds: ThresholdProfile,
    rules: Sequence[MetricRule],
) -> tuple[List[Dict[str, object]], Dict[str, float]]:
    if tile_i <= 0 or tile_j <= 0:
        raise ValueError("Tile sizes must be > 0")

    if mode == MODE_TRANSPOSE:
        rows = matrix.cols
        cols = matrix.rows
        coords = [(col, row) for row, col in matrix.coords]
    else:
        rows = matrix.rows
        cols = matrix.cols
        coords = matrix.coords

    total_tiles = math.ceil(rows / tile_i) * math.ceil(cols / tile_j) if rows and cols else 0
    tile_states: Dict[tuple[int, int], TileState] = {}

    for row, col in coords:
        tile_p = row // tile_i
        tile_q = col // tile_j
        key = (tile_p, tile_q)
        if key not in tile_states:
            row_start = tile_p * tile_i
            col_start = tile_q * tile_j
            tile_states[key] = TileState(
                row_start=row_start,
                row_end=min(row_start + tile_i, rows),
                col_start=col_start,
                col_end=min(col_start + tile_j, cols),
            )
        state = tile_states[key]
        local_row = row - state.row_start
        state.tile_nnz += 1
        state.min_col = col if state.min_col is None else min(state.min_col, col)
        state.max_col = col if state.max_col is None else max(state.max_col, col)
        state.row_counts[local_row] += 1
        state.row_cols[local_row].append(col)

    tile_rows: List[Dict[str, object]] = []
    good_tiles = 0
    good_nnz = 0
    tile_nnz_values: List[float] = []
    density_values: List[float] = []
    span_frac_values: List[float] = []
    row_cv_values: List[float] = []
    runs_values: List[float] = []

    for (tile_p, tile_q), state in sorted(tile_states.items()):
        rows_in_tile = state.row_end - state.row_start
        cols_in_tile = state.col_end - state.col_start
        tile_area = rows_in_tile * cols_in_tile
        density = state.tile_nnz / tile_area if tile_area > 0 else 0.0
        span = 0 if state.min_col is None or state.max_col is None else (state.max_col - state.min_col + 1)
        span_frac = span / cols_in_tile if cols_in_tile > 0 else 0.0
        row_mean = state.tile_nnz / rows_in_tile if rows_in_tile > 0 else 0.0
        row_stddev = statistics.pstdev(state.row_counts) if rows_in_tile > 1 else 0.0
        row_cv = row_stddev / row_mean if row_mean > 0 else 0.0
        runs_avg = 0.0
        if rows_in_tile > 0:
            runs_avg = float(
                statistics.fmean(
                    count_runs(sorted(state.row_cols.get(local_row, [])))
                    for local_row in range(rows_in_tile)
                )
            )

        metrics = TileMetrics(
            tile_nnz=state.tile_nnz,
            density=density,
            span=span,
            span_frac=span_frac,
            row_cv=row_cv,
            runs_avg=runs_avg,
        )
        status = evaluate_tile(metrics, thresholds, rules)
        is_good = all(status.values()) if status else True
        if is_good:
            good_tiles += 1
            good_nnz += state.tile_nnz

        tile_nnz_values.append(float(metrics.tile_nnz))
        density_values.append(metrics.density)
        span_frac_values.append(metrics.span_frac)
        row_cv_values.append(metrics.row_cv)
        runs_values.append(metrics.runs_avg)

        tile_rows.append(
            {
                "mode": mode,
                "tile_i": tile_i,
                "tile_j": tile_j,
                "tile_p": tile_p,
                "tile_q": tile_q,
                "tile_nnz": metrics.tile_nnz,
                "density": metrics.density,
                "span": metrics.span,
                "span_frac": metrics.span_frac,
                "row_cv": metrics.row_cv,
                "runs_avg": metrics.runs_avg,
                "is_good": int(is_good),
                **status,
            }
        )

    nonempty_tiles = len(tile_states)
    empty_tiles = total_tiles - nonempty_tiles
    bad_tiles = nonempty_tiles - good_tiles
    aggregate = {
        "tiles_nonempty": float(nonempty_tiles),
        "tiles_empty": float(empty_tiles),
        "tiles_good": float(good_tiles),
        "tiles_bad": float(bad_tiles),
        "empty_tile_frac": (empty_tiles / total_tiles) if total_tiles > 0 else 0.0,
        "r_tile_nonempty": (good_tiles / nonempty_tiles) if nonempty_tiles > 0 else 0.0,
        "r_tile_all": (good_tiles / total_tiles) if total_tiles > 0 else 0.0,
        "r_nnz_useful": (good_nnz / len(coords)) if coords else 0.0,
        "good_to_bad_ratio": (good_tiles / bad_tiles) if bad_tiles > 0 else math.nan,
    }
    for metric_name, values in (
        ("tile_nnz", tile_nnz_values),
        ("density", density_values),
        ("span_frac", span_frac_values),
        ("row_cv", row_cv_values),
        ("runs_avg", runs_values),
    ):
        stats = aggregate_metric(values)
        for suffix, value in stats.items():
            aggregate[f"{metric_name}_{suffix}"] = value

    return tile_rows, aggregate


def build_threshold_row(thresholds: ThresholdProfile) -> Dict[str, object]:
    return {
        "threshold_profile_id": thresholds.profile_id,
        "nnz_min": thresholds.nnz_min,
        "density_min": thresholds.density_min,
        "span_frac_max": thresholds.span_frac_max,
        "cv_max": thresholds.cv_max,
        "runs_min": thresholds.runs_min,
        "use_nnz": int(thresholds.use_nnz),
        "use_density": int(thresholds.use_density),
        "use_span": int(thresholds.use_span),
        "use_cv": int(thresholds.use_cv),
        "use_runs": int(thresholds.use_runs),
    }


def with_mode_prefix(values: Dict[str, float], mode: str) -> Dict[str, float]:
    return {f"{key}_{mode}": value for key, value in values.items()}


def selected_mode(format_name: str) -> str:
    if format_name == "csc":
        return MODE_TRANSPOSE
    return MODE_AS_IS


def safe_speedup(baseline_time: float, candidate_time: float) -> float:
    if baseline_time <= 0 or candidate_time <= 0:
        return math.nan
    return baseline_time / candidate_time


def write_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames: List[str] = []
    seen = set()
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


def main() -> int:
    args = parse_args()
    thresholds = build_threshold_profile(args)
    rules = build_rules(thresholds)
    impls = parse_csv_list(args.impls)
    benchmark_rows = load_benchmark_rows([args.spmm_csv, args.sddmm_csv], impls)
    baseline_map = build_baseline_map(benchmark_rows)

    matrices: Dict[str, MatrixData] = {}
    analysis_cache: Dict[tuple[str, int, int, str], tuple[List[Dict[str, object]], Dict[str, float]]] = {}
    dataset_rows: List[Dict[str, object]] = []
    tile_output_rows: List[Dict[str, object]] = []
    emitted_tile_keys: set[tuple[str, str, int, int, str]] = set()
    threshold_row = build_threshold_row(thresholds)

    for row in benchmark_rows:
        if row["kernel"] not in {"spmm", "sddmm"}:
            continue
        config_info = parse_2d_config(row["config"])
        if not config_info:
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
            cache_key = (matrix_name, int(config_info["tile_i"]), int(config_info["tile_j"]), mode)
            if cache_key not in analysis_cache:
                analysis_cache[cache_key] = compute_tile_analysis(
                    matrix,
                    int(config_info["tile_i"]),
                    int(config_info["tile_j"]),
                    mode,
                    thresholds,
                    rules,
                )
            tile_rows, aggregate = analysis_cache[cache_key]
            mode_aggregates[mode] = aggregate
            emitted_key = (row["kernel"], matrix_name, int(config_info["tile_i"]), int(config_info["tile_j"]), mode)
            if emitted_key not in emitted_tile_keys:
                emitted_tile_keys.add(emitted_key)
                for tile_row in tile_rows:
                    tile_output_rows.append(
                        {
                            "threshold_profile_id": thresholds.profile_id,
                            "kernel": row["kernel"],
                            "matrix_name": matrix_name,
                            **tile_row,
                        }
                    )

        chosen_mode = selected_mode(row["format"])
        dataset_row: Dict[str, object] = dict(row)
        dataset_row.update(
            {
                "threshold_profile_id": thresholds.profile_id,
                "config_kind": config_info["config_kind"],
                "order": config_info["order"],
                "tile_i": config_info["tile_i"],
                "tile_j": config_info["tile_j"],
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
            "No 2D benchmark rows found to analyze. Re-run spmm/sddmm benchmarks with --sweep-block-2d "
            "or provide CSVs that contain block2d/all2d configs."
        )

    write_csv(dataset_rows, args.output)
    write_csv(tile_output_rows, args.tiles_output)
    write_csv([threshold_row], args.thresholds_output)

    print(f"Wrote dataset: {args.output}")
    print(f"Wrote tiles: {args.tiles_output}")
    print(f"Wrote thresholds: {args.thresholds_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
