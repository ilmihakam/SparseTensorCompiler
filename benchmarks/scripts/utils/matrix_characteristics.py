#!/usr/bin/env python3
"""MatrixMarket sparse characteristics used by generation and benchmark metadata."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


CHARACTERISTIC_BASE_NAMES = [
    "m_rows",
    "n_cols",
    "aspect_ratio",
    "nnz",
    "density",
    "avg_nnz_row",
    "avg_nnz_col",
    "row_variance",
    "row_cv",
    "max_row_length",
    "min_row_length",
    "nnz_row_min",
    "nnz_row_max",
    "col_variance",
    "col_cv",
    "nnz_col_min",
    "nnz_col_max",
    "bandwidth",
    "avg_row_span",
    "avg_col_span",
    "avg_runs_per_row",
    "avg_run_length",
    "avg_gap",
    "frac_adjacent",
    "normalized_entropy",
    "avg_entry_matches",
    "jaccard_adjacent_rows_avg",
]

CHARACTERISTIC_NAMES = [f"char_{name}" for name in CHARACTERISTIC_BASE_NAMES]


def zero_characteristics(*, prefix: str = "char_") -> Dict[str, float]:
    return {f"{prefix}{name}": 0.0 for name in CHARACTERISTIC_BASE_NAMES}


def read_matrix_coords(matrix_file: Path) -> Tuple[int, int, List[Tuple[int, int]]]:
    with matrix_file.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        if not first:
            raise ValueError(f"Empty matrix file: {matrix_file}")
        banner = first.strip().split()
        if len(banner) != 5 or banner[0] != "%%MatrixMarket":
            raise ValueError(f"Invalid MatrixMarket banner in {matrix_file}")
        if banner[1].lower() != "matrix" or banner[2].lower() != "coordinate":
            raise ValueError(f"Only coordinate MatrixMarket files are supported: {matrix_file}")

        dims_line = ""
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            dims_line = stripped
            break
        if not dims_line:
            raise ValueError(f"Missing dimensions line in {matrix_file}")
        dims = dims_line.split()
        if len(dims) < 3:
            raise ValueError(f"Invalid dimensions line in {matrix_file}: {dims_line}")
        rows = int(dims[0])
        cols = int(dims[1])
        nnz_declared = int(dims[2])

        coords: List[Tuple[int, int]] = []
        while len(coords) < nnz_declared:
            line = handle.readline()
            if line == "":
                raise ValueError(f"Unexpected EOF while reading entries in {matrix_file}")
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid entry in {matrix_file}: {stripped}")
            r = int(parts[0]) - 1
            c = int(parts[1]) - 1
            if r < 0 or r >= rows or c < 0 or c >= cols:
                raise ValueError(f"Entry out of bounds in {matrix_file}: {parts[0]} {parts[1]}")
            coords.append((r, c))
    return rows, cols, coords


def _variance(values: Sequence[int], mean: float) -> float:
    if not values:
        return 0.0
    total = 0.0
    for value in values:
        diff = float(value) - mean
        total += diff * diff
    return total / float(len(values))


def _safe_cv(variance: float, mean: float) -> float:
    if mean <= 0.0:
        return 0.0
    return math.sqrt(max(variance, 0.0)) / mean


def _average(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def compute_characteristics_from_coords(
    rows: int, cols: int, coords: Sequence[Tuple[int, int]], *, prefix: str = "char_"
) -> Dict[str, float]:
    nnz = len(coords)
    result = zero_characteristics(prefix=prefix)
    if rows <= 0 or cols <= 0:
        result[f"{prefix}m_rows"] = float(rows)
        result[f"{prefix}n_cols"] = float(cols)
        return result

    row_cols: List[List[int]] = [[] for _ in range(rows)]
    col_rows: List[List[int]] = [[] for _ in range(cols)]

    for row, col in coords:
        row_cols[row].append(col)
        col_rows[col].append(row)

    row_nnz = [len(items) for items in row_cols]
    col_nnz = [len(items) for items in col_rows]

    avg_nnz_row = float(nnz) / float(rows)
    avg_nnz_col = float(nnz) / float(cols)

    row_variance = _variance(row_nnz, avg_nnz_row)
    col_variance = _variance(col_nnz, avg_nnz_col)
    row_cv = _safe_cv(row_variance, avg_nnz_row)
    col_cv = _safe_cv(col_variance, avg_nnz_col)

    max_row_len = float(max(row_nnz) if row_nnz else 0)
    min_row_len = float(min(row_nnz) if row_nnz else 0)
    max_col_len = float(max(col_nnz) if col_nnz else 0)
    min_col_len = float(min(col_nnz) if col_nnz else 0)

    bandwidth = 0
    row_spans: List[float] = []
    col_spans: List[float] = []
    total_runs = 0.0
    total_run_length = 0.0
    total_gaps = 0.0
    gap_sum = 0.0
    adjacent_count = 0.0

    row_sets: List[set[int]] = [set(items) for items in row_cols]
    jaccard_values: List[float] = []

    for idx, cols_in_row in enumerate(row_cols):
        if idx > 0:
            left = row_sets[idx - 1]
            right = row_sets[idx]
            union = left | right
            if union:
                jaccard_values.append(float(len(left & right)) / float(len(union)))
        if not cols_in_row:
            continue
        ordered = sorted(cols_in_row)
        row_spans.append(float(ordered[-1] - ordered[0] + 1))
        runs = 1
        for pos in range(1, len(ordered)):
            gap = ordered[pos] - ordered[pos - 1]
            if gap != 1:
                runs += 1
            else:
                adjacent_count += 1.0
            gap_sum += float(gap)
            total_gaps += 1.0
        total_runs += float(runs)
        total_run_length += float(len(ordered))

    for col, rows_in_col in enumerate(col_rows):
        if not rows_in_col:
            continue
        ordered = sorted(rows_in_col)
        col_spans.append(float(ordered[-1] - ordered[0] + 1))
        for row in ordered:
            distance = abs(row - col)
            if distance > bandwidth:
                bandwidth = distance

    entropy = 0.0
    if nnz > 0:
        for count in col_nnz:
            if count <= 0:
                continue
            probability = float(count) / float(nnz)
            entropy -= probability * math.log(probability)
    norm_entropy = entropy / math.log(cols) if cols > 1 else 0.0

    col_pair_sum = 0.0
    for count in col_nnz:
        col_pair_sum += float(count) * float(count - 1) / 2.0
    avg_entry_matches = 0.0
    if rows > 1:
        avg_entry_matches = 2.0 * col_pair_sum / (float(rows) * float(rows - 1))

    result.update(
        {
            f"{prefix}m_rows": float(rows),
            f"{prefix}n_cols": float(cols),
            f"{prefix}aspect_ratio": float(rows) / float(cols),
            f"{prefix}nnz": float(nnz),
            f"{prefix}density": float(nnz) / (float(rows) * float(cols)),
            f"{prefix}avg_nnz_row": avg_nnz_row,
            f"{prefix}avg_nnz_col": avg_nnz_col,
            f"{prefix}row_variance": row_variance,
            f"{prefix}row_cv": row_cv,
            f"{prefix}max_row_length": max_row_len,
            f"{prefix}min_row_length": min_row_len,
            f"{prefix}nnz_row_min": min_row_len,
            f"{prefix}nnz_row_max": max_row_len,
            f"{prefix}col_variance": col_variance,
            f"{prefix}col_cv": col_cv,
            f"{prefix}nnz_col_min": min_col_len,
            f"{prefix}nnz_col_max": max_col_len,
            f"{prefix}bandwidth": float(bandwidth),
            f"{prefix}avg_row_span": _average(row_spans),
            f"{prefix}avg_col_span": _average(col_spans),
            f"{prefix}avg_runs_per_row": (total_runs / float(rows)) if rows > 0 else 0.0,
            f"{prefix}avg_run_length": (total_run_length / total_runs) if total_runs > 0 else 0.0,
            f"{prefix}avg_gap": (gap_sum / total_gaps) if total_gaps > 0 else 0.0,
            f"{prefix}frac_adjacent": (adjacent_count / total_gaps) if total_gaps > 0 else 0.0,
            f"{prefix}normalized_entropy": norm_entropy,
            f"{prefix}avg_entry_matches": avg_entry_matches,
            f"{prefix}jaccard_adjacent_rows_avg": _average(jaccard_values),
        }
    )
    return result


def compute_characteristics_from_file(matrix_file: Path, *, prefix: str = "char_") -> Dict[str, float]:
    rows, cols, coords = read_matrix_coords(matrix_file)
    return compute_characteristics_from_coords(rows, cols, coords, prefix=prefix)
