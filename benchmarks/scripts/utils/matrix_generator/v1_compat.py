"""V1 structured_random generator — moved verbatim from generate_matrices.py."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence, Tuple

from .registry import register
from .types import GeneratedMatrix, MatrixSpec


def _parse_value_mode(value_mode: str) -> str:
    mode = value_mode.strip().lower()
    if mode not in ("ones", "uniform[-1,1]"):
        raise ValueError(f"Unsupported value_mode '{value_mode}'")
    return mode


def _sample_value(rng: random.Random, value_mode: str) -> float:
    if value_mode == "ones":
        return 1.0
    return rng.uniform(-1.0, 1.0)


def _resolve_total_nnz(nnz_cfg: Dict[str, object], rows: int, cols: int) -> int:
    mode = str(nnz_cfg.get("mode", "density")).strip().lower()
    if mode == "density":
        density = float(nnz_cfg.get("density", 0.001))
        if density <= 0 or density > 1:
            raise ValueError("nnz.density must be in (0, 1]")
        total = int(round(density * rows * cols))
    elif mode == "avg_nnz_row":
        avg_nnz_row = float(nnz_cfg.get("avg_nnz_row", 1.0))
        if avg_nnz_row < 0:
            raise ValueError("nnz.avg_nnz_row must be >= 0")
        total = int(round(avg_nnz_row * rows))
    elif mode == "nnz_total":
        total = int(nnz_cfg.get("nnz_total", 0))
    else:
        raise ValueError(f"Unsupported nnz.mode '{mode}'")
    total = max(0, min(total, rows * cols))
    return total


def _build_row_targets(rng: random.Random, nnz_cfg: Dict[str, object], rows: int, cols: int) -> List[int]:
    total_nnz = _resolve_total_nnz(nnz_cfg, rows, cols)
    if rows <= 0:
        return []

    distribution = str(nnz_cfg.get("row_distribution", "fixed")).strip().lower()
    min_per_row = int(nnz_cfg.get("min_per_row", 0))
    max_per_row = int(nnz_cfg.get("max_per_row", cols))
    min_per_row = max(0, min(min_per_row, cols))
    max_per_row = max(min_per_row, min(max_per_row, cols))
    mean = float(total_nnz) / float(rows)

    weights: List[float] = []
    if distribution == "fixed":
        weights = [1.0] * rows
    elif distribution == "lognormal":
        row_cv = float(nnz_cfg.get("row_cv", 1.0))
        row_cv = max(row_cv, 0.0)
        sigma = math.sqrt(math.log(row_cv * row_cv + 1.0))
        mu = math.log(mean + 1e-9) - 0.5 * sigma * sigma
        for _ in range(rows):
            weights.append(max(1e-9, rng.lognormvariate(mu, sigma)))
    elif distribution == "poisson":
        lam = max(mean, 0.01)
        for _ in range(rows):
            # Sample from Poisson using inverse-transform (ok for moderate lambda)
            if lam < 30:
                L = math.exp(-lam)
                k = 0
                p = 1.0
                while True:
                    k += 1
                    p *= rng.random()
                    if p < L:
                        break
                weights.append(float(max(k - 1, 0)))
            else:
                # Normal approximation for large lambda
                weights.append(max(0.0, rng.gauss(lam, math.sqrt(lam))))
    else:
        raise ValueError(f"Unsupported nnz.row_distribution '{distribution}'")

    weight_sum = sum(weights)
    if weight_sum <= 0:
        weights = [1.0] * rows
        weight_sum = float(rows)

    raw = [float(total_nnz) * (weight / weight_sum) for weight in weights]
    targets = [int(math.floor(value)) for value in raw]
    frac = sorted(((raw[i] - float(targets[i]), i) for i in range(rows)), reverse=True)
    missing = total_nnz - sum(targets)
    for _, idx in frac[:missing]:
        targets[idx] += 1

    clamped = [max(min_per_row, min(max_per_row, value)) for value in targets]
    current = sum(clamped)
    if current < total_nnz:
        order = sorted(range(rows), key=lambda i: clamped[i])
        for idx in order:
            if current >= total_nnz:
                break
            cap = max_per_row - clamped[idx]
            if cap <= 0:
                continue
            add = min(cap, total_nnz - current)
            clamped[idx] += add
            current += add
    elif current > total_nnz:
        order = sorted(range(rows), key=lambda i: clamped[i], reverse=True)
        for idx in order:
            if current <= total_nnz:
                break
            reducible = clamped[idx] - min_per_row
            if reducible <= 0:
                continue
            dec = min(reducible, current - total_nnz)
            clamped[idx] -= dec
            current -= dec

    return clamped


def _support_window(row: int, rows: int, cols: int, support_cfg: Dict[str, object], rng: random.Random) -> Tuple[int, int]:
    mode = str(support_cfg.get("mode", "global")).strip().lower()
    if mode == "global":
        return 0, cols - 1
    if mode != "banded":
        raise ValueError(f"Unsupported support.mode '{mode}'")
    bandwidth = int(support_cfg.get("bandwidth", max(1, cols // 16)))
    bandwidth = max(0, bandwidth)
    if rows == 1:
        center = 0
    else:
        center = int(round((float(row) / float(rows - 1)) * float(cols - 1)))
    jitter = int(support_cfg.get("diagonal_jitter", 0))
    if jitter > 0:
        center += rng.randint(-jitter, jitter)
    center = max(0, min(cols - 1, center))
    left = max(0, center - bandwidth)
    right = min(cols - 1, center + bandwidth)
    return left, right


def _sample_run_length(rng: random.Random, avg_run_length: float) -> int:
    avg_run_length = max(1.0, avg_run_length)
    probability = 1.0 / avg_run_length
    run_len = 1
    while run_len < 1024 and rng.random() > probability:
        run_len += 1
    return run_len


def _maybe_pick_hotspot(
    rng: random.Random, left: int, right: int, hotspots: Sequence[int], hotspot_prob: float, hotspot_spread: int
) -> int:
    if hotspots and rng.random() < hotspot_prob:
        base = hotspots[rng.randrange(len(hotspots))]
        if hotspot_spread > 0:
            base += rng.randint(-hotspot_spread, hotspot_spread)
        return max(left, min(right, base))
    return rng.randint(left, right)


def _fill_row(
    rng: random.Random,
    row: int,
    target_nnz: int,
    rows: int,
    cols: int,
    support_cfg: Dict[str, object],
    clustering_cfg: Dict[str, object],
    columns_cfg: Dict[str, object],
    prev_rows: Sequence[set],
    similarity_cfg: Dict[str, object],
) -> set:
    row_cols: set = set()
    if target_nnz <= 0:
        return row_cols

    left, right = _support_window(row, rows, cols, support_cfg, rng)
    if left > right:
        return row_cols

    mode = str(columns_cfg.get("mode", "uniform")).strip().lower()
    hotspots: List[int] = []
    if mode == "hotspots":
        hotspot_count = int(columns_cfg.get("hotspot_count", min(64, max(1, (right - left + 1) // 4))))
        hotspot_count = max(1, min(hotspot_count, right - left + 1))
        hotspot_prob = float(columns_cfg.get("hotspot_prob", 0.5))
        hotspot_prob = max(0.0, min(1.0, hotspot_prob))
        hotspot_spread = int(columns_cfg.get("hotspot_spread", 2))
        hotspots = sorted(rng.sample(range(left, right + 1), hotspot_count))
    elif mode == "uniform":
        hotspot_prob = 0.0
        hotspot_spread = 0
    else:
        raise ValueError(f"Unsupported columns.mode '{mode}'")

    sim_mode = str(similarity_cfg.get("mode", "none")).strip().lower()
    if sim_mode == "window_share":
        window = max(1, int(similarity_cfg.get("window", 4)))
        share_prob = max(0.0, min(1.0, float(similarity_cfg.get("share_prob", 0.2))))
        if prev_rows and rng.random() < share_prob:
            start = max(0, len(prev_rows) - window)
            candidate_rows = list(prev_rows[start:])
            if candidate_rows:
                inherited = list(candidate_rows[rng.randrange(len(candidate_rows))])
                rng.shuffle(inherited)
                for col in inherited:
                    if left <= col <= right:
                        row_cols.add(col)
                        if len(row_cols) >= target_nnz:
                            return row_cols

    cluster_mode = str(clustering_cfg.get("mode", "runs")).strip().lower()
    if cluster_mode != "runs":
        raise ValueError(f"Unsupported clustering.mode '{cluster_mode}'")
    avg_run_length = float(clustering_cfg.get("avg_run_length", 3.0))
    avg_gap = float(clustering_cfg.get("avg_gap", 4.0))
    gap_scale = max(1, int(round(max(avg_gap, 0.0))))

    while len(row_cols) < target_nnz:
        run_len = _sample_run_length(rng, avg_run_length)
        start_col = _maybe_pick_hotspot(rng, left, right, hotspots, hotspot_prob, hotspot_spread)
        for offset in range(run_len):
            col = start_col + offset
            if col > right:
                break
            row_cols.add(col)
            if len(row_cols) >= target_nnz:
                break
        if len(row_cols) >= target_nnz:
            break
        if right - left + 1 <= len(row_cols):
            break
        start_col = _maybe_pick_hotspot(rng, left, right, hotspots, hotspot_prob, hotspot_spread)
        start_col = min(right, start_col + rng.randint(0, gap_scale))
        row_cols.add(start_col)
    return row_cols


def _apply_block_overlay(
    rng: random.Random,
    rows: int,
    cols: int,
    per_row_cols: List[set],
    block_cfg: Dict[str, object],
) -> None:
    if not bool(block_cfg.get("enabled", False)):
        return
    block_rows = max(1, int(block_cfg.get("block_rows", 8)))
    block_cols = max(1, int(block_cfg.get("block_cols", 8)))
    block_density = max(0.0, min(1.0, float(block_cfg.get("block_density", 0.5))))
    block_prob = max(0.0, min(1.0, float(block_cfg.get("block_prob", 0.0))))

    for row_start in range(0, rows, block_rows):
        for col_start in range(0, cols, block_cols):
            if rng.random() > block_prob:
                continue
            row_end = min(rows, row_start + block_rows)
            col_end = min(cols, col_start + block_cols)
            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    if rng.random() <= block_density:
                        per_row_cols[row].add(col)


def generate_matrix_from_dict(matrix_spec: Dict[str, object]) -> Tuple[int, int, List[Tuple[int, int, float]], Dict[str, object]]:
    """Generate a matrix from a v1-style dict spec. Returns (rows, cols, entries, params)."""
    generator = str(matrix_spec.get("generator", "structured_random_v1"))
    if generator != "structured_random_v1":
        raise ValueError(f"Unsupported generator '{generator}'")

    rows = int(matrix_spec["rows"])
    cols = int(matrix_spec["cols"])
    seed = int(matrix_spec.get("seed", 0))
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be > 0")

    value_mode = _parse_value_mode(str(matrix_spec.get("value_mode", "ones")))
    nnz_cfg = dict(matrix_spec.get("nnz", {}))
    support_cfg = dict(matrix_spec.get("support", {}))
    clustering_cfg = dict(matrix_spec.get("clustering", {}))
    columns_cfg = dict(matrix_spec.get("columns", {}))
    similarity_cfg = dict(matrix_spec.get("inter_row_similarity", {}))
    block_cfg = dict(matrix_spec.get("block_structure", {}))

    rng = random.Random(seed)
    row_targets = _build_row_targets(rng, nnz_cfg, rows, cols)
    per_row_cols: List[set] = []

    for row in range(rows):
        row_cols = _fill_row(
            rng,
            row,
            row_targets[row],
            rows,
            cols,
            support_cfg,
            clustering_cfg,
            columns_cfg,
            per_row_cols,
            similarity_cfg,
        )
        per_row_cols.append(row_cols)

    _apply_block_overlay(rng, rows, cols, per_row_cols, block_cfg)

    entries: List[Tuple[int, int, float]] = []
    for row, cols_set in enumerate(per_row_cols):
        for col in sorted(cols_set):
            entries.append((row, col, _sample_value(rng, value_mode)))

    params = {
        "value_mode": value_mode,
        "nnz": nnz_cfg,
        "support": support_cfg,
        "clustering": clustering_cfg,
        "columns": columns_cfg,
        "inter_row_similarity": similarity_cfg,
        "block_structure": block_cfg,
    }
    return rows, cols, entries, params


@register("v1_structured_random")
def generate_v1(spec: MatrixSpec, rng: random.Random) -> GeneratedMatrix:
    """Registry-compatible wrapper around the v1 generation logic."""
    rows, cols, entries, params = generate_matrix_from_dict(spec.family_params)
    return GeneratedMatrix(rows=rows, cols=cols, entries=entries, params=params)
