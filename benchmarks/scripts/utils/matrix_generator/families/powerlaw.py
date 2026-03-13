"""F6: Power-law row/column distributions."""

from __future__ import annotations

import math
import random
from typing import List

from ..registry import register
from ..types import COOEntries, GeneratedMatrix, MatrixSpec


def _sample_value(rng: random.Random, value_mode: str) -> float:
    if value_mode == "ones":
        return 1.0
    return rng.uniform(-1.0, 1.0)


def _zipf_sample(n: int, alpha: float, avg_target: float, rng: random.Random) -> List[int]:
    """Generate n values following an approximate Zipf distribution, scaled to hit avg_target on average."""
    raw = []
    for _ in range(n):
        # Zipf via inverse CDF: draw uniform, invert power law
        u = rng.random()
        raw.append(max(1.0, (1.0 - u) ** (-1.0 / max(alpha, 0.01))))

    # Scale to target total
    raw_sum = sum(raw)
    if raw_sum <= 0:
        return [max(1, int(avg_target))] * n
    scale = avg_target * n / raw_sum
    targets = [max(0, int(round(v * scale))) for v in raw]
    return targets


def _fill_row_random(row: int, nnz: int, cols: int, rng: random.Random,
                     value_mode: str, col_left: int = 0, col_right: int = -1) -> COOEntries:
    """Place nnz random entries in a single row within [col_left, col_right]."""
    if col_right < 0:
        col_right = cols - 1
    span = col_right - col_left + 1
    if span <= 0 or nnz <= 0:
        return []
    nnz = min(nnz, span)
    chosen = rng.sample(range(col_left, col_right + 1), nnz)
    return [(row, c, _sample_value(rng, value_mode)) for c in sorted(chosen)]


@register("powerlaw")
def generate_powerlaw(spec: MatrixSpec, rng: random.Random) -> GeneratedMatrix:
    fp = spec.family_params
    rows, cols = spec.rows, spec.cols
    vm = spec.value_mode
    v = spec.variant

    alpha = float(fp.get("alpha", 1.2))
    avg_nnz_row = float(fp.get("avg_nnz_row", 10))

    if v == "row_zipf":
        row_nnz = _zipf_sample(rows, alpha, avg_nnz_row, rng)
        entries: COOEntries = []
        for r, nnz in enumerate(row_nnz):
            entries.extend(_fill_row_random(r, nnz, cols, rng, vm))

    elif v == "col_zipf":
        col_nnz = _zipf_sample(cols, alpha, avg_nnz_row, rng)
        entries: COOEntries = []
        for c, nnz in enumerate(col_nnz):
            nnz = min(nnz, rows)
            if nnz <= 0:
                continue
            chosen_rows = rng.sample(range(rows), nnz)
            for r in chosen_rows:
                entries.append((r, c, _sample_value(rng, vm)))

    elif v == "row_col_zipf":
        row_nnz = _zipf_sample(rows, alpha, avg_nnz_row, rng)
        # Bias column selection toward power-law popular columns
        col_weights = _zipf_sample(cols, alpha, 1.0, rng)
        col_weights_sum = sum(col_weights) or 1
        col_probs = [w / col_weights_sum for w in col_weights]
        entries: COOEntries = []
        for r, nnz in enumerate(row_nnz):
            nnz = min(nnz, cols)
            if nnz <= 0:
                continue
            # Weighted sample without replacement (approximate)
            chosen = set()
            for _ in range(nnz * 3):
                if len(chosen) >= nnz:
                    break
                # Roulette selection
                u = rng.random()
                cumulative = 0.0
                for c, p in enumerate(col_probs):
                    cumulative += p
                    if u <= cumulative:
                        chosen.add(c)
                        break
            for c in sorted(chosen):
                entries.append((r, c, _sample_value(rng, vm)))

    elif v == "super_rows":
        # Few super-dense rows, rest very sparse
        num_super = int(fp.get("num_super", 5))
        super_nnz = int(fp.get("super_nnz", cols // 2))
        sparse_nnz = int(fp.get("sparse_nnz", 2))
        super_rows_set = set(rng.sample(range(rows), min(num_super, rows)))
        entries: COOEntries = []
        for r in range(rows):
            nnz = super_nnz if r in super_rows_set else sparse_nnz
            entries.extend(_fill_row_random(r, nnz, cols, rng, vm))

    elif v == "super_cols":
        num_super = int(fp.get("num_super", 5))
        super_nnz = int(fp.get("super_nnz", rows // 2))
        sparse_nnz = int(fp.get("sparse_nnz", 2))
        super_cols_set = set(rng.sample(range(cols), min(num_super, cols)))
        entries: COOEntries = []
        for c in range(cols):
            nnz = super_nnz if c in super_cols_set else sparse_nnz
            nnz = min(nnz, rows)
            if nnz <= 0:
                continue
            chosen = rng.sample(range(rows), nnz)
            for r in chosen:
                entries.append((r, c, _sample_value(rng, vm)))

    elif v == "banded_powerlaw":
        bandwidth = int(fp.get("bandwidth", cols // 10))
        row_nnz = _zipf_sample(rows, alpha, avg_nnz_row, rng)
        entries: COOEntries = []
        for r, nnz in enumerate(row_nnz):
            center = int(round(float(r) / max(1, rows - 1) * max(1, cols - 1)))
            left = max(0, center - bandwidth)
            right = min(cols - 1, center + bandwidth)
            entries.extend(_fill_row_random(r, nnz, cols, rng, vm, left, right))

    elif v == "shared_span_powerlaw":
        # Heavy rows share same column span
        row_nnz = _zipf_sample(rows, alpha, avg_nnz_row, rng)
        span_center = cols // 2
        span_width = int(fp.get("span_width", cols // 3))
        left = max(0, span_center - span_width // 2)
        right = min(cols - 1, span_center + span_width // 2)
        entries: COOEntries = []
        for r, nnz in enumerate(row_nnz):
            entries.extend(_fill_row_random(r, nnz, cols, rng, vm, left, right))

    elif v == "disjoint_span_powerlaw":
        # Heavy rows have disjoint column spans
        row_nnz = _zipf_sample(rows, alpha, avg_nnz_row, rng)
        entries: COOEntries = []
        num_spans = int(fp.get("num_spans", 4))
        span_width = cols // num_spans
        for r, nnz in enumerate(row_nnz):
            span_idx = r % num_spans
            left = span_idx * span_width
            right = min(cols - 1, left + span_width - 1)
            entries.extend(_fill_row_random(r, nnz, cols, rng, vm, left, right))

    elif v == "aspect_skewed":
        row_nnz = _zipf_sample(rows, alpha, avg_nnz_row, rng)
        entries: COOEntries = []
        for r, nnz in enumerate(row_nnz):
            entries.extend(_fill_row_random(r, nnz, cols, rng, vm))

    else:
        raise ValueError(f"Unknown powerlaw variant '{v}'")

    entries.sort()
    # Deduplicate
    deduped: COOEntries = []
    seen = set()
    for r, c, val in entries:
        if (r, c) not in seen:
            deduped.append((r, c, val))
            seen.add((r, c))

    return GeneratedMatrix(rows=rows, cols=cols, entries=deduped, params={"variant": v, **fp})
