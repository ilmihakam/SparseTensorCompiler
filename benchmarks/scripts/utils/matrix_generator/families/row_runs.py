"""F4+F5: Row-clustered long runs / Column-clustered (via transpose)."""

from __future__ import annotations

import random
from typing import List

from ..registry import register
from ..types import COOEntries, GeneratedMatrix, MatrixSpec


def _sample_value(rng: random.Random, value_mode: str) -> float:
    if value_mode == "ones":
        return 1.0
    return rng.uniform(-1.0, 1.0)


def _place_segments(rows: int, cols: int, segments: List[dict], rng: random.Random, value_mode: str) -> COOEntries:
    """Place contiguous segments per row.

    Each entry in `segments` is a dict with:
      count: number of segments
      length: segment length (or "bimodal" for mixed)
      gap: gap between segments (0 = random placement)
      length_short/length_long/long_fraction: for bimodal
    """
    entries: COOEntries = []
    for row in range(rows):
        placed_cols: set = set()
        for seg_spec in segments:
            count = int(seg_spec.get("count", 1))
            for _ in range(count):
                length_mode = seg_spec.get("length", 50)
                if isinstance(length_mode, str) and length_mode == "bimodal":
                    short = int(seg_spec.get("length_short", 10))
                    long = int(seg_spec.get("length_long", 100))
                    frac = float(seg_spec.get("long_fraction", 0.5))
                    seg_len = long if rng.random() < frac else short
                else:
                    seg_len = int(length_mode)

                gap = int(seg_spec.get("gap", 0))
                if gap > 0 and placed_cols:
                    last_col = max(placed_cols)
                    start = last_col + gap
                else:
                    start = rng.randint(0, max(0, cols - seg_len))

                start = max(0, min(start, cols - 1))
                for c in range(start, min(start + seg_len, cols)):
                    placed_cols.add(c)

        for c in sorted(placed_cols):
            entries.append((row, c, _sample_value(rng, value_mode)))

    entries.sort()
    return entries


@register("row_runs")
def generate_row_runs(spec: MatrixSpec, rng: random.Random) -> GeneratedMatrix:
    fp = spec.family_params
    rows, cols = spec.rows, spec.cols
    vm = spec.value_mode
    v = spec.variant

    if v == "single_segment":
        length = int(fp.get("length", 50))
        entries = _place_segments(rows, cols, [{"count": 1, "length": length}], rng, vm)

    elif v == "long_segment":
        length = int(fp.get("length", 200))
        entries = _place_segments(rows, cols, [{"count": 1, "length": length}], rng, vm)

    elif v == "double_segment_fixed_gap":
        length = int(fp.get("length", 25))
        gap = int(fp.get("gap", 50))
        entries = _place_segments(rows, cols, [
            {"count": 1, "length": length},
            {"count": 1, "length": length, "gap": gap},
        ], rng, vm)

    elif v == "bimodal_lengths":
        entries = _place_segments(rows, cols, [{
            "count": 1,
            "length": "bimodal",
            "length_short": int(fp.get("length_short", 10)),
            "length_long": int(fp.get("length_long", 100)),
            "long_fraction": float(fp.get("long_fraction", 0.5)),
        }], rng, vm)

    elif v == "grouped_bimodal":
        # Top fraction gets long runs, rest get short
        long_frac = float(fp.get("long_fraction", 0.1))
        short_len = int(fp.get("length_short", 10))
        long_len = int(fp.get("length_long", 100))
        entries: COOEntries = []
        for row in range(rows):
            is_long = (row < int(rows * long_frac))
            seg_len = long_len if is_long else short_len
            start = rng.randint(0, max(0, cols - seg_len))
            for c in range(start, min(start + seg_len, cols)):
                entries.append((row, c, _sample_value(rng, vm)))
        entries.sort()

    elif v == "shared_overlap":
        # All rows share overlapping column ranges
        length = int(fp.get("length", 50))
        center = cols // 2
        spread = int(fp.get("spread", length // 2))
        entries: COOEntries = []
        for row in range(rows):
            start = center - spread + rng.randint(0, spread)
            start = max(0, min(start, cols - length))
            for c in range(start, min(start + length, cols)):
                entries.append((row, c, _sample_value(rng, vm)))
        entries.sort()

    elif v == "staggered_disjoint":
        # Runs staggered to minimize overlap
        length = int(fp.get("length", 50))
        entries: COOEntries = []
        for row in range(rows):
            start = (row * length) % max(1, cols - length)
            for c in range(start, min(start + length, cols)):
                entries.append((row, c, _sample_value(rng, vm)))
        entries.sort()

    elif v == "wide_span":
        # Runs span large fraction of columns
        frac = float(fp.get("span_fraction", 0.8))
        length = max(1, int(cols * frac))
        entries: COOEntries = []
        for row in range(rows):
            start = rng.randint(0, max(0, cols - length))
            for c in range(start, min(start + length, cols)):
                entries.append((row, c, _sample_value(rng, vm)))
        entries.sort()

    elif v == "narrow_span":
        # Runs confined to small column interval
        frac = float(fp.get("span_fraction", 0.05))
        length = max(1, int(cols * frac))
        center = cols // 2
        entries: COOEntries = []
        for row in range(rows):
            start = center - length // 2 + rng.randint(-length // 4, length // 4)
            start = max(0, min(start, cols - length))
            for c in range(start, min(start + length, cols)):
                entries.append((row, c, _sample_value(rng, vm)))
        entries.sort()

    elif v == "tall_skinny_runs":
        length = int(fp.get("length", 10))
        entries = _place_segments(rows, cols, [{"count": 1, "length": length}], rng, vm)

    else:
        raise ValueError(f"Unknown row_runs variant '{v}'")

    return GeneratedMatrix(rows=rows, cols=cols, entries=entries, params={"variant": v, **fp})


@register("col_runs")
def generate_col_runs(spec: MatrixSpec, rng: random.Random) -> GeneratedMatrix:
    """F5: Column-clustered = transpose of row_runs.

    Generates as row_runs on transposed dimensions, then transposes result.
    """
    from ..postprocess import transpose as do_transpose

    # Build as row-runs on (cols x rows), then transpose
    transposed_spec = MatrixSpec(
        name=spec.name,
        family="row_runs",
        variant=spec.variant,
        family_number=spec.family_number,
        rows=spec.cols,
        cols=spec.rows,
        seed=spec.seed,
        tags=spec.tags,
        family_params=spec.family_params,
        postprocess={},
        value_mode=spec.value_mode,
    )
    matrix = generate_row_runs(transposed_spec, rng)
    return do_transpose(matrix)
