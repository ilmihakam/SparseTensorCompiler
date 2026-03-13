"""F8: Low-rank-ish / few dominant rows/columns / template reuse."""

from __future__ import annotations

import random
from typing import List, Set

from ..registry import register
from ..types import COOEntries, GeneratedMatrix, MatrixSpec


def _sample_value(rng: random.Random, value_mode: str) -> float:
    if value_mode == "ones":
        return 1.0
    return rng.uniform(-1.0, 1.0)


@register("template_reuse")
def generate_template_reuse(spec: MatrixSpec, rng: random.Random) -> GeneratedMatrix:
    fp = spec.family_params
    rows, cols = spec.rows, spec.cols
    vm = spec.value_mode
    v = spec.variant

    if v == "dense_columns":
        # A few columns are dense, rest have 1 nnz
        num_dense = int(fp.get("num_dense", 5))
        dense_density = float(fp.get("dense_density", 0.5))
        sparse_nnz = int(fp.get("sparse_nnz", 1))
        dense_cols = set(rng.sample(range(cols), min(num_dense, cols)))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        # Dense columns
        for c in dense_cols:
            for r in range(rows):
                if rng.random() <= dense_density and (r, c) not in seen:
                    entries.append((r, c, _sample_value(rng, vm)))
                    seen.add((r, c))
        # Sparse columns: each row gets sparse_nnz in non-dense cols
        non_dense = [c for c in range(cols) if c not in dense_cols]
        for r in range(rows):
            nnz = min(sparse_nnz, len(non_dense))
            if nnz > 0:
                chosen = rng.sample(non_dense, nnz)
                for c in chosen:
                    if (r, c) not in seen:
                        entries.append((r, c, _sample_value(rng, vm)))
                        seen.add((r, c))

    elif v == "dense_rows":
        num_dense = int(fp.get("num_dense", 5))
        dense_density = float(fp.get("dense_density", 0.5))
        sparse_nnz = int(fp.get("sparse_nnz", 1))
        dense_rows_set = set(rng.sample(range(rows), min(num_dense, rows)))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for r in dense_rows_set:
            for c in range(cols):
                if rng.random() <= dense_density and (r, c) not in seen:
                    entries.append((r, c, _sample_value(rng, vm)))
                    seen.add((r, c))
        for r in range(rows):
            if r in dense_rows_set:
                continue
            nnz = min(sparse_nnz, cols)
            if nnz > 0:
                chosen = rng.sample(range(cols), nnz)
                for c in chosen:
                    if (r, c) not in seen:
                        entries.append((r, c, _sample_value(rng, vm)))
                        seen.add((r, c))

    elif v == "sparse_uv_overlap":
        # Rank-k structure: k template column sets, each row picks one
        k = int(fp.get("k", 5))
        template_size = int(fp.get("template_size", cols // 10))
        templates: List[List[int]] = []
        for _ in range(k):
            templates.append(sorted(rng.sample(range(cols), min(template_size, cols))))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for r in range(rows):
            t = templates[rng.randrange(k)]
            for c in t:
                if (r, c) not in seen:
                    entries.append((r, c, _sample_value(rng, vm)))
                    seen.add((r, c))

    elif v == "row_templates":
        # Each row reuses one of a small set of column patterns
        num_templates = int(fp.get("num_templates", 4))
        template_nnz = int(fp.get("template_nnz", 20))
        noise_prob = float(fp.get("noise_prob", 0.05))
        templates: List[List[int]] = []
        for _ in range(num_templates):
            templates.append(sorted(rng.sample(range(cols), min(template_nnz, cols))))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for r in range(rows):
            t = templates[rng.randrange(num_templates)]
            for c in t:
                if (r, c) not in seen:
                    entries.append((r, c, _sample_value(rng, vm)))
                    seen.add((r, c))
            # Add noise
            if rng.random() < noise_prob:
                c = rng.randrange(cols)
                if (r, c) not in seen:
                    entries.append((r, c, _sample_value(rng, vm)))
                    seen.add((r, c))

    elif v == "column_templates":
        # Each column reuses one of a small set of row patterns
        num_templates = int(fp.get("num_templates", 4))
        template_nnz = int(fp.get("template_nnz", 20))
        templates: List[List[int]] = []
        for _ in range(num_templates):
            templates.append(sorted(rng.sample(range(rows), min(template_nnz, rows))))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for c in range(cols):
            t = templates[rng.randrange(num_templates)]
            for r in t:
                if (r, c) not in seen:
                    entries.append((r, c, _sample_value(rng, vm)))
                    seen.add((r, c))

    elif v == "disjoint_row_groups":
        # Two groups of rows share disjoint sets of columns
        num_groups = int(fp.get("num_groups", 2))
        nnz_per_row = int(fp.get("nnz_per_row", 10))
        group_width = cols // num_groups
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for r in range(rows):
            group = r % num_groups
            left = group * group_width
            right = min(cols - 1, left + group_width - 1)
            span = right - left + 1
            nnz = min(nnz_per_row, span)
            if nnz > 0:
                chosen = rng.sample(range(left, right + 1), nnz)
                for c in chosen:
                    if (r, c) not in seen:
                        entries.append((r, c, _sample_value(rng, vm)))
                        seen.add((r, c))

    elif v == "column_stripe_plus_noise":
        # Dense column stripe shared by all rows + random noise
        stripe_start = int(fp.get("stripe_start", cols // 4))
        stripe_width = int(fp.get("stripe_width", cols // 10))
        stripe_density = float(fp.get("stripe_density", 0.8))
        noise_nnz = int(fp.get("noise_nnz", 3))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for r in range(rows):
            for c in range(stripe_start, min(stripe_start + stripe_width, cols)):
                if rng.random() <= stripe_density and (r, c) not in seen:
                    entries.append((r, c, _sample_value(rng, vm)))
                    seen.add((r, c))
            for _ in range(noise_nnz):
                c = rng.randrange(cols)
                if (r, c) not in seen:
                    entries.append((r, c, _sample_value(rng, vm)))
                    seen.add((r, c))

    elif v == "row_stripe_plus_noise":
        stripe_start = int(fp.get("stripe_start", rows // 4))
        stripe_width = int(fp.get("stripe_width", rows // 10))
        stripe_density = float(fp.get("stripe_density", 0.8))
        noise_nnz = int(fp.get("noise_nnz", 3))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for c in range(cols):
            for r in range(stripe_start, min(stripe_start + stripe_width, rows)):
                if rng.random() <= stripe_density and (r, c) not in seen:
                    entries.append((r, c, _sample_value(rng, vm)))
                    seen.add((r, c))
            for _ in range(noise_nnz):
                r = rng.randrange(rows)
                if (r, c) not in seen:
                    entries.append((r, c, _sample_value(rng, vm)))
                    seen.add((r, c))

    elif v == "lowrank_plus_sparse":
        # Low-rank component (k dense cols) + sparse random component
        k = int(fp.get("k", 5))
        dense_density = float(fp.get("dense_density", 0.3))
        sparse_nnz_per_row = int(fp.get("sparse_nnz_per_row", 2))
        dense_cols = sorted(rng.sample(range(cols), min(k, cols)))
        entries: COOEntries = []
        seen: Set[tuple] = set()
        for r in range(rows):
            for c in dense_cols:
                if rng.random() <= dense_density and (r, c) not in seen:
                    entries.append((r, c, _sample_value(rng, vm)))
                    seen.add((r, c))
            for _ in range(sparse_nnz_per_row):
                c = rng.randrange(cols)
                if (r, c) not in seen:
                    entries.append((r, c, _sample_value(rng, vm)))
                    seen.add((r, c))

    else:
        raise ValueError(f"Unknown template_reuse variant '{v}'")

    entries.sort()
    return GeneratedMatrix(rows=rows, cols=cols, entries=entries, params={"variant": v, **fp})
