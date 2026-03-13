"""F1: Uniform random baselines — wraps v1 with simplified variant-specific defaults."""

from __future__ import annotations

import random
from typing import Dict

from ..registry import register
from ..types import GeneratedMatrix, MatrixSpec
from ..v1_compat import generate_matrix_from_dict


def _build_v1_dict(spec: MatrixSpec) -> Dict[str, object]:
    """Map F1 family_params + variant into a v1-compatible dict."""
    fp = spec.family_params
    v = spec.variant
    base: Dict[str, object] = {
        "generator": "structured_random_v1",
        "rows": spec.rows,
        "cols": spec.cols,
        "seed": spec.seed,
        "value_mode": spec.value_mode,
    }

    # Defaults per variant
    if v == "full_random":
        base["nnz"] = {
            "mode": "density",
            "density": float(fp.get("density", 0.001)),
            "row_distribution": "poisson",
        }
        base["support"] = {"mode": "global"}
    elif v == "banded_random":
        base["nnz"] = {
            "mode": "density",
            "density": float(fp.get("density", 0.01)),
            "row_distribution": "poisson",
        }
        base["support"] = {
            "mode": "banded",
            "bandwidth": int(fp.get("bandwidth", max(1, spec.cols // 10))),
        }
    elif v in ("tall_skinny", "wide_short"):
        base["nnz"] = {
            "mode": "avg_nnz_row",
            "avg_nnz_row": float(fp.get("avg_nnz_row", 10)),
            "row_distribution": "poisson",
        }
        base["support"] = {"mode": "global"}
    elif v == "dense_stress":
        base["nnz"] = {
            "mode": "density",
            "density": float(fp.get("density", 0.05)),
            "row_distribution": "poisson",
        }
        base["support"] = {"mode": "global"}
    else:
        # Fallback: use family_params as-is with sensible defaults
        base["nnz"] = dict(fp.get("nnz", {"mode": "density", "density": float(fp.get("density", 0.001)), "row_distribution": "poisson"}))
        base["support"] = dict(fp.get("support", {"mode": "global"}))

    # Common defaults
    base.setdefault("clustering", dict(fp.get("clustering", {"mode": "runs", "avg_run_length": 1.0, "avg_gap": 1.0})))
    base.setdefault("columns", dict(fp.get("columns", {"mode": "uniform"})))
    base.setdefault("inter_row_similarity", dict(fp.get("inter_row_similarity", {"mode": "none"})))
    base.setdefault("block_structure", dict(fp.get("block_structure", {"enabled": False})))

    return base


@register("uniform_random")
def generate_uniform_random(spec: MatrixSpec, rng: random.Random) -> GeneratedMatrix:
    v1_dict = _build_v1_dict(spec)
    rows, cols, entries, params = generate_matrix_from_dict(v1_dict)
    params["variant"] = spec.variant
    return GeneratedMatrix(rows=rows, cols=cols, entries=entries, params=params)
