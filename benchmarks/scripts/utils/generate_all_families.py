#!/usr/bin/env python3
"""Batch-generate all family matrices (F1-F8, F10) with size scale-downs for
problematic specs.  Outputs to benchmarks/matrices/generated/families/<family>/.

Usage:
    python generate_all_families.py            # generate all families
    python generate_all_families.py f01 f07    # generate only selected families
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent          # benchmarks/scripts/utils
SPECS_DIR = SCRIPT_DIR.parent.parent / "matrices" / "generated" / "specs"
FAMILIES_DIR = SCRIPT_DIR.parent.parent / "matrices" / "generated" / "families"
GENERATOR = SCRIPT_DIR / "generate_matrices.py"

# ── Scale-down patches ────────────────────────────────────────────────
# Keyed by matrix *name* → dict of fields to override.
# Keeps generation fast without editing the canonical spec files.
SCALE_DOWNS: dict[str, dict] = {
    # F1 large / aspect
    "f1_full_random_100k":  {"rows": 10000, "cols": 10000},
    "f1_tall_skinny":       {"rows": 10000},
    "f1_wide_short":        {"cols": 10000},
    # F2 aspect
    "f2_narrow_band_tall":  {"rows": 10000},
    "f2_narrow_band_wide":  {"cols": 10000},
    # F3 coarse
    "f3_coarse_100k":       {"rows": 10000, "cols": 10000},
    # F4 wide_span + tall
    "f4_wide_span":         {"rows": 2000, "cols": 2000},
    "f4_tall_skinny_runs":  {"rows": 10000},
    # F5 wide_span + wide
    "f5_wide_span":         {"rows": 2000, "cols": 2000},
    "f5_tall_skinny_runs":  {"cols": 10000},
    # F6 aspect
    "f6_aspect_skewed":     {"rows": 10000},
    # F7 large stencil grid
    "f7_2d_5pt_1000":       {"rows": 10000, "cols": 10000,
                              "family_params": {"grid_n": 100, "grid_m": 100}},
    # F8 tall
    "f8_tall_dense_cols":   {"rows": 10000},
    # F10 extreme aspect
    "f10_extreme_aspect":   {"rows": 10000},
}


def _apply_patches(spec: dict) -> dict:
    """Return a copy of the spec with large-matrix patches applied."""
    spec = copy.deepcopy(spec)
    for mat in spec["matrices"]:
        patch = SCALE_DOWNS.get(mat["name"])
        if patch is None:
            continue
        for key, val in patch.items():
            if key == "family_params":
                mat.setdefault("family_params", {}).update(val)
            else:
                mat[key] = val
    return spec


def _family_name_from_spec(spec_path: Path) -> str:
    """f01_uniform_random.json → f01_uniform_random"""
    return spec_path.stem


def generate_family(spec_path: Path) -> bool:
    """Generate one family.  Returns True on success."""
    family = _family_name_from_spec(spec_path)
    out_dir = FAMILIES_DIR / family

    with spec_path.open() as f:
        spec = json.load(f)

    patched = _apply_patches(spec)
    n = len(patched["matrices"])

    # Write patched spec to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f"{family}_", delete=False
    ) as tmp:
        json.dump(patched, tmp, indent=2)
        tmp_path = Path(tmp.name)

    print(f"── {family}: {n} matrices → {out_dir}")
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(GENERATOR),
                "--spec", str(tmp_path),
                "--out", str(out_dir),
                "--force",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        # Forward generator stdout (summary lines)
        for line in result.stdout.strip().splitlines():
            print(f"   {line}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"   FAILED ({exc.returncode})")
        if exc.stderr:
            for line in exc.stderr.strip().splitlines():
                print(f"   stderr: {line}")
        return False
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> int:
    # Discover spec files (exclude example.json)
    all_specs = sorted(
        p for p in SPECS_DIR.glob("f*.json")
    )
    if not all_specs:
        print(f"No spec files found in {SPECS_DIR}", file=sys.stderr)
        return 1

    # Optional filter: pass family prefixes on CLI  (e.g. f01 f07)
    filters = [a.lower() for a in sys.argv[1:]]
    if filters:
        all_specs = [
            p for p in all_specs
            if any(p.stem.startswith(f) for f in filters)
        ]
        if not all_specs:
            print(f"No specs matched filters: {filters}", file=sys.stderr)
            return 1

    FAMILIES_DIR.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for spec_path in all_specs:
        if generate_family(spec_path):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} families generated, {fail} failed.")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
