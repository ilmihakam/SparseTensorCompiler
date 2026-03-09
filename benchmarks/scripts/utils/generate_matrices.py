#!/usr/bin/env python3
"""Generate configurable synthetic sparse matrices for unified benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from canonicalize_mtx import canonicalize_directory
from matrix_characteristics import CHARACTERISTIC_NAMES, compute_characteristics_from_file


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = SCRIPT_DIR.parent
DEFAULT_OUT = BENCHMARKS_DIR / "matrices" / "generated"


def _csv_tags(tags: Sequence[str]) -> str:
    return ";".join(str(tag) for tag in tags if str(tag).strip())


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
    prev_rows: Sequence[set[int]],
    similarity_cfg: Dict[str, object],
) -> set[int]:
    row_cols: set[int] = set()
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
    per_row_cols: List[set[int]],
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


def generate_matrix_from_spec(matrix_spec: Dict[str, object]) -> Tuple[int, int, List[Tuple[int, int, float]], Dict[str, object]]:
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
    per_row_cols: List[set[int]] = []

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


def write_matrix_market(path: Path, rows: int, cols: int, entries: Sequence[Tuple[int, int, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("%%MatrixMarket matrix coordinate real general\n")
        handle.write("% generated by generate_matrices.py\n")
        handle.write(f"{rows} {cols} {len(entries)}\n")
        for row, col, value in entries:
            handle.write(f"{row + 1} {col + 1} {value:.17g}\n")


def _default_name(index: int, generator: str, rows: int, cols: int, seed: int) -> str:
    return f"{generator}_{rows}x{cols}_s{seed}_{index:03d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate custom benchmark matrices and manifest")
    parser.add_argument("--spec", type=Path, required=True, help="JSON spec file")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory root")
    parser.add_argument("--manifest", type=Path, default=None, help="Manifest CSV path (default: <out>/manifest.csv)")
    parser.add_argument("--pairs-output", type=Path, default=None, help="Pairs CSV path (default: <out>/pairs.csv)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing raw/canonical matrices")
    return parser.parse_args()


def _read_spec(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("version", 1)) != 1:
        raise ValueError("Only spec version 1 is supported")
    if "matrices" not in payload or not isinstance(payload["matrices"], list):
        raise ValueError("spec must include a 'matrices' list")
    return payload


def _write_manifest(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "source",
        "generator",
        "seed",
        "params_json",
        "tags",
        "raw_path",
        "canonical_path",
        "rows",
        "cols",
        "nnz",
        *CHARACTERISTIC_NAMES,
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_pairs(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["kernel", "matrix_a", "matrix_b"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pairs_from_spec(spec: Dict[str, object]) -> List[Dict[str, object]]:
    all_rows: List[Dict[str, object]] = []
    for pair_set in spec.get("pairs", []):
        kernel = str(pair_set.get("kernel", "")).strip().lower()
        if kernel not in {"spadd", "spelmul", "spgemm"}:
            raise ValueError(f"Unsupported pair kernel '{kernel}' in spec pairs")
        pairs = pair_set.get("pairs", [])
        if not isinstance(pairs, list):
            raise ValueError("pairs entry must contain a list under 'pairs'")
        for entry in pairs:
            matrix_a = str(entry["a"]).strip()
            matrix_b = str(entry["b"]).strip()
            if not matrix_a or not matrix_b:
                raise ValueError("Pair entries require non-empty 'a' and 'b'")
            all_rows.append({"kernel": kernel, "matrix_a": matrix_a, "matrix_b": matrix_b})
    return all_rows


def main() -> int:
    args = parse_args()
    spec = _read_spec(args.spec)

    out_root = args.out
    raw_dir = out_root / "raw"
    canonical_dir = out_root / "canonical"
    manifest_path = args.manifest if args.manifest is not None else out_root / "manifest.csv"
    pairs_path = args.pairs_output if args.pairs_output is not None else out_root / "pairs.csv"

    manifest_rows: List[Dict[str, object]] = []
    generated_names: List[str] = []
    generated_specs: List[Dict[str, object]] = []

    for idx, matrix_spec in enumerate(spec["matrices"]):
        if not isinstance(matrix_spec, dict):
            raise ValueError("Each matrix spec entry must be a JSON object")
        rows, cols, entries, params = generate_matrix_from_spec(matrix_spec)
        generator = str(matrix_spec.get("generator", "structured_random_v1"))
        seed = int(matrix_spec.get("seed", 0))
        name = str(matrix_spec.get("name") or _default_name(idx, generator, rows, cols, seed)).strip()
        if not name:
            raise ValueError("Matrix name resolved to empty string")
        if name in generated_names:
            raise ValueError(f"Duplicate matrix name in spec: {name}")
        generated_names.append(name)

        raw_file = raw_dir / f"{name}.mtx"
        canonical_file = canonical_dir / f"{name}.mtx"
        if (raw_file.exists() or canonical_file.exists()) and not args.force:
            raise FileExistsError(f"Matrix output already exists for {name}; use --force to overwrite")
        write_matrix_market(raw_file, rows, cols, entries)

        tags = matrix_spec.get("tags", [])
        if tags is None:
            tags = []
        if not isinstance(tags, list):
            raise ValueError("tags must be a list of strings")
        generated_specs.append(
            {
                "name": name,
                "generator": generator,
                "seed": seed,
                "params": params,
                "tags": list(tags),
            }
        )

    canonicalize_directory(
        input_dir=raw_dir,
        output_dir=canonical_dir,
        matrix_stems=generated_names,
        force=True,
        skip_invalid=False,
    )

    for generated in generated_specs:
        name = str(generated["name"])
        generator = str(generated["generator"])
        seed = int(generated["seed"])
        params = dict(generated["params"])
        tags = list(generated["tags"])
        canonical_file = canonical_dir / f"{name}.mtx"
        char_map = compute_characteristics_from_file(canonical_file, prefix="char_")

        manifest_row: Dict[str, object] = {
            "name": name,
            "source": "generated",
            "generator": generator,
            "seed": seed,
            "params_json": json.dumps(params, sort_keys=True, separators=(",", ":")),
            "tags": _csv_tags(tags if isinstance(tags, list) else []),
            "raw_path": str((raw_dir / f"{name}.mtx").resolve()),
            "canonical_path": str(canonical_file.resolve()),
            "rows": int(char_map["char_m_rows"]),
            "cols": int(char_map["char_n_cols"]),
            "nnz": int(char_map["char_nnz"]),
        }
        for key in CHARACTERISTIC_NAMES:
            manifest_row[key] = char_map[key]
        manifest_rows.append(manifest_row)

    _write_manifest(manifest_rows, manifest_path)
    pair_rows = _pairs_from_spec(spec)
    if pair_rows:
        _write_pairs(pair_rows, pairs_path)

    print(f"Generated {len(manifest_rows)} matrix/matrices")
    print(f"Raw dir: {raw_dir}")
    print(f"Canonical dir: {canonical_dir}")
    print(f"Manifest: {manifest_path}")
    if pair_rows:
        print(f"Pairs: {pairs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
