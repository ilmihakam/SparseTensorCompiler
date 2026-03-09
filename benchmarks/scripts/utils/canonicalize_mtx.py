#!/usr/bin/env python3
"""
Canonicalize MatrixMarket coordinate matrices.

Output contract:
- format: coordinate real general
- 1-indexed indices
- sorted coordinates
- duplicate coordinates merged by summation
- symmetric/skew-symmetric expanded
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class CanonicalMatrix:
    rows: int
    cols: int
    nnz: int
    path: Path


def _next_data_line(handle) -> str:
    for line in handle:
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        return stripped
    raise ValueError("Unexpected end of file while reading MatrixMarket")


def _parse_banner(line: str) -> Tuple[str, str, str, str]:
    parts = line.strip().split()
    if len(parts) != 5 or parts[0] != "%%MatrixMarket":
        raise ValueError("Invalid MatrixMarket banner")
    object_type = parts[1].lower()
    storage = parts[2].lower()
    field = parts[3].lower()
    symmetry = parts[4].lower()
    return object_type, storage, field, symmetry


def _parse_entries(
    matrix_path: Path, rows: int, cols: int, nnz_declared: int, field: str, symmetry: str
) -> List[Tuple[int, int, float]]:
    entries: Dict[Tuple[int, int], float] = {}
    with matrix_path.open("r", encoding="utf-8") as handle:
        _ = handle.readline()
        _ = _next_data_line(handle)
        read_entries = 0
        while read_entries < nnz_declared:
            raw = handle.readline()
            if raw == "":
                raise ValueError(f"Unexpected EOF while reading entries in {matrix_path}")
            stripped = raw.strip()
            if not stripped or stripped.startswith("%"):
                continue
            parts = stripped.split()
            if field == "pattern":
                if len(parts) < 2:
                    raise ValueError(f"Invalid pattern entry in {matrix_path}: {stripped}")
                r, c = int(parts[0]), int(parts[1])
                v = 1.0
            elif field == "integer":
                if len(parts) < 3:
                    raise ValueError(f"Invalid integer entry in {matrix_path}: {stripped}")
                r, c = int(parts[0]), int(parts[1])
                v = float(int(parts[2]))
            else:
                if len(parts) < 3:
                    raise ValueError(f"Invalid real entry in {matrix_path}: {stripped}")
                r, c = int(parts[0]), int(parts[1])
                v = float(parts[2])

            r0 = r - 1
            c0 = c - 1
            if r0 < 0 or r0 >= rows or c0 < 0 or c0 >= cols:
                raise ValueError(f"Entry out of bounds in {matrix_path}: {(r, c)}")

            entries[(r0, c0)] = entries.get((r0, c0), 0.0) + v
            if symmetry in ("symmetric", "skew-symmetric") and r0 != c0:
                mirrored_v = v if symmetry == "symmetric" else -v
                entries[(c0, r0)] = entries.get((c0, r0), 0.0) + mirrored_v

            read_entries += 1

    ordered = sorted((r, c, v) for (r, c), v in entries.items())
    return ordered


def canonicalize_matrix(input_path: Path, output_path: Path, force: bool = False) -> CanonicalMatrix:
    if output_path.exists() and not force:
        rows, cols, nnz = read_matrix_header(output_path)
        return CanonicalMatrix(rows=rows, cols=cols, nnz=nnz, path=output_path)

    with input_path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        if not first:
            raise ValueError(f"Empty file: {input_path}")
        object_type, storage, field, symmetry = _parse_banner(first)
        if object_type != "matrix" or storage != "coordinate":
            raise ValueError(f"Only coordinate matrix inputs are supported: {input_path}")
        if field not in ("real", "double", "integer", "pattern"):
            raise ValueError(f"Unsupported MatrixMarket field '{field}' in {input_path}")
        if symmetry not in ("general", "symmetric", "skew-symmetric"):
            raise ValueError(f"Unsupported MatrixMarket symmetry '{symmetry}' in {input_path}")
        dims = _next_data_line(handle).split()
        if len(dims) < 3:
            raise ValueError(f"Invalid size line in {input_path}")
        rows, cols, nnz_declared = int(dims[0]), int(dims[1]), int(dims[2])

    entries = _parse_entries(input_path, rows, cols, nnz_declared, field, symmetry)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("%%MatrixMarket matrix coordinate real general\n")
        handle.write(f"% canonicalized from {input_path.name}\n")
        handle.write(f"{rows} {cols} {len(entries)}\n")
        for r0, c0, value in entries:
            if value == 0.0:
                value = 0.0
            handle.write(f"{r0 + 1} {c0 + 1} {value:.17g}\n")

    return CanonicalMatrix(rows=rows, cols=cols, nnz=len(entries), path=output_path)


def read_matrix_header(matrix_file: Path) -> Tuple[int, int, int]:
    with matrix_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                raise ValueError(f"Invalid matrix header in {matrix_file}")
            return int(parts[0]), int(parts[1]), int(parts[2])
    raise ValueError(f"No matrix header found in {matrix_file}")


def canonicalize_directory(
    input_dir: Path,
    output_dir: Path,
    matrix_stems: Sequence[str] | None = None,
    force: bool = False,
    skip_invalid: bool = True,
) -> List[CanonicalMatrix]:
    if matrix_stems:
        inputs = [input_dir / f"{stem}.mtx" for stem in matrix_stems]
    else:
        inputs = sorted(input_dir.glob("*.mtx"))
    outputs: List[CanonicalMatrix] = []
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Matrix not found: {path}")
        out = output_dir / path.name
        try:
            outputs.append(canonicalize_matrix(path, out, force=force))
        except Exception as exc:  # noqa: BLE001
            if not skip_invalid:
                raise
            print(f"Warning: skipping invalid matrix {path}: {exc}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonicalize MatrixMarket files")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../matrices/suitesparse/raw"),
        help="Input directory containing .mtx files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../matrices/suitesparse/canonical"),
        help="Output directory for canonical .mtx files",
    )
    parser.add_argument(
        "--matrices",
        nargs="*",
        default=None,
        help="Optional list of matrix stems (without .mtx)",
    )
    parser.add_argument("--force", action="store_true", help="Rewrite canonical outputs even if they exist")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = canonicalize_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        matrix_stems=args.matrices,
        force=args.force,
    )
    print(f"Canonicalized {len(outputs)} matrices into {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
