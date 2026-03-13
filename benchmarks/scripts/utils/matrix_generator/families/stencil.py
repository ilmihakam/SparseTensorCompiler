"""F7: 2D/3D stencil / grid-like patterns."""

from __future__ import annotations

import random
from typing import List, Tuple

from ..registry import register
from ..types import COOEntries, GeneratedMatrix, MatrixSpec


def _sample_value(rng: random.Random, value_mode: str) -> float:
    if value_mode == "ones":
        return 1.0
    return rng.uniform(-1.0, 1.0)


def _grid_node_to_row(coords: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
    """Flatten N-D grid coordinates to a linear index (row-major)."""
    idx = 0
    for i, c in enumerate(coords):
        stride = 1
        for j in range(i + 1, len(shape)):
            stride *= shape[j]
        idx += c * stride
    return idx


def _stencil_matrix_2d(grid_n: int, grid_m: int, offsets: List[Tuple[int, int]],
                       rng: random.Random, value_mode: str,
                       hole_mask: set = None) -> Tuple[int, COOEntries]:
    """Build a 2D stencil matrix. Returns (matrix_size, entries)."""
    n = grid_n * grid_m
    entries: COOEntries = []
    shape = (grid_n, grid_m)
    for i in range(grid_n):
        for j in range(grid_m):
            if hole_mask and (i, j) in hole_mask:
                continue
            row = _grid_node_to_row((i, j), shape)
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_n and 0 <= nj < grid_m:
                    if hole_mask and (ni, nj) in hole_mask:
                        continue
                    col = _grid_node_to_row((ni, nj), shape)
                    entries.append((row, col, _sample_value(rng, value_mode)))
    entries.sort()
    return n, entries


def _stencil_matrix_3d(gx: int, gy: int, gz: int, offsets: List[Tuple[int, int, int]],
                       rng: random.Random, value_mode: str) -> Tuple[int, COOEntries]:
    """Build a 3D stencil matrix."""
    n = gx * gy * gz
    entries: COOEntries = []
    shape = (gx, gy, gz)
    for i in range(gx):
        for j in range(gy):
            for k in range(gz):
                row = _grid_node_to_row((i, j, k), shape)
                for di, dj, dk in offsets:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 0 <= ni < gx and 0 <= nj < gy and 0 <= nk < gz:
                        col = _grid_node_to_row((ni, nj, nk), shape)
                        entries.append((row, col, _sample_value(rng, value_mode)))
    entries.sort()
    return n, entries


# Standard stencil offset sets
_5PT = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
_9PT = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
_7PT_3D = [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
_27PT_3D = [(di, dj, dk) for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1)]


@register("stencil")
def generate_stencil(spec: MatrixSpec, rng: random.Random) -> GeneratedMatrix:
    fp = spec.family_params
    vm = spec.value_mode
    v = spec.variant

    if v in ("2d_5pt", "2d_5pt_small", "2d_5pt_medium", "2d_5pt_large"):
        grid_n = int(fp.get("grid_n", 100))
        grid_m = int(fp.get("grid_m", grid_n))
        n, entries = _stencil_matrix_2d(grid_n, grid_m, _5PT, rng, vm)

    elif v == "2d_9pt":
        grid_n = int(fp.get("grid_n", 100))
        grid_m = int(fp.get("grid_m", grid_n))
        n, entries = _stencil_matrix_2d(grid_n, grid_m, _9PT, rng, vm)

    elif v == "2d_anisotropic":
        grid_n = int(fp.get("grid_n", 100))
        grid_m = int(fp.get("grid_m", grid_n))
        # Stronger coupling in x-direction: add (0, +-2)
        offsets = list(_5PT) + [(0, 2), (0, -2)]
        n, entries = _stencil_matrix_2d(grid_n, grid_m, offsets, rng, vm)

    elif v == "3d_7pt":
        gx = int(fp.get("grid_x", 20))
        gy = int(fp.get("grid_y", gx))
        gz = int(fp.get("grid_z", gx))
        n, entries = _stencil_matrix_3d(gx, gy, gz, _7PT_3D, rng, vm)

    elif v == "3d_27pt":
        gx = int(fp.get("grid_x", 10))
        gy = int(fp.get("grid_y", gx))
        gz = int(fp.get("grid_z", gx))
        n, entries = _stencil_matrix_3d(gx, gy, gz, _27PT_3D, rng, vm)

    elif v == "2d_with_holes":
        grid_n = int(fp.get("grid_n", 100))
        grid_m = int(fp.get("grid_m", grid_n))
        hole_fraction = float(fp.get("hole_fraction", 0.1))
        hole_mask = set()
        for i in range(grid_n):
            for j in range(grid_m):
                if rng.random() < hole_fraction:
                    hole_mask.add((i, j))
        n, entries = _stencil_matrix_2d(grid_n, grid_m, _5PT, rng, vm, hole_mask=hole_mask)

    elif v == "two_subdomains":
        grid_n = int(fp.get("grid_n", 100))
        grid_m = int(fp.get("grid_m", grid_n))
        coupling_width = int(fp.get("coupling_width", 2))
        # Build full 5pt stencil, then remove most cross-boundary connections
        n, entries = _stencil_matrix_2d(grid_n, grid_m, _5PT, rng, vm)
        mid = grid_m // 2
        filtered: COOEntries = []
        for r, c, val in entries:
            # Convert back to 2D coords
            ri, rj = divmod(r, grid_m)
            ci, cj = divmod(c, grid_m)
            # Keep if both in same half, or within coupling_width of boundary
            same_half = (rj < mid) == (cj < mid)
            near_boundary = abs(rj - mid) < coupling_width or abs(cj - mid) < coupling_width
            if same_half or near_boundary:
                filtered.append((r, c, val))
        entries = filtered

    elif v == "locally_refined":
        grid_n = int(fp.get("grid_n", 100))
        grid_m = int(fp.get("grid_m", grid_n))
        # Base 5pt stencil + denser 9pt in a refined region
        n, base_entries = _stencil_matrix_2d(grid_n, grid_m, _5PT, rng, vm)
        refine_start = grid_n // 4
        refine_end = grid_n * 3 // 4
        existing = set((r, c) for r, c, _ in base_entries)
        extra: COOEntries = []
        for i in range(refine_start, refine_end):
            for j in range(refine_start, min(refine_end, grid_m)):
                row = _grid_node_to_row((i, j), (grid_n, grid_m))
                for di, dj in _9PT:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_n and 0 <= nj < grid_m:
                        col = _grid_node_to_row((ni, nj), (grid_n, grid_m))
                        if (row, col) not in existing:
                            extra.append((row, col, _sample_value(rng, vm)))
                            existing.add((row, col))
        entries = base_entries + extra
        entries.sort()

    else:
        raise ValueError(f"Unknown stencil variant '{v}'")

    return GeneratedMatrix(rows=n, cols=n, entries=entries, params={"variant": v, **fp})
