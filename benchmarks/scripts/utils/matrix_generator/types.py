"""Shared data types for the matrix generator framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

COOEntries = List[Tuple[int, int, float]]


@dataclass
class GeneratedMatrix:
    rows: int
    cols: int
    entries: COOEntries
    params: Dict[str, object]


@dataclass
class MatrixSpec:
    name: str
    family: str
    variant: str
    family_number: str
    rows: int
    cols: int
    seed: int
    tags: List[str]
    family_params: Dict[str, object]
    postprocess: Dict[str, object]
    value_mode: str = "ones"
