#!/usr/bin/env python3
"""
Shared utilities for unified benchmark runners.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import signal
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Sequence, Tuple

_UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from matrix_characteristics import CHARACTERISTIC_BASE_NAMES, CHARACTERISTIC_NAMES, zero_characteristics


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
COMPILER = REPO_ROOT / "build" / "sparse_compiler"
GCC_FLAGS = ["-O2", "-march=native", "-std=c11"]

DEFAULT_SWEEP_BLOCK_SIZES = [16, 32, 64]
DEFAULT_SWEEP_BLOCK2D_SIZES = [(16, 16), (32, 32), (64, 64)]
DEFAULT_SWEEP_ORDERS = ["I_THEN_B", "B_THEN_I", "I_B_I"]
DEFAULT_HWC_MODE = "off"

CONFIGS = {
    "baseline": [],
    "interchange_only": ["--opt-interchange"],
    "block_only": ["--opt-block=32"],
    "i_then_b": ["--opt-all=32", "--opt-order=I_THEN_B"],
    "b_then_i": ["--opt-all=32", "--opt-order=B_THEN_I"],
    "i_b_i": ["--opt-all=32", "--opt-order=I_B_I"],
}

DEFAULT_CONFIG_ORDER = [
    "baseline",
    "interchange_only",
    "block_only",
    "i_then_b",
    "b_then_i",
    "i_b_i",
]

POSBLOCK_RE = re.compile(r"^posblock_p(?P<p>\d+)$")
BLOCKPOS_RE = re.compile(r"^blockpos_b(?P<b>\d+)_p(?P<p>\d+)$")
ALLPOS_RE = re.compile(r"^allpos_(?P<order>[A-Z_]+)_b(?P<b>\d+)_p(?P<p>\d+)$")
BLOCK2D_TARGET_RE = re.compile(
    r"^block2d_(?P<t1>[A-Za-z0-9_]+)_(?P<t2>[A-Za-z0-9_]+)_b(?P<b1>\d+)x(?P<b2>\d+)$"
)
ALL2D_TARGET_RE = re.compile(
    r"^all2d_(?P<t1>[A-Za-z0-9_]+)_(?P<t2>[A-Za-z0-9_]+)_(?P<order>[A-Z_]+)_b(?P<b1>\d+)x(?P<b2>\d+)$"
)
FULL_TARGET_RE = re.compile(
    r"^full_(?P<t1>[A-Za-z0-9_]+)_(?P<t2>[A-Za-z0-9_]+)_p(?P<p>\d+)_b(?P<b1>\d+)x(?P<b2>\d+)$"
)


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    flags: List[str]
    kind: Literal["static", "sweep_1d", "sweep_2d"] = "static"


@dataclass(frozen=True)
class MatrixMeta:
    name: str
    path: Path
    rows: int
    cols: int
    nnz: int


@dataclass(frozen=True)
class MatrixManifestEntry:
    name: str
    source: str
    generator: str
    seed: int
    params_json: str
    tags: str
    characteristics: Dict[str, float]


MATRIX_META_FIELDS = [
    "matrix_source",
    "matrix_generator",
    "matrix_seed",
    "matrix_params_json",
    "matrix_tags",
]
PAIR_A_META_FIELDS = [
    "matrix_a_source",
    "matrix_a_generator",
    "matrix_a_seed",
    "matrix_a_params_json",
    "matrix_a_tags",
]
PAIR_B_META_FIELDS = [
    "matrix_b_source",
    "matrix_b_generator",
    "matrix_b_seed",
    "matrix_b_params_json",
    "matrix_b_tags",
]
PAIR_A_CHAR_FIELDS = [f"char_a_{name}" for name in CHARACTERISTIC_BASE_NAMES]
PAIR_B_CHAR_FIELDS = [f"char_b_{name}" for name in CHARACTERISTIC_BASE_NAMES]


FEATURE_NAMES = [
    "feat_m_rows", "feat_n_cols", "feat_aspect_ratio", "feat_nnz", "feat_density",
    "feat_avg_nnz_row", "feat_avg_nnz_col",
    "feat_row_variance", "feat_row_cv", "feat_max_row_length", "feat_min_row_length",
    "feat_col_variance", "feat_col_cv",
    "feat_bandwidth", "feat_avg_row_span",
    "feat_avg_runs_per_row", "feat_avg_run_length", "feat_avg_gap", "feat_frac_adjacent",
    "feat_normalized_entropy",
    "feat_avg_entry_matches",
]

# Map from FEATURE: output key to dataclass field name
_FEATURE_KEY_TO_FIELD = {name.replace("feat_", ""): name for name in FEATURE_NAMES}
_NORMALIZED_HWC_FIELDS = {
    "cycles": "hwc_cycles",
    "instructions": "hwc_instructions",
    "cache-references": "hwc_cache_references",
    "cache-misses": "hwc_cache_misses",
    "branches": "hwc_branches",
    "branch-misses": "hwc_branch_misses",
    # lauka counter names (Apple Silicon):
    "core_active_cycle": "hwc_cycles",
    "fixed_cycles": "hwc_cycles",
    "inst_all": "hwc_instructions",
    "fixed_instructions": "hwc_instructions",
    "l1d_cache_miss_ld_nonspec": "hwc_cache_misses",
    "branch_mispred_nonspec": "hwc_branch_misses",
}


@dataclass(frozen=True)
class HardwareCounterConfig:
    mode: Literal["off", "perf", "lauka"] = "off"
    events: Tuple[str, ...] = ()
    strict: bool = False
    lauka_bin: str = "lauka"
    lauka_runs: int = 5
    lauka_warmup: int = 1


@dataclass
class SpMVBenchmarkResult:
    matrix_name: str
    format: str
    kernel: str
    impl: str
    config: str
    rows: int
    cols: int
    nnz: int
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    stddev_ms: float
    variance_pct: float
    max_error: float
    hwc_status: str = "off"
    hwc_tool: str = ""
    hwc_events_requested: str = ""
    hwc_events_recorded: str = ""
    hwc_event_values_json: str = ""
    hwc_cycles: float = 0.0
    hwc_instructions: float = 0.0
    hwc_cache_references: float = 0.0
    hwc_cache_misses: float = 0.0
    hwc_branches: float = 0.0
    hwc_branch_misses: float = 0.0
    matrix_source: str = ""
    matrix_generator: str = ""
    matrix_seed: int = 0
    matrix_params_json: str = ""
    matrix_tags: str = ""
    char_m_rows: float = 0.0
    char_n_cols: float = 0.0
    char_aspect_ratio: float = 0.0
    char_nnz: float = 0.0
    char_density: float = 0.0
    char_avg_nnz_row: float = 0.0
    char_avg_nnz_col: float = 0.0
    char_row_variance: float = 0.0
    char_row_cv: float = 0.0
    char_max_row_length: float = 0.0
    char_min_row_length: float = 0.0
    char_nnz_row_min: float = 0.0
    char_nnz_row_max: float = 0.0
    char_col_variance: float = 0.0
    char_col_cv: float = 0.0
    char_nnz_col_min: float = 0.0
    char_nnz_col_max: float = 0.0
    char_bandwidth: float = 0.0
    char_avg_row_span: float = 0.0
    char_avg_col_span: float = 0.0
    char_avg_runs_per_row: float = 0.0
    char_avg_run_length: float = 0.0
    char_avg_gap: float = 0.0
    char_frac_adjacent: float = 0.0
    char_normalized_entropy: float = 0.0
    char_avg_entry_matches: float = 0.0
    char_jaccard_adjacent_rows_avg: float = 0.0
    feat_m_rows: float = 0.0
    feat_n_cols: float = 0.0
    feat_aspect_ratio: float = 0.0
    feat_nnz: float = 0.0
    feat_density: float = 0.0
    feat_avg_nnz_row: float = 0.0
    feat_avg_nnz_col: float = 0.0
    feat_row_variance: float = 0.0
    feat_row_cv: float = 0.0
    feat_max_row_length: float = 0.0
    feat_min_row_length: float = 0.0
    feat_col_variance: float = 0.0
    feat_col_cv: float = 0.0
    feat_bandwidth: float = 0.0
    feat_avg_row_span: float = 0.0
    feat_avg_runs_per_row: float = 0.0
    feat_avg_run_length: float = 0.0
    feat_avg_gap: float = 0.0
    feat_frac_adjacent: float = 0.0
    feat_normalized_entropy: float = 0.0
    feat_avg_entry_matches: float = 0.0
    config_flags: str = ""
    trials: int = 1
    trial_selected: int = 0
    trial_avg_time_ms_median: float = 0.0
    trial_avg_time_ms_min: float = 0.0
    trial_avg_time_ms_max: float = 0.0
    trial_avg_time_ms_stddev: float = 0.0
    trial_avg_time_ms_variance_pct: float = 0.0
    trial_max_error_max: float = 0.0


@dataclass
class SpMMBenchmarkResult:
    matrix_name: str
    format: str
    kernel: str
    impl: str
    config: str
    rows: int
    cols: int
    nnz: int
    N: int
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    stddev_ms: float
    variance_pct: float
    max_error: float
    hwc_status: str = "off"
    hwc_tool: str = ""
    hwc_events_requested: str = ""
    hwc_events_recorded: str = ""
    hwc_event_values_json: str = ""
    hwc_cycles: float = 0.0
    hwc_instructions: float = 0.0
    hwc_cache_references: float = 0.0
    hwc_cache_misses: float = 0.0
    hwc_branches: float = 0.0
    hwc_branch_misses: float = 0.0
    matrix_source: str = ""
    matrix_generator: str = ""
    matrix_seed: int = 0
    matrix_params_json: str = ""
    matrix_tags: str = ""
    char_m_rows: float = 0.0
    char_n_cols: float = 0.0
    char_aspect_ratio: float = 0.0
    char_nnz: float = 0.0
    char_density: float = 0.0
    char_avg_nnz_row: float = 0.0
    char_avg_nnz_col: float = 0.0
    char_row_variance: float = 0.0
    char_row_cv: float = 0.0
    char_max_row_length: float = 0.0
    char_min_row_length: float = 0.0
    char_nnz_row_min: float = 0.0
    char_nnz_row_max: float = 0.0
    char_col_variance: float = 0.0
    char_col_cv: float = 0.0
    char_nnz_col_min: float = 0.0
    char_nnz_col_max: float = 0.0
    char_bandwidth: float = 0.0
    char_avg_row_span: float = 0.0
    char_avg_col_span: float = 0.0
    char_avg_runs_per_row: float = 0.0
    char_avg_run_length: float = 0.0
    char_avg_gap: float = 0.0
    char_frac_adjacent: float = 0.0
    char_normalized_entropy: float = 0.0
    char_avg_entry_matches: float = 0.0
    char_jaccard_adjacent_rows_avg: float = 0.0
    feat_m_rows: float = 0.0
    feat_n_cols: float = 0.0
    feat_aspect_ratio: float = 0.0
    feat_nnz: float = 0.0
    feat_density: float = 0.0
    feat_avg_nnz_row: float = 0.0
    feat_avg_nnz_col: float = 0.0
    feat_row_variance: float = 0.0
    feat_row_cv: float = 0.0
    feat_max_row_length: float = 0.0
    feat_min_row_length: float = 0.0
    feat_col_variance: float = 0.0
    feat_col_cv: float = 0.0
    feat_bandwidth: float = 0.0
    feat_avg_row_span: float = 0.0
    feat_avg_runs_per_row: float = 0.0
    feat_avg_run_length: float = 0.0
    feat_avg_gap: float = 0.0
    feat_frac_adjacent: float = 0.0
    feat_normalized_entropy: float = 0.0
    feat_avg_entry_matches: float = 0.0
    config_flags: str = ""
    trials: int = 1
    trial_selected: int = 0
    trial_avg_time_ms_median: float = 0.0
    trial_avg_time_ms_min: float = 0.0
    trial_avg_time_ms_max: float = 0.0
    trial_avg_time_ms_stddev: float = 0.0
    trial_avg_time_ms_variance_pct: float = 0.0
    trial_max_error_max: float = 0.0


@dataclass
class TwoSparseBenchmarkResult:
    matrix_a: str
    matrix_b: str
    format: str
    kernel: str
    impl: str
    config: str
    rows_a: int
    cols_a: int
    nnz_a: int
    rows_b: int
    cols_b: int
    nnz_b: int
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    stddev_ms: float
    variance_pct: float
    max_error: float
    hwc_status: str = "off"
    hwc_tool: str = ""
    hwc_events_requested: str = ""
    hwc_events_recorded: str = ""
    hwc_event_values_json: str = ""
    hwc_cycles: float = 0.0
    hwc_instructions: float = 0.0
    hwc_cache_references: float = 0.0
    hwc_cache_misses: float = 0.0
    hwc_branches: float = 0.0
    hwc_branch_misses: float = 0.0
    matrix_a_source: str = ""
    matrix_a_generator: str = ""
    matrix_a_seed: int = 0
    matrix_a_params_json: str = ""
    matrix_a_tags: str = ""
    matrix_b_source: str = ""
    matrix_b_generator: str = ""
    matrix_b_seed: int = 0
    matrix_b_params_json: str = ""
    matrix_b_tags: str = ""
    char_a_m_rows: float = 0.0
    char_a_n_cols: float = 0.0
    char_a_aspect_ratio: float = 0.0
    char_a_nnz: float = 0.0
    char_a_density: float = 0.0
    char_a_avg_nnz_row: float = 0.0
    char_a_avg_nnz_col: float = 0.0
    char_a_row_variance: float = 0.0
    char_a_row_cv: float = 0.0
    char_a_max_row_length: float = 0.0
    char_a_min_row_length: float = 0.0
    char_a_nnz_row_min: float = 0.0
    char_a_nnz_row_max: float = 0.0
    char_a_col_variance: float = 0.0
    char_a_col_cv: float = 0.0
    char_a_nnz_col_min: float = 0.0
    char_a_nnz_col_max: float = 0.0
    char_a_bandwidth: float = 0.0
    char_a_avg_row_span: float = 0.0
    char_a_avg_col_span: float = 0.0
    char_a_avg_runs_per_row: float = 0.0
    char_a_avg_run_length: float = 0.0
    char_a_avg_gap: float = 0.0
    char_a_frac_adjacent: float = 0.0
    char_a_normalized_entropy: float = 0.0
    char_a_avg_entry_matches: float = 0.0
    char_a_jaccard_adjacent_rows_avg: float = 0.0
    char_b_m_rows: float = 0.0
    char_b_n_cols: float = 0.0
    char_b_aspect_ratio: float = 0.0
    char_b_nnz: float = 0.0
    char_b_density: float = 0.0
    char_b_avg_nnz_row: float = 0.0
    char_b_avg_nnz_col: float = 0.0
    char_b_row_variance: float = 0.0
    char_b_row_cv: float = 0.0
    char_b_max_row_length: float = 0.0
    char_b_min_row_length: float = 0.0
    char_b_nnz_row_min: float = 0.0
    char_b_nnz_row_max: float = 0.0
    char_b_col_variance: float = 0.0
    char_b_col_cv: float = 0.0
    char_b_nnz_col_min: float = 0.0
    char_b_nnz_col_max: float = 0.0
    char_b_bandwidth: float = 0.0
    char_b_avg_row_span: float = 0.0
    char_b_avg_col_span: float = 0.0
    char_b_avg_runs_per_row: float = 0.0
    char_b_avg_run_length: float = 0.0
    char_b_avg_gap: float = 0.0
    char_b_frac_adjacent: float = 0.0
    char_b_normalized_entropy: float = 0.0
    char_b_avg_entry_matches: float = 0.0
    char_b_jaccard_adjacent_rows_avg: float = 0.0
    feat_m_rows: float = 0.0
    feat_n_cols: float = 0.0
    feat_aspect_ratio: float = 0.0
    feat_nnz: float = 0.0
    feat_density: float = 0.0
    feat_avg_nnz_row: float = 0.0
    feat_avg_nnz_col: float = 0.0
    feat_row_variance: float = 0.0
    feat_row_cv: float = 0.0
    feat_max_row_length: float = 0.0
    feat_min_row_length: float = 0.0
    feat_col_variance: float = 0.0
    feat_col_cv: float = 0.0
    feat_bandwidth: float = 0.0
    feat_avg_row_span: float = 0.0
    feat_avg_runs_per_row: float = 0.0
    feat_avg_run_length: float = 0.0
    feat_avg_gap: float = 0.0
    feat_frac_adjacent: float = 0.0
    feat_normalized_entropy: float = 0.0
    feat_avg_entry_matches: float = 0.0
    config_flags: str = ""
    trials: int = 1
    trial_selected: int = 0
    trial_avg_time_ms_median: float = 0.0
    trial_avg_time_ms_min: float = 0.0
    trial_avg_time_ms_max: float = 0.0
    trial_avg_time_ms_stddev: float = 0.0
    trial_avg_time_ms_variance_pct: float = 0.0
    trial_max_error_max: float = 0.0


@dataclass
class SDDMMBenchmarkResult:
    matrix_name: str
    format: str
    kernel: str
    impl: str
    config: str
    rows: int
    cols: int
    nnz: int
    K: int
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    stddev_ms: float
    variance_pct: float
    max_error: float
    hwc_status: str = "off"
    hwc_tool: str = ""
    hwc_events_requested: str = ""
    hwc_events_recorded: str = ""
    hwc_event_values_json: str = ""
    hwc_cycles: float = 0.0
    hwc_instructions: float = 0.0
    hwc_cache_references: float = 0.0
    hwc_cache_misses: float = 0.0
    hwc_branches: float = 0.0
    hwc_branch_misses: float = 0.0
    matrix_source: str = ""
    matrix_generator: str = ""
    matrix_seed: int = 0
    matrix_params_json: str = ""
    matrix_tags: str = ""
    char_m_rows: float = 0.0
    char_n_cols: float = 0.0
    char_aspect_ratio: float = 0.0
    char_nnz: float = 0.0
    char_density: float = 0.0
    char_avg_nnz_row: float = 0.0
    char_avg_nnz_col: float = 0.0
    char_row_variance: float = 0.0
    char_row_cv: float = 0.0
    char_max_row_length: float = 0.0
    char_min_row_length: float = 0.0
    char_nnz_row_min: float = 0.0
    char_nnz_row_max: float = 0.0
    char_col_variance: float = 0.0
    char_col_cv: float = 0.0
    char_nnz_col_min: float = 0.0
    char_nnz_col_max: float = 0.0
    char_bandwidth: float = 0.0
    char_avg_row_span: float = 0.0
    char_avg_col_span: float = 0.0
    char_avg_runs_per_row: float = 0.0
    char_avg_run_length: float = 0.0
    char_avg_gap: float = 0.0
    char_frac_adjacent: float = 0.0
    char_normalized_entropy: float = 0.0
    char_avg_entry_matches: float = 0.0
    char_jaccard_adjacent_rows_avg: float = 0.0
    feat_m_rows: float = 0.0
    feat_n_cols: float = 0.0
    feat_aspect_ratio: float = 0.0
    feat_nnz: float = 0.0
    feat_density: float = 0.0
    feat_avg_nnz_row: float = 0.0
    feat_avg_nnz_col: float = 0.0
    feat_row_variance: float = 0.0
    feat_row_cv: float = 0.0
    feat_max_row_length: float = 0.0
    feat_min_row_length: float = 0.0
    feat_col_variance: float = 0.0
    feat_col_cv: float = 0.0
    feat_bandwidth: float = 0.0
    feat_avg_row_span: float = 0.0
    feat_avg_runs_per_row: float = 0.0
    feat_avg_run_length: float = 0.0
    feat_avg_gap: float = 0.0
    feat_frac_adjacent: float = 0.0
    feat_normalized_entropy: float = 0.0
    feat_avg_entry_matches: float = 0.0
    config_flags: str = ""
    trials: int = 1
    trial_selected: int = 0
    trial_avg_time_ms_median: float = 0.0
    trial_avg_time_ms_min: float = 0.0
    trial_avg_time_ms_max: float = 0.0
    trial_avg_time_ms_stddev: float = 0.0
    trial_avg_time_ms_variance_pct: float = 0.0
    trial_max_error_max: float = 0.0


@dataclass
class FailedRun:
    kernel: str
    impl: str
    config: str
    format: str
    item: str
    reason: str
    hwc_status: str = "off"
    hwc_tool: str = ""
    hwc_events_requested: str = ""
    hwc_events_recorded: str = ""
    hwc_event_values_json: str = ""
    hwc_cycles: float = 0.0
    hwc_instructions: float = 0.0
    hwc_cache_references: float = 0.0
    hwc_cache_misses: float = 0.0
    hwc_branches: float = 0.0
    hwc_branch_misses: float = 0.0
    config_flags: str = ""
    trials: int = 1
    trial_selected: int = 0
    trial_avg_time_ms_median: float = 0.0
    trial_avg_time_ms_min: float = 0.0
    trial_avg_time_ms_max: float = 0.0
    trial_avg_time_ms_stddev: float = 0.0
    trial_avg_time_ms_variance_pct: float = 0.0
    trial_max_error_max: float = 0.0


@dataclass(frozen=True)
class TrialSummary:
    trials: int
    trial_selected: int
    trial_avg_time_ms_median: float
    trial_avg_time_ms_min: float
    trial_avg_time_ms_max: float
    trial_avg_time_ms_stddev: float
    trial_avg_time_ms_variance_pct: float
    trial_max_error_max: float


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str


def ensure_compiler_exists() -> None:
    if not COMPILER.exists():
        raise FileNotFoundError(f"Compiler not found: {COMPILER}")


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_hwc_events(raw: str) -> List[str]:
    events: List[str] = []
    seen = set()
    for event in parse_csv_list(raw):
        if event in seen:
            continue
        seen.add(event)
        events.append(event)
    return events


def add_hwc_args(parser) -> None:
    parser.add_argument(
        "--hwc-mode",
        choices=["off", "perf", "lauka"],
        default=DEFAULT_HWC_MODE,
        help="Hardware counter collection mode",
    )
    parser.add_argument(
        "--hwc-events",
        default="",
        help="Comma-separated events to collect when --hwc-mode=perf or lauka",
    )
    parser.add_argument(
        "--hwc-strict",
        action="store_true",
        help="Fail a benchmark row if requested events are unavailable",
    )
    parser.add_argument(
        "--hwc-lauka-bin",
        default="lauka",
        help="Path to lauka binary (default: 'lauka')",
    )
    parser.add_argument(
        "--hwc-lauka-runs",
        type=int,
        default=5,
        help="Number of lauka measurement runs (min 3, default: 5)",
    )
    parser.add_argument(
        "--hwc-lauka-warmup",
        type=int,
        default=1,
        help="Number of lauka warmup runs (default: 1)",
    )


def build_hwc_config(args) -> HardwareCounterConfig:
    mode = getattr(args, "hwc_mode", DEFAULT_HWC_MODE)
    events = tuple(parse_hwc_events(getattr(args, "hwc_events", "")))
    strict = bool(getattr(args, "hwc_strict", False))
    if mode == "perf" and not events:
        raise ValueError("--hwc-events must be non-empty when --hwc-mode=perf")
    if mode == "lauka" and not events:
        raise ValueError("--hwc-events must be non-empty when --hwc-mode=lauka")
    lauka_bin = getattr(args, "hwc_lauka_bin", "lauka")
    lauka_runs = int(getattr(args, "hwc_lauka_runs", 5))
    lauka_warmup = int(getattr(args, "hwc_lauka_warmup", 1))
    if mode == "lauka" and lauka_runs < 3:
        raise ValueError("--hwc-lauka-runs must be >= 3")
    return HardwareCounterConfig(
        mode=mode, events=events, strict=strict,
        lauka_bin=lauka_bin, lauka_runs=lauka_runs, lauka_warmup=lauka_warmup,
    )


def default_hwc_fields(
    *,
    status: str = "off",
    tool: str = "",
    events_requested: Sequence[str] = (),
    events_recorded: Sequence[str] = (),
    event_values: Dict[str, float] | None = None,
) -> Dict[str, object]:
    values: Dict[str, object] = {
        "hwc_status": status,
        "hwc_tool": tool,
        "hwc_events_requested": ",".join(events_requested),
        "hwc_events_recorded": ",".join(events_recorded),
        "hwc_event_values_json": json.dumps(event_values or {}, sort_keys=True),
        "hwc_cycles": 0.0,
        "hwc_instructions": 0.0,
        "hwc_cache_references": 0.0,
        "hwc_cache_misses": 0.0,
        "hwc_branches": 0.0,
        "hwc_branch_misses": 0.0,
    }
    for event_name, raw_value in (event_values or {}).items():
        field_name = _NORMALIZED_HWC_FIELDS.get(normalize_event_name(event_name))
        if field_name:
            values[field_name] = float(raw_value)
    return values


def validate_formats(formats: Sequence[str]) -> List[str]:
    normalized = [item.lower() for item in formats]
    for fmt in normalized:
        if fmt not in ("csr", "csc"):
            raise ValueError(f"Unsupported format '{fmt}'")
    return normalized


def validate_config_names(config_names: Sequence[str]) -> List[str]:
    names = list(config_names)
    for config_name in names:
        if config_name not in CONFIGS:
            raise ValueError(f"Unknown config '{config_name}'")
    return names


def parse_int_csv(raw: str) -> List[int]:
    values: List[int] = []
    for token in parse_csv_list(raw):
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid integer '{token}'") from exc
        if value <= 0:
            raise ValueError(f"Expected positive integer, got {value}")
        values.append(value)
    if not values:
        raise ValueError("Empty integer list")
    return values


def parse_orders_csv(raw: str) -> List[str]:
    orders = parse_csv_list(raw)
    allowed = set(DEFAULT_SWEEP_ORDERS)
    for order in orders:
        if order not in allowed:
            raise ValueError(f"Unsupported opt order '{order}'. Allowed: {', '.join(DEFAULT_SWEEP_ORDERS)}")
    if not orders:
        raise ValueError("Empty orders list")
    return orders


def parse_block2d_csv(raw: str) -> List[Tuple[int, int]]:
    sizes: List[Tuple[int, int]] = []
    for token in parse_csv_list(raw):
        if "x" in token:
            left, right = token.split("x", 1)
            bx = int(left)
            by = int(right)
        else:
            bx = int(token)
            by = bx
        if bx <= 0 or by <= 0:
            raise ValueError(f"Expected positive 2D block size, got {token}")
        sizes.append((bx, by))
    if not sizes:
        raise ValueError("Empty 2D block size list")
    return sizes


def parse_dynamic_config_spec(name: str) -> ConfigSpec:
    if match := POSBLOCK_RE.match(name):
        p = match.group("p")
        return ConfigSpec(name=name, flags=[f"--opt-block-pos={p}"], kind="static")

    if match := BLOCKPOS_RE.match(name):
        b = match.group("b")
        p = match.group("p")
        return ConfigSpec(name=name, flags=[f"--opt-block={b}", f"--opt-block-pos={p}"], kind="static")

    if match := ALLPOS_RE.match(name):
        order = match.group("order")
        b = match.group("b")
        p = match.group("p")
        return ConfigSpec(
            name=name,
            flags=[f"--opt-all={b}", f"--opt-order={order}", f"--opt-block-pos={p}"],
            kind="static",
        )

    if match := BLOCK2D_TARGET_RE.match(name):
        t1 = match.group("t1")
        t2 = match.group("t2")
        b1 = match.group("b1")
        b2 = match.group("b2")
        return ConfigSpec(
            name=name,
            flags=[f"--opt-block-2d={b1}x{b2}", f"--opt-block-2d-targets={t1},{t2}"],
            kind="static",
        )

    if match := ALL2D_TARGET_RE.match(name):
        t1 = match.group("t1")
        t2 = match.group("t2")
        order = match.group("order")
        b1 = match.group("b1")
        b2 = match.group("b2")
        return ConfigSpec(
            name=name,
            flags=[
                "--opt-interchange",
                f"--opt-block-2d={b1}x{b2}",
                f"--opt-block-2d-targets={t1},{t2}",
                f"--opt-order={order}",
            ],
            kind="static",
        )

    if match := FULL_TARGET_RE.match(name):
        t1 = match.group("t1")
        t2 = match.group("t2")
        p = match.group("p")
        b1 = match.group("b1")
        b2 = match.group("b2")
        return ConfigSpec(
            name=name,
            flags=[
                f"--opt-block-2d={b1}x{b2}",
                f"--opt-block-2d-targets={t1},{t2}",
                f"--opt-block-pos={p}",
            ],
            kind="static",
        )

    raise ValueError(f"Unknown benchmark config '{name}'")


def build_config_specs(
    kernel: str,
    static_names: Sequence[str],
    *,
    sweep_block_sizes: Sequence[int] | None = None,
    sweep_orders: Sequence[str] | None = None,
    sweep_block2d_sizes: Sequence[Tuple[int, int]] | None = None,
) -> List[ConfigSpec]:
    specs: List[ConfigSpec] = []
    for name in static_names:
        if name in CONFIGS:
            specs.append(ConfigSpec(name=name, flags=list(CONFIGS[name]), kind="static"))
        else:
            specs.append(parse_dynamic_config_spec(name))

    orders = list(sweep_orders) if sweep_orders else list(DEFAULT_SWEEP_ORDERS)

    if sweep_block_sizes:
        for b in sweep_block_sizes:
            specs.append(ConfigSpec(name=f"block_b{b}", flags=[f"--opt-block={b}"], kind="sweep_1d"))
            for order in orders:
                specs.append(
                    ConfigSpec(
                        name=f"all_{order}_b{b}",
                        flags=[f"--opt-all={b}", f"--opt-order={order}"],
                        kind="sweep_1d",
                    )
                )

    if sweep_block2d_sizes:
        if kernel not in ("spmm", "sddmm"):
            raise ValueError(f"2D blocking sweeps only supported for spmm/sddmm (got kernel={kernel})")
        for bx, by in sweep_block2d_sizes:
            bxby = f"{bx}x{by}"
            specs.append(ConfigSpec(name=f"block2d_b{bxby}", flags=[f"--opt-block-2d={bxby}"], kind="sweep_2d"))
            for order in orders:
                specs.append(
                    ConfigSpec(
                        name=f"all2d_{order}_b{bxby}",
                        flags=["--opt-interchange", f"--opt-block-2d={bxby}", f"--opt-order={order}"],
                        kind="sweep_2d",
                    )
                )

    return specs


def normalize_event_name(event_name: str) -> str:
    base = event_name.strip()
    if ":" in base:
        base = base.split(":", 1)[0]
    return base


def build_perf_command(command: Sequence[str], events: Sequence[str]) -> List[str]:
    return [
        "perf",
        "stat",
        "--no-big-num",
        "-x,",
        "-e",
        ",".join(events),
        "--",
        *command,
    ]


def parse_perf_stat_output(stderr: str, requested_events: Sequence[str]) -> Dict[str, object]:
    requested = [event.strip() for event in requested_events if event.strip()]
    recorded: List[str] = []
    values: Dict[str, float] = {}

    for row in csv.reader(io.StringIO(stderr)):
        if len(row) < 3:
            continue
        raw_value = row[0].strip()
        raw_event = row[2].strip()
        if not raw_event:
            continue
        if raw_value in ("", "<not counted>", "<not supported>"):
            continue
        try:
            value = float(raw_value.replace(",", ""))
        except ValueError:
            continue
        recorded.append(raw_event)
        values[raw_event] = value

    requested_set = set(requested)
    recorded_set = set(recorded)
    if not recorded:
        status = "unavailable"
    elif requested_set.issubset(recorded_set):
        status = "ok"
    else:
        status = "partial"

    return default_hwc_fields(
        status=status,
        tool="perf",
        events_requested=requested,
        events_recorded=recorded,
        event_values=values,
    )


_SI_MULTIPLIERS = {"K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}
_UNIT_SUFFIXES = ("ms", "MB", "cy", "s", "B")


def parse_si_value(text: str) -> float:
    """Parse lauka SI-suffixed numbers: '2.51G' → 2.51e9, '591' → 591.0."""
    s = text.strip()
    if not s:
        raise ValueError("empty value")
    # Strip known unit suffixes first (e.g. 'ms', 'MB', 'cy')
    for suffix in _UNIT_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    # Check for SI multiplier as last character
    if s and s[-1] in _SI_MULTIPLIERS:
        return float(s[:-1]) * _SI_MULTIPLIERS[s[-1]]
    return float(s)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def build_lauka_command(
    command: Sequence[str],
    events: Sequence[str],
    *,
    lauka_bin: str = "lauka",
    runs: int = 5,
    warmup: int = 1,
) -> List[str]:
    child_cmd = " ".join(str(a) for a in command)
    return [
        "sudo", lauka_bin,
        "-n", str(runs),
        "--warmup", str(warmup),
        "--color", "never",
        "-m", ",".join(events),
        "--", child_cmd,
    ]


def parse_lauka_output(stdout: str, requested_events: Sequence[str]) -> Dict[str, object]:
    """Parse lauka table output into hwc fields.

    Each measurement row looks like:
      core_active_cycle           2.51G ± 22.1M     2.48G … 2.54G        0 (0%)
    """
    cleaned = _ANSI_RE.sub("", stdout)
    requested = [e.strip() for e in requested_events if e.strip()]
    recorded: List[str] = []
    values: Dict[str, float] = {}

    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        # Split on whitespace; first token is counter name, second is mean value
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        # Skip header/separator lines
        if name.startswith("-") or name.startswith("=") or name.lower() == "counter":
            continue
        # Try to parse the mean value (second token)
        try:
            mean_val = parse_si_value(parts[1])
        except (ValueError, IndexError):
            continue
        recorded.append(name)
        values[name] = mean_val

    requested_set = set(requested)
    recorded_set = set(recorded)
    if not recorded:
        status = "unavailable"
    elif requested_set.issubset(recorded_set):
        status = "ok"
    else:
        status = "partial"

    return default_hwc_fields(
        status=status,
        tool="lauka",
        events_requested=requested,
        events_recorded=recorded,
        event_values=values,
    )


def run_lauka_collection(
    command: Sequence[str],
    hwc_config: HardwareCounterConfig,
    timeout: int = 1200,
) -> Dict[str, object]:
    """Run a separate lauka invocation to collect hardware counters."""
    lauka_cmd = build_lauka_command(
        command,
        hwc_config.events,
        lauka_bin=hwc_config.lauka_bin,
        runs=hwc_config.lauka_runs,
        warmup=hwc_config.lauka_warmup,
    )
    process = subprocess.Popen(
        lauka_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:  # noqa: BLE001
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:  # noqa: BLE001
                process.kill()
            stdout, stderr = process.communicate()
        raise RuntimeError(f"lauka timed out after {timeout}s")

    if process.returncode == 2:
        raise RuntimeError(
            f"lauka PMU scheduling error — counter combination may be incompatible. "
            f"stderr: {(stderr or '').strip()}"
        )
    if process.returncode != 0:
        msg = (stderr or "").strip() or (stdout or "").strip()
        if "sudo" in msg.lower() or "password" in msg.lower():
            raise RuntimeError(f"lauka requires sudo privileges: {msg}")
        raise RuntimeError(f"lauka failed (exit {process.returncode}): {msg}")

    return parse_lauka_output(stdout, hwc_config.events)


def run_benchmark_command(
    args: Sequence[str],
    *,
    timeout: int = 600,
    hwc_config: HardwareCounterConfig | None = None,
) -> CommandResult:
    command = list(args)
    if hwc_config and hwc_config.mode == "perf":
        command = build_perf_command(command, hwc_config.events)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:  # noqa: BLE001
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception:  # noqa: BLE001
                process.kill()
            stdout, stderr = process.communicate()
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(command)}")

    if process.returncode != 0:
        stderr = (stderr or "").strip()
        stdout = (stdout or "").strip()
        raise RuntimeError(stderr or stdout or f"Command failed: {' '.join(command)}")
    return CommandResult(stdout=stdout, stderr=stderr)


def run_trials(
    command: Sequence[str],
    *,
    trials: int,
    timeout: int,
    hwc_config: HardwareCounterConfig | None = None,
) -> Tuple[Dict[str, float], TrialSummary, str, Dict[str, object]]:
    if trials <= 0:
        raise ValueError("--trials must be > 0")

    stdouts: List[str] = []
    metrics_list: List[Dict[str, float]] = []
    hwc_results: List[Dict[str, object]] = []
    avg_times: List[float] = []
    max_errors: List[float] = []

    # For lauka mode, don't wrap trials — run them directly for timing,
    # then collect counters in a separate lauka invocation afterward.
    # For perf mode, wrap each trial as before.
    trial_hwc_config = hwc_config if (hwc_config and hwc_config.mode == "perf") else None

    for _ in range(trials):
        result = run_benchmark_command(command, timeout=timeout, hwc_config=trial_hwc_config)
        metrics = parse_timing_metrics(result.stdout)
        stdouts.append(result.stdout)
        metrics_list.append(metrics)
        if trial_hwc_config:
            hwc_fields = parse_perf_stat_output(result.stderr, trial_hwc_config.events)
            if trial_hwc_config.strict and hwc_fields["hwc_status"] != "ok":
                raise RuntimeError(
                    "Requested perf events unavailable or partial: "
                    f"{hwc_fields['hwc_events_recorded'] or '<none>'}"
                )
        else:
            hwc_fields = default_hwc_fields()
        hwc_results.append(hwc_fields)
        avg_times.append(float(metrics["avg_time_ms"]))
        max_errors.append(float(metrics.get("max_error", 0.0)))

    ranked = sorted(range(trials), key=lambda i: (avg_times[i], i))
    selected_rank = (trials - 1) // 2  # median-low in sorted order
    selected_idx = ranked[selected_rank]

    ordered_avg = sorted(avg_times)
    median_avg = ordered_avg[selected_rank]
    mean_avg = statistics.fmean(avg_times)
    stddev_avg = statistics.pstdev(avg_times) if trials > 1 else 0.0
    variance_pct = (stddev_avg / mean_avg * 100.0) if mean_avg > 0 else 0.0
    max_error_max = max(max_errors) if max_errors else 0.0

    summary = TrialSummary(
        trials=trials,
        trial_selected=selected_idx,
        trial_avg_time_ms_median=float(median_avg),
        trial_avg_time_ms_min=float(min(avg_times)),
        trial_avg_time_ms_max=float(max(avg_times)),
        trial_avg_time_ms_stddev=float(stddev_avg),
        trial_avg_time_ms_variance_pct=float(variance_pct),
        trial_max_error_max=float(max_error_max),
    )

    # After trials, do one lauka collection if mode == "lauka"
    if hwc_config and hwc_config.mode == "lauka":
        try:
            hwc_final = run_lauka_collection(command, hwc_config, timeout=timeout)
            if hwc_config.strict and hwc_final["hwc_status"] != "ok":
                raise RuntimeError(
                    "Requested lauka events unavailable or partial: "
                    f"{hwc_final['hwc_events_recorded'] or '<none>'}"
                )
        except Exception as exc:
            if hwc_config.strict:
                raise
            hwc_final = default_hwc_fields(
                status="error", tool="lauka",
                events_requested=hwc_config.events,
            )
    else:
        hwc_final = hwc_results[selected_idx]

    return metrics_list[selected_idx], summary, stdouts[selected_idx], hwc_final


def run_command(args: Sequence[str], timeout: int = 600) -> str:
    return run_benchmark_command(args, timeout=timeout).stdout


def generate_kernel(dsl_file: Path, config_flags: Sequence[str], output_c: Path) -> None:
    output_c.parent.mkdir(parents=True, exist_ok=True)
    args = [str(COMPILER), str(dsl_file), *config_flags, "-o", str(output_c)]
    run_command(args, timeout=300)


def compile_c_to_executable(c_file: Path, output_exe: Path) -> None:
    args = ["gcc", *GCC_FLAGS, str(c_file), "-o", str(output_exe)]
    run_command(args, timeout=300)


def parse_timing_metrics(stdout: str) -> Dict[str, float]:
    iter_match = re.search(r"Iterations: (\d+)", stdout)
    total_match = re.search(r"Total time: ([\d.]+) ms", stdout)
    avg_match = re.search(r"Avg time per iteration: ([\d.]+) ms", stdout)
    min_match = re.search(r"Min time per iteration: ([\d.]+) ms", stdout)
    max_match = re.search(r"Max time per iteration: ([\d.]+) ms", stdout)
    stddev_match = re.search(r"Stddev time: ([\d.]+) ms", stdout)
    variance_match = re.search(r"Variance: ([\d.]+)%", stdout)
    error_match = re.search(r"Max error vs reference: ([\d.eE+-]+)", stdout)
    format_match = re.search(r"Format: ([A-Za-z0-9_]+)", stdout)

    if not (iter_match and total_match and avg_match):
        raise ValueError(f"Failed to parse benchmark output:\n{stdout}")

    return {
        "format": format_match.group(1).lower() if format_match else "",
        "iterations": int(iter_match.group(1)),
        "total_time_ms": float(total_match.group(1)),
        "avg_time_ms": float(avg_match.group(1)),
        "min_time_ms": float(min_match.group(1)) if min_match else 0.0,
        "max_time_ms": float(max_match.group(1)) if max_match else 0.0,
        "stddev_ms": float(stddev_match.group(1)) if stddev_match else 0.0,
        "variance_pct": float(variance_match.group(1)) if variance_match else 0.0,
        "max_error": float(error_match.group(1)) if error_match else 0.0,
    }


def parse_features(stdout: str) -> Dict[str, float]:
    """Parse FEATURE: lines from generated binary output into dataclass field dict."""
    features: Dict[str, float] = {}
    for match in re.finditer(r"FEATURE: (\w+)=([\d.eE+-]+)", stdout):
        key = match.group(1)
        field = _FEATURE_KEY_TO_FIELD.get(key)
        if field:
            features[field] = float(match.group(2))
    return features


def save_csv(rows: Sequence[object], output_file: Path, row_type) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(row_type)])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def reset_output_csvs(output_file: Path, failed_output: Path) -> None:
    if output_file.exists():
        output_file.unlink()
    if failed_output.exists():
        failed_output.unlink()


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


def collect_matrix_meta(matrices_dir: Path) -> List[MatrixMeta]:
    matrices: List[MatrixMeta] = []
    for matrix_path in sorted(matrices_dir.glob("*.mtx")):
        rows, cols, nnz = read_matrix_header(matrix_path)
        matrices.append(
            MatrixMeta(
                name=matrix_path.stem,
                path=matrix_path,
                rows=rows,
                cols=cols,
                nnz=nnz,
            )
        )
    return matrices


def select_matrices(matrices: Sequence[MatrixMeta], max_matrices: int) -> List[MatrixMeta]:
    if max_matrices <= 0:
        raise ValueError("--max-matrices must be > 0")
    ordered = sorted(matrices, key=lambda item: item.name)
    return ordered[:max_matrices]


def select_matrices_custom(
    matrices: Sequence[MatrixMeta],
    max_matrices: int,
    *,
    ordered_names: Sequence[str] | None = None,
    requested_names: Sequence[str] | None = None,
) -> List[MatrixMeta]:
    if max_matrices <= 0:
        raise ValueError("--max-matrices must be > 0")
    matrix_by_name = {matrix.name: matrix for matrix in matrices}

    if ordered_names is not None:
        ordered: List[MatrixMeta] = [matrix_by_name[name] for name in ordered_names if name in matrix_by_name]
    else:
        ordered = sorted(matrices, key=lambda item: item.name)

    if requested_names:
        requested_unique = []
        seen = set()
        for name in requested_names:
            if name in seen:
                continue
            seen.add(name)
            requested_unique.append(name)
        missing = [name for name in requested_unique if name not in matrix_by_name]
        if missing:
            raise ValueError(f"Unknown matrix name(s): {', '.join(missing)}")
        if ordered_names is not None:
            allowed = set(requested_unique)
            ordered = [matrix for matrix in ordered if matrix.name in allowed]
        else:
            ordered = [matrix_by_name[name] for name in requested_unique]

    return ordered[:max_matrices]


def load_matrix_manifest(path: Path) -> List[MatrixManifestEntry]:
    if not path.exists():
        raise FileNotFoundError(f"Matrix manifest not found: {path}")
    entries: List[MatrixManifestEntry] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "name" not in (reader.fieldnames or []):
            raise ValueError(f"Manifest missing required 'name' column: {path}")
        for row in reader:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            char_map = zero_characteristics(prefix="char_")
            for key in CHARACTERISTIC_NAMES:
                raw = row.get(key, "")
                if raw not in ("", None):
                    char_map[key] = float(raw)
            entries.append(
                MatrixManifestEntry(
                    name=name,
                    source=(row.get("source") or "").strip(),
                    generator=(row.get("generator") or "").strip(),
                    seed=int(row.get("seed") or 0),
                    params_json=(row.get("params_json") or "").strip(),
                    tags=(row.get("tags") or "").strip(),
                    characteristics=char_map,
                )
            )
    return entries


def manifest_lookup(entries: Sequence[MatrixManifestEntry]) -> Dict[str, MatrixManifestEntry]:
    out: Dict[str, MatrixManifestEntry] = {}
    for entry in entries:
        out[entry.name] = entry
    return out


def build_single_matrix_metadata(
    matrix_name: str,
    manifest_by_name: Dict[str, MatrixManifestEntry] | None,
) -> Dict[str, object]:
    if not manifest_by_name or matrix_name not in manifest_by_name:
        defaults: Dict[str, object] = {
            "matrix_source": "",
            "matrix_generator": "",
            "matrix_seed": 0,
            "matrix_params_json": "",
            "matrix_tags": "",
        }
        defaults.update(zero_characteristics(prefix="char_"))
        return defaults

    entry = manifest_by_name[matrix_name]
    metadata: Dict[str, object] = {
        "matrix_source": entry.source,
        "matrix_generator": entry.generator,
        "matrix_seed": entry.seed,
        "matrix_params_json": entry.params_json,
        "matrix_tags": entry.tags,
    }
    for key in CHARACTERISTIC_NAMES:
        metadata[key] = entry.characteristics[key]
    return metadata


def build_pair_matrix_metadata(
    matrix_a: str,
    matrix_b: str,
    manifest_by_name: Dict[str, MatrixManifestEntry] | None,
) -> Dict[str, object]:
    values: Dict[str, object] = {}
    if not manifest_by_name:
        for key in PAIR_A_META_FIELDS + PAIR_B_META_FIELDS:
            values[key] = "" if not key.endswith("_seed") else 0
        for key in PAIR_A_CHAR_FIELDS + PAIR_B_CHAR_FIELDS:
            values[key] = 0.0
        return values

    entry_a = manifest_by_name.get(matrix_a)
    entry_b = manifest_by_name.get(matrix_b)

    values["matrix_a_source"] = entry_a.source if entry_a else ""
    values["matrix_a_generator"] = entry_a.generator if entry_a else ""
    values["matrix_a_seed"] = entry_a.seed if entry_a else 0
    values["matrix_a_params_json"] = entry_a.params_json if entry_a else ""
    values["matrix_a_tags"] = entry_a.tags if entry_a else ""
    values["matrix_b_source"] = entry_b.source if entry_b else ""
    values["matrix_b_generator"] = entry_b.generator if entry_b else ""
    values["matrix_b_seed"] = entry_b.seed if entry_b else 0
    values["matrix_b_params_json"] = entry_b.params_json if entry_b else ""
    values["matrix_b_tags"] = entry_b.tags if entry_b else ""

    char_a = entry_a.characteristics if entry_a else zero_characteristics(prefix="char_")
    char_b = entry_b.characteristics if entry_b else zero_characteristics(prefix="char_")
    for base in CHARACTERISTIC_BASE_NAMES:
        values[f"char_a_{base}"] = char_a.get(f"char_{base}", 0.0)
        values[f"char_b_{base}"] = char_b.get(f"char_{base}", 0.0)
    return values


def is_compatible_pair(kernel: str, matrix_a: MatrixMeta, matrix_b: MatrixMeta) -> bool:
    if kernel in ("spadd", "spelmul"):
        return matrix_a.rows == matrix_b.rows and matrix_a.cols == matrix_b.cols
    if kernel == "spgemm":
        return matrix_a.cols == matrix_b.rows
    raise ValueError(f"Unsupported two-sparse kernel '{kernel}'")


def load_pairs_file(path: Path, *, kernel: str | None = None) -> List[Tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Pairs file not found: {path}")
    pairs: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "matrix_a" not in reader.fieldnames or "matrix_b" not in reader.fieldnames:
            raise ValueError(f"Pairs file must include matrix_a,matrix_b columns: {path}")
        for row in reader:
            row_kernel = (row.get("kernel") or "").strip().lower()
            if kernel and row_kernel and row_kernel != kernel:
                continue
            matrix_a = (row.get("matrix_a") or "").strip()
            matrix_b = (row.get("matrix_b") or "").strip()
            if not matrix_a or not matrix_b:
                continue
            pairs.append((matrix_a, matrix_b))
    return pairs


def select_pairs(
    kernel: str,
    matrices: Sequence[MatrixMeta],
    max_pairs: int,
    *,
    explicit_pairs: Sequence[Tuple[str, str]] | None = None,
) -> List[Tuple[MatrixMeta, MatrixMeta]]:
    if max_pairs <= 0:
        raise ValueError("--max-pairs must be > 0")
    matrix_by_name = {matrix.name: matrix for matrix in matrices}
    if explicit_pairs is not None:
        resolved: List[Tuple[MatrixMeta, MatrixMeta]] = []
        for matrix_a_name, matrix_b_name in explicit_pairs:
            if matrix_a_name not in matrix_by_name or matrix_b_name not in matrix_by_name:
                raise ValueError(f"Unknown matrix in pair ({matrix_a_name}, {matrix_b_name})")
            matrix_a = matrix_by_name[matrix_a_name]
            matrix_b = matrix_by_name[matrix_b_name]
            if not is_compatible_pair(kernel, matrix_a, matrix_b):
                raise ValueError(f"Incompatible pair for {kernel}: ({matrix_a_name}, {matrix_b_name})")
            resolved.append((matrix_a, matrix_b))
        return resolved[:max_pairs]

    pairs: List[Tuple[MatrixMeta, MatrixMeta]] = []
    for matrix_a in matrices:
        for matrix_b in matrices:
            if is_compatible_pair(kernel, matrix_a, matrix_b):
                pairs.append((matrix_a, matrix_b))
    pairs.sort(key=lambda item: (item[0].name, item[1].name))
    return pairs[:max_pairs]


def write_pairs_csv(pairs: Sequence[Tuple[MatrixMeta, MatrixMeta]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "matrix_a",
                "matrix_b",
                "rows_a",
                "cols_a",
                "nnz_a",
                "rows_b",
                "cols_b",
                "nnz_b",
            ],
        )
        writer.writeheader()
        for matrix_a, matrix_b in pairs:
            writer.writerow(
                {
                    "matrix_a": matrix_a.name,
                    "matrix_b": matrix_b.name,
                    "rows_a": matrix_a.rows,
                    "cols_a": matrix_a.cols,
                    "nnz_a": matrix_a.nnz,
                    "rows_b": matrix_b.rows,
                    "cols_b": matrix_b.cols,
                    "nnz_b": matrix_b.nnz,
                }
            )
