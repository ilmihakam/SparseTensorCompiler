#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_RUN_DIR = Path(__file__).resolve().parent.parent / "run"
if str(_RUN_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_DIR))

from benchmark_common import build_config_specs


class ConfigSweepsUnitTest(unittest.TestCase):
    def test_1d_sweep_expansion_order(self) -> None:
        specs = build_config_specs(
            "spmv",
            ["baseline"],
            sweep_block_sizes=[16, 32],
            sweep_orders=["I_THEN_B", "B_THEN_I", "I_B_I"],
        )
        self.assertEqual(
            [spec.name for spec in specs],
            [
                "baseline",
                "block_b16",
                "all_I_THEN_B_b16",
                "all_B_THEN_I_b16",
                "all_I_B_I_b16",
                "block_b32",
                "all_I_THEN_B_b32",
                "all_B_THEN_I_b32",
                "all_I_B_I_b32",
            ],
        )

    def test_2d_sweep_only_for_spmm_sddmm(self) -> None:
        specs = build_config_specs(
            "spmm",
            ["baseline"],
            sweep_block2d_sizes=[(16, 16)],
            sweep_orders=["I_THEN_B", "B_THEN_I", "I_B_I"],
        )
        self.assertEqual(
            [spec.name for spec in specs],
            [
                "baseline",
                "block2d_b16x16",
                "all2d_I_THEN_B_b16x16",
                "all2d_B_THEN_I_b16x16",
                "all2d_I_B_I_b16x16",
            ],
        )

        with self.assertRaises(ValueError):
            _ = build_config_specs(
                "spmv",
                ["baseline"],
                sweep_block2d_sizes=[(16, 16)],
            )


if __name__ == "__main__":
    unittest.main()

