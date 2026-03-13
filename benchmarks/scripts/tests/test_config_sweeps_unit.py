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

    def test_dynamic_position_and_explicit_2d_configs(self) -> None:
        specs = build_config_specs(
            "spmv",
            ["posblock_p32", "block2d_i_j_b16x32", "full_i_j_p8_b16x32"],
        )
        self.assertEqual(
            [spec.name for spec in specs],
            ["posblock_p32", "block2d_i_j_b16x32", "full_i_j_p8_b16x32"],
        )
        self.assertEqual(specs[0].flags, ["--opt-block-pos=32"])
        self.assertEqual(
            specs[1].flags,
            ["--opt-block-2d=16x32", "--opt-block-2d-targets=i,j"],
        )
        self.assertEqual(
            specs[2].flags,
            [
                "--opt-block-2d=16x32",
                "--opt-block-2d-targets=i,j",
                "--opt-block-pos=8",
            ],
        )


if __name__ == "__main__":
    unittest.main()
