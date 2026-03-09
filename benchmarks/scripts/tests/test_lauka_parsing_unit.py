#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_RUN_DIR = Path(__file__).resolve().parent.parent / "run"
if str(_RUN_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_DIR))

from benchmark_common import (
    _NORMALIZED_HWC_FIELDS,
    build_hwc_config,
    build_lauka_command,
    normalize_event_name,
    parse_lauka_output,
    parse_si_value,
)


class TestParseSiValue(unittest.TestCase):
    def test_plain_integer(self) -> None:
        self.assertAlmostEqual(parse_si_value("591"), 591.0)

    def test_plain_float(self) -> None:
        self.assertAlmostEqual(parse_si_value("3.14"), 3.14)

    def test_giga(self) -> None:
        self.assertAlmostEqual(parse_si_value("2.51G"), 2.51e9)

    def test_mega(self) -> None:
        self.assertAlmostEqual(parse_si_value("3.58M"), 3.58e6)

    def test_kilo(self) -> None:
        self.assertAlmostEqual(parse_si_value("22.1K"), 22100.0)

    def test_tera(self) -> None:
        self.assertAlmostEqual(parse_si_value("1.5T"), 1.5e12)

    def test_with_ms_unit(self) -> None:
        self.assertAlmostEqual(parse_si_value("591ms"), 591.0)

    def test_with_MB_unit(self) -> None:
        self.assertAlmostEqual(parse_si_value("137MB"), 137.0)

    def test_with_cy_unit_and_si(self) -> None:
        self.assertAlmostEqual(parse_si_value("2.51Gcy"), 2.51e9)

    def test_with_s_unit(self) -> None:
        self.assertAlmostEqual(parse_si_value("12.3s"), 12.3)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_si_value("")

    def test_whitespace_stripped(self) -> None:
        self.assertAlmostEqual(parse_si_value("  42  "), 42.0)


class TestParseLaukaOutput(unittest.TestCase):
    SAMPLE_OUTPUT = """\
  core_active_cycle           2.51G ± 22.1M     2.48G … 2.54G        0 (0%)
  inst_all                    5.02G ± 15.3M     5.00G … 5.05G        0 (0%)
  l1d_cache_miss_ld_nonspec   1.23M ± 45.2K     1.18M … 1.28M        0 (0%)
  branch_mispred_nonspec      591K  ± 12.3K     578K  … 604K         0 (0%)
  l1i_cache_miss_demand       8.45K ± 234       8.12K … 8.78K        0 (0%)
  map_stall_dispatch          3.21M ± 98.7K     3.11M … 3.31M        0 (0%)
"""

    def test_parses_all_counters(self) -> None:
        events = [
            "core_active_cycle", "inst_all", "l1d_cache_miss_ld_nonspec",
            "branch_mispred_nonspec", "l1i_cache_miss_demand", "map_stall_dispatch",
        ]
        result = parse_lauka_output(self.SAMPLE_OUTPUT, events)
        self.assertEqual(result["hwc_status"], "ok")
        self.assertEqual(result["hwc_tool"], "lauka")
        vals = result["hwc_event_values_json"]
        import json
        parsed = json.loads(vals)
        self.assertAlmostEqual(parsed["core_active_cycle"], 2.51e9)
        self.assertAlmostEqual(parsed["inst_all"], 5.02e9)
        self.assertAlmostEqual(parsed["l1d_cache_miss_ld_nonspec"], 1.23e6)
        self.assertAlmostEqual(parsed["branch_mispred_nonspec"], 591e3)

    def test_normalized_fields_populated(self) -> None:
        events = ["core_active_cycle", "inst_all"]
        result = parse_lauka_output(self.SAMPLE_OUTPUT, events)
        self.assertAlmostEqual(result["hwc_cycles"], 2.51e9)
        self.assertAlmostEqual(result["hwc_instructions"], 5.02e9)

    def test_empty_input(self) -> None:
        result = parse_lauka_output("", ["core_active_cycle"])
        self.assertEqual(result["hwc_status"], "unavailable")
        self.assertEqual(result["hwc_cycles"], 0.0)

    def test_malformed_input(self) -> None:
        result = parse_lauka_output("not a valid table\ngarbage data", ["core_active_cycle"])
        # Should not crash; status should reflect missing events
        self.assertIn(result["hwc_status"], ("unavailable", "partial"))

    def test_strips_ansi(self) -> None:
        ansi_line = (
            "  \x1b[32mcore_active_cycle\x1b[0m           2.51G ± 22.1M"
            "     2.48G … 2.54G        0 (0%)\n"
        )
        result = parse_lauka_output(ansi_line, ["core_active_cycle"])
        self.assertEqual(result["hwc_status"], "ok")
        self.assertAlmostEqual(result["hwc_cycles"], 2.51e9)

    def test_partial_when_event_missing(self) -> None:
        result = parse_lauka_output(self.SAMPLE_OUTPUT, ["core_active_cycle", "nonexistent_counter"])
        self.assertEqual(result["hwc_status"], "partial")


class TestBuildLaukaCommand(unittest.TestCase):
    def test_basic_command(self) -> None:
        cmd = build_lauka_command(
            ["/tmp/kernel", "matrix.mtx"],
            ["core_active_cycle", "inst_all"],
            lauka_bin="/usr/local/bin/lauka",
            runs=5,
            warmup=1,
        )
        self.assertEqual(cmd[0], "sudo")
        self.assertEqual(cmd[1], "/usr/local/bin/lauka")
        self.assertIn("-n", cmd)
        self.assertEqual(cmd[cmd.index("-n") + 1], "5")
        self.assertIn("--warmup", cmd)
        self.assertEqual(cmd[cmd.index("--warmup") + 1], "1")
        self.assertIn("--color", cmd)
        self.assertEqual(cmd[cmd.index("--color") + 1], "never")
        self.assertIn("-m", cmd)
        self.assertEqual(cmd[cmd.index("-m") + 1], "core_active_cycle,inst_all")
        self.assertIn("--", cmd)
        # Child command is joined as a single string after --
        dash_idx = cmd.index("--")
        self.assertEqual(cmd[dash_idx + 1], "/tmp/kernel matrix.mtx")

    def test_defaults(self) -> None:
        cmd = build_lauka_command(["./test"], ["inst_all"])
        self.assertEqual(cmd[0], "sudo")
        self.assertEqual(cmd[1], "lauka")
        self.assertEqual(cmd[cmd.index("-n") + 1], "5")
        self.assertEqual(cmd[cmd.index("--warmup") + 1], "1")


class TestNormalizedFieldMapping(unittest.TestCase):
    def test_lauka_cycles(self) -> None:
        self.assertEqual(_NORMALIZED_HWC_FIELDS["core_active_cycle"], "hwc_cycles")
        self.assertEqual(_NORMALIZED_HWC_FIELDS["fixed_cycles"], "hwc_cycles")

    def test_lauka_instructions(self) -> None:
        self.assertEqual(_NORMALIZED_HWC_FIELDS["inst_all"], "hwc_instructions")
        self.assertEqual(_NORMALIZED_HWC_FIELDS["fixed_instructions"], "hwc_instructions")

    def test_lauka_cache_misses(self) -> None:
        self.assertEqual(_NORMALIZED_HWC_FIELDS["l1d_cache_miss_ld_nonspec"], "hwc_cache_misses")

    def test_lauka_branch_misses(self) -> None:
        self.assertEqual(_NORMALIZED_HWC_FIELDS["branch_mispred_nonspec"], "hwc_branch_misses")

    def test_perf_mappings_unchanged(self) -> None:
        self.assertEqual(_NORMALIZED_HWC_FIELDS["cycles"], "hwc_cycles")
        self.assertEqual(_NORMALIZED_HWC_FIELDS["instructions"], "hwc_instructions")
        self.assertEqual(_NORMALIZED_HWC_FIELDS["cache-misses"], "hwc_cache_misses")
        self.assertEqual(_NORMALIZED_HWC_FIELDS["branch-misses"], "hwc_branch_misses")


class TestBuildHwcConfigLauka(unittest.TestCase):
    def test_lauka_requires_events(self) -> None:
        args = SimpleNamespace(
            hwc_mode="lauka", hwc_events="", hwc_strict=False,
            hwc_lauka_bin="lauka", hwc_lauka_runs=5, hwc_lauka_warmup=1,
        )
        with self.assertRaises(ValueError):
            build_hwc_config(args)

    def test_lauka_min_runs(self) -> None:
        args = SimpleNamespace(
            hwc_mode="lauka", hwc_events="inst_all", hwc_strict=False,
            hwc_lauka_bin="lauka", hwc_lauka_runs=2, hwc_lauka_warmup=1,
        )
        with self.assertRaises(ValueError):
            build_hwc_config(args)

    def test_lauka_config_populates_fields(self) -> None:
        args = SimpleNamespace(
            hwc_mode="lauka", hwc_events="core_active_cycle,inst_all", hwc_strict=True,
            hwc_lauka_bin="/opt/bin/lauka", hwc_lauka_runs=7, hwc_lauka_warmup=2,
        )
        config = build_hwc_config(args)
        self.assertEqual(config.mode, "lauka")
        self.assertEqual(config.events, ("core_active_cycle", "inst_all"))
        self.assertTrue(config.strict)
        self.assertEqual(config.lauka_bin, "/opt/bin/lauka")
        self.assertEqual(config.lauka_runs, 7)
        self.assertEqual(config.lauka_warmup, 2)


if __name__ == "__main__":
    unittest.main()
