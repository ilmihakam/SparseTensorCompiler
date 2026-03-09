#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

_RUN_DIR = Path(__file__).resolve().parent.parent / "run"
if str(_RUN_DIR) not in sys.path:
    sys.path.insert(0, str(_RUN_DIR))

from benchmark_common import build_hwc_config, parse_hwc_events, parse_perf_stat_output


class HardwareCountersUnitTest(unittest.TestCase):
    def test_parse_hwc_events_deduplicates(self) -> None:
        self.assertEqual(parse_hwc_events("cycles,instructions,cycles"), ["cycles", "instructions"])

    def test_build_hwc_config_requires_events_for_perf(self) -> None:
        args = SimpleNamespace(hwc_mode="perf", hwc_events="", hwc_strict=False)
        with self.assertRaises(ValueError):
            build_hwc_config(args)

    def test_parse_perf_stat_output_maps_generic_fields(self) -> None:
        stderr = "\n".join(
            [
                "1000,,cycles:u,100.00,,,",
                "2500,,instructions:u,100.00,,,",
                "40,,cache-misses,100.00,,,",
            ]
        )
        fields = parse_perf_stat_output(stderr, ["cycles:u", "instructions:u", "cache-misses"])
        self.assertEqual(fields["hwc_status"], "ok")
        self.assertEqual(fields["hwc_events_recorded"], "cycles:u,instructions:u,cache-misses")
        self.assertEqual(fields["hwc_cycles"], 1000.0)
        self.assertEqual(fields["hwc_instructions"], 2500.0)
        self.assertEqual(fields["hwc_cache_misses"], 40.0)

    def test_parse_perf_stat_output_marks_partial_when_event_missing(self) -> None:
        stderr = "1000,,cycles,100.00,,,"
        fields = parse_perf_stat_output(stderr, ["cycles", "instructions"])
        self.assertEqual(fields["hwc_status"], "partial")
        self.assertEqual(fields["hwc_cycles"], 1000.0)
        self.assertEqual(fields["hwc_instructions"], 0.0)


if __name__ == "__main__":
    unittest.main()
