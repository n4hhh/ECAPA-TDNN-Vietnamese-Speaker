"""Focused tests for deterministic cache-benchmark metadata helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.benchmark_cached_fbank_access import diagnostic_order, percentile, summary


class CachedFbankBenchmarkHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            SimpleNamespace(feature_shard_path="train/shard_00000.pt"),
            SimpleNamespace(feature_shard_path="train/shard_00000.pt"),
            SimpleNamespace(feature_shard_path="train/shard_00001.pt"),
            SimpleNamespace(feature_shard_path="train/shard_00001.pt"),
        ]

    def test_diagnostic_orders_are_deterministic_and_complete(self) -> None:
        for pattern in ("sequential", "sample-random", "shard-local"):
            first = diagnostic_order(self.rows, pattern, 20260727)
            second = diagnostic_order(self.rows, pattern, 20260727)
            self.assertEqual(first, second)
            self.assertEqual(sorted(first), list(range(len(self.rows))))

    def test_shard_local_order_keeps_each_shard_contiguous(self) -> None:
        order = diagnostic_order(self.rows, "shard-local", 20260727)
        shard_sequence = [self.rows[index].feature_shard_path for index in order]
        self.assertIn(shard_sequence, [
            ["train/shard_00000.pt"] * 2 + ["train/shard_00001.pt"] * 2,
            ["train/shard_00001.pt"] * 2 + ["train/shard_00000.pt"] * 2,
        ])

    def test_percentile_and_summary_schema(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4], 0.25), 1.75)
        self.assertEqual(
            set(summary([1, 2, 3])),
            {"min", "mean", "q1", "median", "q3", "max"},
        )


if __name__ == "__main__":
    unittest.main()
