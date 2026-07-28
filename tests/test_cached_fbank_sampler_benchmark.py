"""Focused tests for sampler benchmark helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_cached_fbank_samplers import (
    OutputPaths,
    STANDARD_BATCH_CSV,
    STANDARD_MARKDOWN,
    STANDARD_OUTPUT,
    aggregate_repeats,
    resolve_output_paths,
    rotated_candidate_order,
    validate_report_payload,
    write_validated_outputs,
)


class SamplerBenchmarkHelperTests(unittest.TestCase):
    def test_candidate_rotation_is_deterministic(self) -> None:
        candidates = ["a", "b", "c", "d"]
        self.assertEqual(
            rotated_candidate_order(candidates, 2, 17),
            rotated_candidate_order(candidates, 2, 17),
        )
        self.assertEqual(sorted(rotated_candidate_order(candidates, 2, 17)), candidates)
        self.assertNotEqual(
            rotated_candidate_order(candidates, 1, 17),
            rotated_candidate_order(candidates, 2, 17),
        )

    def test_median_min_max_aggregation(self) -> None:
        rows = [{"value": 3}, {"value": 1}, {"value": 8}]
        self.assertEqual(
            aggregate_repeats(rows, "value"),
            {"median": 3.0, "min": 1.0, "max": 8.0},
        )

    def test_standard_and_custom_output_resolution(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit --output"):
            resolve_output_paths(None, standard_run=False)
        self.assertEqual(
            resolve_output_paths(None, standard_run=True),
            OutputPaths(STANDARD_OUTPUT, STANDARD_MARKDOWN, STANDARD_BATCH_CSV),
        )
        explicit = Path("reports/debug_hybrid_w8.json")
        self.assertEqual(
            resolve_output_paths(explicit, standard_run=False),
            OutputPaths(
                explicit, Path("reports/debug_hybrid_w8.md"),
                Path("reports/debug_hybrid_w8_batches.csv"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "standard full-run artifact"):
            resolve_output_paths(STANDARD_OUTPUT, standard_run=False)

    @staticmethod
    def _valid_payload() -> dict[str, object]:
        return {
            "candidate_analysis": [{
                "candidate": "valid",
                "duplicate_indexes_inside_batches": 0,
                "out_of_range_indexes": 0,
                "pk_structure_violating_batches": 0,
                "malformed_batch_sizes": 0,
                "determinism": {
                    "same_seed_epoch_identical": True,
                    "different_epoch_different": True,
                    "dataset_unmodified": True,
                },
                "speaker_exposure": {"all_speakers_selected": True},
            }],
            "benchmark_runs": [{
                "candidate": "valid", "completed_successfully": True,
                "all_batches_valid": True,
            }],
        }

    def test_valid_candidate_analysis_passes(self) -> None:
        validate_report_payload(self._valid_payload(), benchmark_required=True)

    def test_invalid_metric_fails_closed(self) -> None:
        payload = self._valid_payload()
        payload["candidate_analysis"][0]["duplicate_indexes_inside_batches"] = 1
        with self.assertRaisesRegex(AssertionError, "duplicate indexes"):
            validate_report_payload(payload, benchmark_required=True)

    def test_pass_artifacts_are_not_written_for_invalid_payload(self) -> None:
        payload = self._valid_payload()
        payload["candidate_analysis"][0]["determinism"]["same_seed_epoch_identical"] = False
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = OutputPaths(
                root / "result.json", root / "result.md", root / "result_batches.csv"
            )
            with self.assertRaisesRegex(AssertionError, "same-seed determinism"):
                write_validated_outputs(
                    paths, payload, [{"candidate": "invalid"}],
                    benchmark_required=True,
                )
            self.assertFalse(paths.json.exists())
            self.assertFalse(paths.markdown.exists())
            self.assertFalse(paths.batch_csv.exists())


if __name__ == "__main__":
    unittest.main()
