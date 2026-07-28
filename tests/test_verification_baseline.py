from __future__ import annotations

import unittest

import torch

from src.verification_baseline import numeric_summary, score_trials, validate_saved_artifacts
from src.verification_trials import Trial
from scripts.recompute_pretrained_validation_metrics import render_markdown


class BaselineHelperTests(unittest.TestCase):
    def test_scoring_order_and_values(self):
        embeddings = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
        trials = [Trial(0, 0, "a", "c", "x", "y"), Trial(1, 1, "a", "b", "x", "x")]
        scores = score_trials(embeddings, ["a", "b", "c"], trials)
        self.assertEqual(scores.tolist(), [0., 1.])

    def test_missing_and_duplicate_paths(self):
        trial = [Trial(0, 0, "a", "b", "x", "y")]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            score_trials(torch.ones(2, 2), ["a", "a"], trial)
        with self.assertRaisesRegex(ValueError, "missing"):
            score_trials(torch.ones(2, 2), ["a", "c"], trial)

    def test_summary(self):
        summary = numeric_summary([1, 2, 3])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["median"], 2)

    def test_saved_artifact_validation(self):
        trials = [Trial(0, 0, "a", "b", "x", "y")]
        embedding = {
            "relative_audio_paths": ["a", "b"], "speaker_ids": ["x", "y"],
            "embeddings": torch.ones(2, 192, dtype=torch.float32),
        }
        score = {
            "trial_ids": [0], "targets": torch.tensor([0]),
            "scores": torch.tensor([.2], dtype=torch.float32),
        }
        returned = validate_saved_artifacts(embedding, score, trials, ["a", "b"], ["x", "y"])
        self.assertEqual(tuple(returned[0].shape), (2, 192))
        bad = dict(score, trial_ids=[1])
        with self.assertRaisesRegex(ValueError, "trial IDs"):
            validate_saved_artifacts(embedding, bad, trials, ["a", "b"], ["x", "y"])

    def test_metrics_report_separates_interpolated_and_empirical(self):
        payload = {
            "validation_rows": 2, "validation_speakers": 2, "embedding_shape": [2, 192],
            "positive_trials": 1, "negative_trials": 1, "score_count": 2,
            "score_range": [0.1, 0.9], "protected_hashes": {},
            "old_interpolated_eer": 0.0, "interpolated_eer_absolute_difference": 0.0,
            "old_interpolated_threshold": 0.5, "optimized_metric_runtime_seconds": 0.001,
            "unique_score_count": 2, "elapsed_seconds": 1.0, "utterances_per_second": 2.0,
            "peak_allocated_vram_bytes": 0, "peak_reserved_vram_bytes": 0,
            "embedding_save_load_max_difference": 0.0, "score_save_load_max_difference": 0.0,
            "metrics": {
                "interpolated_eer": 0.0, "interpolated_eer_percentage": 0.0,
                "interpolated_threshold": 0.5, "interpolated_far": 0.0,
                "interpolated_frr": 0.0, "eer_kind": "linearly_interpolated_roc_crossing",
                "eer_threshold_kind": "interpolated_non_empirical", "empirical_threshold": 0.5,
                "empirical_far": 0.0, "empirical_frr": 0.0, "empirical_far_frr_gap": 0.0,
                "empirical_average_error": 0.0,
                "empirical_threshold_semantics": "accept same speaker when score >= threshold",
            },
        }
        report = render_markdown(payload)
        self.assertIn("## Interpolated result", report)
        self.assertIn("## Executable empirical operating point", report)
        self.assertIn("may not be an empirical operating point", report)
        self.assertIn("No embedding extraction", report)


if __name__ == "__main__":
    unittest.main()
