from __future__ import annotations

import os
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path

from src.verification_v2 import (
    ValidationManifestRow,
    empirical_confusion,
    generate_validation_trials,
    trials_csv_bytes,
)


def fixture_rows() -> tuple[ValidationManifestRow, ...]:
    rows = []
    for speaker in ("10", "20", "30", "40"):
        for index in range(4):
            rows.append(
                ValidationManifestRow(
                    audio_path=f"{speaker}/{index}.wav",
                    speaker_id=speaker,
                    duplicate_group=(
                        f"dup-{speaker}" if index in (0, 1) else ""
                    ),
                )
            )
    return tuple(rows)


class VerificationV2Tests(unittest.TestCase):
    def test_balanced_protocol_and_duplicate_safety(self) -> None:
        trials, metadata = generate_validation_trials(
            fixture_rows(),
            seed=20260729,
            positive_per_speaker=2,
            negative_rounds=3,
        )
        self.assertEqual(len(trials), 14)
        positives = [trial for trial in trials if trial.target == 1]
        negatives = [trial for trial in trials if trial.target == 0]
        self.assertEqual(Counter(t.left_speaker_id for t in positives), {
            "10": 2, "20": 2, "30": 2, "40": 2,
        })
        participation = Counter()
        for trial in negatives:
            participation[trial.left_speaker_id] += 1
            participation[trial.right_speaker_id] += 1
        self.assertEqual(participation, {
            "10": 3, "20": 3, "30": 3, "40": 3,
        })
        groups = {
            row.audio_path: row.duplicate_group for row in fixture_rows()
        }
        for trial in positives:
            left, right = groups[trial.left_audio_path], groups[trial.right_audio_path]
            self.assertFalse(left and left == right)
        pairs = {
            tuple(sorted((trial.left_audio_path, trial.right_audio_path)))
            for trial in trials
        }
        self.assertEqual(len(pairs), len(trials))
        self.assertEqual(metadata["positive_duplicate_group_pair_rejections"], 4)

    def test_trial_bytes_are_repeatable(self) -> None:
        first, _ = generate_validation_trials(
            fixture_rows(), positive_per_speaker=2, negative_rounds=3
        )
        second, _ = generate_validation_trials(
            fixture_rows(), positive_per_speaker=2, negative_rounds=3
        )
        self.assertEqual(trials_csv_bytes(first), trials_csv_bytes(second))

    def test_trial_bytes_are_pythonhashseed_independent(self) -> None:
        code = (
            "from src.verification_v2 import ValidationManifestRow as R,"
            "generate_validation_trials as g,trials_csv_bytes as b;"
            "rows=tuple(R(f'{s}/{i}.wav',s,f'dup-{s}' if i<2 else '') "
            "for s in ('10','20','30','40') for i in range(4));"
            "print(b(g(rows,positive_per_speaker=2,negative_rounds=3)[0]).hex())"
        )
        outputs = []
        for hash_seed in ("2", "123456"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = hash_seed
            outputs.append(
                subprocess.check_output(
                    [sys.executable, "-c", code],
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    text=True,
                ).strip()
            )
        self.assertEqual(outputs[0], outputs[1])

    def test_empirical_confusion_uses_greater_equal_threshold(self) -> None:
        result = empirical_confusion(
            [0.9, 0.5, 0.5, 0.1],
            [1, 1, 0, 0],
            0.5,
        )
        self.assertEqual(
            {key: result[key] for key in ("tp", "tn", "fp", "fn")},
            {"tp": 2, "tn": 1, "fp": 1, "fn": 0},
        )
        self.assertEqual(result["far"], 0.5)
        self.assertEqual(result["frr"], 0.0)


if __name__ == "__main__":
    unittest.main()
