from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.verification_trials import (
    Trial, Utterance, generate_trials, read_trials, sha256_bytes, trials_csv_text,
    validate_trials, validate_trials_against_metadata,
)


class TrialTests(unittest.TestCase):
    def rows(self):
        return [Utterance(f"audio/{speaker}_{i}.wav", speaker) for speaker in ("a", "b", "c") for i in range(4)]

    def test_determinism_seed_validity_and_balance(self):
        first = generate_trials(self.rows(), 7, 3)
        second = generate_trials(self.rows(), 7, 3)
        other = generate_trials(self.rows(), 8, 3)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        validate_trials(first, {"a", "b", "c"})
        self.assertEqual(sum(t.target for t in first), len(first) // 2)
        self.assertEqual([t.trial_id for t in first], list(range(len(first))))

    def test_positive_cap_and_low_resource_all_pairs(self):
        rows = [Utterance(f"x/a{i}.wav", "a") for i in range(3)]
        rows += [Utterance(f"x/b{i}.wav", "b") for i in range(5)]
        trials = generate_trials(rows, 1, 4)
        self.assertEqual(sum(t.target == 1 and t.left_speaker_id == "a" for t in trials), 3)
        self.assertEqual(sum(t.target == 1 and t.left_speaker_id == "b" for t in trials), 4)

    def test_invalid_metadata_and_paths(self):
        with self.assertRaisesRegex(ValueError, "fewer than two"):
            generate_trials([Utterance("x/a.wav", "a"), Utterance("x/b.wav", "b"), Utterance("x/c.wav", "b")], 1)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            generate_trials([Utterance("C:/bad.wav", "a"), Utterance("x/a.wav", "a")], 1)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            generate_trials([Utterance("C:audio/file.wav", "a"), Utterance("x/a.wav", "a")], 1)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            generate_trials([Utterance("x\\a.wav", "a"), Utterance("x/b.wav", "a")], 1)
        with self.assertRaisesRegex(ValueError, "speaker_id"):
            generate_trials([Utterance("x/a.wav", ""), Utterance("x/b.wav", "")], 1)

    def test_round_trip_and_hash(self):
        trials = generate_trials(self.rows(), 22, 2)
        text = trials_csv_text(trials)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.csv"
            path.write_text(text, encoding="utf-8")
            self.assertEqual(read_trials(path), trials)
        self.assertEqual(sha256_bytes(text.encode()), sha256_bytes(trials_csv_text(generate_trials(self.rows(), 22, 2)).encode()))

    def test_trials_against_metadata(self):
        trials = [Trial(0, 0, "x/a.wav", "x/b.wav", "a", "b")]
        validate_trials_against_metadata(trials, {"x/a.wav": "a", "x/b.wav": "b"}, {"a", "b"})
        for changed, pattern in (
            (Trial(0, 0, "x/a.wav", "x/b.wav", "wrong", "b"), "left path"),
            (Trial(0, 0, "x/a.wav", "x/b.wav", "a", "wrong"), "right path"),
            (Trial(0, 0, "x/a.wav", "x/missing.wav", "a", "b"), "missing right path"),
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, pattern):
                validate_trials_against_metadata([changed], {"x/a.wav": "a", "x/b.wav": "b"})

    def test_swapped_fabricated_duplicate_and_real_target_contradiction(self):
        metadata = [("x/a.wav", "a"), ("x/b.wav", "b")]
        with self.assertRaisesRegex(ValueError, "belongs to"):
            validate_trials_against_metadata(
                [Trial(0, 0, "x/a.wav", "x/b.wav", "b", "a")], metadata
            )
        with self.assertRaisesRegex(ValueError, "duplicate metadata path"):
            validate_trials_against_metadata(
                [Trial(0, 0, "x/a.wav", "x/b.wav", "a", "b")],
                metadata + [("x/a.wav", "a")],
            )
        with self.assertRaisesRegex(ValueError, "contradicts real"):
            validate_trials_against_metadata(
                [Trial(0, 1, "x/a.wav", "x/b.wav", "a", "b")], metadata
            )


if __name__ == "__main__":
    unittest.main()
