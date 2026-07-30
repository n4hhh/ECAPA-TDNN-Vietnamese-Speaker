"""Metrics-only verification from immutable saved scores.

This command intentionally imports neither torch nor SpeechBrain and never
reads WAVs, Fbank configuration payloads, cache indexes, or cache shards.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.final_evaluation_v2 import (  # noqa: E402
    atomic_json,
    calculate_final_metrics,
    load_score_artifact,
    read_json,
    sha256_file,
)


def main() -> None:
    lock_identity_path = REPO_ROOT / "configs/v2/final_evaluation_v2_identity.json"
    lock_identity = read_json(lock_identity_path)
    report = read_json(REPO_ROOT / "reports/final_evaluation_v2.json")
    if (
        report.get("one_time_status") != "finalized"
        or report.get("lock_identity_sha256") != sha256_file(lock_identity_path)
        or not lock_identity.get("protocol_locked")
    ):
        raise RuntimeError("metrics-only recomputation requires finalized evaluation")
    score_path = REPO_ROOT / "outputs/final_evaluation_v2/final_test_scores_v2.json"
    artifact = load_score_artifact(score_path)
    reproduced = calculate_final_metrics(
        artifact["scores"],
        artifact["targets"],
        report["validation_reference"],
    )
    if reproduced != report["metrics"]:
        raise RuntimeError("metrics-only reproduction differs from finalized report")
    result = {
        "schema_version": 2,
        "result": "PASS",
        "mode": "metrics_only",
        "score_artifact_sha256": sha256_file(score_path),
        "metrics_exactly_reproduced": True,
        "speechbrain_imported": False,
        "cuda_loaded": False,
        "wav_accessed": False,
        "fbank_accessed": False,
        "trials_rescored": False,
    }
    output = (
        REPO_ROOT
        / "outputs/final_evaluation_v2/metrics_only_recalculation_v2.json"
    )
    atomic_json(output, result)
    print("Metrics-only final-test reproduction: PASS")


if __name__ == "__main__":
    main()
