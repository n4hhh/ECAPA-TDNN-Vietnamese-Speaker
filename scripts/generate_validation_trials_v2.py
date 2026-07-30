#!/usr/bin/env python
"""Generate fixed, deterministic VieSpeaker2.0 validation verification trials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training_readiness_v2 import (  # noqa: E402
    APPROVED_INPUTS,
    atomic_write_bytes,
    canonical_json_bytes,
    sha256_file,
    verify_approved_inputs,
)
from src.verification_v2 import (  # noqa: E402
    generate_validation_trials,
    read_validation_manifest,
    sha256_bytes,
    trials_csv_bytes,
)


DEFAULT_OUTPUT = Path("manifests/verification_v2")


def write_exact_or_fail(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError(f"refusing to replace incompatible artifact: {path}")
    atomic_write_bytes(path, payload)


def markdown_report(report: dict[str, object]) -> bytes:
    identity = report["trial_identity"]
    protocol = report["protocol_validation"]
    lines = [
        "# VieSpeaker2.0 fixed validation trials",
        "",
        f"Result: **{report['result']}**",
        "",
        "The fixed validation protocol contains 10,000 positive and 10,000 "
        "negative trials over all 100 approved validation speakers.",
        "",
        "- Positive balance: exactly 100 pairs per speaker",
        "- Negative balance: exactly 200 round-robin participations per speaker",
        "- Negative schedule: two complete 99-round cycles plus the first 2 rounds of cycle three",
        "- Pair identity: canonical unordered relative-path pairs with no duplicates",
        "- Positive duplicate protection: same nonempty duplicate group is rejected",
        "- Utterance choice: least-used selection with stable SHA-256 tie breaks",
        "",
        f"- Trial CSV SHA-256: `{identity['trial_csv_sha256']}`",
        f"- Trial config SHA-256: `{identity['trial_config_sha256']}`",
        f"- Speakers: {protocol['speaker_count']}",
        f"- Positive duplicate-group candidate rejections: {protocol['positive_duplicate_group_pair_rejections']}",
        "",
        "The artifacts contain no timestamps or absolute local paths.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def build_artifacts(output_dir: Path) -> dict[str, object]:
    upstream = verify_approved_inputs(REPO_ROOT)
    sampler_path = REPO_ROOT / "configs/v2/training_sampler_v2.json"
    sampler = json.loads(sampler_path.read_text(encoding="utf-8"))
    if (
        sampler.get("identity_kind") != "training_sampler_v2"
        or sampler.get("approved") is not True
    ):
        raise ValueError("training sampler config is not approved")
    sampler_hash = sha256_file(sampler_path)
    plan_hash = sampler["sampler_plan"]["sampler_plan_identity_sha256"]

    manifest_path = REPO_ROOT / APPROVED_INPUTS["validation_manifest"]["path"]
    rows = read_validation_manifest(manifest_path)
    trials, protocol = generate_validation_trials(
        rows,
        seed=20260729,
        positive_per_speaker=100,
        negative_rounds=200,
    )
    csv_payload = trials_csv_bytes(trials)
    repeat_trials, repeat_protocol = generate_validation_trials(
        rows,
        seed=20260729,
        positive_per_speaker=100,
        negative_rounds=200,
    )
    if (
        trials_csv_bytes(repeat_trials) != csv_payload
        or repeat_protocol != protocol
    ):
        raise RuntimeError("in-process trial reproduction failed")

    config = {
        "schema_version": 2,
        "identity_kind": "validation_trials_config_v2",
        "seed": 20260729,
        "input_split": "validation",
        "speaker_count": 100,
        "trial_counts": {
            "positive": 10000,
            "negative": 10000,
            "total": 20000,
        },
        "positive_protocol": {
            "pairs_per_speaker": 100,
            "selection": "stable_sha256_ranked_canonical_unordered_path_pairs",
            "same_nonempty_duplicate_group_pair_forbidden": True,
        },
        "negative_protocol": {
            "rounds": 200,
            "schedule": "two_99_round_cycles_plus_first_2_rounds_of_third_cycle",
            "participations_per_speaker": 200,
            "utterance_selection": "least_used_then_stable_sha256_tie_break",
        },
        "pair_uniqueness": "canonical_unordered_relative_audio_path_pair",
        "trial_order": "all_hash_ranked_positives_then_round_robin_negatives",
        "timestamps_in_identity": False,
        "absolute_paths_in_artifacts": False,
        "upstream_bindings": upstream,
        "training_sampler_binding": {
            "path": "configs/v2/training_sampler_v2.json",
            "sha256": sampler_hash,
            "sampler_plan_identity_sha256": plan_hash,
        },
    }
    config_payload = canonical_json_bytes(config)
    identity = {
        "schema_version": 2,
        "identity_kind": "validation_trials_v2",
        "trial_csv_path": "manifests/verification_v2/validation_trials_v2.csv",
        "trial_csv_sha256": sha256_bytes(csv_payload),
        "trial_config_path": "manifests/verification_v2/validation_trials_config_v2.json",
        "trial_config_sha256": sha256_bytes(config_payload),
        "trial_counts": config["trial_counts"],
        "speaker_count": 100,
        "seed": 20260729,
        "upstream_bindings": upstream,
        "training_sampler_binding": config["training_sampler_binding"],
        "deterministic_reproduction": {
            "same_process_byte_identical": True,
            "pythonhashseed_independent": True,
        },
        "timestamps_in_identity": False,
        "absolute_paths_in_identity": False,
    }
    identity_payload = canonical_json_bytes(identity)

    output_dir = output_dir.resolve()
    write_exact_or_fail(output_dir / "validation_trials_v2.csv", csv_payload)
    write_exact_or_fail(
        output_dir / "validation_trials_config_v2.json", config_payload
    )
    write_exact_or_fail(
        output_dir / "validation_trials_identity_v2.json", identity_payload
    )
    return {
        "schema_version": 2,
        "report_kind": "validation_trials_v2",
        "result": "PASS",
        "protocol_validation": protocol,
        "trial_config": config,
        "trial_identity": identity,
        "trial_identity_sha256": sha256_bytes(identity_payload),
        "same_process_byte_identical": True,
        "final_test_quarantine": {
            "manifest_opened": False,
            "audio_opened": False,
            "features_extracted": False,
            "trials_generated": False,
            "evaluation_run": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="repository-relative output directory",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    report = build_artifacts(output_dir)
    if output_dir.resolve() == (REPO_ROOT / DEFAULT_OUTPUT).resolve():
        write_exact_or_fail(
            REPO_ROOT / "reports/validation_trials_v2.json",
            canonical_json_bytes(report),
        )
        write_exact_or_fail(
            REPO_ROOT / "reports/validation_trials_v2.md",
            markdown_report(report),
        )
    print(json.dumps({
        "result": report["result"],
        "trial_csv_sha256": report["trial_identity"]["trial_csv_sha256"],
        "trial_identity_sha256": report["trial_identity_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
