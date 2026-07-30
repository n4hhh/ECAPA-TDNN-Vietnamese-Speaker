"""Create and permanently lock the metadata-only final-test protocol."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.final_evaluation_v2 import (  # noqa: E402
    SEED,
    APPROVED_INPUTS,
    assert_no_absolute_paths,
    atomic_bytes,
    atomic_json,
    build_cache_config,
    build_lock_config,
    build_lock_identity,
    build_trial_config,
    build_trial_identity,
    canonical_json,
    final_trials_csv_bytes,
    generate_final_trials,
    initialize_state,
    read_final_manifest,
    read_json,
    sha256_bytes,
    sha256_file,
    validate_final_trials,
    validate_lock,
    validate_split_disjointness,
    verify_approved_inputs,
)
from src.verification_v2 import read_trials_csv  # noqa: E402


TRIAL_DIR = REPO_ROOT / "manifests/verification_v2"
CSV_PATH = TRIAL_DIR / "final_test_trials_v2.csv"
TRIAL_CONFIG_PATH = TRIAL_DIR / "final_test_trials_config_v2.json"
TRIAL_IDENTITY_PATH = TRIAL_DIR / "final_test_trials_identity_v2.json"
CACHE_CONFIG_PATH = REPO_ROOT / "configs/v2/final_test_fbank_cache_v2.json"
LOCK_PATH = REPO_ROOT / "configs/v2/final_evaluation_v2.json"
LOCK_IDENTITY_PATH = REPO_ROOT / "configs/v2/final_evaluation_v2_identity.json"
REPRO_DIR = REPO_ROOT / "outputs/final_test_trials_v2_reproduction"


def artifact_payloads(seed: int) -> tuple[bytes, bytes, bytes]:
    verified = verify_approved_inputs(REPO_ROOT)
    rows = read_final_manifest(
        REPO_ROOT / APPROVED_INPUTS["final_test_manifest"]["path"]
    )
    validate_split_disjointness(
        rows,
        REPO_ROOT / APPROVED_INPUTS["train_manifest"]["path"],
        REPO_ROOT / APPROVED_INPUTS["validation_manifest"]["path"],
        REPO_ROOT / APPROVED_INPUTS["speaker_split"]["path"],
    )
    trials, metadata = generate_final_trials(rows, seed)
    csv_payload = final_trials_csv_bytes(trials)
    config_payload = canonical_json(build_trial_config(verified, seed)).encode("utf-8")
    identity = build_trial_identity(
        verified,
        sha256_bytes(config_payload),
        sha256_bytes(csv_payload),
        seed,
        metadata,
    )
    identity_payload = canonical_json(identity).encode("utf-8")
    return csv_payload, config_payload, identity_payload


def write_reproduction(seed: int, output_dir: Path) -> None:
    payloads = artifact_payloads(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in zip(
        (
            "final_test_trials_v2.csv",
            "final_test_trials_config_v2.json",
            "final_test_trials_identity_v2.json",
        ),
        payloads,
    ):
        atomic_bytes(output_dir / name, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--reproduction-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=REPRO_DIR)
    args = parser.parse_args()
    if args.seed != SEED:
        raise ValueError(f"approved final-test trial seed is exactly {SEED}")
    if args.reproduction_only:
        write_reproduction(args.seed, args.output_dir)
        return

    targets = (
        CSV_PATH,
        TRIAL_CONFIG_PATH,
        TRIAL_IDENTITY_PATH,
        CACHE_CONFIG_PATH,
        LOCK_PATH,
        LOCK_IDENTITY_PATH,
        REPO_ROOT / "outputs/final_evaluation_v2/runtime_state_v2.json",
    )
    existing = [path.relative_to(REPO_ROOT).as_posix() for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"protocol lock targets already exist: {existing}")

    started = time.perf_counter()
    canonical_payloads = artifact_payloads(args.seed)
    reproduction_dir = args.output_dir
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--seed",
        str(args.seed),
        "--reproduction-only",
        "--output-dir",
        str(reproduction_dir),
    ]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "987654"
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
    reproduced = tuple(
        (reproduction_dir / name).read_bytes()
        for name in (
            "final_test_trials_v2.csv",
            "final_test_trials_config_v2.json",
            "final_test_trials_identity_v2.json",
        )
    )
    if canonical_payloads != reproduced:
        raise RuntimeError("trial package differs under another PYTHONHASHSEED")

    for path, payload in zip(
        (CSV_PATH, TRIAL_CONFIG_PATH, TRIAL_IDENTITY_PATH), canonical_payloads
    ):
        atomic_bytes(path, payload)
        if path.read_bytes() != payload:
            raise RuntimeError(f"trial artifact readback mismatch: {path}")

    verified = verify_approved_inputs(REPO_ROOT)
    rows = read_final_manifest(
        REPO_ROOT / APPROVED_INPUTS["final_test_manifest"]["path"]
    )
    validate_final_trials(read_trials_csv(CSV_PATH), rows)
    cache_config = build_cache_config(verified)
    assert_no_absolute_paths(cache_config)
    atomic_json(CACHE_CONFIG_PATH, cache_config)
    lock = build_lock_config(
        verified,
        trial_csv_hash=sha256_file(CSV_PATH),
        trial_config_hash=sha256_file(TRIAL_CONFIG_PATH),
        trial_identity_hash=sha256_file(TRIAL_IDENTITY_PATH),
        fbank_config_hash=sha256_file(CACHE_CONFIG_PATH),
    )
    assert_no_absolute_paths(lock)
    atomic_json(LOCK_PATH, lock)
    identity = build_lock_identity(sha256_file(LOCK_PATH), lock)
    assert_no_absolute_paths(identity)
    atomic_json(LOCK_IDENTITY_PATH, identity)
    validate_lock(REPO_ROOT)
    if read_json(LOCK_IDENTITY_PATH) != identity:
        raise RuntimeError("lock identity readback mismatch")
    initialize_state(
        REPO_ROOT, sha256_file(LOCK_IDENTITY_PATH), identity["evaluation_id"]
    )
    result = {
        "result": "PASS",
        "phase": "protocol_locked",
        "trial_generation_duration_seconds": time.perf_counter() - started,
        "trial_csv_sha256": sha256_file(CSV_PATH),
        "trial_config_sha256": sha256_file(TRIAL_CONFIG_PATH),
        "trial_identity_sha256": sha256_file(TRIAL_IDENTITY_PATH),
        "fbank_config_sha256": sha256_file(CACHE_CONFIG_PATH),
        "lock_config_sha256": sha256_file(LOCK_PATH),
        "lock_identity_sha256": sha256_file(LOCK_IDENTITY_PATH),
        "different_pythonhashseed_byte_identical": True,
    }
    atomic_json(
        REPO_ROOT / "outputs/final_evaluation_v2/protocol_lock_result_v2.json",
        result,
    )
    print(canonical_json(result), end="")


if __name__ == "__main__":
    main()
