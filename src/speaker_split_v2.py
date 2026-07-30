"""Deterministic VieSpeaker2.0 speaker-disjoint split package construction."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.full_manifest_v2 import (
    MANIFEST_FIELDS,
    ManifestRow,
    file_sha256,
    read_manifest,
    validate_portable_path,
)

SPLIT_VERSION = "v2"
SPLIT_SCHEMA_VERSION = 1
IDENTITY_SCHEMA_VERSION = 1
SPLIT_SEED = 20260729
SELECTION_ALGORITHM = "sha256_bucket_largest_remainder_exact_balance_v1"
CREATION_TOOL = "scripts/create_speaker_split_package_v2.py"
CREATION_TOOL_VERSION = "speaker_split_package_v2_schema_1"

FULL_MANIFEST_RELATIVE_PATH = "manifests/v2/full_manifest_v2.csv"
FULL_MANIFEST_IDENTITY_RELATIVE_PATH = (
    "manifests/v2/full_manifest_v2_identity.json"
)
APPROVED_FULL_MANIFEST_SHA256 = (
    "26a0157abce3bb00ce5f0ca16f9b964e180f72484e53f9d2602577f5ce4acf8f"
)
APPROVED_FULL_MANIFEST_IDENTITY_SHA256 = (
    "f7b6c1cbc95b8a0d20b596b4841de7ebc26e69d6bd7c130801937dab681c544e"
)

SPLIT_CSV_RELATIVE_PATH = "splits/v2/speaker_split_v2.csv"
SPLIT_IDENTITY_RELATIVE_PATH = "splits/v2/speaker_split_v2_identity.json"
SPLIT_POLICY_RELATIVE_PATH = "splits/v2/split_policy_v2.json"
PORTABLE_ROOT = "manifests/portable_v2"
TRAIN_MANIFEST_RELATIVE_PATH = f"{PORTABLE_ROOT}/train_manifest_v2.csv"
VALIDATION_MANIFEST_RELATIVE_PATH = (
    f"{PORTABLE_ROOT}/validation_manifest_v2.csv"
)
TEST_MANIFEST_RELATIVE_PATH = f"{PORTABLE_ROOT}/test_manifest_v2.csv"
LABEL_MAPPING_RELATIVE_PATH = f"{PORTABLE_ROOT}/speaker_to_label_v2.json"
PORTABLE_IDENTITY_RELATIVE_PATH = (
    f"{PORTABLE_ROOT}/portable_manifests_v2_identity.json"
)
REPORT_MD_RELATIVE_PATH = "reports/speaker_split_v2_summary.md"
REPORT_JSON_RELATIVE_PATH = "reports/speaker_split_v2_summary.json"

SPLIT_FIELDS = (
    "speaker_id",
    "final_split",
    "utterance_count",
    "duration_seconds",
    "provenance_composition",
    "duplicate_file_count",
    "utterance_bucket",
    "eligibility_status",
    "eligibility_reason",
    "selection_hash",
    "split_seed",
    "split_version",
)
PORTABLE_FIELDS = (
    "relative_audio_path",
    "speaker_id",
    "speaker_label",
    "final_split",
    "filename_group",
    "duplicate_group",
    "manifest_version",
)
FINAL_SPLITS = ("train", "validation", "test", "excluded")
OUTPUT_SPLITS = ("train", "validation", "test")
PROVENANCE_ORDER = ("train", "train_small", "part", "test")
EVALUATION_BUCKETS = (
    ("20-49", 20, 49),
    ("50-99", 50, 99),
    ("100-199", 100, 199),
    ("200-499", 200, 499),
    ("500+", 500, None),
)
REPRODUCIBILITY_PATHS = (
    SPLIT_CSV_RELATIVE_PATH,
    SPLIT_IDENTITY_RELATIVE_PATH,
    SPLIT_POLICY_RELATIVE_PATH,
    TRAIN_MANIFEST_RELATIVE_PATH,
    VALIDATION_MANIFEST_RELATIVE_PATH,
    TEST_MANIFEST_RELATIVE_PATH,
    LABEL_MAPPING_RELATIVE_PATH,
    PORTABLE_IDENTITY_RELATIVE_PATH,
)


@dataclass(frozen=True)
class SplitPolicy:
    seed: int = SPLIT_SEED
    validation_speakers: int = 100
    test_speakers: int = 100
    minimum_train_utterances: int = 2
    minimum_evaluation_utterances: int = 20
    buckets: tuple[tuple[str, int, int | None], ...] = EVALUATION_BUCKETS


@dataclass(frozen=True)
class PackageExpectations:
    total_rows: int
    total_speakers: int
    train_speakers: int
    validation_speakers: int
    test_speakers: int
    excluded_speakers: int
    singleton_speakers: int
    duplicate_groups: int
    duplicate_files: int
    evaluation_eligible_speakers: int


PRODUCTION_EXPECTATIONS = PackageExpectations(
    total_rows=125_847,
    total_speakers=1_675,
    train_speakers=1_347,
    validation_speakers=100,
    test_speakers=100,
    excluded_speakers=128,
    singleton_speakers=128,
    duplicate_groups=4,
    duplicate_files=8,
    evaluation_eligible_speakers=800,
)


@dataclass(frozen=True)
class SpeakerStats:
    speaker_id: str
    utterance_count: int
    duration_seconds: Decimal
    provenance_counts: Mapping[str, int]
    duplicate_file_count: int
    selection_hash: str


@dataclass(frozen=True)
class Assignment:
    speaker: SpeakerStats
    final_split: str
    utterance_bucket: str
    eligibility_status: str
    eligibility_reason: str


@dataclass(frozen=True)
class AuthoritativeInput:
    rows: tuple[ManifestRow, ...]
    speakers: tuple[SpeakerStats, ...]
    manifest_sha256: str
    identity_sha256: str
    manifest_identity: Mapping[str, Any]


@dataclass(frozen=True)
class PackageData:
    assignments: tuple[Assignment, ...]
    portable_rows: Mapping[str, tuple[Mapping[str, Any], ...]]
    labels: Mapping[str, int]
    bucket_capacities: Mapping[str, int]
    bucket_quotas: Mapping[str, int]
    validation_utterances: int
    test_utterances: int
    validation_duplicate_files: int
    test_duplicate_files: int


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def render_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def render_csv(fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=fields,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def decimal_text(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.000001')):.6f}"


def numeric_speaker_key(speaker_id: str) -> tuple[int, str]:
    if not speaker_id.isdigit():
        raise ValueError(f"speaker ID is not numeric: {speaker_id!r}")
    return int(speaker_id), speaker_id


def selection_hash(seed: int, speaker_id: str) -> str:
    numeric_speaker_key(speaker_id)
    return hashlib.sha256(f"{seed}:{speaker_id}".encode("utf-8")).hexdigest()


def evaluation_bucket(
    count: int,
    buckets: Sequence[tuple[str, int, int | None]] = EVALUATION_BUCKETS,
) -> str:
    for name, lower, upper in buckets:
        if count >= lower and (upper is None or count <= upper):
            return name
    raise ValueError(f"utterance count is not evaluation eligible: {count}")


def assignment_bucket(
    count: int,
    buckets: Sequence[tuple[str, int, int | None]] = EVALUATION_BUCKETS,
) -> str:
    if count == 1:
        return "1"
    if count < buckets[0][1]:
        return f"2-{buckets[0][1] - 1}"
    return evaluation_bucket(count, buckets)


def largest_remainder_allocation(
    capacities: Mapping[str, int],
    target: int,
    ordered_names: Sequence[str],
) -> dict[str, int]:
    if set(capacities) != set(ordered_names):
        raise ValueError("bucket capacities do not match configured buckets")
    total = sum(capacities.values())
    if target < 0 or target > total:
        raise ValueError("allocation target exceeds bucket capacity")
    raw = {
        name: Fraction(target * capacities[name], total)
        for name in ordered_names
    }
    allocation = {name: int(raw[name]) for name in ordered_names}
    remaining = target - sum(allocation.values())
    order_index = {name: index for index, name in enumerate(ordered_names)}
    remainder_order = sorted(
        ordered_names,
        key=lambda name: (
            -(raw[name] - int(raw[name])),
            order_index[name],
        ),
    )
    for name in remainder_order:
        if not remaining:
            break
        if allocation[name] < capacities[name]:
            allocation[name] += 1
            remaining -= 1
    if remaining or sum(allocation.values()) != target:
        raise ValueError("largest-remainder allocation could not meet target")
    if any(allocation[name] > capacities[name] for name in ordered_names):
        raise ValueError("largest-remainder allocation exceeds capacity")
    return allocation


def _bucket_subset_states(
    speakers: Sequence[SpeakerStats],
) -> dict[tuple[int, int, int], int]:
    """Return best stable mask for each (chosen, utterances, duplicate files)."""
    ordered = sorted(
        speakers,
        key=lambda speaker: (
            speaker.selection_hash,
            numeric_speaker_key(speaker.speaker_id),
        ),
    )
    lower = len(ordered) // 2
    upper = (len(ordered) + 1) // 2
    states: dict[tuple[int, int, int], int] = {(0, 0, 0): 0}
    for index, speaker in enumerate(ordered):
        bit = 1 << (len(ordered) - index - 1)
        updated = dict(states)
        for (chosen, utterances, duplicates), mask in states.items():
            if chosen >= upper:
                continue
            key = (
                chosen + 1,
                utterances + speaker.utterance_count,
                duplicates + speaker.duplicate_file_count,
            )
            candidate = mask | bit
            if candidate > updated.get(key, -1):
                updated[key] = candidate
        states = updated
    return {
        key: mask
        for key, mask in states.items()
        if key[0] in {lower, upper}
    }


def _suffix_reachability(
    bucket_states: Sequence[Mapping[tuple[int, int, int], int]],
) -> list[dict[tuple[int, int], int]]:
    """Bitset of reachable utterance totals keyed by (speaker count, dup files)."""
    suffix: list[dict[tuple[int, int], int]] = [
        {} for _ in range(len(bucket_states) + 1)
    ]
    suffix[-1] = {(0, 0): 1}
    for index in range(len(bucket_states) - 1, -1, -1):
        reachable: dict[tuple[int, int], int] = {}
        for (chosen, utterances, duplicates) in bucket_states[index]:
            for (rest_count, rest_duplicates), rest_bits in suffix[index + 1].items():
                key = (
                    chosen + rest_count,
                    duplicates + rest_duplicates,
                )
                reachable[key] = reachable.get(key, 0) | (
                    rest_bits << utterances
                )
        suffix[index] = reachable
    return suffix


def _nearest_reachable_sums(bits: int, total: int) -> tuple[int, ...]:
    if not bits:
        return ()
    half_floor = total // 2
    lower_bits = bits & ((1 << (half_floor + 1)) - 1)
    candidates: set[int] = set()
    if lower_bits:
        candidates.add(lower_bits.bit_length() - 1)
    upper_bits = bits >> ((total + 1) // 2)
    if upper_bits:
        candidates.add(((total + 1) // 2) + ((upper_bits & -upper_bits).bit_length() - 1))
    return tuple(sorted(candidates))


def balance_selected_speakers(
    selected_by_bucket: Mapping[str, Sequence[SpeakerStats]],
    policy: SplitPolicy,
) -> tuple[set[str], set[str], dict[str, Any]]:
    bucket_names = [name for name, _, _ in policy.buckets]
    if set(selected_by_bucket) != set(bucket_names):
        raise ValueError("selected evaluation buckets are incomplete")
    selected = [
        speaker
        for name in bucket_names
        for speaker in selected_by_bucket[name]
    ]
    expected_total = policy.validation_speakers + policy.test_speakers
    if len(selected) != expected_total:
        raise ValueError("selected evaluation speaker total is invalid")
    bucket_states = [
        _bucket_subset_states(selected_by_bucket[name]) for name in bucket_names
    ]
    suffix = _suffix_reachability(bucket_states)
    total_utterances = sum(speaker.utterance_count for speaker in selected)
    total_duplicates = sum(speaker.duplicate_file_count for speaker in selected)
    objective_candidates: list[tuple[int, int, int, int]] = []
    for (count, duplicate_files), bits in suffix[0].items():
        if count != policy.validation_speakers:
            continue
        for utterances in _nearest_reachable_sums(bits, total_utterances):
            objective_candidates.append(
                (
                    abs(total_utterances - 2 * utterances),
                    abs(total_duplicates - 2 * duplicate_files),
                    utterances,
                    duplicate_files,
                )
            )
    if not objective_candidates:
        raise ValueError("evaluation balance constraints are infeasible")
    best_objective = min(
        (utterance_difference, duplicate_difference)
        for utterance_difference, duplicate_difference, _, _ in objective_candidates
    )

    def reconstruct(
        target_utterances: int,
        target_duplicates: int,
    ) -> tuple[set[str], dict[str, int], tuple[int, ...]]:
        remaining_count = policy.validation_speakers
        remaining_utterances = target_utterances
        remaining_duplicates = target_duplicates
        reconstructed_ids: set[str] = set()
        reconstructed_bucket_counts: dict[str, int] = {}
        stable_masks: list[int] = []
        for index, name in enumerate(bucket_names):
            speakers = sorted(
                selected_by_bucket[name],
                key=lambda speaker: (
                    speaker.selection_hash,
                    numeric_speaker_key(speaker.speaker_id),
                ),
            )
            candidates = sorted(
                bucket_states[index].items(),
                key=lambda item: item[1],
                reverse=True,
            )
            chosen_state: tuple[tuple[int, int, int], int] | None = None
            for (chosen, utterances, duplicates), mask in candidates:
                rest_key = (
                    remaining_count - chosen,
                    remaining_duplicates - duplicates,
                )
                rest_sum = remaining_utterances - utterances
                if rest_sum < 0:
                    continue
                bits = suffix[index + 1].get(rest_key, 0)
                if (bits >> rest_sum) & 1:
                    chosen_state = ((chosen, utterances, duplicates), mask)
                    break
            if chosen_state is None:
                raise AssertionError(
                    "could not reconstruct optimal evaluation balance"
                )
            (chosen, utterances, duplicates), mask = chosen_state
            for speaker_index, speaker in enumerate(speakers):
                bit = 1 << (len(speakers) - speaker_index - 1)
                if mask & bit:
                    reconstructed_ids.add(speaker.speaker_id)
            reconstructed_bucket_counts[name] = chosen
            stable_masks.append(mask)
            remaining_count -= chosen
            remaining_utterances -= utterances
            remaining_duplicates -= duplicates
        if (
            remaining_count,
            remaining_utterances,
            remaining_duplicates,
        ) != (0, 0, 0):
            raise AssertionError(
                "evaluation balance reconstruction did not reconcile"
            )
        return (
            reconstructed_ids,
            reconstructed_bucket_counts,
            tuple(stable_masks),
        )

    stable_candidates = []
    for (
        utterance_difference,
        duplicate_difference,
        utterances,
        duplicates,
    ) in objective_candidates:
        if (utterance_difference, duplicate_difference) != best_objective:
            continue
        ids, bucket_counts, masks = reconstruct(utterances, duplicates)
        stable_candidates.append(
            (masks, utterances, duplicates, ids, bucket_counts)
        )
    (
        _,
        target_utterances,
        target_duplicates,
        validation_ids,
        per_bucket_validation,
    ) = max(stable_candidates, key=lambda candidate: candidate[0])
    selected_ids = {speaker.speaker_id for speaker in selected}
    test_ids = selected_ids - validation_ids
    if len(validation_ids) != policy.validation_speakers:
        raise AssertionError("validation target was not met")
    if len(test_ids) != policy.test_speakers:
        raise AssertionError("test target was not met")
    for name in bucket_names:
        validation_count = sum(
            speaker.speaker_id in validation_ids
            for speaker in selected_by_bucket[name]
        )
        test_count = len(selected_by_bucket[name]) - validation_count
        if abs(validation_count - test_count) > 1:
            raise AssertionError("per-bucket validation/test imbalance exceeds one")
    diagnostics = {
        "validation_utterances": target_utterances,
        "test_utterances": total_utterances - target_utterances,
        "utterance_absolute_difference": abs(
            total_utterances - 2 * target_utterances
        ),
        "validation_duplicate_files": target_duplicates,
        "test_duplicate_files": total_duplicates - target_duplicates,
        "duplicate_file_absolute_difference": abs(
            total_duplicates - 2 * target_duplicates
        ),
        "validation_speakers_by_bucket": per_bucket_validation,
        "test_speakers_by_bucket": {
            name: len(selected_by_bucket[name]) - per_bucket_validation[name]
            for name in bucket_names
        },
    }
    return validation_ids, test_ids, diagnostics


def aggregate_speakers(
    rows: Sequence[ManifestRow],
    seed: int,
) -> tuple[SpeakerStats, ...]:
    grouped: dict[str, list[ManifestRow]] = defaultdict(list)
    for row in rows:
        validate_portable_path(row.audio_path)
        pure_parts = row.audio_path.split("/")
        if len(pure_parts) != 2:
            raise ValueError(f"full-manifest path depth is invalid: {row.audio_path}")
        if pure_parts[0] != row.speaker_id:
            raise ValueError("full-manifest speaker differs from direct parent")
        if pure_parts[-1] != row.filename:
            raise ValueError("full-manifest filename differs from path basename")
        grouped[row.speaker_id].append(row)
    speakers: list[SpeakerStats] = []
    for speaker_id in sorted(grouped, key=numeric_speaker_key):
        speaker_rows = grouped[speaker_id]
        provenance = Counter(row.provenance for row in speaker_rows)
        if set(provenance) - set(PROVENANCE_ORDER):
            raise ValueError("unsupported provenance in full manifest")
        speakers.append(
            SpeakerStats(
                speaker_id=speaker_id,
                utterance_count=len(speaker_rows),
                duration_seconds=sum(
                    (Decimal(row.duration_seconds) for row in speaker_rows),
                    Decimal(0),
                ),
                provenance_counts={
                    name: provenance[name] for name in PROVENANCE_ORDER
                },
                duplicate_file_count=sum(
                    bool(row.duplicate_group) for row in speaker_rows
                ),
                selection_hash=selection_hash(seed, speaker_id),
            )
        )
    return tuple(speakers)


def validate_authoritative_rows(
    rows: Sequence[ManifestRow],
    identity: Mapping[str, Any],
) -> None:
    if not rows:
        raise ValueError("full manifest is empty")
    paths = [row.audio_path for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("full manifest contains duplicate audio paths")
    expected_order = sorted(
        rows,
        key=lambda row: (numeric_speaker_key(row.speaker_id), row.audio_path),
    )
    if list(rows) != expected_order:
        raise ValueError("full manifest order is invalid")
    speakers = aggregate_speakers(rows, SPLIT_SEED)
    provenance = Counter(row.provenance for row in rows)
    duplicate_groups: dict[str, set[str]] = defaultdict(set)
    duplicate_files = 0
    for row in rows:
        if row.duplicate_group:
            duplicate_files += 1
            duplicate_groups[row.duplicate_group].add(row.speaker_id)
    values = {
        "row_count": len(rows),
        "unique_audio_path_count": len(set(paths)),
        "speaker_count": len(speakers),
        "total_bytes": sum(row.file_size_bytes for row in rows),
        "total_duration_seconds": int(
            sum(
                (Decimal(row.duration_seconds) for row in rows),
                Decimal(0),
            )
        ),
        "readable_wav_count": sum(row.wav_status == "readable" for row in rows),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_file_count": duplicate_files,
        "cross_speaker_duplicate_group_count": sum(
            len(owners) > 1 for owners in duplicate_groups.values()
        ),
        "singleton_speaker_count": sum(
            speaker.utterance_count == 1 for speaker in speakers
        ),
        "provenance_counts": {
            name: provenance[name] for name in PROVENANCE_ORDER
        },
    }
    for key, value in values.items():
        if identity.get(key) != value:
            raise ValueError(f"full-manifest identity mismatch: {key}")
    if any(len(owners) != 1 for owners in duplicate_groups.values()):
        raise ValueError("a duplicate group crosses speakers")


def load_authoritative_input(
    manifest_path: Path,
    identity_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_identity_sha256: str,
    seed: int,
) -> AuthoritativeInput:
    actual_manifest_hash = file_sha256(manifest_path)
    actual_identity_hash = file_sha256(identity_path)
    if actual_manifest_hash != expected_manifest_sha256:
        raise ValueError("authoritative full manifest SHA-256 mismatch")
    if actual_identity_hash != expected_identity_sha256:
        raise ValueError("authoritative full-manifest identity SHA-256 mismatch")
    with identity_path.open("r", encoding="utf-8") as stream:
        identity = json.load(stream)
    if not isinstance(identity, Mapping):
        raise ValueError("full-manifest identity must be a JSON object")
    if (
        identity.get("manifest_relative_path") != FULL_MANIFEST_RELATIVE_PATH
        or identity.get("manifest_sha256") != expected_manifest_sha256
        or identity.get("path_separator") != "/"
        or identity.get("manifest_version") != "v2"
    ):
        raise ValueError("full-manifest identity binding is invalid")
    rows = tuple(read_manifest(manifest_path))
    validate_authoritative_rows(rows, identity)
    speakers = aggregate_speakers(rows, seed)
    return AuthoritativeInput(
        rows=rows,
        speakers=speakers,
        manifest_sha256=actual_manifest_hash,
        identity_sha256=actual_identity_hash,
        manifest_identity=identity,
    )


def create_assignments(
    speakers: Sequence[SpeakerStats],
    policy: SplitPolicy,
    expectations: PackageExpectations | None = None,
) -> tuple[tuple[Assignment, ...], dict[str, int], dict[str, int], dict[str, Any]]:
    if policy.minimum_train_utterances != 2:
        raise ValueError("v2 policy requires minimum train utterances of 2")
    if policy.minimum_evaluation_utterances != 20:
        raise ValueError("v2 policy requires minimum evaluation utterances of 20")
    if len({speaker.speaker_id for speaker in speakers}) != len(speakers):
        raise ValueError("speaker statistics contain duplicate speakers")
    bucket_names = [name for name, _, _ in policy.buckets]
    eligible_by_bucket: dict[str, list[SpeakerStats]] = {
        name: [] for name in bucket_names
    }
    for speaker in speakers:
        expected_hash = selection_hash(policy.seed, speaker.speaker_id)
        if speaker.selection_hash != expected_hash:
            raise ValueError("speaker selection hash is inconsistent")
        if speaker.utterance_count >= policy.minimum_evaluation_utterances:
            eligible_by_bucket[
                evaluation_bucket(speaker.utterance_count, policy.buckets)
            ].append(speaker)
    capacities = {
        name: len(eligible_by_bucket[name]) for name in bucket_names
    }
    evaluation_target = policy.validation_speakers + policy.test_speakers
    if sum(capacities.values()) < evaluation_target:
        raise ValueError("insufficient evaluation-eligible speakers")
    quotas = largest_remainder_allocation(
        capacities, evaluation_target, bucket_names
    )
    selected_by_bucket: dict[str, tuple[SpeakerStats, ...]] = {}
    for name in bucket_names:
        ordered = sorted(
            eligible_by_bucket[name],
            key=lambda speaker: (
                speaker.selection_hash,
                numeric_speaker_key(speaker.speaker_id),
            ),
        )
        selected_by_bucket[name] = tuple(ordered[: quotas[name]])
    validation_ids, test_ids, balance = balance_selected_speakers(
        selected_by_bucket, policy
    )
    assignments: list[Assignment] = []
    for speaker in sorted(speakers, key=lambda item: numeric_speaker_key(item.speaker_id)):
        if speaker.utterance_count == 1:
            split = "excluded"
            status = "ineligible"
            reason = "singleton_speaker"
        elif speaker.speaker_id in validation_ids:
            split = "validation"
            status = "eligible_evaluation"
            reason = "at_least_20_utterances"
        elif speaker.speaker_id in test_ids:
            split = "test"
            status = "eligible_evaluation"
            reason = "at_least_20_utterances"
        elif speaker.utterance_count >= policy.minimum_train_utterances:
            split = "train"
            status = "eligible_train"
            reason = "at_least_2_utterances"
        else:
            raise ValueError("speaker cannot be assigned under approved policy")
        assignments.append(
            Assignment(
                speaker=speaker,
                final_split=split,
                utterance_bucket=assignment_bucket(
                    speaker.utterance_count, policy.buckets
                ),
                eligibility_status=status,
                eligibility_reason=reason,
            )
        )
    validate_assignments(assignments, policy, expectations)
    return tuple(assignments), capacities, quotas, balance


def validate_assignments(
    assignments: Sequence[Assignment],
    policy: SplitPolicy,
    expectations: PackageExpectations | None,
) -> None:
    if [assignment.speaker.speaker_id for assignment in assignments] != sorted(
        (assignment.speaker.speaker_id for assignment in assignments),
        key=numeric_speaker_key,
    ):
        raise ValueError("speaker assignments are not numerically ordered")
    by_split = {
        split: {
            assignment.speaker.speaker_id
            for assignment in assignments
            if assignment.final_split == split
        }
        for split in FINAL_SPLITS
    }
    if sum(len(ids) for ids in by_split.values()) != len(assignments):
        raise ValueError("every speaker must have exactly one assignment")
    for index, left in enumerate(FINAL_SPLITS):
        for right in FINAL_SPLITS[index + 1 :]:
            if by_split[left] & by_split[right]:
                raise ValueError(f"speaker leakage between {left} and {right}")
    if len(by_split["validation"]) != policy.validation_speakers:
        raise ValueError("validation speaker target was not met")
    if len(by_split["test"]) != policy.test_speakers:
        raise ValueError("test speaker target was not met")
    for assignment in assignments:
        count = assignment.speaker.utterance_count
        if assignment.final_split == "excluded" and (
            count != 1
            or assignment.eligibility_status != "ineligible"
            or assignment.eligibility_reason != "singleton_speaker"
        ):
            raise ValueError("excluded speaker is not an approved singleton")
        if assignment.final_split == "train" and count < policy.minimum_train_utterances:
            raise ValueError("train speaker is below the approved minimum")
        if assignment.final_split in {"validation", "test"} and count < policy.minimum_evaluation_utterances:
            raise ValueError("evaluation speaker is below the approved minimum")
    if expectations is not None:
        actual_speakers = {split: len(by_split[split]) for split in FINAL_SPLITS}
        expected_speakers = {
            "train": expectations.train_speakers,
            "validation": expectations.validation_speakers,
            "test": expectations.test_speakers,
            "excluded": expectations.excluded_speakers,
        }
        if len(assignments) != expectations.total_speakers:
            raise ValueError("total speaker count differs from approval")
        if actual_speakers != expected_speakers:
            raise ValueError("split speaker counts differ from approval")
        if sum(
            assignment.speaker.utterance_count == 1
            for assignment in assignments
        ) != expectations.singleton_speakers:
            raise ValueError("singleton speaker count differs from approval")
        if sum(
            assignment.speaker.utterance_count
            >= policy.minimum_evaluation_utterances
            for assignment in assignments
        ) != expectations.evaluation_eligible_speakers:
            raise ValueError("evaluation-eligible speaker count differs from approval")


def split_csv_rows(assignments: Sequence[Assignment], policy: SplitPolicy) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for assignment in assignments:
        speaker = assignment.speaker
        composition = "|".join(
            f"{name}={speaker.provenance_counts.get(name, 0)}"
            for name in PROVENANCE_ORDER
        )
        rows.append(
            {
                "speaker_id": speaker.speaker_id,
                "final_split": assignment.final_split,
                "utterance_count": speaker.utterance_count,
                "duration_seconds": decimal_text(speaker.duration_seconds),
                "provenance_composition": composition,
                "duplicate_file_count": speaker.duplicate_file_count,
                "utterance_bucket": assignment.utterance_bucket,
                "eligibility_status": assignment.eligibility_status,
                "eligibility_reason": assignment.eligibility_reason,
                "selection_hash": speaker.selection_hash,
                "split_seed": policy.seed,
                "split_version": SPLIT_VERSION,
            }
        )
    return rows


def build_portable_rows(
    full_rows: Sequence[ManifestRow],
    assignments: Sequence[Assignment],
) -> tuple[dict[str, tuple[Mapping[str, Any], ...]], dict[str, int]]:
    assignment_by_speaker = {
        assignment.speaker.speaker_id: assignment.final_split
        for assignment in assignments
    }
    train_speakers = sorted(
        (
            speaker_id
            for speaker_id, split in assignment_by_speaker.items()
            if split == "train"
        ),
        key=numeric_speaker_key,
    )
    labels = {
        speaker_id: label for label, speaker_id in enumerate(train_speakers)
    }
    rows: dict[str, list[Mapping[str, Any]]] = {
        split: [] for split in OUTPUT_SPLITS
    }
    seen_paths: set[str] = set()
    for source in full_rows:
        split = assignment_by_speaker.get(source.speaker_id)
        if split is None:
            raise ValueError("full-manifest speaker has no assignment")
        if source.audio_path in seen_paths:
            raise ValueError("full-manifest path is duplicated")
        seen_paths.add(source.audio_path)
        if split == "excluded":
            continue
        rows[split].append(
            {
                "relative_audio_path": source.audio_path,
                "speaker_id": source.speaker_id,
                "speaker_label": (
                    labels[source.speaker_id] if split == "train" else -1
                ),
                "final_split": split,
                "filename_group": source.provenance,
                "duplicate_group": source.duplicate_group,
                "manifest_version": SPLIT_VERSION,
            }
        )
    rows["train"].sort(
        key=lambda row: (
            int(row["speaker_label"]),
            str(row["relative_audio_path"]),
        )
    )
    for split in ("validation", "test"):
        rows[split].sort(
            key=lambda row: (
                numeric_speaker_key(str(row["speaker_id"])),
                str(row["relative_audio_path"]),
            )
        )
    result = {split: tuple(rows[split]) for split in OUTPUT_SPLITS}
    validate_portable_rows(full_rows, assignments, result, labels)
    return result, labels


def validate_portable_rows(
    full_rows: Sequence[ManifestRow],
    assignments: Sequence[Assignment],
    portable_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    labels: Mapping[str, int],
) -> None:
    assignment_by_speaker = {
        assignment.speaker.speaker_id: assignment.final_split
        for assignment in assignments
    }
    all_paths: list[str] = []
    speaker_sets: dict[str, set[str]] = {}
    for split in OUTPUT_SPLITS:
        split_rows = portable_rows[split]
        speaker_sets[split] = {
            str(row["speaker_id"]) for row in split_rows
        }
        for row in split_rows:
            path = str(row["relative_audio_path"])
            validate_portable_path(path)
            all_paths.append(path)
            speaker_id = str(row["speaker_id"])
            if assignment_by_speaker.get(speaker_id) != split:
                raise ValueError("portable row disagrees with speaker assignment")
            if row["final_split"] != split:
                raise ValueError("portable row final_split is invalid")
            if row["manifest_version"] != SPLIT_VERSION:
                raise ValueError("portable row manifest_version is invalid")
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("portable manifest paths are not globally unique")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if speaker_sets[left] & speaker_sets[right]:
            raise ValueError("portable manifest speakers are not disjoint")
    if set(labels) != speaker_sets["train"]:
        raise ValueError("train label mapping does not match train speakers")
    if list(labels.values()) != list(range(len(labels))):
        raise ValueError("train labels are not contiguous")
    if any(
        int(row["speaker_label"]) != labels[str(row["speaker_id"])]
        for row in portable_rows["train"]
    ):
        raise ValueError("train portable label is invalid")
    if any(
        int(row["speaker_label"]) != -1
        for split in ("validation", "test")
        for row in portable_rows[split]
    ):
        raise ValueError("validation/test labels must be -1")
    excluded_paths = {
        row.audio_path
        for row in full_rows
        if assignment_by_speaker[row.speaker_id] == "excluded"
    }
    if set(all_paths) | excluded_paths != {
        row.audio_path for row in full_rows
    }:
        raise ValueError("portable and excluded rows do not reconcile to full manifest")
    if set(all_paths) & excluded_paths:
        raise ValueError("excluded rows appear in portable manifests")


def duplicate_counts_by_split(
    full_rows: Sequence[ManifestRow],
    assignments: Sequence[Assignment],
) -> dict[str, dict[str, int]]:
    assignment_by_speaker = {
        assignment.speaker.speaker_id: assignment.final_split
        for assignment in assignments
    }
    group_splits: dict[str, set[str]] = defaultdict(set)
    group_speakers: dict[str, set[str]] = defaultdict(set)
    file_counts = Counter()
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    for row in full_rows:
        if not row.duplicate_group:
            continue
        split = assignment_by_speaker[row.speaker_id]
        group_splits[row.duplicate_group].add(split)
        group_speakers[row.duplicate_group].add(row.speaker_id)
        file_counts[split] += 1
        groups_by_split[split].add(row.duplicate_group)
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("duplicate group crosses final splits")
    if any(len(speakers) != 1 for speakers in group_speakers.values()):
        raise ValueError("duplicate group crosses speakers")
    return {
        split: {
            "duplicate_group_count": len(groups_by_split[split]),
            "duplicate_file_count": file_counts[split],
        }
        for split in FINAL_SPLITS
    }


def package_statistics(
    full_rows: Sequence[ManifestRow],
    assignments: Sequence[Assignment],
    portable_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    speaker_counts = Counter(
        assignment.final_split for assignment in assignments
    )
    row_counts = Counter(
        {
            split: len(portable_rows.get(split, ()))
            for split in OUTPUT_SPLITS
        }
    )
    duration_counts = {
        split: sum(
            (
                assignment.speaker.duration_seconds
                for assignment in assignments
                if assignment.final_split == split
            ),
            Decimal(0),
        )
        for split in FINAL_SPLITS
    }
    row_counts["excluded"] = sum(
        assignment.speaker.utterance_count
        for assignment in assignments
        if assignment.final_split == "excluded"
    )
    if sum(row_counts.values()) != len(full_rows):
        raise ValueError("split row counts do not reconcile")
    return {
        "speaker_counts": {
            split: speaker_counts[split] for split in FINAL_SPLITS
        },
        "row_counts": {split: row_counts[split] for split in FINAL_SPLITS},
        "duration_seconds": {
            split: decimal_text(duration_counts[split])
            for split in FINAL_SPLITS
        },
        "duplicate_counts_by_split": duplicate_counts_by_split(
            full_rows, assignments
        ),
    }


def build_policy_document(
    authoritative: AuthoritativeInput,
    policy: SplitPolicy,
) -> dict[str, Any]:
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "split_version": SPLIT_VERSION,
        "split_policy_version": "final_speaker_disjoint_v2_policy_1",
        "full_manifest_relative_path": FULL_MANIFEST_RELATIVE_PATH,
        "full_manifest_sha256": authoritative.manifest_sha256,
        "full_manifest_identity_relative_path": (
            FULL_MANIFEST_IDENTITY_RELATIVE_PATH
        ),
        "full_manifest_identity_sha256": authoritative.identity_sha256,
        "split_seed": policy.seed,
        "selection_algorithm": SELECTION_ALGORITHM,
        "selection_hash_input": "UTF-8 bytes of '<split_seed>:<speaker_id>'",
        "selection_hash_algorithm": "SHA-256",
        "evaluation_selection": (
            "largest-remainder proportional allocation by configured bucket; "
            "lowest SHA-256 order within bucket"
        ),
        "evaluation_assignment_objectives": [
            "exact validation/test speaker targets",
            "per-bucket speaker-count difference no greater than one",
            "minimum absolute utterance-count difference",
            "minimum absolute duplicate-file-count difference",
            "configured bucket order then SHA-256/speaker-ID stable tie break",
        ],
        "utterance_buckets": [
            {"name": name, "minimum": lower, "maximum": upper}
            for name, lower, upper in policy.buckets
        ],
        "minimum_train_utterances": policy.minimum_train_utterances,
        "minimum_evaluation_utterances": (
            policy.minimum_evaluation_utterances
        ),
        "validation_speaker_target": policy.validation_speakers,
        "test_speaker_target": policy.test_speakers,
        "singleton_policy": "excluded_from_portable_manifests_only",
        "provenance_assignment_authority": False,
        "candidate_source_group_assignment_authority": False,
        "duplicate_policy": "preserve_all_files_no_representative_selection",
        "final_test_policy": "quarantined_after_creation_and_validation",
        "path_base_semantics": "relative_to_supplied_dataset_root",
        "path_separator": "/",
        "timestamps_in_identity": False,
    }


def build_identity_document(
    *,
    identity_kind: str,
    authoritative: AuthoritativeInput,
    policy: SplitPolicy,
    policy_sha256: str,
    artifact_hashes: Mapping[str, str],
    statistics: Mapping[str, Any],
    bucket_capacities: Mapping[str, int],
    bucket_quotas: Mapping[str, int],
    balance: Mapping[str, Any],
    train_class_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "identity_kind": identity_kind,
        "split_version": SPLIT_VERSION,
        "split_policy_version": "final_speaker_disjoint_v2_policy_1",
        "split_seed": policy.seed,
        "selection_algorithm": SELECTION_ALGORITHM,
        "utterance_buckets": [
            {"name": name, "minimum": lower, "maximum": upper}
            for name, lower, upper in policy.buckets
        ],
        "minimum_train_utterances": policy.minimum_train_utterances,
        "minimum_evaluation_utterances": (
            policy.minimum_evaluation_utterances
        ),
        "full_manifest_relative_path": FULL_MANIFEST_RELATIVE_PATH,
        "full_manifest_sha256": authoritative.manifest_sha256,
        "full_manifest_identity_relative_path": (
            FULL_MANIFEST_IDENTITY_RELATIVE_PATH
        ),
        "full_manifest_identity_sha256": authoritative.identity_sha256,
        "split_csv_relative_path": SPLIT_CSV_RELATIVE_PATH,
        "split_csv_sha256": artifact_hashes[SPLIT_CSV_RELATIVE_PATH],
        "split_policy_relative_path": SPLIT_POLICY_RELATIVE_PATH,
        "split_policy_sha256": policy_sha256,
        "train_manifest_relative_path": TRAIN_MANIFEST_RELATIVE_PATH,
        "train_manifest_sha256": artifact_hashes[
            TRAIN_MANIFEST_RELATIVE_PATH
        ],
        "validation_manifest_relative_path": (
            VALIDATION_MANIFEST_RELATIVE_PATH
        ),
        "validation_manifest_sha256": artifact_hashes[
            VALIDATION_MANIFEST_RELATIVE_PATH
        ],
        "test_manifest_relative_path": TEST_MANIFEST_RELATIVE_PATH,
        "test_manifest_sha256": artifact_hashes[TEST_MANIFEST_RELATIVE_PATH],
        "speaker_to_label_relative_path": LABEL_MAPPING_RELATIVE_PATH,
        "speaker_to_label_sha256": artifact_hashes[
            LABEL_MAPPING_RELATIVE_PATH
        ],
        "speaker_counts": statistics["speaker_counts"],
        "row_counts": statistics["row_counts"],
        "duration_seconds": statistics["duration_seconds"],
        "duplicate_counts_by_split": statistics[
            "duplicate_counts_by_split"
        ],
        "bucket_capacities": dict(bucket_capacities),
        "selected_evaluation_speakers_by_bucket": dict(bucket_quotas),
        "evaluation_balance": dict(balance),
        "train_class_count": train_class_count,
        "train_label_range": (
            [0, train_class_count - 1] if train_class_count else []
        ),
        "validation_test_label": -1,
        "path_base_semantics": "relative_to_supplied_dataset_root",
        "path_separator": "/",
        "sort_order": {
            "speaker_split": "numeric speaker_id ascending",
            "train_manifest": "speaker_label then relative_audio_path ordinal",
            "validation_manifest": (
                "numeric speaker_id then relative_audio_path ordinal"
            ),
            "test_manifest": (
                "numeric speaker_id then relative_audio_path ordinal"
            ),
            "label_mapping": "numeric speaker_id ascending",
        },
        "final_test_quarantine": True,
        "creation_tool": CREATION_TOOL,
        "creation_tool_version": CREATION_TOOL_VERSION,
        "timestamps_in_identity": False,
    }


def build_summary_json(
    *,
    authoritative: AuthoritativeInput,
    policy: SplitPolicy,
    policy_sha256: str,
    artifact_hashes: Mapping[str, str],
    statistics: Mapping[str, Any],
    bucket_capacities: Mapping[str, int],
    bucket_quotas: Mapping[str, int],
    balance: Mapping[str, Any],
    split_identity_sha256: str,
    portable_identity_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "result": "PASS",
        "scope": "final speaker-disjoint split package v2 only",
        "split_approved": True,
        "full_manifest": {
            "relative_path": FULL_MANIFEST_RELATIVE_PATH,
            "sha256": authoritative.manifest_sha256,
        },
        "full_manifest_identity": {
            "relative_path": FULL_MANIFEST_IDENTITY_RELATIVE_PATH,
            "sha256": authoritative.identity_sha256,
        },
        "policy": {
            "relative_path": SPLIT_POLICY_RELATIVE_PATH,
            "sha256": policy_sha256,
            "split_seed": policy.seed,
            "selection_algorithm": SELECTION_ALGORITHM,
            "minimum_train_utterances": policy.minimum_train_utterances,
            "minimum_evaluation_utterances": (
                policy.minimum_evaluation_utterances
            ),
            "bucket_capacities": dict(bucket_capacities),
            "bucket_quotas": dict(bucket_quotas),
        },
        "statistics": statistics,
        "evaluation_balance": dict(balance),
        "artifact_hashes": dict(artifact_hashes),
        "split_identity_sha256": split_identity_sha256,
        "portable_identity_sha256": portable_identity_sha256,
        "train_class_count": statistics["speaker_counts"]["train"],
        "train_label_range": [
            0,
            statistics["speaker_counts"]["train"] - 1,
        ],
        "validation_test_label": -1,
        "duplicates_preserved": True,
        "singletons_excluded_from_portable_only": True,
        "final_test_quarantined": True,
        "source_wav_accessed": False,
        "source_wav_changed": False,
        "full_manifest_changed": False,
        "reproducibility": {
            "reference_compared": False,
            "all_required_artifacts_byte_identical": None,
        },
        "tests": {
            "py_compile": "PENDING",
            "focused": "PENDING",
            "complete_repository": "PENDING",
            "git_diff_check": "PENDING",
        },
        "deferred": [
            "Fbank and cache v2",
            "validation and test trials",
            "SpeechBrain and ECAPA loading",
            "embeddings, scores, EER, thresholds, and accuracy",
            "sampler benchmarks",
            "training and checkpoints",
            "final-test content access and evaluation",
            "commit and push",
        ],
    }


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    stats = summary["statistics"]
    balance = summary["evaluation_balance"]
    hashes = summary["artifact_hashes"]
    lines = [
        "# VieSpeaker2.0 Final Speaker-Disjoint Split Package v2",
        "",
        "## Goal and exact scope",
        "",
        "Created and validated only the approved speaker-disjoint split, portable",
        "manifests, train label mapping, and immutable identity bindings. No audio",
        "content, feature, model, trial, score, or training stage was accessed.",
        "",
        "## Approved input identities",
        "",
        f"- Full manifest: `{summary['full_manifest']['relative_path']}`",
        f"- Full manifest SHA-256: `{summary['full_manifest']['sha256']}`",
        f"- Full-manifest identity: `{summary['full_manifest_identity']['relative_path']}`",
        f"- Full-manifest identity SHA-256: `{summary['full_manifest_identity']['sha256']}`",
        "",
        "## Deterministic selection algorithm",
        "",
        f"- Seed: `{summary['policy']['split_seed']}`",
        f"- Identifier: `{summary['policy']['selection_algorithm']}`",
        "- Eligible speakers have at least 20 utterances.",
        "- Largest-remainder proportional allocation uses configured bucket order",
        "  to resolve equal remainders.",
        "- Within each bucket speakers are ordered by",
        "  `SHA256(UTF8('<seed>:<speaker_id>'))`, then numeric speaker ID.",
        "- The selected 200 speakers are assigned by exact dynamic programming:",
        "  exact 100/100 counts, per-bucket difference at most one, minimum total",
        "  utterance difference, then minimum duplicate-file difference, then",
        "  configured-bucket/SHA-256 stable tie breaking.",
        "- Provenance and candidate source groups do not influence assignment.",
        "",
        "## Bucket allocation",
        "",
        "| Bucket | Eligible | Selected | Validation | Test |",
        "|---|---:|---:|---:|---:|",
    ]
    for bucket in summary["policy"]["bucket_capacities"]:
        lines.append(
            f"| {bucket} | {summary['policy']['bucket_capacities'][bucket]} | "
            f"{summary['policy']['bucket_quotas'][bucket]} | "
            f"{balance['validation_speakers_by_bucket'][bucket]} | "
            f"{balance['test_speakers_by_bucket'][bucket]} |"
        )
    lines.extend(
        [
            "",
            "## Exact counts",
            "",
            "| Split | Speakers | Rows | Duration seconds | Duplicate groups | Duplicate files |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split in FINAL_SPLITS:
        duplicate = stats["duplicate_counts_by_split"][split]
        lines.append(
            f"| {split} | {stats['speaker_counts'][split]} | "
            f"{stats['row_counts'][split]} | {stats['duration_seconds'][split]} | "
            f"{duplicate['duplicate_group_count']} | "
            f"{duplicate['duplicate_file_count']} |"
        )
    lines.extend(
        [
            "",
            f"All {sum(stats['row_counts'].values()):,} full-manifest rows "
            "reconcile across train, validation,",
            "test, and excluded. All 128 excluded speakers are singletons retained",
            "in the authoritative full manifest and omitted only from portable",
            "train/validation/test manifests.",
            "",
            "## Evaluation balance",
            "",
            f"- Validation/test selected utterances: "
            f"`{balance['validation_utterances']} / {balance['test_utterances']}`",
            f"- Absolute utterance difference: "
            f"`{balance['utterance_absolute_difference']}`",
            f"- Validation/test duplicate files: "
            f"`{balance['validation_duplicate_files']} / "
            f"{balance['test_duplicate_files']}`",
            f"- Absolute duplicate-file difference: "
            f"`{balance['duplicate_file_absolute_difference']}`",
            "",
            "## Label mapping and portable manifests",
            "",
            f"- Train classes: `{summary['train_class_count']}`.",
            f"- Train labels: contiguous `{summary['train_label_range'][0]}.."
            f"{summary['train_label_range'][1]}` in numeric speaker-ID order.",
            "- Validation/test labels: `-1`.",
            "- Paths are dataset-root-relative, use `/`, and contain no absolute",
            "  prefix, drive, backslash, or parent traversal.",
            "- Portable schema preserves descriptive provenance as `filename_group`",
            "  and carries `duplicate_group`; neither controls assignment.",
            "",
            "## Duplicate and final-test policy",
            "",
            "All four exact duplicate groups and all eight files remain present in",
            "their owning speaker's single final split. No representative was",
            "selected. Later validation-trial generation must reject a positive",
            "pair when both files share the same non-empty `duplicate_group`.",
            "",
            "`test_manifest_v2.csv` is immutable and quarantined after this",
            "validation. This task did not open test WAV content, extract test",
            "features, create trials, load test data into a model, or compute",
            "embeddings, scores, EER, thresholds, or accuracy.",
            "",
            "## Artifact hashes",
            "",
        ]
    )
    for path in REPRODUCIBILITY_PATHS:
        lines.append(f"- `{path}`: `{hashes[path]}`")
    lines.extend(
        [
            f"- Split identity file: `{summary['split_identity_sha256']}`",
            f"- Portable identity file: `{summary['portable_identity_sha256']}`",
            "",
            "## Reproducibility and formatting",
            "",
            "- UTF-8 without BOM, LF newlines, fixed CSV columns, fixed six-decimal",
            "  duration serialization, sorted JSON keys, two-space JSON indentation,",
            "  numeric speaker ordering, and no identity timestamps.",
            "- Independent reproduction status is recorded in the JSON report.",
            "",
            "## Validation, preservation, and result",
            "",
            "- Final CSV/JSON schema, hash, row, speaker, label, path, duplicate,",
            "  disjointness, and reconciliation read-back: PASS.",
            "- The builder does not accept or access a dataset root. Source WAV",
            "  content was not opened, and no source WAV could be changed.",
            "- Full manifest pre/post hash: identical.",
            "- Result: **PASS** for the split package only.",
            "",
            "## Explicit exclusions",
            "",
            "No Fbank/cache, trial generation, SpeechBrain/ECAPA load, embedding,",
            "score, EER, threshold, accuracy, sampler benchmark, training,",
            "checkpoint, final-test content access/evaluation, commit, or push.",
            "",
        ]
    )
    return "\n".join(lines)


def build_artifacts(
    authoritative: AuthoritativeInput,
    policy: SplitPolicy,
    expectations: PackageExpectations | None,
) -> tuple[dict[str, bytes], PackageData, dict[str, Any]]:
    assignments, capacities, quotas, balance = create_assignments(
        authoritative.speakers, policy, expectations
    )
    portable_rows, labels = build_portable_rows(
        authoritative.rows, assignments
    )
    if expectations is not None:
        if len(labels) != expectations.train_speakers:
            raise ValueError("train class count differs from approval")
        if len(authoritative.rows) != expectations.total_rows:
            raise ValueError("full row count differs from approval")
        duplicate_groups = {
            row.duplicate_group
            for row in authoritative.rows
            if row.duplicate_group
        }
        duplicate_files = sum(
            bool(row.duplicate_group) for row in authoritative.rows
        )
        if (
            len(duplicate_groups) != expectations.duplicate_groups
            or duplicate_files != expectations.duplicate_files
        ):
            raise ValueError("duplicate totals differ from approval")
    statistics = package_statistics(
        authoritative.rows, assignments, portable_rows
    )
    policy_document = build_policy_document(authoritative, policy)
    policy_bytes = render_json(policy_document)
    artifacts: dict[str, bytes] = {
        SPLIT_CSV_RELATIVE_PATH: render_csv(
            SPLIT_FIELDS, split_csv_rows(assignments, policy)
        ),
        SPLIT_POLICY_RELATIVE_PATH: policy_bytes,
        TRAIN_MANIFEST_RELATIVE_PATH: render_csv(
            PORTABLE_FIELDS, portable_rows["train"]
        ),
        VALIDATION_MANIFEST_RELATIVE_PATH: render_csv(
            PORTABLE_FIELDS, portable_rows["validation"]
        ),
        TEST_MANIFEST_RELATIVE_PATH: render_csv(
            PORTABLE_FIELDS, portable_rows["test"]
        ),
        LABEL_MAPPING_RELATIVE_PATH: render_json(dict(labels)),
    }
    artifact_hashes = {
        path: bytes_sha256(payload) for path, payload in artifacts.items()
    }
    identity_common = {
        "authoritative": authoritative,
        "policy": policy,
        "policy_sha256": bytes_sha256(policy_bytes),
        "artifact_hashes": artifact_hashes,
        "statistics": statistics,
        "bucket_capacities": capacities,
        "bucket_quotas": quotas,
        "balance": balance,
        "train_class_count": len(labels),
    }
    split_identity = build_identity_document(
        identity_kind="speaker_split_v2", **identity_common
    )
    portable_identity = build_identity_document(
        identity_kind="portable_manifests_v2", **identity_common
    )
    artifacts[SPLIT_IDENTITY_RELATIVE_PATH] = render_json(split_identity)
    artifacts[PORTABLE_IDENTITY_RELATIVE_PATH] = render_json(portable_identity)
    artifact_hashes.update(
        {
            SPLIT_IDENTITY_RELATIVE_PATH: bytes_sha256(
                artifacts[SPLIT_IDENTITY_RELATIVE_PATH]
            ),
            PORTABLE_IDENTITY_RELATIVE_PATH: bytes_sha256(
                artifacts[PORTABLE_IDENTITY_RELATIVE_PATH]
            ),
        }
    )
    summary = build_summary_json(
        authoritative=authoritative,
        policy=policy,
        policy_sha256=bytes_sha256(policy_bytes),
        artifact_hashes=artifact_hashes,
        statistics=statistics,
        bucket_capacities=capacities,
        bucket_quotas=quotas,
        balance=balance,
        split_identity_sha256=artifact_hashes[SPLIT_IDENTITY_RELATIVE_PATH],
        portable_identity_sha256=artifact_hashes[
            PORTABLE_IDENTITY_RELATIVE_PATH
        ],
    )
    artifacts[REPORT_JSON_RELATIVE_PATH] = render_json(summary)
    artifacts[REPORT_MD_RELATIVE_PATH] = (
        render_summary_markdown(summary) + "\n"
    ).encode("utf-8")
    data = PackageData(
        assignments=assignments,
        portable_rows=portable_rows,
        labels=labels,
        bucket_capacities=capacities,
        bucket_quotas=quotas,
        validation_utterances=balance["validation_utterances"],
        test_utterances=balance["test_utterances"],
        validation_duplicate_files=balance["validation_duplicate_files"],
        test_duplicate_files=balance["test_duplicate_files"],
    )
    return artifacts, data, summary


def _read_csv(path: Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(expected_fields):
            raise ValueError(f"persisted CSV schema is invalid: {path.name}")
        return list(reader)


def validate_persisted_package(
    root: Path,
    artifacts: Mapping[str, bytes],
    authoritative: AuthoritativeInput,
    data: PackageData,
    expectations: PackageExpectations | None,
) -> None:
    for relative_path, expected_bytes in artifacts.items():
        validate_portable_path(relative_path)
        path = root.joinpath(*relative_path.split("/"))
        if not path.is_file() or path.read_bytes() != expected_bytes:
            raise ValueError(f"persisted artifact bytes differ: {relative_path}")
    split_rows = _read_csv(
        root.joinpath(*SPLIT_CSV_RELATIVE_PATH.split("/")), SPLIT_FIELDS
    )
    if len(split_rows) != len(data.assignments):
        raise ValueError("persisted speaker split row count is invalid")
    if [row["speaker_id"] for row in split_rows] != [
        assignment.speaker.speaker_id for assignment in data.assignments
    ]:
        raise ValueError("persisted speaker split ordering is invalid")
    persisted_portable = {
        split: _read_csv(
            root.joinpath(
                *{
                    "train": TRAIN_MANIFEST_RELATIVE_PATH,
                    "validation": VALIDATION_MANIFEST_RELATIVE_PATH,
                    "test": TEST_MANIFEST_RELATIVE_PATH,
                }[split].split("/")
            ),
            PORTABLE_FIELDS,
        )
        for split in OUTPUT_SPLITS
    }
    validate_portable_rows(
        authoritative.rows,
        data.assignments,
        persisted_portable,
        data.labels,
    )
    label_path = root.joinpath(*LABEL_MAPPING_RELATIVE_PATH.split("/"))
    with label_path.open("r", encoding="utf-8") as stream:
        labels = json.load(stream)
    if labels != dict(data.labels):
        raise ValueError("persisted label mapping is invalid")
    numeric_label_values = [
        labels[speaker_id]
        for speaker_id in sorted(labels, key=numeric_speaker_key)
    ]
    if numeric_label_values != list(range(len(labels))):
        raise ValueError("persisted labels are not contiguous")
    for identity_relative in (
        SPLIT_IDENTITY_RELATIVE_PATH,
        PORTABLE_IDENTITY_RELATIVE_PATH,
    ):
        identity_path = root.joinpath(*identity_relative.split("/"))
        with identity_path.open("r", encoding="utf-8") as stream:
            identity = json.load(stream)
        if not isinstance(identity, Mapping):
            raise ValueError("persisted identity must be an object")
        bindings = {
            "split_csv_sha256": SPLIT_CSV_RELATIVE_PATH,
            "split_policy_sha256": SPLIT_POLICY_RELATIVE_PATH,
            "train_manifest_sha256": TRAIN_MANIFEST_RELATIVE_PATH,
            "validation_manifest_sha256": VALIDATION_MANIFEST_RELATIVE_PATH,
            "test_manifest_sha256": TEST_MANIFEST_RELATIVE_PATH,
            "speaker_to_label_sha256": LABEL_MAPPING_RELATIVE_PATH,
        }
        for key, relative_path in bindings.items():
            actual = file_sha256(root.joinpath(*relative_path.split("/")))
            if identity.get(key) != actual:
                raise ValueError(f"persisted identity hash binding is invalid: {key}")
        if (
            identity.get("full_manifest_sha256")
            != authoritative.manifest_sha256
            or identity.get("full_manifest_identity_sha256")
            != authoritative.identity_sha256
            or identity.get("final_test_quarantine") is not True
        ):
            raise ValueError("persisted identity source binding is invalid")
    if expectations is not None:
        row_counts = {
            split: len(persisted_portable[split]) for split in OUTPUT_SPLITS
        }
        excluded_rows = sum(
            assignment.speaker.utterance_count
            for assignment in data.assignments
            if assignment.final_split == "excluded"
        )
        if sum(row_counts.values()) + excluded_rows != expectations.total_rows:
            raise ValueError("persisted row totals do not reconcile")


def _safe_remove_tree(path: Path, parent: Path) -> None:
    try:
        resolved_parent = parent.resolve(strict=True)
        resolved = path.resolve(strict=False)
        resolved.relative_to(resolved_parent)
    except (OSError, ValueError):
        return
    if path.is_dir():
        shutil.rmtree(path)


def publish_transaction(
    *,
    output_root: Path,
    artifacts: Mapping[str, bytes],
    validate_staged: Any,
) -> str:
    root = output_root.resolve(strict=True)
    finals = {
        relative: root.joinpath(*relative.split("/"))
        for relative in artifacts
    }
    existing = {relative: path.exists() for relative, path in finals.items()}
    if any(existing.values()):
        if not all(existing.values()):
            raise FileExistsError("split package has partial existing output")
        if any(
            not finals[relative].is_file()
            or finals[relative].read_bytes() != artifacts[relative]
            for relative in artifacts
        ):
            raise FileExistsError(
                "split package exists with non-identical contents"
            )
        validate_staged(root)
        return "already_present_byte_identical"

    staging = root / f".speaker_split_v2.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    staging.mkdir()
    published: list[Path] = []
    created_dirs: list[Path] = []
    try:
        for relative, payload in artifacts.items():
            path = staging.joinpath(*relative.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        validate_staged(staging)
        for relative in artifacts:
            source = staging.joinpath(*relative.split("/"))
            destination = finals[relative]
            missing_parents: list[Path] = []
            parent = destination.parent
            while parent != root and not parent.exists():
                missing_parents.append(parent)
                parent = parent.parent
            destination.parent.mkdir(parents=True, exist_ok=True)
            created_dirs.extend(reversed(missing_parents))
            os.replace(source, destination)
            published.append(destination)
        validate_staged(root)
        return "published_transactionally"
    except Exception:
        for path in reversed(published):
            if path.is_file():
                path.unlink()
        for directory in reversed(created_dirs):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        raise
    finally:
        if staging.exists():
            _safe_remove_tree(staging, root)


def create_split_package(
    *,
    project_root: Path,
    manifest_path: Path,
    identity_path: Path,
    output_root: Path,
    expected_manifest_sha256: str = APPROVED_FULL_MANIFEST_SHA256,
    expected_identity_sha256: str = APPROVED_FULL_MANIFEST_IDENTITY_SHA256,
    policy: SplitPolicy = SplitPolicy(),
    expectations: PackageExpectations | None = PRODUCTION_EXPECTATIONS,
    reference_root: Path | None = None,
) -> dict[str, Any]:
    project = project_root.resolve(strict=True)
    output = output_root.resolve(strict=True)
    try:
        output.relative_to(project)
    except ValueError as error:
        raise ValueError("output root must be inside the repository") from error
    manifest = manifest_path.resolve(strict=True)
    identity = identity_path.resolve(strict=True)
    input_hashes_before = (
        file_sha256(manifest),
        file_sha256(identity),
    )
    authoritative = load_authoritative_input(
        manifest,
        identity,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_identity_sha256=expected_identity_sha256,
        seed=policy.seed,
    )
    artifacts, data, summary = build_artifacts(
        authoritative, policy, expectations
    )

    def validate_at(root: Path) -> None:
        validate_persisted_package(
            root, artifacts, authoritative, data, expectations
        )

    publication = publish_transaction(
        output_root=output,
        artifacts=artifacts,
        validate_staged=validate_at,
    )
    validate_at(output)
    input_hashes_after = (file_sha256(manifest), file_sha256(identity))
    if input_hashes_after != input_hashes_before:
        raise RuntimeError("authoritative full-manifest inputs changed")

    reproduction = {
        "reference_compared": reference_root is not None,
        "all_required_artifacts_byte_identical": None,
        "artifact_matches": {},
    }
    if reference_root is not None:
        reference = reference_root.resolve(strict=True)
        matches = {}
        for relative in REPRODUCIBILITY_PATHS:
            produced = output.joinpath(*relative.split("/"))
            approved = reference.joinpath(*relative.split("/"))
            if not approved.is_file():
                raise FileNotFoundError(
                    f"reference artifact is missing: {relative}"
                )
            matches[relative] = produced.read_bytes() == approved.read_bytes()
        reproduction["artifact_matches"] = matches
        reproduction["all_required_artifacts_byte_identical"] = all(
            matches.values()
        )
        if not reproduction["all_required_artifacts_byte_identical"]:
            raise ValueError("independent reproduction differs from reference")

    return {
        "result": "PASS",
        "publication": publication,
        "full_manifest_sha256": authoritative.manifest_sha256,
        "full_manifest_identity_sha256": authoritative.identity_sha256,
        "speaker_counts": summary["statistics"]["speaker_counts"],
        "row_counts": summary["statistics"]["row_counts"],
        "duration_seconds": summary["statistics"]["duration_seconds"],
        "duplicate_counts_by_split": summary["statistics"][
            "duplicate_counts_by_split"
        ],
        "bucket_capacities": dict(data.bucket_capacities),
        "bucket_quotas": dict(data.bucket_quotas),
        "evaluation_balance": summary["evaluation_balance"],
        "train_class_count": len(data.labels),
        "label_range": [0, len(data.labels) - 1],
        "artifact_hashes": {
            relative: file_sha256(output.joinpath(*relative.split("/")))
            for relative in REPRODUCIBILITY_PATHS
        },
        "authoritative_inputs_preserved": True,
        "source_wav_accessed": False,
        "source_wav_changed": False,
        "reproducibility": reproduction,
        "formatting": {
            "encoding": "UTF-8 without BOM",
            "newline": "LF",
            "csv_column_order": "fixed",
            "duration_serialization": "fixed six decimal places",
            "json_key_order": "lexicographic sort_keys",
            "json_indent": 2,
            "identity_timestamps": False,
        },
    }
