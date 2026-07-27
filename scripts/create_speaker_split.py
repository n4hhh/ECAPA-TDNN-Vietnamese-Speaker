"""Create deterministic candidate speaker-disjoint split v1 from manifest metadata."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

TRUSTED_GROUPS = ("train", "train_small")
QUARANTINE_GROUPS = ("test", "part")
BUCKETS = ("10-19", "20-49", "50+")
SPLIT_FIELDS = (
    "speaker_id", "final_split", "trusted_utterance_count",
    "train_utterance_count", "train_small_utterance_count",
    "provenance_signature", "utterance_bucket", "eligibility_reason",
)


class InsufficientEligibleSpeakers(RuntimeError):
    pass


@dataclass(frozen=True)
class ManifestData:
    path_column: str
    speaker_column: str
    group_column: str
    row_count: int
    all_speakers: frozenset[str]
    trusted_counts: dict[str, Counter[str]]
    quarantine_counts: dict[str, Counter[str]]


def detect_column(fieldnames: list[str] | None, candidates: tuple[str, ...], role: str) -> str:
    fields = fieldnames or []
    by_fold = {field.strip().casefold(): field for field in fields}
    matches = [by_fold[name] for name in candidates if name in by_fold]
    if len(matches) != 1:
        raise ValueError(f"could not uniquely detect {role} column; candidates={candidates}, fields={fields}")
    return matches[0]


def read_manifest(path: Path) -> ManifestData:
    trusted: dict[str, Counter[str]] = defaultdict(Counter)
    quarantine: dict[str, Counter[str]] = defaultdict(Counter)
    speakers: set[str] = set()
    seen_paths: dict[str, int] = {}
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        path_col = detect_column(reader.fieldnames, ("audio_path", "wav_path", "path", "file"), "WAV path")
        speaker_col = detect_column(reader.fieldnames, ("speaker_id", "speaker", "spk_id", "speakerid"), "speaker ID")
        group_col = detect_column(reader.fieldnames, ("filename_group", "provenance_group", "group", "source_group"), "provenance group")
        for row_number, row in enumerate(reader, start=2):
            rows += 1
            audio_path = row[path_col].strip()
            speaker_id = row[speaker_col].strip()
            group = row[group_col].strip().casefold()
            if not audio_path or not speaker_id or not group:
                raise ValueError(f"row {row_number} has an empty path, speaker ID, or group")
            path_key = audio_path.casefold()
            if path_key in seen_paths:
                raise ValueError(f"duplicate audio path at rows {seen_paths[path_key]} and {row_number}: {audio_path}")
            seen_paths[path_key] = row_number
            speakers.add(speaker_id)
            if group in TRUSTED_GROUPS:
                trusted[speaker_id][group] += 1
            elif group in QUARANTINE_GROUPS:
                quarantine[speaker_id][group] += 1
            else:
                raise ValueError(f"row {row_number} has unsupported provenance group {group!r}")
    if rows == 0:
        raise ValueError("manifest contains no data rows")
    return ManifestData(path_col, speaker_col, group_col, rows, frozenset(speakers), dict(trusted), dict(quarantine))


def provenance(counts: Counter[str]) -> str:
    if counts["train"] and counts["train_small"]:
        return "train_and_train_small"
    if counts["train"]:
        return "train_only"
    if counts["train_small"]:
        return "train_small_only"
    raise AssertionError("trusted speaker has no trusted records")


def bucket(count: int) -> str:
    if count < 10:
        return "under_10"
    if count < 20:
        return "10-19"
    if count < 50:
        return "20-49"
    return "50+"


def apportion(capacities: dict[str, int], target: int) -> dict[str, int]:
    total = sum(capacities.values())
    if target > total:
        raise ValueError("allocation target exceeds capacity")
    if not target:
        return {key: 0 for key in capacities}
    raw = {key: target * value / total for key, value in capacities.items()}
    result = {key: min(capacities[key], int(raw[key])) for key in capacities}
    remaining = target - sum(result.values())
    order = sorted(capacities, key=lambda key: (-(raw[key] - int(raw[key])), key))
    while remaining:
        progressed = False
        for key in order:
            if result[key] < capacities[key]:
                result[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise AssertionError("unable to complete allocation")
    return result


def create_assignments(data: ManifestData, seed: int, min_train: int, min_eval: int,
                       validation_target: int, test_target: int) -> list[dict[str, str | int]]:
    eligible = [sid for sid, counts in data.trusted_counts.items() if sum(counts.values()) >= min_eval]
    evaluation_target = validation_target + test_target
    if len(eligible) < evaluation_target:
        raise InsufficientEligibleSpeakers(
            f"INSUFFICIENT_ELIGIBLE_SPEAKERS: need {evaluation_target}, found {len(eligible)}"
        )

    strata: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for sid in sorted(eligible):
        counts = data.trusted_counts[sid]
        strata[bucket(sum(counts.values()))][provenance(counts)].append(sid)
    rng = random.Random(seed)
    for bucket_name in sorted(strata):
        for signature in sorted(strata[bucket_name]):
            rng.shuffle(strata[bucket_name][signature])

    bucket_quotas = apportion(
        {name: sum(len(ids) for ids in strata[name].values()) for name in BUCKETS},
        evaluation_target,
    )
    selected_by_stratum: dict[tuple[str, str], list[str]] = {}
    for bucket_name in BUCKETS:
        signature_quotas = apportion(
            {sig: len(ids) for sig, ids in strata[bucket_name].items()},
            bucket_quotas[bucket_name],
        )
        for signature in sorted(strata[bucket_name]):
            selected_by_stratum[(bucket_name, signature)] = strata[bucket_name][signature][:signature_quotas[signature]]

    evaluation_assignments: dict[str, str] = {}
    totals = Counter()
    stratum_totals: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    limits = {"validation": validation_target, "test": test_target}
    for stratum in sorted(selected_by_stratum):
        for sid in selected_by_stratum[stratum]:
            choices = [name for name in ("validation", "test") if totals[name] < limits[name]]
            chosen = min(choices, key=lambda name: (stratum_totals[stratum][name], totals[name], name))
            evaluation_assignments[sid] = chosen
            totals[chosen] += 1
            stratum_totals[stratum][chosen] += 1

    rows: list[dict[str, str | int]] = []
    for sid in sorted(data.trusted_counts):
        counts = data.trusted_counts[sid]
        total = sum(counts.values())
        if sid in evaluation_assignments:
            final_split = evaluation_assignments[sid]
            reason = f"evaluation_eligible_at_least_{min_eval}"
        elif total >= min_train:
            final_split = "train"
            reason = "train_eligible" if total < min_eval else "evaluation_eligible_assigned_train"
        else:
            final_split = "excluded"
            reason = f"fewer_than_{min_train}_trusted_utterances"
        rows.append({
            "speaker_id": sid, "final_split": final_split,
            "trusted_utterance_count": total,
            "train_utterance_count": counts["train"],
            "train_small_utterance_count": counts["train_small"],
            "provenance_signature": provenance(counts),
            "utterance_bucket": bucket(total), "eligibility_reason": reason,
        })
    validate(data, rows, min_train, min_eval, validation_target, test_target)
    return rows


def validate(data: ManifestData, rows: list[dict[str, str | int]], min_train: int,
             min_eval: int, validation_target: int, test_target: int) -> None:
    assignments: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        assignments[str(row["final_split"])].add(str(row["speaker_id"]))
    trusted_speakers = set(data.trusted_counts)
    if len(rows) != len(trusted_speakers) or {str(row["speaker_id"]) for row in rows} != trusted_speakers:
        raise AssertionError("every trusted speaker must receive exactly one assignment")
    if len(assignments["validation"]) != validation_target or len(assignments["test"]) != test_target:
        raise AssertionError("evaluation speaker targets were not met exactly")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if assignments[left] & assignments[right]:
            raise AssertionError(f"speaker leakage between {left} and {right}")
    for row in rows:
        split = str(row["final_split"])
        count = int(row["trusted_utterance_count"])
        if split == "train" and count < min_train:
            raise AssertionError("train speaker below minimum")
        if split in {"validation", "test"} and count < min_eval:
            raise AssertionError("evaluation speaker below minimum")
    trusted_records = sum(sum(counts.values()) for counts in data.trusted_counts.values())
    if sum(int(row["trusted_utterance_count"]) for row in rows) != trusted_records:
        raise AssertionError("trusted record totals do not reconcile")
    quarantine_records = sum(sum(counts.values()) for counts in data.quarantine_counts.values())
    if trusted_records + quarantine_records != data.row_count:
        raise AssertionError("assigned, excluded, and quarantined totals do not reconcile")


def quarantine_rows(data: ManifestData) -> list[dict[str, str | int]]:
    trusted = set(data.trusted_counts)
    test_speakers = {sid for sid, counts in data.quarantine_counts.items() if counts["test"]}
    part_speakers = {sid for sid, counts in data.quarantine_counts.items() if counts["part"]}
    rows = []
    for group, speakers in (("test", test_speakers), ("part", part_speakers)):
        rows.append({
            "quarantine_group": group,
            "utterance_count": sum(counts[group] for counts in data.quarantine_counts.values()),
            "unique_speaker_count": len(speakers),
            "trusted_overlap_speaker_count": len(speakers & trusted),
            "quarantine_only_speaker_count": len(speakers - trusted),
            "test_and_part_speaker_count": len(test_speakers & part_speakers),
        })
    return rows


def statistics(data: ManifestData, rows: list[dict[str, str | int]]) -> dict:
    split_speakers = Counter(str(row["final_split"]) for row in rows)
    split_utterances = Counter()
    bucket_counts: dict[str, Counter[str]] = defaultdict(Counter)
    provenance_counts = Counter()
    eligibility = Counter()
    for row in rows:
        split = str(row["final_split"])
        count = int(row["trusted_utterance_count"])
        split_utterances[split] += count
        if str(row["utterance_bucket"]) in BUCKETS:
            bucket_counts[split][str(row["utterance_bucket"])] += 1
        provenance_counts[str(row["provenance_signature"])] += 1
        eligibility["fewer_than_5" if count < 5 else "5_to_9" if count < 10 else "at_least_10"] += 1
    return {
        "split_speakers": split_speakers, "split_utterances": split_utterances,
        "bucket_counts": bucket_counts, "provenance_counts": provenance_counts,
        "eligibility": eligibility, "quarantine": quarantine_rows(data),
    }


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_summary(data: ManifestData, rows: list[dict[str, str | int]], config: dict) -> str:
    stats = statistics(data, rows)
    lines = [
        "# Candidate speaker split v1", "",
        "> Candidate only. This split is not approved for training.", "",
        "## Input", "",
        f"- Manifest: `{config['manifest']}`",
        f"- Detected columns: `{data.path_column}`, `{data.speaker_column}`, `{data.group_column}`",
        f"- Rows: {data.row_count}", f"- Unique speakers: {len(data.all_speakers)}", "",
        "## Trusted pool", "",
        f"- Utterances: {sum(stats['split_utterances'].values())}",
        f"- Speakers: {len(rows)}",
        f"- train_only: {stats['provenance_counts']['train_only']}",
        f"- train_small_only: {stats['provenance_counts']['train_small_only']}",
        f"- train_and_train_small: {stats['provenance_counts']['train_and_train_small']}", "",
        "## Eligibility", "",
        f"- Fewer than 5: {stats['eligibility']['fewer_than_5']}",
        f"- 5-9: {stats['eligibility']['5_to_9']}",
        f"- At least 10: {stats['eligibility']['at_least_10']}", "",
        "## Final split", "",
    ]
    for split in ("train", "validation", "test", "excluded"):
        lines.append(f"- {split}: {stats['split_speakers'][split]} speakers, {stats['split_utterances'][split]} utterances")
    lines.extend(["", "## Speaker bucket distribution", "", "| Split | 10-19 | 20-49 | 50+ |", "|---|---:|---:|---:|"])
    for split in ("train", "validation", "test"):
        lines.append(f"| {split} | {stats['bucket_counts'][split]['10-19']} | {stats['bucket_counts'][split]['20-49']} | {stats['bucket_counts'][split]['50+']} |")
    lines.extend(["", "## Quarantine audit", "", "| Group | Utterances | Speakers | Trusted overlap | Quarantine-only | In both test and part |", "|---|---:|---:|---:|---:|---:|"])
    for row in stats["quarantine"]:
        lines.append(f"| {row['quarantine_group']} | {row['utterance_count']} | {row['unique_speaker_count']} | {row['trusted_overlap_speaker_count']} | {row['quarantine_only_speaker_count']} | {row['test_and_part_speaker_count']} |")
    lines.extend(["", "## Validation", "", "- Train/validation overlap: 0", "- Train/test overlap: 0", "- Validation/test overlap: 0", "- Quarantine records assigned to final splits: 0", "- All manifest totals reconciled: yes", "- Deterministic for the recorded seed: yes", ""])
    return "\n".join(lines)


def write_outputs(output_root: Path, data: ManifestData, rows: list[dict[str, str | int]], config: dict) -> None:
    write_csv(output_root / "splits/speaker_split_v1.csv", SPLIT_FIELDS, rows)
    excluded = [row for row in rows if row["final_split"] == "excluded"]
    write_csv(output_root / "splits/excluded_speakers_v1.csv", SPLIT_FIELDS, excluded)
    config_path = output_root / "splits/split_config_v1.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = quarantine_rows(data)
    write_csv(output_root / "reports/quarantine_audit_v1.csv", audit[0].keys(), audit)
    report_path = output_root / "reports/split_v1_summary.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_summary(data, rows, config), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/full_manifest.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-train-utterances", type=int, default=5)
    parser.add_argument("--min-eval-utterances", type=int, default=10)
    parser.add_argument("--validation-speakers", type=int, default=100)
    parser.add_argument("--test-speakers", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_train_utterances != 5 or args.min_eval_utterances != 10:
        raise ValueError("candidate split v1 requires minimums 5 and 10")
    if args.validation_speakers != 100 or args.test_speakers != 100:
        raise ValueError("candidate split v1 requires exactly 100 validation and 100 test speakers")
    manifest = args.manifest.resolve(strict=True)
    data = read_manifest(manifest)
    rows = create_assignments(data, args.seed, args.min_train_utterances,
                              args.min_eval_utterances, args.validation_speakers,
                              args.test_speakers)
    config = {
        "manifest": str(args.manifest), "seed": args.seed,
        "trusted_groups": list(TRUSTED_GROUPS), "quarantine_groups": list(QUARANTINE_GROUPS),
        "minimum_train_utterances": args.min_train_utterances,
        "minimum_evaluation_utterances": args.min_eval_utterances,
        "validation_target": args.validation_speakers, "test_target": args.test_speakers,
        "utterance_buckets": {"10-19": [10, 19], "20-49": [20, 49], "50+": [50, None]},
    }
    print(render_summary(data, rows, config))
    if args.dry_run:
        print("DRY RUN: no output files written")
    else:
        write_outputs(args.output_root.resolve(), data, rows, config)
        print(f"Wrote candidate split outputs under {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
