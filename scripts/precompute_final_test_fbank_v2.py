"""Build the separate, manifest-allowlisted VieSpeaker2.0 final-test Fbank cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import speechbrain
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.final_evaluation_v2 import (  # noqa: E402
    APPROVED_INPUTS,
    BATCH_SIZE,
    EXPECTED_ROWS,
    EXPECTED_SPEAKERS,
    FEATURE_SHAPE,
    SHARD_SIZE,
    atomic_bytes,
    atomic_json,
    canonical_json,
    read_final_manifest,
    read_json,
    read_state,
    sha256_file,
    transition_state,
    validate_lock,
)
from src.speechbrain_frontend import SpeechBrainECAPAFrontend  # noqa: E402


CONFIG_PATH = REPO_ROOT / "configs/v2/final_test_fbank_cache_v2.json"
REPORT_JSON = REPO_ROOT / "reports/final_test_fbank_cache_v2.json"
REPORT_MD = REPO_ROOT / "reports/final_test_fbank_cache_v2.md"
INDEX_FIELDS = (
    "audio_path",
    "speaker_id",
    "label",
    "final_split",
    "shard_path",
    "within_shard_index",
    "feature_frames",
    "feature_bins",
    "feature_dtype",
    "manifest_row_index",
    "filename_group",
    "duplicate_group",
    "manifest_version",
)
SHARD_KEYS = {
    "schema_version",
    "features",
    "speaker_labels",
    "speaker_ids",
    "relative_audio_paths",
    "manifest_row_indices",
    "shard_number",
    "final_split",
}


def atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def source_path(dataset_root: Path, relative: str) -> Path:
    return dataset_root / Path(*PurePosixPath(relative).parts)


def source_snapshot(dataset_root: Path, rows: Sequence[Any]) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    for row in rows:
        path = source_path(dataset_root, row.relative_audio_path)
        try:
            stat = path.stat()
        except OSError as error:
            raise FileNotFoundError(
                f"unavailable approved test WAV {row.relative_audio_path}"
            ) from error
        if not path.is_file():
            raise ValueError(f"approved source is not a file: {row.relative_audio_path}")
        record = f"{row.relative_audio_path}\x1f{stat.st_size}\x1f{stat.st_mtime_ns}\n"
        digest.update(record.encode("utf-8"))
        total_bytes += stat.st_size
    return {
        "schema_version": 2,
        "snapshot_kind": "manifest_path_size_mtime_ns",
        "row_count": len(rows),
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "contains_absolute_paths": False,
    }


def load_waveform(dataset_root: Path, relative: str) -> torch.Tensor:
    path = source_path(dataset_root, relative)
    try:
        waveform, sample_rate = torchaudio.load(str(path), normalize=True)
    except Exception as error:
        raise RuntimeError(f"failed to load approved WAV {relative}: {error}") from error
    if sample_rate != 16000 or tuple(waveform.shape) != (1, 48000):
        raise ValueError(
            f"{relative}: expected mono [1,48000] at 16000 Hz, "
            f"got {tuple(waveform.shape)} at {sample_rate}"
        )
    waveform = waveform.squeeze(0).to(torch.float32).contiguous()
    if not bool(torch.isfinite(waveform).all().item()):
        raise ValueError(f"{relative}: waveform contains NaN or Inf")
    return waveform


def validate_shard(
    payload: Any, rows: Sequence[Any], shard_number: int
) -> None:
    count = len(rows)
    if not isinstance(payload, dict) or set(payload) != SHARD_KEYS:
        raise ValueError("final-test cache shard schema mismatch")
    features = payload["features"]
    labels = payload["speaker_labels"]
    if (
        payload["schema_version"] != 2
        or payload["shard_number"] != shard_number
        or payload["final_split"] != "test"
        or not isinstance(features, torch.Tensor)
        or tuple(features.shape) != (count, *FEATURE_SHAPE)
        or features.dtype != torch.float32
        or features.device.type != "cpu"
        or not features.is_contiguous()
        or not bool(torch.isfinite(features).all().item())
        or not isinstance(labels, torch.Tensor)
        or labels.dtype != torch.long
        or labels.tolist() != [-1] * count
        or payload["speaker_ids"] != [row.speaker_id for row in rows]
        or payload["relative_audio_paths"]
        != [row.relative_audio_path for row in rows]
        or payload["manifest_row_indices"]
        != [row.manifest_row_index for row in rows]
    ):
        raise ValueError(f"invalid final-test cache shard {shard_number}")


def index_bytes(rows: Sequence[Any]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        shard_number, within = divmod(row.manifest_row_index, SHARD_SIZE)
        writer.writerow(
            {
                "audio_path": row.relative_audio_path,
                "speaker_id": row.speaker_id,
                "label": -1,
                "final_split": "test",
                "shard_path": f"test/shard_{shard_number:05d}.pt",
                "within_shard_index": within,
                "feature_frames": FEATURE_SHAPE[0],
                "feature_bins": FEATURE_SHAPE[1],
                "feature_dtype": "float32",
                "manifest_row_index": row.manifest_row_index,
                "filename_group": row.filename_group,
                "duplicate_group": row.duplicate_group,
                "manifest_version": "v2",
            }
        )
    return stream.getvalue().encode("utf-8")


def validate_complete_cache(
    cache_dir: Path, rows: Sequence[Any], shard_hashes: dict[str, str]
) -> dict[str, Any]:
    test_dir = cache_dir / "test"
    expected_names = {
        f"shard_{number:05d}.pt"
        for number in range(math.ceil(EXPECTED_ROWS / SHARD_SIZE))
    }
    actual_names = {path.name for path in test_dir.iterdir() if path.is_file()}
    if actual_names != expected_names or any(path.is_dir() for path in test_dir.iterdir()):
        raise ValueError("final-test shard set differs from deterministic plan")
    if any(
        path.is_dir() and path.name != "test" for path in cache_dir.iterdir()
    ):
        raise ValueError("train/validation/unexpected directory in final-test cache")
    total_bytes = 0
    for number in range(len(expected_names)):
        path = test_dir / f"shard_{number:05d}.pt"
        relative = f"test/{path.name}"
        actual = sha256_file(path)
        if shard_hashes.get(relative) != actual:
            raise ValueError(f"finalized shard hash mismatch: {relative}")
        start = number * SHARD_SIZE
        selected = rows[start : start + SHARD_SIZE]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_shard(payload, selected, number)
        total_bytes += path.stat().st_size
    return {
        "row_count": len(rows),
        "speaker_count": len({row.speaker_id for row in rows}),
        "shard_count": len(expected_names),
        "total_shard_bytes": total_bytes,
        "feature_shape": list(FEATURE_SHAPE),
        "feature_dtype": "float32",
        "labels": [-1],
        "manifest_index_alignment": True,
        "all_shards_read_back": True,
        "unexpected_split_directories": 0,
    }


def markdown(report: dict[str, Any]) -> str:
    runtime = report["runtime"]
    return f"""# VieSpeaker2.0 Final-Test Fbank Cache

Result: **{report['result']}**

- Rows: {report['cache_validation']['row_count']}
- Speakers: {report['cache_validation']['speaker_count']}
- Shards: {report['cache_validation']['shard_count']}
- Shape/dtype: `[301, 80]` / float32
- Cache bytes: {report['cache_validation']['total_shard_bytes']}
- Extraction seconds: {runtime['extraction_duration_seconds']}
- Utterances/second: {runtime['utterances_per_second']}
- CUDA peak allocated bytes: {runtime['cuda_peak_allocated_bytes']}
- CUDA peak reserved bytes: {runtime['cuda_peak_reserved_bytes']}
- OOM count: 0
- Source path-size-mtime snapshot unchanged: true
- Existing train/validation cache protected bindings unchanged: true
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/fbank_cache_final_test_v2"),
    )
    args = parser.parse_args()
    if (
        args.device != "cuda:0"
        or args.batch_size != BATCH_SIZE
        or args.shard_size != SHARD_SIZE
    ):
        raise ValueError("final-test cache device/batch/shard settings are immutable")
    dataset_root = args.dataset_root.resolve()
    if str(dataset_root).lower() != r"e:\viespeaker2.0\augmented_dataset".lower():
        raise ValueError("unexpected final v2 dataset root")
    cache_dir = args.cache_dir
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    if cache_dir.resolve() != (REPO_ROOT / "outputs/fbank_cache_final_test_v2").resolve():
        raise ValueError("final-test cache must use its locked separate root")

    lock, lock_identity = validate_lock(REPO_ROOT)
    lock_identity_hash = sha256_file(
        REPO_ROOT / "configs/v2/final_evaluation_v2_identity.json"
    )
    state = read_state(REPO_ROOT, lock_identity_hash)
    if state["phase"] != "protocol_locked":
        raise RuntimeError(f"cache extraction forbidden in phase {state['phase']}")
    if sha256_file(CONFIG_PATH) != lock["fbank"]["config_sha256"]:
        raise ValueError("tracked final-test Fbank configuration changed")
    rows = read_final_manifest(
        REPO_ROOT / APPROVED_INPUTS["final_test_manifest"]["path"]
    )
    if len(rows) != EXPECTED_ROWS or len({row.speaker_id for row in rows}) != EXPECTED_SPEAKERS:
        raise ValueError("final-test rows/speakers mismatch")

    protected_paths = (
        REPO_ROOT / "outputs/fbank_cache_v2/fbank_cache_config_v2.json",
        REPO_ROOT / "outputs/fbank_cache_v2/fbank_cache_identity_v2.json",
        REPO_ROOT / "outputs/fbank_cache_v2/train_feature_index_v2.csv",
        REPO_ROOT / "outputs/fbank_cache_v2/validation_feature_index_v2.csv",
    )
    protected_before = {path.as_posix(): sha256_file(path) for path in protected_paths}
    cache_dir.mkdir(parents=True, exist_ok=True)
    test_dir = cache_dir / "test"
    test_dir.mkdir(exist_ok=True)
    config_copy = cache_dir / "fbank_cache_config_final_test_v2.json"
    index_path = cache_dir / "test_feature_index_v2.csv"
    build_state_path = cache_dir / "cache_build_state_final_test_v2.json"
    before_snapshot_path = cache_dir / "source_snapshot_before_v2.json"
    after_snapshot_path = cache_dir / "source_snapshot_after_v2.json"
    identity_path = cache_dir / "fbank_cache_identity_final_test_v2.json"
    runtime_path = cache_dir / "runtime_result_final_test_v2.json"
    planned_index = index_bytes(rows)
    tracked_config = CONFIG_PATH.read_bytes()

    if build_state_path.exists():
        build_state = read_json(build_state_path)
        if (
            build_state.get("lock_identity_sha256") != lock_identity_hash
            or config_copy.read_bytes() != tracked_config
            or index_path.read_bytes() != planned_index
        ):
            raise ValueError("resume rejected incompatible final-test cache metadata")
        before_snapshot = read_json(before_snapshot_path)
        if source_snapshot(dataset_root, rows) != before_snapshot:
            raise RuntimeError("approved source snapshot changed before cache resume")
    else:
        unexpected = {
            path.name
            for path in cache_dir.iterdir()
            if path.name not in {
                "test",
                "extraction.stdout.log",
                "extraction.stderr.log",
            }
        }
        if unexpected or any(test_dir.iterdir()):
            raise FileExistsError(f"cache root is not pristine: {sorted(unexpected)}")
        config_copy.write_bytes(tracked_config)
        index_path.write_bytes(planned_index)
        if config_copy.read_bytes() != tracked_config or index_path.read_bytes() != planned_index:
            raise RuntimeError("cache metadata readback mismatch")
        before_snapshot = source_snapshot(dataset_root, rows)
        atomic_json(before_snapshot_path, before_snapshot)
        build_state = {
            "schema_version": 2,
            "phase": "extracting",
            "lock_identity_sha256": lock_identity_hash,
            "config_sha256": sha256_file(config_copy),
            "index_sha256": sha256_file(index_path),
            "source_snapshot_sha256": before_snapshot["sha256"],
            "completed_shards": {},
        }
        atomic_json(build_state_path, build_state)

    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for authoritative final-test extraction")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    expected_compute_calls = sum(
        math.ceil(
            len(rows[number * SHARD_SIZE : (number + 1) * SHARD_SIZE])
            / BATCH_SIZE
        )
        for number in range(math.ceil(EXPECTED_ROWS / SHARD_SIZE))
        if not (test_dir / f"shard_{number:05d}.pt").exists()
    )
    frontend = SpeechBrainECAPAFrontend(device=device)
    frontend.eval()
    call_counts = {"compute_features": 0, "mean_var_norm": 0, "embedding_model": 0, "classifier": 0}
    hooks = [
        frontend.classifier.mods.compute_features.register_forward_hook(
            lambda *_: call_counts.__setitem__(
                "compute_features", call_counts["compute_features"] + 1
            )
        ),
        frontend.classifier.mods.mean_var_norm.register_forward_hook(
            lambda *_: call_counts.__setitem__(
                "mean_var_norm", call_counts["mean_var_norm"] + 1
            )
        ),
        frontend.classifier.mods.embedding_model.register_forward_hook(
            lambda *_: call_counts.__setitem__(
                "embedding_model", call_counts["embedding_model"] + 1
            )
        ),
        frontend.classifier.mods.classifier.register_forward_hook(
            lambda *_: call_counts.__setitem__(
                "classifier", call_counts["classifier"] + 1
            )
        ),
    ]
    try:
        for number, start in enumerate(range(0, len(rows), SHARD_SIZE)):
            selected = rows[start : start + SHARD_SIZE]
            target = test_dir / f"shard_{number:05d}.pt"
            relative = f"test/{target.name}"
            recorded = build_state["completed_shards"].get(relative)
            if target.exists():
                if recorded is None or sha256_file(target) != recorded:
                    raise ValueError(f"untracked or mismatching finalized shard: {relative}")
                validate_shard(
                    torch.load(target, map_location="cpu", weights_only=False),
                    selected,
                    number,
                )
                print(f"test shard {number + 1}/60 validated and skipped", flush=True)
                continue
            if recorded is not None:
                raise ValueError(f"recorded finalized shard is missing: {relative}")
            batches: list[torch.Tensor] = []
            for batch_start in range(0, len(selected), BATCH_SIZE):
                batch_rows = selected[batch_start : batch_start + BATCH_SIZE]
                waveforms = torch.stack(
                    [
                        load_waveform(dataset_root, row.relative_audio_path)
                        for row in batch_rows
                    ]
                )
                with torch.inference_mode():
                    features = frontend.compute_features(waveforms)
                features = features.detach().cpu().to(torch.float32).contiguous()
                if (
                    tuple(features.shape) != (len(batch_rows), *FEATURE_SHAPE)
                    or not bool(torch.isfinite(features).all().item())
                ):
                    raise ValueError("raw final-test Fbank batch contract mismatch")
                batches.append(features)
            payload = {
                "schema_version": 2,
                "features": torch.cat(batches, dim=0).contiguous(),
                "speaker_labels": torch.full((len(selected),), -1, dtype=torch.long),
                "speaker_ids": [row.speaker_id for row in selected],
                "relative_audio_paths": [
                    row.relative_audio_path for row in selected
                ],
                "manifest_row_indices": [
                    row.manifest_row_index for row in selected
                ],
                "shard_number": number,
                "final_split": "test",
            }
            validate_shard(payload, selected, number)
            atomic_torch_save(payload, target)
            validate_shard(
                torch.load(target, map_location="cpu", weights_only=False),
                selected,
                number,
            )
            build_state["completed_shards"][relative] = sha256_file(target)
            atomic_json(build_state_path, build_state)
            print(f"test shard {number + 1}/60 complete", flush=True)
        torch.cuda.synchronize(device)
    finally:
        for hook in hooks:
            hook.remove()
    extraction_seconds = time.perf_counter() - started
    if (
        call_counts["compute_features"] != expected_compute_calls
        or any(call_counts[name] for name in ("mean_var_norm", "embedding_model", "classifier"))
    ):
        raise RuntimeError(f"forbidden cache-stage module call: {call_counts}")

    after_snapshot = source_snapshot(dataset_root, rows)
    atomic_json(after_snapshot_path, after_snapshot)
    if before_snapshot != after_snapshot:
        raise RuntimeError("approved final-test source snapshot changed")
    validation = validate_complete_cache(
        cache_dir, rows, build_state["completed_shards"]
    )
    protected_after = {path.as_posix(): sha256_file(path) for path in protected_paths}
    if protected_before != protected_after:
        raise RuntimeError("existing train/validation cache artifacts changed")
    identity = {
        "schema_version": 2,
        "identity_kind": "final_test_fbank_cache_v2",
        "config_path": "outputs/fbank_cache_final_test_v2/fbank_cache_config_final_test_v2.json",
        "config_sha256": sha256_file(config_copy),
        "tracked_config_path": "configs/v2/final_test_fbank_cache_v2.json",
        "tracked_config_sha256": sha256_file(CONFIG_PATH),
        "index_path": "outputs/fbank_cache_final_test_v2/test_feature_index_v2.csv",
        "index_sha256": sha256_file(index_path),
        "manifest_binding": lock["final_test"]["manifest"],
        "lock_identity_sha256": lock_identity_hash,
        "row_count": EXPECTED_ROWS,
        "speaker_count": EXPECTED_SPEAKERS,
        "shard_count": 60,
        "shard_hashes": dict(sorted(build_state["completed_shards"].items())),
        "feature_shape": list(FEATURE_SHAPE),
        "feature_dtype": "float32",
        "raw_pre_normalization": True,
        "transposed": False,
        "source_snapshot_sha256": before_snapshot["sha256"],
        "source_state_unchanged": True,
        "protected_train_validation_cache_unchanged": True,
        "timestamps_in_identity": False,
        "absolute_paths_in_identity": False,
    }
    atomic_json(identity_path, identity)
    if read_json(identity_path) != identity:
        raise RuntimeError("cache identity readback mismatch")
    runtime = {
        "result": "PASS",
        "extraction_duration_seconds": extraction_seconds,
        "utterances_per_second": EXPECTED_ROWS / extraction_seconds,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "oom_count": 0,
        "model_inference_processes": 1,
        "module_call_counts": call_counts,
    }
    atomic_json(runtime_path, runtime)
    report = {
        "schema_version": 2,
        "result": "PASS",
        "cache_identity_path": "outputs/fbank_cache_final_test_v2/fbank_cache_identity_final_test_v2.json",
        "cache_identity_sha256": sha256_file(identity_path),
        "cache_config_sha256": sha256_file(config_copy),
        "cache_index_sha256": sha256_file(index_path),
        "cache_validation": validation,
        "source_snapshot": before_snapshot,
        "source_snapshot_unchanged": True,
        "protected_train_validation_cache_unchanged": True,
        "runtime": runtime,
    }
    atomic_json(REPORT_JSON, report)
    atomic_bytes(REPORT_MD, markdown(report).encode("utf-8"))
    report_md_hash = sha256_file(REPORT_MD)
    transition_state(
        REPO_ROOT,
        "protocol_locked",
        "cache_complete",
        {
            "outputs/fbank_cache_final_test_v2/fbank_cache_identity_final_test_v2.json": sha256_file(identity_path),
            "outputs/fbank_cache_final_test_v2/test_feature_index_v2.csv": sha256_file(index_path),
            "outputs/fbank_cache_final_test_v2/runtime_result_final_test_v2.json": sha256_file(runtime_path),
            "reports/final_test_fbank_cache_v2.json": sha256_file(REPORT_JSON),
            "reports/final_test_fbank_cache_v2.md": report_md_hash,
        },
    )
    print(canonical_json(report), end="")


if __name__ == "__main__":
    main()
