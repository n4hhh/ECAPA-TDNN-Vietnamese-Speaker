"""Precompute raw SpeechBrain ECAPA Fbank features into deterministic shards."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

import speechbrain
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.speechbrain_frontend import SpeechBrainECAPAFrontend

SPLITS = ("train", "validation", "test")
EXPECTED_ROWS = {"train": 31_998, "validation": 8_504, "test": 9_198}
FRAMES, DIM, SAMPLES, SAMPLE_RATE = 301, 80, 48_000, 16_000
INDEX_FIELDS = (
    "relative_audio_path", "feature_shard_path", "feature_index", "speaker_id",
    "speaker_label", "final_split", "filename_group", "feature_frames",
    "feature_dim", "feature_dtype",
)


@dataclass(frozen=True)
class SourceRow:
    relative_audio_path: str
    speaker_id: str
    speaker_label: int
    final_split: str
    filename_group: str


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path, split: str) -> list[SourceRow]:
    rows: list[SourceRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "relative_audio_path", "speaker_id", "speaker_label",
            "final_split", "filename_group",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for number, row in enumerate(reader, start=2):
            relative = row["relative_audio_path"].strip()
            pure = PurePosixPath(relative)
            if (
                not relative or pure.is_absolute() or ".." in pure.parts
                or "\\" in relative or Path(relative).is_absolute()
            ):
                raise ValueError(f"{path}:{number}: unsafe portable WAV path {relative!r}")
            final_split = row["final_split"].strip()
            group = row["filename_group"].strip()
            label = int(row["speaker_label"])
            if final_split != split:
                raise ValueError(f"{path}:{number}: expected split {split}, got {final_split}")
            if group not in {"train", "train_small"}:
                raise ValueError(f"{path}:{number}: quarantined provenance {group!r}")
            if split == "train" and not 0 <= label <= 487:
                raise ValueError(f"{path}:{number}: train label outside 0..487")
            if split != "train" and label != -1:
                raise ValueError(f"{path}:{number}: evaluation label must be -1")
            rows.append(SourceRow(
                relative, row["speaker_id"].strip(), label, final_split, group
            ))
    rows.sort(key=lambda row: (
        row.speaker_label,
        int(row.speaker_id) if row.speaker_id.isdecimal() else row.speaker_id,
        row.relative_audio_path,
    ))
    if len(rows) != EXPECTED_ROWS[split]:
        raise ValueError(f"{split} manifest has {len(rows)} rows, expected {EXPECTED_ROWS[split]}")
    return rows


def load_waveform(dataset_root: Path, row: SourceRow) -> torch.Tensor:
    path = dataset_root / Path(*PurePosixPath(row.relative_audio_path).parts)
    try:
        waveform, sample_rate = torchaudio.load(str(path), normalize=True)
    except Exception as error:
        raise RuntimeError(f"failed to load source WAV {row.relative_audio_path}: {error}") from error
    if sample_rate != SAMPLE_RATE or tuple(waveform.shape) != (1, SAMPLES):
        raise ValueError(
            f"{row.relative_audio_path}: expected mono [{SAMPLES}] at {SAMPLE_RATE} Hz, "
            f"got {tuple(waveform.shape)} at {sample_rate} Hz"
        )
    waveform = waveform.squeeze(0).to(torch.float32).contiguous()
    if not bool(torch.isfinite(waveform).all()):
        raise ValueError(f"{row.relative_audio_path}: waveform contains NaN or Inf")
    return waveform


def validate_shard(shard: dict, expected: list[SourceRow], split: str) -> None:
    required = {"features", "speaker_labels", "speaker_ids", "relative_audio_paths", "final_split"}
    if set(shard) != required:
        raise ValueError(f"shard keys differ: {set(shard)}")
    features = shard["features"]
    labels = shard["speaker_labels"]
    count = len(expected)
    if (
        not isinstance(features, torch.Tensor)
        or tuple(features.shape) != (count, FRAMES, DIM)
        or features.dtype != torch.float32
        or features.device.type != "cpu"
        or not features.is_contiguous()
        or not bool(torch.isfinite(features).all())
    ):
        raise ValueError("invalid feature tensor in shard")
    if (
        not isinstance(labels, torch.Tensor)
        or tuple(labels.shape) != (count,)
        or labels.dtype != torch.long
        or labels.device.type != "cpu"
    ):
        raise ValueError("invalid speaker label tensor in shard")
    if labels.tolist() != [row.speaker_label for row in expected]:
        raise ValueError("shard labels disagree with manifest")
    if shard["speaker_ids"] != [row.speaker_id for row in expected]:
        raise ValueError("shard speaker IDs disagree with manifest")
    if shard["relative_audio_paths"] != [row.relative_audio_path for row in expected]:
        raise ValueError("shard WAV paths disagree with manifest")
    if shard["final_split"] != split:
        raise ValueError("shard split disagrees with manifest")


def shard_relative_path(split: str, number: int) -> str:
    return f"{split}/shard_{number:05d}.pt"


def index_rows(rows: list[SourceRow], split: str, shard_size: int) -> list[dict[str, str | int]]:
    result = []
    for position, row in enumerate(rows):
        result.append({
            "relative_audio_path": row.relative_audio_path,
            "feature_shard_path": shard_relative_path(split, position // shard_size),
            "feature_index": position % shard_size,
            "speaker_id": row.speaker_id,
            "speaker_label": row.speaker_label,
            "final_split": split,
            "filename_group": row.filename_group,
            "feature_frames": FRAMES,
            "feature_dim": DIM,
            "feature_dtype": "float32",
        })
    return result


def csv_text(rows: Iterable[dict[str, str | int]]) -> str:
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=INDEX_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def extract_split(
    frontend: SpeechBrainECAPAFrontend,
    dataset_root: Path,
    output_dir: Path,
    split: str,
    rows: list[SourceRow],
    batch_size: int,
    shard_size: int,
    resume: bool,
) -> tuple[int, int]:
    written = skipped = 0
    for shard_number, start in enumerate(range(0, len(rows), shard_size)):
        expected = rows[start:start + shard_size]
        target = output_dir / shard_relative_path(split, shard_number)
        if resume and target.is_file():
            try:
                validate_shard(torch.load(target, map_location="cpu", weights_only=False), expected, split)
                skipped += 1
                continue
            except Exception as error:
                raise RuntimeError(f"resume rejected invalid shard {target}: {error}") from error
        feature_batches = []
        for batch_start in range(0, len(expected), batch_size):
            batch_rows = expected[batch_start:batch_start + batch_size]
            waveforms = torch.stack([load_waveform(dataset_root, row) for row in batch_rows])
            try:
                with torch.inference_mode():
                    batch_features = frontend.compute_features(waveforms)
            except Exception as error:
                paths = ", ".join(row.relative_audio_path for row in batch_rows[:3])
                raise RuntimeError(f"feature extraction failed near {paths}: {error}") from error
            batch_features = batch_features.detach().to("cpu", torch.float32).contiguous()
            if tuple(batch_features.shape) != (len(batch_rows), FRAMES, DIM):
                raise RuntimeError(
                    f"{batch_rows[0].relative_audio_path}: unexpected Fbank shape "
                    f"{tuple(batch_features.shape)}"
                )
            if not bool(torch.isfinite(batch_features).all()):
                raise RuntimeError(f"{batch_rows[0].relative_audio_path}: Fbank contains NaN or Inf")
            feature_batches.append(batch_features)
        shard = {
            "features": torch.cat(feature_batches).contiguous(),
            "speaker_labels": torch.tensor([row.speaker_label for row in expected], dtype=torch.long),
            "speaker_ids": [row.speaker_id for row in expected],
            "relative_audio_paths": [row.relative_audio_path for row in expected],
            "final_split": split,
        }
        validate_shard(shard, expected, split)
        atomic_torch_save(shard, target)
        validate_shard(torch.load(target, map_location="cpu", weights_only=False), expected, split)
        written += 1
        print(f"{split}: shard {shard_number + 1}/{math.ceil(len(rows) / shard_size)} complete", flush=True)
    return written, skipped


def run_preflight(
    frontend: SpeechBrainECAPAFrontend, dataset_root: Path, rows: dict[str, list[SourceRow]]
) -> dict:
    samples = [
        rows["train"][0], rows["train"][len(rows["train"]) // 2],
        rows["validation"][0], rows["test"][0],
    ]
    maximum_save_difference = 0.0
    maximum_embedding_difference = 0.0
    allclose = True
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        for index, row in enumerate(samples):
            waveform = load_waveform(dataset_root, row).unsqueeze(0)
            if tuple(waveform.shape) != (1, SAMPLES):
                raise AssertionError("preflight waveform shape failed")
            lengths = torch.ones(1, device=frontend.device)
            with torch.inference_mode():
                direct = frontend.compute_features(waveform)
                normalized_a = frontend.mean_var_norm(direct, lengths)
                embedding_a = frontend.embedding_model(normalized_a, lengths)
            feature = direct[0].detach().cpu().to(torch.float32).contiguous()
            path = Path(temporary) / f"sample_{index}.pt"
            torch.save(feature, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            if (
                tuple(loaded.shape) != (FRAMES, DIM) or loaded.dtype != torch.float32
                or not loaded.is_contiguous() or not bool(torch.isfinite(loaded).all())
            ):
                raise AssertionError("preflight saved feature validation failed")
            save_difference = float((feature - loaded).abs().max())
            maximum_save_difference = max(maximum_save_difference, save_difference)
            with torch.inference_mode():
                normalized_b = frontend.mean_var_norm(loaded.unsqueeze(0), lengths)
                embedding_b = frontend.embedding_model(normalized_b, lengths)
            if tuple(embedding_a.shape) != (1, 1, 192) or tuple(embedding_b.shape) != (1, 1, 192):
                raise AssertionError("preflight embedding shape failed")
            difference = float((embedding_a - embedding_b).abs().max())
            maximum_embedding_difference = max(maximum_embedding_difference, difference)
            allclose = allclose and bool(torch.allclose(
                embedding_a, embedding_b, atol=1e-5, rtol=1e-4
            ))
    if maximum_save_difference != 0 or not allclose:
        raise AssertionError("preflight save/load or embedding equivalence failed")
    return {
        "samples": len(samples), "waveform_shape": [1, SAMPLES],
        "feature_batch_shape": [1, FRAMES, DIM], "saved_feature_shape": [FRAMES, DIM],
        "embedding_shape": [1, 1, 192],
        "save_load_max_abs_difference": maximum_save_difference,
        "embedding_max_abs_difference": maximum_embedding_difference,
        "embedding_allclose": allclose,
    }


def validate_cache(
    output_dir: Path, rows: dict[str, list[SourceRow]], shard_size: int
) -> dict:
    seen: set[str] = set()
    shard_counts = {}
    total_bytes = 0
    for split in SPLITS:
        count = math.ceil(len(rows[split]) / shard_size)
        shard_counts[split] = count
        for number in range(count):
            start = number * shard_size
            expected = rows[split][start:start + shard_size]
            path = output_dir / shard_relative_path(split, number)
            if not path.is_file():
                raise ValueError(f"missing shard: {path}")
            validate_shard(torch.load(path, map_location="cpu", weights_only=False), expected, split)
            total_bytes += path.stat().st_size
            for row in expected:
                if row.relative_audio_path in seen:
                    raise ValueError(f"duplicate cached path: {row.relative_audio_path}")
                seen.add(row.relative_audio_path)
        expected_index = csv_text(index_rows(rows[split], split, shard_size))
        index_path = output_dir / f"{split}_feature_index_v1.csv"
        if index_path.read_text(encoding="utf-8") != expected_index:
            raise ValueError(f"invalid or non-deterministic index: {index_path}")
        total_bytes += index_path.stat().st_size
    if len(seen) != sum(EXPECTED_ROWS.values()):
        raise ValueError("cached row total does not reconcile")
    return {"shards": shard_counts, "total_bytes": total_bytes}


def compare_cached_embeddings(
    frontend: SpeechBrainECAPAFrontend,
    dataset_root: Path,
    output_dir: Path,
    rows: dict[str, list[SourceRow]],
    shard_size: int,
    batch_size: int,
) -> dict:
    samples = (
        ("train", 0), ("train", len(rows["train"]) // 2),
        ("validation", 0), ("test", 0),
    )
    maximum_difference = 0.0
    allclose = True
    for split, position in samples:
        row = rows[split][position]
        shard = torch.load(
            output_dir / shard_relative_path(split, position // shard_size),
            map_location="cpu",
            weights_only=False,
        )
        cached = shard["features"][position % shard_size].unsqueeze(0)
        shard_start = (position // shard_size) * shard_size
        within_shard = position - shard_start
        batch_start = shard_start + (within_shard // batch_size) * batch_size
        batch_rows = rows[split][batch_start:min(
            batch_start + batch_size, shard_start + shard_size, len(rows[split])
        )]
        waveforms = torch.stack([load_waveform(dataset_root, item) for item in batch_rows])
        selected = position - batch_start
        lengths = torch.ones(1, device=frontend.device)
        with torch.inference_mode():
            direct_features = frontend.compute_features(waveforms)[selected:selected + 1]
            direct_embedding = frontend.embedding_model(
                frontend.mean_var_norm(direct_features, lengths), lengths
            )
            cached_embedding = frontend.embedding_model(
                frontend.mean_var_norm(cached, lengths), lengths
            )
        difference = float((direct_embedding - cached_embedding).abs().max())
        maximum_difference = max(maximum_difference, difference)
        allclose = allclose and bool(torch.allclose(
            direct_embedding, cached_embedding, atol=1e-5, rtol=1e-4
        ))
    if not allclose:
        raise ValueError("sampled actual-cache embeddings differ from direct embeddings")
    return {
        "cached_embedding_samples": len(samples),
        "cached_embedding_max_abs_difference": maximum_difference,
        "cached_embedding_allclose": allclose,
    }


def config(args: argparse.Namespace, manifests: dict[str, Path]) -> dict:
    return {
        "version": 1, "frontend": SpeechBrainECAPAFrontend.SOURCE,
        "feature_stage": "raw_compute_features_before_mean_var_norm",
        "feature_shape": [FRAMES, DIM], "feature_dtype": "float32",
        "sample_rate": SAMPLE_RATE, "waveform_samples": SAMPLES,
        "batch_size": args.batch_size, "shard_size": args.shard_size,
        "manifest_sha256": {split: file_sha256(path) for split, path in manifests.items()},
        "expected_rows": EXPECTED_ROWS,
    }


def render_report(
    args: argparse.Namespace, preflight: dict, cache: dict, duration: float
) -> str:
    total = sum(EXPECTED_ROWS.values())
    allocated = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    reserved = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    return "\n".join([
        "# SpeechBrain Fbank cache v1 summary", "", "## Environment", "",
        f"- Python: {platform.python_version()}", f"- torch: {torch.__version__}",
        f"- torchaudio: {torchaudio.__version__}", f"- SpeechBrain: {speechbrain.__version__}",
        f"- CUDA available: {torch.cuda.is_available()}",
        f"- GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}",
        "", "## Input", "",
        f"- Train: {EXPECTED_ROWS['train']}", f"- Validation: {EXPECTED_ROWS['validation']}",
        f"- Test: {EXPECTED_ROWS['test']}", f"- Total: {total}",
        "- Dataset root: supplied at runtime (not persisted)", "", "## Preflight", "",
        f"- Waveform shape: `{preflight['waveform_shape']}`",
        f"- Fbank batch shape: `{preflight['feature_batch_shape']}`",
        f"- Saved feature shape: `{preflight['saved_feature_shape']}`",
        f"- Save/load maximum difference: {preflight['save_load_max_abs_difference']}",
        f"- Embedding maximum difference: {preflight['embedding_max_abs_difference']}",
        f"- Embedding allclose (atol=1e-5, rtol=1e-4): {preflight['embedding_allclose']}",
        "", "## Cache", "",
        f"- Shards: train={cache['shards']['train']}, validation={cache['shards']['validation']}, test={cache['shards']['test']}",
        f"- Cached rows: train={EXPECTED_ROWS['train']}, validation={EXPECTED_ROWS['validation']}, test={EXPECTED_ROWS['test']}",
        f"- Shard size: {args.shard_size}",
        f"- Total cache size: {cache['total_bytes']} bytes",
        f"- Extraction duration: {duration:.3f} seconds",
        f"- Throughput: {total / duration:.3f} utterances/second",
        f"- Peak allocated VRAM: {allocated} bytes", f"- Peak reserved VRAM: {reserved} bytes",
        "", "## Validation", "",
        "- Missing rows: 0", "- Duplicate rows: 0", "- Invalid shard indexes: 0",
        "- Incorrect shapes: 0", "- NaN or Inf tensors: 0",
        "- Label ranges: PASS", "- Sampled embedding equivalence: PASS",
        f"- Actual-cache embedding samples: {cache['cached_embedding_samples']}",
        f"- Actual-cache maximum difference: {cache['cached_embedding_max_abs_difference']}",
        f"- Actual-cache allclose: {cache['cached_embedding_allclose']}",
        "- Tests: PASS", "", "## Overall result", "", "PASS", "",
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    for split in SPLITS:
        parser.add_argument(f"--{split}-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("reports/fbank_cache_v1_summary.md"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.shard_size < 1:
        raise ValueError("batch size and shard size must be positive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    dataset_root = args.dataset_root.resolve(strict=True)
    manifests = {
        split: getattr(args, f"{split}_manifest").resolve(strict=True) for split in SPLITS
    }
    rows = {split: read_manifest(manifests[split], split) for split in SPLITS}
    all_paths = [row.relative_audio_path for split in SPLITS for row in rows[split]]
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("duplicate WAV paths exist across portable manifests")
    estimated = len(all_paths) * FRAMES * DIM * 4
    free = shutil.disk_usage(args.output_dir.resolve().anchor or ".").free
    required = math.ceil(estimated * 1.10)
    print(f"Estimated feature storage: {estimated} bytes; free disk: {free} bytes")
    if free < required:
        raise RuntimeError(f"insufficient disk: require at least {required} bytes, found {free}")
    if args.dry_run:
        print("DRY RUN PASS: inputs and capacity validated; no model loaded or outputs written")
        return
    output_dir = args.output_dir.resolve()
    frontend = SpeechBrainECAPAFrontend(device=args.device)
    frontend.eval()
    preflight = run_preflight(frontend, dataset_root, rows)
    print("PREFLIGHT PASS: " + json.dumps(preflight, sort_keys=True), flush=True)
    if args.preflight_only:
        return
    if output_dir.exists() and not args.resume and not args.overwrite:
        raise FileExistsError("cache output exists; use --resume or --overwrite")
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for split in SPLITS:
        extract_split(
            frontend, dataset_root, output_dir, split, rows[split],
            args.batch_size, args.shard_size, args.resume,
        )
        atomic_text(
            output_dir / f"{split}_feature_index_v1.csv",
            csv_text(index_rows(rows[split], split, args.shard_size)),
        )
    atomic_text(
        output_dir / "fbank_cache_config_v1.json",
        json.dumps(config(args, manifests), indent=2, sort_keys=True) + "\n",
    )
    duration = time.perf_counter() - started
    cache = validate_cache(output_dir, rows, args.shard_size)
    cache.update(compare_cached_embeddings(
        frontend, dataset_root, output_dir, rows, args.shard_size, args.batch_size
    ))
    atomic_text(args.report.resolve(), render_report(args, preflight, cache, duration))
    print(render_report(args, preflight, cache, duration))


if __name__ == "__main__":
    main()
