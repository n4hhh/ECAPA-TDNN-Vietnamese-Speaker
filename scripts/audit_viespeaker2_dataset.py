"""Run the complete read-only VieSpeaker2.0 Dataset Understanding v2 audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_audit_v2 import scan_dataset, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path("outputs/dataset_understanding_v2"),
    )
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve(strict=True)
    before_children = {path.name for path in dataset_root.iterdir()}
    result = scan_dataset(dataset_root)
    after_children = {path.name for path in dataset_root.iterdir()}
    if before_children != after_children:
        raise RuntimeError("dataset-root entries changed during the audit")
    artifacts = write_outputs(
        result,
        (PROJECT_ROOT / args.runtime_dir).resolve(),
        (PROJECT_ROOT / args.report_dir).resolve(),
    )
    print(
        json.dumps(
            {
                "result": result["result"],
                "counts": result["counts"],
                "artifacts": artifacts,
                "preservation_snapshots_match": result["preservation"][
                    "snapshots_match"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if result["result"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
