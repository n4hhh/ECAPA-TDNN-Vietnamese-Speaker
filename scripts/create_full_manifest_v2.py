"""Create and validate the production VieSpeaker2.0 full manifest v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.full_manifest_v2 import (
    CANONICAL_MANIFEST_RELATIVE_PATH,
    create_full_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--audit-inventory", type=Path, required=True)
    parser.add_argument(
        "--audit-report",
        type=Path,
        default=Path("reports/dataset_understanding_v2.md"),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path("reports/dataset_understanding_v2.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("manifests/v2")
    )
    parser.add_argument(
        "--canonical-manifest-relative-path",
        default=CANONICAL_MANIFEST_RELATIVE_PATH,
    )
    parser.add_argument("--reference-dir", type=Path)
    return parser.parse_args()


def project_path(value: Path) -> Path:
    return value if value.is_absolute() else PROJECT_ROOT / value


def main() -> None:
    args = parse_args()
    result = create_full_manifest(
        project_root=PROJECT_ROOT,
        dataset_root=args.dataset_root,
        inventory_path=project_path(args.audit_inventory),
        audit_report_path=project_path(args.audit_report),
        audit_json_path=project_path(args.audit_json),
        output_dir=project_path(args.output_dir),
        canonical_manifest_relative_path=args.canonical_manifest_relative_path,
        reference_dir=(
            project_path(args.reference_dir) if args.reference_dir else None
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
