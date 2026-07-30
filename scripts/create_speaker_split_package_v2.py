"""Create the approved VieSpeaker2.0 final speaker-disjoint split package v2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.speaker_split_v2 import (
    FULL_MANIFEST_IDENTITY_RELATIVE_PATH,
    FULL_MANIFEST_RELATIVE_PATH,
    create_split_package,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-manifest",
        type=Path,
        default=Path(FULL_MANIFEST_RELATIVE_PATH),
    )
    parser.add_argument(
        "--full-manifest-identity",
        type=Path,
        default=Path(FULL_MANIFEST_IDENTITY_RELATIVE_PATH),
    )
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--reference-root", type=Path)
    return parser.parse_args()


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    args = parse_args()
    result = create_split_package(
        project_root=PROJECT_ROOT,
        manifest_path=project_path(args.full_manifest),
        identity_path=project_path(args.full_manifest_identity),
        output_root=project_path(args.output_root),
        reference_root=(
            project_path(args.reference_root) if args.reference_root else None
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
