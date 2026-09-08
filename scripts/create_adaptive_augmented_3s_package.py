"""CLI for the adaptive_augmented_3s production dataset package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive_augmented_3s_package import DEFAULT_SEED, create_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and verify the adaptive_augmented_3s dataset package."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = create_package(args.dataset_root, PROJECT_ROOT, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
