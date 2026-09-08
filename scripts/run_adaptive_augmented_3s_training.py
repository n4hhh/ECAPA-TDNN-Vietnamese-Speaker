"""CLI for adaptive_augmented_3s_v1 ECAPA/AAM fine-tuning."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adaptive_augmented_3s_training import main


if __name__ == "__main__":
    raise SystemExit(main())
