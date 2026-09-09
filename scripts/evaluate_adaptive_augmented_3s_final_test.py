"""CLI wrapper for the locked one-shot final-test evaluator."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.adaptive_augmented_3s_final_evaluation import main
if __name__ == "__main__":
    main()
