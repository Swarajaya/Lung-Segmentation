"""Backward-compatible alias for scripts/train.py (kept so older documented
commands and notebooks keep working). All logic lives in scripts/train.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from train import main  # noqa: E402

if __name__ == "__main__":
    main()
