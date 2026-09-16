"""Backward-compatible alias for scripts/evaluate.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
