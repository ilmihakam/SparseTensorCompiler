#!/usr/bin/env python3
"""Generate configurable synthetic sparse matrices for unified benchmarks."""

import sys
from pathlib import Path

# Ensure the utils directory is on sys.path so matrix_generator can find siblings
_UTILS_DIR = str(Path(__file__).resolve().parent)
if _UTILS_DIR not in sys.path:
    sys.path.insert(0, _UTILS_DIR)

from matrix_generator import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
