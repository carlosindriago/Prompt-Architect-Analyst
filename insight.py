#!/usr/bin/env python3
"""
prompt-architect-analyst — standalone entry point.

Usage:
    python3 insight.py [options]
    ./insight.py [options]       # after chmod +x insight.py

No pip install required. Delegates to src.cli:main.
"""

import os
import sys

# Ensure the project root is on the path so `src` is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
