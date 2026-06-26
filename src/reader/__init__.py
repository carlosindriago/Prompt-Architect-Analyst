"""
Reader sub-package.

AbstractReader    - base.py     - protocol all readers must implement
OpenCodeReader    - opencode.py - SQLite reader for opencode.db
"""

from .base import AbstractReader
from .opencode import OpenCodeReader

__all__ = ["AbstractReader", "OpenCodeReader"]
