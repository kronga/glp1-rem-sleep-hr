"""Canonical entry point for the clean unified-cohort Figure 1 builder.

The implementation remains in `_test_fig1_unified.py` because that file holds the reviewed
development history and cache format. This public wrapper keeps the paper build target clear.
"""

from __future__ import annotations

from _test_fig1_unified import main


if __name__ == "__main__":
    raise SystemExit(main())
