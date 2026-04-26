"""Retest-All baseline: select every test in the KB."""

from __future__ import annotations


def select_all(all_test_ids: list[str]) -> list[str]:
    """Return every test — the trivial upper bound."""
    return list(all_test_ids)
