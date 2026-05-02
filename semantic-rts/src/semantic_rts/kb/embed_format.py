"""Canonical embedding format shared between Phase 1 (document) and Phase 2 (query).

IMPORTANT: Any change to the output of format_for_embedding() invalidates all existing
FAISS indexes. If you modify this function, force a full KB rebuild with `srts build --force`.
"""

from __future__ import annotations


def format_for_embedding(
    summary: str,
    methods: list[str],
    concepts: list[str],
    risk_areas: list[str] | None = None,
    condition: str | None = None,
    class_simple: str | None = None,
) -> str:
    parts = [f"Behavior: {summary}."]
    if condition:
        parts.append(f"Condition: {condition}.")
    if methods:
        simple_names = [m.split(".")[-1] for m in methods if "." in m]
        if not simple_names:
            simple_names = list(methods)
        parts.append(f"Methods involved: {', '.join(simple_names)}.")
    if concepts:
        parts.append(f"Concepts: {', '.join(concepts)}.")
    if risk_areas:
        parts.append(f"Risk areas: {', '.join(risk_areas)}.")
    if class_simple:
        parts.append(f"Test class: {class_simple}.")
    return " ".join(parts)
