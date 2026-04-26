"""Parse JUnit 4/5 test methods from a Java project using javalang."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import javalang

logger = logging.getLogger(__name__)

# Directories to skip when walking the source tree
_SKIP_DIRS = {"target", "build", "out", ".git", "__pycache__"}

# JUnit 4 test annotations
_JUNIT4_TEST_ANNOTATIONS = {"Test"}
# JUnit 5 test annotations
_JUNIT5_TEST_ANNOTATIONS = {"Test", "ParameterizedTest", "RepeatedTest"}
# Skip-marker annotations
_SKIP_ANNOTATIONS = {"Disabled", "Ignore"}


@dataclass
class TestMethod:
    """One discovered test method, pre-LLM-enrichment."""
    test_id: str           # "pkg.Class::method"
    class_fqn: str
    class_simple: str
    method: str
    file_path: str         # relative to project root
    junit: Literal["4", "5"]
    source: str            # raw method source (may be truncated)
    source_hash: str       # sha1 of *untruncated* source

    # Filled in Phase 1 enrichment
    summary: str = ""
    concepts: list[str] = field(default_factory=list)
    tier: int = 3
    tier_source: Literal["rule", "llm", "default"] = "rule"
    embedding: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_test_files(project_path: Path) -> list[Path]:
    """Walk project tree and return all .java files under test directories, sorted."""
    test_dirs = {"test", "tests"}
    result: list[Path] = []

    for java_file in project_path.rglob("*.java"):
        parts = java_file.relative_to(project_path).parts
        # Skip build output dirs
        if any(p in _SKIP_DIRS for p in parts):
            continue
        # Only include files that live under a "test" or "tests" directory
        if any(p.lower() in test_dirs for p in parts[:-1]):
            result.append(java_file)

    return sorted(result)


def parse_test_methods(files: list[Path], project_root: Path) -> list[TestMethod]:
    """Parse JUnit test methods from the given .java files."""
    methods: list[TestMethod] = []
    for file_path in files:
        try:
            found = _parse_file(file_path, project_root)
            methods.extend(found)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", file_path, exc)
    return methods


# ---------------------------------------------------------------------------
# Internal parsing
# ---------------------------------------------------------------------------

def _detect_junit_version(tree: javalang.tree.CompilationUnit) -> Literal["4", "5", "unknown"]:
    imports = [imp.path for imp in (tree.imports or [])]
    if any("org.junit.jupiter" in p for p in imports):
        return "5"
    if any("org.junit" in p for p in imports):
        return "4"
    return "unknown"


def _parse_file(file_path: Path, project_root: Path) -> list[TestMethod]:
    source_text = file_path.read_text(encoding="utf-8", errors="replace")
    source_lines = source_text.splitlines()

    try:
        tree = javalang.parse.parse(source_text)
    except Exception as exc:
        logger.warning("javalang parse error in %s: %s", file_path, exc)
        return []

    package = tree.package.name if tree.package else ""
    junit_version = _detect_junit_version(tree)
    valid_test_anns = _JUNIT5_TEST_ANNOTATIONS if junit_version == "5" else _JUNIT4_TEST_ANNOTATIONS

    methods: list[TestMethod] = []

    for type_decl in (tree.types or []):
        if not isinstance(type_decl, javalang.tree.ClassDeclaration):
            continue

        # Skip abstract classes — concrete subclasses will be parsed
        if "abstract" in (type_decl.modifiers or set()):
            continue

        class_name = type_decl.name
        class_fqn = f"{package}.{class_name}" if package else class_name

        for member in (type_decl.body or []):
            if not isinstance(member, javalang.tree.MethodDeclaration):
                continue

            annotation_names = {ann.name for ann in (member.annotations or [])}

            # Must have at least one test annotation
            if not (annotation_names & valid_test_anns):
                continue

            # Skip disabled/ignored
            if annotation_names & _SKIP_ANNOTATIONS:
                continue

            # Skip private or static
            modifiers = set(member.modifiers or [])
            if "private" in modifiers or "static" in modifiers:
                continue

            # Find earliest line (method decl or its annotations)
            start_line = _earliest_line(member) - 1   # convert to 0-indexed

            raw_source = _extract_method_source(source_lines, start_line)
            source_hash = hashlib.sha1(
                raw_source.encode("utf-8"), usedforsecurity=False
            ).hexdigest()

            # Truncate for LLM (rough 4-chars-per-token estimate)
            max_chars = 6000   # ~1500 tokens
            truncated = raw_source[:max_chars] + "\n// [truncated]" if len(raw_source) > max_chars else raw_source

            # Prefer the detected version; fall back to "4"
            junit_str: Literal["4", "5"] = "5" if junit_version == "5" else "4"

            test_id = f"{class_fqn}::{member.name}"
            methods.append(TestMethod(
                test_id=test_id,
                class_fqn=class_fqn,
                class_simple=class_name,
                method=member.name,
                file_path=str(file_path.relative_to(project_root)),
                junit=junit_str,
                source=truncated,
                source_hash=source_hash,
            ))

    return methods


def _earliest_line(method: javalang.tree.MethodDeclaration) -> int:
    """Return 1-indexed line number for the earliest part of this method (incl. annotations)."""
    line = method.position.line if method.position else 1
    for ann in (method.annotations or []):
        if ann.position and ann.position.line < line:
            line = ann.position.line
    return line


def _extract_method_source(source_lines: list[str], start_line: int) -> str:
    """Extract complete method source from start_line (0-indexed) by brace counting."""
    depth = 0
    in_body = False
    collected: list[str] = []

    for i in range(start_line, len(source_lines)):
        line = source_lines[i]
        collected.append(line)

        in_string = False
        in_char = False
        skip_next = False

        for j, ch in enumerate(line):
            if skip_next:
                skip_next = False
                continue
            if ch == "\\" and (in_string or in_char):
                skip_next = True
                continue
            if ch == '"' and not in_char:
                in_string = not in_string
            elif ch == "'" and not in_string:
                in_char = not in_char
            elif not in_string and not in_char:
                if ch == "{":
                    depth += 1
                    in_body = True
                elif ch == "}":
                    depth -= 1

        if in_body and depth == 0:
            break

    return "\n".join(collected)
