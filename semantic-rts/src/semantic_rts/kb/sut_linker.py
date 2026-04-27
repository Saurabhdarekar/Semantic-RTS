"""Static analysis: link a test method to the production code it exercises.

Two-strategy class resolution:
  1. Same-package scan  — test in pkg X → find all production .java files in pkg X
                          (handles the common case where no imports are needed)
  2. Explicit imports   — for cross-package tests, resolve non-framework imports

No API calls — pure AST + regex parsing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import javalang

if TYPE_CHECKING:
    from semantic_rts.kb.test_parser import TestMethod

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EXCLUDED = {
    # JUnit assertions
    "assertEquals", "assertNotEquals", "assertTrue", "assertFalse",
    "assertNull", "assertNotNull", "assertSame", "assertNotSame",
    "assertArrayEquals", "assertIterableEquals", "assertThrows",
    "assertDoesNotThrow", "assertThat", "assertAll", "fail",
    # JUnit lifecycle
    "setUp", "tearDown", "beforeEach", "afterEach", "beforeAll", "afterAll",
    "before", "after",
    # Java noise
    "println", "print", "printf", "format", "valueOf", "toString",
    "equals", "hashCode", "getClass", "length", "size",
    "get", "set", "add", "put", "remove", "contains",
    "isEmpty", "iterator", "next", "hasNext", "close",
}

# Import prefixes that are never production code
_FRAMEWORK_PREFIXES = (
    "java.", "javax.", "org.junit", "junit.",
    "org.mockito", "org.hamcrest", "org.testng", "org.easymock",
)

_TEST_DIRS = {"test", "tests"}

_CALL_RE = re.compile(r'\b([a-z][a-zA-Z0-9_]*)\s*\(')

# Hard caps to keep prompt size reasonable
_MAX_SIGS_PER_CLASS = 4
_MAX_TOTAL_SIGS = 12


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_sut_context(tm: "TestMethod", project_path: Path) -> str:
    """Return a formatted SUT context block for the summarizer prompt.

    Returns empty string when nothing can be resolved.
    """
    called = _extract_called_method_names(tm.source)
    if not called:
        return ""

    prod_classes = _find_prod_classes(tm, project_path)
    if not prod_classes:
        return ""

    # Collect signatures from each resolved class
    results: dict[str, list[str]] = {}
    total = 0
    for class_name, class_path in prod_classes.items():
        if total >= _MAX_TOTAL_SIGS:
            break
        sigs = _extract_method_signatures(class_path, called, _MAX_SIGS_PER_CLASS)
        if sigs:
            results[class_name] = sigs
            total += len(sigs)

    if not results:
        return ""

    lines = ["Production classes under test:"]
    for class_name, sigs in results.items():
        lines.append(f"  {class_name}:")
        for sig in sigs:
            lines.append(f"    - {sig}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strategy 1: same-package scan
# ---------------------------------------------------------------------------

def _same_package_classes(tm: "TestMethod", project_path: Path) -> dict[str, Path]:
    """Find all non-test .java files whose directory matches the test's package."""
    package = tm.class_fqn.rsplit(".", 1)[0] if "." in tm.class_fqn else ""
    if not package:
        return {}

    pkg_parts = tuple(package.split("."))
    found: dict[str, Path] = {}

    for java_file in project_path.rglob("*.java"):
        if java_file.stem == tm.class_simple:
            continue
        parts = java_file.relative_to(project_path).parts
        if any(p.lower() in _TEST_DIRS for p in parts):
            continue
        dir_parts = parts[:-1]
        if len(dir_parts) >= len(pkg_parts) and dir_parts[-len(pkg_parts):] == pkg_parts:
            found[java_file.stem] = java_file

    return found


# ---------------------------------------------------------------------------
# Strategy 2: explicit imports
# ---------------------------------------------------------------------------

def _import_based_classes(tm: "TestMethod", project_path: Path) -> dict[str, Path]:
    """Resolve non-framework explicit imports to production .java files."""
    test_file = project_path / tm.file_path
    try:
        source = test_file.read_text(encoding="utf-8", errors="replace")
        tree = javalang.parse.parse(source)
        imports = [imp.path for imp in (tree.imports or [])]
    except Exception:
        return {}

    found: dict[str, Path] = {}
    for imp_path in imports:
        if any(imp_path.startswith(p) for p in _FRAMEWORK_PREFIXES):
            continue
        if imp_path.endswith(".*"):
            continue
        class_simple = imp_path.split(".")[-1]
        if class_simple in found:
            continue
        for java_file in project_path.rglob(f"{class_simple}.java"):
            parts = java_file.relative_to(project_path).parts
            if not any(p.lower() in _TEST_DIRS for p in parts):
                found[class_simple] = java_file
                break

    return found


# ---------------------------------------------------------------------------
# Combined resolver
# ---------------------------------------------------------------------------

def _find_prod_classes(tm: "TestMethod", project_path: Path) -> dict[str, Path]:
    """Merge both strategies; same-package first, then fill in from imports."""
    result = _same_package_classes(tm, project_path)
    for name, path in _import_based_classes(tm, project_path).items():
        result.setdefault(name, path)
    return result


# ---------------------------------------------------------------------------
# Signature extraction
# ---------------------------------------------------------------------------

def _extract_called_method_names(source: str) -> set[str]:
    names: set[str] = set()
    for m in _CALL_RE.finditer(source):
        name = m.group(1)
        if name not in _EXCLUDED and not name.startswith("assert"):
            names.add(name)
    return names


def _extract_method_signatures(
    prod_path: Path,
    method_names: set[str],
    cap: int,
) -> list[str]:
    try:
        source = prod_path.read_text(encoding="utf-8", errors="replace")
        tree = javalang.parse.parse(source)
    except Exception:
        return []

    sigs: list[str] = []
    for type_decl in tree.types or []:
        if not isinstance(type_decl, javalang.tree.ClassDeclaration):
            continue
        for member in type_decl.body or []:
            if not isinstance(member, javalang.tree.MethodDeclaration):
                continue
            if member.name not in method_names:
                continue
            if "private" in (member.modifiers or set()):
                continue
            return_type = (
                getattr(member.return_type, "name", "void")
                if member.return_type else "void"
            )
            params = ", ".join(
                f"{getattr(p.type, 'name', '?')} {p.name}"
                for p in (member.parameters or [])
            )
            sigs.append(f"{return_type} {member.name}({params})")
            if len(sigs) >= cap:
                return sigs

    return sigs
