"""Parse unified diffs into structured change records using unidiff + javalang."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import javalang
import unidiff

logger = logging.getLogger(__name__)

# Matches Java method declaration lines in unified diff output (added or removed)
_SIG_CHANGE_RE = re.compile(
    r'^[+-]\s*(public|protected|private|static)\s+\w[\w<>\[\]]*\s+\w+\s*\(',
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HunkRange:
    target_start: int   # 1-indexed line in the post-change file
    target_end: int     # inclusive


@dataclass
class FileChange:
    path: str                            # normalised (no a/ or b/ prefix)
    hunks: list[HunkRange] = field(default_factory=list)
    is_new_file: bool = False
    is_deleted_file: bool = False


@dataclass
class ParsedDiff:
    diff_hash: str
    file_changes: list[FileChange]
    files_changed: list[str]            # source files only (non-test Java)
    test_files_changed: list[str]       # test files touched by the diff
    methods_changed: list[str]          # populated by extract_changed_methods
    raw_diff: str
    change_type: str = "general"        # micro_fix | api_change | refactoring | new_behavior | config_change | general


# ---------------------------------------------------------------------------
# Change type classifier
# ---------------------------------------------------------------------------

def classify_change_type(parsed: "ParsedDiff") -> str:
    """Classify a parsed diff using rule-based heuristics. No LLM call."""
    java_changes = [fc for fc in parsed.file_changes if fc.path.endswith(".java")]
    if not java_changes:
        return "config_change"

    has_new_files = any(fc.is_new_file for fc in java_changes)
    has_sig_change = bool(_SIG_CHANGE_RE.search(parsed.raw_diff))

    raw_lines = parsed.raw_diff.splitlines()
    added = sum(1 for l in raw_lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in raw_lines if l.startswith("-") and not l.startswith("---"))
    total_delta = added + removed

    if has_new_files:
        return "new_behavior"
    if has_sig_change:
        return "api_change"
    if total_delta <= 5:
        return "micro_fix"
    if total_delta > 30 and not has_sig_change:
        return "refactoring"
    return "general"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_test_path(path: str) -> bool:
    """Return True if this Java file is a test file."""
    p = path.replace("\\", "/")
    stem = Path(path).stem
    return (
        "/test/" in p or "/tests/" in p
        or stem.endswith("Test") or stem.endswith("Tests")
        or stem.startswith("Test")
    )


def _strip_prefix(path: str) -> str:
    """Strip git a/ or b/ path prefix."""
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _build_method_map(source: str) -> list[tuple[str, str, int]]:
    """Return [(class_name, method_name, start_line_1indexed), ...] sorted by line."""
    methods: list[tuple[str, str, int]] = []
    try:
        tree = javalang.parse.parse(source)
        current_class = "Unknown"
        for path, node in tree:
            if isinstance(node, javalang.tree.ClassDeclaration):
                current_class = node.name
            elif isinstance(node, javalang.tree.MethodDeclaration) and node.position:
                methods.append((current_class, node.name, node.position.line))
    except Exception as exc:
        logger.debug("javalang parse error building method map: %s", exc)
    return sorted(methods, key=lambda t: t[2])


def _method_for_line(method_map: list[tuple[str, str, int]], target_line: int) -> str | None:
    """Return 'ClassName.methodName' for the method that contains target_line."""
    containing: str | None = None
    for cls, meth, start_line in method_map:
        if start_line <= target_line:
            containing = f"{cls}.{meth}"
        else:
            break
    return containing


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_unified_diff(diff_text: str) -> ParsedDiff:
    """Parse a unified diff string into a ParsedDiff."""
    diff_hash = hashlib.sha1(diff_text.encode("utf-8"), usedforsecurity=False).hexdigest()

    try:
        patch = unidiff.PatchSet(diff_text)
    except Exception as exc:
        logger.warning("Failed to parse diff (hash %s): %s", diff_hash[:8], exc)
        return ParsedDiff(
            diff_hash=diff_hash,
            file_changes=[],
            files_changed=[],
            methods_changed=[],
            raw_diff=diff_text,
        )

    file_changes: list[FileChange] = []

    for pf in patch:
        # Prefer target (new) file path; fall back to source for deletions
        raw_path = pf.target_file if not pf.is_removed_file else pf.source_file
        path = _strip_prefix(raw_path)

        if not path.endswith(".java"):
            continue  # skip non-Java files

        hunks: list[HunkRange] = []
        for hunk in pf:
            t_start = hunk.target_start
            t_end = t_start + max(hunk.target_length - 1, 0)
            hunks.append(HunkRange(target_start=t_start, target_end=t_end))

        file_changes.append(FileChange(
            path=path,
            hunks=hunks,
            is_new_file=pf.is_added_file,
            is_deleted_file=pf.is_removed_file,
        ))

    source_files = [fc.path for fc in file_changes if not _is_test_path(fc.path)]
    test_files = [fc.path for fc in file_changes if _is_test_path(fc.path)]
    result = ParsedDiff(
        diff_hash=diff_hash,
        file_changes=file_changes,
        files_changed=source_files,
        test_files_changed=test_files,
        methods_changed=[],
        raw_diff=diff_text,
    )
    result.change_type = classify_change_type(result)
    return result


def extract_changed_methods(
    parsed: ParsedDiff,
    project_root: str | None = None,
) -> list[str]:
    """Return deduplicated list of 'ClassName.methodName' touched by the diff.

    Reads post-change Java files from disk (if project_root is given) to map
    hunk line numbers to method names.  Logs and falls back gracefully if a
    file is missing or unparseable.
    """
    results: list[str] = []

    for fc in parsed.file_changes:
        # New file: report all methods in the added file
        if fc.is_new_file and project_root:
            full_path = Path(project_root) / fc.path
            if full_path.exists():
                source = full_path.read_text(encoding="utf-8", errors="replace")
                method_map = _build_method_map(source)
                results.extend(f"{cls}.{meth}" for cls, meth, _ in method_map)
            continue

        # Deleted file: report class-level placeholder
        if fc.is_deleted_file:
            cls_name = Path(fc.path).stem
            results.append(f"{cls_name}.<deleted>")
            continue

        # Modified file: find which method each hunk falls in
        if project_root:
            full_path = Path(project_root) / fc.path
        else:
            full_path = Path(fc.path)

        if not full_path.exists():
            # Can't read file — use filename as a fallback signal
            cls_name = Path(fc.path).stem
            results.append(f"{cls_name}.<class-level>")
            logger.debug("Post-change file not found on disk: %s", full_path)
            continue

        source = full_path.read_text(encoding="utf-8", errors="replace")
        method_map = _build_method_map(source)

        if not method_map:
            results.append(f"{Path(fc.path).stem}.<class-level>")
            continue

        for hunk in fc.hunks:
            found = _method_for_line(method_map, hunk.target_start)
            if found:
                results.append(found)
            else:
                results.append(f"{Path(fc.path).stem}.<class-level>")

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for item in results:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped
