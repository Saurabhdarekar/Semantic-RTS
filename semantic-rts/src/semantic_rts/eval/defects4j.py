"""Defects4J integration: checkout, patch extraction, ground-truth test lists."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BugInfo:
    project: str
    bug_id: int
    diff: str                   # unified diff (buggy → fixed)
    failing_tests: list[str]    # trigger tests (ground truth)
    all_tests: list[str]        # all tests from fixed checkout
    fixed_dir: Path = field(default_factory=Path)
    buggy_dir: Path = field(default_factory=Path)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

class Defects4JError(RuntimeError):
    """Raised when a defects4j command fails."""


class DeprecatedBugError(Defects4JError):
    """Raised when the requested bug ID is deprecated in this Defects4J version."""


def _run(args: list[str], cwd: Path | None = None, check: bool = True) -> str:
    """Run a shell command and return stdout. Raises Defects4JError on failure."""
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
        )
    except FileNotFoundError as exc:
        raise Defects4JError(f"Command not found: {args[0]}") from exc

    if check and result.returncode != 0:
        stderr = result.stderr[:500]
        if "deprecated bug" in stderr.lower():
            raise DeprecatedBugError(f"Deprecated bug: {' '.join(args)}")
        raise Defects4JError(
            f"Command failed (exit {result.returncode}): {' '.join(args)}\n"
            f"stderr: {stderr}"
        )
    return result.stdout.strip()


def _d4j(args: list[str], cwd: Path | None = None) -> str:
    """Run a defects4j subcommand and return stdout."""
    return _run(["defects4j"] + args, cwd=cwd)


# ---------------------------------------------------------------------------
# Checkout
# ---------------------------------------------------------------------------

def checkout(project: str, bug_id: int, version: str, target: Path) -> None:
    """Checkout a Defects4J bug version to *target*.

    version: "b" for buggy, "f" for fixed.
    Skips silently if *target* already contains files.
    """
    if target.exists() and any(target.iterdir()):
        logger.debug("Checkout already exists, skipping: %s", target)
        return
    target.mkdir(parents=True, exist_ok=True)
    logger.info("defects4j checkout %s-%d%s → %s", project, bug_id, version, target)
    _d4j(["checkout", "-p", project, "-v", f"{bug_id}{version}", "-w", str(target)])


# ---------------------------------------------------------------------------
# Diff extraction
# ---------------------------------------------------------------------------

def get_patch_from_framework(d4j_home: Path, project: str, bug_id: int) -> str | None:
    """Read and combine .src.patch + .test.patch from the D4J framework directory.

    D4J stores pre-generated patches at:
      $D4J_HOME/framework/projects/<project>/patches/<bug_id>.src.patch
      $D4J_HOME/framework/projects/<project>/patches/<bug_id>.test.patch

    Both are combined so that test files added alongside the fix are visible
    to the test_in_diff bypass in Phase 2.
    """
    patches_dir = d4j_home / "framework" / "projects" / project / "patches"
    src_patch = patches_dir / f"{bug_id}.src.patch"
    test_patch = patches_dir / f"{bug_id}.test.patch"

    parts = []
    if src_patch.exists():
        parts.append(src_patch.read_text(encoding="utf-8", errors="replace"))
    if test_patch.exists():
        parts.append(test_patch.read_text(encoding="utf-8", errors="replace"))

    if parts:
        return "\n".join(parts)
    logger.debug("No framework patches found for %s-%d", project, bug_id)
    return None


def get_diff_from_git(fixed_dir: Path) -> str:
    """Get the bug-fixing diff by comparing the last two commits in the fixed checkout."""
    diff = _run(
        ["git", "diff", "HEAD~1", "HEAD", "--", "*.java"],
        cwd=fixed_dir,
        check=False,
    )
    if not diff:
        logger.warning("git diff returned empty for %s", fixed_dir)
    return diff


def get_diff(
    d4j_home: Path | None,
    project: str,
    bug_id: int,
    fixed_dir: Path,
) -> str:
    """Get the bug-fixing unified diff.

    Strategy (in order):
    1. Framework patch file ($D4J_HOME/framework/projects/<project>/patches/<id>.src.patch)
    2. Git diff of the last two commits in the fixed checkout
    """
    if d4j_home:
        patch = get_patch_from_framework(d4j_home, project, bug_id)
        if patch:
            return patch
    return get_diff_from_git(fixed_dir)


# ---------------------------------------------------------------------------
# Test list extraction
# ---------------------------------------------------------------------------

def _parse_test_list(output: str) -> list[str]:
    """Parse newline-separated defects4j export output into a list of test IDs."""
    tests: list[str] = []
    for line in output.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tests.append(line)
    return tests


def get_failing_tests(project_dir: Path) -> list[str]:
    """Return the trigger (failing) tests via defects4j export -p tests.trigger."""
    out = _d4j(["export", "-p", "tests.trigger"], cwd=project_dir)
    return _parse_test_list(out)


def get_all_tests(project_dir: Path) -> list[str]:
    """Return all test IDs via defects4j export -p tests.all."""
    out = _d4j(["export", "-p", "tests.all"], cwd=project_dir)
    return _parse_test_list(out)


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------

def load_bug(
    project: str,
    bug_id: int,
    work_dir: Path,
    d4j_home: Path | None = None,
) -> BugInfo:
    """Checkout both versions of a bug and return a BugInfo.

    Checkouts are cached under *work_dir*; re-running skips existing directories.

    Args:
        project:   Defects4J project name, e.g. "Chart".
        bug_id:    Bug number, e.g. 1.
        work_dir:  Root directory for bug checkouts.
        d4j_home:  Path to the Defects4J installation (optional; enables framework patches).
    """
    buggy_dir = work_dir / f"{project}_{bug_id}_buggy"
    fixed_dir = work_dir / f"{project}_{bug_id}_fixed"

    checkout(project, bug_id, "b", buggy_dir)
    checkout(project, bug_id, "f", fixed_dir)

    diff = get_diff(d4j_home, project, bug_id, fixed_dir)
    failing = get_failing_tests(buggy_dir)
    all_tests = get_all_tests(fixed_dir)

    return BugInfo(
        project=project,
        bug_id=bug_id,
        diff=diff,
        failing_tests=failing,
        all_tests=all_tests,
        fixed_dir=fixed_dir,
        buggy_dir=buggy_dir,
    )
