#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class ProjectResult:
    project: str
    num_bugs: int | None
    final_fixed_version: str | None
    tests_all_count: int | None
    workdir: str
    status: str
    error: str


def _run(cmd: list[str], *, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _parse_num_bugs(info_stdout: str) -> int | None:
    # Example line: "Number of bugs: 106"
    m = re.search(r"^Number of bugs:\s*(\d+)\s*$", info_stdout, flags=re.MULTILINE)
    if not m:
        return None
    return int(m.group(1))


def defects4j_info(project: str) -> int | None:
    p = _run(["defects4j", "info", "-p", project])
    if p.returncode != 0:
        return None
    return _parse_num_bugs(p.stdout)


def defects4j_checkout(project: str, version: str, workdir: Path) -> tuple[bool, str]:
    workdir.parent.mkdir(parents=True, exist_ok=True)
    if workdir.exists() and any(workdir.iterdir()):
        return True, "already-exists"

    p = _run(["defects4j", "checkout", "-p", project, "-v", version, "-w", str(workdir)])
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "checkout failed").strip()
        return False, err
    return True, "checked-out"


def defects4j_tests_all_count(workdir: Path) -> tuple[int | None, str]:
    p = _run(["defects4j", "export", "-p", "tests.all", "-w", str(workdir)])
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "export failed").strip()
        return None, err

    tests = [line for line in p.stdout.splitlines() if line.strip()]
    return len(tests), "ok"


def load_projects_from_config(config_path: Path) -> list[str]:
    cfg = yaml.safe_load(config_path.read_text())
    projects = cfg.get("projects", [])
    names: list[str] = []
    for item in projects:
        name = item.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    # de-dupe while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            result.append(n)
    return result


def build_csv(projects: Iterable[str], base_workdir: Path) -> list[ProjectResult]:
    results: list[ProjectResult] = []

    for project in projects:
        num_bugs = defects4j_info(project)
        if not num_bugs:
            results.append(
                ProjectResult(
                    project=project,
                    num_bugs=None,
                    final_fixed_version=None,
                    tests_all_count=None,
                    workdir=str(base_workdir / project),
                    status="error",
                    error="Could not determine number of bugs (defects4j info failed or unexpected output)",
                )
            )
            continue

        final_fixed_version = f"{num_bugs}f"
        workdir = base_workdir / f"{project}-{final_fixed_version}"

        ok, msg = defects4j_checkout(project, final_fixed_version, workdir)
        if not ok:
            results.append(
                ProjectResult(
                    project=project,
                    num_bugs=num_bugs,
                    final_fixed_version=final_fixed_version,
                    tests_all_count=None,
                    workdir=str(workdir),
                    status="error",
                    error=f"checkout: {msg}",
                )
            )
            continue

        tests_count, tests_msg = defects4j_tests_all_count(workdir)
        if tests_count is None:
            results.append(
                ProjectResult(
                    project=project,
                    num_bugs=num_bugs,
                    final_fixed_version=final_fixed_version,
                    tests_all_count=None,
                    workdir=str(workdir),
                    status="error",
                    error=f"tests.all export: {tests_msg}",
                )
            )
            continue

        results.append(
            ProjectResult(
                project=project,
                num_bugs=num_bugs,
                final_fixed_version=final_fixed_version,
                tests_all_count=tests_count,
                workdir=str(workdir),
                status="ok",
                error="",
            )
        )

    return results


def write_csv(out_path: Path, results: list[ProjectResult]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "project",
                "num_bugs",
                "final_fixed_version",
                "tests_all_count",
                "workdir",
                "status",
                "error",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.project,
                    r.num_bugs if r.num_bugs is not None else "",
                    r.final_fixed_version or "",
                    r.tests_all_count if r.tests_all_count is not None else "",
                    r.workdir,
                    r.status,
                    r.error,
                ]
            )


def main(argv: list[str]) -> int:
    if shutil.which("defects4j") is None:
        print("ERROR: defects4j not found on PATH", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    default_config = repo_root / "config" / "projects.yaml"
    default_out = repo_root / "data" / "eval" / "defects4j_final_fixed_test_counts.csv"
    default_workdir = Path(os.environ.get("D4J_WORKDIR", "/tmp/d4j_final_fixed_versions"))

    ap = argparse.ArgumentParser(
        description=(
            "Generate a CSV with (a) total Defects4J bug count per project and "
            "(b) number of tests in the project’s latest fixed version (Nf)."
        )
    )
    ap.add_argument("--config", type=Path, default=default_config, help="Path to projects.yaml")
    ap.add_argument("--out", type=Path, default=default_out, help="Output CSV path")
    ap.add_argument(
        "--workdir",
        type=Path,
        default=default_workdir,
        help="Base directory used for Defects4J checkouts (cached)",
    )
    ap.add_argument(
        "--projects",
        nargs="*",
        default=None,
        help="Optional explicit project IDs (overrides config)",
    )

    args = ap.parse_args(argv)

    if args.projects:
        projects = args.projects
    else:
        if not args.config.exists():
            print(f"ERROR: config not found: {args.config}", file=sys.stderr)
            return 2
        projects = load_projects_from_config(args.config)

    if not projects:
        print("ERROR: no projects specified", file=sys.stderr)
        return 2

    results = build_csv(projects, args.workdir)
    write_csv(args.out, results)

    ok = sum(1 for r in results if r.status == "ok")
    err = len(results) - ok
    print(f"Wrote {args.out} (ok={ok}, error={err})")

    if err:
        print("Errors:")
        for r in results:
            if r.status != "ok":
                print(f"- {r.project}: {r.error}")

    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
