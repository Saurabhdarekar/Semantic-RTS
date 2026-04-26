"""Unit tests for M7 baseline methods."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from semantic_rts.baselines.retest_all import select_all
from semantic_rts.baselines.starts_runner import select_static_fallback, select_starts
from semantic_rts.eval.metrics import BugMetrics

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

ALL_TESTS = [
    "org.example.AuthTest::testLogin",
    "org.example.AuthTest::testLogout",
    "org.example.DatabaseTest::testConnect",
    "org.example.UserServiceTest::testCreateUser",
    "org.example.ModelTest::testGetterSetter",
]
FAILING = ["org.example.AuthTest::testLogin"]

AUTH_DIFF = """\
--- a/src/main/java/org/example/Auth.java
+++ b/src/main/java/org/example/Auth.java
@@ -5,3 +5,4 @@ public class Auth {
     public boolean validate(String pw) {
-        return pw.length() >= 8;
+        // stronger check
+        return pw.length() >= 12;
     }
"""

MULTI_FILE_DIFF = """\
--- a/src/main/java/org/example/Auth.java
+++ b/src/main/java/org/example/Auth.java
@@ -6,1 +6,1 @@
-        return pw.length() >= 8;
+        return pw.length() >= 12;
--- a/src/main/java/org/example/Database.java
+++ b/src/main/java/org/example/Database.java
@@ -5,1 +5,1 @@
-        conn.setAutoCommit(true);
+        conn.setAutoCommit(false);
"""


# ---------------------------------------------------------------------------
# select_all (Retest-All)
# ---------------------------------------------------------------------------

def test_retest_all_returns_everything():
    tests = ["a::m1", "b::m2", "c::m3"]
    assert select_all(tests) == tests


def test_select_all_returns_copy():
    tests = ["a::m1"]
    assert select_all(tests) is not tests


def test_select_all_empty():
    assert select_all([]) == []


def test_select_all_recall_is_one():
    from semantic_rts.eval.metrics import compute_metrics
    selected = select_all(ALL_TESTS)
    m = compute_metrics(selected, FAILING, ALL_TESTS, "retest_all", "Chart", 1)
    assert m.recall == pytest.approx(1.0)
    assert m.selection_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# select_static_fallback (file-level static)
# ---------------------------------------------------------------------------

def test_static_fallback_matches_by_name():
    diff_files = ["src/main/java/org/jfree/chart/ChartFactory.java"]
    all_tests = [
        "org.jfree.chart.ChartFactoryTest::testCreate",
        "org.jfree.chart.AxisTest::testRange",
    ]
    selected = select_static_fallback(diff_files, all_tests)
    assert "org.jfree.chart.ChartFactoryTest::testCreate" in selected
    assert "org.jfree.chart.AxisTest::testRange" not in selected


def test_static_fallback_case_insensitive():
    selected = select_static_fallback(
        ["src/main/java/org/example/Auth.java"],
        ["org.example.AUTHTEST::testFoo"],
    )
    assert "org.example.AUTHTEST::testFoo" in selected


def test_static_fallback_multi_file():
    diff_files = [
        "src/main/java/org/example/Auth.java",
        "src/main/java/org/example/Database.java",
    ]
    selected = select_static_fallback(diff_files, ALL_TESTS)
    assert "org.example.AuthTest::testLogin" in selected
    assert "org.example.DatabaseTest::testConnect" in selected
    assert "org.example.ModelTest::testGetterSetter" not in selected


def test_static_fallback_no_java_files_returns_all():
    selected = select_static_fallback(["README.md"], ALL_TESTS)
    assert set(selected) == set(ALL_TESTS)


def test_static_fallback_empty_diff_files_returns_all():
    selected = select_static_fallback([], ALL_TESTS)
    assert set(selected) == set(ALL_TESTS)


def test_static_fallback_no_match_returns_empty():
    selected = select_static_fallback(
        ["src/main/java/org/example/Auth.java"],
        ["org.example.PaymentTest::testCharge"],
    )
    assert selected == []


def test_static_fallback_test_id_without_method():
    selected = select_static_fallback(
        ["src/main/java/org/example/Auth.java"],
        ["org.example.AuthTest"],
    )
    assert "org.example.AuthTest" in selected


def test_static_fallback_recall_on_auth_change():
    from semantic_rts.eval.metrics import compute_metrics
    from semantic_rts.impact.diff_parser import parse_unified_diff
    parsed = parse_unified_diff(AUTH_DIFF)
    selected = select_static_fallback(parsed.files_changed, ALL_TESTS)
    m = compute_metrics(selected, FAILING, ALL_TESTS, "file_level_static", "Chart", 1)
    assert m.recall == pytest.approx(1.0)
    assert m.selection_rate < 1.0


# ---------------------------------------------------------------------------
# select_starts (delegates to static fallback)
# ---------------------------------------------------------------------------

def test_select_starts_delegates_to_static_fallback():
    diff_files = ["src/main/java/org/example/Auth.java"]
    result_starts = select_starts(diff_files, ALL_TESTS)
    result_static = select_static_fallback(diff_files, ALL_TESTS)
    assert result_starts == result_static


# ---------------------------------------------------------------------------
# eval/baselines wrappers
# ---------------------------------------------------------------------------

class TestEvalBaselineWrappers:
    def test_retest_all_wrapper(self):
        from semantic_rts.eval.baselines import retest_all
        assert set(retest_all(ALL_TESTS)) == set(ALL_TESTS)

    def test_file_level_static_wrapper_with_diff(self):
        from semantic_rts.eval.baselines import file_level_static
        selected = file_level_static(AUTH_DIFF, ALL_TESTS)
        assert "org.example.AuthTest::testLogin" in selected
        assert "org.example.ModelTest::testGetterSetter" not in selected

    def test_file_level_static_empty_diff_returns_all(self):
        from semantic_rts.eval.baselines import file_level_static
        assert set(file_level_static("", ALL_TESTS)) == set(ALL_TESTS)


# ---------------------------------------------------------------------------
# run_baseline_bug
# ---------------------------------------------------------------------------

class TestRunBaselineBug:
    def _make_bug(self, diff=AUTH_DIFF, failing=None, all_tests=None):
        from semantic_rts.eval.defects4j import BugInfo
        return BugInfo(
            project="Chart", bug_id=1, diff=diff,
            failing_tests=failing or FAILING,
            all_tests=all_tests or ALL_TESTS,
            fixed_dir=Path("."), buggy_dir=Path("."),
        )

    def test_retest_all_gives_full_recall(self, tmp_path):
        from semantic_rts.eval.baselines import run_baseline_bug
        with patch("semantic_rts.eval.baselines.load_bug", return_value=self._make_bug()):
            m = run_baseline_bug("Chart", 1, "retest_all", tmp_path)
        assert m.recall == pytest.approx(1.0)
        assert m.selection_rate == pytest.approx(1.0)
        assert m.method == "retest_all"

    def test_file_level_static_partial_selection(self, tmp_path):
        from semantic_rts.eval.baselines import run_baseline_bug
        with patch("semantic_rts.eval.baselines.load_bug", return_value=self._make_bug()):
            m = run_baseline_bug("Chart", 1, "file_level_static", tmp_path)
        assert m.recall == pytest.approx(1.0)   # AuthTest selected
        assert m.selection_rate < 1.0            # not all tests
        assert m.method == "file_level_static"

    def test_invalid_method_raises(self, tmp_path):
        from semantic_rts.eval.baselines import run_baseline_bug
        with patch("semantic_rts.eval.baselines.load_bug", return_value=self._make_bug()):
            with pytest.raises(ValueError, match="Unknown baseline method"):
                run_baseline_bug("Chart", 1, "bad_method", tmp_path)  # type: ignore

    def test_latency_recorded(self, tmp_path):
        from semantic_rts.eval.baselines import run_baseline_bug
        with patch("semantic_rts.eval.baselines.load_bug", return_value=self._make_bug()):
            m = run_baseline_bug("Chart", 1, "retest_all", tmp_path)
        assert m.latency_ms >= 0


# ---------------------------------------------------------------------------
# run_baseline_eval
# ---------------------------------------------------------------------------

class TestRunBaselineEval:
    def _make_metrics(self, bug_id=1, method="retest_all"):
        return BugMetrics(
            project="Chart", bug_id=bug_id, method=method,
            recall=1.0, selection_rate=1.0, precision=1.0,
            latency_ms=1.0, cost_usd=0.0,
            n_failing=1, n_selected=5, n_total=5,
        )

    def test_processes_all_bugs(self, tmp_path):
        from semantic_rts.eval.baselines import run_baseline_eval
        processed = []

        def fake(project, bug_id, method, work_dir, d4j_home=None):
            processed.append(bug_id)
            return self._make_metrics(bug_id)

        with patch("semantic_rts.eval.baselines.run_baseline_bug", side_effect=fake):
            results = run_baseline_eval("Chart", [1, 2, 3], "retest_all", tmp_path)

        assert len(results) == 3
        assert processed == [1, 2, 3]

    def test_skips_failed_bugs(self, tmp_path):
        from semantic_rts.eval.baselines import run_baseline_eval

        def fake(project, bug_id, method, work_dir, d4j_home=None):
            if bug_id == 2:
                raise RuntimeError("checkout failed")
            return self._make_metrics(bug_id)

        with patch("semantic_rts.eval.baselines.run_baseline_bug", side_effect=fake):
            results = run_baseline_eval("Chart", [1, 2, 3], "retest_all", tmp_path)

        assert len(results) == 2
        assert all(r.bug_id != 2 for r in results)

    def test_writes_csv(self, tmp_path):
        from semantic_rts.eval.baselines import run_baseline_eval
        with patch("semantic_rts.eval.baselines.run_baseline_bug", return_value=self._make_metrics()):
            run_baseline_eval("Chart", [1], "retest_all", tmp_path, output_dir=tmp_path)
        assert (tmp_path / "Chart_retest_all_results.csv").exists()

    def test_no_csv_when_output_dir_none(self, tmp_path):
        from semantic_rts.eval.baselines import run_baseline_eval
        with patch("semantic_rts.eval.baselines.run_baseline_bug", return_value=self._make_metrics()):
            run_baseline_eval("Chart", [1], "retest_all", tmp_path, output_dir=None)
        assert not any(tmp_path.rglob("*.csv"))

    def test_file_level_static_csv_named_correctly(self, tmp_path):
        from semantic_rts.eval.baselines import run_baseline_eval
        with patch("semantic_rts.eval.baselines.run_baseline_bug",
                   return_value=self._make_metrics(method="file_level_static")):
            run_baseline_eval("Chart", [1], "file_level_static", tmp_path, output_dir=tmp_path)
        assert (tmp_path / "Chart_file_level_static_results.csv").exists()
