"""Unit tests for M5: Defects4J integration (subprocess-free via mocks)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import call, patch

import pytest

SAMPLE_PATCH = """\
diff --git a/src/main/java/org/example/Foo.java b/src/main/java/org/example/Foo.java
--- a/src/main/java/org/example/Foo.java
+++ b/src/main/java/org/example/Foo.java
@@ -5,3 +5,4 @@ public class Foo {
     public void bar() {
-        x = 1;
+        // fix
+        x = 2;
     }
"""


# ---------------------------------------------------------------------------
# checkout()
# ---------------------------------------------------------------------------

class TestCheckout:
    def test_skips_existing_non_empty_dir(self, tmp_path):
        from semantic_rts.eval.defects4j import checkout
        target = tmp_path / "existing"
        target.mkdir()
        (target / "dummy.txt").touch()

        with patch("semantic_rts.eval.defects4j._d4j") as mock_d4j:
            checkout("Chart", 1, "f", target)
            mock_d4j.assert_not_called()

    def test_calls_d4j_for_new_dir(self, tmp_path):
        from semantic_rts.eval.defects4j import checkout
        target = tmp_path / "new_checkout"

        with patch("semantic_rts.eval.defects4j._d4j", return_value="") as mock_d4j:
            checkout("Chart", 1, "f", target)
            mock_d4j.assert_called_once()
            cmd_args = mock_d4j.call_args[0][0]
            assert "checkout" in cmd_args
            assert "Chart" in cmd_args
            assert "1f" in cmd_args

    def test_buggy_version_uses_b_suffix(self, tmp_path):
        from semantic_rts.eval.defects4j import checkout
        target = tmp_path / "buggy"

        with patch("semantic_rts.eval.defects4j._d4j", return_value="") as mock_d4j:
            checkout("Lang", 5, "b", target)
            cmd_args = mock_d4j.call_args[0][0]
            assert "5b" in cmd_args

    def test_creates_target_dir(self, tmp_path):
        from semantic_rts.eval.defects4j import checkout
        target = tmp_path / "deep" / "nested" / "dir"

        with patch("semantic_rts.eval.defects4j._d4j", return_value=""):
            checkout("Chart", 1, "f", target)
            assert target.exists()


# ---------------------------------------------------------------------------
# get_patch_from_framework()
# ---------------------------------------------------------------------------

class TestGetPatchFromFramework:
    def test_reads_existing_patch(self, tmp_path):
        from semantic_rts.eval.defects4j import get_patch_from_framework
        patch_dir = tmp_path / "framework" / "projects" / "Chart" / "patches"
        patch_dir.mkdir(parents=True)
        (patch_dir / "1.src.patch").write_text(SAMPLE_PATCH)

        result = get_patch_from_framework(tmp_path, "Chart", 1)
        assert result == SAMPLE_PATCH

    def test_returns_none_when_missing(self, tmp_path):
        from semantic_rts.eval.defects4j import get_patch_from_framework
        result = get_patch_from_framework(tmp_path, "Chart", 99)
        assert result is None

    def test_returns_none_for_wrong_project(self, tmp_path):
        from semantic_rts.eval.defects4j import get_patch_from_framework
        patch_dir = tmp_path / "framework" / "projects" / "Chart" / "patches"
        patch_dir.mkdir(parents=True)
        (patch_dir / "1.src.patch").write_text(SAMPLE_PATCH)

        result = get_patch_from_framework(tmp_path, "Lang", 1)
        assert result is None


# ---------------------------------------------------------------------------
# get_diff()
# ---------------------------------------------------------------------------

class TestGetDiff:
    def test_uses_framework_patch_when_available(self, tmp_path):
        from semantic_rts.eval.defects4j import get_diff
        patch_dir = tmp_path / "framework" / "projects" / "Chart" / "patches"
        patch_dir.mkdir(parents=True)
        (patch_dir / "1.src.patch").write_text(SAMPLE_PATCH)

        with patch("semantic_rts.eval.defects4j.get_diff_from_git") as mock_git:
            result = get_diff(tmp_path, "Chart", 1, tmp_path / "fixed")
            mock_git.assert_not_called()
            assert result == SAMPLE_PATCH

    def test_falls_back_to_git_when_no_framework(self, tmp_path):
        from semantic_rts.eval.defects4j import get_diff
        with patch("semantic_rts.eval.defects4j.get_diff_from_git", return_value="git_diff") as mock_git:
            result = get_diff(None, "Chart", 1, tmp_path)
            mock_git.assert_called_once_with(tmp_path)
            assert result == "git_diff"

    def test_falls_back_to_git_when_patch_missing(self, tmp_path):
        from semantic_rts.eval.defects4j import get_diff
        with patch("semantic_rts.eval.defects4j.get_diff_from_git", return_value="git_diff") as mock_git:
            result = get_diff(tmp_path, "Chart", 99, tmp_path)
            mock_git.assert_called_once()
            assert result == "git_diff"


# ---------------------------------------------------------------------------
# _parse_test_list()
# ---------------------------------------------------------------------------

class TestParseTestList:
    def test_parses_fqn_method_ids(self):
        from semantic_rts.eval.defects4j import _parse_test_list
        out = (
            "org.example.FooTest::testBar\n"
            "org.example.FooTest::testBaz\n"
        )
        assert _parse_test_list(out) == [
            "org.example.FooTest::testBar",
            "org.example.FooTest::testBaz",
        ]

    def test_skips_blank_lines(self):
        from semantic_rts.eval.defects4j import _parse_test_list
        out = "\norg.example.FooTest::testBar\n\n"
        assert _parse_test_list(out) == ["org.example.FooTest::testBar"]

    def test_skips_comment_lines(self):
        from semantic_rts.eval.defects4j import _parse_test_list
        out = "# comment\norg.example.FooTest::testBar\n"
        assert _parse_test_list(out) == ["org.example.FooTest::testBar"]

    def test_empty_output_returns_empty_list(self):
        from semantic_rts.eval.defects4j import _parse_test_list
        assert _parse_test_list("") == []


# ---------------------------------------------------------------------------
# get_failing_tests() / get_all_tests()
# ---------------------------------------------------------------------------

class TestGetTestLists:
    def test_get_failing_tests_calls_export_trigger(self, tmp_path):
        from semantic_rts.eval.defects4j import get_failing_tests
        with patch("semantic_rts.eval.defects4j._d4j", return_value="org.ex.T::m") as mock_d4j:
            result = get_failing_tests(tmp_path)
            mock_d4j.assert_called_once_with(
                ["export", "-p", "tests.trigger"], cwd=tmp_path
            )
            assert result == ["org.ex.T::m"]

    def test_get_all_tests_calls_export_all(self, tmp_path):
        from semantic_rts.eval.defects4j import get_all_tests
        with patch("semantic_rts.eval.defects4j._d4j", return_value="org.ex.T::m\norg.ex.T::n") as mock_d4j:
            result = get_all_tests(tmp_path)
            mock_d4j.assert_called_once_with(
                ["export", "-p", "tests.all"], cwd=tmp_path
            )
            assert result == ["org.ex.T::m", "org.ex.T::n"]


# ---------------------------------------------------------------------------
# load_bug()
# ---------------------------------------------------------------------------

class TestLoadBug:
    def test_orchestrates_checkout_diff_and_tests(self, tmp_path):
        from semantic_rts.eval.defects4j import load_bug

        with (
            patch("semantic_rts.eval.defects4j.checkout") as mock_co,
            patch("semantic_rts.eval.defects4j.get_diff", return_value="diff_text"),
            patch("semantic_rts.eval.defects4j.get_failing_tests", return_value=["T::fail"]),
            patch("semantic_rts.eval.defects4j.get_all_tests", return_value=["T::fail", "T::pass"]),
        ):
            info = load_bug("Chart", 1, tmp_path, d4j_home=None)

        assert mock_co.call_count == 2  # buggy + fixed
        assert info.project == "Chart"
        assert info.bug_id == 1
        assert info.diff == "diff_text"
        assert info.failing_tests == ["T::fail"]
        assert info.all_tests == ["T::fail", "T::pass"]

    def test_checkout_dirs_named_correctly(self, tmp_path):
        from semantic_rts.eval.defects4j import load_bug

        checked_out = []
        with (
            patch("semantic_rts.eval.defects4j.checkout", side_effect=lambda *a, **kw: checked_out.append(a)),
            patch("semantic_rts.eval.defects4j.get_diff", return_value=""),
            patch("semantic_rts.eval.defects4j.get_failing_tests", return_value=[]),
            patch("semantic_rts.eval.defects4j.get_all_tests", return_value=[]),
        ):
            load_bug("Lang", 5, tmp_path)

        dirs = [str(a[3]) for a in checked_out]
        assert any("Lang_5_buggy" in d for d in dirs)
        assert any("Lang_5_fixed" in d for d in dirs)

    def test_passes_d4j_home_to_get_diff(self, tmp_path):
        from semantic_rts.eval.defects4j import load_bug
        d4j_home = tmp_path / "d4j"

        with (
            patch("semantic_rts.eval.defects4j.checkout"),
            patch("semantic_rts.eval.defects4j.get_diff", return_value="") as mock_diff,
            patch("semantic_rts.eval.defects4j.get_failing_tests", return_value=[]),
            patch("semantic_rts.eval.defects4j.get_all_tests", return_value=[]),
        ):
            load_bug("Chart", 2, tmp_path, d4j_home=d4j_home)
            assert mock_diff.call_args[0][0] == d4j_home


# ---------------------------------------------------------------------------
# Defects4JError
# ---------------------------------------------------------------------------

class TestDefects4JError:
    def test_raised_on_nonzero_exit(self, tmp_path):
        from semantic_rts.eval.defects4j import Defects4JError, _run
        with pytest.raises(Defects4JError, match="Command failed"):
            _run(["python", "-c", "import sys; sys.exit(1)"])

    def test_raised_on_missing_executable(self):
        from semantic_rts.eval.defects4j import Defects4JError, _run
        with pytest.raises(Defects4JError, match="Command not found"):
            _run(["__nonexistent_binary_xyz__"])
