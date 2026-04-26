"""Unit tests for the Java test parser — no LLM, no API needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from semantic_rts.kb.test_parser import (
    TestMethod,
    _extract_method_source,
    discover_test_files,
    parse_test_methods,
)

MINI_PROJECT = Path(__file__).parent.parent / "fixtures" / "mini_project"


class TestDiscoverTestFiles:
    def test_finds_all_five_java_files(self):
        files = discover_test_files(MINI_PROJECT)
        assert len(files) == 5

    def test_all_under_test_directory(self):
        files = discover_test_files(MINI_PROJECT)
        for f in files:
            parts = f.relative_to(MINI_PROJECT).parts
            assert any(p.lower() in {"test", "tests"} for p in parts)

    def test_sorted(self):
        files = discover_test_files(MINI_PROJECT)
        assert files == sorted(files)

    def test_empty_dir_returns_nothing(self, tmp_path):
        assert discover_test_files(tmp_path) == []

    def test_non_test_dir_excluded(self, tmp_path):
        # File in src/main should NOT be returned
        main_dir = tmp_path / "src" / "main" / "java"
        main_dir.mkdir(parents=True)
        (main_dir / "Foo.java").write_text("public class Foo {}")
        assert discover_test_files(tmp_path) == []


class TestParseTestMethods:
    def test_all_five_tests_discovered(self):
        files = discover_test_files(MINI_PROJECT)
        methods = parse_test_methods(files, MINI_PROJECT)
        assert len(methods) == 5

    def test_test_ids_use_double_colon(self):
        files = discover_test_files(MINI_PROJECT)
        methods = parse_test_methods(files, MINI_PROJECT)
        for m in methods:
            assert "::" in m.test_id

    def test_junit_version_is_4(self):
        files = discover_test_files(MINI_PROJECT)
        methods = parse_test_methods(files, MINI_PROJECT)
        for m in methods:
            assert m.junit == "4"

    def test_source_hash_is_hex(self):
        files = discover_test_files(MINI_PROJECT)
        methods = parse_test_methods(files, MINI_PROJECT)
        for m in methods:
            assert len(m.source_hash) == 40  # SHA-1 hex

    def test_each_class_present(self):
        files = discover_test_files(MINI_PROJECT)
        methods = parse_test_methods(files, MINI_PROJECT)
        simple_names = {m.class_simple for m in methods}
        expected = {"AuthTest", "DatabaseTest", "UserServiceTest", "StringUtilTest", "ModelTest"}
        assert simple_names == expected

    def test_source_not_empty(self):
        files = discover_test_files(MINI_PROJECT)
        methods = parse_test_methods(files, MINI_PROJECT)
        for m in methods:
            assert len(m.source) > 10

    def test_file_path_relative(self):
        files = discover_test_files(MINI_PROJECT)
        methods = parse_test_methods(files, MINI_PROJECT)
        for m in methods:
            assert not Path(m.file_path).is_absolute()

    def test_bad_file_logged_not_raised(self, tmp_path):
        bad = tmp_path / "src" / "test" / "java" / "Bad.java"
        bad.parent.mkdir(parents=True)
        bad.write_text("not valid java }{}{")
        methods = parse_test_methods([bad], tmp_path)
        # Should return empty list, not raise
        assert methods == []


class TestExtractMethodSource:
    def test_simple_method(self):
        lines = [
            "@Test",
            "public void testFoo() {",
            "    assertEquals(1, 1);",
            "}",
            "// after",
        ]
        src = _extract_method_source(lines, 0)
        assert "testFoo" in src
        assert "assertEquals" in src
        assert "// after" not in src

    def test_nested_braces(self):
        lines = [
            "public void testIf() {",
            "    if (true) {",
            "        doSomething();",
            "    }",
            "}",
            "public void next() {}",
        ]
        src = _extract_method_source(lines, 0)
        assert "doSomething" in src
        assert "next" not in src

    def test_string_with_braces_not_counted(self):
        lines = [
            'public void testStr() {',
            '    String s = "{ not a brace }";',
            '    assertEquals("}", s.charAt(0));',
            '}',
        ]
        src = _extract_method_source(lines, 0)
        assert "testStr" in src
        # Should end at the final } of the method
        assert src.strip().endswith("}")
