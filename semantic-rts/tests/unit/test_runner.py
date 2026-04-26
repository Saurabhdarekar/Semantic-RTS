"""Unit tests for M6: eval runner and CSV writer."""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from semantic_rts.eval.metrics import BugMetrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(project="Chart", bug_id=1, recall=1.0, sel_rate=0.1, method="semantic") -> BugMetrics:
    return BugMetrics(
        project=project,
        bug_id=bug_id,
        method=method,
        recall=recall,
        selection_rate=sel_rate,
        precision=1.0,
        latency_ms=500.0,
        cost_usd=0.0,
        n_failing=2,
        n_selected=10,
        n_total=100,
    )


def _make_bug_info(diff="diff_text", failing=None, all_tests=None, fixed_dir=None):
    from semantic_rts.eval.defects4j import BugInfo
    return BugInfo(
        project="Chart",
        bug_id=1,
        diff=diff,
        failing_tests=failing or ["org.ex.T::fail"],
        all_tests=all_tests or ["org.ex.T::fail", "org.ex.T::pass"],
        fixed_dir=fixed_dir or Path("."),
        buggy_dir=Path("."),
    )


# ---------------------------------------------------------------------------
# write_csv / read_csv
# ---------------------------------------------------------------------------

class TestWriteCsv:
    def test_writes_headers_and_rows(self, tmp_path):
        from semantic_rts.eval.runner import write_csv
        rows = [_make_metrics(bug_id=1), _make_metrics(bug_id=2, recall=0.5)]
        path = tmp_path / "out.csv"
        write_csv(rows, path)

        assert path.exists()
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            data = list(reader)
        assert len(data) == 2
        assert data[0]["project"] == "Chart"
        assert data[0]["bug_id"] == "1"
        assert float(data[1]["recall"]) == pytest.approx(0.5)

    def test_creates_parent_dirs(self, tmp_path):
        from semantic_rts.eval.runner import write_csv
        path = tmp_path / "deep" / "nested" / "out.csv"
        write_csv([_make_metrics()], path)
        assert path.exists()

    def test_no_op_on_empty_list(self, tmp_path):
        from semantic_rts.eval.runner import write_csv
        path = tmp_path / "out.csv"
        write_csv([], path)
        assert not path.exists()

    def test_all_fields_present(self, tmp_path):
        from semantic_rts.eval.runner import write_csv
        path = tmp_path / "out.csv"
        m = _make_metrics()
        write_csv([m], path)
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames
        expected = [f.name for f in dataclasses.fields(BugMetrics)]
        assert headers == expected


class TestReadCsv:
    def test_round_trips(self, tmp_path):
        from semantic_rts.eval.runner import read_csv, write_csv
        rows = [_make_metrics(bug_id=1), _make_metrics(bug_id=2)]
        path = tmp_path / "out.csv"
        write_csv(rows, path)
        back = read_csv(path)
        assert len(back) == 2
        assert back[0]["bug_id"] == "1"


# ---------------------------------------------------------------------------
# run_bug()
# ---------------------------------------------------------------------------

class TestRunBug:
    def _mocks(self, diff="diff_text", failing=None, all_tests=None, selected=None):
        """Return a dict of patchers for the full pipeline."""
        from semantic_rts.eval.defects4j import BugInfo
        from semantic_rts.impact.intent_agent import IntentResult
        from semantic_rts.impact.retriever import AnalysisTrace, Candidate
        from semantic_rts.selector.ranker import SelectionTrace, SelectedTest

        bug = _make_bug_info(
            diff=diff,
            failing=failing or ["org.ex.T::fail"],
            all_tests=all_tests or ["org.ex.T::fail", "org.ex.T::pass"],
        )
        intent = IntentResult(
            intent_summary="fix auth", concepts=["auth"], risk_areas=["security"]
        )
        candidate = Candidate(test_id="org.ex.T::fail", score=0.9, tier=1, rank=0)
        trace = AnalysisTrace(
            diff_hash="abc", files_changed=[], methods_changed=[],
            intent_summary="fix auth", intent_concepts=[], risk_areas=[],
            query_text="q", top_k=30, raw_results=[],
        )
        sel_test = SelectedTest(
            test_id=selected[0] if selected else "org.ex.T::fail",
            score=0.9, tier=1, reason="safety_bridge_t1",
        )
        sel_trace = SelectionTrace(
            selected=[sel_test],
            dropped=[],
        )
        return bug, intent, [candidate], trace, sel_trace

    def test_returns_bug_metrics(self, tmp_path):
        from semantic_rts.config import load_config
        from semantic_rts.eval.runner import run_bug

        cfg = load_config()
        store = MagicMock()
        store.all_tests.return_value = [("org.ex.T::fail", 1)]
        client = MagicMock()
        embedder = MagicMock()

        bug, intent, candidates, trace, sel_trace = self._mocks()

        with (
            patch("semantic_rts.eval.runner.load_bug", return_value=bug),
            patch("semantic_rts.eval.runner.parse_unified_diff") as mock_parse,
            patch("semantic_rts.eval.runner.extract_changed_methods", return_value=[]),
            patch("semantic_rts.eval.runner.analyze_intent", return_value=intent),
            patch("semantic_rts.eval.runner.retrieve", return_value=(candidates, trace)),
            patch("semantic_rts.eval.runner.select", return_value=sel_trace),
        ):
            mock_parse.return_value = MagicMock(
                files_changed=["Auth.java"], diff_hash="abc", file_changes=[]
            )
            m = run_bug("Chart", 1, cfg, store, client, embedder, tmp_path)

        assert isinstance(m, BugMetrics)
        assert m.project == "Chart"
        assert m.bug_id == 1
        assert m.recall == pytest.approx(1.0)   # selected "fail" == failing "fail"

    def test_empty_diff_returns_zero_recall(self, tmp_path):
        from semantic_rts.config import load_config
        from semantic_rts.eval.runner import run_bug

        cfg = load_config()
        store = MagicMock()
        client = MagicMock()
        embedder = MagicMock()

        bug = _make_bug_info(diff="")  # empty diff

        with patch("semantic_rts.eval.runner.load_bug", return_value=bug):
            m = run_bug("Chart", 1, cfg, store, client, embedder, tmp_path)

        assert m.recall == pytest.approx(0.0)
        assert m.n_selected == 0

    def test_latency_is_positive(self, tmp_path):
        from semantic_rts.config import load_config
        from semantic_rts.eval.runner import run_bug
        from semantic_rts.impact.intent_agent import IntentResult
        from semantic_rts.impact.retriever import AnalysisTrace
        from semantic_rts.selector.ranker import SelectionTrace

        cfg = load_config()
        store = MagicMock()
        store.all_tests.return_value = []
        client = MagicMock()
        embedder = MagicMock()

        bug = _make_bug_info()
        intent = IntentResult(intent_summary="x", concepts=[], risk_areas=[])
        trace = AnalysisTrace(
            diff_hash="", files_changed=[], methods_changed=[],
            intent_summary="", intent_concepts=[], risk_areas=[],
            query_text="", top_k=30, raw_results=[],
        )
        sel_trace = SelectionTrace(selected=[], dropped=[])

        with (
            patch("semantic_rts.eval.runner.load_bug", return_value=bug),
            patch("semantic_rts.eval.runner.parse_unified_diff") as mock_parse,
            patch("semantic_rts.eval.runner.extract_changed_methods", return_value=[]),
            patch("semantic_rts.eval.runner.analyze_intent", return_value=intent),
            patch("semantic_rts.eval.runner.retrieve", return_value=([], trace)),
            patch("semantic_rts.eval.runner.select", return_value=sel_trace),
        ):
            mock_parse.return_value = MagicMock(
                files_changed=[], diff_hash="", file_changes=[]
            )
            m = run_bug("Chart", 1, cfg, store, client, embedder, tmp_path)

        assert m.latency_ms >= 0


# ---------------------------------------------------------------------------
# run_eval()
# ---------------------------------------------------------------------------

class TestRunEval:
    def _make_store_and_llm(self):
        store = MagicMock()
        store.size = 5
        store.all_tests.return_value = []
        client = MagicMock()
        embedder = MagicMock()
        return store, client, embedder

    def test_processes_all_bugs(self, tmp_path):
        from semantic_rts.config import load_config
        from semantic_rts.eval.runner import run_eval

        cfg = load_config()
        kb_path = tmp_path / "kb"
        kb_path.mkdir()

        metrics_seq = [_make_metrics(bug_id=i) for i in [1, 2, 3]]
        call_count = []

        def fake_run_bug(project, bug_id, *args, **kwargs):
            call_count.append(bug_id)
            return metrics_seq[bug_id - 1]

        with (
            patch("semantic_rts.eval.runner.VectorStore") as MockVS,
            patch("semantic_rts.eval.runner.GeminiClient"),
            patch("semantic_rts.eval.runner.GeminiEmbedder"),
            patch("semantic_rts.eval.runner.run_bug", side_effect=fake_run_bug),
        ):
            MockVS.load.return_value.size = 5
            results = run_eval("Chart", [1, 2, 3], cfg, kb_path, tmp_path)

        assert len(results) == 3
        assert call_count == [1, 2, 3]

    def test_skips_failed_bugs_and_continues(self, tmp_path):
        from semantic_rts.config import load_config
        from semantic_rts.eval.runner import run_eval

        cfg = load_config()
        kb_path = tmp_path / "kb"
        kb_path.mkdir()

        def fake_run_bug(project, bug_id, *args, **kwargs):
            if bug_id == 2:
                raise RuntimeError("D4J error")
            return _make_metrics(bug_id=bug_id)

        with (
            patch("semantic_rts.eval.runner.VectorStore") as MockVS,
            patch("semantic_rts.eval.runner.GeminiClient"),
            patch("semantic_rts.eval.runner.GeminiEmbedder"),
            patch("semantic_rts.eval.runner.run_bug", side_effect=fake_run_bug),
        ):
            MockVS.load.return_value.size = 5
            results = run_eval("Chart", [1, 2, 3], cfg, kb_path, tmp_path)

        assert len(results) == 2
        assert all(r.bug_id != 2 for r in results)

    def test_writes_csv_when_output_dir_given(self, tmp_path):
        from semantic_rts.config import load_config
        from semantic_rts.eval.runner import run_eval

        cfg = load_config()
        kb_path = tmp_path / "kb"
        kb_path.mkdir()
        out_dir = tmp_path / "results"

        with (
            patch("semantic_rts.eval.runner.VectorStore") as MockVS,
            patch("semantic_rts.eval.runner.GeminiClient"),
            patch("semantic_rts.eval.runner.GeminiEmbedder"),
            patch("semantic_rts.eval.runner.run_bug", return_value=_make_metrics()),
        ):
            MockVS.load.return_value.size = 5
            run_eval("Chart", [1], cfg, kb_path, tmp_path, output_dir=out_dir)

        assert (out_dir / "Chart_semantic_results.csv").exists()

    def test_no_csv_when_output_dir_is_none(self, tmp_path):
        from semantic_rts.config import load_config
        from semantic_rts.eval.runner import run_eval

        cfg = load_config()
        kb_path = tmp_path / "kb"
        kb_path.mkdir()

        with (
            patch("semantic_rts.eval.runner.VectorStore") as MockVS,
            patch("semantic_rts.eval.runner.GeminiClient"),
            patch("semantic_rts.eval.runner.GeminiEmbedder"),
            patch("semantic_rts.eval.runner.run_bug", return_value=_make_metrics()),
        ):
            MockVS.load.return_value.size = 5
            run_eval("Chart", [1], cfg, kb_path, tmp_path, output_dir=None)

        # No CSV should be written anywhere in tmp_path
        assert not any(tmp_path.rglob("*.csv"))

    def test_returns_empty_list_on_all_failures(self, tmp_path):
        from semantic_rts.config import load_config
        from semantic_rts.eval.runner import run_eval

        cfg = load_config()
        kb_path = tmp_path / "kb"
        kb_path.mkdir()

        with (
            patch("semantic_rts.eval.runner.VectorStore") as MockVS,
            patch("semantic_rts.eval.runner.GeminiClient"),
            patch("semantic_rts.eval.runner.GeminiEmbedder"),
            patch("semantic_rts.eval.runner.run_bug", side_effect=RuntimeError("boom")),
        ):
            MockVS.load.return_value.size = 5
            results = run_eval("Chart", [1, 2], cfg, kb_path, tmp_path)

        assert results == []
