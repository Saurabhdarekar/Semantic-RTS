"""FAISS IndexFlatIP wrapper: build, save, load, search."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class VectorStore:
    """Thin wrapper around a FAISS IndexFlatIP index.

    Vectors must be unit-normalized before being added (the embedder handles this).
    IndexFlatIP on unit vectors computes cosine similarity.
    """

    def __init__(self, dim: int = 768) -> None:
        self._index = faiss.IndexFlatIP(dim)
        self._dim = dim
        self._test_ids: list[str] = []
        self._tiers: list[int] = []
        self._rows: list[dict[str, Any]] = []  # full rows for inspection / export
        self._bm25 = None  # built on demand via build_bm25()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def add(
        self,
        vectors: list[list[float]],
        test_ids: list[str],
        tiers: list[int] | None = None,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add normalized vectors and metadata to the index."""
        mat = np.array(vectors, dtype=np.float32)
        self._index.add(mat)
        self._test_ids.extend(test_ids)
        self._tiers.extend(tiers if tiers is not None else [3] * len(test_ids))
        self._rows.extend(rows if rows is not None else [{} for _ in test_ids])

    def upsert(
        self,
        vector: list[float],
        test_id: str,
        tier: int,
        row: "dict[str, Any] | None" = None,
    ) -> None:
        """Add a new vector or replace an existing one with the same test_id.

        IndexFlatIP has no native delete, so an update rebuilds the index
        from the reconstructed vectors minus the old entry.
        """
        if test_id in self._test_ids:
            idx = self._test_ids.index(test_id)
            n = self._index.ntotal
            all_vecs = np.zeros((n, self._dim), dtype=np.float32)
            for i in range(n):
                self._index.reconstruct(i, all_vecs[i])
            keep = [i for i in range(n) if i != idx]
            self._index = faiss.IndexFlatIP(self._dim)
            if keep:
                self._index.add(all_vecs[keep])
            self._test_ids.pop(idx)
            self._tiers.pop(idx)
            if idx < len(self._rows):
                self._rows.pop(idx)

        mat = np.array([vector], dtype=np.float32)
        self._index.add(mat)
        self._test_ids.append(test_id)
        self._tiers.append(tier)
        self._rows.append(row or {})

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: list[float], k: int) -> tuple[list[float], list[str]]:
        """Return (scores, test_ids) for the top-k most similar vectors.

        Returns fewer than k results if the index has fewer entries.
        """
        if self._index.ntotal == 0:
            return [], []
        actual_k = min(k, self._index.ntotal)
        q = np.array([query], dtype=np.float32)
        scores_mat, idx_mat = self._index.search(q, actual_k)
        scores = scores_mat[0].tolist()
        ids = [self._test_ids[i] for i in idx_mat[0]]
        return scores, ids

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, kb_dir: Path) -> None:
        """Persist the FAISS index and metadata to kb_dir."""
        kb_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(kb_dir / "index.faiss"))
        meta = {
            "test_ids": self._test_ids,
            "tiers": self._tiers,
            "n_tests": len(self._test_ids),
            "dim": self._dim,
            "build_time": time.time(),
        }
        with open(kb_dir / "index.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, kb_dir: Path) -> "VectorStore":
        """Load a previously saved VectorStore from kb_dir."""
        store = cls.__new__(cls)
        store._index = faiss.read_index(str(kb_dir / "index.faiss"))
        store._dim = store._index.d
        with open(kb_dir / "index.meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        store._test_ids = meta["test_ids"]
        store._tiers = meta["tiers"]
        store._rows = meta.get("rows", [{} for _ in store._test_ids])
        store._bm25 = None
        return store

    # ------------------------------------------------------------------
    # Accessors (used by Phase 3 and Retriever)
    # ------------------------------------------------------------------

    def test_id_at(self, idx: int) -> str:
        return self._test_ids[idx]

    def tier_at(self, idx: int) -> int:
        return self._tiers[idx]

    def all_tests(self) -> list[tuple[str, int]]:
        """Return [(test_id, tier), ...] for every test in the KB."""
        return list(zip(self._test_ids, self._tiers))

    def tier_for_id(self, test_id: str) -> int:
        """Return the tier for a given test_id (default 3 if not found)."""
        try:
            return self._tiers[self._test_ids.index(test_id)]
        except ValueError:
            return 3

    # ------------------------------------------------------------------
    # BM25 hybrid index
    # ------------------------------------------------------------------

    def build_bm25(self, tested_methods_map: dict[str, list[str]]) -> None:
        """Build BM25 index over test identity tokens.

        Must be called explicitly after load() because tested_methods data
        lives in tests.jsonl, not in index.meta.json.
        """
        import re as _re
        from rank_bm25 import BM25Okapi

        corpus = []
        for test_id in self._test_ids:
            tokens = _re.split(r'[.:_\s]+', test_id.lower())
            for method in tested_methods_map.get(test_id, []):
                tokens.extend(_re.split(r'[._\s]+', method.lower()))
            corpus.append(tokens)

        self._bm25 = BM25Okapi(corpus) if corpus else None

    def bm25_scores(self, query_tokens: list[str]) -> dict[str, float]:
        """Return {test_id: normalised_score} for all tests.

        Scores are in [0.0, 1.0]. Returns empty dict if BM25 not built.
        """
        if self._bm25 is None or not self._test_ids:
            return {}
        raw = self._bm25.get_scores(query_tokens)
        max_score = float(raw.max())
        if max_score <= 0:
            return {}
        return dict(zip(self._test_ids, (raw / max_score).tolist()))

    @property
    def size(self) -> int:
        return self._index.ntotal
