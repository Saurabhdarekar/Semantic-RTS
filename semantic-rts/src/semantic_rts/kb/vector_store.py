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

    @property
    def size(self) -> int:
        return self._index.ntotal
