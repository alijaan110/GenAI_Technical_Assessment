"""
BM25 sparse retrieval store. Lexical complement to the dense vector index —
legal queries demand exact matches on `Article 17`, `§ 5(2)(a)` which dense
embeddings smear together.

Backed by `rank-bm25` (Okapi-BM25). The corpus itself lives in the SQLite
chunks table; this index is rebuilt in-memory from that on demand.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from backend.db import get_db


@dataclass
class BM25Hit:
    chunk_id: str
    text: str
    metadata: Dict
    score: float


_TOKEN_RE = re.compile(r"[A-Za-z0-9§()\.\-]+")


def _tokenize(text: str) -> List[str]:
    # Keep §, parentheses, dots and hyphens so legal section identifiers
    # survive tokenization ("§ 5(2)(a)", "Article 17", "5.1.2-bis").
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Store:
    def __init__(self) -> None:
        self.bm25: Optional[BM25Okapi] = None
        self.docs: List[Dict] = []
        self._built = False

    def rebuild(self) -> None:
        db = get_db()
        rows = db.execute("SELECT id, chunk_text, metadata FROM chunks").fetchall()
        self.docs = [
            {
                "chunk_id": r["id"],
                "text": r["chunk_text"],
                "metadata": json.loads(r["metadata"] or "{}"),
            }
            for r in rows
        ]
        if not self.docs:
            self.bm25 = None
            self._built = True
            return
        self.bm25 = BM25Okapi([_tokenize(d["text"]) for d in self.docs])
        self._built = True

    def search(self, query: str, k: int) -> List[BM25Hit]:
        if not self._built:
            self.rebuild()
        if self.bm25 is None or not self.docs:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            BM25Hit(
                chunk_id=self.docs[i]["chunk_id"],
                text=self.docs[i]["text"],
                metadata=self.docs[i]["metadata"],
                score=float(scores[i]),
            )
            for i in ranked
        ]

    def size(self) -> int:
        return len(self.docs)


_singleton: Optional[BM25Store] = None


def get_bm25() -> BM25Store:
    global _singleton
    if _singleton is None:
        _singleton = BM25Store()
    return _singleton


def reset_bm25() -> None:
    global _singleton
    _singleton = None
