"""
Dense vector store with two backends behind one interface:

  - QdrantBackend:  production Qdrant client, probed at startup.
  - MemoryBackend:  in-process cosine search, used when Qdrant is unreachable.

Vectors and chunks are persisted authoritatively in SQLite, so the choice
of backend is purely a runtime cache decision — switching backends never
loses data and never requires re-ingestion.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from backend.db import get_db
from backend.embedder import EMBED_DIM, build_embeddings, embed_query
from backend.settings import get_settings

QDRANT_COLLECTION = "legal_docs"


@dataclass
class DenseHit:
    chunk_id: str
    text: str
    metadata: Dict
    score: float


class DenseBackend(Protocol):
    name: str

    def upsert(self, items: List[Dict]) -> None: ...
    def delete_by_document(self, document_id: str) -> None: ...
    def search(self, query: str, k: int) -> List[DenseHit]: ...
    def size(self) -> int: ...


# ── Qdrant backend ────────────────────────────────────────────────
class QdrantBackend:
    name = "qdrant"

    def __init__(self, client: QdrantClient):
        self.client = client
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self.client.get_collection(QDRANT_COLLECTION)
        except Exception:
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
            )

    @staticmethod
    def _hash_id(s: str) -> int:
        # Deterministic 32-bit unsigned int — Qdrant requires int or UUID
        # for point IDs and the original chunk_id is preserved in payload.
        h = 2166136261
        for ch in s:
            h = (h ^ ord(ch)) & 0xFFFFFFFF
            h = (h * 16777619) & 0xFFFFFFFF
        return h

    def upsert(self, items: List[Dict]) -> None:
        if not items:
            return
        points = [
            PointStruct(
                id=self._hash_id(it["chunk_id"]),
                vector=it["vector"],
                payload={
                    **it["metadata"],
                    "chunk_id": it["chunk_id"],
                    "text": it["text"],
                },
            )
            for it in items
        ]
        self.client.upsert(collection_name=QDRANT_COLLECTION, points=points, wait=True)

    def delete_by_document(self, document_id: str) -> None:
        try:
            self.client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
                ),
                wait=True,
            )
        except Exception as e:
            print(f"[qdrant] delete_by_document failed: {e}")

    def search(self, query: str, k: int) -> List[DenseHit]:
        qv = embed_query(query)
        try:
            results = self.client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=qv,
                limit=k,
                with_payload=True,
            )
        except Exception as e:
            print(f"[qdrant] search failed: {e}")
            return []
        hits: List[DenseHit] = []
        for r in results:
            payload = dict(r.payload or {})
            text = str(payload.pop("text", ""))
            cid = str(payload.get("chunk_id", ""))
            hits.append(DenseHit(chunk_id=cid, text=text, metadata=payload, score=float(r.score)))
        return hits

    def size(self) -> int:
        try:
            info = self.client.get_collection(QDRANT_COLLECTION)
            return int(getattr(info, "points_count", 0) or 0)
        except Exception:
            return 0


# ── In-memory backend ─────────────────────────────────────────────
@dataclass
class _MemRecord:
    chunk_id: str
    text: str
    metadata: Dict
    vector: List[float]


class MemoryBackend:
    name = "memory"

    def __init__(self) -> None:
        self.records: List[_MemRecord] = []

    @staticmethod
    def _cos(a: List[float], b: List[float]) -> float:
        dot = na = nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0 or nb == 0:
            return 0.0
        return dot / (math.sqrt(na) * math.sqrt(nb))

    def upsert(self, items: List[Dict]) -> None:
        existing = {r.chunk_id: i for i, r in enumerate(self.records)}
        for it in items:
            rec = _MemRecord(
                chunk_id=it["chunk_id"],
                text=it["text"],
                metadata=it["metadata"],
                vector=it["vector"],
            )
            if it["chunk_id"] in existing:
                self.records[existing[it["chunk_id"]]] = rec
            else:
                self.records.append(rec)

    def delete_by_document(self, document_id: str) -> None:
        self.records = [r for r in self.records if r.metadata.get("document_id") != document_id]

    def search(self, query: str, k: int) -> List[DenseHit]:
        if not self.records:
            return []
        qv = embed_query(query)
        scored = [
            DenseHit(
                chunk_id=r.chunk_id,
                text=r.text,
                metadata=r.metadata,
                score=self._cos(r.vector, qv),
            )
            for r in self.records
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def size(self) -> int:
        return len(self.records)


# ── Factory + hydration ───────────────────────────────────────────
_backend: Optional[DenseBackend] = None
_hydrated: bool = False


def _try_qdrant() -> Optional[QdrantBackend]:
    s = get_settings()
    url = s.qdrant_url or "http://localhost:6333"
    api_key = s.qdrant_api_key or None
    try:
        client = QdrantClient(url=url, api_key=api_key, timeout=5.0)
        client.get_collections()
        return QdrantBackend(client)
    except Exception as e:
        print(f"[qdrant] not reachable ({e}) — falling back to in-memory store.")
        return None


def get_backend() -> DenseBackend:
    global _backend, _hydrated
    if _backend is not None and _hydrated:
        return _backend
    if _backend is None:
        qb = _try_qdrant()
        _backend = qb if qb is not None else MemoryBackend()

    # Hydrate from SQLite — the canonical source of truth.
    db = get_db()
    rows = db.execute(
        "SELECT id, document_id, chunk_text, embedding_vector, metadata FROM chunks"
    ).fetchall()

    items: List[Dict] = []
    for r in rows:
        ev = r["embedding_vector"]
        if not ev or ev == "[]":
            continue
        try:
            vec = json.loads(ev)
        except Exception:
            continue
        meta = json.loads(r["metadata"] or "{}")
        meta["chunk_id"] = r["id"]
        meta["document_id"] = r["document_id"]
        items.append(
            {
                "chunk_id": r["id"],
                "text": r["chunk_text"],
                "metadata": meta,
                "vector": vec,
            }
        )

    if items:
        _backend.upsert(items)

    _hydrated = True
    return _backend


def reset_backend() -> None:
    """Force a re-probe + re-hydrate (e.g. after Qdrant URL change or doc delete)."""
    global _backend, _hydrated
    _backend = None
    _hydrated = False


def active_backend_name() -> str:
    if _backend is None:
        return "unknown"
    return _backend.name
