"""
Hybrid retrieval = dense + sparse, fused via Reciprocal Rank Fusion (RRF).

  RRF score = Σ 1 / (rrf_k + rank_in_list)
  rrf_k=60 is the standard smoothing constant from the original RRF paper.

RRF doesn't require normalising scores across heterogenous retrievers, which
is exactly why it's the preferred fusion strategy for hybrid RAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from backend.bm25_store import get_bm25
from backend.vector_store import get_backend


@dataclass
class HybridHit:
    chunk_id: str
    text: str
    metadata: Dict
    rrf_score: float
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None


def hybrid_retrieve(
    query: str,
    *,
    k: int = 8,
    use_hybrid: bool = True,
    rrf_k: int = 60,
    over_fetch: int | None = None,
) -> List[HybridHit]:
    # Pull a wider over-fetch so RRF has more candidates to dedupe and rank;
    # bigger k for the final result improves recall on long/multi-clause questions.
    over_fetch = over_fetch or max(k * 4, 20)
    dense = get_backend().search(query, over_fetch)
    sparse = get_bm25().search(query, over_fetch) if use_hybrid else []

    fused: Dict[str, HybridHit] = {}

    for rank, hit in enumerate(dense):
        if not hit.chunk_id:
            continue
        fused[hit.chunk_id] = HybridHit(
            chunk_id=hit.chunk_id,
            text=hit.text,
            metadata=hit.metadata,
            rrf_score=1.0 / (rrf_k + rank + 1),
            dense_rank=rank,
            dense_score=hit.score,
        )

    for rank, hit in enumerate(sparse):
        existing = fused.get(hit.chunk_id)
        if existing is not None:
            existing.rrf_score += 1.0 / (rrf_k + rank + 1)
            existing.sparse_rank = rank
            existing.sparse_score = hit.score
        else:
            fused[hit.chunk_id] = HybridHit(
                chunk_id=hit.chunk_id,
                text=hit.text,
                metadata=hit.metadata,
                rrf_score=1.0 / (rrf_k + rank + 1),
                sparse_rank=rank,
                sparse_score=hit.score,
            )

    return sorted(fused.values(), key=lambda h: h.rrf_score, reverse=True)[:k]
