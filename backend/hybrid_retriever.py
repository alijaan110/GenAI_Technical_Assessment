"""
Hybrid retrieval = dense + sparse, fused via Reciprocal Rank Fusion (RRF).

  RRF score = Σ 1 / (rrf_k + rank_in_list)
  rrf_k=60 is the standard smoothing constant from the original RRF paper.

Enhanced with multi-query expansion: the original query is augmented with
keyword-based variants so retrieval catches different phrasings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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


# ── Lightweight query expansion (no LLM call, zero latency) ──────
_LEGAL_SYNONYMS = {
    "regulation": ["directive", "legislation", "law", "act"],
    "directive": ["regulation", "legislation"],
    "article": ["section", "provision", "clause"],
    "section": ["article", "provision"],
    "controller": ["data controller", "processor"],
    "processor": ["data processor", "controller"],
    "gdpr": ["general data protection regulation", "regulation 2016/679"],
    "data subject": ["individual", "natural person"],
    "consent": ["permission", "agreement", "authorization"],
    "personal data": ["personal information", "data"],
    "member state": ["eu member state", "member country"],
    "lawful basis": ["legal basis", "lawful grounds", "legal grounds"],
    "breach": ["violation", "infringement"],
    "penalty": ["fine", "sanction"],
    "rights": ["entitlements", "freedoms"],
    "residence": ["domicile", "habitual residence"],
    "periodicity": ["frequency", "interval", "period"],
    "transmission": ["reporting", "submission", "delivery"],
    "council": ["european council", "eu council"],
    "commission": ["european commission", "eu commission"],
    "rule of law": ["rule-of-law", "legal principles"],
}


def _expand_queries(question: str) -> List[str]:
    """
    Generate query variants using legal-domain synonym expansion.
    Zero-latency, no LLM calls — purely keyword-based.
    Returns up to 2 additional query variants.
    """
    q_lower = question.lower()
    variants = []

    for term, synonyms in _LEGAL_SYNONYMS.items():
        if term in q_lower:
            # Create a variant replacing the matched term with its first synonym
            for syn in synonyms[:1]:
                variant = re.sub(
                    re.escape(term), syn, q_lower, count=1, flags=re.IGNORECASE
                )
                if variant != q_lower and variant not in variants:
                    variants.append(variant)
                    if len(variants) >= 2:
                        return variants

    return variants


def _fuse_ranked_list(
    fused: Dict[str, HybridHit],
    hits: list,
    rrf_k: int,
    *,
    is_dense: bool,
) -> None:
    """Merge a single ranked list into the running RRF accumulator."""
    for rank, hit in enumerate(hits):
        cid = hit.chunk_id
        if not cid:
            continue
        score_incr = 1.0 / (rrf_k + rank + 1)
        existing = fused.get(cid)
        if existing is not None:
            existing.rrf_score += score_incr
            if is_dense and existing.dense_rank is None:
                existing.dense_rank = rank
                existing.dense_score = hit.score
            elif not is_dense and existing.sparse_rank is None:
                existing.sparse_rank = rank
                existing.sparse_score = hit.score
        else:
            fused[cid] = HybridHit(
                chunk_id=cid,
                text=hit.text,
                metadata=hit.metadata,
                rrf_score=score_incr,
                dense_rank=rank if is_dense else None,
                sparse_rank=None if is_dense else rank,
                dense_score=hit.score if is_dense else None,
                sparse_score=None if is_dense else hit.score,
            )


def hybrid_retrieve(
    query: str,
    *,
    k: int = 10,
    use_hybrid: bool = True,
    rrf_k: int = 60,
    over_fetch: int | None = None,
    expand_queries: bool = True,
) -> List[HybridHit]:
    # Pull a wider over-fetch so RRF has more candidates.
    over_fetch = over_fetch or max(k * 6, 30)
    backend = get_backend()
    bm25 = get_bm25() if use_hybrid else None

    # Build list of queries: original + keyword expansions (zero latency)
    queries = [query]
    if expand_queries:
        queries.extend(_expand_queries(query))

    fused: Dict[str, HybridHit] = {}

    for q in queries:
        dense_hits = backend.search(q, over_fetch)
        _fuse_ranked_list(fused, dense_hits, rrf_k, is_dense=True)

        if bm25 is not None:
            sparse_hits = bm25.search(q, over_fetch)
            _fuse_ranked_list(fused, sparse_hits, rrf_k, is_dense=False)

    return sorted(fused.values(), key=lambda h: h.rrf_score, reverse=True)[:k]
