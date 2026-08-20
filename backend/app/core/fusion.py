"""
Score fusion utilities.
Implements Reciprocal Rank Fusion (RRF) and weighted fusion.
"""

from typing import List, Dict
from collections import defaultdict


def reciprocal_rank_fusion(
    *result_lists: List[Dict],
    k: int = 60,
    limit: int = 200,
) -> List[Dict]:
    """Combine multiple ranked result lists using Reciprocal Rank Fusion.

    RRF is proven to outperform simple weighted averaging for combining
    heterogeneous retrieval systems (semantic + keyword).

    Formula: score(d) = Σ 1/(k + rank(d))

    Args:
        result_lists: Variable number of ranked result lists.
            Each list contains dicts with 'id', 'score', 'metadata'.
        k: RRF constant (default 60, as per original paper).
        limit: Max results to return.

    Returns:
        Fused and re-ranked results.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_store: Dict[str, Dict] = {}

    for result_list in result_lists:
        if not result_list:
            continue
        for rank, doc in enumerate(result_list):
            doc_id = doc["id"]
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            # Keep the doc with best individual metadata
            if doc_id not in doc_store:
                doc_store[doc_id] = doc

    # Build fused results
    fused = []
    for doc_id, score in rrf_scores.items():
        result = doc_store[doc_id].copy()
        result["score"] = score
        fused.append(result)

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:limit]


def weighted_fusion(
    result_lists: List[List[Dict]],
    weights: List[float],
    limit: int = 200,
) -> List[Dict]:
    """Combine result lists using weighted score summation.

    Args:
        result_lists: List of ranked result lists.
        weights: Weight for each result list. Must sum to 1.0.
        limit: Max results.

    Returns:
        Fused results.
    """
    assert len(result_lists) == len(weights), "Lists and weights must match"

    scores: Dict[str, float] = defaultdict(float)
    doc_store: Dict[str, Dict] = {}

    for result_list, weight in zip(result_lists, weights):
        if not result_list:
            continue

        # Normalize scores within this list
        max_score = max(r["score"] for r in result_list) if result_list else 1.0
        min_score = min(r["score"] for r in result_list) if result_list else 0.0
        score_range = max_score - min_score + 1e-6

        for doc in result_list:
            doc_id = doc["id"]
            normalized = (doc["score"] - min_score) / score_range
            scores[doc_id] += normalized * weight

            if doc_id not in doc_store:
                doc_store[doc_id] = doc

    fused = []
    for doc_id, score in scores.items():
        result = doc_store[doc_id].copy()
        result["score"] = score
        fused.append(result)

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:limit]
