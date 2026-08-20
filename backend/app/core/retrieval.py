"""
Main retrieval engine.
Orchestrates hybrid search (semantic + keyword) with RRF fusion.
"""

import logging
import time
import numpy as np
from typing import List, Dict, Optional

from app.services.clip_encoder import get_clip_encoder
from app.services.vector_db import get_vector_db
from app.services.elasticsearch_service import get_es_service
from app.services.llm_service import get_llm_service
from app.core.fusion import reciprocal_rank_fusion
from app.models.schemas import SearchMode

logger = logging.getLogger(__name__)


async def unified_search(
    text_queries: List[str],
    mode: SearchMode = SearchMode.HYBRID,
    limit: int = 200,
    use_llm_enhance: bool = True,
    use_llm_rerank: bool = False,
    image_embeddings: Optional[np.ndarray] = None,
) -> Dict:
    """Unified search engine combining multiple retrieval strategies.

    Args:
        text_queries: Text search queries.
        mode: Search mode (semantic, keyword, hybrid, etc.).
        limit: Max results.
        use_llm_enhance: Whether to enhance queries with LLM.
        use_llm_rerank: Whether to rerank with LLM (slower but more accurate).
        image_embeddings: Optional pre-computed image embeddings for image search.

    Returns:
        Dict with results, enhanced queries, and timing info.
    """
    start_time = time.time()
    encoder = get_clip_encoder()
    vector_db = get_vector_db()
    es_service = get_es_service()
    llm_service = get_llm_service()

    enhanced_queries = text_queries

    # Step 1: Enhance queries with LLM
    if use_llm_enhance and text_queries:
        enhanced_queries = await llm_service.enhance_queries(text_queries)

    results = []

    if mode == SearchMode.SEMANTIC or mode == SearchMode.HYBRID:
        # Semantic search via CLIP + FAISS
        semantic_results = []
        if enhanced_queries:
            text_emb = encoder.encode_text(enhanced_queries)
            semantic_results = vector_db.search(text_emb, limit=limit * 2)

        if image_embeddings is not None:
            img_results = vector_db.search(image_embeddings, limit=limit)
            if semantic_results:
                semantic_results = reciprocal_rank_fusion(
                    semantic_results, img_results, k=60, limit=limit * 2
                )
            else:
                semantic_results = img_results

        if mode == SearchMode.SEMANTIC:
            results = semantic_results[:limit]
        else:
            # Hybrid: also do keyword search
            keyword_results = await es_service.full_text_search(
                text_queries,  # Use original queries for keyword search
                limit=limit * 2,
            )

            if keyword_results and semantic_results:
                results = reciprocal_rank_fusion(
                    semantic_results, keyword_results,
                    k=60, limit=limit,
                )
            elif semantic_results:
                results = semantic_results[:limit]
            elif keyword_results:
                results = keyword_results[:limit]

    elif mode == SearchMode.KEYWORD:
        results = await es_service.full_text_search(text_queries, limit)

    elif mode == SearchMode.OCR:
        results = await es_service.ocr_search(text_queries, limit)

    elif mode == SearchMode.OBJECT:
        # Translate objects to English first
        en_queries = await llm_service.translate_to_english(text_queries)
        results = await es_service.object_search(en_queries, limit)

    elif mode == SearchMode.PLACE:
        en_queries = await llm_service.translate_to_english(text_queries)
        results = await es_service.place_search(en_queries, limit)

    elif mode == SearchMode.AUDIO:
        results = await es_service.audio_search(text_queries, limit)

    # Add thumbnail URLs
    for r in results:
        r["thumbnail_url"] = _get_thumbnail_url(r["id"])

    elapsed = (time.time() - start_time) * 1000

    return {
        "results": results,
        "total": len(results),
        "query_enhanced": enhanced_queries if enhanced_queries != text_queries else None,
        "search_time_ms": round(elapsed, 1),
    }


def _get_thumbnail_url(frame_id: str) -> str:
    """Generate thumbnail URL from frame ID.

    Frame ID format: K01_V001_042 → /static/frames/K01/K01_V001/042.jpg
    """
    parts = frame_id.rsplit("_", 1)
    if len(parts) != 2:
        return ""
    video_id = parts[0]
    frame_num = parts[1]
    group = video_id.split("_")[0]
    return f"/static/frames/{group}/{video_id}/{frame_num}.jpg"
