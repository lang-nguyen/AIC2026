"""
API Router - All endpoints.
"""

import logging
import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query

from app.models.schemas import (
    SearchRequest, SearchResponse, SearchResult, FrameMetadata,
    TemporalSearchRequest, TemporalSearchResponse, TemporalMatch,
    ChatRequest, ChatResponse,
    AutoQueryRequest, AutoQueryResponse,
    NearbyFramesRequest,
    HealthResponse, SearchMode,
)
from app.core.retrieval import unified_search
from app.core.temporal_search import temporal_search
from app.services.clip_encoder import get_clip_encoder
from app.services.vector_db import get_vector_db
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Health ──────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    vector_db = get_vector_db()
    return HealthResponse(
        status="ok",
        services={
            "faiss": vector_db.index is not None and vector_db.index.ntotal > 0,
            "clip": True,
        }
    )


# ─── Unified Search ─────────────────────────────────────

@router.post("/api/v1/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest):
    """Unified search endpoint - handles all search modes."""
    try:
        result = await unified_search(
            text_queries=req.text_queries,
            mode=req.mode,
            limit=req.limit,
            use_llm_enhance=req.use_llm_enhance,
            use_llm_rerank=req.use_llm_rerank,
        )

        return SearchResponse(
            results=[
                SearchResult(
                    id=r["id"],
                    score=r.get("score", 0),
                    metadata=FrameMetadata(**r.get("metadata", {})),
                    thumbnail_url=r.get("thumbnail_url"),
                )
                for r in result["results"]
            ],
            total=result["total"],
            query_enhanced=result.get("query_enhanced"),
            search_time_ms=result.get("search_time_ms"),
        )
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Image Search ───────────────────────────────────────

@router.post("/api/v1/search/image", response_model=SearchResponse)
async def image_search_endpoint(
    image: UploadFile = File(...),
    text_queries: str = Form(default="[]"),
    limit: int = Form(default=200),
):
    """Search using uploaded image + optional text."""
    try:
        encoder = get_clip_encoder()
        image_data = await image.read()
        image_emb = encoder.encode_image(image_data)

        queries = json.loads(text_queries)

        result = await unified_search(
            text_queries=queries,
            mode=SearchMode.HYBRID if queries else SearchMode.SEMANTIC,
            limit=limit,
            use_llm_enhance=bool(queries),
            image_embeddings=image_emb,
        )

        return SearchResponse(
            results=[
                SearchResult(
                    id=r["id"],
                    score=r.get("score", 0),
                    metadata=FrameMetadata(**r.get("metadata", {})),
                    thumbnail_url=r.get("thumbnail_url"),
                )
                for r in result["results"]
            ],
            total=result["total"],
            search_time_ms=result.get("search_time_ms"),
        )
    except Exception as e:
        logger.error(f"Image search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Temporal Search ────────────────────────────────────

@router.post("/api/v1/search/temporal", response_model=TemporalSearchResponse)
async def temporal_search_endpoint(req: TemporalSearchRequest):
    """Search for temporal sequences of events."""
    import time
    start = time.time()

    try:
        event_queries = [e.text for e in req.events]

        # Enhance queries
        llm = get_llm_service()
        enhanced = await llm.enhance_queries(event_queries)

        matches = await temporal_search(
            event_queries=enhanced,
            limit=req.limit,
            max_time_gap=req.max_time_gap,
        )

        elapsed = (time.time() - start) * 1000

        return TemporalSearchResponse(
            matches=[
                TemporalMatch(
                    video_id=m["video_id"],
                    frames=[
                        SearchResult(
                            id=f["id"],
                            score=f["score"],
                            metadata=FrameMetadata(**f.get("metadata", {})),
                        )
                        for f in m["frames"]
                    ],
                    total_score=m["total_score"],
                )
                for m in matches
            ],
            total=len(matches),
            search_time_ms=round(elapsed, 1),
        )
    except Exception as e:
        logger.error(f"Temporal search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Nearby Frames ──────────────────────────────────────

@router.get("/api/v1/frames/nearby")
async def nearby_frames_endpoint(
    id: str = Query(..., description="Frame ID"),
    range: int = Query(default=40, ge=1, le=200),
):
    """Get frames near a given frame."""
    vector_db = get_vector_db()
    results = vector_db.get_nearby_frames(id, range)

    return {
        "results": [
            {
                "id": r["id"],
                "metadata": r["metadata"],
                "thumbnail_url": _get_thumb(r["id"]),
            }
            for r in results
        ]
    }


# ─── Frames by Video ────────────────────────────────────

@router.get("/api/v1/frames/video/{video_id}")
async def frames_by_video_endpoint(video_id: str):
    """Get all frames for a specific video."""
    vector_db = get_vector_db()
    results = vector_db.get_frames_by_video(video_id)
    return {
        "results": [
            {
                "id": r["id"],
                "metadata": r["metadata"],
                "thumbnail_url": _get_thumb(r["id"]),
            }
            for r in results
        ]
    }


# ─── Chat ────────────────────────────────────────────────

@router.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """AI chatbot endpoint."""
    try:
        llm = get_llm_service()
        response_text = await llm.chat_response(req.question)

        return ChatResponse(text=response_text)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Automatic Mode ─────────────────────────────────────

@router.post("/api/v1/automatic/query", response_model=AutoQueryResponse)
async def automatic_query_endpoint(req: AutoQueryRequest):
    """Automatic mode: receive query, return results without user interaction."""
    try:
        llm = get_llm_service()

        # Parse query with LLM
        parsed = await llm.parse_auto_query(req.query_text)

        # Execute search based on parsed strategy
        text_queries = parsed.get("text_queries", [req.query_text])
        is_temporal = parsed.get("is_temporal", False)
        temporal_events = parsed.get("temporal_events")

        if is_temporal and temporal_events and len(temporal_events) >= 2:
            # Temporal search
            matches = await temporal_search(
                event_queries=temporal_events,
                limit=req.max_results,
            )
            results = []
            for m in matches:
                results.extend(m["frames"])
        else:
            # Standard hybrid search
            mode = SearchMode.HYBRID
            if parsed.get("search_mode") == "ocr":
                mode = SearchMode.OCR
                text_queries = parsed.get("ocr_queries", text_queries)

            search_result = await unified_search(
                text_queries=text_queries,
                mode=mode,
                limit=req.max_results,
                use_llm_enhance=True,
            )
            results = search_result["results"]

        # Also search by specific types if available
        if parsed.get("object_queries"):
            from app.services.elasticsearch_service import get_es_service
            es = get_es_service()
            obj_results = await es.object_search(parsed["object_queries"], 50)
            # Merge with main results using RRF
            from app.core.fusion import reciprocal_rank_fusion
            if results and obj_results:
                results = reciprocal_rank_fusion(results, obj_results, limit=req.max_results)
            elif obj_results:
                results = obj_results

        return AutoQueryResponse(
            results=[
                SearchResult(
                    id=r["id"],
                    score=r.get("score", 0),
                    metadata=FrameMetadata(**r.get("metadata", {})),
                    thumbnail_url=r.get("thumbnail_url", _get_thumb(r["id"])),
                )
                for r in results[:req.max_results]
            ],
            confidence=0.8 if results else 0.0,
        )
    except Exception as e:
        logger.error(f"Automatic query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Helpers ─────────────────────────────────────────────

def _get_thumb(frame_id: str) -> str:
    parts = frame_id.rsplit("_", 1)
    if len(parts) != 2:
        return ""
    video_id = parts[0]
    frame_num = parts[1]
    group = video_id.split("_")[0]
    return f"/static/frames/{group}/{video_id}/{frame_num}.jpg"
