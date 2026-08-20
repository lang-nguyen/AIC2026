"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum


# ─── Enums ───────────────────────────────────────────────

class SearchMode(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    OCR = "ocr"
    OBJECT = "object"
    PLACE = "place"
    AUDIO = "audio"
    TEMPORAL = "temporal"


# ─── Request Schemas ─────────────────────────────────────

class SearchRequest(BaseModel):
    """Unified search request."""
    text_queries: List[str] = Field(default_factory=list, description="Text queries")
    mode: SearchMode = Field(default=SearchMode.HYBRID, description="Search mode")
    limit: int = Field(default=200, ge=1, le=2000)
    use_llm_enhance: bool = Field(default=True, description="Use LLM to enhance queries")
    use_llm_rerank: bool = Field(default=False, description="Use LLM to rerank results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")


class TemporalEvent(BaseModel):
    """Single event in a temporal query."""
    text: str = Field(..., description="Description of this event")
    filters: Optional[Dict[str, Any]] = None


class TemporalSearchRequest(BaseModel):
    """Temporal search: find sequences of events in order."""
    events: List[TemporalEvent] = Field(..., min_length=2, max_length=4)
    limit: int = Field(default=100, ge=1, le=500)
    max_time_gap: float = Field(default=300.0, description="Max seconds between events")


class ChatRequest(BaseModel):
    """Chat with AI assistant."""
    question: str = Field(..., min_length=1)
    search_context: Optional[List[str]] = Field(default=None, description="IDs of current search results for context")


class AutoQueryRequest(BaseModel):
    """Automatic mode query."""
    query_text: str = Field(..., description="Raw query text from competition")
    session_id: Optional[str] = None
    max_results: int = Field(default=20, ge=1, le=100)


class NearbyFramesRequest(BaseModel):
    """Get frames near a given frame."""
    frame_id: str = Field(..., description="Frame ID e.g. K01_V001_042")
    range: int = Field(default=40, ge=1, le=200)


class ImageSearchRequest(BaseModel):
    """Search by uploaded image."""
    limit: int = Field(default=200, ge=1, le=2000)


# ─── Response Schemas ────────────────────────────────────

class FrameMetadata(BaseModel):
    """Metadata for a keyframe."""
    frame_idx: Optional[int] = None
    pts_time: Optional[float] = None
    video_id: Optional[str] = None
    youtube_id: Optional[str] = None
    n: Optional[int] = None
    group: Optional[str] = None  # K01, L21, etc.


class SearchResult(BaseModel):
    """Single search result item."""
    id: str
    score: float = 0.0
    metadata: FrameMetadata = Field(default_factory=FrameMetadata)
    thumbnail_url: Optional[str] = None


class SearchResponse(BaseModel):
    """Search response."""
    results: List[SearchResult]
    total: int
    query_enhanced: Optional[List[str]] = None
    search_time_ms: Optional[float] = None


class TemporalMatch(BaseModel):
    """A temporal match: sequence of frames across events."""
    video_id: str
    frames: List[SearchResult]
    total_score: float


class TemporalSearchResponse(BaseModel):
    """Temporal search response."""
    matches: List[TemporalMatch]
    total: int
    search_time_ms: Optional[float] = None


class ChatResponse(BaseModel):
    """Chat response."""
    text: str
    frame_results: List[SearchResult] = Field(default_factory=list)


class AutoQueryResponse(BaseModel):
    """Automatic query response."""
    results: List[SearchResult]
    answer: Optional[str] = None
    confidence: float = 0.0


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    version: str = "2026.1.0"
    services: Dict[str, bool] = Field(default_factory=dict)
