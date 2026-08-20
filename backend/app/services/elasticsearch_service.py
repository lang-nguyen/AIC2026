"""
Elasticsearch service for keyword-based search.
Handles OCR text, objects, places, audio transcripts.
"""

import logging
from typing import List, Dict, Optional
from elasticsearch import AsyncElasticsearch

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ElasticSearchService:
    """Async Elasticsearch service."""

    def __init__(self):
        self.client: Optional[AsyncElasticsearch] = None
        self.index_name = settings.ELASTICSEARCH_INDEX_NAME

    async def connect(self):
        """Initialize async ES client."""
        if not settings.ELASTICSEARCH_URL:
            logger.warning("Elasticsearch URL not configured")
            return
        self.client = AsyncElasticsearch(
            hosts=settings.ELASTICSEARCH_URL,
            api_key=settings.ELASTICSEARCH_API_KEY,
        )
        try:
            info = await self.client.info()
            logger.info(f"Connected to Elasticsearch: {info['version']['number']}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            self.client = None

    async def close(self):
        """Close ES client."""
        if self.client:
            await self.client.close()

    async def multi_field_search(
        self,
        queries: List[str],
        fields: List[str],
        limit: int = 200,
    ) -> List[Dict]:
        """Multi-field search across specified fields.

        Args:
            queries: List of search queries.
            fields: ES fields to search (e.g., ["objects", "places", "text"]).
            limit: Max results.

        Returns:
            List of search result dicts.
        """
        if not self.client or not queries:
            return []

        body = []
        per_query_limit = max(1, limit // len(queries))

        for query in queries:
            body.append({})
            body.append({
                "size": per_query_limit,
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": fields,
                        "fuzziness": "AUTO",
                        "type": "best_fields",
                    }
                },
            })

        try:
            response = await self.client.msearch(index=self.index_name, body=body)
        except Exception as e:
            logger.error(f"ES search error: {e}")
            return []

        results = []
        for resp in response.get("responses", []):
            if "error" in resp:
                continue
            for hit in resp.get("hits", {}).get("hits", []):
                results.append({
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "metadata": hit["_source"].get("metadata", {}),
                })

        # Deduplicate and sort
        seen = {}
        for r in results:
            if r["id"] not in seen or r["score"] > seen[r["id"]]["score"]:
                seen[r["id"]] = r

        deduped = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return deduped[:limit]

    async def ocr_search(self, queries: List[str], limit: int = 200) -> List[Dict]:
        """Search OCR text."""
        return await self.multi_field_search(queries, ["text"], limit)

    async def object_search(self, queries: List[str], limit: int = 200) -> List[Dict]:
        """Search detected objects."""
        return await self.multi_field_search(queries, ["objects"], limit)

    async def place_search(self, queries: List[str], limit: int = 200) -> List[Dict]:
        """Search by place/location."""
        if not self.client or not queries:
            return []

        body = []
        per_query_limit = max(1, limit // len(queries))

        for place in queries:
            body.append({})
            body.append({
                "size": per_query_limit,
                "query": {
                    "bool": {
                        "should": [
                            {"match": {"places": {"query": place, "boost": 2.0}}},
                            {"match_phrase": {"places": {"query": place, "boost": 3.0}}},
                            {"wildcard": {"places": {"value": f"*{place.lower()}*", "boost": 1.5}}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
            })

        try:
            response = await self.client.msearch(index=self.index_name, body=body)
        except Exception as e:
            logger.error(f"ES place search error: {e}")
            return []

        results = []
        for resp in response.get("responses", []):
            if "error" in resp:
                continue
            for hit in resp.get("hits", {}).get("hits", []):
                results.append({
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "metadata": hit["_source"].get("metadata", {}),
                })

        seen = {}
        for r in results:
            if r["id"] not in seen or r["score"] > seen[r["id"]]["score"]:
                seen[r["id"]] = r

        deduped = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return deduped[:limit]

    async def audio_search(self, queries: List[str], limit: int = 200) -> List[Dict]:
        """Search audio transcripts."""
        return await self.multi_field_search(queries, ["transcript"], limit)

    async def full_text_search(self, queries: List[str], limit: int = 200) -> List[Dict]:
        """Search across all text fields (caption, OCR, objects, places, transcript)."""
        return await self.multi_field_search(
            queries,
            ["caption", "text", "objects", "places", "transcript", "actions"],
            limit,
        )


# Singleton
_es_service: Optional[ElasticSearchService] = None


def get_es_service() -> ElasticSearchService:
    """Get or create ES service singleton."""
    global _es_service
    if _es_service is None:
        _es_service = ElasticSearchService()
    return _es_service
