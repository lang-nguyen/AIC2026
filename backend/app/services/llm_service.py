"""
LLM Service - Gemini API integration for:
- Query understanding and enhancement
- Translation (Vietnamese → English)
- Reranking with visual QA
- Answer generation
"""

import logging
import json
from typing import List, Optional, Dict
import google.generativeai as genai

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMService:
    """Gemini-powered LLM service."""

    def __init__(self):
        if settings.GOOGLE_API_KEY:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(settings.LLM_MODEL)
            logger.info(f"LLM service initialized with {settings.LLM_MODEL}")
        else:
            self.model = None
            logger.warning("GOOGLE_API_KEY not set, LLM service disabled")

    async def enhance_queries(self, queries: List[str]) -> List[str]:
        """Enhance search queries using LLM.

        Translates Vietnamese to English, expands queries for CLIP search,
        removes noise, generates sub-queries.

        Args:
            queries: Original user queries (may be in Vietnamese).

        Returns:
            List of enhanced English queries optimized for CLIP search.
        """
        if not self.model or not queries:
            return queries

        prompt = f"""You are an AI assistant for a video/image retrieval system using CLIP model.
Your task: convert user queries into optimized English search queries for CLIP-based visual retrieval.

Rules:
1. Translate Vietnamese to English accurately with visual context
2. Break complex descriptions into 2-3 focused sub-queries
3. Focus on visual elements: objects, actions, colors, scenes, people
4. Remove irrelevant narrative/noise that doesn't help visual search
5. Keep queries concise (under 77 tokens each for CLIP)
6. Return ONLY a JSON array of strings

Input queries: {json.dumps(queries, ensure_ascii=False)}

Output (JSON array of enhanced English queries):"""

        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            result = json.loads(response.text)
            if isinstance(result, list):
                logger.info(f"Enhanced queries: {queries} → {result}")
                return result
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")

        return queries

    async def parse_auto_query(self, query_text: str) -> Dict:
        """Parse competition query for automatic mode.

        Extracts search intent, entities, and generates search strategy.

        Args:
            query_text: Raw query text from competition.

        Returns:
            Parsed query with search strategy.
        """
        if not self.model:
            return {"text_queries": [query_text], "search_mode": "hybrid"}

        prompt = f"""You are parsing a multimedia retrieval competition query.
Analyze the query and extract structured search information.
CRITICAL INSTRUCTION: All extracted queries (text_queries, object_queries, place_queries, temporal_events) MUST be translated to English, because the database only understands English. Only ocr_queries should be kept in the original language if it's looking for exact text on screen.

Query: {query_text}

Return a JSON object with:
{{
    "text_queries": ["list of English text queries for CLIP search"],
    "ocr_queries": ["exact text appearing on screen, keep original language"],
    "object_queries": ["list of specific objects in English"],
    "place_queries": ["list of locations/places in English"],
    "temporal_events": ["event1 in English", "event2 in English"] or null,
    "search_mode": "hybrid" | "temporal" | "ocr",
    "answer_type": "frame" | "text" | "count" | null,
    "is_temporal": true/false
}}"""

        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            result = json.loads(response.text)
            return result
        except Exception as e:
            logger.error(f"Auto query parsing failed: {e}")
            return {"text_queries": [query_text], "search_mode": "hybrid"}

    async def chat_response(
        self,
        question: str,
        context: Optional[str] = None,
    ) -> str:
        """Generate chat response.

        Args:
            question: User's question.
            context: Optional context from search results.

        Returns:
            AI-generated response text.
        """
        if not self.model:
            return "LLM service is not available. Please configure GOOGLE_API_KEY."

        system_prompt = """Bạn là trợ lý AI thông minh cho hệ thống truy xuất multimedia.
Nhiệm vụ: hỗ trợ người dùng tìm kiếm nội dung video/hình ảnh.
- Trả lời ngắn gọn, hữu ích
- Nếu người dùng mô tả cảnh, gợi ý từ khóa tìm kiếm
- Hỗ trợ bằng cả Tiếng Việt và Tiếng Anh"""

        messages = [system_prompt]
        if context:
            messages.append(f"Context from current search results:\n{context}")
        messages.append(f"User: {question}")

        try:
            response = await self.model.generate_content_async(
                "\n\n".join(messages),
                generation_config=genai.types.GenerationConfig(temperature=0.3),
            )
            return response.text
        except Exception as e:
            logger.error(f"Chat response failed: {e}")
            return f"Xin lỗi, có lỗi xảy ra: {str(e)}"

    async def translate_to_english(self, texts: List[str]) -> List[str]:
        """Translate texts to English using LLM (better context than Google Translate)."""
        if not self.model or not texts:
            return texts

        # Check if already English
        has_vietnamese = any(
            any("\u00c0" <= char <= "\u1ef9" for char in text)
            for text in texts
        )
        if not has_vietnamese:
            return texts

        prompt = f"""Translate the following Vietnamese texts to English accurately.
Keep visual descriptions precise. Return ONLY a JSON array of translated strings.

Input: {json.dumps(texts, ensure_ascii=False)}

Output (JSON array):"""

        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            result = json.loads(response.text)
            if isinstance(result, list) and len(result) == len(texts):
                return result
        except Exception as e:
            logger.error(f"Translation failed: {e}")

        return texts


# Singleton
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
