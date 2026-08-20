"""
Application configuration using Pydantic Settings.
All config loaded from .env file or environment variables.
"""

import os
from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- App ---
    APP_NAME: str = "AIC 2026 - Intelligent Multimedia Retrieval"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- Data Paths ---
    KEYFRAMES_DIR: str = "D:/AIC/AIC2025/static/frames"
    CLIP_FEATURES_DIR: str = "D:/AIC/AIC2025/data/clip-features-32"
    MAP_KEYFRAMES_DIR: str = "D:/AIC/AIC2025/data/map-keyframes"
    MEDIA_INFO_DIR: str = "D:/AIC/AIC2025/data/media-info"
    OBJECTS_DIR: str = "D:/AIC/AIC2025/data/objects"
    OCR_CSV_PATH: str = "D:/AIC/AIC2025/data/ocr_results.csv"
    VIDEO_FPS_PATH: str = "D:/AIC/AIC2025/data/video_fps/video_fps.json"
    YOUTUBE_ID_PATH: str = "D:/AIC/AIC2025/data/youtube_id/youtube.json"
    FAISS_INDEX_PATH: str = "D:/AIC/AIC2026/backend/data/faiss_index"

    # --- Elasticsearch ---
    ELASTICSEARCH_URL: str = ""
    ELASTICSEARCH_API_KEY: str = ""
    ELASTICSEARCH_INDEX_NAME: str = "aic-2026"

    # --- Google API ---
    GOOGLE_API_KEY: str = ""
    GOOGLE_TRANSLATE_API_KEY: str = ""

    # --- CLIP ---
    CLIP_MODEL: str = "ViT-B/32"
    CLIP_DOWNLOAD_ROOT: str = "D:/AIC/AIC2025/AI_Models/CLIP"

    # --- Search ---
    DEFAULT_SEARCH_LIMIT: int = 200
    MAX_SEARCH_LIMIT: int = 2000
    RRF_K: int = 60
    RERANK_TOP_N: int = 30

    # --- LLM ---
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
