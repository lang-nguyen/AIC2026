"""
AIC 2026 - Intelligent Virtual Assistant for Multimedia Retrieval
FastAPI Application Entry Point
"""

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.api.routes import router
from app.services.vector_db import get_vector_db
from app.services.elasticsearch_service import get_es_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # --- Startup ---
    logger.info(f"🚀 Starting {settings.APP_NAME}")

    # Initialize FAISS vector DB
    vector_db = get_vector_db()
    if vector_db.index is None or vector_db.index.ntotal == 0:
        logger.warning("⚠️  FAISS index not loaded. Run: python -m app.scripts.build_index")
    else:
        logger.info(f"✅ FAISS index: {vector_db.index.ntotal} vectors")

    # Initialize Elasticsearch
    es = get_es_service()
    await es.connect()

    logger.info("✅ All services initialized")

    yield

    # --- Shutdown ---
    logger.info("Shutting down...")
    await es.close()


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version="2026.1.0",
    description="Intelligent Virtual Assistant for multimedia big data retrieval",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (keyframe images)
frames_dir = settings.KEYFRAMES_DIR
if os.path.exists(frames_dir):
    app.mount("/static/frames", StaticFiles(directory=frames_dir), name="frames")
    logger.info(f"📁 Serving frames from: {frames_dir}")

# Include API routes
app.include_router(router)


# Root endpoint
@app.get("/")
async def root():
    return {
        "name": settings.APP_NAME,
        "version": "2026.1.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
