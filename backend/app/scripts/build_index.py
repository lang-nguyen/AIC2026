"""
Build FAISS index from pre-computed CLIP features.
Run this script first before starting the server.

Usage:
    cd backend
    python -m app.scripts.build_index
"""

import sys
import os
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.config import get_settings
from app.services.vector_db import FAISSVectorDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    settings = get_settings()

    db = FAISSVectorDB()

    logger.info("=" * 60)
    logger.info("Building FAISS index from CLIP features")
    logger.info(f"  Features dir: {settings.CLIP_FEATURES_DIR}")
    logger.info(f"  Map keyframes dir: {settings.MAP_KEYFRAMES_DIR}")
    logger.info(f"  Media info dir: {settings.MEDIA_INFO_DIR}")
    logger.info("=" * 60)

    db.build_index(
        features_dir=settings.CLIP_FEATURES_DIR,
        map_keyframes_dir=settings.MAP_KEYFRAMES_DIR,
        media_info_dir=settings.MEDIA_INFO_DIR,
    )

    if db.index and db.index.ntotal > 0:
        logger.info(f"✅ Index built: {db.index.ntotal} vectors")

        # Save
        os.makedirs(settings.FAISS_INDEX_PATH, exist_ok=True)
        db.save(settings.FAISS_INDEX_PATH)
        logger.info(f"✅ Index saved to: {settings.FAISS_INDEX_PATH}")
    else:
        logger.error("❌ Failed to build index. Check data paths.")


if __name__ == "__main__":
    main()
