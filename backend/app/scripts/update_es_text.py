import asyncio
import csv
import os
import logging
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from app.config import get_settings
from app.services.vector_db import get_vector_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()

async def update_ocr(client, index_name):
    """Updates only the 'text' field from OCR CSV."""
    if not settings.OCR_CSV_PATH or not os.path.exists(settings.OCR_CSV_PATH):
        logger.warning(f"OCR CSV not found at {settings.OCR_CSV_PATH}. Skipping.")
        return 0

    logger.info(f"Preparing to inject OCR data from {settings.OCR_CSV_PATH}...")
    
    def generate_ocr_updates():
        with open(settings.OCR_CSV_PATH, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip header if it exists
            first_row = True
            for row in reader:
                if not row or len(row) < 2: continue
                if first_row and "frame_id" in row[0].lower():
                    first_row = False
                    continue
                first_row = False
                
                frame_id = row[0]
                text = row[1]
                
                yield {
                    "_op_type": "update",
                    "_index": index_name,
                    "_id": frame_id,
                    "doc": {
                        "text": text
                    },
                    "upsert": {
                        "text": text,
                        "metadata": {"frame_id": frame_id}
                    }
                }
    
    try:
        success, _ = await async_bulk(client, generate_ocr_updates(), chunk_size=2000, request_timeout=60)
        logger.info(f"Successfully updated/upserted {success} OCR documents!")
        return success
    except Exception as e:
        logger.error(f"Error during bulk update: {e}")
        return 0

async def update_asr(client, index_name):
    """Updates the 'transcript' field from ASR CSV by broadcasting to all frames of the video."""
    # Try getting the path from settings, fallback if not there yet
    asr_path = getattr(settings, "ASR_CSV_PATH", "D:/AIC/AIC2026/backend/data/asr_results.csv")
    if not os.path.exists(asr_path):
        logger.warning(f"ASR CSV not found at {asr_path}. Skipping.")
        return 0

    logger.info(f"Preparing to inject ASR data from {asr_path}...")
    vector_db = get_vector_db()
    
    # 1. Group transcripts by video_id
    video_transcripts = {}
    with open(asr_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id")
            text = row.get("text")
            if vid and text:
                if vid not in video_transcripts:
                    video_transcripts[vid] = []
                video_transcripts[vid].append(text)
                
    if not video_transcripts:
        logger.warning("No ASR data parsed.")
        return 0
        
    # 2. Map video_id transcripts to all its frames
    logger.info(f"Found transcripts for {len(video_transcripts)} videos. Mapping to frames...")
    
    def generate_asr_updates():
        for vid, texts in video_transcripts.items():
            full_transcript = " ".join(texts)
            # Get all frames for this video from FAISS metadata
            frames = vector_db.get_frames_by_video(vid)
            for f in frames:
                frame_id = f["id"]
                yield {
                    "_op_type": "update",
                    "_index": index_name,
                    "_id": frame_id,
                    "doc": {
                        "transcript": full_transcript
                    },
                    "upsert": {
                        "transcript": full_transcript,
                        "metadata": {"frame_id": frame_id}
                    }
                }
                
    try:
        success, _ = await async_bulk(client, generate_asr_updates(), chunk_size=2000, request_timeout=60)
        logger.info(f"Successfully updated/upserted {success} ASR frame documents!")
        return success
    except Exception as e:
        logger.error(f"Error during bulk ASR update: {e}")
        return 0

async def main():
    if not settings.ELASTICSEARCH_URL:
        logger.error("ELASTICSEARCH_URL not set in .env")
        return

    client = AsyncElasticsearch(
        hosts=settings.ELASTICSEARCH_URL,
        api_key=settings.ELASTICSEARCH_API_KEY if settings.ELASTICSEARCH_API_KEY else None,
    )
    
    index_name = settings.ELASTICSEARCH_INDEX_NAME
    
    exists = await client.indices.exists(index=index_name)
    if not exists:
        logger.error(f"Index {index_name} does not exist! Please run build_es_index.py first.")
        await client.close()
        return

    logger.info("Starting rapid text update (OCR & ASR)...")
    
    await update_ocr(client, index_name)
    await update_asr(client, index_name)
    
    await client.close()
    logger.info("Rapid text update finished!")

if __name__ == "__main__":
    asyncio.run(main())
