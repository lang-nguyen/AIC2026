"""
Build Elasticsearch Index from OCR, Objects, and Metadata.
"""
import asyncio
import json
import os
import csv
import logging
from glob import glob
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

from app.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()

async def main():
    if not settings.ELASTICSEARCH_URL:
        logger.error("ELASTICSEARCH_URL not set in .env")
        return

    # Connect to Elasticsearch
    client = AsyncElasticsearch(
        hosts=settings.ELASTICSEARCH_URL,
        api_key=settings.ELASTICSEARCH_API_KEY if settings.ELASTICSEARCH_API_KEY else None,
    )
    
    index_name = settings.ELASTICSEARCH_INDEX_NAME

    try:
        info = await client.info()
        logger.info(f"Connected to ES: {info['version']['number']}")
    except Exception as e:
        logger.error(f"Cannot connect to ES: {e}")
        return

    # 1. Create Index with mapping
    mapping = {
        "mappings": {
            "properties": {
                "objects": {"type": "text"},
                "text": {"type": "text"},  # OCR
                "caption": {"type": "text"},
                "places": {"type": "text"},
                "transcript": {"type": "text"},
                "metadata": {"type": "object"}
            }
        }
    }
    
    exists = await client.indices.exists(index=index_name)
    if not exists:
        await client.indices.create(index=index_name, body=mapping)
        logger.info(f"Created index: {index_name}")
    else:
        logger.info(f"Index {index_name} already exists. Updating documents...")

    # Dictionary to hold the data before bulk index
    # frame_id -> dict of fields
    documents = {}

    def get_doc(fid):
        if fid not in documents:
            documents[fid] = {
                "_index": index_name,
                "_id": fid,
                "objects": "",
                "text": "",
                "metadata": {"frame_id": fid}
            }
        return documents[fid]

    # 2. Process Objects
    if settings.OBJECTS_DIR and os.path.exists(settings.OBJECTS_DIR):
        logger.info(f"Processing objects from {settings.OBJECTS_DIR}...")
        json_files = glob(os.path.join(settings.OBJECTS_DIR, "**", "*.json"), recursive=True)
        logger.info(f"Found {len(json_files)} object JSON files.")
        
        for jpath in json_files:
            video_id = os.path.basename(os.path.dirname(jpath))
            frame_num = os.path.splitext(os.path.basename(jpath))[0]
            try:
                frame_n = int(frame_num)
                frame_id = f"{video_id}_{frame_n:03d}"
                with open(jpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    entities = data.get("detection_class_entities", [])
                    if entities:
                        # Join unique entities for keyword search
                        unique_entities = list(set(entities))
                        get_doc(frame_id)["objects"] = " ".join(unique_entities)
            except Exception as e:
                pass
    else:
        logger.warning("OBJECTS_DIR not found or empty.")

    # 3. Process OCR if exists
    if settings.OCR_CSV_PATH and os.path.exists(settings.OCR_CSV_PATH):
        logger.info(f"Processing OCR from {settings.OCR_CSV_PATH}...")
        try:
            with open(settings.OCR_CSV_PATH, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                count = 0
                for row in reader:
                    if len(row) >= 2:
                        frame_id = row[0]
                        text = row[1]
                        get_doc(frame_id)["text"] = text
                        count += 1
                logger.info(f"Loaded {count} OCR entries.")
        except Exception as e:
            logger.error(f"Error reading OCR: {e}")
    else:
        logger.info("OCR_CSV_PATH not found (Skipping OCR data).")

    # 4. Bulk insert
    docs_to_insert = list(documents.values())
    logger.info(f"Ready to insert {len(docs_to_insert)} documents into {index_name}...")
    
    if docs_to_insert:
        success, _ = await async_bulk(client, docs_to_insert, chunk_size=1000)
        logger.info(f"✅ Successfully inserted {success} documents into Elasticsearch!")
    else:
        logger.info("No documents to insert.")

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
