"""
FAISS Vector Database Service.
Uses FAISS for fast vector similarity search - no Docker/server needed.
"""

import faiss
import numpy as np
import json
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from glob import glob

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FAISSVectorDB:
    """FAISS-based vector search engine with metadata storage."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.id_map: List[str] = []  # Maps FAISS internal idx → frame_id
        self.metadata: Dict[str, dict] = {}  # frame_id → metadata
        self.embed_dim: int = 512  # CLIP ViT-B/32

    def build_index(
        self,
        features_dir: str,
        map_keyframes_dir: str,
        media_info_dir: str,
    ):
        """Build FAISS index from pre-computed CLIP features.

        Args:
            features_dir: Directory containing .npy files (one per video).
            map_keyframes_dir: Directory containing keyframe CSV mappings.
            media_info_dir: Directory containing media info JSON files.
        """
        logger.info("Building FAISS index from CLIP features...")
        all_embeddings = []
        all_ids = []

        # Discover all .npy feature files
        npy_files = sorted(glob(os.path.join(features_dir, "**", "*.npy"), recursive=True))
        if not npy_files:
            npy_files = sorted(glob(os.path.join(features_dir, "*.npy")))

        logger.info(f"Found {len(npy_files)} feature files")

        for npy_path in npy_files:
            video_id = Path(npy_path).stem  # e.g., K01_V001
            features = np.load(npy_path).astype(np.float32)

            # Load keyframe mapping
            csv_path = os.path.join(map_keyframes_dir, f"{video_id}.csv")
            if not os.path.exists(csv_path):
                continue

            import csv
            keyframes = []
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    keyframes.append(row)

            # Load media info
            youtube_id = ""
            media_path = os.path.join(media_info_dir, f"{video_id}.json")
            if os.path.exists(media_path):
                try:
                    with open(media_path, "r", encoding="utf-8") as f:
                        media_data = json.load(f)
                        watch_url = media_data.get("watch_url", "")
                        youtube_id = watch_url.rsplit("=", 1)[-1] if "=" in watch_url else ""
                except Exception:
                    pass

            # Match features with keyframes
            group = video_id.split("_")[0]  # K01, L21, etc.
            num_kf = min(len(features), len(keyframes))

            for i in range(num_kf):
                kf = keyframes[i]
                n = int(kf["n"])
                frame_id = f"{video_id}_{n:03d}"

                self.metadata[frame_id] = {
                    "frame_idx": int(kf.get("frame_idx", 0)),
                    "pts_time": float(kf.get("pts_time", 0.0)),
                    "video_id": video_id,
                    "youtube_id": youtube_id,
                    "n": n,
                    "group": group,
                }

                all_ids.append(frame_id)
                all_embeddings.append(features[i])

        if not all_embeddings:
            logger.warning("No embeddings found!")
            return

        # Stack and normalize
        embeddings_matrix = np.stack(all_embeddings).astype(np.float32)
        faiss.normalize_L2(embeddings_matrix)

        self.embed_dim = embeddings_matrix.shape[1]
        self.id_map = all_ids

        # Build FAISS index with Inner Product (cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(embeddings_matrix)

        logger.info(
            f"FAISS index built: {self.index.ntotal} vectors, dim={self.embed_dim}"
        )

    def save(self, path: str):
        """Save index and metadata to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "id_map.json"), "w") as f:
            json.dump(self.id_map, f)
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(self.metadata, f)
        logger.info(f"FAISS index saved to {path}")

    def load(self, path: str) -> bool:
        """Load index and metadata from disk."""
        index_path = os.path.join(path, "index.faiss")
        if not os.path.exists(index_path):
            logger.warning(f"No FAISS index found at {path}")
            return False

        self.index = faiss.read_index(index_path)
        with open(os.path.join(path, "id_map.json"), "r") as f:
            self.id_map = json.load(f)
        with open(os.path.join(path, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

        self.embed_dim = self.index.d
        logger.info(
            f"FAISS index loaded: {self.index.ntotal} vectors, dim={self.embed_dim}"
        )
        return True

    def search(
        self,
        query_embedding: np.ndarray,
        limit: int = 200,
        filter_fn=None,
    ) -> List[Dict]:
        """Search for similar vectors.

        Args:
            query_embedding: Query vector(s), shape (N, dim) or (dim,).
            limit: Max results per query.
            filter_fn: Optional callable that filters results by metadata.

        Returns:
            List of search result dicts.
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Search more if we need to filter
        search_limit = limit * 3 if filter_fn else limit

        scores, indices = self.index.search(query_embedding, min(search_limit, self.index.ntotal))

        results = []
        seen = set()

        for q_idx in range(len(scores)):
            for rank in range(len(scores[q_idx])):
                idx = int(indices[q_idx][rank])
                score = float(scores[q_idx][rank])

                if idx < 0 or idx >= len(self.id_map):
                    continue

                frame_id = self.id_map[idx]
                if frame_id in seen:
                    continue
                seen.add(frame_id)

                meta = self.metadata.get(frame_id, {})

                if filter_fn and not filter_fn(meta):
                    continue

                results.append({
                    "id": frame_id,
                    "score": score,
                    "metadata": meta,
                })

                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break

        return results

    def get_metadata(self, frame_id: str) -> Optional[dict]:
        """Get metadata for a specific frame."""
        return self.metadata.get(frame_id)

    def get_nearby_frames(self, frame_id: str, range_n: int = 40) -> List[Dict]:
        """Get frames near the given frame in the same video."""
        meta = self.metadata.get(frame_id)
        if not meta:
            return []

        video_id = meta["video_id"]
        n = meta["n"]

        results = []
        for offset in range(-range_n, range_n + 1):
            nearby_n = n + offset
            if nearby_n < 1:
                continue
            nearby_id = f"{video_id}_{nearby_n:03d}"
            nearby_meta = self.metadata.get(nearby_id)
            if nearby_meta:
                results.append({
                    "id": nearby_id,
                    "metadata": nearby_meta,
                })

        return results

    def get_frames_by_video(self, video_id: str) -> List[Dict]:
        """Get all frames for a given video."""
        results = []
        for fid, meta in self.metadata.items():
            if meta.get("video_id") == video_id:
                results.append({"id": fid, "metadata": meta})
        results.sort(key=lambda x: x["metadata"].get("n", 0))
        return results


# Singleton instance
_vector_db: Optional[FAISSVectorDB] = None


def get_vector_db() -> FAISSVectorDB:
    """Get or initialize the vector DB singleton."""
    global _vector_db
    if _vector_db is None:
        _vector_db = FAISSVectorDB()
        # Try to load from disk
        if not _vector_db.load(settings.FAISS_INDEX_PATH):
            logger.info("No pre-built index found. Will build on first request or run indexing script.")
    return _vector_db
