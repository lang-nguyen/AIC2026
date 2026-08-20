"""
Temporal Search Engine.
Finds sequences of events occurring in order within the same video.
Key differentiator for competition - similar to NII-UIT VBS 2025 approach.
"""

import logging
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict

from app.services.clip_encoder import get_clip_encoder
from app.services.vector_db import get_vector_db

logger = logging.getLogger(__name__)


async def temporal_search(
    event_queries: List[str],
    limit: int = 100,
    max_time_gap: float = 300.0,
) -> List[Dict]:
    """Search for temporal sequences of events.

    Algorithm:
    1. For each event, find top-K matching frames via CLIP
    2. Group results by video_id
    3. Find valid temporal chains: frame_A.time < frame_B.time (within gap)
    4. Score = sum of individual CLIP scores
    5. Return top matches

    Args:
        event_queries: Ordered list of event descriptions.
        limit: Max temporal matches to return.
        max_time_gap: Maximum seconds allowed between consecutive events.

    Returns:
        List of temporal matches, each containing a video_id and ordered frames.
    """
    encoder = get_clip_encoder()
    vector_db = get_vector_db()

    if len(event_queries) < 2:
        return []

    # Step 1: Search for each event independently
    per_event_results = []
    for query in event_queries:
        embedding = encoder.encode_text([query])
        results = vector_db.search(embedding, limit=500)
        per_event_results.append(results)

    # Step 2: Group by video_id
    video_frames: Dict[str, List[List[Dict]]] = defaultdict(lambda: [[] for _ in range(len(event_queries))])

    for event_idx, results in enumerate(per_event_results):
        for r in results:
            vid = r["metadata"].get("video_id", "")
            if vid:
                video_frames[vid][event_idx].append(r)

    # Step 3: Find valid temporal chains
    matches = []

    for video_id, event_frame_lists in video_frames.items():
        # Check if this video has candidates for ALL events
        if any(len(frames) == 0 for frames in event_frame_lists):
            continue

        # Sort each event's frames by pts_time
        for i in range(len(event_frame_lists)):
            event_frame_lists[i].sort(
                key=lambda x: x["metadata"].get("pts_time", 0)
            )

        # Find valid chains using greedy approach
        best_chain = _find_best_chain(event_frame_lists, max_time_gap)
        if best_chain:
            total_score = sum(frame["score"] for frame in best_chain)
            matches.append({
                "video_id": video_id,
                "frames": best_chain,
                "total_score": total_score,
            })

    # Step 4: Sort by total score
    matches.sort(key=lambda x: x["total_score"], reverse=True)
    return matches[:limit]


def _find_best_chain(
    event_frame_lists: List[List[Dict]],
    max_time_gap: float,
) -> Optional[List[Dict]]:
    """Find the best temporal chain across events.

    Uses dynamic programming approach: for each candidate of event[0],
    find the best subsequent match for event[1], etc.
    """
    n_events = len(event_frame_lists)
    best_chain = None
    best_score = -1

    # Try top candidates for the first event
    for first_frame in event_frame_lists[0][:10]:
        chain = [first_frame]
        prev_time = first_frame["metadata"].get("pts_time", 0)

        valid = True
        for event_idx in range(1, n_events):
            # Find best frame for this event that comes AFTER prev_time
            best_next = None
            best_next_score = -1

            for candidate in event_frame_lists[event_idx]:
                cand_time = candidate["metadata"].get("pts_time", 0)
                time_diff = cand_time - prev_time

                if time_diff > 0 and time_diff <= max_time_gap:
                    # Prefer higher CLIP score with slight preference for closer time
                    adjusted_score = candidate["score"] * (1.0 - 0.001 * time_diff)
                    if adjusted_score > best_next_score:
                        best_next = candidate
                        best_next_score = adjusted_score

            if best_next is None:
                valid = False
                break

            chain.append(best_next)
            prev_time = best_next["metadata"].get("pts_time", 0)

        if valid:
            chain_score = sum(f["score"] for f in chain)
            if chain_score > best_score:
                best_score = chain_score
                best_chain = chain

    return best_chain
