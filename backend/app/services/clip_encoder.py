"""
CLIP encoder service.
Handles text and image encoding using OpenAI CLIP model.
Runs on CPU (sufficient for text queries at competition time).
"""

import clip
import torch
import numpy as np
from PIL import Image
import io
import logging
from functools import lru_cache
from typing import List, Optional, Union

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class CLIPEncoder:
    """Singleton CLIP encoder for text and image embedding."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "model", None) is not None:
            return
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading CLIP model {settings.CLIP_MODEL} on {self.device}...")
        try:
            self.model, self.preprocess = clip.load(
                settings.CLIP_MODEL,
                device=self.device,
                download_root=settings.CLIP_DOWNLOAD_ROOT
            )
            self.model.eval()
            self.embed_dim = self.model.visual.output_dim
            logger.info(f"CLIP model loaded. Embedding dim: {self.embed_dim}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise e

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """Encode text queries to embeddings.

        Args:
            texts: List of text strings.

        Returns:
            numpy array of shape (N, embed_dim), normalized.
        """
        if not texts:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        tokens = clip.tokenize(texts, truncate=True).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_text(tokens)
            # L2 normalize for cosine similarity
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().astype(np.float32)

    def encode_image(self, image_data: Union[bytes, Image.Image]) -> np.ndarray:
        """Encode a single image to embedding.

        Args:
            image_data: Raw image bytes or PIL Image.

        Returns:
            numpy array of shape (1, embed_dim), normalized.
        """
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            image = image_data.convert("RGB")

        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().astype(np.float32)

    def encode_images(self, images: List[Union[bytes, Image.Image]]) -> np.ndarray:
        """Encode multiple images to embeddings.

        Args:
            images: List of raw image bytes or PIL Images.

        Returns:
            numpy array of shape (N, embed_dim), normalized.
        """
        if not images:
            return np.empty((0, self.embed_dim), dtype=np.float32)

        processed = []
        for img_data in images:
            if isinstance(img_data, bytes):
                img = Image.open(io.BytesIO(img_data)).convert("RGB")
            else:
                img = img_data.convert("RGB")
            processed.append(self.preprocess(img))

        batch = torch.stack(processed).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_image(batch)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.cpu().numpy().astype(np.float32)


@lru_cache()
def get_clip_encoder() -> CLIPEncoder:
    """Get singleton CLIP encoder instance."""
    return CLIPEncoder()
