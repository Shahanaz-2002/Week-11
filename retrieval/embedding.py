# =========================================================
# IMPORTS
# =========================================================

from sentence_transformers import SentenceTransformer

import numpy as np

from typing import (
    List,
    Union
)

import logging

import torch


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(__name__)


# =========================================================
# BIOMEDICAL EMBEDDING ENGINE
# =========================================================

class BioBERTEmbedding:

    # =====================================================
    # SHARED MODEL INSTANCE
    # =====================================================

    _model = None

    # =====================================================
    # INITIALIZATION
    # =====================================================

    def __init__(self):

        try:

            # =============================================
            # LOAD MODEL ONLY ONCE
            # =============================================

            if BioBERTEmbedding._model is None:

                logger.info(
                    "Loading BioBERT embedding model..."
                )

                device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                )

                logger.info(
                    f"Using device: {device}"
                )

                BioBERTEmbedding._model = SentenceTransformer(

                    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",

                    device=device
                )

                logger.info(
                    "BioBERT model loaded successfully."
                )

            self.model = BioBERTEmbedding._model

            # =============================================
            # EMBEDDING DIMENSION
            # =============================================

            self.embedding_dimension = (

                self.model.get_sentence_embedding_dimension()
            )

        except Exception as error:

            logger.error(
                f"Error loading BioBERT model: {error}"
            )

            raise error

    # =====================================================
    # CLEAN TEXT
    # =====================================================

    def _clean_text(
        self,
        text: Union[str, None]
    ) -> str:

        if text is None:
            return ""

        return str(text).strip()

    # =====================================================
    # SINGLE TEXT EMBEDDING
    # =====================================================

    def encode(
        self,
        text: str
    ) -> np.ndarray:

        try:

            cleaned_text = self._clean_text(text)

            embedding = self.model.encode(

                cleaned_text,

                convert_to_numpy=True,

                normalize_embeddings=True,

                show_progress_bar=False
            )

            return embedding.astype(np.float32)

        except Exception as error:

            logger.error(
                f"Encoding failed: {error}"
            )

            return np.zeros(

                self.embedding_dimension,

                dtype=np.float32
            )

    # =====================================================
    # ALIAS METHOD
    # =====================================================

    # Some project files may expect this method
    # so we keep compatibility

    def generate_embedding(
        self,
        text: str
    ) -> np.ndarray:

        return self.encode(text)

    # =====================================================
    # MULTIPLE TEXT EMBEDDINGS
    # =====================================================

    def encode_batch(
        self,
        texts: List[str]
    ) -> np.ndarray:

        try:

            cleaned_texts = [

                self._clean_text(text)

                for text in texts
            ]

            embeddings = self.model.encode(

                cleaned_texts,

                convert_to_numpy=True,

                normalize_embeddings=True,

                show_progress_bar=False
            )

            return embeddings.astype(np.float32)

        except Exception as error:

            logger.error(
                f"Batch encoding failed: {error}"
            )

            return np.zeros(

                (
                    len(texts),
                    self.embedding_dimension
                ),

                dtype=np.float32
            )

    # =====================================================
    # COSINE SIMILARITY
    # =====================================================

    def cosine_similarity(

        self,

        embedding_1: np.ndarray,

        embedding_2: np.ndarray

    ) -> float:

        try:

            norm_1 = np.linalg.norm(
                embedding_1
            )

            norm_2 = np.linalg.norm(
                embedding_2
            )

            # =============================================
            # PREVENT DIVISION BY ZERO
            # =============================================

            if norm_1 == 0 or norm_2 == 0:

                return 0.0

            similarity = np.dot(

                embedding_1,
                embedding_2

            ) / (

                norm_1 * norm_2
            )

            return float(similarity)

        except Exception as error:

            logger.error(
                f"Similarity computation failed: {error}"
            )

            return 0.0

    # =====================================================
    # VECTOR DIMENSION
    # =====================================================

    def get_embedding_dimension(
        self
    ) -> int:

        return self.embedding_dimension


# =========================================================
# TEST BLOCK
# =========================================================

if __name__ == "__main__":

    model = BioBERTEmbedding()

    sample_text = (
        "Patient has severe lower back pain "
        "with difficulty walking."
    )

    embedding = model.encode(
        sample_text
    )

    print(
        "Embedding Shape:",
        embedding.shape
    )

    print(
        "Embedding Dimension:",
        model.get_embedding_dimension()
    )

    print(
        "Embedding Sample:",
        embedding[:10]
    )