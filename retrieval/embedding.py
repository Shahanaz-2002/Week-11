import numpy as np
import torch
import logging
import json
import time
import re
from typing import List

from transformers import (
    AutoTokenizer,
    AutoModel
)

logger = logging.getLogger(__name__)


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    force=True
)


# =========================================================
# LOGGING HELPER
# =========================================================

def log_event(event_type, message, extra=None):

    log_data = {
        "event": event_type,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if extra:
        log_data.update(extra)

    logger.info(json.dumps(log_data))


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.strip()

    text = re.sub(r"\s+", " ", text)

    return text


# =========================================================
# CLINICAL KEYWORD ENHANCEMENT
# =========================================================

def enrich_clinical_text(text):

    text = clean_text(text)

    clinical_keywords = [

        "pain",
        "swelling",
        "fracture",
        "injury",
        "stiffness",
        "mobility",
        "inflammation",
        "muscle",
        "joint",
        "sprain",
        "tenderness",
        "weakness",
        "arthritis",
        "back pain",
        "knee pain",
        "shoulder pain",
        "neck pain",
        "posture",
        "movement restriction",
        "functional limitation",
        "physical examination",
        "objective findings",
        "clinical history",
        "range of motion",
        "sports injury",
        "ligament injury",
        "rehabilitation"
    ]

    lower_text = text.lower()

    matched_keywords = []

    for keyword in clinical_keywords:

        if keyword in lower_text:

            matched_keywords.append(keyword)

    matched_keywords = list(set(matched_keywords))

    if matched_keywords:

        text += (
            " | " +
            " ".join(matched_keywords)
        )

    return text


# =========================================================
# BIOBERT EMBEDDING CLASS
# =========================================================

class BioBERTEmbedding:

    _instance = None

    # =====================================================
    # SINGLETON INITIALIZATION
    # =====================================================

    def __new__(cls):

        if cls._instance is None:

            cls._instance = super(
                BioBERTEmbedding,
                cls
            ).__new__(cls)

            try:

                # -------------------------------------------------
                # DEVICE
                # -------------------------------------------------

                cls._instance.device = torch.device(

                    "cuda"

                    if torch.cuda.is_available()

                    else "cpu"
                )

                # -------------------------------------------------
                # MODEL NAME
                # -------------------------------------------------

                cls._instance.MODEL_NAME = (
                    "emilyalsentzer/Bio_ClinicalBERT"
                )

                print(
                    "\n🔄 Loading BioClinicalBERT model..."
                )

                log_event(
                    "model_loading",
                    "Loading BioClinicalBERT model"
                )

                # -------------------------------------------------
                # TOKENIZER
                # -------------------------------------------------

                cls._instance.tokenizer = (
                    AutoTokenizer.from_pretrained(
                        cls._instance.MODEL_NAME
                    )
                )

                # -------------------------------------------------
                # MODEL
                # -------------------------------------------------

                cls._instance.model = (
                    AutoModel.from_pretrained(
                        cls._instance.MODEL_NAME
                    )
                )

                cls._instance.model.to(
                    cls._instance.device
                )

                cls._instance.model.eval()

                print(
                    "✅ BioClinicalBERT loaded successfully\n"
                )

                log_event(
                    "model_loaded",
                    "BioClinicalBERT loaded successfully",
                    {
                        "device":
                            str(
                                cls._instance.device
                            )
                    }
                )

            except Exception as e:

                log_event(
                    "model_load_error",
                    "Failed to load BioClinicalBERT",
                    {
                        "error": str(e)
                    }
                )

                raise e

        return cls._instance

    # =====================================================
    # MEAN POOLING
    # =====================================================

    def mean_pooling(
        self,
        token_embeddings,
        attention_mask
    ):

        mask = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()

        masked_embeddings = (
            token_embeddings * mask
        )

        summed_embeddings = torch.sum(
            masked_embeddings,
            dim=1
        )

        summed_mask = torch.clamp(
            mask.sum(dim=1),
            min=1e-9
        )

        pooled_embedding = (
            summed_embeddings / summed_mask
        )

        return pooled_embedding

    # =====================================================
    # GENERATE SINGLE EMBEDDING
    # =====================================================

    def get_embedding(
        self,
        text: str
    ) -> np.ndarray:

        start_time = time.time()

        # -------------------------------------------------
        # INPUT VALIDATION
        # -------------------------------------------------

        if not text:

            raise ValueError(
                "Input text is empty"
            )

        if not isinstance(text, str):

            raise ValueError(
                "Input text must be string"
            )

        text = clean_text(text)

        if not text.strip():

            raise ValueError(
                "Input text became empty after cleaning"
            )

        # -------------------------------------------------
        # CLINICAL ENRICHMENT
        # -------------------------------------------------

        enriched_text = enrich_clinical_text(
            text
        )

        log_event(
            "text_enriched",
            "Clinical text enhanced",
            {
                "original_length":
                    len(text),

                "enhanced_length":
                    len(enriched_text)
            }
        )

        # -------------------------------------------------
        # TOKENIZATION
        # -------------------------------------------------

        try:

            inputs = self.tokenizer(

                enriched_text,

                return_tensors="pt",

                truncation=True,

                padding=True,

                max_length=512
            )

            inputs = {

                k: v.to(self.device)

                for k, v in inputs.items()
            }

        except Exception as e:

            log_event(
                "tokenization_error",
                "Error during tokenization",
                {
                    "error": str(e)
                }
            )

            raise e

        # -------------------------------------------------
        # MODEL INFERENCE
        # -------------------------------------------------

        try:

            with torch.no_grad():

                outputs = self.model(
                    **inputs
                )

        except Exception as e:

            log_event(
                "model_inference_error",
                "Model inference failed",
                {
                    "error": str(e)
                }
            )

            raise e

        # -------------------------------------------------
        # MEAN POOLING
        # -------------------------------------------------

        try:

            token_embeddings = (
                outputs.last_hidden_state
            )

            attention_mask = (
                inputs["attention_mask"]
            )

            pooled_embedding = self.mean_pooling(

                token_embeddings,

                attention_mask
            )

            embedding = (

                pooled_embedding

                .cpu()

                .numpy()[0]
            )

        except Exception as e:

            log_event(
                "pooling_error",
                "Mean pooling failed",
                {
                    "error": str(e)
                }
            )

            raise e

        # -------------------------------------------------
        # NORMALIZATION
        # -------------------------------------------------

        try:

            norm = np.linalg.norm(
                embedding
            )

            if norm == 0:

                raise ValueError(
                    "Zero norm embedding encountered"
                )

            embedding = embedding / norm

        except Exception as e:

            log_event(
                "normalization_error",
                "Embedding normalization failed",
                {
                    "error": str(e)
                }
            )

            raise e

        # -------------------------------------------------
        # FINAL LOGGING
        # -------------------------------------------------

        total_time = round(
            (
                time.time() -
                start_time
            ) * 1000,
            2
        )

        log_event(
            "embedding_generated",
            "Clinical embedding generated successfully",
            {
                "embedding_dimension":
                    len(embedding),

                "processing_time_ms":
                    total_time
            }
        )

        return embedding

    # =====================================================
    # GENERATE BATCH EMBEDDINGS
    # =====================================================

    def get_batch_embeddings(
        self,
        texts: List[str]
    ) -> List[np.ndarray]:

        embeddings = []

        if not isinstance(texts, list):

            raise ValueError(
                "texts must be a list"
            )

        for index, text in enumerate(texts):

            try:

                embedding = self.get_embedding(
                    text
                )

                embeddings.append(
                    embedding
                )

            except Exception as e:

                log_event(
                    "batch_embedding_error",
                    "Error generating batch embedding",
                    {
                        "index": index,
                        "error": str(e)
                    }
                )

                continue

        log_event(
            "batch_embedding_completed",
            "Batch embedding generation completed",
            {
                "total_inputs":
                    len(texts),

                "successful_embeddings":
                    len(embeddings)
            }
        )

        return embeddings

    # =====================================================
    # EMBEDDING DIMENSION
    # =====================================================

    def get_embedding_dimension(self):

        try:

            sample_embedding = self.get_embedding(
                "sample clinical text"
            )

            return len(sample_embedding)

        except Exception:

            return 768