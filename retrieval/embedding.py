import numpy as np
import torch
import logging
import json
import time
import re

from transformers import (
    AutoTokenizer,
    AutoModel
)

logger = logging.getLogger(__name__)


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

    if not text:
        return ""

    text = str(text)

    text = text.strip()

    text = re.sub(r"\s+", " ", text)

    return text


# =========================================================
# DERMATOLOGY KEYWORD ENHANCEMENT
# =========================================================

def enrich_dermatology_text(text):

    text = clean_text(text)

    dermatology_keywords = [
        "acne",
        "eczema",
        "psoriasis",
        "rash",
        "lesion",
        "pigmentation",
        "hyperpigmentation",
        "papules",
        "pustules",
        "itching",
        "skin inflammation",
        "dermatitis",
        "redness",
        "dry skin",
        "fungal infection"
    ]

    lower_text = text.lower()

    matched_keywords = []

    for keyword in dermatology_keywords:

        if keyword in lower_text:
            matched_keywords.append(keyword)

    if matched_keywords:

        text += " | " + " ".join(matched_keywords)

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
                    "🔄 Loading BioClinicalBERT model "
                    "(only once)..."
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

                log_event(
                    "model_loaded",
                    "BioClinicalBERT loaded successfully",
                    {
                        "device":
                            str(cls._instance.device)
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

        masked_embeddings = token_embeddings * mask

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
    # GENERATE EMBEDDING
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
        # DERMATOLOGY ENRICHMENT
        # -------------------------------------------------

        enriched_text = enrich_dermatology_text(
            text
        )

        log_event(
            "text_enriched",
            "Dermatology text enhanced",
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

                outputs = self.model(**inputs)

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
            (time.time() - start_time) * 1000,
            2
        )

        log_event(
            "embedding_generated",
            "Dermatology embedding generated successfully",
            {
                "embedding_dimension":
                    len(embedding),

                "processing_time_ms":
                    total_time
            }
        )

        return embedding

    # =====================================================
    # BATCH EMBEDDINGS
    # =====================================================

    def get_batch_embeddings(
        self,
        texts
    ):

        embeddings = []

        for text in texts:

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
                        "error": str(e)
                    }
                )

                continue

        return embeddings