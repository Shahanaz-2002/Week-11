from typing import List, Dict
import numpy as np
import logging
import json
import time

from retrieval.embedding import BioBERTEmbedding

logger = logging.getLogger(__name__)


# =========================================================
# SAFE EMBEDDER INITIALIZATION
# =========================================================

try:
    embedder = BioBERTEmbedding()

except Exception:

    embedder = None


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

    text = " ".join(text.split())

    return text


# =========================================================
# DERMATOLOGY QUERY ENHANCEMENT
# =========================================================

def enhance_dermatology_query(query_text):

    query_text = clean_text(query_text)

    dermatology_keywords = [
        "skin",
        "lesion",
        "rash",
        "pigmentation",
        "itching",
        "acne",
        "eczema",
        "psoriasis",
        "papules",
        "pustules",
        "hyperpigmentation"
    ]

    enhanced_query = query_text

    detected_keywords = []

    lower_query = query_text.lower()

    for keyword in dermatology_keywords:

        if keyword in lower_query:
            detected_keywords.append(keyword)

    if detected_keywords:

        enhanced_query += " | " + " ".join(detected_keywords)

    return enhanced_query.strip()


# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:

    try:

        norm_a = np.linalg.norm(a)

        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = np.dot(a, b) / (norm_a * norm_b)

        return float(similarity)

    except Exception as e:

        log_event(
            "cosine_error",
            "Error during cosine similarity",
            {
                "error": str(e)
            }
        )

        return 0.0


# =========================================================
# BUILD CASE SEARCH TEXT
# =========================================================

def build_case_search_text(case_data):

    text_parts = []

    fields = [
        "case_description",
        "chief_complaint",
        "symptoms",
        "physical_examination",
        "objective_findings",
        "doctor_notes",
        "resolution_notes",
        "category"
    ]

    for field in fields:

        value = case_data.get(field, "")

        if value:
            text_parts.append(str(value))

    return " | ".join(text_parts)


# =========================================================
# MAIN RETRIEVAL FUNCTION
# =========================================================

def retrieve_similar_cases(
    query_text: str,
    case_database: List[Dict],
    top_k: int = 3
) -> List[Dict]:

    start_time = time.time()

    # =====================================================
    # LOG RETRIEVAL START
    # =====================================================

    log_event(
        "retrieval_started",
        "Dermatology retrieval started",
        {
            "query_length": len(query_text)
            if query_text else 0,

            "database_size": len(case_database)
            if isinstance(case_database, list)
            else 0,

            "top_k": top_k
        }
    )

    # =====================================================
    # INPUT VALIDATION
    # =====================================================

    if not query_text:

        log_event(
            "validation_error",
            "Empty dermatology query received"
        )

        return []

    if not isinstance(query_text, str):

        log_event(
            "validation_error",
            "Query is not string"
        )

        return []

    query_text = clean_text(query_text)

    if not query_text.strip():

        log_event(
            "validation_error",
            "Query became empty after cleaning"
        )

        return []

    if not isinstance(case_database, list):

        log_event(
            "validation_error",
            "Invalid database format"
        )

        return []

    if not isinstance(top_k, int) or top_k <= 0:

        log_event(
            "top_k_warning",
            "Invalid top_k supplied, defaulting to 3"
        )

        top_k = 3

    # =====================================================
    # QUERY ENHANCEMENT
    # =====================================================

    enhanced_query = enhance_dermatology_query(
        query_text
    )

    log_event(
        "query_enhanced",
        "Dermatology query enhanced",
        {
            "enhanced_query": enhanced_query
        }
    )

    # =====================================================
    # EMBEDDING VALIDATION
    # =====================================================

    if embedder is None:

        log_event(
            "embedding_unavailable",
            "BioBERT embedder unavailable"
        )

        return []

    # =====================================================
    # GENERATE QUERY EMBEDDING
    # =====================================================

    try:

        embedding_start = time.time()

        query_embedding = embedder.get_embedding(
            enhanced_query
        )

        if query_embedding is None:

            log_event(
                "embedding_failure",
                "Query embedding generation failed"
            )

            return []

        query_embedding = np.array(query_embedding)

        if query_embedding.size == 0:

            log_event(
                "embedding_failure",
                "Empty query embedding generated"
            )

            return []

        embedding_time = round(
            (time.time() - embedding_start) * 1000,
            2
        )

        log_event(
            "embedding_generated",
            "Dermatology query embedding created",
            {
                "embedding_dimension":
                    len(query_embedding),

                "embedding_time_ms":
                    embedding_time
            }
        )

    except Exception as e:

        log_event(
            "embedding_error",
            "Embedding generation error",
            {
                "error": str(e)
            }
        )

        return []

    # =====================================================
    # SIMILARITY SEARCH
    # =====================================================

    results = []

    processed_cases = 0

    for case_data in case_database:

        if not isinstance(case_data, dict):

            log_event(
                "invalid_case_skipped",
                "Skipping invalid dermatology case"
            )

            continue

        try:

            # -------------------------------------------------
            # CASE EMBEDDING
            # -------------------------------------------------

            case_embedding = np.array(
                case_data.get("embedding", [])
            )

            if case_embedding.size == 0:
                continue

            # -------------------------------------------------
            # DIMENSION CHECK
            # -------------------------------------------------

            if len(query_embedding) != len(case_embedding):

                log_event(
                    "dimension_mismatch",
                    "Embedding dimension mismatch",
                    {
                        "query_dim":
                            len(query_embedding),

                        "case_dim":
                            len(case_embedding)
                    }
                )

                continue

            # -------------------------------------------------
            # COSINE SIMILARITY
            # -------------------------------------------------

            similarity_score = cosine_similarity(
                query_embedding,
                case_embedding
            )

            # -------------------------------------------------
            # BUILD SEARCHABLE TEXT
            # -------------------------------------------------

            searchable_text = build_case_search_text(
                case_data
            )

            # -------------------------------------------------
            # RESULT OBJECT
            # -------------------------------------------------

            results.append({

                "case_id":
                    case_data.get(
                        "case_id",
                        "Unknown"
                    ),

                "similarity":
                    round(similarity_score, 4),

                "category":
                    case_data.get(
                        "category",
                        "Dermatology"
                    ),

                "location":
                    case_data.get(
                        "location",
                        "Unknown"
                    ),

                "resolution_notes":
                    case_data.get(
                        "resolution_notes",
                        "No dermatologist notes available"
                    ),

                "case_description":
                    case_data.get(
                        "case_description",
                        ""
                    ),

                "searchable_text":
                    searchable_text
            })

            processed_cases += 1

        except Exception as e:

            log_event(
                "case_processing_error",
                "Error processing dermatology case",
                {
                    "error": str(e)
                }
            )

            continue

    # =====================================================
    # NO RESULTS
    # =====================================================

    if len(results) == 0:

        log_event(
            "no_results",
            "No dermatology similarity results generated"
        )

        return []

    # =====================================================
    # SORT RESULTS
    # =====================================================

    try:

        results = sorted(
            results,
            key=lambda x: x["similarity"],
            reverse=True
        )

        top_results = results[:top_k]

        log_event(
            "top_k_selected",
            "Top dermatology matches selected",
            {
                "top_k":
                    top_k,

                "top_case_ids":
                    [
                        r["case_id"]
                        for r in top_results
                    ],

                "top_scores":
                    [
                        r["similarity"]
                        for r in top_results
                    ]
            }
        )

    except Exception as e:

        log_event(
            "sorting_error",
            "Sorting failed",
            {
                "error": str(e)
            }
        )

        return []

    # =====================================================
    # TOTAL TIME
    # =====================================================

    total_time = round(
        (time.time() - start_time) * 1000,
        2
    )

    log_event(
        "retrieval_completed",
        "Dermatology retrieval completed successfully",
        {
            "processed_cases":
                processed_cases,

            "returned_cases":
                len(top_results),

            "total_time_ms":
                total_time
        }
    )

    return top_results