from typing import List, Dict
import numpy as np
import logging
import json
import time
import re

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

        "timestamp":
            time.strftime("%Y-%m-%d %H:%M:%S")
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
# QUERY ENHANCEMENT
# =========================================================

def enhance_query(query_text):

    query_text = clean_text(query_text)

    medical_keywords = [

        "pain",
        "swelling",
        "injury",
        "fracture",
        "stiffness",
        "inflammation",
        "weakness",
        "mobility",
        "muscle",
        "joint",
        "tenderness",
        "sprain",
        "posture",
        "movement"
    ]

    detected_keywords = []

    lower_query = query_text.lower()

    for keyword in medical_keywords:

        if keyword in lower_query:

            detected_keywords.append(keyword)

    if detected_keywords:

        query_text += (
            " | " +
            " ".join(detected_keywords)
        )

    return query_text.strip()


# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray
) -> float:

    try:

        norm_a = np.linalg.norm(a)

        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:

            return 0.0

        similarity = np.dot(a, b) / (
            norm_a * norm_b
        )

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

        "chief_complaint",
        "affected_body_part",
        "symptoms",
        "physical_examination",
        "objective_findings",
        "doctor_notes",
        "clinical_history",
        "case_description"
    ]

    for field in fields:

        value = case_data.get(field, "")

        if value not in [None, ""]:

            text_parts.append(
                str(value)
            )

    return " | ".join(text_parts)


# =========================================================
# MATCHED KEYWORD EXTRACTION
# =========================================================

def extract_matched_keywords(
    query_text,
    searchable_text
):

    query_words = set(
        clean_text(query_text).lower().split()
    )

    searchable_words = set(
        clean_text(searchable_text).lower().split()
    )

    matched = list(
        query_words.intersection(
            searchable_words
        )
    )

    matched = [
        word
        for word in matched
        if len(word) > 3
    ]

    return matched[:10]


# =========================================================
# BOOST SCORE FOR IMPORTANT MATCHES
# =========================================================

def apply_keyword_boost(
    similarity_score,
    matched_keywords
):

    boost = min(
        0.10,
        len(matched_keywords) * 0.01
    )

    boosted_score = similarity_score + boost

    return min(boosted_score, 1.0)


# =========================================================
# MAIN RETRIEVAL FUNCTION
# =========================================================

def retrieve_similar_cases(
    query_text: str,
    case_database: List[Dict],
    top_k: int = 2
) -> List[Dict]:

    start_time = time.time()

    # =====================================================
    # LOG RETRIEVAL START
    # =====================================================

    log_event(
        "retrieval_started",
        "Clinical retrieval started",
        {
            "query_length":
                len(query_text)
                if query_text else 0,

            "database_size":
                len(case_database)
                if isinstance(case_database, list)
                else 0,

            "top_k":
                top_k
        }
    )

    # =====================================================
    # INPUT VALIDATION
    # =====================================================

    if not query_text:

        log_event(
            "validation_error",
            "Empty clinical query received"
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
            "Query empty after cleaning"
        )

        return []

    if not isinstance(case_database, list):

        log_event(
            "validation_error",
            "Invalid case database"
        )

        return []

    # =====================================================
    # QUERY ENHANCEMENT
    # =====================================================

    enhanced_query = enhance_query(
        query_text
    )

    log_event(
        "query_enhanced",
        "Clinical query enhanced",
        {
            "enhanced_query":
                enhanced_query
        }
    )

    # =====================================================
    # EMBEDDER VALIDATION
    # =====================================================

    if embedder is None:

        log_event(
            "embedding_error",
            "Embedding model unavailable"
        )

        return []

    # =====================================================
    # QUERY EMBEDDING
    # =====================================================

    try:

        embedding_start = time.time()

        query_embedding = (
            embedder.get_embedding(
                enhanced_query
            )
        )

        if query_embedding is None:

            return []

        query_embedding = np.array(
            query_embedding
        )

        if query_embedding.size == 0:

            return []

        embedding_time = round(
            (
                time.time() -
                embedding_start
            ) * 1000,
            2
        )

        log_event(
            "embedding_generated",
            "Query embedding generated",
            {
                "embedding_dimension":
                    len(query_embedding),

                "embedding_time_ms":
                    embedding_time
            }
        )

    except Exception as e:

        log_event(
            "embedding_failure",
            "Embedding generation failed",
            {
                "error": str(e)
            }
        )

        return []

    # =====================================================
    # RETRIEVAL LOOP
    # =====================================================

    results = []

    processed_cases = 0

    for case_data in case_database:

        if not isinstance(case_data, dict):

            continue

        try:

            # ------------------------------------------------
            # CASE EMBEDDING
            # ------------------------------------------------

            case_embedding = np.array(
                case_data.get(
                    "embedding",
                    []
                )
            )

            if case_embedding.size == 0:

                continue

            # ------------------------------------------------
            # DIMENSION CHECK
            # ------------------------------------------------

            if len(query_embedding) != len(case_embedding):

                continue

            # ------------------------------------------------
            # COSINE SIMILARITY
            # ------------------------------------------------

            similarity_score = cosine_similarity(
                query_embedding,
                case_embedding
            )

            # ------------------------------------------------
            # SEARCHABLE TEXT
            # ------------------------------------------------

            searchable_text = (
                build_case_search_text(
                    case_data
                )
            )

            # ------------------------------------------------
            # MATCHED KEYWORDS
            # ------------------------------------------------

            matched_keywords = (
                extract_matched_keywords(
                    enhanced_query,
                    searchable_text
                )
            )

            # ------------------------------------------------
            # BOOSTED SCORE
            # ------------------------------------------------

            boosted_score = apply_keyword_boost(
                similarity_score,
                matched_keywords
            )

            # ------------------------------------------------
            # FILTER LOW SCORES
            # ------------------------------------------------

            if boosted_score < 0.20:

                continue

            # ------------------------------------------------
            # RESULT OBJECT
            # ------------------------------------------------

            results.append({

                "case_id":
                    case_data.get(
                        "case_id",
                        "Unknown"
                    ),

                "similarity":
                    round(
                        boosted_score,
                        4
                    ),

                "chief_complaint":
                    case_data.get(
                        "chief_complaint",
                        "Unknown"
                    ),

                "affected_body_part":
                    case_data.get(
                        "affected_body_part",
                        "Unknown"
                    ),

                "symptoms_duration":
                    case_data.get(
                        "symptoms_duration",
                        "Unknown"
                    ),

                "doctor_notes":
                    case_data.get(
                        "doctor_notes",
                        "No notes available"
                    ),

                "clinical_history":
                    case_data.get(
                        "clinical_history",
                        ""
                    ),

                "matched_keywords":
                    matched_keywords,

                "searchable_text":
                    searchable_text
            })

            processed_cases += 1

        except Exception as e:

            log_event(
                "case_processing_error",
                "Error processing case",
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
            "No similar clinical cases found"
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
    # FINAL LOGGING
    # =====================================================

    total_time = round(
        (time.time() - start_time) * 1000,
        2
    )

    log_event(
        "retrieval_completed",
        "Clinical retrieval completed",
        {
            "processed_cases":
                processed_cases,

            "returned_cases":
                len(top_results),

            "top_case_ids":
                [
                    r["case_id"]
                    for r in top_results
                ],

            "top_scores":
                [
                    r["similarity"]
                    for r in top_results
                ],

            "total_time_ms":
                total_time
        }
    )

    return top_results