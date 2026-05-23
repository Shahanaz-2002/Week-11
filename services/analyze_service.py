# =========================================================
# services/analyze_service.py
# =========================================================

import time
import traceback
import re
import logging
from typing import Dict, List, Any

from fastapi import HTTPException

from retrieval.retrieval_engine import retrieve_similar_cases
from retrieval.database import fetch_case_database

from config import (
    TOP_K,
    MAX_MATCH_RESULTS,
    VERY_HIGH_CONFIDENCE,
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    DEFAULT_RECOMMENDED_TESTS,
    DEFAULT_RECOMMENDED_MEDICINES
)

# =========================================================
# LOGGER CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    force=True
)

logger = logging.getLogger(__name__)


# =========================================================
# LOAD CASE DATABASE
# =========================================================

try:

    case_database = fetch_case_database()

    if not isinstance(case_database, list):

        case_database = []

except Exception as e:

    logger.error(
        f"Failed to load case database: {str(e)}"
    )

    case_database = []


# =========================================================
# SAFE TEXT HELPER
# =========================================================

def safe_text(value) -> str:

    if value in [None, "", [], {}]:

        return ""

    return str(value).strip()


# =========================================================
# NORMALIZE TEXT
# =========================================================

def normalize_text(text: str) -> str:

    text = safe_text(text)

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"[^\w\s|,-]", " ", text)

    return text.strip()


# =========================================================
# REMOVE DUPLICATES
# =========================================================

def remove_duplicates(values: List[str]) -> List[str]:

    seen = set()

    cleaned = []

    for value in values:

        normalized = normalize_text(
            value
        ).lower()

        if normalized and normalized not in seen:

            seen.add(normalized)

            cleaned.append(value)

    return cleaned


# =========================================================
# BUILD SEARCH QUERY
# =========================================================

def build_search_query(request) -> str:

    query_parts = []

    weighted_fields = [

        getattr(request, "chief_complaint", ""),

        getattr(request, "affected_body_part", ""),

        getattr(request, "symptoms", ""),

        getattr(request, "subjective_assessment", ""),

        getattr(request, "physical_examination", ""),

        getattr(request, "objective_findings", ""),

        getattr(request, "patient_pain_classification", ""),

        getattr(request, "previous_injuries", ""),

        getattr(request, "functional_assessment", ""),

        getattr(request, "doctor_notes", ""),

        getattr(request, "clinical_history", "")
    ]

    for field in weighted_fields:

        cleaned = normalize_text(field)

        if cleaned:

            query_parts.append(cleaned)

    query_parts = remove_duplicates(query_parts)

    return " | ".join(query_parts).strip()


# =========================================================
# BUILD CLINICAL CONTEXT
# =========================================================

def build_clinical_context(request) -> str:

    context_parts = []

    field_mapping = {

        "Age":
            getattr(request, "age", ""),

        "Gender":
            getattr(request, "gender", ""),

        "Occupation":
            getattr(request, "occupation", ""),

        "Activity Levels":
            getattr(request, "activity_levels", ""),

        "Chief Complaint":
            getattr(request, "chief_complaint", ""),

        "Affected Body Part":
            getattr(request, "affected_body_part", ""),

        "Symptoms Duration":
            getattr(request, "symptoms_duration", ""),

        "Previous Injuries":
            getattr(request, "previous_injuries", ""),

        "Current Medications":
            getattr(request, "current_medications", ""),

        "Allergies":
            getattr(request, "allergies", ""),

        "Subjective Assessment":
            getattr(request, "subjective_assessment", ""),

        "Functional Assessment":
            getattr(request, "functional_assessment", ""),

        "Physical Examination":
            getattr(request, "physical_examination", ""),

        "Objective Findings":
            getattr(request, "objective_findings", ""),

        "Pain Classification":
            getattr(
                request,
                "patient_pain_classification",
                ""
            ),

        "Symptoms":
            getattr(request, "symptoms", ""),

        "Doctor Notes":
            getattr(request, "doctor_notes", ""),

        "Clinical History":
            getattr(request, "clinical_history", ""),

        "Doctor Name":
            getattr(request, "doctor_name", "")
    }

    for field_name, value in field_mapping.items():

        cleaned = normalize_text(value)

        if cleaned:

            context_parts.append(
                f"{field_name}: {cleaned}"
            )

    return "\n".join(context_parts)


# =========================================================
# GENERATE RECOMMENDATIONS
# =========================================================

def generate_recommendations(
    case
) -> Dict[str, List[str]]:

    recommendations = {

        "recommended_tests": [],

        "recommended_medicines": []
    }

    complaint = safe_text(
        case.get("chief_complaint")
    ).lower()

    body_part = safe_text(
        case.get("affected_body_part")
    ).lower()

    symptoms = safe_text(
        case.get("searchable_text")
    ).lower()

    combined_text = (
        complaint +
        " " +
        body_part +
        " " +
        symptoms
    ).lower()

    # =====================================================
    # KNEE CONDITIONS
    # =====================================================

    if "knee" in combined_text:

        recommendations["recommended_tests"] = [

            "Knee X-Ray",

            "MRI Knee",

            "Anterior Drawer Test",

            "Physical Stability Assessment"
        ]

        recommendations["recommended_medicines"] = [

            "Ibuprofen",

            "Paracetamol"
        ]

    # =====================================================
    # BACK CONDITIONS
    # =====================================================

    elif "back" in combined_text:

        recommendations["recommended_tests"] = [

            "Lumbar Spine MRI",

            "Posture Assessment",

            "Straight Leg Raise Test"
        ]

        recommendations["recommended_medicines"] = [

            "Diclofenac",

            "Muscle Relaxant"
        ]

    # =====================================================
    # SHOULDER CONDITIONS
    # =====================================================

    elif "shoulder" in combined_text:

        recommendations["recommended_tests"] = [

            "Shoulder MRI",

            "Rotator Cuff Examination",

            "Shoulder Mobility Test"
        ]

        recommendations["recommended_medicines"] = [

            "Naproxen",

            "Pain Relief Gel"
        ]

    # =====================================================
    # NECK CONDITIONS
    # =====================================================

    elif "neck" in combined_text:

        recommendations["recommended_tests"] = [

            "Cervical Spine X-Ray",

            "Neck Mobility Assessment"
        ]

        recommendations["recommended_medicines"] = [

            "Muscle Relaxant",

            "Paracetamol"
        ]

    # =====================================================
    # SKIN CONDITIONS
    # =====================================================

    elif (

        "skin" in combined_text or
        "rash" in combined_text or
        "acne" in combined_text

    ):

        recommendations["recommended_tests"] = [

            "Skin Examination",

            "Allergy Test"
        ]

        recommendations["recommended_medicines"] = [

            "Topical Cream",

            "Antihistamine"
        ]

    # =====================================================
    # DEFAULT RECOMMENDATIONS
    # =====================================================

    else:

        recommendations["recommended_tests"] = (
            DEFAULT_RECOMMENDED_TESTS
        )

        recommendations["recommended_medicines"] = (
            DEFAULT_RECOMMENDED_MEDICINES
        )

    return recommendations


# =========================================================
# CONFIDENCE LEVEL
# =========================================================

def get_confidence_level(
    score: float
) -> str:

    if score >= VERY_HIGH_CONFIDENCE:

        return "Very High"

    elif score >= HIGH_CONFIDENCE:

        return "High"

    elif score >= MEDIUM_CONFIDENCE:

        return "Moderate"

    return "Low"


# =========================================================
# GENERATE MATCH EXPLANATION
# =========================================================

def generate_similarity_reason(
    case
) -> str:

    keywords = []

    complaint = safe_text(
        case.get("chief_complaint")
    )

    body_part = safe_text(
        case.get("affected_body_part")
    )

    findings = safe_text(
        case.get("objective_findings")
    )

    if complaint:
        keywords.append(complaint)

    if body_part:
        keywords.append(body_part)

    if findings:
        keywords.append(findings)

    keywords = remove_duplicates(keywords)

    if len(keywords) == 0:

        return (
            "Matched using semantic "
            "clinical similarity"
        )

    return (
        "Matched based on: " +
        ", ".join(keywords[:3])
    )


# =========================================================
# SANITIZE MATCH
# =========================================================

def sanitize_match(
    case
) -> Dict[str, Any]:

    try:

        similarity_score = round(

            float(
                case.get(
                    "similarity",
                    0.0
                )
            ),

            4
        )

    except Exception:

        similarity_score = 0.0

    similarity_score = max(
        0.0,
        min(
            1.0,
            similarity_score
        )
    )

    recommendations = (
        generate_recommendations(case)
    )

    return {

        "case_id":
            str(
                case.get(
                    "case_id",
                    "Unknown"
                )
            ),

        "match_score":
            similarity_score,

        "confidence_level":
            get_confidence_level(
                similarity_score
            ),

        "chief_complaint":
            safe_text(
                case.get(
                    "chief_complaint",
                    "Unknown"
                )
            ),

        "affected_body_part":
            safe_text(
                case.get(
                    "affected_body_part",
                    "Unknown"
                )
            ),

        "symptoms_duration":
            safe_text(
                case.get(
                    "symptoms_duration",
                    "Unknown"
                )
            ),

        "doctor_notes":
            safe_text(
                case.get(
                    "doctor_notes",
                    "No notes available"
                )
            ),

        "matched_keywords":
            case.get(
                "matched_keywords",
                []
            ),

        "semantic_score":
            similarity_score,

        "retrieval_source":
            "BioBERT Semantic Retrieval",

        "explanation":
            generate_similarity_reason(
                case
            ),

        "recommendation": {

            "recommended_tests":
                case.get(
                    "recommended_tests",
                    recommendations[
                        "recommended_tests"
                    ]
                ),

            "recommended_medicines":
                case.get(
                    "recommended_medicines",
                    recommendations[
                        "recommended_medicines"
                    ]
                )
        }
    }


# =========================================================
# MAIN PIPELINE
# =========================================================

def clinical_match_pipeline(

    request,

    request_id,

    search_query="",

    generated_context="",

    combined_symptoms="",

    patient_metadata=None,

    log_event=None
):

    start_time = time.time()

    try:

        # =================================================
        # DATABASE VALIDATION
        # =================================================

        if not isinstance(
            case_database,
            list
        ):

            raise HTTPException(

                status_code=500,

                detail={

                    "status":
                        "Failed",

                    "message":
                        "Invalid case database"
                }
            )

        if len(case_database) == 0:

            raise HTTPException(

                status_code=500,

                detail={

                    "status":
                        "Failed",

                    "message":
                        "Case database is empty"
                }
            )

        # =================================================
        # DYNAMIC QUERY GENERATION
        # =================================================

        if not search_query:

            search_query = (
                build_search_query(
                    request
                )
            )

        if not generated_context:

            generated_context = (
                build_clinical_context(
                    request
                )
            )

        if not combined_symptoms:

            combined_symptoms = " ".join([

                safe_text(
                    getattr(
                        request,
                        "chief_complaint",
                        ""
                    )
                ),

                safe_text(
                    getattr(
                        request,
                        "subjective_assessment",
                        ""
                    )
                ),

                safe_text(
                    getattr(
                        request,
                        "objective_findings",
                        ""
                    )
                ),

                safe_text(
                    getattr(
                        request,
                        "symptoms",
                        ""
                    )
                )

            ]).strip()

        search_query = normalize_text(
            search_query
        )

        generated_context = normalize_text(
            generated_context
        )

        combined_symptoms = normalize_text(
            combined_symptoms
        )

        # =================================================
        # EMPTY QUERY VALIDATION
        # =================================================

        if not search_query:

            raise HTTPException(

                status_code=400,

                detail={

                    "error":
                        "Invalid Input",

                    "message":
                        "Clinical search query is empty"
                }
            )

        # =================================================
        # LOG INPUT PROCESSING
        # =================================================

        if log_event:

            log_event(
                "input_processed",
                request_id,
                "Clinical input processed successfully",
                {
                    "search_query":
                        search_query,

                    "combined_symptoms":
                        combined_symptoms
                }
            )

        # =================================================
        # RETRIEVAL ENGINE
        # =================================================

        try:

            retrieved_cases = (
                retrieve_similar_cases(

                    query_text=search_query,

                    case_database=case_database,

                    top_k=max(
                        TOP_K,
                        MAX_MATCH_RESULTS
                    )
                )
            )

        except Exception as e:

            if log_event:

                log_event(
                    "retrieval_error",
                    request_id,
                    "Similarity retrieval failed",
                    {
                        "error": str(e),
                        "traceback":
                            traceback.format_exc()
                    }
                )

            raise HTTPException(

                status_code=500,

                detail={

                    "error":
                        "Retrieval Failure",

                    "message":
                        "Error during clinical similarity retrieval"
                }
            )

        # =================================================
        # EMPTY RETRIEVAL FALLBACK
        # =================================================

        if not retrieved_cases:

            return {

                "status":
                    "No Match",

                "message":
                    "No similar clinical cases found",

                "matches":
                    [],

                "total_matches_found":
                    0,

                "confidence_score":
                    0.0,

                "generated_context":
                    generated_context,

                "search_query":
                    search_query,

                "processing_time_ms":
                    round(
                        (
                            time.time() -
                            start_time
                        ) * 1000,
                        2
                    ),

                "explanation":
                    (
                        "No relevant historical "
                        "clinical cases available"
                    )
            }

        # =================================================
        # LIMIT MATCHES
        # =================================================

        top_matches = retrieved_cases[
            :MAX_MATCH_RESULTS
        ]

        # =================================================
        # FORMAT MATCHES
        # =================================================

        formatted_matches = []

        for case in top_matches:

            try:

                formatted_case = (
                    sanitize_match(case)
                )

                formatted_matches.append(
                    formatted_case
                )

            except Exception as formatting_error:

                if log_event:

                    log_event(
                        "match_format_error",
                        request_id,
                        "Failed to format match",
                        {
                            "error":
                                str(
                                    formatting_error
                                )
                        }
                    )

                continue

        # =================================================
        # VALID MATCH CHECK
        # =================================================

        if len(formatted_matches) == 0:

            raise HTTPException(

                status_code=500,

                detail={

                    "error":
                        "Formatting Failure",

                    "message":
                        "Retrieved cases could not be formatted"
                }
            )

        # =================================================
        # CONFIDENCE SCORE
        # =================================================

        confidence_score = round(

            sum([

                match["match_score"]

                for match in formatted_matches

            ]) / len(formatted_matches),

            4
        )

        confidence_score = max(
            0.0,
            min(
                1.0,
                confidence_score
            )
        )

        # =================================================
        # TOTAL PROCESSING TIME
        # =================================================

        total_time = round(

            (
                time.time() -
                start_time
            ) * 1000,

            2
        )

        # =================================================
        # PIPELINE SUCCESS LOGGING
        # =================================================

        if log_event:

            log_event(
                "pipeline_completed",
                request_id,
                "Clinical pipeline executed successfully",
                {
                    "matches_found":
                        len(
                            formatted_matches
                        ),

                    "processing_time_ms":
                        total_time,

                    "confidence_score":
                        confidence_score
                }
            )

        # =================================================
        # FINAL RESPONSE
        # =================================================

        return {

            "status":
                "Success",

            "message":
                (
                    "Top clinical matches "
                    "retrieved successfully"
                ),

            "matches":
                formatted_matches,

            "total_matches_found":
                len(formatted_matches),

            "confidence_score":
                confidence_score,

            "generated_context":
                generated_context,

            "search_query":
                search_query,

            "processing_time_ms":
                total_time,

            "explanation":
                (
                    "Top clinical matches generated "
                    "using AI-powered semantic "
                    "similarity retrieval"
                )
        }

    # =====================================================
    # EXPECTED ERRORS
    # =====================================================

    except HTTPException:
        raise

    # =====================================================
    # UNEXPECTED ERRORS
    # =====================================================

    except Exception as e:

        if log_event:

            log_event(
                "pipeline_error",
                request_id,
                "Clinical pipeline failure",
                {
                    "error": str(e),
                    "traceback":
                        traceback.format_exc()
                }
            )

        raise HTTPException(

            status_code=500,

            detail={

                "status":
                    "Failed",

                "message":
                    "Clinical pipeline execution failed",

                "matches":
                    [],

                "confidence_score":
                    0.0,

                "explanation":
                    str(e)
            }
        )