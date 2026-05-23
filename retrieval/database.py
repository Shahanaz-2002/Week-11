# =========================================================
# services/analyze_service.py
# =========================================================

import time

import traceback

from fastapi import HTTPException

from retrieval.retrieval_engine import (
    retrieve_similar_cases
)

from retrieval.database import (
    fetch_case_database
)

from config import TOP_K


# =========================================================
# SAFE TEXT
# =========================================================

def safe_text(value):

    if value in [None, "", [], {}]:

        return ""

    return str(value).strip()


# =========================================================
# SAFE ATTRIBUTE ACCESS
# =========================================================

def safe_attr(obj, field):

    try:

        return getattr(obj, field, "")

    except Exception:

        return ""


# =========================================================
# BUILD SEARCH QUERY
# =========================================================

def build_search_query(request):

    query_parts = []

    weighted_fields = [

        safe_attr(request, "skin_condition"),

        safe_attr(request, "affected_skin_area"),

        safe_attr(request, "symptoms"),

        safe_attr(request, "skin_type"),

        safe_attr(request, "subjective_assessment"),

        safe_attr(request, "physical_examination"),

        safe_attr(request, "objective_findings"),

        safe_attr(request, "previous_skin_conditions"),

        safe_attr(request, "allergies"),

        safe_attr(request, "doctor_notes"),

        safe_attr(request, "clinical_history"),

        # ============================================
        # SUPPORT GENERIC CLINICAL FIELDS
        # ============================================

        safe_attr(request, "chief_complaint"),

        safe_attr(request, "affected_body_part")
    ]

    for field in weighted_fields:

        cleaned = safe_text(field)

        if cleaned:

            query_parts.append(cleaned)

    return " | ".join(query_parts).strip()


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_dermatology_context(request):

    context_parts = []

    field_mapping = {

        "Age":
            safe_attr(request, "age"),

        "Gender":
            safe_attr(request, "gender"),

        "Skin Type":
            safe_attr(request, "skin_type"),

        "Occupation":
            safe_attr(request, "occupation"),

        "Skin Condition":
            safe_attr(request, "skin_condition"),

        "Affected Skin Area":
            safe_attr(request, "affected_skin_area"),

        "Symptoms":
            safe_attr(request, "symptoms"),

        "Doctor Notes":
            safe_attr(request, "doctor_notes"),

        "Clinical History":
            safe_attr(request, "clinical_history")
    }

    for field_name, value in field_mapping.items():

        cleaned = safe_text(value)

        if cleaned:

            context_parts.append(
                f"{field_name}: {cleaned}"
            )

    return "\n".join(context_parts)


# =========================================================
# GENERATE RECOMMENDATIONS
# =========================================================

def generate_recommendations(case):

    recommendations = {

        "recommended_tests": [],

        "recommended_medicines": [],

        "skincare_plan": [],

        "precautions": []
    }

    combined_text = (

        safe_text(case.get("skin_condition")) +

        " " +

        safe_text(case.get("symptoms")) +

        " " +

        safe_text(case.get("doctor_notes"))

    ).lower()

    # =====================================================
    # ACNE
    # =====================================================

    if "acne" in combined_text:

        recommendations["recommended_tests"] = [

            "Dermatoscopy",

            "Hormonal Evaluation"
        ]

        recommendations["recommended_medicines"] = [

            "Benzoyl Peroxide",

            "Topical Retinoid"
        ]

    # =====================================================
    # ECZEMA
    # =====================================================

    elif "eczema" in combined_text:

        recommendations["recommended_tests"] = [

            "Patch Allergy Test"
        ]

        recommendations["recommended_medicines"] = [

            "Topical Corticosteroid",

            "Moisturizer"
        ]

    # =====================================================
    # PSORIASIS
    # =====================================================

    elif "psoriasis" in combined_text:

        recommendations["recommended_tests"] = [

            "Skin Biopsy"
        ]

        recommendations["recommended_medicines"] = [

            "Topical Steroids"
        ]

    # =====================================================
    # FALLBACK
    # =====================================================

    else:

        recommendations["recommended_tests"] = [

            "Dermatology Consultation"
        ]

        recommendations["recommended_medicines"] = [

            "Symptomatic Skin Care"
        ]

    return recommendations


# =========================================================
# CONFIDENCE
# =========================================================

def get_confidence_level(score):

    if score >= 0.85:

        return "High"

    elif score >= 0.60:

        return "Moderate"

    return "Low"


# =========================================================
# GENERATE MATCH REASON
# =========================================================

def generate_similarity_reason(case):

    keywords = []

    for field in [

        "skin_condition",

        "affected_skin_area",

        "objective_findings",

        "chief_complaint",

        "affected_body_part"
    ]:

        value = safe_text(
            case.get(field)
        )

        if value:

            keywords.append(value)

    if not keywords:

        return (
            "Matched using semantic similarity"
        )

    return (
        "Matched based on: " +
        ", ".join(keywords[:3])
    )


# =========================================================
# SANITIZE MATCH
# =========================================================

def sanitize_match(case):

    similarity_score = round(

        float(case.get("similarity", 0.0)),

        4
    )

    similarity_score = max(

        0.0,

        min(1.0, similarity_score)
    )

    return {

        "case_id":
            str(case.get("case_id", "Unknown")),

        "match_score":
            similarity_score,

        "confidence_level":
            get_confidence_level(similarity_score),

        "matched_keywords":
            case.get("matched_keywords", []),

        "semantic_score":
            similarity_score,

        "retrieval_source":
            "BioBERT Clinical Retrieval",

        "explanation":
            generate_similarity_reason(case),

        "recommendation":
            generate_recommendations(case),

        # ============================================
        # RETURN RAW CASE TOO
        # ============================================

        "case_data":
            case
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
        # FETCH DATABASE DYNAMICALLY
        # =================================================

        case_database = fetch_case_database()

        if not isinstance(case_database, list):

            raise HTTPException(

                status_code=500,

                detail={
                    "status": "Failed",
                    "message": "Invalid case database"
                }
            )

        # =================================================
        # BUILD SEARCH QUERY
        # =================================================

        if not search_query:

            search_query = build_search_query(
                request
            )

        if not generated_context:

            generated_context = (
                build_dermatology_context(
                    request
                )
            )

        if not combined_symptoms:

            combined_symptoms = " ".join([

                safe_text(
                    safe_attr(request, "skin_condition")
                ),

                safe_text(
                    safe_attr(request, "symptoms")
                ),

                safe_text(
                    safe_attr(request, "objective_findings")
                )
            ]).strip()

        if not search_query.strip():

            raise HTTPException(

                status_code=400,

                detail={
                    "error": "Invalid Input",
                    "message": "Search query empty"
                }
            )

        # =================================================
        # RETRIEVE CASES
        # =================================================

        retrieved_cases = retrieve_similar_cases(

            query_text=search_query,

            case_database=case_database,

            top_k=max(TOP_K, 2)
        )

        # =================================================
        # NO MATCH
        # =================================================

        if not retrieved_cases:

            return {

                "status":
                    "No Match",

                "message":
                    "No similar cases found",

                "matches":
                    [],

                "total_matches_found":
                    0,

                "confidence_score":
                    0.0,

                "generated_context":
                    generated_context,

                "search_query":
                    search_query
            }

        # =================================================
        # FORMAT RESULTS
        # =================================================

        formatted_matches = [

            sanitize_match(case)

            for case in retrieved_cases[:2]
        ]

        confidence_score = round(

            sum([
                x["match_score"]
                for x in formatted_matches
            ]) / len(formatted_matches),

            4
        )

        return {

            "status":
                "Success",

            "message":
                "Clinical matches retrieved successfully",

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
                round(
                    (
                        time.time() -
                        start_time
                    ) * 1000,
                    2
                ),

            "explanation":
                "AI-powered semantic retrieval completed"
        }

    except HTTPException:

        raise

    except Exception as e:

        if log_event:

            log_event(

                "pipeline_error",

                request_id,

                "Clinical pipeline failure",

                {
                    "error":
                        str(e),

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
                    "Pipeline execution failed",

                "matches":
                    [],

                "confidence_score":
                    0.0,

                "explanation":
                    str(e)
            }
        )