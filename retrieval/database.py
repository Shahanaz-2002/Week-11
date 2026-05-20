# =========================================================
# services/analyze_service.py
# =========================================================

import time
import traceback
from fastapi import HTTPException

from retrieval.retrieval_engine import retrieve_similar_cases
from retrieval.database import fetch_case_database

from config import TOP_K


try:
    case_database = fetch_case_database()
except Exception:
    case_database = []


def safe_text(value):

    if value in [None, "", [], {}]:
        return ""

    return str(value).strip()


def build_search_query(request):

    query_parts = []

    weighted_fields = [

        request.skin_condition,
        request.affected_skin_area,
        request.symptoms,
        request.skin_type,
        request.subjective_assessment,
        request.physical_examination,
        request.objective_findings,
        request.previous_skin_conditions,
        request.allergies,
        request.doctor_notes,
        request.clinical_history
    ]

    for field in weighted_fields:

        cleaned = safe_text(field)

        if cleaned:
            query_parts.append(cleaned)

    return " | ".join(query_parts).strip()


def build_dermatology_context(request):

    context_parts = []

    field_mapping = {

        "Age": request.age,
        "Gender": request.gender,
        "Skin Type": request.skin_type,
        "Occupation": request.occupation,
        "Skin Condition": request.skin_condition,
        "Affected Skin Area": request.affected_skin_area,
        "Symptoms Duration": request.symptoms_duration,
        "Previous Skin Conditions": request.previous_skin_conditions,
        "Current Medications": request.current_medications,
        "Allergies": request.allergies,
        "Symptoms": request.symptoms,
        "Physical Examination": request.physical_examination,
        "Objective Findings": request.objective_findings,
        "Doctor Notes": request.doctor_notes,
        "Clinical History": request.clinical_history,
        "Doctor Name": request.doctor_name
    }

    for field_name, value in field_mapping.items():

        cleaned = safe_text(value)

        if cleaned:

            context_parts.append(
                f"{field_name}: {cleaned}"
            )

    return "\n".join(context_parts)


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

    if "acne" in combined_text:

        recommendations["recommended_tests"] = [
            "Dermatoscopy",
            "Hormonal Evaluation"
        ]

        recommendations["recommended_medicines"] = [
            "Benzoyl Peroxide",
            "Topical Retinoid"
        ]

        recommendations["skincare_plan"] = [
            "Use oil-free cleanser",
            "Avoid comedogenic cosmetics"
        ]

        recommendations["precautions"] = [
            "Avoid touching pimples",
            "Use sunscreen daily"
        ]

    elif "eczema" in combined_text:

        recommendations["recommended_tests"] = [
            "Patch Allergy Test"
        ]

        recommendations["recommended_medicines"] = [
            "Topical Corticosteroid",
            "Moisturizer"
        ]

        recommendations["skincare_plan"] = [
            "Use fragrance-free moisturizer",
            "Hydrate skin regularly"
        ]

        recommendations["precautions"] = [
            "Avoid harsh soaps",
            "Avoid allergens"
        ]

    elif "psoriasis" in combined_text:

        recommendations["recommended_tests"] = [
            "Skin Biopsy"
        ]

        recommendations["recommended_medicines"] = [
            "Topical Steroids",
            "Vitamin D Analogues"
        ]

        recommendations["skincare_plan"] = [
            "Keep skin moisturized",
            "Use medicated shampoo if scalp affected"
        ]

        recommendations["precautions"] = [
            "Avoid smoking",
            "Reduce stress"
        ]

    elif "fungal" in combined_text:

        recommendations["recommended_tests"] = [
            "KOH Test",
            "Fungal Culture"
        ]

        recommendations["recommended_medicines"] = [
            "Clotrimazole",
            "Terbinafine"
        ]

        recommendations["skincare_plan"] = [
            "Keep affected area dry"
        ]

        recommendations["precautions"] = [
            "Avoid sharing towels",
            "Maintain hygiene"
        ]

    else:

        recommendations["recommended_tests"] = [
            "Dermatology Consultation"
        ]

        recommendations["recommended_medicines"] = [
            "Symptomatic Skin Care"
        ]

    return recommendations


def get_confidence_level(score):

    if score >= 0.85:
        return "High"

    elif score >= 0.60:
        return "Moderate"

    return "Low"


def generate_similarity_reason(case):

    keywords = []

    condition = safe_text(
        case.get("skin_condition")
    )

    area = safe_text(
        case.get("affected_skin_area")
    )

    findings = safe_text(
        case.get("objective_findings")
    )

    if condition:
        keywords.append(condition)

    if area:
        keywords.append(area)

    if findings:
        keywords.append(findings)

    if len(keywords) == 0:

        return (
            "Matched using semantic dermatology similarity"
        )

    return (
        "Matched based on: " +
        ", ".join(keywords[:3])
    )


def sanitize_match(case):

    similarity_score = round(
        float(case.get("similarity", 0.0)),
        4
    )

    similarity_score = max(
        0.0,
        min(1.0, similarity_score)
    )

    recommendations = generate_recommendations(case)

    return {

        "case_id":
            str(case.get("case_id", "Unknown")),

        "match_score":
            similarity_score,

        "confidence_level":
            get_confidence_level(similarity_score),

        "skin_condition":
            case.get("skin_condition", "Unknown"),

        "affected_skin_area":
            case.get("affected_skin_area", "Unknown"),

        "symptoms_duration":
            case.get("symptoms_duration", "Unknown"),

        "doctor_notes":
            case.get(
                "doctor_notes",
                "No notes available"
            ),

        "matched_keywords":
            case.get(
                "matched_keywords",
                []
            ),

        "semantic_score":
            similarity_score,

        "retrieval_source":
            "BioBERT Dermatology Retrieval",

        "explanation":
            generate_similarity_reason(case),

        "recommendation": recommendations
    }


def dermatology_match_pipeline(
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

        if not isinstance(case_database, list):

            raise HTTPException(
                status_code=500,
                detail={
                    "status": "Failed",
                    "message": "Invalid case database"
                }
            )

        if not search_query:

            search_query = build_search_query(
                request
            )

        if not generated_context:

            generated_context = build_dermatology_context(
                request
            )

        if not combined_symptoms:

            combined_symptoms = " ".join([

                safe_text(request.skin_condition),
                safe_text(request.symptoms),
                safe_text(request.objective_findings)

            ]).strip()

        if not search_query.strip():

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid Input",
                    "message": "Search query empty"
                }
            )

        retrieved_cases = retrieve_similar_cases(

            query_text=search_query,
            case_database=case_database,
            top_k=max(TOP_K, 2)
        )

        if not retrieved_cases:

            return {

                "status": "No Match",
                "message": "No similar dermatology cases found",
                "matches": [],
                "total_matches_found": 0,
                "confidence_score": 0.0,
                "generated_context": generated_context,
                "search_query": search_query,
                "processing_time_ms": round(
                    (time.time() - start_time) * 1000,
                    2
                )
            }

        formatted_matches = []

        for case in retrieved_cases[:2]:

            formatted_matches.append(
                sanitize_match(case)
            )

        confidence_score = round(

            sum([
                x["match_score"]
                for x in formatted_matches
            ]) / len(formatted_matches),

            4
        )

        return {

            "status": "Success",

            "message":
                "Top dermatology matches retrieved successfully",

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
                    (time.time() - start_time) * 1000,
                    2
                ),

            "explanation":
                "AI-powered dermatology semantic retrieval completed"
        }

    except HTTPException:
        raise

    except Exception as e:

        if log_event:

            log_event(
                "pipeline_error",
                request_id,
                "Dermatology pipeline failure",
                {
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            )

        raise HTTPException(

            status_code=500,

            detail={

                "status": "Failed",

                "message":
                    "Dermatology pipeline execution failed",

                "matches": [],

                "confidence_score": 0.0,

                "explanation": str(e)
            }
        )