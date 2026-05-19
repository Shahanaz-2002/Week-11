import time
from fastapi import HTTPException

from retrieval.retrieval_engine import retrieve_similar_cases
from retrieval.database import fetch_case_database

from config import TOP_K


# =========================================================
# LOAD CASE DATABASE
# =========================================================

try:

    case_database = fetch_case_database()

except Exception:

    case_database = []


# =========================================================
# HELPER FUNCTION:
# BUILD SEARCH QUERY
# =========================================================

def build_search_query(request):

    query_parts = []

    weighted_fields = [

        request.chief_complaint,
        request.affected_body_part,
        request.symptoms,
        request.subjective_assessment,
        request.physical_examination,
        request.objective_findings,
        request.patient_pain_classification,
        request.previous_injuries,
        request.functional_assessment,
        request.doctor_notes,
        request.clinical_history
    ]

    for field in weighted_fields:

        if field not in [None, "", [], {}]:

            query_parts.append(
                str(field).strip()
            )

    return " | ".join(query_parts).strip()


# =========================================================
# HELPER FUNCTION:
# BUILD CLINICAL CONTEXT
# =========================================================

def build_clinical_context(request):

    context_parts = []

    field_mapping = {

        "Age": request.age,
        "Gender": request.gender,
        "Occupation": request.occupation,
        "Activity Levels": request.activity_levels,
        "Chief Complaint": request.chief_complaint,
        "Affected Body Part": request.affected_body_part,
        "Symptoms Duration": request.symptoms_duration,
        "Previous Injuries": request.previous_injuries,
        "Current Medications": request.current_medications,
        "Allergies": request.allergies,
        "Subjective Assessment": request.subjective_assessment,
        "Functional Assessment": request.functional_assessment,
        "Physical Examination": request.physical_examination,
        "Objective Findings": request.objective_findings,
        "Pain Classification": request.patient_pain_classification,
        "Symptoms": request.symptoms,
        "Doctor Notes": request.doctor_notes,
        "Clinical History": request.clinical_history,
        "Doctor Name": request.doctor_name
    }

    for field_name, value in field_mapping.items():

        if value not in [None, "", [], {}]:

            context_parts.append(
                f"{field_name}: {value}"
            )

    return "\n".join(context_parts)


# =========================================================
# HELPER FUNCTION:
# GENERATE RECOMMENDATIONS
# =========================================================

def generate_recommendations(case):

    recommendations = {

        "recommended_tests": [],
        "recommended_medicines": []
    }

    complaint = str(
        case.get("chief_complaint", "")
    ).lower()

    body_part = str(
        case.get("affected_body_part", "")
    ).lower()

    combined_text = (
        complaint + " " + body_part
    ).lower()

    # =====================================================
    # KNEE CONDITIONS
    # =====================================================

    if "knee" in combined_text:

        recommendations["recommended_tests"] = [

            "Knee X-Ray",
            "MRI Knee",
            "Physical Stability Test"
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

            "Spine MRI",
            "Posture Assessment"
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
            "Rotator Cuff Examination"
        ]

        recommendations["recommended_medicines"] = [

            "Naproxen",
            "Pain Relief Gel"
        ]

    # =====================================================
    # SKIN / DERMATOLOGY
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
    # DEFAULT
    # =====================================================

    else:

        recommendations["recommended_tests"] = [

            "Clinical Evaluation"
        ]

        recommendations["recommended_medicines"] = [

            "General Pain Management"
        ]

    return recommendations


# =========================================================
# HELPER FUNCTION:
# CONFIDENCE LEVEL
# =========================================================

def get_confidence_level(score):

    if score >= 0.85:

        return "High"

    elif score >= 0.60:

        return "Moderate"

    return "Low"


# =========================================================
# HELPER FUNCTION:
# GENERATE MATCH EXPLANATION
# =========================================================

def generate_similarity_reason(case):

    keywords = []

    complaint = case.get(
        "chief_complaint",
        ""
    )

    body_part = case.get(
        "affected_body_part",
        ""
    )

    findings = case.get(
        "objective_findings",
        ""
    )

    if complaint:
        keywords.append(complaint)

    if body_part:
        keywords.append(body_part)

    if findings:
        keywords.append(findings)

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
        # DYNAMIC QUERY GENERATION
        # =================================================

        if not search_query:

            search_query = build_search_query(
                request
            )

        if not generated_context:

            generated_context = build_clinical_context(
                request
            )

        if not combined_symptoms:

            combined_symptoms = " ".join([

                str(request.chief_complaint),

                str(request.subjective_assessment),

                str(request.objective_findings),

                str(request.symptoms)

            ]).strip()

        # =================================================
        # EMPTY QUERY VALIDATION
        # =================================================

        if not search_query.strip():

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
                    "search_query": search_query
                }
            )

        # =================================================
        # RETRIEVAL ENGINE
        # =================================================

        try:

            retrieved_cases = retrieve_similar_cases(

                query_text=search_query,

                case_database=case_database,

                top_k=max(TOP_K, 2)
            )

        except Exception as e:

            if log_event:

                log_event(
                    "retrieval_error",
                    request_id,
                    "Similarity retrieval failed",
                    {
                        "error": str(e)
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
        # NO MATCH FOUND
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
        # TOP 2 MATCHES ONLY
        # =================================================

        top_matches = retrieved_cases[:2]

        # =================================================
        # FORMAT MATCHES
        # =================================================

        formatted_matches = []

        for case in top_matches:

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

                similarity_score = max(
                    0.0,
                    min(
                        1.0,
                        similarity_score
                    )
                )

                recommendations = (
                    generate_recommendations(
                        case
                    )
                )

                formatted_case = {

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
                        case.get(
                            "chief_complaint",
                            "Unknown"
                        ),

                    "affected_body_part":
                        case.get(
                            "affected_body_part",
                            "Unknown"
                        ),

                    "symptoms_duration":
                        case.get(
                            "symptoms_duration",
                            "Unknown"
                        ),

                    "doctor_notes":
                        case.get(
                            "doctor_notes",
                            "No notes available"
                        ),

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
                        ),

                    "matched_keywords": [

                        str(
                            case.get(
                                "chief_complaint",
                                ""
                            )
                        ),

                        str(
                            case.get(
                                "affected_body_part",
                                ""
                            )
                        )
                    ],

                    "similarity_reason":
                        generate_similarity_reason(
                            case
                        )
                }

                formatted_matches.append(
                    formatted_case
                )

            except Exception:
                continue

        # =================================================
        # VALID FORMATTED MATCHES CHECK
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
        # OVERALL CONFIDENCE SCORE
        # =================================================

        confidence_score = round(

            sum([

                match["match_score"]

                for match in formatted_matches

            ]) / len(formatted_matches),

            4
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
                        len(formatted_matches),

                    "processing_time_ms":
                        total_time
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
                    "error": str(e)
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