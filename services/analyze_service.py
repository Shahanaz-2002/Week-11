import time
from fastapi import HTTPException

from models.models import SimilarCase

from retrieval.retrieval_engine import retrieve_similar_cases
from retrieval.database import fetch_case_database

from insight.insight_aggregator import InsightAggregator
from insight.confidence_engine import ConfidenceEngine
from insight.explanation_generator import ExplanationGenerator

from config import TOP_K


# =========================================================
# INITIALIZE COMPONENTS
# =========================================================

try:
    case_database = fetch_case_database()
except Exception:
    case_database = []


insight_aggregator = InsightAggregator()
confidence_engine = ConfidenceEngine()
explanation_generator = ExplanationGenerator()


# =========================================================
# HELPER FUNCTION:
# BUILD DERMATOLOGY QUERY
# =========================================================

def build_dermatology_query(request):

    query_parts = []

    if request.chief_complaint:
        query_parts.append(request.chief_complaint)

    if request.symptoms:
        query_parts.append(request.symptoms)

    if request.physical_examination:
        query_parts.append(request.physical_examination)

    if request.objective_findings:
        query_parts.append(request.objective_findings)

    if request.patient_pain_classification:
        query_parts.append(request.patient_pain_classification)

    return " | ".join(query_parts)


# =========================================================
# HELPER FUNCTION:
# BUILD DERMATOLOGY CONTEXT
# =========================================================

def build_dermatology_context(request):

    context_parts = []

    if request.age:
        context_parts.append(f"Age: {request.age}")

    if request.gender:
        context_parts.append(f"Gender: {request.gender}")

    if request.chief_complaint:
        context_parts.append(
            f"Chief Complaint: {request.chief_complaint}"
        )

    if request.symptoms_duration:
        context_parts.append(
            f"Symptoms Duration: {request.symptoms_duration}"
        )

    if request.previous_injuries:
        context_parts.append(
            f"Previous Skin History: {request.previous_injuries}"
        )

    if request.current_medications:
        context_parts.append(
            f"Current Medications: {request.current_medications}"
        )

    if request.allergies:
        context_parts.append(
            f"Allergies: {request.allergies}"
        )

    if request.subjective_assessment:
        context_parts.append(
            f"Subjective Assessment: {request.subjective_assessment}"
        )

    if request.physical_examination:
        context_parts.append(
            f"Physical Examination: {request.physical_examination}"
        )

    if request.objective_findings:
        context_parts.append(
            f"Objective Findings: {request.objective_findings}"
        )

    if request.doctor_notes:
        context_parts.append(
            f"Doctor Notes: {request.doctor_notes}"
        )

    return "\n".join(context_parts)


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
            search_query = build_dermatology_query(request)

        if not generated_context:
            generated_context = build_dermatology_context(request)

        if not combined_symptoms:
            combined_symptoms = (
                f"{request.chief_complaint} "
                f"{request.subjective_assessment} "
                f"{request.objective_findings}"
            ).strip()

        # =================================================
        # EMPTY QUERY CHECK
        # =================================================

        if not search_query.strip():

            raise HTTPException(
                status_code=400,
                detail="Dermatology clinical query is empty"
            )

        # =================================================
        # LOG INPUT PROCESSING
        # =================================================

        if log_event:

            log_event(
                "input_processed",
                request_id,
                "Dermatology input processed successfully",
                {
                    "search_query": search_query
                }
            )

        # =================================================
        # RETRIEVAL ENGINE
        # =================================================

        try:

            top_matches = retrieve_similar_cases(
                query_text=search_query,
                case_database=case_database,
                top_k=TOP_K
            )

        except Exception as e:

            if log_event:

                log_event(
                    "retrieval_error",
                    request_id,
                    "Dermatology retrieval failed",
                    {
                        "error": str(e)
                    }
                )

            raise HTTPException(
                status_code=500,
                detail="Error during dermatology similarity retrieval"
            )

        # =================================================
        # NO MATCH FOUND
        # =================================================

        if not top_matches:

            if log_event:

                log_event(
                    "no_matches",
                    request_id,
                    "No similar dermatology cases found"
                )

            return {
                "suggested_resolution": (
                    "No similar dermatology cases found.\n"
                    "Recommended Action: Manual dermatologist review suggested."
                ),

                "similar_cases": [],

                "confidence_score": 0.0,

                "explanation": (
                    "No dermatology matches found.\n"
                    "Reason: Insufficient historical dermatology data.\n"
                    "Recommendation: Proceed with expert dermatologist analysis."
                )
            }

        # =================================================
        # LOG RETRIEVAL SUCCESS
        # =================================================

        if log_event:

            log_event(
                "retrieval_completed",
                request_id,
                "Dermatology cases retrieved successfully",
                {
                    "num_cases": len(top_matches)
                }
            )

        # =================================================
        # FORMAT SIMILAR CASES
        # =================================================

        similar_cases_formatted = []

        for c in top_matches:

            try:

                formatted_case = SimilarCase(

                    case_id=str(
                        c.get("case_id", "Unknown")
                    ),

                    similarity_score=max(
                        0.0,
                        min(
                            1.0,
                            float(c.get("similarity", 0.0))
                        )
                    ),

                    category=c.get(
                        "category",
                        "Dermatology"
                    ) or "Dermatology",

                    location=c.get(
                        "location",
                        "Unknown"
                    ) or "Unknown",

                    resolution_notes=c.get(
                        "resolution_notes",
                        "No dermatologist notes available"
                    ) or "No dermatologist notes available"
                )

                similar_cases_formatted.append(
                    formatted_case
                )

            except Exception:
                continue

        # =================================================
        # FORMAT FAILURE
        # =================================================

        if not similar_cases_formatted:

            if log_event:

                log_event(
                    "formatting_issue",
                    request_id,
                    "Retrieved dermatology cases invalid"
                )

            return {

                "suggested_resolution": (
                    "Relevant dermatology cases found "
                    "but could not be processed."
                ),

                "similar_cases": [],

                "confidence_score": 0.0,

                "explanation":
                    "Retrieved dermatology data format invalid."
            }

        # =================================================
        # CONFIDENCE ENGINE
        # =================================================

        try:

            confidence_data = (
                confidence_engine.compute_confidence(
                    top_matches
                )
            )

        except Exception as e:

            if log_event:

                log_event(
                    "confidence_error",
                    request_id,
                    "Confidence calculation failed",
                    {
                        "error": str(e)
                    }
                )

            confidence_data = {
                "confidence_score": 0.0
            }

        # =================================================
        # EXPLANATION GENERATION
        # =================================================

        try:

            explanation = (
                explanation_generator.generate_explanation(
                    top_matches
                )
            )

        except Exception as e:

            if log_event:

                log_event(
                    "explanation_error",
                    request_id,
                    "Explanation generation failed",
                    {
                        "error": str(e)
                    }
                )

            explanation = (
                "Dermatology explanation could not be generated."
            )

        # =================================================
        # INSIGHT AGGREGATION
        # =================================================

        try:

            final_insight = (
                insight_aggregator.aggregate_insights(
                    top_matches=top_matches,
                    explanation=explanation,
                    confidence_data=confidence_data
                )
            )

        except Exception as e:

            if log_event:

                log_event(
                    "aggregation_error",
                    request_id,
                    "Insight aggregation failed",
                    {
                        "error": str(e)
                    }
                )

            raise HTTPException(
                status_code=500,
                detail="Error during dermatology insight aggregation"
            )

        # =================================================
        # TOTAL PROCESSING TIME
        # =================================================

        total_time = round(
            (time.time() - start_time) * 1000,
            2
        )

        # =================================================
        # PIPELINE SUCCESS
        # =================================================

        if log_event:

            log_event(
                "pipeline_completed",
                request_id,
                "Dermatology pipeline executed successfully",
                {
                    "total_time_ms": total_time
                }
            )

        # =================================================
        # FINAL RESPONSE
        # =================================================

        return {

            "suggested_resolution":

                final_insight.get(
                    "suggested_resolution",
                    "No dermatology recommendation available"
                ),

            "similar_cases":
                similar_cases_formatted,

            "confidence_score":

                float(
                    final_insight.get(
                        "confidence_score",
                        0.0
                    )
                ),

            "explanation":

                final_insight.get(
                    "explanation",
                    explanation
                ),

            "search_query":
                search_query,

            "generated_context":
                generated_context,

            "processing_time_ms":
                total_time
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
                "Dermatology pipeline failure",
                {
                    "error": str(e)
                }
            )

        raise HTTPException(
            status_code=500,
            detail={
                "suggested_resolution":
                    "Error occurred while processing dermatology request",

                "similar_cases": [],

                "confidence_score": 0.0,

                "explanation": str(e)
            }
        )