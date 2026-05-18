import time
import logging
import uuid
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from models.models import (
    ClinicalMatchRequest,
    ClinicalMatchResponse
)

from services.clinical_match_service import (
    clinical_match_pipeline
)


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    force=True
)

logger = logging.getLogger(__name__)


def log_event(event_type, request_id, message, extra=None):

    log_data = {
        "event": event_type,
        "request_id": request_id,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if extra:
        log_data.update(extra)

    logger.info(json.dumps(log_data))


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Clinical Match API",
    version="3.0.0",
    description="""
    AI-powered Clinical Similarity Matching API

    Features:
    - Dynamic Clinical Input Processing
    - Optional Field Handling
    - Search Query Generation
    - Clinical Context Generation
    - Similar Patient Case Retrieval
    - Top 2 Clinical Match Recommendation
    """
)


# =========================================================
# CORS CONFIGURATION
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# ROOT ROUTE
# =========================================================

@app.get("/")
def root():

    return {
        "message": "Welcome to Clinical Match API",
        "version": "3.0.0",
        "docs": "/docs",
        "status": "running"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    return {
        "status": "Clinical Match API is running"
    }


# =========================================================
# MAIN CLINICAL MATCH ROUTE
# =========================================================

@app.post(
    "/clinical/match",
    response_model=ClinicalMatchResponse
)
def clinical_match(request: ClinicalMatchRequest):

    # -----------------------------------------------------
    # REQUEST ID
    # -----------------------------------------------------

    request_id = str(uuid.uuid4())

    start_time = time.time()

    # -----------------------------------------------------
    # LOG INCOMING REQUEST
    # -----------------------------------------------------

    log_event(
        "request_received",
        request_id,
        "Incoming clinical request"
    )

    try:

        # =================================================
        # REQUEST VALIDATION
        # =================================================

        request_data = request.model_dump()

        non_empty_fields = [
            value
            for value in request_data.values()
            if value not in [None, "", [], {}]
        ]

        if len(non_empty_fields) == 0:

            log_event(
                "validation_error",
                request_id,
                "Empty clinical request received"
            )

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid Input",
                    "message": "At least one clinical field is required"
                }
            )

        # =================================================
        # DYNAMIC INPUT PROCESSING
        # =================================================

        processed_inputs = request.generate_dynamic_inputs()

        search_query = processed_inputs.get(
            "search_query",
            ""
        )

        generated_context = processed_inputs.get(
            "generated_context",
            ""
        )

        combined_symptoms = processed_inputs.get(
            "combined_symptoms",
            ""
        )

        patient_metadata = processed_inputs.get(
            "patient_metadata",
            {}
        )

        available_fields = processed_inputs.get(
            "available_fields",
            []
        )

        # -------------------------------------------------
        # LOG SEARCH QUERY
        # -------------------------------------------------

        log_event(
            "search_query_generated",
            request_id,
            "Clinical search query generated",
            {
                "search_query": search_query,
                "available_fields": available_fields
            }
        )

        # -------------------------------------------------
        # LOG GENERATED CONTEXT
        # -------------------------------------------------

        log_event(
            "context_generated",
            request_id,
            "Clinical context generated",
            {
                "generated_context": generated_context
            }
        )

        # =================================================
        # PIPELINE EXECUTION
        # =================================================

        result = clinical_match_pipeline(
            request=request,
            request_id=request_id,
            search_query=search_query,
            generated_context=generated_context,
            combined_symptoms=combined_symptoms,
            patient_metadata=patient_metadata
        )

        # =================================================
        # NO RESULTS FOUND
        # =================================================

        if not result or len(result.get("matches", [])) == 0:

            log_event(
                "no_result",
                request_id,
                "No similar patient cases found"
            )

            raise HTTPException(
                status_code=404,
                detail={
                    "error": "No Results",
                    "message": "No similar patient cases found",
                    "matches": [],
                    "total_matches_found": 0,
                    "confidence_score": 0.0
                }
            )

        # =================================================
        # LIMIT TO TOP 2 MATCHES
        # =================================================

        top_matches = result.get("matches", [])[:2]

        result["matches"] = top_matches
        result["total_matches_found"] = len(top_matches)

        # =================================================
        # RESPONSE TIME
        # =================================================

        response_time = round(
            (time.time() - start_time) * 1000,
            2
        )

        # =================================================
        # SUCCESS LOGGING
        # =================================================

        log_event(
            "response_ready",
            request_id,
            "Clinical response prepared successfully",
            {
                "response_time_ms": response_time,
                "matches_returned": len(top_matches)
            }
        )

        # =================================================
        # FINAL RESPONSE
        # =================================================

        final_response = {
            **result,
            "request_id": request_id,
            "search_query": search_query,
            "generated_context": generated_context,
            "processing_time_ms": response_time,
            "input_fields_used": available_fields
        }

        return ClinicalMatchResponse(**final_response)

    # =====================================================
    # EXPECTED ERRORS
    # =====================================================

    except HTTPException as http_err:

        log_event(
            "http_error",
            request_id,
            "Handled HTTP exception",
            {
                "status_code": http_err.status_code,
                "detail": str(http_err.detail)
            }
        )

        raise http_err

    # =====================================================
    # UNEXPECTED ERRORS
    # =====================================================

    except Exception as e:

        log_event(
            "pipeline_error",
            request_id,
            "Clinical pipeline execution failure",
            {
                "error": str(e)
            }
        )

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal Server Error",
                "message": "Error occurred while processing clinical request",
                "matches": [],
                "confidence_score": 0.0,
                "explanation": str(e)
            }
        )


# =========================================================
# DEBUG SAMPLE ROUTE
# =========================================================

@app.get("/debug/sample-query")
def sample_query():

    return {

        "sample_search_query":
            "Knee pain | Swelling | ACL injury | Difficulty walking",

        "sample_context":
            """
            Chief Complaint: Knee pain and instability
            Affected Body Part: Right Knee
            Symptoms Duration: 3 months
            Previous Injuries: ACL tear history
            Current Medications: Ibuprofen
            Allergies: None
            Occupation: Athlete
            Activity Levels: High
            Gender: Male
            Age: 24
            Doctor Name: Dr. Smith
            Subjective Assessment: Pain increases during running
            Functional Assessment: Difficulty climbing stairs
            Physical Examination: Swelling and tenderness over ACL region
            Objective Findings: Reduced range of motion
            Patient Pain Classification: Moderate
            """
    }


# =========================================================
# SERVER START
# =========================================================

if __name__ == "__main__":

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )