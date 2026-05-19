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


# =========================================================
# LOG EVENT FUNCTION
# =========================================================

def log_event(
    event_type,
    request_id,
    message,
    extra=None
):

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
    version="4.0.0",
    description="""
    AI-powered Clinical Similarity Matching API

    Features:
    - Dynamic Clinical Input Processing
    - Semantic Clinical Retrieval
    - Top-2 Similar Patient Matching
    - Recommendation Generation
    - Confidence Score Analysis
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
    allow_headers=["*"]
)


# =========================================================
# ROOT ROUTE
# =========================================================

@app.get("/")
def root():

    return {
        "message": "Clinical Match API Running",
        "version": "4.0.0",
        "docs": "/docs",
        "status": "active"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "api": "Clinical Match API"
    }


# =========================================================
# MAIN CLINICAL MATCH ENDPOINT
# =========================================================

@app.post(
    "/clinical/match",
    response_model=ClinicalMatchResponse
)
def clinical_match(
    request: ClinicalMatchRequest
):

    request_id = str(uuid.uuid4())

    start_time = time.time()

    # =====================================================
    # LOG REQUEST
    # =====================================================

    log_event(
        "request_received",
        request_id,
        "Clinical request received"
    )

    try:

        # =================================================
        # PROCESS DYNAMIC INPUTS
        # =================================================

        processed_inputs = (
            request.generate_dynamic_inputs()
        )

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

        # =================================================
        # EMPTY QUERY VALIDATION
        # =================================================

        if not search_query.strip():

            raise HTTPException(
                status_code=400,
                detail="Clinical search query is empty"
            )

        # =================================================
        # LOG QUERY
        # =================================================

        log_event(
            "query_generated",
            request_id,
            "Clinical search query generated",
            {
                "search_query": search_query,
                "available_fields": available_fields
            }
        )

        # =================================================
        # EXECUTE PIPELINE
        # =================================================

        result = clinical_match_pipeline(

            request=request,

            request_id=request_id,

            search_query=search_query,

            generated_context=generated_context,

            combined_symptoms=combined_symptoms,

            patient_metadata=patient_metadata,

            log_event=log_event
        )

        # =================================================
        # RESPONSE TIME
        # =================================================

        processing_time = round(
            (
                time.time() - start_time
            ) * 1000,
            2
        )

        # =================================================
        # FINAL RESPONSE
        # =================================================

        final_response = {

            "status":
                result.get(
                    "status",
                    "Success"
                ),

            "message":
                result.get(
                    "message",
                    "Clinical matching completed"
                ),

            "request_id":
                request_id,

            "api_version":
                "4.0.0",

            "request_timestamp":
                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "matches":
                result.get(
                    "matches",
                    []
                )[:2],

            "total_matches_found":
                min(
                    2,
                    result.get(
                        "total_matches_found",
                        0
                    )
                ),

            "confidence_score":
                result.get(
                    "confidence_score",
                    0.0
                ),

            "search_query":
                search_query,

            "generated_context":
                generated_context,

            "input_fields_used":
                available_fields,

            "processing_time_ms":
                processing_time,

            "explanation":
                result.get(
                    "explanation",
                    "Top clinical matches retrieved using semantic similarity"
                )
        }

        # =================================================
        # SUCCESS LOG
        # =================================================

        log_event(
            "response_generated",
            request_id,
            "Clinical response generated",
            {
                "matches":
                    final_response[
                        "total_matches_found"
                    ],

                "processing_time_ms":
                    processing_time
            }
        )

        return ClinicalMatchResponse(
            **final_response
        )

    # =====================================================
    # HTTP ERRORS
    # =====================================================

    except HTTPException as http_error:

        log_event(
            "http_error",
            request_id,
            "HTTP exception occurred",
            {
                "status_code":
                    http_error.status_code,

                "detail":
                    str(http_error.detail)
            }
        )

        raise http_error

    # =====================================================
    # UNKNOWN ERRORS
    # =====================================================

    except Exception as e:

        log_event(
            "pipeline_failure",
            request_id,
            "Clinical pipeline failure",
            {
                "error": str(e)
            }
        )

        raise HTTPException(
            status_code=500,
            detail={
                "status": "Failed",
                "message":
                    "Internal clinical pipeline failure",
                "error":
                    str(e)
            }
        )


# =========================================================
# DEBUG ROUTE
# =========================================================

@app.get("/debug/sample")
def debug_sample():

    return {

        "sample_query":
            "Knee pain | ACL injury | swelling",

        "sample_context":
            "Patient with chronic knee pain and swelling"
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