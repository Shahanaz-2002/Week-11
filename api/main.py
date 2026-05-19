import time
import logging
import uuid
import json
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    Request
)

from fastapi.responses import JSONResponse

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

        "timestamp":
            time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if extra:

        log_data.update(extra)

    logger.info(json.dumps(log_data))


# =========================================================
# APPLICATION LIFECYCLE
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        json.dumps({
            "event": "startup",
            "message": "Clinical Match API Started",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    )

    yield

    logger.info(
        json.dumps({
            "event": "shutdown",
            "message": "Clinical Match API Shutdown",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    )


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(

    title="Clinical Match API",

    version="5.0.0",

    description="""
    AI-Powered Clinical Similarity Matching API

    Features:
    - Dynamic Optional Clinical Input Processing
    - Semantic Similarity Retrieval
    - Top-2 Similar Patient Matching
    - Clinical Recommendation Generation
    - Confidence Score Analysis
    - Error Stabilization & Validation
    - Swagger/OpenAPI Documentation
    """,

    lifespan=lifespan
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
# GLOBAL EXCEPTION HANDLER
# =========================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception
):

    request_id = str(uuid.uuid4())

    log_event(
        "global_exception",
        request_id,
        "Unhandled exception occurred",
        {
            "error": str(exc)
        }
    )

    return JSONResponse(

        status_code=500,

        content={

            "status": "Failed",

            "message": "Internal Server Error",

            "request_id": request_id,

            "error": str(exc)
        }
    )


# =========================================================
# ROOT ROUTE
# =========================================================

@app.get("/")
def root():

    return {

        "message":
            "Clinical Match API Running",

        "version":
            "5.0.0",

        "status":
            "active",

        "docs":
            "/docs",

        "redoc":
            "/redoc"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    return {

        "status":
            "healthy",

        "api":
            "Clinical Match API",

        "timestamp":
            time.strftime("%Y-%m-%d %H:%M:%S")
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
    # REQUEST RECEIVED
    # =====================================================

    log_event(

        "request_received",

        request_id,

        "Clinical request received"
    )

    try:

        # =================================================
        # PROCESS INPUTS
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
        # VALIDATION
        # =================================================

        if not search_query.strip():

            raise HTTPException(

                status_code=400,

                detail={
                    "status": "Failed",
                    "message":
                        "No valid clinical input fields provided"
                }
            )

        # =================================================
        # LOG QUERY
        # =================================================

        log_event(

            "query_generated",

            request_id,

            "Clinical query generated",

            {
                "search_query":
                    search_query,

                "available_fields":
                    available_fields
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
        # RESPONSE PROCESSING
        # =================================================

        matches = result.get(
            "matches",
            []
        )

        matches = matches[:2]

        confidence_score = result.get(
            "confidence_score",
            0.0
        )

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
                "Success",

            "message":
                "Clinical matching completed successfully",

            "request_id":
                request_id,

            "api_version":
                "5.0.0",

            "request_timestamp":
                time.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "matches":
                matches,

            "total_matches_found":
                len(matches),

            "confidence_score":
                round(
                    float(confidence_score),
                    4
                ),

            "search_query":
                search_query,

            "generated_context":
                generated_context,

            "combined_symptoms":
                combined_symptoms,

            "input_fields_used":
                available_fields,

            "processing_time_ms":
                processing_time,

            "patient_metadata":
                patient_metadata,

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
                "matches_found":
                    len(matches),

                "processing_time_ms":
                    processing_time,

                "confidence_score":
                    confidence_score
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
    # PIPELINE FAILURE
    # =====================================================

    except Exception as e:

        log_event(

            "pipeline_failure",

            request_id,

            "Clinical pipeline failure",

            {
                "error":
                    str(e)
            }
        )

        raise HTTPException(

            status_code=500,

            detail={

                "status":
                    "Failed",

                "message":
                    "Internal clinical pipeline failure",

                "request_id":
                    request_id,

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
            "Patient with chronic knee pain and swelling",

        "sample_fields": [

            "chief_complaint",
            "affected_body_part",
            "symptoms_duration",
            "subjective_assessment",
            "physical_examination"
        ]
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