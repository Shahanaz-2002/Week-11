import time
import logging
import uuid
import json

from fastapi import FastAPI, HTTPException
import uvicorn

from models.models import (
    ClinicalMatchRequest,
    ClinicalMatchResponse
)

from services.clinical_match_service import (
    clinical_match_pipeline
)


# ---------------- LOGGING CONFIG ----------------
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


# ---------------- FASTAPI APP ----------------
app = FastAPI(
    title="Clinical Match API",
    version="1.0.0",
    description="AI-powered Clinical Similarity Matching API"
)


# ---------------- MAIN API ROUTE ----------------
@app.post(
    "/clinical/match",
    response_model=ClinicalMatchResponse
)
def clinical_match(request: ClinicalMatchRequest):

    request_id = str(uuid.uuid4())

    start_time = time.time()

    # ---------------- REQUEST LOGGING ----------------
    log_event(
        "request_received",
        request_id,
        "Incoming clinical match request"
    )

    try:

        # ---------------- EMPTY REQUEST VALIDATION ----------------
        request_data = request.model_dump()

        non_empty_fields = [
            value for value in request_data.values()
            if value not in [None, ""]
        ]

        if len(non_empty_fields) == 0:

            log_event(
                "validation_error",
                request_id,
                "Empty clinical request"
            )

            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid Input",
                    "message": "At least one clinical field is required"
                }
            )

        # ---------------- PIPELINE EXECUTION ----------------
        result = clinical_match_pipeline(
            request=request,
            request_id=request_id
        )

        # ---------------- EMPTY RESULT HANDLING ----------------
        if not result:

            log_event(
                "no_result",
                request_id,
                "No matching cases found"
            )

            raise HTTPException(
                status_code=404,
                detail={
                    "error": "No Results",
                    "message": "No similar patient cases found",
                    "matches": [],
                    "confidence_score": 0.0
                }
            )

        # ---------------- RESPONSE TIME ----------------
        response_time = round(
            (time.time() - start_time) * 1000,
            2
        )

        log_event(
            "response_ready",
            request_id,
            "Clinical response prepared",
            {
                "response_time_ms": response_time
            }
        )

        return ClinicalMatchResponse(**result)

    # ---------------- EXPECTED ERRORS ----------------
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

    # ---------------- UNEXPECTED ERRORS ----------------
    except Exception as e:

        log_event(
            "pipeline_error",
            request_id,
            "Pipeline execution failure",
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


# ---------------- HEALTH CHECK ----------------
@app.get("/health")
def health_check():

    return {
        "status": "Clinical Match API is running"
    }


# ---------------- ROOT ROUTE ----------------
@app.get("/")
def root():

    return {
        "message": "Welcome to Clinical Match API",
        "version": "1.0.0",
        "docs": "/docs"
    }


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )