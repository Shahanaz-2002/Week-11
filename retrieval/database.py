from pymongo import MongoClient
from pymongo.errors import (
    ConnectionFailure,
    ServerSelectionTimeoutError
)

import logging
import json
import time
from typing import List, Dict, Any

from config import (
    MONGO_URI,
    DATABASE_NAME,
    COLLECTION_NAME
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
# LOGGING HELPER
# =========================================================

def log_event(
    event_type: str,
    message: str,
    extra: Dict[str, Any] = None
):

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
# DATABASE CONNECTION
# =========================================================

client = None

db = None

collection = None

try:

    client = MongoClient(

        MONGO_URI,

        serverSelectionTimeoutMS=5000,

        connectTimeoutMS=5000,

        socketTimeoutMS=5000
    )

    # -----------------------------------------------------
    # TEST CONNECTION
    # -----------------------------------------------------

    client.admin.command("ping")

    db = client[DATABASE_NAME]

    collection = db[COLLECTION_NAME]

    log_event(
        "db_connected",
        "MongoDB connection established successfully",
        {
            "database": DATABASE_NAME,
            "collection": COLLECTION_NAME
        }
    )

except (
    ConnectionFailure,
    ServerSelectionTimeoutError
) as e:

    log_event(
        "db_connection_error",
        "MongoDB connection failed",
        {
            "error": str(e)
        }
    )

    collection = None

except Exception as e:

    log_event(
        "db_unknown_error",
        "Unexpected MongoDB initialization failure",
        {
            "error": str(e)
        }
    )

    collection = None


# =========================================================
# SAFE VALUE CLEANER
# =========================================================

def safe_value(
    value,
    default=""
):

    if value in [None, ""]:

        return default

    return value


# =========================================================
# VALIDATE CASE RECORD
# =========================================================

def validate_case_record(
    record: Dict
) -> bool:

    if not isinstance(record, dict):

        return False

    required_fields = [

        "case_id",
        "embedding"
    ]

    for field in required_fields:

        if field not in record:

            return False

    embedding = record.get("embedding")

    if not isinstance(embedding, list):

        return False

    if len(embedding) == 0:

        return False

    return True


# =========================================================
# NORMALIZE CASE RECORD
# =========================================================

def normalize_case_record(
    record: Dict
) -> Dict:

    return {

        "case_id":
            str(
                safe_value(
                    record.get("case_id"),
                    "Unknown"
                )
            ),

        "embedding":
            record.get(
                "embedding",
                []
            ),

        "chief_complaint":
            safe_value(
                record.get(
                    "chief_complaint"
                ),
                ""
            ),

        "affected_body_part":
            safe_value(
                record.get(
                    "affected_body_part"
                ),
                ""
            ),

        "symptoms":
            safe_value(
                record.get(
                    "symptoms"
                ),
                ""
            ),

        "symptoms_duration":
            safe_value(
                record.get(
                    "symptoms_duration"
                ),
                ""
            ),

        "subjective_assessment":
            safe_value(
                record.get(
                    "subjective_assessment"
                ),
                ""
            ),

        "functional_assessment":
            safe_value(
                record.get(
                    "functional_assessment"
                ),
                ""
            ),

        "physical_examination":
            safe_value(
                record.get(
                    "physical_examination"
                ),
                ""
            ),

        "objective_findings":
            safe_value(
                record.get(
                    "objective_findings"
                ),
                ""
            ),

        "patient_pain_classification":
            safe_value(
                record.get(
                    "patient_pain_classification"
                ),
                ""
            ),

        "doctor_notes":
            safe_value(
                record.get(
                    "doctor_notes"
                ),
                "No notes available"
            ),

        "clinical_history":
            safe_value(
                record.get(
                    "clinical_history"
                ),
                ""
            ),

        "previous_injuries":
            safe_value(
                record.get(
                    "previous_injuries"
                ),
                ""
            ),

        "recommended_tests":
            record.get(
                "recommended_tests",
                []
            ),

        "recommended_medicines":
            record.get(
                "recommended_medicines",
                []
            ),

        "case_description":
            safe_value(
                record.get(
                    "case_description"
                ),
                ""
            ),

        "category":
            safe_value(
                record.get(
                    "category"
                ),
                "Unknown"
            ),

        "location":
            safe_value(
                record.get(
                    "location"
                ),
                "Unknown"
            ),

        "resolution_notes":
            safe_value(
                record.get(
                    "resolution_notes"
                ),
                ""
            )
    }


# =========================================================
# FETCH COMPLETE CASE DATABASE
# =========================================================

def fetch_case_database() -> List[Dict]:

    start_time = time.time()

    case_database = []

    # -----------------------------------------------------
    # DATABASE VALIDATION
    # -----------------------------------------------------

    if collection is None:

        log_event(
            "db_unavailable",
            "MongoDB collection unavailable"
        )

        return case_database

    try:

        # -------------------------------------------------
        # FETCH RECORDS
        # -------------------------------------------------

        records = list(

            collection.find(
                {},
                {"_id": 0}
            )
        )

        processed_records = 0

        skipped_records = 0

        # -------------------------------------------------
        # PROCESS RECORDS
        # -------------------------------------------------

        for record in records:

            try:

                if not validate_case_record(record):

                    skipped_records += 1

                    log_event(
                        "invalid_record_skipped",
                        "Invalid MongoDB record skipped"
                    )

                    continue

                normalized_record = (
                    normalize_case_record(
                        record
                    )
                )

                case_database.append(
                    normalized_record
                )

                processed_records += 1

            except Exception as e:

                skipped_records += 1

                log_event(
                    "record_processing_error",
                    "Error processing MongoDB record",
                    {
                        "error": str(e)
                    }
                )

                continue

        # -------------------------------------------------
        # FINAL LOGGING
        # -------------------------------------------------

        total_time = round(

            (
                time.time() -
                start_time
            ) * 1000,

            2
        )

        log_event(
            "db_fetch_success",
            "Case database loaded successfully",
            {
                "processed_records":
                    processed_records,

                "skipped_records":
                    skipped_records,

                "total_records":
                    len(case_database),

                "fetch_time_ms":
                    total_time
            }
        )

    except Exception as e:

        log_event(
            "db_fetch_error",
            "MongoDB fetch operation failed",
            {
                "error": str(e)
            }
        )

    return case_database


# =========================================================
# INSERT SINGLE CASE
# =========================================================

def insert_case(
    case_data: Dict
) -> bool:

    if collection is None:

        log_event(
            "db_unavailable",
            "Insert failed - database unavailable"
        )

        return False

    try:

        # -------------------------------------------------
        # INPUT VALIDATION
        # -------------------------------------------------

        if not isinstance(case_data, dict):

            raise ValueError(
                "case_data must be dictionary"
            )

        required_fields = [

            "case_id",
            "embedding"
        ]

        for field in required_fields:

            if field not in case_data:

                raise ValueError(
                    f"Missing required field: {field}"
                )

        embedding = case_data.get("embedding")

        if not isinstance(embedding, list):

            raise ValueError(
                "Embedding must be list"
            )

        if len(embedding) == 0:

            raise ValueError(
                "Embedding cannot be empty"
            )

        # -------------------------------------------------
        # INSERT RECORD
        # -------------------------------------------------

        collection.insert_one(case_data)

        log_event(
            "db_insert_success",
            "Clinical case inserted successfully",
            {
                "case_id":
                    case_data.get(
                        "case_id"
                    )
            }
        )

        return True

    except Exception as e:

        log_event(
            "db_insert_error",
            "MongoDB insert operation failed",
            {
                "error": str(e)
            }
        )

        return False


# =========================================================
# FETCH SINGLE CASE BY ID
# =========================================================

def fetch_case_by_id(
    case_id: str
):

    if collection is None:

        return None

    try:

        record = collection.find_one(

            {
                "case_id": str(case_id)
            },

            {
                "_id": 0
            }
        )

        if not record:

            return None

        return normalize_case_record(
            record
        )

    except Exception as e:

        log_event(
            "fetch_case_error",
            "Failed to fetch case by ID",
            {
                "case_id": case_id,
                "error": str(e)
            }
        )

        return None


# =========================================================
# DATABASE HEALTH CHECK
# =========================================================

def check_database_health() -> Dict[str, Any]:

    try:

        if collection is None:

            return {

                "status": "unavailable",

                "database":
                    DATABASE_NAME,

                "collection":
                    COLLECTION_NAME
            }

        total_records = (
            collection.count_documents({})
        )

        return {

            "status": "healthy",

            "database":
                DATABASE_NAME,

            "collection":
                COLLECTION_NAME,

            "total_records":
                total_records
        }

    except Exception as e:

        return {

            "status": "error",

            "error": str(e)
        }


# =========================================================
# CLOSE DATABASE CONNECTION
# =========================================================

def close_database_connection():

    global client

    try:

        if client:

            client.close()

            log_event(
                "db_connection_closed",
                "MongoDB connection closed successfully"
            )

    except Exception as e:

        log_event(
            "db_close_error",
            "Error closing MongoDB connection",
            {
                "error": str(e)
            }
        )