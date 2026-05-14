import logging
import json
import time
from datetime import datetime

from retrieval.database import collection
from retrieval.embedding import BioBERTEmbedding

from config import EMBEDDING_VERSION


# =========================================================
# LOGGER CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# EMBEDDER INITIALIZATION
# =========================================================

try:

    embedder = BioBERTEmbedding()

except Exception as e:

    logger.error(
        f"Failed to initialize BioBERTEmbedding: {str(e)}"
    )

    embedder = None


# =========================================================
# LOGGING HELPER
# =========================================================

def log_event(event_type, message, extra=None):

    log_data = {
        "event": event_type,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if extra:
        log_data.update(extra)

    logger.info(json.dumps(log_data))


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    if not text:
        return ""

    text = str(text)

    text = text.strip()

    text = " ".join(text.split())

    return text


# =========================================================
# BUILD DERMATOLOGY TEXT
# =========================================================

def build_dermatology_text(record):

    text_parts = [

        # -------------------------------------------------
        # PRIMARY CLINICAL FIELDS
        # -------------------------------------------------

        clean_text(
            record.get("chief_complaint", "")
        ),

        clean_text(
            record.get("symptoms", "")
        ),

        clean_text(
            record.get("diagnosis", "")
        ),

        clean_text(
            record.get("doctor_notes", "")
        ),

        clean_text(
            record.get("subjective_assessment", "")
        ),

        clean_text(
            record.get("physical_examination", "")
        ),

        clean_text(
            record.get("objective_findings", "")
        ),

        clean_text(
            record.get("patient_pain_classification", "")
        ),

        clean_text(
            record.get("current_medications", "")
        ),

        clean_text(
            record.get("allergies", "")
        ),

        clean_text(
            record.get("symptoms_duration", "")
        ),

        clean_text(
            record.get("resolution_notes", "")
        )
    ]

    filtered_parts = [

        part
        for part in text_parts
        if part
    ]

    return " | ".join(filtered_parts)


# =========================================================
# GENERATE AND STORE EMBEDDINGS
# =========================================================

def generate_and_store_embeddings():

    # -----------------------------------------------------
    # EMBEDDER CHECK
    # -----------------------------------------------------

    if embedder is None:

        log_event(
            "embedder_error",
            "Embedding model unavailable"
        )

        return

    try:

        records = collection.find({})

        total_records = collection.count_documents({})

    except Exception as e:

        log_event(
            "database_error",
            "Failed to fetch records",
            {
                "error": str(e)
            }
        )

        return

    # -----------------------------------------------------
    # COUNTERS
    # -----------------------------------------------------

    processed = 0

    skipped = 0

    errors = 0

    start_time = time.time()

    # -----------------------------------------------------
    # START LOG
    # -----------------------------------------------------

    log_event(
        "embedding_generation_started",
        "Dermatology embedding generation started",
        {
            "total_records": total_records
        }
    )

    # =====================================================
    # PROCESS RECORDS
    # =====================================================

    for record in records:

        try:

            case_id = record.get(
                "case_id",
                "Unknown"
            )

            # -------------------------------------------------
            # SKIP EXISTING EMBEDDINGS
            # -------------------------------------------------

            if (
                record.get("embedding_version")
                == EMBEDDING_VERSION
            ):

                skipped += 1

                continue

            # -------------------------------------------------
            # BUILD SEARCHABLE TEXT
            # -------------------------------------------------

            text = build_dermatology_text(
                record
            )

            if not text:

                log_event(
                    "empty_text_skipped",
                    "Empty dermatology text skipped",
                    {
                        "case_id": case_id
                    }
                )

                skipped += 1

                continue

            # -------------------------------------------------
            # GENERATE EMBEDDING
            # -------------------------------------------------

            embedding = embedder.get_embedding(
                text
            )

            if embedding is None:

                log_event(
                    "embedding_failed",
                    "Embedding generation failed",
                    {
                        "case_id": case_id
                    }
                )

                errors += 1

                continue

            # -------------------------------------------------
            # STORE IN DATABASE
            # -------------------------------------------------

            collection.update_one(

                {"case_id": case_id},

                {
                    "$set": {

                        "embedding":
                            embedding.tolist(),

                        "embedding_text":
                            text,

                        "embedding_version":
                            EMBEDDING_VERSION,

                        "embedding_created_at":
                            datetime.utcnow(),

                        "embedding_model":
                            "BioClinicalBERT",

                        "embedding_domain":
                            "Dermatology"
                    }
                }
            )

            processed += 1

            # -------------------------------------------------
            # SUCCESS LOG
            # -------------------------------------------------

            log_event(
                "embedding_stored",
                "Dermatology embedding stored",
                {
                    "case_id": case_id,
                    "processed": processed
                }
            )

        # =================================================
        # RECORD ERROR
        # =================================================

        except Exception as e:

            errors += 1

            log_event(
                "record_processing_error",
                "Error processing dermatology case",
                {
                    "case_id":
                        record.get(
                            "case_id",
                            "Unknown"
                        ),

                    "error":
                        str(e)
                }
            )

            continue

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    total_time = round(
        (time.time() - start_time) * 1000,
        2
    )

    summary = {

        "processed": processed,

        "skipped": skipped,

        "errors": errors,

        "total_records": total_records,

        "total_time_ms": total_time
    }

    log_event(
        "embedding_generation_completed",
        "Dermatology embedding generation completed",
        summary
    )

    print("\n=================================================")
    print("DERMATOLOGY EMBEDDING GENERATION COMPLETED")
    print("=================================================")

    print(f"Total Records : {total_records}")
    print(f"Processed     : {processed}")
    print(f"Skipped       : {skipped}")
    print(f"Errors        : {errors}")
    print(f"Time Taken    : {total_time} ms")

    print("=================================================\n")


# =========================================================
# MAIN ENTRY
# =========================================================

if __name__ == "__main__":

    generate_and_store_embeddings()