# =========================================================
# utils.py
# =========================================================

import pandas as pd
import logging
import json
import re
import time
from typing import (
    Dict,
    Any,
    List
)

# =========================================================
# LOGGER CONFIGURATION
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# =========================================================
# LOGGING HELPER
# =========================================================

def log_event(
    event_type,
    message,
    extra=None
):

    log_data = {
        "event": event_type,
        "message": message,
        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    if extra:
        log_data.update(extra)

    logger.info(
        json.dumps(log_data)
    )

# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text

# =========================================================
# SAFE INTEGER CONVERSION
# =========================================================

def safe_int(value, default=0):

    try:

        if value in [None, "", "nan"]:
            return default

        return int(float(value))

    except Exception:

        return default

# =========================================================
# SAFE FLOAT CONVERSION
# =========================================================

def safe_float(value, default=0.0):

    try:

        if value in [None, "", "nan"]:
            return default

        return float(value)

    except Exception:

        return default

# =========================================================
# EMPTY VALUE CHECK
# =========================================================

def is_empty(value):

    return value in [
        None,
        "",
        [],
        {},
        "nan",
        "None"
    ]

# =========================================================
# DERMATOLOGY QUERY GENERATION
# =========================================================

def build_search_query(
    clinical_data: Dict[str, Any]
) -> str:

    query_parts = [

        clean_text(
            clinical_data.get(
                "chief_complaint",
                ""
            )
        ),

        clean_text(
            clinical_data.get(
                "symptoms",
                ""
            )
        ),

        clean_text(
            clinical_data.get(
                "diagnosis",
                ""
            )
        ),

        clean_text(
            clinical_data.get(
                "physical_examination",
                ""
            )
        ),

        clean_text(
            clinical_data.get(
                "objective_findings",
                ""
            )
        ),

        clean_text(
            clinical_data.get(
                "patient_pain_classification",
                ""
            )
        ),

        clean_text(
            clinical_data.get(
                "doctor_notes",
                ""
            )
        )
    ]

    filtered_parts = [

        part
        for part in query_parts
        if part
    ]

    search_query = " | ".join(
        filtered_parts
    )

    return search_query

# =========================================================
# CONTEXT GENERATION
# =========================================================

def generate_context(
    clinical_data: Dict[str, Any]
) -> str:

    context_parts = []

    field_mapping = {

        "age": "Age",

        "gender": "Gender",

        "occupation": "Occupation",

        "activity_levels": "Activity Levels",

        "chief_complaint":
            "Chief Complaint",

        "symptoms":
            "Symptoms",

        "symptoms_duration":
            "Symptoms Duration",

        "diagnosis":
            "Diagnosis",

        "subjective_assessment":
            "Subjective Assessment",

        "physical_examination":
            "Physical Examination",

        "objective_findings":
            "Objective Findings",

        "doctor_notes":
            "Doctor Notes",

        "current_medications":
            "Current Medications",

        "allergies":
            "Allergies",

        "resolution_notes":
            "Resolution Notes",

        "doctor_name":
            "Doctor Name"
    }

    for field, label in field_mapping.items():

        value = clean_text(
            clinical_data.get(
                field,
                ""
            )
        )

        if value:

            context_parts.append(
                f"{label}: {value}"
            )

    return "\n".join(
        context_parts
    )

# =========================================================
# DERMATOLOGY KEYWORD EXTRACTION
# =========================================================

def extract_dermatology_keywords(
    text: str
) -> List[str]:

    if not text:
        return []

    text = clean_text(text).lower()

    dermatology_keywords = [

        "acne",
        "eczema",
        "psoriasis",
        "dermatitis",
        "rash",
        "lesion",
        "papules",
        "pustules",
        "pigmentation",
        "hyperpigmentation",
        "itching",
        "redness",
        "dry skin",
        "fungal infection",
        "melasma",
        "skin allergy",
        "inflammation",
        "comedones",
        "scarring",
        "skin irritation",
        "blisters",
        "skin peeling",
        "skin sensitivity",
        "allergic reaction"
    ]

    found_keywords = []

    for keyword in dermatology_keywords:

        if keyword in text:

            found_keywords.append(
                keyword
            )

    return list(
        set(found_keywords)
    )

# =========================================================
# BUILD SEARCHABLE TEXT
# =========================================================

def build_searchable_text(
    case_data: Dict[str, Any]
) -> str:

    searchable_fields = [

        "chief_complaint",

        "symptoms",

        "diagnosis",

        "doctor_notes",

        "subjective_assessment",

        "physical_examination",

        "objective_findings",

        "resolution_notes",

        "category",

        "current_medications",

        "allergies"
    ]

    text_parts = []

    for field in searchable_fields:

        value = clean_text(
            case_data.get(
                field,
                ""
            )
        )

        if value:

            text_parts.append(
                value
            )

    return " | ".join(
        text_parts
    )

# =========================================================
# VALIDATE CASE RECORD
# =========================================================

def validate_case_record(
    case_record: Dict[str, Any]
) -> bool:

    required_fields = [

        "case_id",
        "chief_complaint"
    ]

    for field in required_fields:

        value = case_record.get(field)

        if is_empty(value):

            return False

    return True

# =========================================================
# NORMALIZE CASE RECORD
# =========================================================

def normalize_case_record(
    case_record: Dict[str, Any]
) -> Dict[str, Any]:

    normalized_record = {}

    for key, value in case_record.items():

        if isinstance(value, str):

            normalized_record[key] = clean_text(
                value
            )

        else:

            normalized_record[key] = value

    return normalized_record

# =========================================================
# CSV LOADER
# =========================================================

def load_cases_from_csv(
    file_path: str
) -> Dict[str, Dict[str, Any]]:

    """
    Load dermatology clinical cases
    from CSV into dictionary format
    for MongoDB insertion.
    """

    try:

        df = pd.read_csv(
            file_path
        )

        df = df.fillna("")

        case_database = {}

        skipped_records = 0

        log_event(
            "csv_loading_started",
            "Loading dermatology CSV",
            {
                "file_path": file_path,
                "rows": len(df)
            }
        )

        # =================================================
        # ITERATE ROWS
        # =================================================

        for _, row in df.iterrows():

            try:

                case_id = clean_text(
                    row.get(
                        "case_id",
                        ""
                    )
                )

                if not case_id:

                    skipped_records += 1
                    continue

                # ---------------------------------------------
                # BUILD RECORD
                # ---------------------------------------------

                case_record = {

                    "case_id":
                        case_id,

                    "chief_complaint":
                        clean_text(
                            row.get(
                                "chief_complaint",
                                ""
                            )
                        ),

                    "symptoms":
                        clean_text(
                            row.get(
                                "symptoms",
                                ""
                            )
                        ),

                    "diagnosis":
                        clean_text(
                            row.get(
                                "diagnosis",
                                ""
                            )
                        ),

                    "doctor_notes":
                        clean_text(
                            row.get(
                                "doctor_notes",
                                ""
                            )
                        ),

                    "subjective_assessment":
                        clean_text(
                            row.get(
                                "subjective_assessment",
                                ""
                            )
                        ),

                    "physical_examination":
                        clean_text(
                            row.get(
                                "physical_examination",
                                ""
                            )
                        ),

                    "objective_findings":
                        clean_text(
                            row.get(
                                "objective_findings",
                                ""
                            )
                        ),

                    "patient_pain_classification":
                        clean_text(
                            row.get(
                                "patient_pain_classification",
                                ""
                            )
                        ),

                    "category":
                        clean_text(
                            row.get(
                                "category",
                                "Dermatology"
                            )
                        ),

                    "location":
                        clean_text(
                            row.get(
                                "location",
                                ""
                            )
                        ),

                    "resolution_notes":
                        clean_text(
                            row.get(
                                "resolution_notes",
                                ""
                            )
                        ),

                    "status":
                        clean_text(
                            row.get(
                                "status",
                                "Closed"
                            )
                        ),

                    "resolution_days":
                        safe_int(
                            row.get(
                                "resolution_days",
                                0
                            )
                        ),

                    "current_medications":
                        clean_text(
                            row.get(
                                "current_medications",
                                ""
                            )
                        ),

                    "allergies":
                        clean_text(
                            row.get(
                                "allergies",
                                ""
                            )
                        ),

                    "symptoms_duration":
                        clean_text(
                            row.get(
                                "symptoms_duration",
                                ""
                            )
                        ),

                    "gender":
                        clean_text(
                            row.get(
                                "gender",
                                ""
                            )
                        ),

                    "age":
                        safe_int(
                            row.get(
                                "age",
                                0
                            )
                        ),

                    "doctor_name":
                        clean_text(
                            row.get(
                                "doctor_name",
                                ""
                            )
                        ),

                    "occupation":
                        clean_text(
                            row.get(
                                "occupation",
                                ""
                            )
                        ),

                    "activity_levels":
                        clean_text(
                            row.get(
                                "activity_levels",
                                ""
                            )
                        )
                }

                # ---------------------------------------------
                # NORMALIZATION
                # ---------------------------------------------

                case_record = normalize_case_record(
                    case_record
                )

                # ---------------------------------------------
                # VALIDATION
                # ---------------------------------------------

                if not validate_case_record(
                    case_record
                ):

                    skipped_records += 1

                    log_event(
                        "invalid_case_skipped",
                        "Invalid dermatology case skipped",
                        {
                            "case_id": case_id
                        }
                    )

                    continue

                # ---------------------------------------------
                # DYNAMIC SEARCH QUERY
                # ---------------------------------------------

                case_record[
                    "search_query"
                ] = build_search_query(
                    case_record
                )

                # ---------------------------------------------
                # GENERATED CONTEXT
                # ---------------------------------------------

                case_record[
                    "generated_context"
                ] = generate_context(
                    case_record
                )

                # ---------------------------------------------
                # SEARCHABLE TEXT
                # ---------------------------------------------

                case_record[
                    "searchable_text"
                ] = build_searchable_text(
                    case_record
                )

                # ---------------------------------------------
                # DERMATOLOGY KEYWORDS
                # ---------------------------------------------

                case_record[
                    "dermatology_keywords"
                ] = extract_dermatology_keywords(
                    case_record[
                        "searchable_text"
                    ]
                )

                # ---------------------------------------------
                # DEFAULT EMBEDDING FIELDS
                # ---------------------------------------------

                case_record[
                    "embedding"
                ] = []

                case_record[
                    "embedding_version"
                ] = ""

                case_record[
                    "embedding_model"
                ] = ""

                case_record[
                    "embedding_text"
                ] = case_record[
                    "searchable_text"
                ]

                # ---------------------------------------------
                # ADD TO DATABASE
                # ---------------------------------------------

                case_database[
                    case_id
                ] = case_record

            except Exception as row_error:

                skipped_records += 1

                log_event(
                    "row_processing_error",
                    "Error processing CSV row",
                    {
                        "error": str(row_error)
                    }
                )

                continue

        # =================================================
        # SUCCESS LOG
        # =================================================

        log_event(
            "csv_loading_completed",
            "Dermatology CSV loaded successfully",
            {
                "total_cases":
                    len(case_database),

                "skipped_records":
                    skipped_records
            }
        )

        return case_database

    # =====================================================
    # ERROR HANDLING
    # =====================================================

    except FileNotFoundError:

        log_event(
            "csv_file_not_found",
            "CSV file not found",
            {
                "file_path": file_path
            }
        )

        return {}

    except pd.errors.EmptyDataError:

        log_event(
            "csv_empty",
            "CSV file is empty",
            {
                "file_path": file_path
            }
        )

        return {}

    except Exception as e:

        log_event(
            "csv_loading_error",
            "Failed to load dermatology CSV",
            {
                "error": str(e),
                "file_path": file_path
            }
        )

        return {}