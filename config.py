# =========================================================
# config.py
# =========================================================

import os

import torch


# =========================================================
# ENVIRONMENT
# =========================================================

ENVIRONMENT = os.getenv(
    "ENVIRONMENT",
    "development"
)


# =========================================================
# MONGODB CONFIGURATION
# =========================================================

MONGO_URI = os.getenv(

    "MONGO_URI",

    "mongodb://localhost:27017"
)

DATABASE_NAME = os.getenv(

    "DATABASE_NAME",

    "dermatology_ai"
)

COLLECTION_NAME = os.getenv(

    "COLLECTION_NAME",

    "dermatology_cases"
)


# =========================================================
# DATABASE TIMEOUTS
# =========================================================

MONGO_SERVER_SELECTION_TIMEOUT_MS = 5000

MONGO_CONNECT_TIMEOUT_MS = 5000

MONGO_SOCKET_TIMEOUT_MS = 5000


# =========================================================
# RETRIEVAL CONFIGURATION
# =========================================================

TOP_K = 2

DEFAULT_TOP_K = 2

MAX_MATCH_RESULTS = 2

MIN_SIMILARITY_SCORE = 0.20

DEFAULT_SIMILARITY_THRESHOLD = 0.20

ENABLE_LOW_SCORE_FILTER = True

LOW_SCORE_FILTER_THRESHOLD = 0.20


# =========================================================
# EMBEDDING CONFIGURATION
# =========================================================

# IMPORTANT:
# USE SAME MODEL EVERYWHERE
# embedding.py
# embedding_store.py
# retrieval_engine.py

EMBEDDING_MODEL_NAME = (

    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)

DEFAULT_EMBEDDING_MODEL = EMBEDDING_MODEL_NAME

EMBEDDING_DIM = 768

EMBEDDING_VERSION = "biobert_semantic_v2"

BATCH_SIZE = 16

NORMALIZE_EMBEDDINGS = True


# =========================================================
# DEVICE CONFIGURATION
# =========================================================

USE_GPU = torch.cuda.is_available()

DEVICE = (

    "cuda"

    if USE_GPU

    else "cpu"
)


# =========================================================
# KEYWORD BOOST CONFIGURATION
# =========================================================

ENABLE_KEYWORD_BOOST = True

KEYWORD_BOOST_FACTOR = 0.01

MAX_KEYWORD_BOOST = 0.10


# =========================================================
# API CONFIGURATION
# =========================================================

API_TITLE = (

    "AI Dermatology Clinical Match API"
)

API_DESCRIPTION = (

    "AI-powered semantic dermatology retrieval "
    "using BioBERT embeddings"
)

API_VERSION = "6.0.0"

API_HOST = "0.0.0.0"

API_PORT = 8000

REQUEST_TIMEOUT_SECONDS = 30


# =========================================================
# INPUT LIMITS
# =========================================================

MAX_TEXT_INPUT_LENGTH = 2000

MAX_QUERY_LENGTH = 2000

MAX_CONTEXT_LENGTH = 5000


# =========================================================
# CONFIDENCE THRESHOLDS
# =========================================================

VERY_HIGH_CONFIDENCE = 0.90

HIGH_CONFIDENCE = 0.75

MEDIUM_CONFIDENCE = 0.55

LOW_CONFIDENCE = 0.30


# =========================================================
# RESPONSE CONFIGURATION
# =========================================================

INCLUDE_SEARCHABLE_TEXT = False

INCLUDE_EMBEDDINGS_IN_RESPONSE = False

ENABLE_EXPLANATION_GENERATION = True


# =========================================================
# LOGGING CONFIGURATION
# =========================================================

LOG_LEVEL = "INFO"

ENABLE_JSON_LOGGING = True

LOG_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


# =========================================================
# SUPPORTED CONDITIONS
# =========================================================

SUPPORTED_DERMATOLOGY_CONDITIONS = [

    "Acne",

    "Eczema",

    "Psoriasis",

    "Fungal Infection",

    "Dermatitis",

    "Rosacea",

    "Vitiligo",

    "Melasma",

    "Skin Allergy",

    "Urticaria",

    "Seborrheic Dermatitis",

    "Folliculitis",

    "Hyperpigmentation",

    "Tinea",

    "Scabies"
]


# =========================================================
# SUPPORTED SKIN TYPES
# =========================================================

SUPPORTED_SKIN_TYPES = [

    "Oily",

    "Dry",

    "Combination",

    "Sensitive",

    "Normal"
]


# =========================================================
# DEFAULT RECOMMENDATIONS
# =========================================================

DEFAULT_RECOMMENDED_TESTS = [

    "Dermatology Clinical Examination"
]

DEFAULT_RECOMMENDED_MEDICINES = [

    "Symptomatic Skin Treatment"
]

DEFAULT_SKINCARE_PLAN = [

    "Use gentle cleanser",

    "Apply moisturizer regularly",

    "Use sunscreen daily"
]

DEFAULT_PRECAUTIONS = [

    "Avoid harsh chemicals",

    "Maintain proper skin hygiene"
]


# =========================================================
# SECURITY SETTINGS
# =========================================================

ALLOWED_GENDERS = [

    "Male",

    "Female",

    "Other",

    "Prefer Not To Say"
]


# =========================================================
# DEBUG MODE
# =========================================================

DEBUG_MODE = (

    ENVIRONMENT.lower() == "development"
)


# =========================================================
# STARTUP VALIDATION
# =========================================================

if EMBEDDING_DIM <= 0:

    raise ValueError(
        "Invalid embedding dimension"
    )

if TOP_K <= 0:

    raise ValueError(
        "TOP_K must be greater than 0"
    )

if MIN_SIMILARITY_SCORE < 0:

    raise ValueError(
        "MIN_SIMILARITY_SCORE invalid"
    )