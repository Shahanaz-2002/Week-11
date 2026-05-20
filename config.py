
MONGO_URI = "mongodb://localhost:27017"

DATABASE_NAME = "dermatology_ai"

COLLECTION_NAME = "dermatology_cases"


# =========================================================
# RETRIEVAL CONFIGURATION
# =========================================================

TOP_K = 2

MIN_SIMILARITY_SCORE = 0.20

MAX_MATCH_RESULTS = 2

EMBEDDING_DIM = 768

EMBEDDING_VERSION = "dermatology_biobert_v1"

ENABLE_KEYWORD_BOOST = True

KEYWORD_BOOST_FACTOR = 0.01

MAX_KEYWORD_BOOST = 0.10


# =========================================================
# EMBEDDING MODEL CONFIGURATION
# =========================================================

EMBEDDING_MODEL_NAME = (
    "dmis-lab/biobert-base-cased-v1.1"
)

USE_GPU = False

BATCH_SIZE = 16


# =========================================================
# API CONFIGURATION
# =========================================================

API_TITLE = (
    "AI Dermatology Clinical Match API"
)

API_DESCRIPTION = (
    "Semantic dermatology case retrieval using BioBERT embeddings"
)

API_VERSION = "6.0.0"

API_HOST = "0.0.0.0"

API_PORT = 8000

REQUEST_TIMEOUT_SECONDS = 30


# =========================================================
# CONFIDENCE THRESHOLDS
# =========================================================

VERY_HIGH_CONFIDENCE = 0.90

HIGH_CONFIDENCE = 0.75

MEDIUM_CONFIDENCE = 0.55

LOW_CONFIDENCE = 0.30


# =========================================================
# DERMATOLOGY CONDITIONS
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
# DEFAULT DERMATOLOGY RECOMMENDATIONS
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
# LOGGING CONFIGURATION
# =========================================================

LOG_LEVEL = "INFO"

ENABLE_JSON_LOGGING = True

LOG_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


# =========================================================
# DATABASE HEALTH SETTINGS
# =========================================================

MONGO_SERVER_SELECTION_TIMEOUT_MS = 5000

MONGO_CONNECT_TIMEOUT_MS = 5000

MONGO_SOCKET_TIMEOUT_MS = 5000


# =========================================================
# RETRIEVAL FILTERING
# =========================================================

ENABLE_LOW_SCORE_FILTER = True

LOW_SCORE_FILTER_THRESHOLD = 0.20


# =========================================================
# RESPONSE SETTINGS
# =========================================================

INCLUDE_SEARCHABLE_TEXT = False

INCLUDE_EMBEDDINGS_IN_RESPONSE = False

ENABLE_EXPLANATION_GENERATION = True


# =========================================================
# SECURITY SETTINGS
# =========================================================

MAX_TEXT_INPUT_LENGTH = 2000

ALLOWED_GENDERS = [

    "Male",

    "Female",

    "Other",

    "Prefer Not To Say"
]