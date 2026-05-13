import requests
import json
import time
import uuid

# =========================
# API CONFIGURATION
# =========================

API_URL = "http://127.0.0.1:8000/clinical/match"


# =========================
# HELPER FUNCTIONS
# =========================

def print_divider():
    print("=" * 80)


def generate_request_id():
    return f"REQ-{uuid.uuid4().hex[:8].upper()}"


# =========================
# SUCCESS RESPONSE VALIDATION
# =========================

def validate_success_response(data):

    required_fields = [
        "status",
        "message",
        "matches",
        "confidence_score",
        "generated_clinical_context",
        "explanation"
    ]

    # ---------------- FIELD CHECK ----------------
    for field in required_fields:

        if field not in data:
            return False, f"Missing field: {field}"

    # ---------------- TYPE CHECKS ----------------
    if not isinstance(data["matches"], list):
        return False, "matches should be a list"

    if not isinstance(data["confidence_score"], (float, int)):
        return False, "confidence_score should be numeric"

    if not (0.0 <= data["confidence_score"] <= 1.0):
        return False, "confidence_score out of range"

    # ---------------- MATCH STRUCTURE CHECK ----------------
    for match in data["matches"]:

        required_match_fields = [
            "case_id",
            "match_score",
            "recommended_tests",
            "recommended_medicines"
        ]

        for field in required_match_fields:

            if field not in match:
                return False, f"Missing match field: {field}"

    return True, "Valid response structure"


# =========================
# ERROR RESPONSE VALIDATION
# =========================

def validate_error_response(data):

    if "detail" not in data:
        return False, "Missing 'detail' field"

    return True, "Valid error response"


# =========================
# REQUEST SENDER
# =========================

def send_request(payload, test_name, expected_status):

    print_divider()

    print(f"TEST NAME : {test_name}")

    request_id = generate_request_id()

    print(f"REQUEST ID : {request_id}")

    start_time = time.time()

    try:

        response = requests.post(
            API_URL,
            json=payload
        )

        response_time = round(
            (time.time() - start_time) * 1000,
            2
        )

        print(f"STATUS CODE : {response.status_code}")

        print(f"RESPONSE TIME : {response_time} ms")

        try:

            data = response.json()

            print("\nRESPONSE JSON:")

            print(json.dumps(data, indent=4))

        except Exception:

            print("FAIL : Invalid JSON response")

            return False

        # ---------------- STATUS CHECK ----------------
        if response.status_code != expected_status:

            print(
                f"FAIL : Expected {expected_status}, "
                f"got {response.status_code}"
            )

            return False

        # ---------------- SUCCESS VALIDATION ----------------
        if response.status_code == 200:

            is_valid, message = validate_success_response(data)

            if not is_valid:

                print(f"FAIL : {message}")

                return False

        # ---------------- ERROR VALIDATION ----------------
        else:

            is_valid, message = validate_error_response(data)

            if not is_valid:

                print(f"FAIL : {message}")

                return False

        print("PASS")

        return True

    except Exception as e:

        print(f"ERROR : Request failed -> {e}")

        return False


# =========================
# TEST CASES
# =========================

test_cases = [

    # ==================================================
    # FULL VALID INPUT
    # ==================================================
    {
        "name": "Full Clinical Input",

        "payload": {
            "chief_complaint": "Right knee pain while climbing stairs",

            "affected_body_part": "Right Knee",

            "symptoms_duration": "3 weeks",

            "previous_injuries": "ACL tear 2 years ago",

            "current_medications": "Ibuprofen",

            "allergies": "Penicillin",

            "occupation": "Construction Worker",

            "activity_levels": "High",

            "gender": "Male",

            "age": 35,

            "doctor_name": "Dr Kumar",

            "subjective_assessment":
                "Pain increases during movement",

            "functional_assessment":
                "Difficulty squatting",

            "physical_examination":
                "Swelling near patella",

            "objective_findings":
                "Reduced range of motion",

            "patient_pain_classification":
                "Moderate"
        },

        "expected_status": 200
    },

    # ==================================================
    # PARTIAL INPUT
    # ==================================================
    {
        "name": "Partial Clinical Input",

        "payload": {
            "chief_complaint": "Lower back pain",

            "occupation": "Driver",

            "age": 45
        },

        "expected_status": 200
    },

    # ==================================================
    # ONLY ONE FIELD
    # ==================================================
    {
        "name": "Single Input Field",

        "payload": {
            "chief_complaint": "Shoulder stiffness"
        },

        "expected_status": 200
    },

    # ==================================================
    # ONLY AGE
    # ==================================================
    {
        "name": "Only Age Provided",

        "payload": {
            "age": 60
        },

        "expected_status": 200
    },

    # ==================================================
    # EMPTY REQUEST
    # ==================================================
    {
        "name": "Empty Request",

        "payload": {},

        "expected_status": 422
    },

    # ==================================================
    # INVALID AGE
    # ==================================================
    {
        "name": "Invalid Age Input",

        "payload": {
            "chief_complaint": "Neck pain",

            "age": 150
        },

        "expected_status": 422
    },

    # ==================================================
    # LONG INPUT
    # ==================================================
    {
        "name": "Long Clinical Description",

        "payload": {
            "subjective_assessment":
                "Patient reports chronic lower back pain "
                * 30
        },

        "expected_status": 200
    },

    # ==================================================
    # NULL FIELDS
    # ==================================================
    {
        "name": "Null Optional Fields",

        "payload": {
            "chief_complaint": None,

            "occupation": None,

            "age": 32
        },

        "expected_status": 200
    }
]


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":

    print("\nStarting Clinical Match API Simulation...\n")

    passed = 0

    failed = 0

    results_summary = []

    for test in test_cases:

        result = send_request(
            payload=test["payload"],
            test_name=test["name"],
            expected_status=test["expected_status"]
        )

        if result:

            passed += 1

            results_summary.append({
                "test": test["name"],
                "status": "PASS"
            })

        else:

            failed += 1

            results_summary.append({
                "test": test["name"],
                "status": "FAIL"
            })

    print_divider()

    print("FINAL SUMMARY")

    print(f"TOTAL TESTS : {len(test_cases)}")

    print(f"PASSED      : {passed}")

    print(f"FAILED      : {failed}")

    # =========================
    # SAVE TEST RESULTS
    # =========================

    with open("test_results.json", "w") as f:

        json.dump(
            results_summary,
            f,
            indent=4
        )

    print("\nResults saved to test_results.json")

    print("Clinical Match API Simulation Completed!")