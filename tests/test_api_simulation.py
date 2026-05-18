import requests
import json
import time
import uuid
from typing import Dict, Any


# =========================================================
# API CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000/clinical/match"

REQUEST_TIMEOUT = 30


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def print_divider():

    print("=" * 100)


def generate_request_id():

    return f"REQ-{uuid.uuid4().hex[:8].upper()}"


# =========================================================
# RESPONSE VALIDATION:
# SUCCESS RESPONSE
# =========================================================

def validate_success_response(data):

    required_fields = [

        "status",
        "message",
        "request_id",
        "matches",
        "total_matches_found",
        "confidence_score",
        "generated_context",
        "explanation"
    ]

    # -----------------------------------------------------
    # REQUIRED FIELD CHECK
    # -----------------------------------------------------

    for field in required_fields:

        if field not in data:

            return (
                False,
                f"Missing field: {field}"
            )

    # -----------------------------------------------------
    # TYPE CHECKS
    # -----------------------------------------------------

    if not isinstance(data["matches"], list):

        return (
            False,
            "matches should be a list"
        )

    if not isinstance(
        data["confidence_score"],
        (float, int)
    ):

        return (
            False,
            "confidence_score should be numeric"
        )

    if not (
        0.0 <=
        data["confidence_score"] <=
        1.0
    ):

        return (
            False,
            "confidence_score out of range"
        )

    # -----------------------------------------------------
    # MATCH STRUCTURE CHECK
    # -----------------------------------------------------

    for match in data["matches"]:

        required_match_fields = [

            "case_id",
            "match_score",
            "recommended_tests",
            "recommended_medicines",
            "confidence_level"
        ]

        for field in required_match_fields:

            if field not in match:

                return (
                    False,
                    f"Missing match field: {field}"
                )

    return (
        True,
        "Valid success response"
    )


# =========================================================
# RESPONSE VALIDATION:
# ERROR RESPONSE
# =========================================================

def validate_error_response(data):

    if "detail" not in data:

        return (
            False,
            "Missing detail field"
        )

    return (
        True,
        "Valid error response"
    )


# =========================================================
# PRINT TEST RESULT
# =========================================================

def print_test_result(
    passed: bool,
    message: str
):

    if passed:

        print(f"\nPASS : {message}")

    else:

        print(f"\nFAIL : {message}")


# =========================================================
# REQUEST SENDER
# =========================================================

def send_request(
    payload: Dict[str, Any],
    test_name: str,
    expected_status: int
):

    print_divider()

    print(f"TEST NAME   : {test_name}")

    request_id = generate_request_id()

    print(f"REQUEST ID  : {request_id}")

    print("\nREQUEST PAYLOAD:\n")

    print(
        json.dumps(
            payload,
            indent=4
        )
    )

    start_time = time.time()

    try:

        # =================================================
        # API REQUEST
        # =================================================

        response = requests.post(

            API_URL,

            json=payload,

            timeout=REQUEST_TIMEOUT
        )

        response_time = round(
            (
                time.time() -
                start_time
            ) * 1000,
            2
        )

        print(f"\nSTATUS CODE : {response.status_code}")

        print(f"RESPONSE TIME : {response_time} ms")

        # =================================================
        # JSON PARSING
        # =================================================

        try:

            data = response.json()

            print("\nRESPONSE JSON:\n")

            print(
                json.dumps(
                    data,
                    indent=4
                )
            )

        except Exception:

            print_test_result(
                False,
                "Invalid JSON response"
            )

            return False

        # =================================================
        # STATUS CODE CHECK
        # =================================================

        if response.status_code != expected_status:

            print_test_result(

                False,

                (
                    f"Expected status "
                    f"{expected_status}, "
                    f"got {response.status_code}"
                )
            )

            return False

        # =================================================
        # SUCCESS RESPONSE VALIDATION
        # =================================================

        if response.status_code == 200:

            is_valid, validation_message = (
                validate_success_response(
                    data
                )
            )

        else:

            # =============================================
            # ERROR RESPONSE VALIDATION
            # =============================================

            is_valid, validation_message = (
                validate_error_response(
                    data
                )
            )

        if not is_valid:

            print_test_result(
                False,
                validation_message
            )

            return False

        # =================================================
        # PASS
        # =================================================

        print_test_result(
            True,
            validation_message
        )

        return True

    # =====================================================
    # REQUEST FAILURE
    # =====================================================

    except requests.exceptions.ConnectionError:

        print_test_result(
            False,
            "Unable to connect to API server"
        )

        return False

    except requests.exceptions.Timeout:

        print_test_result(
            False,
            "Request timeout"
        )

        return False

    except Exception as e:

        print_test_result(
            False,
            f"Unexpected error -> {e}"
        )

        return False


# =========================================================
# TEST CASES
# =========================================================

test_cases = [

    # =====================================================
    # FULL VALID INPUT
    # =====================================================

    {
        "name":
            "Full Clinical Input",

        "payload": {

            "chief_complaint":
                "Right knee pain while climbing stairs",

            "affected_body_part":
                "Right Knee",

            "symptoms_duration":
                "3 weeks",

            "previous_injuries":
                "ACL tear 2 years ago",

            "current_medications":
                "Ibuprofen",

            "allergies":
                "Penicillin",

            "occupation":
                "Construction Worker",

            "activity_levels":
                "High",

            "gender":
                "Male",

            "age":
                35,

            "doctor_name":
                "Dr Kumar",

            "subjective_assessment":
                "Pain increases during movement",

            "functional_assessment":
                "Difficulty squatting",

            "physical_examination":
                "Swelling near patella",

            "objective_findings":
                "Reduced range of motion",

            "patient_pain_classification":
                "Moderate",

            "doctor_notes":
                "Possible ligament instability"
        },

        "expected_status":
            200
    },

    # =====================================================
    # PARTIAL INPUT
    # =====================================================

    {
        "name":
            "Partial Clinical Input",

        "payload": {

            "chief_complaint":
                "Lower back pain",

            "occupation":
                "Driver",

            "age":
                45
        },

        "expected_status":
            200
    },

    # =====================================================
    # SINGLE FIELD
    # =====================================================

    {
        "name":
            "Single Input Field",

        "payload": {

            "chief_complaint":
                "Shoulder stiffness"
        },

        "expected_status":
            200
    },

    # =====================================================
    # ONLY AGE
    # =====================================================

    {
        "name":
            "Only Age Provided",

        "payload": {

            "age":
                60
        },

        "expected_status":
            200
    },

    # =====================================================
    # EMPTY REQUEST
    # =====================================================

    {
        "name":
            "Empty Request",

        "payload": {},

        "expected_status":
            422
    },

    # =====================================================
    # INVALID AGE
    # =====================================================

    {
        "name":
            "Invalid Age Input",

        "payload": {

            "chief_complaint":
                "Neck pain",

            "age":
                150
        },

        "expected_status":
            422
    },

    # =====================================================
    # INVALID GENDER
    # =====================================================

    {
        "name":
            "Invalid Gender Input",

        "payload": {

            "chief_complaint":
                "Knee pain",

            "gender":
                "Alien"
        },

        "expected_status":
            422
    },

    # =====================================================
    # LONG INPUT
    # =====================================================

    {
        "name":
            "Long Clinical Description",

        "payload": {

            "subjective_assessment":
                (
                    "Patient reports chronic "
                    "lower back pain "
                ) * 30
        },

        "expected_status":
            200
    },

    # =====================================================
    # NULL OPTIONAL FIELDS
    # =====================================================

    {
        "name":
            "Null Optional Fields",

        "payload": {

            "chief_complaint":
                None,

            "occupation":
                None,

            "age":
                32
        },

        "expected_status":
            200
    },

    # =====================================================
    # MULTIPLE OPTIONAL INPUTS
    # =====================================================

    {
        "name":
            "Dynamic Optional Inputs",

        "payload": {

            "symptoms":
                "Swelling and tenderness",

            "doctor_notes":
                "Suspected inflammation",

            "clinical_history":
                "Previous sports injury"
        },

        "expected_status":
            200
    }
]


# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":

    print("\nStarting Clinical Match API Simulation...\n")

    passed = 0

    failed = 0

    results_summary = []

    total_start_time = time.time()

    # =====================================================
    # EXECUTE TEST CASES
    # =====================================================

    for test in test_cases:

        result = send_request(

            payload=test["payload"],

            test_name=test["name"],

            expected_status=test["expected_status"]
        )

        if result:

            passed += 1

            results_summary.append({

                "test":
                    test["name"],

                "status":
                    "PASS"
            })

        else:

            failed += 1

            results_summary.append({

                "test":
                    test["name"],

                "status":
                    "FAIL"
            })

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    total_execution_time = round(
        (
            time.time() -
            total_start_time
        ) * 1000,
        2
    )

    print_divider()

    print("FINAL SUMMARY\n")

    print(f"TOTAL TESTS        : {len(test_cases)}")

    print(f"PASSED             : {passed}")

    print(f"FAILED             : {failed}")

    print(
        f"TOTAL EXECUTION MS : "
        f"{total_execution_time}"
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    final_results = {

        "total_tests":
            len(test_cases),

        "passed":
            passed,

        "failed":
            failed,

        "execution_time_ms":
            total_execution_time,

        "results":
            results_summary
    }

    with open(
        "test_results.json",
        "w"
    ) as f:

        json.dump(
            final_results,
            f,
            indent=4
        )

    print(
        "\nResults saved to "
        "test_results.json"
    )

    print(
        "\nClinical Match API "
        "Simulation Completed!"
    )

    print_divider()