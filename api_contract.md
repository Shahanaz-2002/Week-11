# API Contract – Clinical Match API

---

# Overview

This document defines the API contract for the
AI-powered Clinical Match API.

The system intelligently analyzes available clinical
patient inputs and retrieves the most similar
historical patient cases.

The API supports:
- Dynamic optional field processing
- AI-powered semantic similarity matching
- Clinical recommendation retrieval
- Partial input handling
- Top matched patient case retrieval

This API is designed to serve as the foundation for:
- AI-assisted clinical recommendations
- Patient case similarity search
- Clinical decision support systems
- Intelligent healthcare assistance pipelines

---

# 🔹 1. Endpoint Details

| Property | Value |
|---|---|
| URL | `/clinical/match` |
| Method | `POST` |
| Content-Type | `application/json` |

---

# 🔹 2. Request Schema (Input)

# Description

The API accepts patient clinical information.

IMPORTANT:
All fields are OPTIONAL.

The API dynamically processes only the available inputs
and generates an intelligent clinical search context.

---

# Request Fields

| Field Name | Type | Required | Description |
|---|---|---|---|
| chief_complaint | string | No | Primary complaint reported by patient |
| affected_body_part | string | No | Affected body region |
| symptoms_duration | string | No | Duration of symptoms |
| previous_injuries | string | No | Injury or trauma history |
| current_medications | string | No | Current medications |
| allergies | string | No | Known allergies |
| occupation | string | No | Patient occupation |
| activity_levels | string | No | Physical activity level |
| gender | string | No | Patient gender |
| age | integer | No | Patient age |
| doctor_name | string | No | Clinician or doctor name |
| subjective_assessment | string | No | Subjective assessment notes |
| functional_assessment | string | No | Functional limitations |
| physical_examination | string | No | Physical examination findings |
| objective_findings | string | No | Objective clinical findings |
| patient_pain_classification | string | No | Pain severity classification |

---

# 🔹 3. Example Request

## Full Clinical Input

```json
{
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
  "subjective_assessment": "Pain increases during movement",
  "functional_assessment": "Difficulty squatting",
  "physical_examination": "Swelling near patella",
  "objective_findings": "Reduced range of motion",
  "patient_pain_classification": "Moderate"
}
```

---

# Partial Input Example

```json
{
  "chief_complaint": "Lower back pain",
  "occupation": "Driver",
  "age": 45
}
```

---

# Minimal Input Example

```json
{
  "chief_complaint": "Shoulder stiffness"
}
```

---

# 🔹 4. Response Schema (Output)

# Description

The API returns:
- Top matched historical patient cases
- Similarity match scores
- Recommended tests
- Recommended medicines
- AI-generated clinical context
- Confidence score
- Explanation of result generation

---

# Response Fields

| Field Name | Type | Description |
|---|---|---|
| status | string | API response status |
| message | string | Response message |
| matches | array | Top matched patient cases |
| confidence_score | float | Overall confidence score |
| generated_clinical_context | string | Dynamically generated clinical summary |
| explanation | string | Explanation of similarity result |

---

# Match Object Fields

| Field Name | Type | Description |
|---|---|---|
| case_id | string | Historical patient case ID |
| match_score | float | Similarity score |
| chief_complaint | string | Complaint from matched case |
| affected_body_part | string | Body part from matched case |
| recommended_tests | array | Suggested diagnostic tests |
| recommended_medicines | array | Suggested medicines |
| doctor_notes | string | Additional clinician notes |

---

# 🔹 5. Example Success Response

```json
{
  "status": "success",
  "message": "Top matching patient cases retrieved successfully",

  "matches": [
    {
      "case_id": "CASE_102",

      "match_score": 0.91,

      "chief_complaint": "Knee pain during walking",

      "affected_body_part": "Right Knee",

      "recommended_tests": [
        "MRI",
        "X-Ray"
      ],

      "recommended_medicines": [
        "Ibuprofen",
        "Muscle Relaxant"
      ],

      "doctor_notes":
        "Possible ligament strain with inflammation"
    },

    {
      "case_id": "CASE_045",

      "match_score": 0.87,

      "chief_complaint": "Patellar discomfort",

      "affected_body_part": "Knee",

      "recommended_tests": [
        "CT Scan"
      ],

      "recommended_medicines": [
        "Paracetamol"
      ],

      "doctor_notes":
        "Observe swelling progression"
    }
  ],

  "confidence_score": 0.89,

  "generated_clinical_context":
    "35 year old male construction worker presenting with right knee pain while climbing stairs with previous ACL injury history.",

  "explanation":
    "Matches were retrieved using AI-based semantic similarity analysis on available clinical inputs."
}
```

---

# 🔹 6. Validation Rules

| Validation Rule | Description |
|---|---|
| At least one field required | Empty request not allowed |
| age range | Must be between 0 and 120 |
| Optional fields | Empty values handled safely |
| Null values | Automatically cleaned |
| Semantic processing | Works even with partial input |

---

# 🔹 7. Error Handling

# Validation Error Example

```json
{
  "detail": {
    "error": "Invalid Input",
    "message": "At least one clinical field is required"
  }
}
```

---

# Invalid Age Example

```json
{
  "detail": [
    {
      "type": "less_than_equal",
      "loc": [
        "body",
        "age"
      ],
      "msg": "Input should be less than or equal to 120"
    }
  ]
}
```

---

# No Similar Cases Found

```json
{
  "detail": {
    "error": "No Results",
    "message": "No similar patient cases found",
    "matches": [],
    "confidence_score": 0.0
  }
}
```

---

# Internal Server Error

```json
{
  "detail": {
    "error": "Internal Server Error",
    "message": "Error occurred while processing clinical request"
  }
}
```

---

# 🔹 8. Health Check Endpoint

| Property | Value |
|---|---|
| URL | `/health` |
| Method | `GET` |

---

# Example Response

```json
{
  "status": "Clinical Match API is running"
}
```

---

# 🔹 9. Swagger Documentation

FastAPI automatically generates Swagger documentation.

| URL | Purpose |
|---|---|
| `/docs` | Swagger UI |
| `/redoc` | ReDoc Documentation |

---

# 🔹 10. Future Enhancements

Planned future improvements include:
- FAISS vector similarity search
- BioClinicalBERT integration
- Multi-patient recommendation ranking
- Clinical risk scoring
- Real-time database retrieval
- Recommendation explainability engine
- Medical ontology integration

---

# 🔹 11. Technology Stack

| Component | Technology |
|---|---|
| Backend Framework | FastAPI |
| AI Embeddings | Sentence Transformers |
| Similarity Engine | Cosine Similarity |
| Validation | Pydantic |
| API Documentation | Swagger UI |
| Testing | Postman |
| Language | Python |

---

# 🔹 12. API Status Codes

| Status Code | Meaning |
|---|---|
| 200 | Success |
| 400 | Invalid Input |
| 404 | No Similar Cases Found |
| 422 | Validation Error |
| 500 | Internal Server Error |

---