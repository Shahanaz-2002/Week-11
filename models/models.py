from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional


# ---------------- REQUEST MODEL ----------------
class ClinicalMatchRequest(BaseModel):

    chief_complaint: Optional[str] = Field(
        default="",
        description="Primary complaint reported by the patient"
    )

    affected_body_part: Optional[str] = Field(
        default="",
        description="Affected body part or region"
    )

    symptoms_duration: Optional[str] = Field(
        default="",
        description="Duration of symptoms"
    )

    previous_injuries: Optional[str] = Field(
        default="",
        description="Any previous injuries or trauma history"
    )

    current_medications: Optional[str] = Field(
        default="",
        description="Current medications being used"
    )

    allergies: Optional[str] = Field(
        default="",
        description="Known allergies"
    )

    occupation: Optional[str] = Field(
        default="",
        description="Patient occupation"
    )

    activity_levels: Optional[str] = Field(
        default="",
        description="Daily activity level"
    )

    gender: Optional[str] = Field(
        default="",
        description="Gender of the patient"
    )

    age: Optional[int] = Field(
        default=None,
        ge=0,
        le=120,
        description="Age of the patient"
    )

    doctor_name: Optional[str] = Field(
        default="",
        description="Doctor or clinician name"
    )

    subjective_assessment: Optional[str] = Field(
        default="",
        description="Subjective assessment provided by clinician"
    )

    functional_assessment: Optional[str] = Field(
        default="",
        description="Functional assessment details"
    )

    physical_examination: Optional[str] = Field(
        default="",
        description="Physical examination findings"
    )

    objective_findings: Optional[str] = Field(
        default="",
        description="Objective or additional findings"
    )

    patient_pain_classification: Optional[str] = Field(
        default="",
        description="Pain severity or classification"
    )

    # ---------------- FIELD VALIDATORS ----------------

    @field_validator(
        "chief_complaint",
        "affected_body_part",
        "symptoms_duration",
        "previous_injuries",
        "current_medications",
        "allergies",
        "occupation",
        "activity_levels",
        "gender",
        "doctor_name",
        "subjective_assessment",
        "functional_assessment",
        "physical_examination",
        "objective_findings",
        "patient_pain_classification",
        mode="before"
    )
    @classmethod
    def clean_optional_fields(cls, v):

        if v is None:
            return ""

        return str(v).strip()

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):

        if v is None:
            return v

        if v < 0 or v > 120:
            raise ValueError("age must be between 0 and 120")

        return v

    # ---------------- MODEL VALIDATOR ----------------

    @model_validator(mode="after")
    def validate_at_least_one_field(self):

        values = self.model_dump()

        non_empty_fields = [
            value for value in values.values()
            if value not in [None, ""]
        ]

        if len(non_empty_fields) == 0:
            raise ValueError(
                "At least one clinical input field is required"
            )

        return self


# ---------------- MATCH RESULT MODEL ----------------
class MatchResult(BaseModel):

    case_id: str = Field(
        ...,
        min_length=1,
        description="Unique matched case identifier"
    )

    match_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score between 0 and 1"
    )

    chief_complaint: str = Field(
        default="Unknown",
        description="Chief complaint of matched patient"
    )

    affected_body_part: str = Field(
        default="Unknown",
        description="Affected body part of matched patient"
    )

    recommended_tests: List[str] = Field(
        default_factory=list,
        description="Recommended diagnostic tests"
    )

    recommended_medicines: List[str] = Field(
        default_factory=list,
        description="Recommended medicines"
    )

    doctor_notes: str = Field(
        default="No notes available",
        description="Doctor or clinician notes"
    )


# ---------------- FINAL RESPONSE MODEL ----------------
class ClinicalMatchResponse(BaseModel):

    status: str = Field(
        ...,
        description="API response status"
    )

    message: str = Field(
        ...,
        description="Response message"
    )

    matches: List[MatchResult] = Field(
        default_factory=list,
        description="Top matched clinical cases"
    )

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence score"
    )

    generated_clinical_context: str = Field(
        ...,
        description="Dynamically generated clinical context"
    )

    explanation: str = Field(
        ...,
        description="Explanation of how the result was generated"
    )