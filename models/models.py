from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator
)

from typing import (
    List,
    Optional,
    Dict,
    Any
)

import re


# =========================================================
# REQUEST MODEL
# =========================================================

class ClinicalMatchRequest(BaseModel):

    # -----------------------------------------------------
    # BASIC PATIENT DETAILS
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # CLINICAL ASSESSMENTS
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # ADDITIONAL OPTIONAL INPUTS
    # -----------------------------------------------------

    symptoms: Optional[str] = Field(
        default="",
        description="Additional symptoms"
    )

    doctor_notes: Optional[str] = Field(
        default="",
        description="Doctor notes or comments"
    )

    clinical_history: Optional[str] = Field(
        default="",
        description="Clinical history of patient"
    )

    # -----------------------------------------------------
    # FIELD VALIDATORS
    # -----------------------------------------------------

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
        "symptoms",
        "doctor_notes",
        "clinical_history",
        mode="before"
    )
    @classmethod
    def clean_optional_fields(cls, v):

        if v is None:
            return ""

        # Convert to string
        value = str(v)

        # Remove leading/trailing spaces
        value = value.strip()

        # Remove multiple spaces
        value = re.sub(r"\s+", " ", value)

        return value

    # -----------------------------------------------------
    # AGE VALIDATION
    # -----------------------------------------------------

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):

        if v is None:
            return v

        if v < 0 or v > 120:

            raise ValueError(
                "Age must be between 0 and 120"
            )

        return v

    # -----------------------------------------------------
    # GENDER VALIDATION
    # -----------------------------------------------------

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):

        if not v:
            return ""

        allowed = [
            "male",
            "female",
            "other",
            "prefer not to say"
        ]

        if v.lower() not in allowed:

            raise ValueError(
                "Gender must be Male, Female, Other, or Prefer not to say"
            )

        return v.title()

    # -----------------------------------------------------
    # AT LEAST ONE FIELD VALIDATION
    # -----------------------------------------------------

    @model_validator(mode="after")
    def validate_at_least_one_field(self):

        values = self.model_dump()

        non_empty_fields = [
            value
            for value in values.values()
            if value not in [None, "", [], {}]
        ]

        if len(non_empty_fields) == 0:

            raise ValueError(
                "At least one clinical input field is required"
            )

        return self

    # -----------------------------------------------------
    # AVAILABLE INPUT FIELDS
    # -----------------------------------------------------

    def get_available_fields(self) -> List[str]:

        available_fields = []

        for field_name, value in self.model_dump().items():

            if value not in [None, "", [], {}]:

                available_fields.append(field_name)

        return available_fields

    # -----------------------------------------------------
    # SEARCH QUERY GENERATION
    # -----------------------------------------------------

    def build_search_query(self) -> str:

        components = []

        weighted_inputs = [
            self.chief_complaint,
            self.affected_body_part,
            self.symptoms_duration,
            self.symptoms,
            self.subjective_assessment,
            self.physical_examination,
            self.objective_findings,
            self.patient_pain_classification,
            self.previous_injuries
        ]

        for item in weighted_inputs:

            if item not in [None, ""]:

                components.append(item)

        return " | ".join(components).strip()

    # -----------------------------------------------------
    # CONTEXT GENERATION
    # -----------------------------------------------------

    def build_context(self) -> str:

        context_parts = []

        field_map = {
            "Age": self.age,
            "Gender": self.gender,
            "Occupation": self.occupation,
            "Activity Levels": self.activity_levels,
            "Chief Complaint": self.chief_complaint,
            "Affected Body Part": self.affected_body_part,
            "Symptoms Duration": self.symptoms_duration,
            "Previous Injuries": self.previous_injuries,
            "Current Medications": self.current_medications,
            "Allergies": self.allergies,
            "Subjective Assessment": self.subjective_assessment,
            "Functional Assessment": self.functional_assessment,
            "Physical Examination": self.physical_examination,
            "Objective Findings": self.objective_findings,
            "Pain Classification": self.patient_pain_classification,
            "Symptoms": self.symptoms,
            "Doctor Notes": self.doctor_notes,
            "Clinical History": self.clinical_history,
            "Doctor Name": self.doctor_name
        }

        for field_name, value in field_map.items():

            if value not in [None, "", [], {}]:

                context_parts.append(
                    f"{field_name}: {value}"
                )

        return "\n".join(context_parts)

    # -----------------------------------------------------
    # COMBINED SYMPTOMS
    # -----------------------------------------------------

    def build_combined_symptoms(self) -> str:

        symptoms = [
            self.chief_complaint,
            self.subjective_assessment,
            self.objective_findings,
            self.physical_examination,
            self.symptoms
        ]

        combined = " ".join([
            item
            for item in symptoms
            if item not in [None, ""]
        ])

        return combined.strip()

    # -----------------------------------------------------
    # PATIENT METADATA
    # -----------------------------------------------------

    def build_patient_metadata(self) -> Dict[str, Any]:

        return {
            "age": self.age,
            "gender": self.gender,
            "occupation": self.occupation,
            "activity_levels": self.activity_levels,
            "doctor_name": self.doctor_name
        }

    # -----------------------------------------------------
    # COMPLETE DYNAMIC INPUT GENERATION
    # -----------------------------------------------------

    def generate_dynamic_inputs(self) -> Dict[str, Any]:

        return {

            "search_query":
                self.build_search_query(),

            "generated_context":
                self.build_context(),

            "combined_symptoms":
                self.build_combined_symptoms(),

            "patient_metadata":
                self.build_patient_metadata(),

            "available_fields":
                self.get_available_fields()
        }


# =========================================================
# MATCH RESULT MODEL
# =========================================================

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

    symptoms_duration: str = Field(
        default="Unknown",
        description="Symptoms duration"
    )

    doctor_notes: str = Field(
        default="No notes available",
        description="Doctor or clinician notes"
    )

    explanation: str = Field(
        default="Match generated using clinical similarity search",
        description="Explanation for similarity match"
    )

    recommended_tests: List[str] = Field(
        default_factory=list,
        description="Recommended diagnostic tests"
    )

    recommended_medicines: List[str] = Field(
        default_factory=list,
        description="Recommended medicines"
    )

    matched_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords contributing to the match"
    )

    confidence_level: str = Field(
        default="Moderate",
        description="Confidence level of prediction"
    )


# =========================================================
# FINAL RESPONSE MODEL
# =========================================================

class ClinicalMatchResponse(BaseModel):

    status: str = Field(
        ...,
        description="API response status"
    )

    message: str = Field(
        ...,
        description="Response message"
    )

    request_id: str = Field(
        ...,
        description="Unique request identifier"
    )

    matches: List[MatchResult] = Field(
        default_factory=list,
        description="Top matched clinical cases"
    )

    total_matches_found: int = Field(
        default=0,
        description="Number of matches returned"
    )

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall confidence score"
    )

    search_query: str = Field(
        default="",
        description="Generated search query"
    )

    generated_context: str = Field(
        default="",
        description="Dynamically generated clinical context"
    )

    input_fields_used: List[str] = Field(
        default_factory=list,
        description="Clinical fields used for matching"
    )

    processing_time_ms: float = Field(
        default=0.0,
        description="API response processing time in milliseconds"
    )

    explanation: str = Field(
        default="Clinical similarity matching completed successfully",
        description="Explanation of how the result was generated"
    )