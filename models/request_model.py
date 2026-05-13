from pydantic import BaseModel, Field, validator
from typing import Optional


class ClinicalMatchRequest(BaseModel):

    chief_complaint: Optional[str] = Field(
        None,
        max_length=500,
        description="Primary complaint reported by the patient"
    )

    affected_body_part: Optional[str] = Field(
        None,
        max_length=200,
        description="Affected body part or region"
    )

    symptoms_duration: Optional[str] = Field(
        None,
        max_length=100,
        description="Duration of symptoms"
    )

    previous_injuries: Optional[str] = Field(
        None,
        max_length=500,
        description="Any previous injuries or trauma history"
    )

    current_medications: Optional[str] = Field(
        None,
        max_length=500,
        description="Current medications being used"
    )

    allergies: Optional[str] = Field(
        None,
        max_length=300,
        description="Known allergies"
    )

    occupation: Optional[str] = Field(
        None,
        max_length=200,
        description="Patient occupation"
    )

    activity_levels: Optional[str] = Field(
        None,
        max_length=100,
        description="Daily activity level of patient"
    )

    gender: Optional[str] = Field(
        None,
        max_length=20,
        description="Gender of the patient"
    )

    age: Optional[int] = Field(
        None,
        ge=0,
        le=120,
        description="Age of the patient"
    )

    doctor_name: Optional[str] = Field(
        None,
        max_length=100,
        description="Doctor or clinician name"
    )

    subjective_assessment: Optional[str] = Field(
        None,
        max_length=1000,
        description="Subjective assessment provided by clinician"
    )

    functional_assessment: Optional[str] = Field(
        None,
        max_length=1000,
        description="Functional assessment details"
    )

    physical_examination: Optional[str] = Field(
        None,
        max_length=1000,
        description="Physical examination findings"
    )

    objective_findings: Optional[str] = Field(
        None,
        max_length=1000,
        description="Objective or additional findings"
    )

    patient_pain_classification: Optional[str] = Field(
        None,
        max_length=100,
        description="Pain severity or classification"
    )

    # ---------------- VALIDATORS ----------------

    @validator(
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
        pre=True,
        always=True
    )
    def validate_optional_strings(cls, v):

        if v is None:
            return ""

        v = v.strip()

        return v if v else ""

    @validator("age")
    def validate_age(cls, v):

        if v is None:
            return v

        if v < 0 or v > 120:
            raise ValueError("age must be between 0 and 120")

        return v

    @validator("*", pre=False)
    def validate_at_least_one_field(cls, v, values, **kwargs):

        all_fields = list(values.values())

        if kwargs["field"].name == "patient_pain_classification":

            if not any(
                field not in [None, ""]
                for field in all_fields + [v]
            ):
                raise ValueError(
                    "At least one clinical input field is required"
                )

        return v