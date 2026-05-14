from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import re


class ClinicalMatchRequest(BaseModel):

    # ---------------- BASIC PATIENT DETAILS ----------------

    chief_complaint: Optional[str] = Field(
        default="",
        max_length=500,
        description="Primary complaint reported by the patient"
    )

    affected_body_part: Optional[str] = Field(
        default="",
        max_length=200,
        description="Affected body part or region"
    )

    symptoms_duration: Optional[str] = Field(
        default="",
        max_length=100,
        description="Duration of symptoms"
    )

    previous_injuries: Optional[str] = Field(
        default="",
        max_length=500,
        description="Any previous injuries or trauma history"
    )

    current_medications: Optional[str] = Field(
        default="",
        max_length=500,
        description="Current medications being used"
    )

    allergies: Optional[str] = Field(
        default="",
        max_length=300,
        description="Known allergies"
    )

    occupation: Optional[str] = Field(
        default="",
        max_length=200,
        description="Patient occupation"
    )

    activity_levels: Optional[str] = Field(
        default="",
        max_length=100,
        description="Daily activity level of patient"
    )

    gender: Optional[str] = Field(
        default="",
        max_length=20,
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
        max_length=100,
        description="Doctor or clinician name"
    )

    # ---------------- CLINICAL ASSESSMENT ----------------

    subjective_assessment: Optional[str] = Field(
        default="",
        max_length=1000,
        description="Subjective assessment provided by clinician"
    )

    functional_assessment: Optional[str] = Field(
        default="",
        max_length=1000,
        description="Functional assessment details"
    )

    physical_examination: Optional[str] = Field(
        default="",
        max_length=1000,
        description="Physical examination findings"
    )

    objective_findings: Optional[str] = Field(
        default="",
        max_length=1000,
        description="Objective or additional findings"
    )

    patient_pain_classification: Optional[str] = Field(
        default="",
        max_length=100,
        description="Pain severity or classification"
    )

    # ---------------- NEW DAY 2 FIELDS ----------------

    symptoms: Optional[str] = Field(
        default="",
        max_length=1500,
        description="Combined symptom description"
    )

    doctor_notes: Optional[str] = Field(
        default="",
        max_length=2000,
        description="Additional doctor notes"
    )

    clinical_history: Optional[str] = Field(
        default="",
        max_length=2000,
        description="Complete clinical history"
    )

    search_query: Optional[str] = Field(
        default="",
        description="Generated retrieval search query"
    )

    generated_context: Optional[str] = Field(
        default="",
        description="Generated clinical context for retrieval"
    )

    # ---------------- TEXT CLEANING VALIDATOR ----------------

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
        "symptoms",
        "doctor_notes",
        "clinical_history",
        pre=True,
        always=True
    )
    def validate_optional_strings(cls, v):

        if v is None:
            return ""

        # Convert to string
        v = str(v)

        # Remove extra spaces
        v = v.strip()

        # Remove multiple spaces
        v = re.sub(r"\s+", " ", v)

        return v

    # ---------------- AGE VALIDATION ----------------

    @validator("age")
    def validate_age(cls, v):

        if v is None:
            return v

        if v < 0 or v > 120:
            raise ValueError("Age must be between 0 and 120")

        return v

    # ---------------- GENDER VALIDATION ----------------

    @validator("gender")
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

    # ---------------- AT LEAST ONE FIELD VALIDATION ----------------

    @validator("*", pre=False)
    def validate_at_least_one_field(cls, v, values, **kwargs):

        if kwargs["field"].name == "generated_context":

            valid_fields = [
                field
                for field in values.values()
                if field not in [None, "", []]
            ]

            if len(valid_fields) == 0:
                raise ValueError(
                    "At least one clinical input field is required"
                )

        return v

    # ---------------- SEARCH QUERY GENERATION ----------------

    def build_search_query(self) -> str:

        components = []

        if self.chief_complaint:
            components.append(self.chief_complaint)

        if self.affected_body_part:
            components.append(self.affected_body_part)

        if self.symptoms_duration:
            components.append(self.symptoms_duration)

        if self.patient_pain_classification:
            components.append(self.patient_pain_classification)

        if self.objective_findings:
            components.append(self.objective_findings)

        if self.physical_examination:
            components.append(self.physical_examination)

        if self.symptoms:
            components.append(self.symptoms)

        search_query = " | ".join(components)

        return search_query.strip()

    # ---------------- CONTEXT GENERATION ----------------

    def build_context(self) -> str:

        context_parts = []

        if self.age:
            context_parts.append(f"Age: {self.age}")

        if self.gender:
            context_parts.append(f"Gender: {self.gender}")

        if self.occupation:
            context_parts.append(f"Occupation: {self.occupation}")

        if self.activity_levels:
            context_parts.append(
                f"Activity Level: {self.activity_levels}"
            )

        if self.chief_complaint:
            context_parts.append(
                f"Chief Complaint: {self.chief_complaint}"
            )

        if self.affected_body_part:
            context_parts.append(
                f"Affected Body Part: {self.affected_body_part}"
            )

        if self.symptoms_duration:
            context_parts.append(
                f"Symptoms Duration: {self.symptoms_duration}"
            )

        if self.previous_injuries:
            context_parts.append(
                f"Previous Injuries: {self.previous_injuries}"
            )

        if self.current_medications:
            context_parts.append(
                f"Current Medications: {self.current_medications}"
            )

        if self.allergies:
            context_parts.append(
                f"Allergies: {self.allergies}"
            )

        if self.subjective_assessment:
            context_parts.append(
                f"Subjective Assessment: {self.subjective_assessment}"
            )

        if self.functional_assessment:
            context_parts.append(
                f"Functional Assessment: {self.functional_assessment}"
            )

        if self.physical_examination:
            context_parts.append(
                f"Physical Examination: {self.physical_examination}"
            )

        if self.objective_findings:
            context_parts.append(
                f"Objective Findings: {self.objective_findings}"
            )

        if self.patient_pain_classification:
            context_parts.append(
                f"Pain Classification: {self.patient_pain_classification}"
            )

        if self.doctor_notes:
            context_parts.append(
                f"Doctor Notes: {self.doctor_notes}"
            )

        if self.clinical_history:
            context_parts.append(
                f"Clinical History: {self.clinical_history}"
            )

        return "\n".join(context_parts)

    # ---------------- COMPLETE PROCESSING ----------------

    def generate_dynamic_inputs(self) -> Dict[str, Any]:

        search_query = self.build_search_query()

        generated_context = self.build_context()

        combined_symptoms = " ".join([
            self.chief_complaint,
            self.subjective_assessment,
            self.objective_findings,
            self.symptoms
        ]).strip()

        return {
            "search_query": search_query,
            "generated_context": generated_context,
            "combined_symptoms": combined_symptoms,
            "patient_metadata": {
                "age": self.age,
                "gender": self.gender,
                "occupation": self.occupation,
                "activity_levels": self.activity_levels
            }
        }