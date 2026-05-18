from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
import re


class ClinicalMatchRequest(BaseModel):

    # =====================================================
    # BASIC PATIENT DETAILS
    # =====================================================

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

    # =====================================================
    # CLINICAL ASSESSMENT
    # =====================================================

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

    # =====================================================
    # ADDITIONAL CLINICAL FIELDS
    # =====================================================

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

    # =====================================================
    # SYSTEM GENERATED FIELDS
    # =====================================================

    search_query: Optional[str] = Field(
        default="",
        description="Generated retrieval search query"
    )

    generated_context: Optional[str] = Field(
        default="",
        description="Generated clinical context for retrieval"
    )

    # =====================================================
    # TEXT CLEANING VALIDATOR
    # =====================================================

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

        # Remove leading/trailing spaces
        v = v.strip()

        # Remove multiple spaces
        v = re.sub(r"\s+", " ", v)

        # Remove unwanted special characters
        v = re.sub(r"[^\w\s,.\-:/()]", "", v)

        return v

    # =====================================================
    # AGE VALIDATION
    # =====================================================

    @validator("age")
    def validate_age(cls, v):

        if v is None:
            return v

        if v < 0 or v > 120:
            raise ValueError(
                "Age must be between 0 and 120"
            )

        return v

    # =====================================================
    # GENDER VALIDATION
    # =====================================================

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

    # =====================================================
    # AT LEAST ONE FIELD VALIDATION
    # =====================================================

    @validator("*", pre=False)
    def validate_at_least_one_field(
        cls,
        v,
        values,
        **kwargs
    ):

        if kwargs["field"].name == "generated_context":

            valid_fields = [
                field
                for field in values.values()
                if field not in [None, "", [], {}]
            ]

            if len(valid_fields) == 0:

                raise ValueError(
                    "At least one clinical input field is required"
                )

        return v

    # =====================================================
    # AVAILABLE INPUT FIELDS
    # =====================================================

    def get_available_fields(self) -> List[str]:

        available_fields = []

        for field_name, value in self.dict().items():

            if value not in [None, "", [], {}]:

                available_fields.append(field_name)

        return available_fields

    # =====================================================
    # SEARCH QUERY GENERATION
    # =====================================================

    def build_search_query(self) -> str:

        components = []

        weighted_fields = [
            self.chief_complaint,
            self.affected_body_part,
            self.symptoms,
            self.subjective_assessment,
            self.physical_examination,
            self.objective_findings,
            self.patient_pain_classification,
            self.previous_injuries,
            self.functional_assessment
        ]

        for item in weighted_fields:

            if item:
                components.append(item)

        search_query = " | ".join(components)

        return search_query.strip()

    # =====================================================
    # CONTEXT GENERATION
    # =====================================================

    def build_context(self) -> str:

        context_parts = []

        field_mappings = {
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

        for field_name, value in field_mappings.items():

            if value not in [None, "", [], {}]:

                context_parts.append(
                    f"{field_name}: {value}"
                )

        return "\n".join(context_parts)

    # =====================================================
    # COMBINED SYMPTOMS GENERATION
    # =====================================================

    def build_combined_symptoms(self) -> str:

        symptom_parts = [
            self.chief_complaint,
            self.subjective_assessment,
            self.objective_findings,
            self.physical_examination,
            self.symptoms,
            self.patient_pain_classification
        ]

        combined = " ".join([
            str(part)
            for part in symptom_parts
            if part not in [None, "", []]
        ])

        return combined.strip()

    # =====================================================
    # PATIENT METADATA GENERATION
    # =====================================================

    def build_patient_metadata(self) -> Dict[str, Any]:

        return {
            "age": self.age,
            "gender": self.gender,
            "occupation": self.occupation,
            "activity_levels": self.activity_levels,
            "doctor_name": self.doctor_name
        }

    # =====================================================
    # COMPLETE DYNAMIC PROCESSING
    # =====================================================

    def generate_dynamic_inputs(self) -> Dict[str, Any]:

        search_query = self.build_search_query()

        generated_context = self.build_context()

        combined_symptoms = self.build_combined_symptoms()

        patient_metadata = self.build_patient_metadata()

        available_fields = self.get_available_fields()

        return {
            "search_query": search_query,
            "generated_context": generated_context,
            "combined_symptoms": combined_symptoms,
            "patient_metadata": patient_metadata,
            "available_fields": available_fields
        }