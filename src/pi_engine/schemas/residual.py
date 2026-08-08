"""First-class forecast residual records with provisional classifications."""

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pi_engine.schemas.common import NumericValue, Provenance
from pi_engine.schemas.outcome import Outcome


NonEmptyString = Annotated[str, Field(min_length=1)]


class _ImmutableResidualSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class ResidualCategory(str, Enum):
    """Distinct provisional explanations retained for later analysis."""

    PROCESS_NOISE = "process_noise"
    PARAMETER_UNCERTAINTY = "parameter_uncertainty"
    MODEL_DISCREPANCY = "model_discrepancy"
    PHASE_TIMING = "phase_timing"
    TOPOLOGY_COUPLING = "topology_coupling"
    MISSING_VARIABLE = "missing_variable"
    REGIME_CHANGE = "regime_change"
    STRUCTURED_UNKNOWN = "structured_unknown"


class ResidualClassification(_ImmutableResidualSchema):
    """A required provisional category with its inspectable evidence basis."""

    category: ResidualCategory
    basis: NonEmptyString


class Residual(_ImmutableResidualSchema):
    """A prediction/outcome difference retained as auditable data."""

    residual_id: NonEmptyString
    model_id: NonEmptyString
    model_version: NonEmptyString
    case_id: NonEmptyString
    variable: NonEmptyString
    unit: NonEmptyString
    predicted_value: NumericValue | None = None
    predicted_distribution_ref: NonEmptyString | None = None
    observed_outcome: Outcome
    error: NumericValue
    error_convention: Literal[
        "predicted_minus_observed", "observed_minus_predicted"
    ]
    classification: ResidualClassification
    provenance: Provenance

    @field_validator("observed_outcome", mode="before")
    @classmethod
    def revalidate_observed_outcome(cls, value: object) -> object:
        if isinstance(value, Outcome):
            return Outcome.model_validate(value.model_dump())
        return value

    @field_validator("classification", mode="before")
    @classmethod
    def revalidate_classification(cls, value: object) -> object:
        if isinstance(value, ResidualClassification):
            return ResidualClassification.model_validate(value.model_dump())
        return value

    @field_validator("provenance", mode="before")
    @classmethod
    def revalidate_provenance(cls, value: object) -> object:
        if isinstance(value, Provenance):
            return Provenance.model_validate(value.model_dump())
        return value

    @model_validator(mode="after")
    def validate_prediction_and_outcome_identity(self) -> "Residual":
        if self.predicted_value is None and self.predicted_distribution_ref is None:
            raise ValueError("a predicted value or distribution reference is required")

        observed_identity = (
            self.observed_outcome.case_id,
            self.observed_outcome.variable,
            self.observed_outcome.unit,
        )
        residual_identity = (self.case_id, self.variable, self.unit)
        if observed_identity != residual_identity:
            raise ValueError("observed outcome identity must match residual identity")
        return self
