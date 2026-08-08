"""Canonical, theory-neutral schema primitives shared across PI-engine."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UncertaintyClass(str, Enum):
    """Distinct sources of uncertainty retained by the engine."""

    MEASUREMENT = "measurement"
    PARAMETER = "parameter"
    PROCESS_NOISE = "process_noise"
    MODEL_DISCREPANCY = "model_discrepancy"
    STRUCTURED_UNKNOWN = "structured_unknown"


class ModelStatus(str, Enum):
    """Operational evidence status for an explicit model."""

    EXPERIMENTAL = "experimental"
    VALIDATED = "validated"
    DEGRADED = "degraded"
    REJECTED = "rejected"


class ClosureType(str, Enum):
    """Separate descriptive closure categories with no metaphysical claim."""

    EPISTEMIC = "epistemic"
    CAUSAL = "causal"


class Confidence(BaseModel):
    """A bounded confidence value with an optional inspectable basis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float = Field(ge=0.0, le=1.0)
    basis: str | None = None


class Provenance(BaseModel):
    """Source metadata needed to audit an observation or model input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1)
    observed_at: datetime
    reference: str | None = None

    @field_validator("observed_at")
    @classmethod
    def observed_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observed_at must include a timezone")
        return value
