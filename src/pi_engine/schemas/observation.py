"""Cutoff-aware raw observations for PI-engine cases."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pi_engine.schemas.common import (
    Confidence,
    ObservationValue,
    Provenance,
    UncertaintyClass,
)


class Observation(BaseModel):
    """A sourced measurement and the time at which it became usable."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    observation_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    variable: str = Field(min_length=1)
    value: ObservationValue
    unit: str = Field(min_length=1)
    event_time: datetime
    available_at: datetime
    uncertainty_class: UncertaintyClass
    confidence: Confidence
    provenance: Provenance

    @field_validator("event_time", "available_at")
    @classmethod
    def times_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observation times must include a timezone")
        return value
