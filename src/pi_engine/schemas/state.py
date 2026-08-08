"""Canonical state estimates for PI-engine cases."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator


NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class StateEstimate(BaseModel):
    """Observed, latent, uncertain, and boundary components at one time."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    at: datetime
    observed: dict[str, float]
    latent: dict[str, float]
    uncertainty: dict[str, NonNegativeFloat]
    boundary: dict[str, float]

    @field_validator("at")
    @classmethod
    def at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("state time must include a timezone")
        return value
