"""Canonical state estimates for PI-engine cases."""

from datetime import datetime
from types import MappingProxyType
from typing import Mapping

from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from pi_engine.schemas.common import NumericValue


class StateEstimate(BaseModel):
    """Observed, latent, uncertain, and boundary components at one time."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    at: datetime
    observed: Mapping[str, NumericValue]
    latent: Mapping[str, NumericValue]
    uncertainty: Mapping[str, NumericValue]
    boundary: Mapping[str, NumericValue]

    @field_validator("at")
    @classmethod
    def at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("state time must include a timezone")
        return value

    @field_validator("uncertainty")
    @classmethod
    def uncertainty_must_be_nonnegative(
        cls, value: Mapping[str, NumericValue]
    ) -> Mapping[str, NumericValue]:
        for uncertainty in value.values():
            values = uncertainty if isinstance(uncertainty, tuple) else (uncertainty,)
            if any(item < 0.0 for item in values):
                raise ValueError("uncertainty values must be nonnegative")
        return value

    @field_validator("observed", "latent", "uncertainty", "boundary")
    @classmethod
    def freeze_component_mapping(
        cls, value: Mapping[str, NumericValue]
    ) -> Mapping[str, NumericValue]:
        return MappingProxyType(dict(value))

    @field_serializer("observed", "latent", "uncertainty", "boundary")
    def serialize_component_mapping(
        self, value: Mapping[str, NumericValue]
    ) -> dict[str, NumericValue]:
        return dict(value)
