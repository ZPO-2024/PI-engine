"""Withheld outcome records revealed only for post-prediction evaluation."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pi_engine.schemas.common import ObservationValue, Provenance


NonEmptyString = Annotated[str, Field(min_length=1)]


class _ImmutableOutcomeSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


def _require_timezone(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return value


class ComparisonWindow(_ImmutableOutcomeSchema):
    """Optional interval over which an outcome is compared with a forecast."""

    start_at: datetime
    end_at: datetime

    @field_validator("start_at", "end_at")
    @classmethod
    def times_must_be_timezone_aware(
        cls, value: datetime, info: object
    ) -> datetime:
        field_name = getattr(info, "field_name", "comparison time")
        return _require_timezone(value, field_name)

    @model_validator(mode="after")
    def end_must_not_precede_start(self) -> "ComparisonWindow":
        if self.end_at < self.start_at:
            raise ValueError("comparison window end_at must not precede start_at")
        return self


class Outcome(_ImmutableOutcomeSchema):
    """One withheld, sourced observation used only after prediction."""

    outcome_id: NonEmptyString
    case_id: NonEmptyString
    variable: NonEmptyString
    unit: NonEmptyString
    value: ObservationValue
    event_time: datetime
    available_at: datetime
    comparison_window: ComparisonWindow | None = None
    provenance: Provenance

    @field_validator("event_time", "available_at")
    @classmethod
    def times_must_be_timezone_aware(
        cls, value: datetime, info: object
    ) -> datetime:
        field_name = getattr(info, "field_name", "outcome time")
        return _require_timezone(value, field_name)

    @field_validator("comparison_window", mode="before")
    @classmethod
    def revalidate_comparison_window(cls, value: object) -> object:
        if isinstance(value, ComparisonWindow):
            return ComparisonWindow.model_validate(value.model_dump())
        return value

    @field_validator("provenance", mode="before")
    @classmethod
    def revalidate_provenance(cls, value: object) -> object:
        if isinstance(value, Provenance):
            return Provenance.model_validate(value.model_dump())
        return value

    @model_validator(mode="after")
    def validate_chronology(self) -> "Outcome":
        if self.available_at < self.event_time:
            raise ValueError("available_at must not precede event_time")
        if self.comparison_window is not None and not (
            self.comparison_window.start_at
            <= self.event_time
            <= self.comparison_window.end_at
        ):
            raise ValueError("comparison window must contain event_time")
        return self
