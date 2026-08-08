"""Immutable model-conditioned trajectory and ensemble records."""

from datetime import datetime
from types import MappingProxyType
from typing import Annotated, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from pi_engine.schemas.common import FiniteFloat, NumericValue, Provenance
from pi_engine.schemas.state import StateEstimate


NonEmptyString = Annotated[str, Field(min_length=1)]
# Accept no more than one part per billion of floating-point summation drift.
PROBABILITY_SUM_TOLERANCE = 1e-9


class _ImmutablePredictionSchema(BaseModel):
    """Shared validation policy for prediction artifacts."""

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


class TrajectoryHorizon(_ImmutablePredictionSchema):
    """Concrete start and end of a simulated trajectory."""

    start_at: datetime
    end_at: datetime

    @field_validator("start_at", "end_at")
    @classmethod
    def times_must_be_timezone_aware(
        cls, value: datetime, info: object
    ) -> datetime:
        field_name = getattr(info, "field_name", "horizon time")
        return _require_timezone(value, field_name)

    @model_validator(mode="after")
    def end_must_follow_start(self) -> "TrajectoryHorizon":
        if self.end_at <= self.start_at:
            raise ValueError("horizon end_at must be after start_at")
        return self


class TrajectoryPoint(_ImmutablePredictionSchema):
    """A finite state-vector sample at one trajectory time."""

    at: datetime
    values: Annotated[Mapping[NonEmptyString, NumericValue], Field(min_length=1)]

    @field_validator("at")
    @classmethod
    def time_must_be_timezone_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value, "trajectory point time")

    @field_validator("values")
    @classmethod
    def freeze_values(
        cls, value: Mapping[str, NumericValue]
    ) -> Mapping[str, NumericValue]:
        return MappingProxyType(dict(value))

    @field_serializer("values")
    def serialize_values(
        self, value: Mapping[str, NumericValue]
    ) -> dict[str, NumericValue]:
        return dict(value)


class ScenarioWeight(_ImmutablePredictionSchema):
    """A justified probability or relative scenario weight."""

    kind: Literal["probability", "relative_weight"]
    value: FiniteFloat = Field(ge=0.0)
    justification: NonEmptyString

    @model_validator(mode="after")
    def probability_must_not_exceed_one(self) -> "ScenarioWeight":
        if self.kind == "probability" and self.value > 1.0:
            raise ValueError("probability value must not exceed 1")
        return self


class Trajectory(_ImmutablePredictionSchema):
    """One explicit model-conditioned deterministic or stochastic scenario."""

    trajectory_id: NonEmptyString
    model_id: NonEmptyString
    model_version: NonEmptyString
    case_id: NonEmptyString
    initial_state: StateEstimate
    horizon: TrajectoryHorizon
    points: Annotated[tuple[TrajectoryPoint, ...], Field(min_length=1)]
    scenario_weight: ScenarioWeight | None = None
    constraints_encountered: tuple[NonEmptyString, ...]
    provenance: Provenance

    @field_validator("initial_state", mode="before")
    @classmethod
    def revalidate_initial_state(cls, value: object) -> object:
        if isinstance(value, StateEstimate):
            return StateEstimate.model_validate(value.model_dump())
        return value

    @field_validator("horizon", mode="before")
    @classmethod
    def revalidate_horizon(cls, value: object) -> object:
        if isinstance(value, TrajectoryHorizon):
            return TrajectoryHorizon.model_validate(value.model_dump())
        return value

    @field_validator("points", mode="before")
    @classmethod
    def revalidate_points(cls, value: object) -> object:
        if isinstance(value, (tuple, list)):
            return tuple(
                TrajectoryPoint.model_validate(item.model_dump())
                if isinstance(item, TrajectoryPoint)
                else item
                for item in value
            )
        return value

    @field_validator("scenario_weight", mode="before")
    @classmethod
    def revalidate_scenario_weight(cls, value: object) -> object:
        if isinstance(value, ScenarioWeight):
            return ScenarioWeight.model_validate(value.model_dump())
        return value

    @field_validator("provenance", mode="before")
    @classmethod
    def revalidate_provenance(cls, value: object) -> object:
        if isinstance(value, Provenance):
            return Provenance.model_validate(value.model_dump())
        return value

    @model_validator(mode="after")
    def validate_timeline(self) -> "Trajectory":
        if self.initial_state.at != self.horizon.start_at:
            raise ValueError("initial state time must match horizon start_at")

        previous_at: datetime | None = None
        for point in self.points:
            if not self.horizon.start_at <= point.at <= self.horizon.end_at:
                raise ValueError("trajectory points must fall within the horizon")
            if previous_at is not None and point.at <= previous_at:
                raise ValueError("trajectory points must be strictly ordered")
            previous_at = point.at
        return self


class TrajectoryEnsemble(_ImmutablePredictionSchema):
    """Raw stochastic samples retained under one model/case identity."""

    ensemble_id: NonEmptyString
    model_id: NonEmptyString
    model_version: NonEmptyString
    case_id: NonEmptyString
    trajectories: Annotated[tuple[Trajectory, ...], Field(min_length=1)]
    seed: int | None = None
    provenance: Provenance

    @field_validator("trajectories", mode="before")
    @classmethod
    def revalidate_trajectories(cls, value: object) -> object:
        if isinstance(value, (tuple, list)):
            return tuple(
                Trajectory.model_validate(item.model_dump())
                if isinstance(item, Trajectory)
                else item
                for item in value
            )
        return value

    @field_validator("provenance", mode="before")
    @classmethod
    def revalidate_provenance(cls, value: object) -> object:
        if isinstance(value, Provenance):
            return Provenance.model_validate(value.model_dump())
        return value

    @model_validator(mode="after")
    def validate_member_identity(self) -> "TrajectoryEnsemble":
        expected = (self.model_id, self.model_version, self.case_id)
        expected_initial_state = self.trajectories[0].initial_state
        expected_horizon = self.trajectories[0].horizon
        for trajectory in self.trajectories:
            actual = (
                trajectory.model_id,
                trajectory.model_version,
                trajectory.case_id,
            )
            if actual != expected:
                raise ValueError("trajectory ensemble member identity must match ensemble")
            if trajectory.initial_state != expected_initial_state:
                raise ValueError("trajectory ensemble members must share initial state")
            if trajectory.horizon != expected_horizon:
                raise ValueError("trajectory ensemble members must share requested horizon")

        weights = [trajectory.scenario_weight for trajectory in self.trajectories]
        if all(weight is None for weight in weights):
            return self
        if any(weight is None for weight in weights):
            raise ValueError("trajectory ensemble weight scheme cannot be partial")

        declared_weights = [weight for weight in weights if weight is not None]
        kinds = {weight.kind for weight in declared_weights}
        if len(kinds) != 1:
            raise ValueError("trajectory ensemble weight scheme cannot mix kinds")

        total = sum(weight.value for weight in declared_weights)
        kind = declared_weights[0].kind
        if kind == "probability" and abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise ValueError(
                "trajectory ensemble probability weights must sum to 1 "
                f"within {PROBABILITY_SUM_TOLERANCE}"
            )
        if kind == "relative_weight" and total <= 0.0:
            raise ValueError("trajectory ensemble relative weights need a positive total")
        return self
