"""Immutable model-conditioned trajectory and ensemble records."""

from datetime import UTC, datetime
import math
from types import MappingProxyType
from typing import Annotated, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
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
    sample_seed: StrictInt | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
    )
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
        if self.initial_state.at.astimezone(UTC) != self.horizon.start_at.astimezone(
            UTC
        ):
            raise ValueError("initial state time must match horizon start_at")

        previous_at: datetime | None = None
        for point in self.points:
            if not self.horizon.start_at <= point.at <= self.horizon.end_at:
                raise ValueError("trajectory points must fall within the horizon")
            if previous_at is not None and point.at <= previous_at:
                raise ValueError("trajectory points must be strictly ordered")
            previous_at = point.at
        return self


class ScalarSummaryStatistics(_ImmutablePredictionSchema):
    """Population statistics for one scalar variable at one trajectory time."""

    count: StrictInt = Field(ge=1)
    mean: FiniteFloat
    population_std: FiniteFloat = Field(ge=0.0)
    population_variance: FiniteFloat = Field(ge=0.0)
    minimum: FiniteFloat
    maximum: FiniteFloat

    @model_validator(mode="after")
    def validate_internal_consistency(self) -> "ScalarSummaryStatistics":
        if not self.minimum <= self.mean <= self.maximum:
            raise ValueError("summary mean must fall within its minimum and maximum")
        if not math.isclose(
            self.population_std * self.population_std,
            self.population_variance,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("summary population std and variance must agree")
        return self


class TrajectorySummaryPoint(_ImmutablePredictionSchema):
    """Per-variable scalar population statistics at one aligned time."""

    at: datetime
    statistics: Annotated[
        Mapping[NonEmptyString, ScalarSummaryStatistics], Field(min_length=1)
    ]

    @field_validator("at")
    @classmethod
    def time_must_be_timezone_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value, "trajectory summary point time")

    @field_validator("statistics", mode="before")
    @classmethod
    def revalidate_statistics(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return {
                key: ScalarSummaryStatistics.model_validate(item.model_dump())
                if isinstance(item, ScalarSummaryStatistics)
                else item
                for key, item in value.items()
            }
        return value

    @field_validator("statistics")
    @classmethod
    def freeze_statistics(
        cls, value: Mapping[str, ScalarSummaryStatistics]
    ) -> Mapping[str, ScalarSummaryStatistics]:
        return MappingProxyType(dict(value))

    @field_serializer("statistics")
    def serialize_statistics(
        self, value: Mapping[str, ScalarSummaryStatistics]
    ) -> dict[str, ScalarSummaryStatistics]:
        return dict(value)


class TrajectoryEnsembleSummary(_ImmutablePredictionSchema):
    """Derived statistics that retain their aligned trajectory time axis."""

    points: Annotated[tuple[TrajectorySummaryPoint, ...], Field(min_length=1)]

    @field_validator("points", mode="before")
    @classmethod
    def revalidate_points(cls, value: object) -> object:
        if isinstance(value, (tuple, list)):
            return tuple(
                TrajectorySummaryPoint.model_validate(item.model_dump())
                if isinstance(item, TrajectorySummaryPoint)
                else item
                for item in value
            )
        return value


def summarize_trajectories(
    trajectories: tuple[Trajectory, ...],
) -> TrajectoryEnsembleSummary:
    """Derive scalar population statistics from aligned raw trajectories."""
    if not trajectories:
        raise ValueError("summary requires at least one raw trajectory")
    reference_points = trajectories[0].points
    summary_points: list[TrajectorySummaryPoint] = []
    for point_index, reference_point in enumerate(reference_points):
        expected_variables = tuple(reference_point.values)
        values_by_variable: dict[str, list[float]] = {
            variable: [] for variable in expected_variables
        }
        for trajectory in trajectories:
            if len(trajectory.points) != len(reference_points):
                raise ValueError("trajectory samples must share point alignment")
            point = trajectory.points[point_index]
            if point.at.astimezone(UTC) != reference_point.at.astimezone(UTC):
                raise ValueError("trajectory samples must share time alignment")
            if set(point.values) != set(expected_variables):
                raise ValueError("trajectory samples must share variable alignment")
            for variable in expected_variables:
                value = point.values[variable]
                if isinstance(value, tuple):
                    raise ValueError(
                        "vector trajectory values cannot be summarized without "
                        "explicit component semantics"
                    )
                values_by_variable[variable].append(float(value))

        statistics: dict[str, ScalarSummaryStatistics] = {}
        for variable, values in values_by_variable.items():
            count = len(values)
            mean = math.fsum(values) / count
            variance = math.fsum((value - mean) ** 2 for value in values) / count
            statistics[variable] = ScalarSummaryStatistics(
                count=count,
                mean=float(mean),
                population_std=float(math.sqrt(variance)),
                population_variance=float(variance),
                minimum=float(min(values)),
                maximum=float(max(values)),
            )
        summary_points.append(
            TrajectorySummaryPoint(
                at=reference_point.at,
                statistics=statistics,
            )
        )
    return TrajectoryEnsembleSummary(points=tuple(summary_points))


class TrajectoryEnsemble(_ImmutablePredictionSchema):
    """Raw stochastic samples retained under one model/case identity."""

    ensemble_id: NonEmptyString
    model_id: NonEmptyString
    model_version: NonEmptyString
    case_id: NonEmptyString
    trajectories: Annotated[tuple[Trajectory, ...], Field(min_length=1)]
    seed: StrictInt | None = None
    summary: TrajectoryEnsembleSummary | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
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

    @field_validator("summary", mode="before")
    @classmethod
    def revalidate_summary(cls, value: object) -> object:
        if isinstance(value, TrajectoryEnsembleSummary):
            return TrajectoryEnsembleSummary.model_validate(value.model_dump())
        return value

    @model_validator(mode="after")
    def validate_member_identity(self) -> "TrajectoryEnsemble":
        expected = (self.model_id, self.model_version, self.case_id)
        expected_initial_state = self.trajectories[0].initial_state
        expected_horizon = self.trajectories[0].horizon
        trajectory_ids = {
            trajectory.trajectory_id for trajectory in self.trajectories
        }
        if len(trajectory_ids) != len(self.trajectories):
            raise ValueError("trajectory ensemble member identities must be distinct")
        sample_seeds = [
            trajectory.sample_seed for trajectory in self.trajectories
        ]
        if any(value is not None for value in sample_seeds):
            if any(value is None for value in sample_seeds):
                raise ValueError(
                    "trajectory ensemble sample seed identities cannot be partial"
                )
            if len(set(sample_seeds)) != len(sample_seeds):
                raise ValueError(
                    "trajectory ensemble sample seed identities must be distinct"
                )
        for trajectory in self.trajectories:
            actual = (
                trajectory.model_id,
                trajectory.model_version,
                trajectory.case_id,
            )
            if actual != expected:
                raise ValueError(
                    "trajectory ensemble member identity must match ensemble"
                )
            if trajectory.initial_state != expected_initial_state:
                raise ValueError("trajectory ensemble members must share initial state")
            if trajectory.horizon != expected_horizon:
                raise ValueError(
                    "trajectory ensemble members must share requested horizon"
                )

        weights = [trajectory.scenario_weight for trajectory in self.trajectories]
        if not all(weight is None for weight in weights):
            if any(weight is None for weight in weights):
                raise ValueError("trajectory ensemble weight scheme cannot be partial")

            declared_weights = [
                weight for weight in weights if weight is not None
            ]
            kinds = {weight.kind for weight in declared_weights}
            if len(kinds) != 1:
                raise ValueError("trajectory ensemble weight scheme cannot mix kinds")

            total = sum(weight.value for weight in declared_weights)
            kind = declared_weights[0].kind
            if (
                kind == "probability"
                and abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE
            ):
                raise ValueError(
                    "trajectory ensemble probability weights must sum to 1 "
                    f"within {PROBABILITY_SUM_TOLERANCE}"
                )
            if kind == "relative_weight" and total <= 0.0:
                raise ValueError(
                    "trajectory ensemble relative weights need a positive total"
                )

        if self.summary is not None:
            expected_summary = summarize_trajectories(self.trajectories)
            if self.summary != expected_summary:
                raise ValueError(
                    "trajectory ensemble summary must match aligned raw samples"
                )
        return self
