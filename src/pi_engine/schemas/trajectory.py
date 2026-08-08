"""Immutable model-conditioned trajectory and ensemble records."""

from datetime import datetime
import math
import sys
from types import MappingProxyType
from typing import Annotated, Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from pi_engine._numeric import stable_population_mean_std, utc_instant_key
from pi_engine.schemas.common import FiniteFloat, NumericValue, Provenance
from pi_engine.schemas.state import StateEstimate


NonEmptyString = Annotated[str, Field(min_length=1)]
# Accept no more than one part per billion of floating-point summation drift.
PROBABILITY_SUM_TOLERANCE = 1e-9


class TrajectorySummaryError(ValueError):
    """Aligned finite samples could not produce representable statistics."""


class _ImmutablePredictionSchema(BaseModel):
    """Shared validation policy for prediction artifacts."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


def _require_timezone(value: datetime, field_name: str) -> datetime:
    try:
        offset = value.utcoffset()
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field_name} must include a valid timezone") from exc
    if value.tzinfo is None or offset is None:
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
        if utc_instant_key(self.end_at) <= utc_instant_key(self.start_at):
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


class ConstraintActivation(_ImmutablePredictionSchema):
    """One source-retained constraint activation with exact audit scope."""

    constraint_id: NonEmptyString
    activated_at: datetime
    variable: NonEmptyString
    component: NonEmptyString | None
    modeled_domain: NonEmptyString
    structurally_hard: StrictBool
    irreversible_within_horizon: StrictBool
    hardness_basis: NonEmptyString
    available_at: datetime
    provenance: Provenance

    @field_validator("activated_at", "available_at")
    @classmethod
    def times_must_be_timezone_aware(
        cls, value: datetime, info: object
    ) -> datetime:
        return _require_timezone(
            value, getattr(info, "field_name", "constraint activation time")
        )

    @field_validator("provenance", mode="before")
    @classmethod
    def revalidate_provenance(cls, value: object) -> object:
        if isinstance(value, Provenance):
            return Provenance.model_validate(value.model_dump())
        return value

    @model_validator(mode="after")
    def validate_availability(self) -> "ConstraintActivation":
        if self.provenance.reference is None or not self.provenance.reference:
            raise ValueError("constraint activation provenance requires a reference")
        if utc_instant_key(self.available_at) < utc_instant_key(
            self.provenance.observed_at
        ):
            raise ValueError(
                "constraint activation available_at cannot precede provenance "
                "observed_at"
            )
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
    rng_scheme: NonEmptyString | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    initial_state: StateEstimate
    horizon: TrajectoryHorizon
    points: Annotated[tuple[TrajectoryPoint, ...], Field(min_length=1)]
    scenario_weight: ScenarioWeight | None = None
    constraints_encountered: tuple[NonEmptyString, ...]
    constraint_activations: tuple[ConstraintActivation, ...] = Field(
        default=(),
        exclude_if=lambda value: not value,
    )
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

    @field_validator("constraint_activations", mode="before")
    @classmethod
    def revalidate_constraint_activations(cls, value: object) -> object:
        if isinstance(value, (tuple, list)):
            return tuple(
                ConstraintActivation.model_validate(item.model_dump())
                if isinstance(item, ConstraintActivation)
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
    def validate_timeline(self) -> "Trajectory":
        start_key = utc_instant_key(self.horizon.start_at)
        end_key = utc_instant_key(self.horizon.end_at)
        if utc_instant_key(self.initial_state.at) != start_key:
            raise ValueError("initial state time must match horizon start_at")

        previous_key: int | None = None
        for point in self.points:
            point_key = utc_instant_key(point.at)
            if not start_key <= point_key <= end_key:
                raise ValueError("trajectory points must fall within the horizon")
            if previous_key is not None and point_key <= previous_key:
                raise ValueError("trajectory points must be strictly ordered")
            previous_key = point_key

        activation_keys: list[tuple[object, ...]] = []
        for activation in self.constraint_activations:
            activated_key = utc_instant_key(activation.activated_at)
            if not start_key <= activated_key <= end_key:
                raise ValueError(
                    "constraint activation must occur within the trajectory horizon"
                )
            if self.constraints_encountered.count(activation.constraint_id) != 1:
                raise ValueError(
                    "constraint activation id must occur exactly once in "
                    "constraints_encountered"
                )
            activation_keys.append(
                (
                    activation.constraint_id,
                    activated_key,
                    activation.variable,
                    activation.component,
                    activation.modeled_domain,
                )
            )
        if len(set(activation_keys)) != len(activation_keys):
            raise ValueError("constraint activations must be unique")
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
            if utc_instant_key(point.at) != utc_instant_key(reference_point.at):
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
            mean, standard_deviation, minimum, maximum = (
                stable_population_mean_std(
                    tuple(values),
                    tuple(1.0 for _ in values),
                    error_type=TrajectorySummaryError,
                )
            )
            if standard_deviation == 0.0:
                variance = 0.0
            else:
                if standard_deviation > math.sqrt(sys.float_info.max):
                    raise TrajectorySummaryError(
                        "summary population variance is not representable "
                        f"for {variable}"
                    )
                variance = standard_deviation * standard_deviation
                if variance == 0.0:
                    raise TrajectorySummaryError(
                        "summary population variance is not representable "
                        f"for {variable}"
                    )
            statistics[variable] = ScalarSummaryStatistics(
                count=count,
                mean=float(mean),
                population_std=float(standard_deviation),
                population_variance=float(variance),
                minimum=float(minimum),
                maximum=float(maximum),
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
    rng_scheme: NonEmptyString | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
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
        expected_constraints = self.trajectories[0].constraints_encountered
        expected_activations = tuple(
            item.model_dump(mode="json")
            for item in self.trajectories[0].constraint_activations
        )
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
            if trajectory.rng_scheme != self.rng_scheme:
                raise ValueError(
                    "trajectory ensemble member RNG scheme must match ensemble"
                )
            if (
                utc_instant_key(trajectory.initial_state.at)
                != utc_instant_key(expected_initial_state.at)
                or trajectory.initial_state.observed != expected_initial_state.observed
                or trajectory.initial_state.latent != expected_initial_state.latent
                or trajectory.initial_state.uncertainty
                != expected_initial_state.uncertainty
                or trajectory.initial_state.boundary != expected_initial_state.boundary
            ):
                raise ValueError("trajectory ensemble members must share initial state")
            if (
                utc_instant_key(trajectory.horizon.start_at)
                != utc_instant_key(expected_horizon.start_at)
                or utc_instant_key(trajectory.horizon.end_at)
                != utc_instant_key(expected_horizon.end_at)
            ):
                raise ValueError(
                    "trajectory ensemble members must share requested horizon"
                )
            if trajectory.constraints_encountered != expected_constraints:
                raise ValueError(
                    "trajectory ensemble members must share encountered constraints"
                )
            actual_activations = tuple(
                item.model_dump(mode="json")
                for item in trajectory.constraint_activations
            )
            if actual_activations != expected_activations:
                raise ValueError(
                    "trajectory ensemble members must share constraint activations"
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
            if self.summary.model_dump(
                mode="json"
            ) != expected_summary.model_dump(mode="json"):
                raise ValueError(
                    "trajectory ensemble summary must match aligned raw samples"
                )
        return self
