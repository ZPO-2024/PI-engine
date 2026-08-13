"""Time-indexed ensemble spread evidence with explicit singleton semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import math
import sys
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from pi_engine._numeric import (
    normalize_nonnegative_weights,
    normalized_absolute_difference,
    stable_population_mean_std,
)
from pi_engine.analysis._shared import (
    canonical_model_content,
    canonical_record_content,
)
from pi_engine.analysis.convergence import (
    ConvergenceAnalysisError,
    RepresentedMagnitude,
    VariableNormalization,
    _absolute_difference,
    _aligned_rows,
    _normalization_from_inputs,
    _revalidate_trajectory,
    _require_timezone,
    _utc,
)
from pi_engine.schemas.common import FiniteFloat
from pi_engine.schemas.trajectory import Trajectory, TrajectoryEnsemble


NonEmptyString = Annotated[str, Field(min_length=1)]
ComponentName = NonEmptyString | None
SpreadSource = Trajectory | TrajectoryEnsemble
SpreadWeightingPolicy = Literal[
    "single_member",
    "equal_member",
    "probability",
    "normalized_relative_weight",
]


class SpreadAnalysisError(ValueError):
    """Aligned raw members cannot produce the requested finite spread evidence."""


class _ImmutableSpreadSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class SpreadWeighting(_ImmutableSpreadSchema):
    """Explicit normalized member weights in retained source order."""

    policy: SpreadWeightingPolicy
    weights: Annotated[tuple[FiniteFloat, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_weights(self) -> SpreadWeighting:
        if any(item < 0.0 for item in self.weights) or not any(
            item > 0.0 for item in self.weights
        ):
            raise ValueError("spread weights must be nonnegative with positive support")
        return self


class SpreadPoint(_ImmutableSpreadSchema):
    """Stable population evidence for one component at one UTC-aligned time."""

    at: datetime
    variable: NonEmptyString
    component: ComponentName
    normalization_scale: FiniteFloat = Field(gt=0.0)
    count: StrictInt = Field(ge=1)
    values: Annotated[tuple[FiniteFloat, ...], Field(min_length=1)]
    weighting_policy: SpreadWeightingPolicy
    weights: Annotated[tuple[FiniteFloat, ...], Field(min_length=1)]
    computation_scale: FiniteFloat = Field(ge=0.0)
    mean: FiniteFloat
    minimum: FiniteFloat
    maximum: FiniteFloat
    spread_range: RepresentedMagnitude
    population_std: FiniteFloat = Field(ge=0.0)
    population_variance: RepresentedMagnitude
    normalized_population_std: FiniteFloat = Field(ge=0.0)
    normalized_range: FiniteFloat = Field(ge=0.0)

    @field_validator("at")
    @classmethod
    def time_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value, "spread point time")

    @field_validator("spread_range", "population_variance", mode="before")
    @classmethod
    def revalidate_magnitude(cls, value: object) -> object:
        if isinstance(value, RepresentedMagnitude):
            return RepresentedMagnitude.model_validate(
                value.model_dump(warnings=False)
            )
        return value

    @model_validator(mode="after")
    def validate_statistics(self) -> SpreadPoint:
        if self.count != len(self.values):
            raise ValueError("spread count must match retained raw values")
        if len(self.weights) != self.count:
            raise ValueError("spread weights must align with retained raw values")
        expected = _spread_statistics(
            self.values, self.weights, self.normalization_scale
        )
        actual = (
            self.computation_scale,
            self.mean,
            self.minimum,
            self.maximum,
            self.spread_range,
            self.population_std,
            self.population_variance,
            self.normalized_population_std,
            self.normalized_range,
        )
        if actual != expected:
            raise ValueError("spread statistics are inconsistent with raw values")
        return self


ObservedSpreadPattern = Literal[
    "deterministic_singleton_no_spread",
    "stochastic_sample_singleton_no_spread",
    "ensemble_singleton_no_spread",
    "ensemble_no_spread",
    "strictly_expanding_spread",
    "strictly_contracting_spread",
    "constant_positive_spread",
    "insufficient_spread_points",
    "mixed_spread",
]


class SpreadClassificationWindow(_ImmutableSpreadSchema):
    """Exact retained time window used for the mechanical spread label."""

    basis: Literal[
        "forecast_after_shared_horizon_start",
        "all_retained_points",
    ]
    initial_evidence_source: Literal[
        "injected_initial_state",
        "explicit_horizon_start",
        "none",
    ]
    excluded_horizon_start: StrictBool
    start_point_index: StrictInt = Field(ge=0)
    classified_point_count: StrictInt = Field(ge=1)
    start_at: datetime
    end_at: datetime

    @field_validator("start_at", "end_at")
    @classmethod
    def times_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value, "spread classification time")

    @model_validator(mode="after")
    def validate_policy(self) -> SpreadClassificationWindow:
        if self.excluded_horizon_start != (self.start_point_index == 1):
            raise ValueError("spread classification exclusion metadata is inconsistent")
        if self.basis == "forecast_after_shared_horizon_start":
            if not self.excluded_horizon_start:
                raise ValueError(
                    "forecast spread classification must exclude horizon start"
                )
            if self.initial_evidence_source == "none":
                raise ValueError(
                    "forecast spread classification requires initial evidence"
                )
        elif self.excluded_horizon_start or self.initial_evidence_source != "none":
            raise ValueError(
                "all-point spread classification cannot exclude initial evidence"
            )
        if _utc(self.end_at, ValueError) < _utc(self.start_at, ValueError):
            raise ValueError("spread classification window cannot run backward")
        return self


class ComponentSpreadPattern(_ImmutableSpreadSchema):
    """Mechanical label for one ordered normalized standard-deviation sequence."""

    variable: NonEmptyString
    component: ComponentName
    point_count: StrictInt = Field(ge=2)
    normalized_population_std: Annotated[
        tuple[FiniteFloat, ...], Field(min_length=2)
    ]
    classification_start_index: StrictInt = Field(ge=0)
    classified_point_count: StrictInt = Field(ge=1)
    classified_normalized_population_std: Annotated[
        tuple[FiniteFloat, ...], Field(min_length=1)
    ]
    observed_pattern: ObservedSpreadPattern

    @model_validator(mode="after")
    def validate_count(self) -> ComponentSpreadPattern:
        if self.point_count != len(self.normalized_population_std):
            raise ValueError("spread pattern count must match retained evidence")
        if self.classification_start_index >= self.point_count:
            raise ValueError("spread pattern classification start is out of bounds")
        expected = self.normalized_population_std[self.classification_start_index :]
        if self.classified_point_count != len(expected):
            raise ValueError("spread classified count must match retained evidence")
        if self.classified_normalized_population_std != expected:
            raise ValueError(
                "spread classified evidence must be an exact retained suffix"
            )
        return self


class SpreadAnalysis(_ImmutableSpreadSchema):
    """Raw source-bound spread; deterministic path growth is kept separate."""

    source: SpreadSource
    source_kind: Literal[
        "deterministic_singleton",
        "stochastic_sample_singleton",
        "ensemble_singleton",
        "ensemble",
    ]
    member_count: StrictInt = Field(ge=1)
    weighting: SpreadWeighting
    classification_window: SpreadClassificationWindow
    normalization: Annotated[
        tuple[VariableNormalization, ...], Field(min_length=1)
    ]
    points: Annotated[tuple[SpreadPoint, ...], Field(min_length=2)]
    patterns: Annotated[tuple[ComponentSpreadPattern, ...], Field(min_length=1)]

    @field_validator("source", mode="before")
    @classmethod
    def revalidate_source(cls, value: object) -> object:
        return _revalidate_spread_source(value, allow_serialized=True)

    @field_validator("normalization", mode="before")
    @classmethod
    def revalidate_normalization(cls, value: object) -> object:
        return _revalidate_records(value, VariableNormalization)

    @field_validator("weighting", mode="before")
    @classmethod
    def revalidate_weighting(cls, value: object) -> object:
        if isinstance(value, SpreadWeighting):
            return SpreadWeighting.model_validate(
                value.model_dump(warnings=False)
            )
        return value

    @field_validator("classification_window", mode="before")
    @classmethod
    def revalidate_classification_window(cls, value: object) -> object:
        if isinstance(value, SpreadClassificationWindow):
            return SpreadClassificationWindow.model_validate(
                value.model_dump(warnings=False)
            )
        return value

    @field_validator("points", mode="before")
    @classmethod
    def revalidate_points(cls, value: object) -> object:
        return _revalidate_records(value, SpreadPoint)

    @field_validator("patterns", mode="before")
    @classmethod
    def revalidate_patterns(cls, value: object) -> object:
        return _revalidate_records(value, ComponentSpreadPattern)

    @model_validator(mode="after")
    def validate_recomputation(self) -> SpreadAnalysis:
        expected = _derive_spread(self.source, self.normalization)
        if (
            self.source_kind != expected[0]
            or self.member_count != expected[1]
            or canonical_model_content(self.weighting)
            != canonical_model_content(expected[2])
            or canonical_model_content(self.classification_window)
            != canonical_model_content(expected[3])
            or canonical_record_content(self.points)
            != canonical_record_content(expected[4])
            or canonical_record_content(self.patterns)
            != canonical_record_content(expected[5])
        ):
            raise ValueError("spread evidence must exactly recompute from its source")
        return self


def _revalidate_records(value: object, model: type[BaseModel]) -> object:
    if not isinstance(value, (tuple, list)):
        return value
    return tuple(
        model.model_validate(item.model_dump(warnings=False))
        if isinstance(item, model)
        else item
        for item in value
    )


def _revalidate_spread_source(
    value: object, *, allow_serialized: bool
) -> SpreadSource:
    if isinstance(value, TrajectoryEnsemble):
        try:
            ensemble = TrajectoryEnsemble.model_validate(
                value.model_dump(warnings=False)
            )
        except (OverflowError, TypeError, ValueError, ValidationError) as exc:
            raise SpreadAnalysisError(
                "source must be a valid TrajectoryEnsemble"
            ) from exc
        for member in ensemble.trajectories:
            _revalidate_trajectory(member, SpreadAnalysisError)
        return ensemble
    if isinstance(value, Trajectory):
        return _revalidate_trajectory(value, SpreadAnalysisError)
    if allow_serialized and isinstance(value, dict):
        has_ensemble = "ensemble_id" in value
        has_trajectory = "trajectory_id" in value
        if has_ensemble == has_trajectory:
            raise SpreadAnalysisError("spread source type cannot be determined")
        try:
            if has_ensemble:
                ensemble = TrajectoryEnsemble.model_validate(value)
                for member in ensemble.trajectories:
                    _revalidate_trajectory(member, SpreadAnalysisError)
                return ensemble
            return _revalidate_trajectory(
                Trajectory.model_validate(value), SpreadAnalysisError
            )
        except (OverflowError, TypeError, ValueError, ValidationError) as exc:
            raise SpreadAnalysisError("spread source is not valid") from exc
    raise TypeError("source must be a Trajectory or TrajectoryEnsemble")


def _members(source: SpreadSource) -> tuple[Trajectory, ...]:
    return source.trajectories if isinstance(source, TrajectoryEnsemble) else (source,)


def _source_kind(source: SpreadSource) -> Literal[
    "deterministic_singleton",
    "stochastic_sample_singleton",
    "ensemble_singleton",
    "ensemble",
]:
    if isinstance(source, Trajectory):
        if source.sample_seed is None and source.rng_scheme is None:
            return "deterministic_singleton"
        if source.sample_seed is not None and source.rng_scheme is not None:
            return "stochastic_sample_singleton"
        raise SpreadAnalysisError(
            "stochastic sample identity requires sample seed and RNG scheme"
        )
    if len(source.trajectories) == 1:
        return "ensemble_singleton"
    return "ensemble"


def _magnitude_square(value: float) -> RepresentedMagnitude:
    if value == 0.0:
        return RepresentedMagnitude(kind="finite", value=0.0)
    if value > math.sqrt(sys.float_info.max):
        return RepresentedMagnitude(kind="above_float_range")
    result = value * value
    if result == 0.0:
        return RepresentedMagnitude(kind="below_float_resolution")
    return RepresentedMagnitude(kind="finite", value=float(result))


def _finite_ratio(numerator: float, denominator: float, role: str) -> float:
    if numerator == 0.0:
        return 0.0
    if numerator > sys.float_info.max * denominator:
        raise SpreadAnalysisError(f"normalized {role} is not representable as finite")
    result = numerator / denominator
    if not math.isfinite(result):
        raise SpreadAnalysisError(f"normalized {role} is not representable as finite")
    if result == 0.0:
        raise SpreadAnalysisError(f"normalized {role} is not representable as finite")
    return float(result)


def _spread_statistics(
    values: tuple[float, ...],
    weights: tuple[float, ...],
    normalization_scale: float,
) -> tuple[
    float,
    float,
    float,
    float,
    RepresentedMagnitude,
    float,
    RepresentedMagnitude,
    float,
    float,
]:
    mean, std, minimum, maximum = stable_population_mean_std(
        values, weights, error_type=SpreadAnalysisError
    )
    computation_scale = max(abs(minimum), abs(maximum))
    spread_range = _absolute_difference(minimum, maximum)
    normalized_std = _finite_ratio(
        float(std), normalization_scale, "population standard deviation"
    )
    normalized_range = _finite_ratio(
        _finite_range_factor(minimum, maximum, normalization_scale),
        1.0,
        "range",
    )
    return (
        float(computation_scale),
        float(mean),
        float(minimum),
        float(maximum),
        spread_range,
        float(std),
        _magnitude_square(float(std)),
        normalized_std,
        normalized_range,
    )


def _finite_range_factor(
    minimum: float, maximum: float, normalization_scale: float
) -> float:
    return normalized_absolute_difference(
        minimum,
        maximum,
        normalization_scale,
        role="normalized range",
        error_type=SpreadAnalysisError,
    )


def _spread_pattern(
    values: tuple[float, ...], source_kind: str
) -> ObservedSpreadPattern:
    if source_kind == "deterministic_singleton":
        return "deterministic_singleton_no_spread"
    if source_kind == "stochastic_sample_singleton":
        return "stochastic_sample_singleton_no_spread"
    if source_kind == "ensemble_singleton":
        return "ensemble_singleton_no_spread"
    if all(item == 0.0 for item in values):
        return "ensemble_no_spread"
    if len(values) < 2:
        return "insufficient_spread_points"
    comparisons = tuple(
        (later > earlier) - (later < earlier)
        for earlier, later in zip(values, values[1:])
    )
    if all(item > 0 for item in comparisons):
        return "strictly_expanding_spread"
    if all(item < 0 for item in comparisons):
        return "strictly_contracting_spread"
    if all(item == 0 for item in comparisons):
        return "constant_positive_spread"
    return "mixed_spread"


def _aligned_member_rows(
    source: SpreadSource,
    normalization: tuple[VariableNormalization, ...],
) -> tuple[tuple[tuple[datetime, dict[tuple[str, str | None], float]], ...], ...]:
    rows_by_member = tuple(
        _aligned_rows(member, normalization, SpreadAnalysisError)
        for member in _members(source)
    )
    reference = rows_by_member[0]
    for rows in rows_by_member[1:]:
        if len(rows) != len(reference):
            raise SpreadAnalysisError("ensemble members must share exact horizon shape")
        for (expected_at, expected_values), (at, values) in zip(
            reference, rows, strict=True
        ):
            if _utc(at, SpreadAnalysisError) != _utc(
                expected_at, SpreadAnalysisError
            ):
                raise SpreadAnalysisError(
                    "ensemble members must share exact UTC time alignment"
                )
            if tuple(values) != tuple(expected_values):
                raise SpreadAnalysisError(
                    "ensemble members must share exact variable and shape alignment"
                )
    return rows_by_member


def _spread_weighting(source: SpreadSource) -> SpreadWeighting:
    members = _members(source)
    if isinstance(source, Trajectory):
        return SpreadWeighting(policy="single_member", weights=(1.0,))
    declared = tuple(member.scenario_weight for member in members)
    if all(item is None for item in declared):
        weights = normalize_nonnegative_weights(
            tuple(1.0 for _ in members), error_type=SpreadAnalysisError
        )
        return SpreadWeighting(policy="equal_member", weights=weights)
    if any(item is None for item in declared):
        raise SpreadAnalysisError("ensemble weight scheme cannot be partial")
    retained = tuple(item for item in declared if item is not None)
    weights = normalize_nonnegative_weights(
        tuple(item.value for item in retained), error_type=SpreadAnalysisError
    )
    policy: SpreadWeightingPolicy = (
        "probability"
        if retained[0].kind == "probability"
        else "normalized_relative_weight"
    )
    return SpreadWeighting(policy=policy, weights=weights)


def _derive_spread(
    source: SpreadSource,
    normalization: tuple[VariableNormalization, ...],
) -> tuple[
    Literal[
        "deterministic_singleton",
        "stochastic_sample_singleton",
        "ensemble_singleton",
        "ensemble",
    ],
    int,
    SpreadWeighting,
    SpreadClassificationWindow,
    tuple[SpreadPoint, ...],
    tuple[ComponentSpreadPattern, ...],
]:
    rows_by_member = _aligned_member_rows(source, normalization)
    kind = _source_kind(source)
    weighting = _spread_weighting(source)
    scale_by_variable = {item.variable: item.scale for item in normalization}
    points: list[SpreadPoint] = []
    pattern_values: dict[tuple[str, str | None], list[float]] = {
        key: [] for key in rows_by_member[0][0][1]
    }
    for point_index, (at, reference_values) in enumerate(rows_by_member[0]):
        for variable, component in reference_values:
            values = tuple(
                rows[point_index][1][(variable, component)]
                for rows in rows_by_member
            )
            scale = scale_by_variable[variable]
            statistics = _spread_statistics(values, weighting.weights, scale)
            point = SpreadPoint(
                at=at,
                variable=variable,
                component=component,
                normalization_scale=scale,
                count=len(values),
                values=values,
                weighting_policy=weighting.policy,
                weights=weighting.weights,
                computation_scale=statistics[0],
                mean=statistics[1],
                minimum=statistics[2],
                maximum=statistics[3],
                spread_range=statistics[4],
                population_std=statistics[5],
                population_variance=statistics[6],
                normalized_population_std=statistics[7],
                normalized_range=statistics[8],
            )
            points.append(point)
            pattern_values[(variable, component)].append(
                point.normalized_population_std
            )
    first_member = _members(source)[0]
    first_source_at = first_member.points[0].at
    horizon_start = first_member.horizon.start_at
    has_retained_shared_initial = all(
        values[0] == 0.0 for values in pattern_values.values()
    )
    if has_retained_shared_initial and len(rows_by_member[0]) > 1:
        initial_evidence_source: Literal[
            "injected_initial_state", "explicit_horizon_start", "none"
        ] = (
            "explicit_horizon_start"
            if _utc(first_source_at, SpreadAnalysisError)
            == _utc(horizon_start, SpreadAnalysisError)
            else "injected_initial_state"
        )
        classification_start_index = 1
        classification_window = SpreadClassificationWindow(
            basis="forecast_after_shared_horizon_start",
            initial_evidence_source=initial_evidence_source,
            excluded_horizon_start=True,
            start_point_index=classification_start_index,
            classified_point_count=len(rows_by_member[0]) - 1,
            start_at=rows_by_member[0][classification_start_index][0],
            end_at=rows_by_member[0][-1][0],
        )
    else:
        classification_start_index = 0
        classification_window = SpreadClassificationWindow(
            basis="all_retained_points",
            initial_evidence_source="none",
            excluded_horizon_start=False,
            start_point_index=classification_start_index,
            classified_point_count=len(rows_by_member[0]),
            start_at=rows_by_member[0][0][0],
            end_at=rows_by_member[0][-1][0],
        )
    patterns = tuple(
        ComponentSpreadPattern(
            variable=variable,
            component=component,
            point_count=len(values),
            normalized_population_std=tuple(values),
            classification_start_index=classification_start_index,
            classified_point_count=len(values) - classification_start_index,
            classified_normalized_population_std=tuple(
                values[classification_start_index:]
            ),
            observed_pattern=_spread_pattern(
                tuple(values[classification_start_index:]), kind
            ),
        )
        for (variable, component), values in pattern_values.items()
    )
    return (
        kind,
        len(rows_by_member),
        weighting,
        classification_window,
        tuple(points),
        patterns,
    )


def analyze_trajectory_spread(
    source: SpreadSource,
    *,
    normalization_scales: Mapping[str, object],
    vector_components: Mapping[str, Sequence[str]] | None = None,
) -> SpreadAnalysis:
    """Describe population spread without equating it to path instability."""
    validated = _revalidate_spread_source(source, allow_serialized=False)
    first = _members(validated)[0]
    try:
        normalization = _normalization_from_inputs(
            first, normalization_scales, vector_components or {}
        )
    except ConvergenceAnalysisError as exc:
        raise SpreadAnalysisError(str(exc)) from exc
    kind, count, weighting, classification_window, points, patterns = _derive_spread(
        validated, normalization
    )
    return SpreadAnalysis(
        source=validated,
        source_kind=kind,
        member_count=count,
        weighting=weighting,
        classification_window=classification_window,
        normalization=normalization,
        points=points,
        patterns=patterns,
    )


__all__ = [
    "ComponentSpreadPattern",
    "SpreadAnalysis",
    "SpreadAnalysisError",
    "SpreadClassificationWindow",
    "SpreadPoint",
    "SpreadWeighting",
    "analyze_trajectory_spread",
]
