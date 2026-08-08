"""Descriptive consecutive-step trajectory geometry with explicit scales."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import math
import sys
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from pi_engine.schemas.common import FiniteFloat
from pi_engine.schemas.trajectory import Trajectory


NonEmptyString = Annotated[str, Field(min_length=1)]
ComponentName = NonEmptyString | None


class ConvergenceAnalysisError(ValueError):
    """A trajectory cannot be described without changing its geometry."""


class _ImmutableAnalysisSchema(BaseModel):
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


def _utc(value: datetime, error_type: type[ValueError]) -> datetime:
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as exc:
        raise error_type("analysis time cannot be normalized to UTC") from exc


def _strict_scalar(value: object, role: str, error_type: type[ValueError]) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error_type(f"{role} must be a scalar numeric value, not bool")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise error_type(f"{role} must be representable as a finite float") from exc
    if not math.isfinite(result):
        raise error_type(f"{role} must be finite")
    return result


class RepresentedMagnitude(_ImmutableAnalysisSchema):
    """A nonnegative magnitude or a structural float-range overflow."""

    kind: Literal[
        "finite", "above_float_range", "below_float_resolution"
    ]
    value: FiniteFloat | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_representation(self) -> RepresentedMagnitude:
        if self.kind == "finite" and self.value is None:
            raise ValueError("finite magnitude requires a value")
        if self.kind != "finite" and self.value is not None:
            raise ValueError("structural magnitude cannot carry a finite value")
        return self


class VariableNormalization(_ImmutableAnalysisSchema):
    """Caller-declared scale and optional vector-component semantics."""

    variable: NonEmptyString
    scale: FiniteFloat = Field(gt=0.0)
    components: tuple[NonEmptyString, ...]

    @model_validator(mode="after")
    def components_must_be_unique(self) -> VariableNormalization:
        if len(set(self.components)) != len(self.components):
            raise ValueError("component semantics must be unique")
        return self


class VariableStepDistance(_ImmutableAnalysisSchema):
    """One scalar/component distance across one ordered adjacent pair."""

    variable: NonEmptyString
    component: ComponentName
    normalization_scale: FiniteFloat = Field(gt=0.0)
    from_value: FiniteFloat
    to_value: FiniteFloat
    absolute_distance: RepresentedMagnitude
    normalized_distance: FiniteFloat = Field(ge=0.0)

    @field_validator("absolute_distance", mode="before")
    @classmethod
    def revalidate_absolute_distance(cls, value: object) -> object:
        if isinstance(value, RepresentedMagnitude):
            return RepresentedMagnitude.model_validate(
                value.model_dump(warnings=False)
            )
        return value

    @model_validator(mode="after")
    def validate_arithmetic(self) -> VariableStepDistance:
        expected_absolute = _absolute_difference(
            self.from_value, self.to_value
        )
        expected_normalized = _normalized_difference(
            self.from_value,
            self.to_value,
            self.normalization_scale,
            ConvergenceAnalysisError,
        )
        if (
            self.absolute_distance != expected_absolute
            or self.normalized_distance != expected_normalized
        ):
            raise ValueError("step-distance arithmetic is inconsistent")
        return self


class OrderedStepPair(_ImmutableAnalysisSchema):
    """All component distances for one adjacent pair of trajectory states."""

    pair_index: StrictInt = Field(ge=0)
    from_at: datetime
    to_at: datetime
    distances: Annotated[tuple[VariableStepDistance, ...], Field(min_length=1)]

    @field_validator("from_at", "to_at")
    @classmethod
    def times_must_be_aware(cls, value: datetime, info: object) -> datetime:
        return _require_timezone(value, getattr(info, "field_name", "pair time"))

    @field_validator("distances", mode="before")
    @classmethod
    def revalidate_distances(cls, value: object) -> object:
        if isinstance(value, (tuple, list)):
            return tuple(
                VariableStepDistance.model_validate(item.model_dump(warnings=False))
                if isinstance(item, VariableStepDistance)
                else item
                for item in value
            )
        return value

    @model_validator(mode="after")
    def validate_pair(self) -> OrderedStepPair:
        if _utc(self.to_at, ValueError) <= _utc(self.from_at, ValueError):
            raise ValueError("ordered step pair must advance in absolute time")
        keys = tuple((item.variable, item.component) for item in self.distances)
        if len(set(keys)) != len(keys):
            raise ValueError("step-pair component identities must be unique")
        return self


ObservedStepPattern = Literal[
    "insufficient_step_pairs",
    "strictly_contracting_step_distance",
    "strictly_expanding_step_distance",
    "constant_step_distance",
    "mixed_step_distance",
]


class ComponentConvergencePattern(_ImmutableAnalysisSchema):
    """Mechanical label for one ordered normalized-distance sequence."""

    variable: NonEmptyString
    component: ComponentName
    pair_count: StrictInt = Field(ge=1)
    normalized_distances: Annotated[tuple[FiniteFloat, ...], Field(min_length=1)]
    observed_pattern: ObservedStepPattern

    @model_validator(mode="after")
    def validate_pattern(self) -> ComponentConvergencePattern:
        if self.pair_count != len(self.normalized_distances):
            raise ValueError("pattern pair count must match retained distances")
        if any(item < 0.0 for item in self.normalized_distances):
            raise ValueError("normalized distances must be nonnegative")
        if self.observed_pattern != _step_pattern(self.normalized_distances):
            raise ValueError("observed step-distance pattern is inconsistent")
        return self


class ConvergenceAnalysis(_ImmutableAnalysisSchema):
    """Source-bound adjacent-step evidence without a causal or attractor claim."""

    source: Trajectory
    trajectory_kind: Literal["deterministic", "stochastic_sample"]
    normalization: Annotated[
        tuple[VariableNormalization, ...], Field(min_length=1)
    ]
    pairs: Annotated[tuple[OrderedStepPair, ...], Field(min_length=1)]
    patterns: Annotated[
        tuple[ComponentConvergencePattern, ...], Field(min_length=1)
    ]

    @field_validator("source", mode="before")
    @classmethod
    def revalidate_source(cls, value: object) -> object:
        return _revalidate_trajectory(value, ConvergenceAnalysisError)

    @field_validator("normalization", mode="before")
    @classmethod
    def revalidate_normalization(cls, value: object) -> object:
        return _revalidate_records(value, VariableNormalization)

    @field_validator("pairs", mode="before")
    @classmethod
    def revalidate_pairs(cls, value: object) -> object:
        return _revalidate_records(value, OrderedStepPair)

    @field_validator("patterns", mode="before")
    @classmethod
    def revalidate_patterns(cls, value: object) -> object:
        return _revalidate_records(value, ComponentConvergencePattern)

    @model_validator(mode="after")
    def validate_recomputation(self) -> ConvergenceAnalysis:
        expected = _derive_convergence(self.source, self.normalization)
        if (
            self.trajectory_kind != expected[0]
            or self.pairs != expected[1]
            or self.patterns != expected[2]
        ):
            raise ValueError(
                "convergence evidence must exactly recompute from its source"
            )
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


def _revalidate_trajectory(
    value: object, error_type: type[ValueError]
) -> Trajectory:
    try:
        if isinstance(value, Trajectory):
            payload = value.model_dump(warnings=False)
        elif isinstance(value, dict):
            payload = value
        else:
            raise TypeError("source must be a Trajectory")
        result = Trajectory.model_validate(payload)
    except (TypeError, ValueError, ValidationError) as exc:
        raise error_type("source must be a valid Trajectory") from exc
    _validated_timeline(result, error_type)
    return result


def _validated_timeline(
    trajectory: Trajectory, error_type: type[ValueError]
) -> None:
    start = _utc(trajectory.horizon.start_at, error_type)
    end = _utc(trajectory.horizon.end_at, error_type)
    if _utc(trajectory.initial_state.at, error_type) != start:
        raise error_type("initial state must align exactly with horizon start")
    times = tuple(_utc(point.at, error_type) for point in trajectory.points)
    if any(later <= earlier for earlier, later in zip((start, *times), times)):
        raise error_type("trajectory times must be strictly ordered UTC instants")
    if times[-1] != end:
        raise error_type("trajectory must align exactly with the requested horizon")


def _state_values(
    trajectory: Trajectory, error_type: type[ValueError]
) -> dict[str, object]:
    values: dict[str, object] = {}
    for component in (
        trajectory.initial_state.observed,
        trajectory.initial_state.latent,
        trajectory.initial_state.boundary,
    ):
        overlap = set(values) & set(component)
        if overlap:
            raise error_type(
                "initial-state variables must be unique across state components"
            )
        values.update(component)
    return values


def _sequence_names(value: object, role: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConvergenceAnalysisError(f"{role} must be a sequence of names")
    names = tuple(value)
    if any(not isinstance(item, str) or not item for item in names):
        raise ConvergenceAnalysisError(f"{role} must contain nonempty strings")
    if len(set(names)) != len(names):
        raise ConvergenceAnalysisError(f"{role} must be unique")
    return names


def _normalization_from_inputs(
    trajectory: Trajectory,
    normalization_scales: Mapping[str, object],
    vector_components: Mapping[str, Sequence[str]],
) -> tuple[VariableNormalization, ...]:
    if not isinstance(normalization_scales, Mapping):
        raise TypeError("normalization_scales must be a mapping")
    if not isinstance(vector_components, Mapping):
        raise TypeError("vector_components must be a mapping")
    variables = tuple(trajectory.points[0].values)
    if set(normalization_scales) != set(variables):
        raise ConvergenceAnalysisError(
            "normalization scale keys must exactly match trajectory variables"
        )
    initial = _state_values(trajectory, ConvergenceAnalysisError)
    specs: list[VariableNormalization] = []
    vector_names: set[str] = set()
    for variable in variables:
        if variable not in initial:
            raise ConvergenceAnalysisError(
                f"initial state lacks trajectory variable {variable}"
            )
        scale = normalization_scales[variable]
        if (
            isinstance(scale, bool)
            or not isinstance(scale, float)
            or not math.isfinite(scale)
            or scale <= 0.0
        ):
            raise ConvergenceAnalysisError(
                "normalization scales must be explicit positive finite floats"
            )
        first = trajectory.points[0].values[variable]
        if isinstance(first, tuple):
            vector_names.add(variable)
            if variable not in vector_components:
                raise ConvergenceAnalysisError(
                    f"vector variable {variable} requires explicit component semantics"
                )
            components = _sequence_names(
                vector_components[variable], f"component semantics for {variable}"
            )
            if len(components) != len(first):
                raise ConvergenceAnalysisError(
                    f"component semantics for {variable} must match vector shape"
                )
        else:
            if variable in vector_components:
                raise ConvergenceAnalysisError(
                    f"scalar variable {variable} cannot declare vector component semantics"
                )
            components = ()
        specs.append(
            VariableNormalization(
                variable=variable, scale=scale, components=components
            )
        )
    if set(vector_components) != vector_names:
        raise ConvergenceAnalysisError(
            "vector component semantics must exactly match vector variables"
        )
    _aligned_rows(trajectory, tuple(specs), ConvergenceAnalysisError)
    return tuple(specs)


def _aligned_rows(
    trajectory: Trajectory,
    normalization: tuple[VariableNormalization, ...],
    error_type: type[ValueError],
) -> tuple[tuple[datetime, dict[tuple[str, str | None], float]], ...]:
    variables = tuple(spec.variable for spec in normalization)
    if len(set(variables)) != len(variables):
        raise error_type("normalization variable identities must be unique")
    expected = set(variables)
    initial = _state_values(trajectory, error_type)
    point_rows: list[tuple[datetime, Mapping[str, object]]] = [
        (trajectory.horizon.start_at, initial)
    ]
    point_rows.extend((point.at, point.values) for point in trajectory.points)
    rows: list[tuple[datetime, dict[tuple[str, str | None], float]]] = []
    for index, (at, values) in enumerate(point_rows):
        if index and set(values) != expected:
            raise error_type("trajectory points must share exact variable alignment")
        flattened: dict[tuple[str, str | None], float] = {}
        for spec in normalization:
            if spec.variable not in values:
                raise error_type(
                    f"initial state lacks trajectory variable {spec.variable}"
                )
            raw = values[spec.variable]
            if spec.components:
                if not isinstance(raw, tuple) or len(raw) != len(spec.components):
                    raise error_type(
                        f"trajectory shape must remain aligned for {spec.variable}"
                    )
                for component, item in zip(spec.components, raw, strict=True):
                    flattened[(spec.variable, component)] = _strict_scalar(
                        item, f"value for {spec.variable}.{component}", error_type
                    )
            else:
                if isinstance(raw, tuple):
                    raise error_type(
                        f"vector variable {spec.variable} requires component semantics"
                    )
                flattened[(spec.variable, None)] = _strict_scalar(
                    raw, f"value for {spec.variable}", error_type
                )
        rows.append((at, flattened))
    return tuple(rows)


def _absolute_difference(left: float, right: float) -> RepresentedMagnitude:
    scale = max(abs(left), abs(right))
    if scale == 0.0:
        return RepresentedMagnitude(kind="finite", value=0.0)
    factor = abs((right / scale) - (left / scale))
    if factor != 0.0 and scale > sys.float_info.max / factor:
        return RepresentedMagnitude(kind="above_float_range")
    result = scale * factor
    if left != right and result == 0.0:
        return RepresentedMagnitude(kind="below_float_resolution")
    return RepresentedMagnitude(kind="finite", value=float(result))


def _normalized_difference(
    left: float,
    right: float,
    normalization_scale: float,
    error_type: type[ValueError],
) -> float:
    computation_scale = max(abs(left), abs(right))
    if computation_scale == 0.0:
        return 0.0
    factor = abs(
        (right / computation_scale) - (left / computation_scale)
    )
    ratio = computation_scale / normalization_scale
    if factor != 0.0 and ratio > sys.float_info.max / factor:
        raise error_type("normalized distance is not representable as finite")
    result = ratio * factor
    if not math.isfinite(result):
        raise error_type("normalized distance is not representable as finite")
    if left != right and result == 0.0:
        raise error_type("normalized distance is not representable as finite")
    return float(result)


def _step_pattern(distances: tuple[float, ...]) -> ObservedStepPattern:
    if len(distances) < 2:
        return "insufficient_step_pairs"
    comparisons = tuple(
        (later > earlier) - (later < earlier)
        for earlier, later in zip(distances, distances[1:])
    )
    if all(item < 0 for item in comparisons):
        return "strictly_contracting_step_distance"
    if all(item > 0 for item in comparisons):
        return "strictly_expanding_step_distance"
    if all(item == 0 for item in comparisons):
        return "constant_step_distance"
    return "mixed_step_distance"


def _trajectory_kind(trajectory: Trajectory) -> Literal[
    "deterministic", "stochastic_sample"
]:
    if trajectory.sample_seed is None and trajectory.rng_scheme is None:
        return "deterministic"
    if trajectory.sample_seed is not None and trajectory.rng_scheme is not None:
        return "stochastic_sample"
    raise ConvergenceAnalysisError(
        "trajectory stochastic identity must contain both sample seed and RNG scheme"
    )


def _derive_convergence(
    trajectory: Trajectory,
    normalization: tuple[VariableNormalization, ...],
) -> tuple[
    Literal["deterministic", "stochastic_sample"],
    tuple[OrderedStepPair, ...],
    tuple[ComponentConvergencePattern, ...],
]:
    rows = _aligned_rows(trajectory, normalization, ConvergenceAnalysisError)
    spec_by_variable = {item.variable: item for item in normalization}
    pairs: list[OrderedStepPair] = []
    by_component: dict[tuple[str, str | None], list[float]] = {
        key: [] for key in rows[0][1]
    }
    for index, ((from_at, left), (to_at, right)) in enumerate(
        zip(rows, rows[1:])
    ):
        distances: list[VariableStepDistance] = []
        for variable, component in left:
            scale = spec_by_variable[variable].scale
            normalized = _normalized_difference(
                left[(variable, component)],
                right[(variable, component)],
                scale,
                ConvergenceAnalysisError,
            )
            by_component[(variable, component)].append(normalized)
            distances.append(
                VariableStepDistance(
                    variable=variable,
                    component=component,
                    normalization_scale=scale,
                    from_value=left[(variable, component)],
                    to_value=right[(variable, component)],
                    absolute_distance=_absolute_difference(
                        left[(variable, component)], right[(variable, component)]
                    ),
                    normalized_distance=normalized,
                )
            )
        pairs.append(
            OrderedStepPair(
                pair_index=index,
                from_at=from_at,
                to_at=to_at,
                distances=tuple(distances),
            )
        )
    patterns = tuple(
        ComponentConvergencePattern(
            variable=variable,
            component=component,
            pair_count=len(distances),
            normalized_distances=tuple(distances),
            observed_pattern=_step_pattern(tuple(distances)),
        )
        for (variable, component), distances in by_component.items()
    )
    return _trajectory_kind(trajectory), tuple(pairs), patterns


def analyze_trajectory_convergence(
    trajectory: Trajectory,
    *,
    normalization_scales: Mapping[str, object],
    vector_components: Mapping[str, Sequence[str]] | None = None,
) -> ConvergenceAnalysis:
    """Describe adjacent-step distances without asserting an attractor or cause."""
    validated = _revalidate_trajectory(trajectory, ConvergenceAnalysisError)
    normalization = _normalization_from_inputs(
        validated,
        normalization_scales,
        vector_components or {},
    )
    kind, pairs, patterns = _derive_convergence(validated, normalization)
    return ConvergenceAnalysis(
        source=validated,
        trajectory_kind=kind,
        normalization=normalization,
        pairs=pairs,
        patterns=patterns,
    )


__all__ = [
    "ComponentConvergencePattern",
    "ConvergenceAnalysis",
    "ConvergenceAnalysisError",
    "OrderedStepPair",
    "RepresentedMagnitude",
    "VariableNormalization",
    "VariableStepDistance",
    "analyze_trajectory_convergence",
]
