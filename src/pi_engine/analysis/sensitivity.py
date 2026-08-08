"""Explicit one-at-a-time local trajectory sensitivity evidence."""

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
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from pi_engine.analysis.convergence import (
    ConvergenceAnalysisError,
    VariableNormalization,
    _aligned_rows,
    _normalization_from_inputs,
    _revalidate_trajectory,
    _require_timezone,
    _utc,
)
from pi_engine.analysis._shared import finite_difference
from pi_engine.schemas.common import FiniteFloat
from pi_engine.schemas.trajectory import Trajectory


NonEmptyString = Annotated[str, Field(min_length=1)]
ComponentName = NonEmptyString | None


class LocalSensitivityError(ValueError):
    """Declared baseline and perturbation runs do not support a local slope."""


class _ImmutableSensitivitySchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class ParameterPerturbation(_ImmutableSensitivitySchema):
    """Caller assertion that exactly one named parameter changed for this run."""

    parameter_name: NonEmptyString
    delta: FiniteFloat
    trajectory: Trajectory

    @field_validator("delta")
    @classmethod
    def delta_must_be_nonzero(cls, value: float) -> float:
        if value == 0.0:
            raise ValueError("parameter delta must be finite and nonzero")
        return value

    @field_validator("trajectory", mode="before")
    @classmethod
    def revalidate_trajectory(cls, value: object) -> object:
        return _revalidate_trajectory(value, LocalSensitivityError)


class SensitivityPointSlope(_ImmutableSensitivitySchema):
    """One aligned finite-difference slope for one output component and time."""

    parameter_name: NonEmptyString
    delta: FiniteFloat
    at: datetime
    variable: NonEmptyString
    component: ComponentName
    normalization_scale: FiniteFloat = Field(gt=0.0)
    baseline_value: FiniteFloat
    perturbed_value: FiniteFloat
    raw_difference: FiniteFloat
    slope: FiniteFloat
    normalized_slope: FiniteFloat

    @field_validator("delta")
    @classmethod
    def delta_must_be_nonzero(cls, value: float) -> float:
        if value == 0.0:
            raise ValueError("parameter delta must be finite and nonzero")
        return value

    @field_validator("at")
    @classmethod
    def time_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value, "sensitivity point time")

    @model_validator(mode="after")
    def validate_arithmetic(self) -> SensitivityPointSlope:
        expected = _slope_values(
            self.baseline_value,
            self.perturbed_value,
            self.delta,
            self.normalization_scale,
        )
        if (
            self.raw_difference,
            self.slope,
            self.normalized_slope,
        ) != expected:
            raise ValueError("sensitivity slope arithmetic is inconsistent")
        return self


class SensitivitySummary(_ImmutableSensitivitySchema):
    """Transparent per-parameter/component absolute normalized-slope summary."""

    parameter_name: NonEmptyString
    variable: NonEmptyString
    component: ComponentName
    point_count: StrictInt = Field(ge=1)
    mean_absolute_normalized_slope: FiniteFloat = Field(ge=0.0)
    maximum_absolute_normalized_slope: FiniteFloat = Field(ge=0.0)
    final_absolute_normalized_slope: FiniteFloat = Field(ge=0.0)
    observed_pattern: Literal["no_observed_response", "observed_local_response"]


class LocalSensitivityAnalysis(_ImmutableSensitivitySchema):
    """Source-bound OAT evidence; parameter identity is caller-declared metadata."""

    baseline: Trajectory
    perturbations: Annotated[
        tuple[ParameterPerturbation, ...], Field(min_length=1)
    ]
    analysis_kind: Literal["deterministic_oat", "paired_stochastic_oat"]
    normalization: Annotated[
        tuple[VariableNormalization, ...], Field(min_length=1)
    ]
    point_slopes: Annotated[
        tuple[SensitivityPointSlope, ...], Field(min_length=1)
    ]
    summaries: Annotated[tuple[SensitivitySummary, ...], Field(min_length=1)]

    @field_validator("baseline", mode="before")
    @classmethod
    def revalidate_baseline(cls, value: object) -> object:
        return _revalidate_trajectory(value, LocalSensitivityError)

    @field_validator("perturbations", mode="before")
    @classmethod
    def revalidate_perturbations(cls, value: object) -> object:
        return _revalidate_records(value, ParameterPerturbation)

    @field_validator("normalization", mode="before")
    @classmethod
    def revalidate_normalization(cls, value: object) -> object:
        return _revalidate_records(value, VariableNormalization)

    @field_validator("point_slopes", mode="before")
    @classmethod
    def revalidate_point_slopes(cls, value: object) -> object:
        return _revalidate_records(value, SensitivityPointSlope)

    @field_validator("summaries", mode="before")
    @classmethod
    def revalidate_summaries(cls, value: object) -> object:
        return _revalidate_records(value, SensitivitySummary)

    @model_validator(mode="after")
    def validate_recomputation(self) -> LocalSensitivityAnalysis:
        expected = _derive_sensitivity(
            self.baseline, self.perturbations, self.normalization
        )
        if (
            self.analysis_kind != expected[0]
            or self.point_slopes != expected[1]
            or self.summaries != expected[2]
        ):
            raise ValueError(
                "local sensitivity evidence must exactly recompute from its sources"
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


def _finite_product(left: float, right: float, role: str) -> float:
    if left == 0.0 or right == 0.0:
        return 0.0
    if abs(left) > sys.float_info.max / abs(right):
        raise LocalSensitivityError(f"{role} is not representable as finite")
    result = left * right
    if not math.isfinite(result):
        raise LocalSensitivityError(f"{role} is not representable as finite")
    if result == 0.0:
        raise LocalSensitivityError(f"{role} is not representable as finite")
    return float(result)


def _finite_division(numerator: float, denominator: float, role: str) -> float:
    if numerator == 0.0:
        return 0.0
    if denominator == 0.0:
        raise LocalSensitivityError(f"{role} requires a nonzero denominator")
    if abs(numerator) > sys.float_info.max * abs(denominator):
        raise LocalSensitivityError(f"{role} is not representable as finite")
    result = numerator / denominator
    if not math.isfinite(result):
        raise LocalSensitivityError(f"{role} is not representable as finite")
    if result == 0.0:
        raise LocalSensitivityError(f"{role} is not representable as finite")
    return float(result)


def _signed_difference(left: float, right: float) -> float:
    return finite_difference(
        left,
        right,
        role="raw sensitivity difference",
        error_type=LocalSensitivityError,
    )


def _slope_values(
    baseline: float,
    perturbed: float,
    delta: float,
    normalization_scale: float,
) -> tuple[float, float, float]:
    difference = _signed_difference(baseline, perturbed)
    slope = _finite_division(difference, delta, "sensitivity slope")
    normalized_difference = _finite_division(
        difference, normalization_scale, "normalized sensitivity difference"
    )
    normalized_slope = _finite_division(
        normalized_difference, delta, "normalized sensitivity slope"
    )
    return difference, slope, normalized_slope


def _stable_mean(values: tuple[float, ...]) -> float:
    scale = max(abs(item) for item in values)
    if scale == 0.0:
        return 0.0
    result = scale * (math.fsum(item / scale for item in values) / len(values))
    if not math.isfinite(result):
        raise LocalSensitivityError(
            "mean absolute normalized sensitivity is not finite"
        )
    if result == 0.0 and any(item != 0.0 for item in values):
        raise LocalSensitivityError(
            "mean absolute normalized sensitivity is not representable as finite"
        )
    return float(result)


def _state_identity_equal(left: Trajectory, right: Trajectory) -> bool:
    return (
        _utc(left.initial_state.at, LocalSensitivityError)
        == _utc(right.initial_state.at, LocalSensitivityError)
        and left.initial_state.observed == right.initial_state.observed
        and left.initial_state.latent == right.initial_state.latent
        and left.initial_state.uncertainty == right.initial_state.uncertainty
        and left.initial_state.boundary == right.initial_state.boundary
    )


def _validate_perturbations(
    baseline: Trajectory,
    perturbations: tuple[ParameterPerturbation, ...],
    normalization: tuple[VariableNormalization, ...],
) -> Literal["deterministic_oat", "paired_stochastic_oat"]:
    names = tuple(item.parameter_name for item in perturbations)
    if len(set(names)) != len(names):
        raise LocalSensitivityError(
            "one-at-a-time perturbation parameter names must be unique"
        )
    ids = (baseline.trajectory_id,) + tuple(
        item.trajectory.trajectory_id for item in perturbations
    )
    if len(set(ids)) != len(ids):
        raise LocalSensitivityError(
            "baseline and perturbation trajectory identities must be unique"
        )
    baseline_rows = _aligned_rows(
        baseline, normalization, LocalSensitivityError
    )
    baseline_stochastic = baseline.sample_seed is not None or baseline.rng_scheme is not None
    if baseline_stochastic and (
        baseline.sample_seed is None or baseline.rng_scheme is None
    ):
        raise LocalSensitivityError(
            "baseline stochastic identity requires sample seed and RNG scheme"
        )
    for perturbation in perturbations:
        trajectory = perturbation.trajectory
        if (
            trajectory.model_id != baseline.model_id
            or trajectory.model_version != baseline.model_version
        ):
            raise LocalSensitivityError(
                "perturbation model identity must exactly match baseline"
            )
        if trajectory.case_id != baseline.case_id:
            raise LocalSensitivityError(
                "perturbation case identity must exactly match baseline"
            )
        if (
            _utc(trajectory.horizon.start_at, LocalSensitivityError)
            != _utc(baseline.horizon.start_at, LocalSensitivityError)
            or _utc(trajectory.horizon.end_at, LocalSensitivityError)
            != _utc(baseline.horizon.end_at, LocalSensitivityError)
        ):
            raise LocalSensitivityError(
                "perturbation horizon must exactly match baseline"
            )
        if not _state_identity_equal(baseline, trajectory):
            raise LocalSensitivityError(
                "perturbation initial state must exactly match baseline"
            )
        if trajectory.constraints_encountered != baseline.constraints_encountered:
            raise LocalSensitivityError(
                "perturbation constraints must exactly match baseline"
            )
        rows = _aligned_rows(trajectory, normalization, LocalSensitivityError)
        if len(rows) != len(baseline_rows):
            raise LocalSensitivityError(
                "perturbation point shape must exactly match baseline"
            )
        for (baseline_at, baseline_values), (at, values) in zip(
            baseline_rows, rows, strict=True
        ):
            if _utc(at, LocalSensitivityError) != _utc(
                baseline_at, LocalSensitivityError
            ):
                raise LocalSensitivityError(
                    "perturbation time must exactly match baseline UTC instants"
                )
            if tuple(values) != tuple(baseline_values):
                raise LocalSensitivityError(
                    "perturbation variables and shapes must exactly match baseline"
                )
        if baseline_stochastic:
            if (
                trajectory.sample_seed != baseline.sample_seed
                or trajectory.rng_scheme != baseline.rng_scheme
            ):
                raise LocalSensitivityError(
                    "stochastic sensitivity requires the same paired sample seed and RNG scheme"
                )
        elif trajectory.sample_seed is not None or trajectory.rng_scheme is not None:
            raise LocalSensitivityError(
                "deterministic baseline requires deterministic perturbation runs"
            )
    return "paired_stochastic_oat" if baseline_stochastic else "deterministic_oat"


def _derive_sensitivity(
    baseline: Trajectory,
    perturbations: tuple[ParameterPerturbation, ...],
    normalization: tuple[VariableNormalization, ...],
) -> tuple[
    Literal["deterministic_oat", "paired_stochastic_oat"],
    tuple[SensitivityPointSlope, ...],
    tuple[SensitivitySummary, ...],
]:
    kind = _validate_perturbations(baseline, perturbations, normalization)
    baseline_rows = _aligned_rows(
        baseline, normalization, LocalSensitivityError
    )[1:]
    scales = {item.variable: item.scale for item in normalization}
    points: list[SensitivityPointSlope] = []
    for perturbation in perturbations:
        perturbed_rows = _aligned_rows(
            perturbation.trajectory, normalization, LocalSensitivityError
        )[1:]
        for (at, baseline_values), (_, perturbed_values) in zip(
            baseline_rows, perturbed_rows, strict=True
        ):
            for variable, component in baseline_values:
                raw_difference, slope, normalized_slope = _slope_values(
                    baseline_values[(variable, component)],
                    perturbed_values[(variable, component)],
                    perturbation.delta,
                    scales[variable],
                )
                points.append(
                    SensitivityPointSlope(
                        parameter_name=perturbation.parameter_name,
                        delta=perturbation.delta,
                        at=at,
                        variable=variable,
                        component=component,
                        normalization_scale=scales[variable],
                        baseline_value=baseline_values[(variable, component)],
                        perturbed_value=perturbed_values[(variable, component)],
                        raw_difference=raw_difference,
                        slope=slope,
                        normalized_slope=normalized_slope,
                    )
                )
    grouped: dict[tuple[str, str, str | None], list[SensitivityPointSlope]] = {}
    for point in points:
        grouped.setdefault(
            (point.parameter_name, point.variable, point.component), []
        ).append(point)
    summaries: list[SensitivitySummary] = []
    for (parameter, variable, component), items in grouped.items():
        absolute = tuple(abs(item.normalized_slope) for item in items)
        summaries.append(
            SensitivitySummary(
                parameter_name=parameter,
                variable=variable,
                component=component,
                point_count=len(items),
                mean_absolute_normalized_slope=_stable_mean(absolute),
                maximum_absolute_normalized_slope=max(absolute),
                final_absolute_normalized_slope=absolute[-1],
                observed_pattern=(
                    "no_observed_response"
                    if all(item == 0.0 for item in absolute)
                    else "observed_local_response"
                ),
            )
        )
    return kind, tuple(points), tuple(summaries)


def analyze_local_sensitivity(
    baseline: Trajectory,
    perturbations: Sequence[ParameterPerturbation],
    *,
    normalization_scales: Mapping[str, object],
    vector_components: Mapping[str, Sequence[str]] | None = None,
) -> LocalSensitivityAnalysis:
    """Compute caller-declared OAT slopes without inspecting model code.

    The caller is responsible for producing each run with exactly one changed
    parameter. This boundary requires one unique explicit parameter name and
    delta per run; it does not infer or certify parameter changes from equations.
    """
    validated_baseline = _revalidate_trajectory(
        baseline, LocalSensitivityError
    )
    if not isinstance(perturbations, Sequence) or isinstance(
        perturbations, (str, bytes)
    ):
        raise TypeError("perturbations must be a sequence")
    if not perturbations:
        raise LocalSensitivityError("at least one perturbation run is required")
    validated_perturbations: list[ParameterPerturbation] = []
    for item in perturbations:
        if not isinstance(item, ParameterPerturbation):
            raise TypeError("perturbations must contain ParameterPerturbation records")
        try:
            validated_perturbations.append(
                ParameterPerturbation.model_validate(
                    item.model_dump(warnings=False)
                )
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise LocalSensitivityError(
                "perturbation record is not valid"
            ) from exc
    try:
        normalization = _normalization_from_inputs(
            validated_baseline,
            normalization_scales,
            vector_components or {},
        )
    except ConvergenceAnalysisError as exc:
        raise LocalSensitivityError(str(exc)) from exc
    validated_tuple = tuple(validated_perturbations)
    kind, points, summaries = _derive_sensitivity(
        validated_baseline, validated_tuple, normalization
    )
    return LocalSensitivityAnalysis(
        baseline=validated_baseline,
        perturbations=validated_tuple,
        analysis_kind=kind,
        normalization=normalization,
        point_slopes=points,
        summaries=summaries,
    )


__all__ = [
    "LocalSensitivityAnalysis",
    "LocalSensitivityError",
    "ParameterPerturbation",
    "SensitivityPointSlope",
    "SensitivitySummary",
    "analyze_local_sensitivity",
]
