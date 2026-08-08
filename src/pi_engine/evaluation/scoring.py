"""Reveal-bound, artifact-bound forecast scoring without a master score."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from hmac import compare_digest
import json
import math
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

from pi_engine.evaluation.holdout import (
    PredictionReference,
    RevealedEvaluation,
)
from pi_engine.schemas.common import FiniteFloat
from pi_engine.schemas.outcome import Outcome
from pi_engine.schemas.trajectory import Trajectory, TrajectoryEnsemble


NonEmptyString = Annotated[str, Field(min_length=1)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ArtifactType = Literal["trajectory", "trajectory_ensemble"]
PredictionArtifact = Trajectory | TrajectoryEnsemble


class ScoringError(ValueError):
    """A revealed evaluation cannot be scored without changing its meaning."""


class _ImmutableScoreSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class ContinuousPointScore(_ImmutableScoreSchema):
    """One aligned scalar forecast error, retaining outcome identity."""

    outcome_id: NonEmptyString
    variable: NonEmptyString
    event_time: datetime
    forecast: FiniteFloat
    observed: FiniteFloat
    error: FiniteFloat
    absolute_error: FiniteFloat = Field(ge=0.0)
    squared_error: FiniteFloat = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_arithmetic(self) -> ContinuousPointScore:
        expected_error = self.forecast - self.observed
        if (
            not math.isfinite(expected_error)
            or self.error != expected_error
            or self.absolute_error != abs(expected_error)
            or self.squared_error != expected_error * expected_error
        ):
            raise ValueError("continuous point arithmetic is inconsistent")
        return self


class ContinuousMetrics(_ImmutableScoreSchema):
    """Transparent deterministic metrics for one artifact and variable."""

    variable: NonEmptyString
    count: StrictInt = Field(ge=1)
    mean_error: FiniteFloat
    mean_absolute_error: FiniteFloat = Field(ge=0.0)
    mean_squared_error: FiniteFloat = Field(ge=0.0)
    root_mean_squared_error: FiniteFloat = Field(ge=0.0)


class ProbabilityPointScore(_ImmutableScoreSchema):
    """One explicitly declared binary probability forecast."""

    outcome_id: NonEmptyString
    variable: NonEmptyString
    event_time: datetime
    predicted_probability: FiniteFloat = Field(ge=0.0, le=1.0)
    label: StrictInt = Field(ge=0, le=1)
    brier_score: FiniteFloat = Field(ge=0.0, le=1.0)
    log_score: FiniteFloat | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_arithmetic(self) -> ProbabilityPointScore:
        expected_brier = (self.predicted_probability - self.label) ** 2
        if self.brier_score != expected_brier:
            raise ValueError("probability point arithmetic is inconsistent")
        if self.log_score is not None:
            event_probability = (
                self.predicted_probability
                if self.label == 1
                else 1.0 - self.predicted_probability
            )
            if (
                event_probability == 0.0
                or self.log_score != -math.log(event_probability)
            ):
                raise ValueError("probability log score is inconsistent")
        return self


class ProbabilityMetrics(_ImmutableScoreSchema):
    """Proper scoring summaries for one declared binary variable."""

    variable: NonEmptyString
    count: StrictInt = Field(ge=1)
    mean_brier_score: FiniteFloat = Field(ge=0.0, le=1.0)
    mean_log_score: FiniteFloat | None = Field(default=None, ge=0.0)


class IntervalPointScore(_ImmutableScoreSchema):
    """One distribution-free central empirical interval observation."""

    outcome_id: NonEmptyString
    variable: NonEmptyString
    event_time: datetime
    nominal_coverage: FiniteFloat = Field(gt=0.0, lt=1.0)
    lower: FiniteFloat
    upper: FiniteFloat
    observed: FiniteFloat
    covered: StrictBool
    interval_method: Literal["empirical_equal_tail_inverse_cdf"]

    @model_validator(mode="after")
    def validate_interval(self) -> IntervalPointScore:
        if self.lower > self.upper:
            raise ValueError("interval lower must not exceed upper")
        expected_coverage = self.lower <= self.observed <= self.upper
        if self.covered != expected_coverage:
            raise ValueError("interval coverage flag is inconsistent")
        return self


class ArtifactScore(_ImmutableScoreSchema):
    """Scores remain separated by immutable prediction-artifact identity."""

    artifact_type: ArtifactType
    artifact_id: NonEmptyString
    artifact_sha256: Sha256Hex
    model_id: NonEmptyString
    model_version: NonEmptyString
    point_estimate_method: Literal[
        "trajectory_value",
        "equal_weight_raw_samples",
        "probability_weighted_raw_samples",
        "normalized_relative_weight_raw_samples",
    ]
    continuous_points: tuple[ContinuousPointScore, ...]
    continuous_metrics: tuple[ContinuousMetrics, ...]
    probability_points: tuple[ProbabilityPointScore, ...]
    probability_metrics: tuple[ProbabilityMetrics, ...]
    intervals: tuple[IntervalPointScore, ...]

    @field_validator(
        "continuous_points",
        "continuous_metrics",
        "probability_points",
        "probability_metrics",
        "intervals",
        mode="before",
    )
    @classmethod
    def revalidate_score_records(cls, value: object, info: object) -> object:
        if not isinstance(value, (tuple, list)):
            return value
        model_by_field: dict[str, type[BaseModel]] = {
            "continuous_points": ContinuousPointScore,
            "continuous_metrics": ContinuousMetrics,
            "probability_points": ProbabilityPointScore,
            "probability_metrics": ProbabilityMetrics,
            "intervals": IntervalPointScore,
        }
        field_name = getattr(info, "field_name", "")
        model = model_by_field[field_name]
        return tuple(
            model.model_validate(item.model_dump(warnings=False))
            if isinstance(item, model)
            else item
            for item in value
        )

    @model_validator(mode="after")
    def validate_derived_metrics(self) -> ArtifactScore:
        if self.continuous_metrics != _continuous_metrics(
            self.continuous_points
        ):
            raise ValueError("continuous metrics do not match point scores")
        if self.probability_metrics != _probability_metrics(
            self.probability_points
        ):
            raise ValueError("probability metrics do not match point scores")
        return self


class ForecastScoreReport(_ImmutableScoreSchema):
    """Per-artifact, per-variable scores; deliberately no combined ranking."""

    case_id: NonEmptyString
    case_sha256: Sha256Hex
    artifacts: Annotated[tuple[ArtifactScore, ...], Field(min_length=1)]

    @field_validator("artifacts", mode="before")
    @classmethod
    def revalidate_artifacts(cls, value: object) -> object:
        if isinstance(value, (tuple, list)):
            return tuple(
                ArtifactScore.model_validate(item.model_dump(warnings=False))
                if isinstance(item, ArtifactScore)
                else item
                for item in value
            )
        return value


def _canonical_json_bytes(value: BaseModel) -> bytes:
    return json.dumps(
        value.model_dump(mode="json", warnings=False),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _artifact_sha256(artifact: PredictionArtifact) -> str:
    return sha256(_canonical_json_bytes(artifact)).hexdigest()


def _utc(value: datetime) -> datetime:
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError) as exc:
        raise ScoringError("score time cannot be normalized to UTC") from exc


def _revalidate_revealed(value: object) -> RevealedEvaluation:
    if not isinstance(value, RevealedEvaluation):
        raise TypeError("revealed must be a RevealedEvaluation")
    try:
        return RevealedEvaluation.model_validate(
            value.model_dump(warnings=False)
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise ScoringError("revealed evaluation is not valid") from exc


def _revalidate_artifact(value: object) -> PredictionArtifact:
    if isinstance(value, TrajectoryEnsemble):
        try:
            return TrajectoryEnsemble.model_validate(
                value.model_dump(warnings=False)
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise ScoringError(
                "artifact must be a valid TrajectoryEnsemble"
            ) from exc
    if isinstance(value, Trajectory):
        try:
            return Trajectory.model_validate(value.model_dump(warnings=False))
        except (TypeError, ValueError, ValidationError) as exc:
            raise ScoringError("artifact must be a valid Trajectory") from exc
    raise TypeError("artifacts must contain Trajectory or TrajectoryEnsemble")


def _artifact_identity(
    artifact: PredictionArtifact,
) -> tuple[ArtifactType, str, tuple[str, ...], object]:
    if isinstance(artifact, TrajectoryEnsemble):
        return (
            "trajectory_ensemble",
            artifact.ensemble_id,
            tuple(item.trajectory_id for item in artifact.trajectories),
            artifact.trajectories[0].horizon,
        )
    return (
        "trajectory",
        artifact.trajectory_id,
        (artifact.trajectory_id,),
        artifact.horizon,
    )


def _bind_artifact(
    artifact: PredictionArtifact,
    reference: PredictionReference,
    revealed: RevealedEvaluation,
) -> str:
    artifact_type, artifact_id, trajectory_ids, horizon = _artifact_identity(
        artifact
    )
    if artifact_type != reference.artifact_type:
        raise ScoringError("prediction artifact type does not match reference")
    if artifact_id != reference.artifact_id:
        raise ScoringError("prediction artifact identity does not match reference")
    if (
        artifact.model_id != reference.model_id
        or artifact.model_version != reference.model_version
    ):
        raise ScoringError("prediction model identity does not match reference")
    if artifact.case_id != reference.case_id:
        raise ScoringError("prediction case identity does not match reference")
    if (
        reference.case_id != revealed.case_id
        or not compare_digest(reference.case_sha256, revealed.case_sha256)
    ):
        raise ScoringError("prediction reference Case digest does not match reveal")
    if (
        _utc(horizon.start_at) != _utc(reference.horizon.start_at)
        or _utc(horizon.end_at) != _utc(reference.horizon.end_at)
    ):
        raise ScoringError("prediction horizon does not match reference")
    if trajectory_ids != reference.trajectory_ids:
        raise ScoringError(
            "prediction trajectory identities do not match reference"
        )
    digest = _artifact_sha256(artifact)
    if not compare_digest(digest, reference.artifact_sha256):
        raise ScoringError("prediction artifact SHA-256 does not match reference")
    return digest


def _explicit_names(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of names")
    names = tuple(value)
    if any(not isinstance(item, str) or not item for item in names):
        raise ScoringError(f"{field_name} must contain nonempty strings")
    if len(set(names)) != len(names):
        raise ScoringError(f"{field_name} must not contain duplicates")
    return names


def _interval_levels(value: object) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("interval_levels must be a sequence")
    levels: list[float] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, float)
            or not math.isfinite(item)
            or not 0.0 < item < 1.0
        ):
            raise ScoringError(
                "interval levels must be finite floats strictly between 0 and 1"
            )
        levels.append(item)
    if any(later <= earlier for earlier, later in zip(levels, levels[1:])):
        raise ScoringError("interval levels must be strictly increasing")
    return tuple(levels)


def _scalar(value: object, role: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoringError(f"{role} must be a scalar numeric value, not bool")
    converted = float(value)
    if not math.isfinite(converted):
        raise ScoringError(f"{role} must be finite")
    return converted


def _label(value: object) -> int:
    if isinstance(value, bool):
        raise ScoringError("binary label must not be bool")
    numeric = _scalar(value, "binary label")
    if numeric not in (0.0, 1.0):
        raise ScoringError("binary label must be exactly 0 or 1")
    return int(numeric)


def _probability(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ScoringError("predicted probability must be within [0, 1]")
    return value


def _trajectories(
    artifact: PredictionArtifact,
) -> tuple[Trajectory, ...]:
    if isinstance(artifact, TrajectoryEnsemble):
        return artifact.trajectories
    return (artifact,)


def _weights_and_method(
    artifact: PredictionArtifact,
) -> tuple[tuple[float, ...], str]:
    trajectories = _trajectories(artifact)
    if isinstance(artifact, Trajectory):
        return (1.0,), "trajectory_value"
    declared = tuple(item.scenario_weight for item in trajectories)
    if all(item is None for item in declared):
        count = len(trajectories)
        return tuple(1.0 / count for _ in trajectories), "equal_weight_raw_samples"
    weights = tuple(item for item in declared if item is not None)
    total = math.fsum(item.value for item in weights)
    normalized = tuple(item.value / total for item in weights)
    if weights[0].kind == "probability":
        return normalized, "probability_weighted_raw_samples"
    return normalized, "normalized_relative_weight_raw_samples"


def _aligned_values(
    artifact: PredictionArtifact, outcome: Outcome
) -> tuple[float, ...]:
    at = _utc(outcome.event_time)
    values: list[float] = []
    for trajectory in _trajectories(artifact):
        matches = tuple(point for point in trajectory.points if _utc(point.at) == at)
        if len(matches) != 1:
            raise ScoringError(
                "prediction must have exactly one point at outcome UTC time"
            )
        if outcome.variable not in matches[0].values:
            raise ScoringError(
                "prediction point is not aligned to the outcome variable"
            )
        values.append(
            _scalar(
                matches[0].values[outcome.variable],
                f"forecast for {outcome.variable}",
            )
        )
    return tuple(values)


def _weighted_mean(values: tuple[float, ...], weights: tuple[float, ...]) -> float:
    try:
        result = math.fsum(
            value * weight
            for value, weight in zip(values, weights, strict=True)
        )
    except OverflowError as exc:
        raise ScoringError("point estimate is not finite") from exc
    if not math.isfinite(result):
        raise ScoringError("point estimate is not finite")
    return result


def _finite_operation(value: float, message: str) -> float:
    if not math.isfinite(value):
        raise ScoringError(message)
    return value


def _empirical_quantile(
    values: tuple[float, ...], weights: tuple[float, ...], probability: float
) -> float:
    ordered = sorted(zip(values, weights, strict=True), key=lambda item: item[0])
    threshold = probability * math.fsum(weights)
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def _continuous_metrics(
    points: tuple[ContinuousPointScore, ...],
) -> tuple[ContinuousMetrics, ...]:
    by_variable: dict[str, list[ContinuousPointScore]] = {}
    for point in points:
        by_variable.setdefault(point.variable, []).append(point)
    result: list[ContinuousMetrics] = []
    for variable, items in by_variable.items():
        count = len(items)
        try:
            mean_error = math.fsum(item.error for item in items) / count
            mean_absolute_error = (
                math.fsum(item.absolute_error for item in items) / count
            )
            mean_squared_error = (
                math.fsum(item.squared_error for item in items) / count
            )
        except OverflowError as exc:
            raise ScoringError("continuous metric is not finite") from exc
        for value in (mean_error, mean_absolute_error, mean_squared_error):
            _finite_operation(value, "continuous metric is not finite")
        root_mean_squared_error = math.sqrt(mean_squared_error)
        result.append(
            ContinuousMetrics(
                variable=variable,
                count=count,
                mean_error=float(mean_error),
                mean_absolute_error=float(mean_absolute_error),
                mean_squared_error=float(mean_squared_error),
                root_mean_squared_error=float(root_mean_squared_error),
            )
        )
    return tuple(result)


def _probability_metrics(
    points: tuple[ProbabilityPointScore, ...],
) -> tuple[ProbabilityMetrics, ...]:
    by_variable: dict[str, list[ProbabilityPointScore]] = {}
    for point in points:
        by_variable.setdefault(point.variable, []).append(point)
    result: list[ProbabilityMetrics] = []
    for variable, items in by_variable.items():
        count = len(items)
        mean_brier = math.fsum(item.brier_score for item in items) / count
        log_scores = tuple(
            item.log_score for item in items if item.log_score is not None
        )
        mean_log = None
        if log_scores:
            if len(log_scores) != count:
                raise ScoringError("log-score availability cannot be partial")
            mean_log = math.fsum(log_scores) / count
        result.append(
            ProbabilityMetrics(
                variable=variable,
                count=count,
                mean_brier_score=float(mean_brier),
                mean_log_score=None if mean_log is None else float(mean_log),
            )
        )
    return tuple(result)


def _score_artifact(
    revealed: RevealedEvaluation,
    artifact: PredictionArtifact,
    reference: PredictionReference,
    digest: str,
    binary_variables: tuple[str, ...],
    include_log_score: bool,
    interval_levels: tuple[float, ...],
) -> ArtifactScore:
    weights, point_method = _weights_and_method(artifact)
    continuous: list[ContinuousPointScore] = []
    probabilities: list[ProbabilityPointScore] = []
    intervals: list[IntervalPointScore] = []
    for outcome in revealed.outcomes:
        raw_values = _aligned_values(artifact, outcome)
        forecast = _weighted_mean(raw_values, weights)
        if outcome.variable in binary_variables:
            probability = _probability(forecast)
            if any(not 0.0 <= item <= 1.0 for item in raw_values):
                raise ScoringError(
                    "each raw predicted probability must be within [0, 1]"
                )
            label = _label(outcome.value)
            brier = (probability - label) ** 2
            log_score: float | None = None
            if include_log_score:
                event_probability = probability if label == 1 else 1.0 - probability
                if event_probability == 0.0:
                    raise ScoringError(
                        "log score has zero probability for the observed label"
                    )
                log_score = -math.log(event_probability)
            probabilities.append(
                ProbabilityPointScore(
                    outcome_id=outcome.outcome_id,
                    variable=outcome.variable,
                    event_time=outcome.event_time,
                    predicted_probability=float(probability),
                    label=label,
                    brier_score=float(brier),
                    log_score=None if log_score is None else float(log_score),
                )
            )
            continue

        observed = _scalar(outcome.value, f"outcome for {outcome.variable}")
        error = _finite_operation(
            forecast - observed, "nonfinite continuous error"
        )
        absolute_error = abs(error)
        squared_error = _finite_operation(
            error * error, "nonfinite continuous error"
        )
        continuous.append(
            ContinuousPointScore(
                outcome_id=outcome.outcome_id,
                variable=outcome.variable,
                event_time=outcome.event_time,
                forecast=float(forecast),
                observed=float(observed),
                error=float(error),
                absolute_error=float(absolute_error),
                squared_error=float(squared_error),
            )
        )
        if isinstance(artifact, TrajectoryEnsemble):
            for level in interval_levels:
                tail = (1.0 - level) / 2.0
                lower = _empirical_quantile(raw_values, weights, tail)
                upper = _empirical_quantile(raw_values, weights, 1.0 - tail)
                if lower > upper:
                    raise ScoringError("empirical interval lower exceeds upper")
                intervals.append(
                    IntervalPointScore(
                        outcome_id=outcome.outcome_id,
                        variable=outcome.variable,
                        event_time=outcome.event_time,
                        nominal_coverage=level,
                        lower=float(lower),
                        upper=float(upper),
                        observed=float(observed),
                        covered=lower <= observed <= upper,
                        interval_method="empirical_equal_tail_inverse_cdf",
                    )
                )

    artifact_type, artifact_id, _, _ = _artifact_identity(artifact)
    return ArtifactScore(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        artifact_sha256=digest,
        model_id=artifact.model_id,
        model_version=artifact.model_version,
        point_estimate_method=point_method,
        continuous_points=tuple(continuous),
        continuous_metrics=_continuous_metrics(tuple(continuous)),
        probability_points=tuple(probabilities),
        probability_metrics=_probability_metrics(tuple(probabilities)),
        intervals=tuple(intervals),
    )


def score_revealed_evaluation(
    revealed: RevealedEvaluation,
    artifacts: Sequence[PredictionArtifact],
    *,
    binary_probability_variables: Sequence[str] = (),
    include_log_score: bool = False,
    interval_levels: Sequence[float] = (),
) -> ForecastScoreReport:
    """Score only revealed outcomes against the exact authorized artifacts.

    Binary probability semantics and empirical interval levels are explicit
    caller declarations.  The function neither fits a distribution nor combines
    competing model scores into a ranking.
    """
    validated_revealed = _revalidate_revealed(revealed)
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise TypeError("artifacts must be a sequence of prediction artifacts")
    if len(artifacts) != len(validated_revealed.prediction_references):
        raise ScoringError(
            "artifacts must align exactly with revealed prediction references"
        )
    if not isinstance(include_log_score, bool):
        raise TypeError("include_log_score must be bool")
    binary_variables = _explicit_names(
        binary_probability_variables, "binary_probability_variables"
    )
    outcome_variables = {item.variable for item in validated_revealed.outcomes}
    unused = tuple(item for item in binary_variables if item not in outcome_variables)
    if unused:
        raise ScoringError(
            "binary probability variables must identify revealed outcomes"
        )
    levels = _interval_levels(interval_levels)

    validated_artifacts = tuple(_revalidate_artifact(item) for item in artifacts)
    bound: list[tuple[PredictionArtifact, PredictionReference, str]] = []
    for artifact, reference in zip(
        validated_artifacts,
        validated_revealed.prediction_references,
        strict=True,
    ):
        digest = _bind_artifact(artifact, reference, validated_revealed)
        bound.append((artifact, reference, digest))

    if levels and any(
        not isinstance(artifact, TrajectoryEnsemble)
        for artifact, _, _ in bound
    ):
        raise ScoringError("interval levels require TrajectoryEnsemble artifacts")
    if levels and not (outcome_variables - set(binary_variables)):
        raise ScoringError("interval levels require continuous revealed outcomes")

    scores = tuple(
        _score_artifact(
            validated_revealed,
            artifact,
            reference,
            digest,
            binary_variables,
            include_log_score,
            levels,
        )
        for artifact, reference, digest in bound
    )
    return ForecastScoreReport(
        case_id=validated_revealed.case_id,
        case_sha256=validated_revealed.case_sha256,
        artifacts=scores,
    )
