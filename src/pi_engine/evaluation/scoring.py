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


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("score event_time must include a timezone")
    return value


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

    @field_validator("event_time")
    @classmethod
    def event_time_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value)

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


class LogScoreResult(_ImmutableScoreSchema):
    """Finite negative-log loss or structural positive infinity."""

    kind: Literal["finite", "positive_infinity"]
    value: FiniteFloat | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_representation(self) -> LogScoreResult:
        if self.kind == "finite" and self.value is None:
            raise ValueError("finite log score requires a value")
        if self.kind == "positive_infinity" and self.value is not None:
            raise ValueError("positive-infinity log score cannot carry a value")
        return self


class ProbabilityPointScore(_ImmutableScoreSchema):
    """One explicitly declared binary probability forecast."""

    outcome_id: NonEmptyString
    variable: NonEmptyString
    event_time: datetime
    predicted_probability: FiniteFloat = Field(ge=0.0, le=1.0)
    label: StrictInt = Field(ge=0, le=1)
    brier_score: FiniteFloat = Field(ge=0.0, le=1.0)
    log_score: LogScoreResult | None = None

    @field_validator("event_time")
    @classmethod
    def event_time_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @field_validator("log_score", mode="before")
    @classmethod
    def revalidate_log_score(cls, value: object) -> object:
        if isinstance(value, LogScoreResult):
            return LogScoreResult.model_validate(
                value.model_dump(warnings=False)
            )
        return value

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
            if event_probability == 0.0:
                expected = LogScoreResult(kind="positive_infinity")
            else:
                expected = LogScoreResult(
                    kind="finite", value=float(-math.log(event_probability))
                )
            if self.log_score != expected:
                raise ValueError("probability log score is inconsistent")
        return self


class ProbabilityMetrics(_ImmutableScoreSchema):
    """Proper scoring summaries for one declared binary variable."""

    variable: NonEmptyString
    count: StrictInt = Field(ge=1)
    mean_brier_score: FiniteFloat = Field(ge=0.0, le=1.0)
    mean_log_score: LogScoreResult | None = None

    @field_validator("mean_log_score", mode="before")
    @classmethod
    def revalidate_mean_log_score(cls, value: object) -> object:
        if isinstance(value, LogScoreResult):
            return LogScoreResult.model_validate(
                value.model_dump(warnings=False)
            )
        return value


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

    @field_validator("event_time")
    @classmethod
    def event_time_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value)

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
    reference: PredictionReference
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

    @field_validator("reference", mode="before")
    @classmethod
    def revalidate_reference(cls, value: object) -> object:
        if isinstance(value, PredictionReference):
            return PredictionReference.model_validate(
                value.model_dump(warnings=False)
            )
        return value

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
        if (
            self.artifact_type != self.reference.artifact_type
            or self.artifact_id != self.reference.artifact_id
            or not compare_digest(
                self.artifact_sha256, self.reference.artifact_sha256
            )
            or self.model_id != self.reference.model_id
            or self.model_version != self.reference.model_version
        ):
            raise ValueError("artifact score identity does not match reference")
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
    """Self-consistent post-reveal scores with no combined ranking.

    Retaining the revealed source detects detached or internally inconsistent
    copies.  It is not an authenticity signature: adversarial storage or
    transport still requires an external signature over the serialized report.
    """

    source: RevealedEvaluation
    case_id: NonEmptyString
    case_sha256: Sha256Hex
    binary_probability_variables: tuple[NonEmptyString, ...]
    include_log_score: StrictBool
    interval_levels: tuple[FiniteFloat, ...]
    artifacts: Annotated[tuple[ArtifactScore, ...], Field(min_length=1)]

    @field_validator("source", mode="before")
    @classmethod
    def revalidate_source(cls, value: object) -> object:
        if isinstance(value, RevealedEvaluation):
            return RevealedEvaluation.model_validate(
                value.model_dump(warnings=False)
            )
        return value

    @field_validator("binary_probability_variables", mode="before")
    @classmethod
    def validate_binary_variables(cls, value: object) -> object:
        return _explicit_names(value, "binary_probability_variables")

    @field_validator("interval_levels", mode="before")
    @classmethod
    def validate_interval_levels(cls, value: object) -> object:
        return _interval_levels(value)

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

    @model_validator(mode="after")
    def validate_source_binding(self) -> ForecastScoreReport:
        _validate_report_source(self)
        return self


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
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise ScoringError(
            f"{role} must be representable as a finite float"
        ) from exc
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


def _reference_key(reference: PredictionReference) -> tuple[str, str]:
    return reference.artifact_type, reference.artifact_id


def _score_key(score: ArtifactScore) -> tuple[str, str]:
    return score.artifact_type, score.artifact_id


def _validate_report_source(report: ForecastScoreReport) -> None:
    if (
        report.case_id != report.source.case_id
        or not compare_digest(report.case_sha256, report.source.case_sha256)
    ):
        raise ValueError("report identity must match source Case")

    references = report.source.prediction_references
    reference_keys = tuple(_reference_key(item) for item in references)
    if len(set(reference_keys)) != len(reference_keys):
        raise ValueError("prediction reference identities must be unique")
    artifact_keys = tuple(_score_key(item) for item in report.artifacts)
    if len(set(artifact_keys)) != len(artifact_keys):
        raise ValueError("report artifact identities must be unique")
    if set(artifact_keys) != set(reference_keys):
        raise ValueError("report artifacts must exactly match source references")
    references_by_key = dict(zip(reference_keys, references, strict=True))

    outcomes_by_id = {item.outcome_id: item for item in report.source.outcomes}
    binary_variables = set(report.binary_probability_variables)
    outcome_variables = {item.variable for item in report.source.outcomes}
    if not binary_variables <= outcome_variables:
        raise ValueError(
            "binary probability variables must identify source outcomes"
        )
    for artifact in report.artifacts:
        reference = references_by_key[_score_key(artifact)]
        if artifact.reference != reference:
            raise ValueError("artifact score reference must match source reference")
        points = (*artifact.continuous_points, *artifact.probability_points)
        point_ids = tuple(item.outcome_id for item in points)
        if (
            len(point_ids) != len(outcomes_by_id)
            or len(set(point_ids)) != len(point_ids)
            or set(point_ids) != set(outcomes_by_id)
        ):
            raise ValueError(
                "score points must match each source outcome record exactly once"
            )
        for point in artifact.continuous_points:
            outcome = outcomes_by_id[point.outcome_id]
            if point.variable in binary_variables:
                raise ValueError("binary source outcome requires probability score")
            try:
                expected_observed = _scalar(
                    outcome.value, f"source outcome for {point.variable}"
                )
            except ScoringError as exc:
                raise ValueError(str(exc)) from exc
            if (
                point.variable != outcome.variable
                or _utc(point.event_time) != _utc(outcome.event_time)
                or point.observed != expected_observed
            ):
                raise ValueError(
                    "continuous score point does not match source outcome"
                )
        for point in artifact.probability_points:
            outcome = outcomes_by_id[point.outcome_id]
            try:
                expected_label = _label(outcome.value)
            except ScoringError as exc:
                raise ValueError(str(exc)) from exc
            if (
                point.variable not in binary_variables
                or point.variable != outcome.variable
                or _utc(point.event_time) != _utc(outcome.event_time)
                or point.label != expected_label
            ):
                raise ValueError(
                    "probability score point does not match source outcome"
                )
            if report.include_log_score != (point.log_score is not None):
                raise ValueError("log-score presence must match report configuration")

        continuous_outcomes = tuple(
            item
            for item in report.source.outcomes
            if item.variable not in binary_variables
        )
        expected_interval_keys = {
            (outcome.outcome_id, level)
            for outcome in continuous_outcomes
            for level in report.interval_levels
        }
        interval_keys = tuple(
            (item.outcome_id, item.nominal_coverage)
            for item in artifact.intervals
        )
        if (
            len(interval_keys) != len(set(interval_keys))
            or set(interval_keys) != expected_interval_keys
        ):
            raise ValueError(
                "interval scores must match source outcomes and configured levels"
            )
        if report.interval_levels and reference.artifact_type != "trajectory_ensemble":
            raise ValueError("configured intervals require ensemble references")
        for interval in artifact.intervals:
            outcome = outcomes_by_id[interval.outcome_id]
            try:
                expected_observed = _scalar(
                    outcome.value, f"source outcome for {interval.variable}"
                )
            except ScoringError as exc:
                raise ValueError(str(exc)) from exc
            if (
                interval.variable != outcome.variable
                or interval.variable in binary_variables
                or _utc(interval.event_time) != _utc(outcome.event_time)
                or interval.observed != expected_observed
            ):
                raise ValueError("interval score does not match source outcome")


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
    raw = tuple(item.value for item in weights)
    scale = max(raw)
    if scale <= 0.0 or not math.isfinite(scale):
        raise ScoringError("scenario weights need a finite positive scale")
    try:
        scaled = tuple(value / scale for value in raw)
        total = math.fsum(scaled)
        normalized = tuple(value / total for value in scaled)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ScoringError("scenario weights cannot be normalized finitely") from exc
    if not all(math.isfinite(item) and item >= 0.0 for item in normalized):
        raise ScoringError("scenario weights cannot be normalized finitely")
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
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    try:
        numerator = math.fsum(
            (value / scale) * weight
            for value, weight in zip(values, weights, strict=True)
        )
        denominator = math.fsum(weights)
        result = scale * (numerator / denominator)
    except (OverflowError, ZeroDivisionError) as exc:
        raise ScoringError("point estimate is not finite") from exc
    if not math.isfinite(result):
        raise ScoringError("point estimate is not finite")
    return result


def _stable_mean(values: Sequence[float], message: str) -> float:
    if not values:
        raise ScoringError("mean requires at least one value")
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    try:
        normalized_mean = math.fsum(value / scale for value in values) / len(
            values
        )
        result = scale * normalized_mean
    except (OverflowError, ZeroDivisionError) as exc:
        raise ScoringError(message) from exc
    if not math.isfinite(result):
        raise ScoringError(message)
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
        mean_error = _stable_mean(
            tuple(item.error for item in items),
            "continuous metric is not finite",
        )
        mean_absolute_error = _stable_mean(
            tuple(item.absolute_error for item in items),
            "continuous metric is not finite",
        )
        mean_squared_error = _stable_mean(
            tuple(item.squared_error for item in items),
            "continuous metric is not finite",
        )
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
        mean_brier = _stable_mean(
            tuple(item.brier_score for item in items),
            "mean Brier score is not finite",
        )
        log_scores = tuple(
            item.log_score for item in items if item.log_score is not None
        )
        mean_log: LogScoreResult | None = None
        if log_scores:
            if len(log_scores) != count:
                raise ScoringError("log-score availability cannot be partial")
            if any(item.kind == "positive_infinity" for item in log_scores):
                mean_log = LogScoreResult(kind="positive_infinity")
            else:
                finite_values = tuple(
                    item.value
                    for item in log_scores
                    if item.value is not None
                )
                mean_log = LogScoreResult(
                    kind="finite",
                    value=float(
                        _stable_mean(
                            finite_values,
                            "mean log score is not finite",
                        )
                    ),
                )
        result.append(
            ProbabilityMetrics(
                variable=variable,
                count=count,
                mean_brier_score=float(mean_brier),
                mean_log_score=mean_log,
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
            log_score: LogScoreResult | None = None
            if include_log_score:
                event_probability = probability if label == 1 else 1.0 - probability
                if event_probability == 0.0:
                    log_score = LogScoreResult(kind="positive_infinity")
                else:
                    log_score = LogScoreResult(
                        kind="finite", value=float(-math.log(event_probability))
                    )
            probabilities.append(
                ProbabilityPointScore(
                    outcome_id=outcome.outcome_id,
                    variable=outcome.variable,
                    event_time=outcome.event_time,
                    predicted_probability=float(probability),
                    label=label,
                    brier_score=float(brier),
                    log_score=log_score,
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
        reference=reference,
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
    competing model scores into a ranking.  Each distinct outcome ID remains a
    scored record even when variable and event time repeat.  In v0.1 a
    comparison window is retained as source metadata, but scoring still uses
    only the prediction point at the outcome's exact UTC event time.

    The returned source-bound report is self-consistent, not cryptographically
    authentic; adversarial persistence or transport requires an external
    signature over its serialization.
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
    references = validated_revealed.prediction_references
    reference_keys = tuple(_reference_key(item) for item in references)
    if len(set(reference_keys)) != len(reference_keys):
        raise ScoringError("prediction reference identities must be unique")
    artifact_keys = tuple(
        _artifact_identity(item)[:2] for item in validated_artifacts
    )
    if len(set(artifact_keys)) != len(artifact_keys):
        raise ScoringError("supplied artifact identities must be unique")
    if set(artifact_keys) != set(reference_keys):
        raise ScoringError(
            "prediction artifact identity does not align exactly with revealed "
            "prediction references"
        )
    artifacts_by_key = dict(
        zip(artifact_keys, validated_artifacts, strict=True)
    )
    bound: list[tuple[PredictionArtifact, PredictionReference, str]] = []
    for reference in references:
        artifact = artifacts_by_key[_reference_key(reference)]
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
        source=validated_revealed,
        case_id=validated_revealed.case_id,
        case_sha256=validated_revealed.case_sha256,
        binary_probability_variables=binary_variables,
        include_log_score=include_log_score,
        interval_levels=levels,
        artifacts=scores,
    )
