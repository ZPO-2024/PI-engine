"""Calibration summaries derived only from reveal-bound score records."""

from __future__ import annotations

from collections.abc import Sequence
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

from pi_engine.evaluation.scoring import (
    ArtifactScore,
    ForecastScoreReport,
    IntervalPointScore,
    ProbabilityPointScore,
)
from pi_engine.schemas.common import FiniteFloat


NonEmptyString = Annotated[str, Field(min_length=1)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ArtifactType = Literal["trajectory", "trajectory_ensemble"]


class CalibrationError(ValueError):
    """A requested calibration summary is invalid or unrepresentable."""


class _ImmutableCalibrationSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class IntervalCalibrationSummary(_ImmutableCalibrationSchema):
    """Observed versus nominal coverage for one model variable and level."""

    artifact_type: ArtifactType
    artifact_id: NonEmptyString
    artifact_sha256: Sha256Hex
    model_id: NonEmptyString
    model_version: NonEmptyString
    variable: NonEmptyString
    nominal_coverage: FiniteFloat = Field(gt=0.0, lt=1.0)
    count: StrictInt = Field(ge=1)
    covered_count: StrictInt = Field(ge=0)
    observed_coverage: FiniteFloat = Field(ge=0.0, le=1.0)
    calibration_error: FiniteFloat
    mean_interval_width: FiniteFloat = Field(ge=0.0)
    interval_method: Literal["empirical_equal_tail_inverse_cdf"]

    @model_validator(mode="after")
    def validate_counts_and_coverage(self) -> IntervalCalibrationSummary:
        if self.covered_count > self.count:
            raise ValueError("covered_count must not exceed count")
        expected_coverage = self.covered_count / self.count
        if self.observed_coverage != expected_coverage:
            raise ValueError("observed coverage does not match covered_count")
        if self.calibration_error != expected_coverage - self.nominal_coverage:
            raise ValueError("interval calibration error is inconsistent")
        return self


class ProbabilityCalibrationBin(_ImmutableCalibrationSchema):
    """One exact left-closed probability bin; the final bin is right-closed."""

    lower_inclusive: FiniteFloat = Field(ge=0.0, le=1.0)
    upper: FiniteFloat = Field(ge=0.0, le=1.0)
    upper_inclusive: StrictBool
    count: StrictInt = Field(ge=0)
    mean_predicted_probability: FiniteFloat | None = Field(
        default=None, ge=0.0, le=1.0
    )
    observed_frequency: FiniteFloat | None = Field(
        default=None, ge=0.0, le=1.0
    )
    calibration_error: FiniteFloat | None = None

    @model_validator(mode="after")
    def validate_bin(self) -> ProbabilityCalibrationBin:
        if self.lower_inclusive >= self.upper:
            raise ValueError("probability bin lower edge must precede upper")
        summaries = (
            self.mean_predicted_probability,
            self.observed_frequency,
            self.calibration_error,
        )
        if self.count == 0 and any(value is not None for value in summaries):
            raise ValueError("empty probability bins cannot report summaries")
        if self.count > 0 and any(value is None for value in summaries):
            raise ValueError("populated probability bins require summaries")
        if self.count > 0:
            assert self.mean_predicted_probability is not None
            assert self.observed_frequency is not None
            assert self.calibration_error is not None
            if self.calibration_error != (
                self.observed_frequency - self.mean_predicted_probability
            ):
                raise ValueError("probability bin calibration is inconsistent")
        return self


class ProbabilityCalibrationSummary(_ImmutableCalibrationSchema):
    """Explicit-bin reliability summary for one model and binary variable."""

    artifact_type: ArtifactType
    artifact_id: NonEmptyString
    artifact_sha256: Sha256Hex
    model_id: NonEmptyString
    model_version: NonEmptyString
    variable: NonEmptyString
    count: StrictInt = Field(ge=1)
    bins: Annotated[tuple[ProbabilityCalibrationBin, ...], Field(min_length=1)]

    @field_validator("bins", mode="before")
    @classmethod
    def revalidate_bins(cls, value: object) -> object:
        if isinstance(value, (tuple, list)):
            return tuple(
                ProbabilityCalibrationBin.model_validate(
                    item.model_dump(warnings=False)
                )
                if isinstance(item, ProbabilityCalibrationBin)
                else item
                for item in value
            )
        return value

    @model_validator(mode="after")
    def validate_count(self) -> ProbabilityCalibrationSummary:
        if sum(item.count for item in self.bins) != self.count:
            raise ValueError("probability calibration bin counts must match count")
        return self


class CalibrationSummary(_ImmutableCalibrationSchema):
    """Separate interval and probability calibration; no model aggregation."""

    intervals: tuple[IntervalCalibrationSummary, ...]
    probability: tuple[ProbabilityCalibrationSummary, ...]

    @field_validator("intervals", "probability", mode="before")
    @classmethod
    def revalidate_summaries(cls, value: object, info: object) -> object:
        if not isinstance(value, (tuple, list)):
            return value
        model: type[BaseModel]
        if getattr(info, "field_name", "") == "intervals":
            model = IntervalCalibrationSummary
        else:
            model = ProbabilityCalibrationSummary
        return tuple(
            model.model_validate(item.model_dump(warnings=False))
            if isinstance(item, model)
            else item
            for item in value
        )


def _revalidate_report(value: object) -> ForecastScoreReport:
    if not isinstance(value, ForecastScoreReport):
        raise TypeError("report must be a ForecastScoreReport")
    try:
        return ForecastScoreReport.model_validate(
            value.model_dump(warnings=False)
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise CalibrationError("score report is not valid") from exc


def _bin_edges(value: object | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("probability bin edges must be a sequence")
    edges = tuple(value)
    if (
        len(edges) < 2
        or any(
            isinstance(item, bool)
            or not isinstance(item, float)
            or not math.isfinite(item)
            or not 0.0 <= item <= 1.0
            for item in edges
        )
        or edges[0] != 0.0
        or edges[-1] != 1.0
        or any(later <= earlier for earlier, later in zip(edges, edges[1:]))
    ):
        raise CalibrationError(
            "probability bin edges must be finite, strictly increasing floats "
            "from exactly 0.0 to 1.0"
        )
    return edges


def _interval_calibration(
    artifacts: tuple[ArtifactScore, ...],
) -> tuple[IntervalCalibrationSummary, ...]:
    grouped: dict[
        tuple[str, str, str, str, str, str, float, str],
        list[IntervalPointScore],
    ] = {}
    for artifact in artifacts:
        for item in artifact.intervals:
            key = (
                artifact.artifact_type,
                artifact.artifact_id,
                artifact.artifact_sha256,
                artifact.model_id,
                artifact.model_version,
                item.variable,
                item.nominal_coverage,
                item.interval_method,
            )
            grouped.setdefault(key, []).append(item)

    summaries: list[IntervalCalibrationSummary] = []
    for key, items in grouped.items():
        (
            artifact_type,
            artifact_id,
            artifact_sha256,
            model_id,
            model_version,
            variable,
            nominal_coverage,
            interval_method,
        ) = key
        count = len(items)
        covered_count = sum(1 for item in items if item.covered)
        observed_coverage = covered_count / count
        calibration_error = observed_coverage - nominal_coverage
        try:
            mean_width = math.fsum(
                item.upper - item.lower for item in items
            ) / count
        except OverflowError as exc:
            raise CalibrationError("interval width is not finite") from exc
        if not all(
            math.isfinite(value)
            for value in (
                observed_coverage,
                calibration_error,
                mean_width,
            )
        ):
            raise CalibrationError("interval calibration is not finite")
        summaries.append(
            IntervalCalibrationSummary(
                artifact_type=artifact_type,
                artifact_id=artifact_id,
                artifact_sha256=artifact_sha256,
                model_id=model_id,
                model_version=model_version,
                variable=variable,
                nominal_coverage=float(nominal_coverage),
                count=count,
                covered_count=covered_count,
                observed_coverage=float(observed_coverage),
                calibration_error=float(calibration_error),
                mean_interval_width=float(mean_width),
                interval_method=interval_method,
            )
        )
    return tuple(summaries)


def _point_in_bin(
    point: ProbabilityPointScore,
    lower: float,
    upper: float,
    upper_inclusive: bool,
) -> bool:
    if upper_inclusive:
        return lower <= point.predicted_probability <= upper
    return lower <= point.predicted_probability < upper


def _probability_bins(
    points: list[ProbabilityPointScore], edges: tuple[float, ...]
) -> tuple[ProbabilityCalibrationBin, ...]:
    bins: list[ProbabilityCalibrationBin] = []
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        upper_inclusive = index == len(edges) - 2
        members = tuple(
            item
            for item in points
            if _point_in_bin(item, lower, upper, upper_inclusive)
        )
        if not members:
            bins.append(
                ProbabilityCalibrationBin(
                    lower_inclusive=float(lower),
                    upper=float(upper),
                    upper_inclusive=upper_inclusive,
                    count=0,
                )
            )
            continue
        count = len(members)
        mean_probability = (
            math.fsum(item.predicted_probability for item in members) / count
        )
        observed_frequency = math.fsum(item.label for item in members) / count
        calibration_error = observed_frequency - mean_probability
        bins.append(
            ProbabilityCalibrationBin(
                lower_inclusive=float(lower),
                upper=float(upper),
                upper_inclusive=upper_inclusive,
                count=count,
                mean_predicted_probability=float(mean_probability),
                observed_frequency=float(observed_frequency),
                calibration_error=float(calibration_error),
            )
        )
    return tuple(bins)


def _probability_calibration(
    artifacts: tuple[ArtifactScore, ...], edges: tuple[float, ...] | None
) -> tuple[ProbabilityCalibrationSummary, ...]:
    if edges is None:
        return ()
    summaries: list[ProbabilityCalibrationSummary] = []
    for artifact in artifacts:
        by_variable: dict[str, list[ProbabilityPointScore]] = {}
        for point in artifact.probability_points:
            by_variable.setdefault(point.variable, []).append(point)
        for variable, points in by_variable.items():
            summaries.append(
                ProbabilityCalibrationSummary(
                    artifact_type=artifact.artifact_type,
                    artifact_id=artifact.artifact_id,
                    artifact_sha256=artifact.artifact_sha256,
                    model_id=artifact.model_id,
                    model_version=artifact.model_version,
                    variable=variable,
                    count=len(points),
                    bins=_probability_bins(points, edges),
                )
            )
    return tuple(summaries)


def summarize_calibration(
    report: ForecastScoreReport,
    *,
    probability_bin_edges: Sequence[float] | None = None,
) -> CalibrationSummary:
    """Summarize score records using only caller-declared probability bins."""
    validated = _revalidate_report(report)
    edges = _bin_edges(probability_bin_edges)
    return CalibrationSummary(
        intervals=_interval_calibration(validated.artifacts),
        probability=_probability_calibration(validated.artifacts, edges),
    )
