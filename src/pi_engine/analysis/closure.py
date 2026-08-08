"""Conservative typed closure events derived from retained spread evidence."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from fractions import Fraction
import math
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

from pi_engine._numeric import utc_instant_key
from pi_engine.analysis._shared import (
    canonical_model_content,
    canonical_record_content,
)
from pi_engine.analysis.divergence import SpreadAnalysis, SpreadPoint
from pi_engine.schemas.common import FiniteFloat, Provenance
from pi_engine.schemas.trajectory import Trajectory, TrajectoryEnsemble


NonEmptyString = Annotated[str, Field(min_length=1)]
ComponentName = NonEmptyString | None
ClosureDomain = Literal["epistemic", "causal", "unknown"]
ClosureFirmness = Literal["provisional", "hard", "unknown"]
ClosureTransitionStyle = Literal["gradual", "abrupt", "unknown"]
ProxyName = Literal[
    "weighted_population_standard_deviation_proxy",
    "weighted_exact_value_gini_impurity_proxy",
    "positive_weight_bounding_interval_width_proxy",
]


class ClosureAnalysisError(ValueError):
    """Spread or caller evidence cannot support an auditable closure report."""


class _ImmutableClosureSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


def _utc(value: datetime, error_type: type[ValueError] = ValueError) -> int:
    return utc_instant_key(
        value,
        role="closure evidence time",
        error_type=error_type,
    )


def _require_timezone(value: datetime, role: str) -> datetime:
    _utc(value)
    return value


class ClosureThresholds(_ImmutableClosureSchema):
    """Caller-selected relative thresholds; no implicit epsilon is applied."""

    minimum_relative_contraction: FiniteFloat = Field(gt=0.0, le=1.0)
    abrupt_relative_contraction: FiniteFloat = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def abrupt_threshold_must_cover_candidate_threshold(self) -> ClosureThresholds:
        if (
            self.abrupt_relative_contraction
            < self.minimum_relative_contraction
        ):
            raise ValueError(
                "abrupt contraction threshold cannot be below the minimum "
                "closure threshold"
            )
        return self


class InformationUpdateEvidence(_ImmutableClosureSchema):
    """Caller assertion that information changed at one exact analyzed scope."""

    kind: Literal["information_update"] = "information_update"
    evidence_id: NonEmptyString
    effective_at: datetime
    variable: NonEmptyString
    component: ComponentName
    description: NonEmptyString
    provenance: Provenance

    @field_validator("effective_at")
    @classmethod
    def effective_time_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value, "information effective_at")

    @field_validator("provenance", mode="before")
    @classmethod
    def revalidate_provenance(cls, value: object) -> object:
        return _revalidate_provenance(value)

    @model_validator(mode="after")
    def provenance_must_have_audit_identity(self) -> InformationUpdateEvidence:
        if self.provenance.reference is None or not self.provenance.reference:
            raise ValueError("closure evidence provenance requires a reference")
        return self


class KnownHardConstraintEvidence(_ImmutableClosureSchema):
    """Caller assertion of a known hard causal constraint at one exact scope."""

    kind: Literal["known_causal_hard_constraint"] = (
        "known_causal_hard_constraint"
    )
    evidence_id: NonEmptyString
    constraint_id: NonEmptyString
    effective_at: datetime
    variable: NonEmptyString
    component: ComponentName
    description: NonEmptyString
    provenance: Provenance

    @field_validator("effective_at")
    @classmethod
    def effective_time_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value, "hard constraint effective_at")

    @field_validator("provenance", mode="before")
    @classmethod
    def revalidate_provenance(cls, value: object) -> object:
        return _revalidate_provenance(value)

    @model_validator(mode="after")
    def provenance_must_have_audit_identity(
        self,
    ) -> KnownHardConstraintEvidence:
        if self.provenance.reference is None or not self.provenance.reference:
            raise ValueError("closure evidence provenance requires a reference")
        return self


class ClosureConfiguration(_ImmutableClosureSchema):
    """All caller declarations retained beside the derived events."""

    thresholds: ClosureThresholds
    information_updates: tuple[InformationUpdateEvidence, ...]
    hard_constraints: tuple[KnownHardConstraintEvidence, ...]

    @field_validator("thresholds", mode="before")
    @classmethod
    def revalidate_thresholds(cls, value: object) -> object:
        return _revalidate_model(value, ClosureThresholds)

    @field_validator("information_updates", mode="before")
    @classmethod
    def revalidate_information_updates(cls, value: object) -> object:
        return _revalidate_records(value, InformationUpdateEvidence)

    @field_validator("hard_constraints", mode="before")
    @classmethod
    def revalidate_hard_constraints(cls, value: object) -> object:
        return _revalidate_records(value, KnownHardConstraintEvidence)

    @model_validator(mode="after")
    def evidence_identities_must_be_unique(self) -> ClosureConfiguration:
        evidence_ids = tuple(
            item.evidence_id
            for item in (*self.information_updates, *self.hard_constraints)
        )
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("closure evidence identities must be unique")
        return self


class ClosureProxyMagnitude(_ImmutableClosureSchema):
    """Finite proxy value or an explicit structural numeric state."""

    kind: Literal[
        "finite",
        "structural_zero",
        "above_float_range",
        "below_float_resolution",
    ]
    value: FiniteFloat | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_representation(self) -> ClosureProxyMagnitude:
        if self.kind == "finite" and self.value is None:
            raise ValueError("finite closure proxy requires a positive value")
        if self.kind != "finite" and self.value is not None:
            raise ValueError("structural closure proxy cannot carry a float value")
        return self


class ClosureProxyMetric(_ImmutableClosureSchema):
    """An explicitly named descriptive proxy, not a latent master score."""

    name: ProxyName
    magnitude: ClosureProxyMagnitude

    @field_validator("magnitude", mode="before")
    @classmethod
    def revalidate_magnitude(cls, value: object) -> object:
        return _revalidate_model(value, ClosureProxyMagnitude)


class ClosureProxySnapshot(_ImmutableClosureSchema):
    """Three distinct empirical proxies at one retained spread point."""

    at: datetime
    spread: ClosureProxyMetric
    entropy_like: ClosureProxyMetric
    reachable_set: ClosureProxyMetric

    @field_validator("at")
    @classmethod
    def time_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value, "closure proxy time")

    @field_validator("spread", "entropy_like", "reachable_set", mode="before")
    @classmethod
    def revalidate_metric(cls, value: object) -> object:
        return _revalidate_model(value, ClosureProxyMetric)

    @model_validator(mode="after")
    def proxy_names_must_match_their_meaning(self) -> ClosureProxySnapshot:
        expected = (
            "weighted_population_standard_deviation_proxy",
            "weighted_exact_value_gini_impurity_proxy",
            "positive_weight_bounding_interval_width_proxy",
        )
        actual = (
            self.spread.name,
            self.entropy_like.name,
            self.reachable_set.name,
        )
        if actual != expected:
            raise ValueError("closure proxy names must match their retained basis")
        return self


class ClosureTransitionWindow(_ImmutableClosureSchema):
    """Exact source point indexes, representations, and absolute instants."""

    basis: Literal[
        "forecast_after_shared_horizon_start",
        "all_retained_points",
    ]
    before_point_index: StrictInt = Field(ge=0)
    after_point_index: StrictInt = Field(ge=1)
    from_at: datetime
    to_at: datetime
    from_utc_instant_key: NonEmptyString
    to_utc_instant_key: NonEmptyString

    @field_validator("from_at", "to_at")
    @classmethod
    def times_must_be_aware(cls, value: datetime) -> datetime:
        return _require_timezone(value, "closure transition time")

    @model_validator(mode="after")
    def validate_exact_window(self) -> ClosureTransitionWindow:
        from_key = _utc(self.from_at)
        to_key = _utc(self.to_at)
        if self.after_point_index != self.before_point_index + 1:
            raise ValueError("closure transition point indexes must be adjacent")
        if to_key <= from_key:
            raise ValueError("closure transition window must advance in UTC")
        if (
            self.from_utc_instant_key != str(from_key)
            or self.to_utc_instant_key != str(to_key)
        ):
            raise ValueError("closure UTC instant identity is inconsistent")
        return self


ClassificationBasis = Literal[
    "information_update",
    "known_causal_hard_constraint",
    "absent",
    "ambiguous_or_inconsistent",
]


class ClosureEvent(_ImmutableClosureSchema):
    """One qualifying spread contraction with raw and contextual evidence."""

    event_id: NonEmptyString
    variable: NonEmptyString
    component: ComponentName
    window: ClosureTransitionWindow
    before: SpreadPoint
    after: SpreadPoint
    before_proxies: ClosureProxySnapshot
    after_proxies: ClosureProxySnapshot
    relative_spread_contraction: ClosureProxyMagnitude
    thresholds: ClosureThresholds
    domain: ClosureDomain
    firmness: ClosureFirmness
    transition_style: ClosureTransitionStyle
    classification_basis: ClassificationBasis
    evidence_ids: tuple[NonEmptyString, ...]

    @field_validator("window", mode="before")
    @classmethod
    def revalidate_window(cls, value: object) -> object:
        return _revalidate_model(value, ClosureTransitionWindow)

    @field_validator("before", "after", mode="before")
    @classmethod
    def revalidate_spread_point(cls, value: object) -> object:
        return _revalidate_model(value, SpreadPoint)

    @field_validator("before_proxies", "after_proxies", mode="before")
    @classmethod
    def revalidate_snapshot(cls, value: object) -> object:
        return _revalidate_model(value, ClosureProxySnapshot)

    @field_validator("relative_spread_contraction", mode="before")
    @classmethod
    def revalidate_relative_contraction(cls, value: object) -> object:
        return _revalidate_model(value, ClosureProxyMagnitude)

    @field_validator("thresholds", mode="before")
    @classmethod
    def revalidate_thresholds(cls, value: object) -> object:
        return _revalidate_model(value, ClosureThresholds)

    @model_validator(mode="after")
    def validate_event_arithmetic_and_labels(self) -> ClosureEvent:
        scope = (self.variable, self.component)
        if (self.before.variable, self.before.component) != scope or (
            self.after.variable,
            self.after.component,
        ) != scope:
            raise ValueError("closure event raw evidence must share exact scope")
        if self.before.count != self.after.count:
            raise ValueError("closure event member counts must remain aligned")
        before_at = canonical_model_content(self.before)["at"]
        after_at = canonical_model_content(self.after)["at"]
        window = canonical_model_content(self.window)
        if window["from_at"] != before_at or window["to_at"] != after_at:
            raise ValueError("closure event window must retain raw point times")
        expected_before = _proxy_snapshot(self.before)
        expected_after = _proxy_snapshot(self.after)
        if (
            canonical_model_content(self.before_proxies)
            != canonical_model_content(expected_before)
            or canonical_model_content(self.after_proxies)
            != canonical_model_content(expected_after)
        ):
            raise ValueError("closure proxies must exactly recompute from raw evidence")
        relative = _relative_contraction(self.before, self.after)
        if relative is None:
            raise ValueError("closure event requires strict spread contraction")
        expected_relative = _fraction_magnitude(relative)
        if self.relative_spread_contraction != expected_relative:
            raise ValueError("relative spread contraction is inconsistent")
        if not _meets_threshold(
            relative, self.thresholds.minimum_relative_contraction
        ):
            raise ValueError("closure event does not meet the caller threshold")
        expected_style = _transition_style(relative, self.thresholds)
        if self.transition_style != expected_style:
            raise ValueError("closure transition style is inconsistent")
        expected_labels = {
            "information_update": ("epistemic", "provisional"),
            "known_causal_hard_constraint": ("causal", "hard"),
            "absent": ("unknown", "unknown"),
            "ambiguous_or_inconsistent": ("unknown", "unknown"),
        }[self.classification_basis]
        if (self.domain, self.firmness) != expected_labels:
            raise ValueError("closure domain or firmness lacks matching evidence")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("closure event evidence identities must be unique")
        if self.classification_basis == "absent" and self.evidence_ids:
            raise ValueError("absent closure context cannot cite evidence")
        if self.classification_basis != "absent" and not self.evidence_ids:
            raise ValueError("classified closure context must cite evidence")
        return self


class ClosureAnalysis(_ImmutableClosureSchema):
    """Immutable source-bound closure events with conservative classification."""

    source: SpreadAnalysis
    configuration: ClosureConfiguration
    events: tuple[ClosureEvent, ...]

    @field_validator("source", mode="before")
    @classmethod
    def revalidate_source(cls, value: object) -> object:
        return _revalidate_spread(value)

    @field_validator("configuration", mode="before")
    @classmethod
    def revalidate_configuration(cls, value: object) -> object:
        return _revalidate_model(value, ClosureConfiguration)

    @field_validator("events", mode="before")
    @classmethod
    def revalidate_events(cls, value: object) -> object:
        return _revalidate_records(value, ClosureEvent)

    @model_validator(mode="after")
    def validate_source_bound_recomputation(self) -> ClosureAnalysis:
        _validate_evidence_alignment(self.source, self.configuration)
        expected = _derive_events(self.source, self.configuration)
        if canonical_record_content(self.events) != canonical_record_content(
            expected
        ):
            raise ValueError(
                "closure events must exactly recompute from spread and configuration"
            )
        return self


def _revalidate_provenance(value: object) -> object:
    if isinstance(value, Provenance):
        return Provenance.model_validate(value.model_dump(warnings=False))
    return value


def _revalidate_model(value: object, model: type[BaseModel]) -> object:
    if isinstance(value, model):
        return model.model_validate(value.model_dump(warnings=False))
    return value


def _revalidate_records(value: object, model: type[BaseModel]) -> object:
    if not isinstance(value, (tuple, list)):
        return value
    return tuple(_revalidate_model(item, model) for item in value)


def _revalidate_spread(value: object) -> SpreadAnalysis:
    try:
        if isinstance(value, SpreadAnalysis):
            payload = value.model_dump(warnings=False)
        elif isinstance(value, dict):
            payload = value
        else:
            raise TypeError("source must be a SpreadAnalysis")
        return SpreadAnalysis.model_validate(payload)
    except (OverflowError, TypeError, ValueError, ValidationError) as exc:
        raise ClosureAnalysisError(
            "source must be a deeply valid SpreadAnalysis"
        ) from exc


def _sequence(value: object, role: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ClosureAnalysisError(f"{role} must be a sequence")
    return tuple(value)


def _configuration_from_inputs(
    thresholds: object,
    information_updates: object,
    hard_constraints: object,
) -> ClosureConfiguration:
    try:
        threshold_value = _revalidate_model(thresholds, ClosureThresholds)
        information_values = _sequence(
            information_updates, "information_updates"
        )
        hard_values = _sequence(hard_constraints, "hard_constraints")
        return ClosureConfiguration(
            thresholds=threshold_value,
            information_updates=information_values,
            hard_constraints=hard_values,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, ClosureAnalysisError):
            raise
        raise ClosureAnalysisError(
            f"closure configuration is not valid: {exc}"
        ) from exc


def _source_constraints(source: SpreadAnalysis) -> tuple[str, ...]:
    raw = source.source
    member = (
        raw.trajectories[0]
        if isinstance(raw, TrajectoryEnsemble)
        else raw
    )
    return member.constraints_encountered


def _points_by_scope(
    source: SpreadAnalysis,
) -> dict[tuple[str, str | None], tuple[SpreadPoint, ...]]:
    result: dict[tuple[str, str | None], list[SpreadPoint]] = {
        (pattern.variable, pattern.component): [] for pattern in source.patterns
    }
    for point in source.points:
        scope = (point.variable, point.component)
        if scope not in result:
            raise ClosureAnalysisError(
                "spread points contain an undeclared component scope"
            )
        result[scope].append(point)
    return {key: tuple(values) for key, values in result.items()}


def _validate_evidence_alignment(
    source: SpreadAnalysis,
    configuration: ClosureConfiguration,
) -> None:
    points_by_scope = _points_by_scope(source)
    for evidence in (
        *configuration.information_updates,
        *configuration.hard_constraints,
    ):
        scope = (evidence.variable, evidence.component)
        if scope not in points_by_scope:
            raise ClosureAnalysisError(
                "closure evidence must name an exact spread variable/component scope"
            )
        valid_times = {
            _utc(point.at, ClosureAnalysisError)
            for point in points_by_scope[scope]
        }
        if _utc(evidence.effective_at, ClosureAnalysisError) not in valid_times:
            raise ClosureAnalysisError(
                "closure evidence effective_at must match an exact retained "
                "spread instant"
            )


def _positive_magnitude(value: float) -> ClosureProxyMagnitude:
    if value == 0.0:
        return ClosureProxyMagnitude(kind="structural_zero")
    if not math.isfinite(value) or value < 0.0:
        raise ClosureAnalysisError("closure proxy is not a finite nonnegative value")
    return ClosureProxyMagnitude(kind="finite", value=float(value))


def _gini_impurity_proxy(point: SpreadPoint) -> ClosureProxyMagnitude:
    grouped: dict[float, Fraction] = {}
    for value, weight in zip(point.values, point.weights, strict=True):
        if weight == 0.0:
            continue
        grouped[value] = grouped.get(value, Fraction(0)) + Fraction.from_float(
            weight
        )
    if len(grouped) <= 1:
        return ClosureProxyMagnitude(kind="structural_zero")
    total = sum(grouped.values(), Fraction(0))
    probabilities = tuple(value / total for value in grouped.values())
    exact = Fraction(1) - sum(
        (probability * probability for probability in probabilities),
        Fraction(0),
    )
    if exact == 0:
        return ClosureProxyMagnitude(kind="structural_zero")
    value = float(exact)
    if value == 0.0:
        return ClosureProxyMagnitude(kind="below_float_resolution")
    if not math.isfinite(value):
        return ClosureProxyMagnitude(kind="above_float_range")
    return ClosureProxyMagnitude(kind="finite", value=value)


def _reachable_set_proxy(point: SpreadPoint) -> ClosureProxyMagnitude:
    magnitude = point.spread_range
    if magnitude.kind == "above_float_range":
        return ClosureProxyMagnitude(kind="above_float_range")
    if magnitude.kind == "below_float_resolution":
        return ClosureProxyMagnitude(kind="below_float_resolution")
    if magnitude.value is None:
        raise ClosureAnalysisError("finite spread range lacks a retained value")
    return _positive_magnitude(magnitude.value)


def _proxy_snapshot(point: SpreadPoint) -> ClosureProxySnapshot:
    return ClosureProxySnapshot(
        at=point.at,
        spread=ClosureProxyMetric(
            name="weighted_population_standard_deviation_proxy",
            magnitude=_positive_magnitude(point.population_std),
        ),
        entropy_like=ClosureProxyMetric(
            name="weighted_exact_value_gini_impurity_proxy",
            magnitude=_gini_impurity_proxy(point),
        ),
        reachable_set=ClosureProxyMetric(
            name="positive_weight_bounding_interval_width_proxy",
            magnitude=_reachable_set_proxy(point),
        ),
    )


def _relative_contraction(
    before: SpreadPoint,
    after: SpreadPoint,
) -> Fraction | None:
    if after.population_std >= before.population_std:
        return None
    if before.population_std <= 0.0:
        return None
    before_fraction = Fraction.from_float(before.population_std)
    after_fraction = Fraction.from_float(after.population_std)
    return (before_fraction - after_fraction) / before_fraction


def _fraction_magnitude(value: Fraction) -> ClosureProxyMagnitude:
    if value == 0:
        return ClosureProxyMagnitude(kind="structural_zero")
    converted = float(value)
    if converted == 0.0:
        return ClosureProxyMagnitude(kind="below_float_resolution")
    if not math.isfinite(converted):
        return ClosureProxyMagnitude(kind="above_float_range")
    return ClosureProxyMagnitude(kind="finite", value=converted)


def _meets_threshold(value: Fraction, threshold: float) -> bool:
    return value >= Fraction.from_float(threshold)


def _transition_style(
    contraction: Fraction,
    thresholds: ClosureThresholds,
) -> Literal["gradual", "abrupt"]:
    if _meets_threshold(
        contraction, thresholds.abrupt_relative_contraction
    ):
        return "abrupt"
    return "gradual"


def _matching_evidence(
    before: SpreadPoint,
    after: SpreadPoint,
    configuration: ClosureConfiguration,
) -> tuple[
    tuple[InformationUpdateEvidence, ...],
    tuple[KnownHardConstraintEvidence, ...],
]:
    del before
    scope = (after.variable, after.component)
    effective_key = _utc(after.at, ClosureAnalysisError)
    information = tuple(
        item
        for item in configuration.information_updates
        if (item.variable, item.component) == scope
        and _utc(item.effective_at, ClosureAnalysisError) == effective_key
    )
    hard = tuple(
        item
        for item in configuration.hard_constraints
        if (item.variable, item.component) == scope
        and _utc(item.effective_at, ClosureAnalysisError) == effective_key
    )
    return information, hard


def _classify_context(
    source: SpreadAnalysis,
    before: SpreadPoint,
    after: SpreadPoint,
    configuration: ClosureConfiguration,
) -> tuple[ClosureDomain, ClosureFirmness, ClassificationBasis, tuple[str, ...]]:
    information, hard = _matching_evidence(before, after, configuration)
    evidence_ids = tuple(
        item.evidence_id for item in (*information, *hard)
    )
    if not information and not hard:
        return "unknown", "unknown", "absent", ()
    constraints = _source_constraints(source)
    hard_is_consistent = bool(hard) and all(
        constraints.count(item.constraint_id) == 1 for item in hard
    )
    if information and not hard:
        return (
            "epistemic",
            "provisional",
            "information_update",
            evidence_ids,
        )
    if hard and not information and hard_is_consistent:
        return (
            "causal",
            "hard",
            "known_causal_hard_constraint",
            evidence_ids,
        )
    return (
        "unknown",
        "unknown",
        "ambiguous_or_inconsistent",
        evidence_ids,
    )


def _event_id(
    variable: str,
    component: str | None,
    before_index: int,
    after: SpreadPoint,
) -> str:
    component_identity = component if component is not None else "scalar"
    return (
        f"closure:{variable}:{component_identity}:{before_index}:"
        f"{before_index + 1}:{_utc(after.at, ClosureAnalysisError)}"
    )


def _derive_events(
    source: SpreadAnalysis,
    configuration: ClosureConfiguration,
) -> tuple[ClosureEvent, ...]:
    points_by_scope = _points_by_scope(source)
    pattern_by_scope = {
        (pattern.variable, pattern.component): pattern
        for pattern in source.patterns
    }
    events: list[ClosureEvent] = []
    for scope, points in points_by_scope.items():
        pattern = pattern_by_scope[scope]
        start = pattern.classification_start_index
        for before_index in range(start, len(points) - 1):
            before = points[before_index]
            after = points[before_index + 1]
            contraction = _relative_contraction(before, after)
            if contraction is None or not _meets_threshold(
                contraction,
                configuration.thresholds.minimum_relative_contraction,
            ):
                continue
            domain, firmness, basis, evidence_ids = _classify_context(
                source, before, after, configuration
            )
            events.append(
                ClosureEvent(
                    event_id=_event_id(
                        scope[0], scope[1], before_index, after
                    ),
                    variable=scope[0],
                    component=scope[1],
                    window=ClosureTransitionWindow(
                        basis=source.classification_window.basis,
                        before_point_index=before_index,
                        after_point_index=before_index + 1,
                        from_at=before.at,
                        to_at=after.at,
                        from_utc_instant_key=str(
                            _utc(before.at, ClosureAnalysisError)
                        ),
                        to_utc_instant_key=str(
                            _utc(after.at, ClosureAnalysisError)
                        ),
                    ),
                    before=before,
                    after=after,
                    before_proxies=_proxy_snapshot(before),
                    after_proxies=_proxy_snapshot(after),
                    relative_spread_contraction=_fraction_magnitude(
                        contraction
                    ),
                    thresholds=configuration.thresholds,
                    domain=domain,
                    firmness=firmness,
                    transition_style=_transition_style(
                        contraction, configuration.thresholds
                    ),
                    classification_basis=basis,
                    evidence_ids=evidence_ids,
                )
            )
    return tuple(events)


def analyze_closure(
    source: SpreadAnalysis,
    *,
    thresholds: ClosureThresholds,
    information_updates: Sequence[InformationUpdateEvidence] = (),
    hard_constraints: Sequence[KnownHardConstraintEvidence] = (),
) -> ClosureAnalysis:
    """Derive thresholded closure events without inferring cause from shape."""
    validated_source = _revalidate_spread(source)
    configuration = _configuration_from_inputs(
        thresholds,
        information_updates,
        hard_constraints,
    )
    _validate_evidence_alignment(validated_source, configuration)
    events = _derive_events(validated_source, configuration)
    return ClosureAnalysis(
        source=validated_source,
        configuration=configuration,
        events=events,
    )


__all__ = [
    "ClosureAnalysis",
    "ClosureAnalysisError",
    "ClosureConfiguration",
    "ClosureEvent",
    "ClosureProxyMagnitude",
    "ClosureProxyMetric",
    "ClosureProxySnapshot",
    "ClosureThresholds",
    "ClosureTransitionWindow",
    "InformationUpdateEvidence",
    "KnownHardConstraintEvidence",
    "analyze_closure",
]
