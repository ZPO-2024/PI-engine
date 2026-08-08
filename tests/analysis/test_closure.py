"""Closure reports preserve spread evidence without inventing causal claims."""

from datetime import UTC, datetime, timedelta, timezone
import math
import sys
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from pi_engine.analysis.closure import (
    ClosureAnalysis,
    ClosureAnalysisError,
    ClosureEvent,
    ClosureThresholds,
    InformationUpdateEvidence,
    KnownHardConstraintEvidence,
    analyze_closure,
)
from pi_engine.analysis.divergence import SpreadAnalysis, analyze_trajectory_spread
from pi_engine.schemas.common import Provenance
from pi_engine.schemas.state import StateEstimate
from pi_engine.schemas.trajectory import (
    ScenarioWeight,
    Trajectory,
    TrajectoryEnsemble,
    TrajectoryHorizon,
    TrajectoryPoint,
)


START = datetime(2026, 8, 8, 12, tzinfo=UTC)


def _provenance(reference: str, *, observed_at: datetime = START) -> Provenance:
    return Provenance(
        source="Task 12 controlled evidence",
        observed_at=observed_at,
        reference=reference,
    )


def _spread(
    member_values: tuple[tuple[float, ...], ...],
    *,
    point_times: tuple[datetime, ...] | None = None,
    constraints: tuple[str, ...] = (),
    start_at: datetime = START,
    weights: tuple[float, ...] | None = None,
    weight_kind: str = "probability",
    normalization_scale: float = 1.0,
) -> SpreadAnalysis:
    times = point_times or tuple(
        start_at + timedelta(hours=index)
        for index in range(1, len(member_values[0]) + 1)
    )
    if weights is not None and len(weights) != len(member_values):
        raise ValueError("test weights must align with members")
    initial_state = StateEstimate(
        at=start_at,
        observed={"x": 0.0},
        latent={},
        uncertainty={},
        boundary={},
    )
    members = tuple(
        Trajectory(
            trajectory_id=f"closure-member-{member_index}",
            model_id="closure-model",
            model_version="1",
            case_id="closure-case",
            sample_seed=member_index,
            rng_scheme="closure-test-stream-v1",
            initial_state=initial_state,
            horizon=TrajectoryHorizon(start_at=start_at, end_at=times[-1]),
            points=tuple(
                TrajectoryPoint(at=at, values={"x": value})
                for at, value in zip(times, values, strict=True)
            ),
            scenario_weight=(
                ScenarioWeight(
                    kind=weight_kind,
                    value=weights[member_index],
                    justification="Task 12 explicit test weighting",
                )
                if weights is not None
                else None
            ),
            constraints_encountered=constraints,
            provenance=_provenance(
                f"trajectory:{member_index}", observed_at=start_at
            ),
        )
        for member_index, values in enumerate(member_values)
    )
    ensemble = TrajectoryEnsemble(
        ensemble_id="closure-ensemble",
        model_id="closure-model",
        model_version="1",
        case_id="closure-case",
        trajectories=members,
        seed=17,
        rng_scheme="closure-test-stream-v1",
        provenance=_provenance("ensemble:closure", observed_at=start_at),
    )
    return analyze_trajectory_spread(
        ensemble,
        normalization_scales={"x": normalization_scale},
    )


def _thresholds() -> ClosureThresholds:
    return ClosureThresholds(
        minimum_relative_contraction=0.25,
        abrupt_relative_contraction=0.75,
    )


def _information(effective_at: datetime, evidence_id: str) -> InformationUpdateEvidence:
    return InformationUpdateEvidence(
        evidence_id=evidence_id,
        effective_at=effective_at,
        variable="x",
        component=None,
        description="A new measurement narrowed the forecast ensemble.",
        provenance=_provenance(f"information:{evidence_id}"),
    )


def _hard_constraint(
    effective_at: datetime,
    evidence_id: str,
    *,
    constraint_id: str = "capacity-lock",
) -> KnownHardConstraintEvidence:
    return KnownHardConstraintEvidence(
        evidence_id=evidence_id,
        constraint_id=constraint_id,
        effective_at=effective_at,
        variable="x",
        component=None,
        description="The declared capacity lock became effective.",
        provenance=_provenance(f"constraint:{evidence_id}"),
    )


def test_gradual_contraction_retains_raw_proxy_and_epistemic_evidence() -> None:
    """Dropping a proxy or mislabeling a measured update must fail this contract."""
    spread = _spread(((-4.0, -2.0, -1.0), (4.0, 2.0, 1.0)))
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        information_updates=(
            _information(START + timedelta(hours=2), "measurement-2"),
            _information(START + timedelta(hours=3), "measurement-3"),
        ),
    )

    assert len(report.events) == 2
    first = report.events[0]
    assert (first.domain, first.firmness, first.transition_style) == (
        "epistemic",
        "provisional",
        "gradual",
    )
    assert first.before.values == (-4.0, 4.0)
    assert first.after.values == (-2.0, 2.0)
    assert first.thresholds == _thresholds()
    assert first.relative_spread_contraction.kind == "finite"
    assert first.relative_spread_contraction.value == 0.5
    assert first.before_proxies.spread.name == (
        "weighted_population_standard_deviation_proxy"
    )
    assert first.before_proxies.entropy_like.name == (
        "weighted_exact_value_gini_impurity_proxy"
    )
    assert first.before_proxies.reachable_set.name == (
        "positive_weight_bounding_interval_width_proxy"
    )
    assert first.window.from_at == START + timedelta(hours=1)
    assert first.window.to_at == START + timedelta(hours=2)
    assert first.evidence_ids == ("measurement-2",)


def test_abrupt_known_retained_hard_constraint_is_causal_and_hard() -> None:
    """Only matching hard-constraint metadata may support causal hard closure."""
    spread = _spread(
        ((-4.0, 0.0), (4.0, 0.0)),
        constraints=("capacity-lock",),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        hard_constraints=(
            _hard_constraint(START + timedelta(hours=2), "lock-effective"),
        ),
    )

    assert len(report.events) == 1
    event = report.events[0]
    assert (event.domain, event.firmness, event.transition_style) == (
        "causal",
        "hard",
        "abrupt",
    )
    assert event.classification_basis == "known_causal_hard_constraint"
    assert event.after_proxies.spread.magnitude.kind == "structural_zero"
    assert event.after_proxies.entropy_like.magnitude.kind == "structural_zero"
    assert event.after_proxies.reachable_set.magnitude.kind == "structural_zero"


def test_information_only_narrowing_is_never_causal_or_hard() -> None:
    """An information update cannot masquerade as a physical constraint."""
    spread = _spread(
        ((-4.0, 0.0), (4.0, 0.0)),
        constraints=("capacity-lock",),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        information_updates=(
            _information(START + timedelta(hours=2), "new-observation"),
        ),
    )

    event = report.events[0]
    assert event.domain == "epistemic"
    assert event.firmness == "provisional"
    assert event.classification_basis == "information_update"
    assert event.domain != "causal"
    assert event.firmness != "hard"


@pytest.mark.parametrize("evidence_mode", ["absent", "conflicting"])
def test_absent_or_conflicting_context_falls_back_to_unknown(
    evidence_mode: str,
) -> None:
    """Contraction alone or conflicting context cannot choose domain or firmness."""
    spread = _spread(
        ((-4.0, -2.0), (4.0, 2.0)),
        constraints=("capacity-lock",),
    )
    information = ()
    constraints = ()
    if evidence_mode == "conflicting":
        information = (
            _information(START + timedelta(hours=2), "measurement-conflict"),
        )
        constraints = (
            _hard_constraint(START + timedelta(hours=2), "constraint-conflict"),
        )

    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        information_updates=information,
        hard_constraints=constraints,
    )

    event = report.events[0]
    assert (event.domain, event.firmness) == ("unknown", "unknown")
    assert event.classification_basis in {"absent", "ambiguous_or_inconsistent"}
    assert event.transition_style == "gradual"


@pytest.mark.parametrize(
    "member_values",
    [
        ((8.0, 4.0, 2.0),),
        ((8.0, 4.0, 2.0), (8.0, 4.0, 2.0)),
    ],
    ids=("ensemble-singleton", "multi-member-zero-spread"),
)
def test_singleton_or_zero_spread_cannot_create_closure(
    member_values: tuple[tuple[float, ...], ...],
) -> None:
    """Path contraction without positive population spread is not closure."""
    report = analyze_closure(_spread(member_values), thresholds=_thresholds())

    assert report.events == ()
    assert all(point.population_std == 0.0 for point in report.source.points)


def test_weighted_spread_drives_and_remains_in_closure_evidence() -> None:
    """Replacing declared probabilities with equal weights must change this result."""
    spread = _spread(
        ((-4.0, -2.0), (4.0, 2.0)),
        weights=(0.75, 0.25),
    )
    report = analyze_closure(spread, thresholds=_thresholds())

    event = report.events[0]
    assert event.before.weighting_policy == "probability"
    assert event.before.weights == (0.75, 0.25)
    assert event.before.population_std == pytest.approx(math.sqrt(12.0))
    assert event.after.population_std == pytest.approx(math.sqrt(3.0))
    assert event.relative_spread_contraction.value == 0.5


def test_zero_weight_outlier_is_auditable_but_cannot_create_closure() -> None:
    """An unsupported member must not fabricate either spread or contraction."""
    maximum = sys.float_info.max
    spread = _spread(
        ((0.0, 0.0), (0.0, 0.0), (maximum, 0.0)),
        weights=(0.5, 0.5, 0.0),
    )
    report = analyze_closure(spread, thresholds=_thresholds())

    retained = report.source.points[1]
    assert retained.values == (0.0, 0.0, maximum)
    assert retained.weights == (0.5, 0.5, 0.0)
    assert retained.population_std == 0.0
    assert report.events == ()


def test_extreme_supported_range_is_structural_not_infinite() -> None:
    """A float-range overflow must remain structural in reachable-set evidence."""
    maximum = sys.float_info.max
    spread = _spread(
        ((-maximum, -maximum / 2.0), (maximum, maximum / 2.0)),
        normalization_scale=maximum,
    )
    report = analyze_closure(spread, thresholds=_thresholds())

    event = report.events[0]
    assert event.before.spread_range.kind == "above_float_range"
    assert event.before_proxies.reachable_set.magnitude.kind == (
        "above_float_range"
    )
    assert event.before_proxies.reachable_set.magnitude.value is None
    assert event.after_proxies.reachable_set.magnitude.kind == "finite"
    assert event.relative_spread_contraction.value == 0.5


def test_subnormal_supported_spread_reaches_structural_zero_without_epsilon() -> None:
    """A real subnormal spread must survive until an exact zero is observed."""
    minimum = float.fromhex("0x0.0000000000001p-1022")
    spread = _spread(((-minimum, 0.0), (minimum, 0.0)))
    report = analyze_closure(spread, thresholds=_thresholds())

    event = report.events[0]
    assert event.before.population_std == minimum
    assert event.before.population_variance.kind == "below_float_resolution"
    assert event.before_proxies.spread.magnitude.value == minimum
    assert event.after_proxies.spread.magnitude.kind == "structural_zero"
    assert event.relative_spread_contraction.value == 1.0
    assert event.transition_style == "abrupt"


def test_same_zone_dst_folds_remain_distinct_closure_window_instants() -> None:
    """Fold-blind datetime equality must not collapse an observed transition."""
    new_york = ZoneInfo("America/New_York")
    start_at = datetime(2026, 11, 1, 0, 30, tzinfo=new_york)
    first_fold = datetime(2026, 11, 1, 1, 30, tzinfo=new_york, fold=0)
    second_fold = datetime(2026, 11, 1, 1, 30, tzinfo=new_york, fold=1)
    spread = _spread(
        ((-4.0, -2.0), (4.0, 2.0)),
        start_at=start_at,
        point_times=(first_fold, second_fold),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        information_updates=(
            InformationUpdateEvidence(
                evidence_id="fold-update",
                effective_at=second_fold,
                variable="x",
                component=None,
                description="Information arrived during the second repeated hour.",
                provenance=_provenance(
                    "information:fold", observed_at=start_at
                ),
            ),
        ),
    )

    window = report.events[0].window
    assert window.from_at.fold == 0
    assert window.to_at.fold == 1
    assert window.from_at.utcoffset() == -timedelta(hours=4)
    assert window.to_at.utcoffset() == -timedelta(hours=5)
    assert window.from_utc_instant_key != window.to_utc_instant_key


@pytest.mark.parametrize(
    ("start_at", "point_times"),
    [
        (
            datetime(1, 1, 1, 0, tzinfo=timezone(timedelta(hours=14))),
            (
                datetime(1, 1, 1, 1, tzinfo=timezone(timedelta(hours=14))),
                datetime(1, 1, 1, 2, tzinfo=timezone(timedelta(hours=14))),
            ),
        ),
        (
            datetime(
                9999,
                12,
                31,
                21,
                tzinfo=timezone(-timedelta(hours=12)),
            ),
            (
                datetime(
                    9999,
                    12,
                    31,
                    22,
                    tzinfo=timezone(-timedelta(hours=12)),
                ),
                datetime(
                    9999,
                    12,
                    31,
                    23,
                    tzinfo=timezone(-timedelta(hours=12)),
                ),
            ),
        ),
    ],
    ids=("year-one-positive-offset", "year-9999-negative-offset"),
)
def test_boundary_year_instants_need_no_constructible_utc_datetime(
    start_at: datetime,
    point_times: tuple[datetime, datetime],
) -> None:
    """Absolute-time identity must work outside datetime's UTC year range."""
    spread = _spread(
        ((-4.0, -2.0), (4.0, 2.0)),
        start_at=start_at,
        point_times=point_times,
    )
    report = analyze_closure(spread, thresholds=_thresholds())

    event = report.events[0]
    assert event.window.from_at == point_times[0]
    assert event.window.to_at == point_times[1]
    assert int(event.window.to_utc_instant_key) > int(
        event.window.from_utc_instant_key
    )


def test_inconsistent_declared_hard_constraint_falls_back_to_unknown() -> None:
    """An unretained constraint identity cannot establish a hard causal label."""
    spread = _spread(
        ((-4.0, 0.0), (4.0, 0.0)),
        constraints=("different-constraint",),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        hard_constraints=(
            _hard_constraint(START + timedelta(hours=2), "unmatched-lock"),
        ),
    )

    event = report.events[0]
    assert (event.domain, event.firmness) == ("unknown", "unknown")
    assert event.classification_basis == "ambiguous_or_inconsistent"
    assert event.evidence_ids == ("unmatched-lock",)


@pytest.mark.parametrize(
    ("effective_at", "variable", "component", "message"),
    [
        (START + timedelta(minutes=90), "x", None, "exact retained spread instant"),
        (START + timedelta(hours=2), "missing", None, "exact spread variable"),
        (START + timedelta(hours=2), "x", "axis", "exact spread variable"),
    ],
    ids=("unaligned-time", "wrong-variable", "wrong-component"),
)
def test_evidence_requires_exact_source_time_and_component_scope(
    effective_at: datetime,
    variable: str,
    component: str | None,
    message: str,
) -> None:
    """Loose time or scope matching would attribute evidence to the wrong event."""
    spread = _spread(((-4.0, -2.0), (4.0, 2.0)))
    evidence = InformationUpdateEvidence(
        evidence_id="misbound-update",
        effective_at=effective_at,
        variable=variable,
        component=component,
        description="This declaration must align exactly.",
        provenance=_provenance("information:misbound"),
    )

    with pytest.raises(ClosureAnalysisError, match=message):
        analyze_closure(
            spread,
            thresholds=_thresholds(),
            information_updates=(evidence,),
        )


def test_duplicate_evidence_identity_is_rejected_across_evidence_kinds() -> None:
    """One audit identity cannot count as two independent explanations."""
    spread = _spread(
        ((-4.0, 0.0), (4.0, 0.0)),
        constraints=("capacity-lock",),
    )
    effective_at = START + timedelta(hours=2)

    with pytest.raises(ClosureAnalysisError, match="identities must be unique"):
        analyze_closure(
            spread,
            thresholds=_thresholds(),
            information_updates=(_information(effective_at, "duplicate"),),
            hard_constraints=(_hard_constraint(effective_at, "duplicate"),),
        )


@pytest.mark.parametrize(
    "invalid",
    [
        {"minimum_relative_contraction": True, "abrupt_relative_contraction": 0.75},
        {"minimum_relative_contraction": 0.25, "abrupt_relative_contraction": 0.2},
        {"minimum_relative_contraction": math.nan, "abrupt_relative_contraction": 0.75},
    ],
    ids=("bool", "abrupt-below-minimum", "nonfinite"),
)
def test_thresholds_are_explicit_strict_finite_and_ordered(
    invalid: dict[str, object],
) -> None:
    """Coercion or implicit numeric tolerance would silently change classification."""
    with pytest.raises(ValidationError):
        ClosureThresholds.model_validate(invalid)


def test_evidence_requires_aware_time_and_deeply_valid_provenance() -> None:
    """Construct-bypassed provenance cannot enter an authoritative context claim."""
    with pytest.raises(ValidationError, match="timezone"):
        _information(datetime(2026, 8, 8, 14), "naive")

    invalid_provenance = Provenance.model_construct(
        source="",
        observed_at=START,
        reference="constructed",
    )
    with pytest.raises(ValidationError, match="at least 1 character"):
        InformationUpdateEvidence(
            evidence_id="invalid-provenance",
            effective_at=START + timedelta(hours=2),
            variable="x",
            component=None,
            description="Invalid nested provenance must be rejected.",
            provenance=invalid_provenance,
        )


def test_closure_analysis_json_round_trip_preserves_source_and_event() -> None:
    """A standards-compliant JSON round trip must retain all closure evidence."""
    spread = _spread(((-4.0, -2.0), (4.0, 2.0)))
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        information_updates=(
            _information(START + timedelta(hours=2), "json-update"),
        ),
    )

    assert ClosureAnalysis.model_validate_json(report.model_dump_json()) == report


@pytest.mark.parametrize(
    "tamper_target",
    ["nested_raw_source", "retained_spread_point", "derived_event"],
)
def test_serialized_source_or_derived_tampering_is_rejected(
    tamper_target: str,
) -> None:
    """Serialized evidence cannot be edited without invalidating recomputation."""
    spread = _spread(((-4.0, -2.0), (4.0, 2.0)))
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        information_updates=(
            _information(START + timedelta(hours=2), "tamper-update"),
        ),
    )
    payload = report.model_dump(warnings=False)
    if tamper_target == "nested_raw_source":
        payload["source"]["source"]["trajectories"][0]["points"][1][
            "values"
        ]["x"] = -1.0
    elif tamper_target == "retained_spread_point":
        payload["source"]["points"][-1]["population_std"] = 1.5
    else:
        payload["events"][0]["domain"] = "causal"

    with pytest.raises((ValidationError, ClosureAnalysisError)):
        ClosureAnalysis.model_validate(payload)


def test_constructed_source_and_derived_tampering_is_deeply_revalidated() -> None:
    """Trusted construction APIs cannot bypass source or event derivation checks."""
    spread = _spread(((-4.0, -2.0), (4.0, 2.0)))
    report = analyze_closure(spread, thresholds=_thresholds())
    changed_point = spread.points[-1].model_copy(
        update={"population_std": 1.5}
    )
    changed_source = SpreadAnalysis.model_construct(
        **{
            **spread.__dict__,
            "points": (*spread.points[:-1], changed_point),
        }
    )
    with pytest.raises(ClosureAnalysisError, match="deeply valid"):
        analyze_closure(changed_source, thresholds=_thresholds())

    changed_event = ClosureEvent.model_construct(
        **{**report.events[0].__dict__, "transition_style": "abrupt"}
    )
    changed_report = ClosureAnalysis.model_construct(
        source=report.source,
        configuration=report.configuration,
        events=(changed_event,),
    )
    with pytest.raises(ValidationError):
        ClosureAnalysis.model_validate(changed_report)
