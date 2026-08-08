"""Closure reports preserve spread evidence without inventing causal claims."""

from datetime import UTC, datetime, timedelta, timezone
import math
import re
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
    ConstraintActivation,
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
    activations: tuple[ConstraintActivation, ...] = (),
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
            constraint_activations=activations,
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


def _scope_collision_spread() -> SpreadAnalysis:
    times = (
        START + timedelta(hours=1),
        START + timedelta(hours=2),
    )
    initial_state = StateEstimate(
        at=START,
        observed={"a:b": (0.0,), "a": (0.0,)},
        latent={},
        uncertainty={},
        boundary={},
    )
    member_rows = (
        (((-4.0,), (-4.0,)), ((-2.0,), (-2.0,))),
        (((4.0,), (4.0,)), ((2.0,), (2.0,))),
    )
    members = tuple(
        Trajectory(
            trajectory_id=f"scope-member-{member_index}",
            model_id="scope-model",
            model_version="1",
            case_id="scope-case",
            sample_seed=member_index,
            rng_scheme="scope-collision-v1",
            initial_state=initial_state,
            horizon=TrajectoryHorizon(start_at=START, end_at=times[-1]),
            points=tuple(
                TrajectoryPoint(
                    at=at,
                    values={"a:b": values[0], "a": values[1]},
                )
                for at, values in zip(times, rows, strict=True)
            ),
            constraints_encountered=(),
            provenance=_provenance(f"scope-member:{member_index}"),
        )
        for member_index, rows in enumerate(member_rows)
    )
    source = TrajectoryEnsemble(
        ensemble_id="scope-collision-ensemble",
        model_id="scope-model",
        model_version="1",
        case_id="scope-case",
        trajectories=members,
        seed=23,
        rng_scheme="scope-collision-v1",
        provenance=_provenance("scope-collision-ensemble"),
    )
    return analyze_trajectory_spread(
        source,
        normalization_scales={"a:b": 1.0, "a": 1.0},
        vector_components={"a:b": ("c",), "a": ("b:c",)},
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
    variable: str = "x",
    component: str | None = None,
    modeled_domain: str = "closure modeled domain",
    structurally_hard: bool = True,
    irreversible_within_horizon: bool = True,
    hardness_basis: str = "The modeled boundary cannot be crossed.",
    source_activation_available_at: datetime | None = None,
    source_activation_provenance: Provenance | None = None,
    declaration_observed_at: datetime | None = None,
) -> KnownHardConstraintEvidence:
    activation_provenance = source_activation_provenance or _provenance(
        f"activation:{constraint_id}", observed_at=effective_at
    )
    return KnownHardConstraintEvidence(
        evidence_id=evidence_id,
        constraint_id=constraint_id,
        effective_at=effective_at,
        variable=variable,
        component=component,
        modeled_domain=modeled_domain,
        structurally_hard=structurally_hard,
        irreversible_within_horizon=irreversible_within_horizon,
        hardness_basis=hardness_basis,
        source_activation_available_at=(
            source_activation_available_at or effective_at
        ),
        source_activation_provenance=activation_provenance,
        description="The declared capacity lock became effective.",
        provenance=_provenance(
            f"constraint:{evidence_id}",
            observed_at=declaration_observed_at or effective_at,
        ),
    )


def _activation(
    activated_at: datetime,
    *,
    constraint_id: str = "capacity-lock",
    variable: str = "x",
    component: str | None = None,
    modeled_domain: str = "closure modeled domain",
    structurally_hard: bool = True,
    irreversible_within_horizon: bool = True,
    hardness_basis: str = "The modeled boundary cannot be crossed.",
    available_at: datetime | None = None,
    provenance: Provenance | None = None,
) -> ConstraintActivation:
    source_provenance = provenance or _provenance(
        f"activation:{constraint_id}", observed_at=activated_at
    )
    return ConstraintActivation(
        constraint_id=constraint_id,
        activated_at=activated_at,
        variable=variable,
        component=component,
        modeled_domain=modeled_domain,
        structurally_hard=structurally_hard,
        irreversible_within_horizon=irreversible_within_horizon,
        hardness_basis=hardness_basis,
        available_at=available_at or activated_at,
        provenance=source_provenance,
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
    effective_at = START + timedelta(hours=2)
    activation = _activation(effective_at)
    spread = _spread(
        ((-4.0, 0.0), (4.0, 0.0)),
        constraints=("capacity-lock",),
        activations=(activation,),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        hard_constraints=(
            _hard_constraint(
                effective_at,
                "lock-effective",
                source_activation_provenance=activation.provenance,
            ),
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


def test_matching_constraint_string_without_activation_remains_unknown() -> None:
    """An encountered constraint label alone cannot establish causal hard closure."""
    effective_at = START + timedelta(hours=2)
    spread = _spread(
        ((-4.0, 0.0), (4.0, 0.0)),
        constraints=("capacity-lock",),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        hard_constraints=(
            _hard_constraint(effective_at, "unbound-constraint-label"),
        ),
    )

    event = report.events[0]
    assert (event.domain, event.firmness) == ("unknown", "unknown")
    assert event.classification_basis == "ambiguous_or_inconsistent"


def test_arbitrary_activation_time_joins_only_its_exact_transition() -> None:
    """Point labels cannot substitute for an exact activation instant join."""
    point_times = tuple(
        START + timedelta(hours=index) for index in (1, 2, 3)
    )
    activated_at = START + timedelta(hours=2, minutes=17)
    activation = _activation(activated_at)
    spread = _spread(
        ((-8.0, -4.0, -2.0), (8.0, 4.0, 2.0)),
        point_times=point_times,
        constraints=("capacity-lock",),
        activations=(activation,),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        hard_constraints=(
            _hard_constraint(
                activated_at,
                "arbitrary-time-lock",
                source_activation_provenance=activation.provenance,
            ),
        ),
    )

    assert len(report.events) == 2
    assert (report.events[0].domain, report.events[0].firmness) == (
        "unknown",
        "unknown",
    )
    assert (report.events[1].domain, report.events[1].firmness) == (
        "causal",
        "hard",
    )
    assert report.events[1].evidence_ids == ("arbitrary-time-lock",)


@pytest.mark.parametrize(
    "declared_at",
    [
        START + timedelta(hours=2, minutes=18),
        START + timedelta(hours=3),
    ],
    ids=("adjacent-minute", "transition-end"),
)
def test_hard_evidence_time_must_exactly_match_retained_activation(
    declared_at: datetime,
) -> None:
    """Nearby or endpoint evidence cannot rebind an arbitrary activation time."""
    activated_at = START + timedelta(hours=2, minutes=17)
    activation = _activation(activated_at)
    spread = _spread(
        ((-8.0, -4.0, -2.0), (8.0, 4.0, 2.0)),
        constraints=("capacity-lock",),
        activations=(activation,),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        hard_constraints=(
            _hard_constraint(
                declared_at,
                "wrong-activation-time",
                source_activation_provenance=activation.provenance,
            ),
        ),
    )

    assert all(event.domain == "unknown" for event in report.events)
    assert all(event.firmness == "unknown" for event in report.events)


@pytest.mark.parametrize(
    "late_field",
    ["source_provenance", "source_availability", "declaration_provenance"],
)
def test_thirty_day_late_hard_evidence_is_never_contemporaneous(
    late_field: str,
) -> None:
    """Post-outcome knowledge cannot retroactively create causal hard closure."""
    activated_at = START + timedelta(hours=2)
    late = activated_at + timedelta(days=30)
    source_provenance = _provenance(
        "activation:capacity-lock", observed_at=activated_at
    )
    available_at = activated_at
    declaration_at = START
    if late_field == "source_provenance":
        source_provenance = _provenance(
            "activation:capacity-lock", observed_at=late
        )
        available_at = late
    elif late_field == "source_availability":
        available_at = late
    else:
        declaration_at = late
    activation = _activation(
        activated_at,
        available_at=available_at,
        provenance=source_provenance,
    )
    spread = _spread(
        ((-4.0, 0.0), (4.0, 0.0)),
        constraints=("capacity-lock",),
        activations=(activation,),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        hard_constraints=(
            _hard_constraint(
                activated_at,
                f"late-{late_field}",
                source_activation_available_at=available_at,
                source_activation_provenance=source_provenance,
                declaration_observed_at=declaration_at,
            ),
        ),
    )

    event = report.events[0]
    assert (event.domain, event.firmness) == ("unknown", "unknown")
    assert event.classification_basis == "ambiguous_or_inconsistent"


def test_activation_outside_transition_cannot_explain_later_contraction() -> None:
    """A horizon-start activation is not evidence for a later forecast transition."""
    activation = _activation(START)
    spread = _spread(
        ((-4.0, -2.0), (4.0, 2.0)),
        constraints=("capacity-lock",),
        activations=(activation,),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        hard_constraints=(
            _hard_constraint(
                START,
                "pre-forecast-lock",
                source_activation_provenance=activation.provenance,
            ),
        ),
    )

    assert (report.events[0].domain, report.events[0].firmness) == (
        "unknown",
        "unknown",
    )


@pytest.mark.parametrize(
    ("structurally_hard", "irreversible"),
    [(False, True), (True, False), (False, False)],
)
def test_nonhard_or_reversible_activation_never_gets_hard_classification(
    structurally_hard: bool,
    irreversible: bool,
) -> None:
    """A basis label cannot override explicit hardness or reversibility facts."""
    effective_at = START + timedelta(hours=2)
    activation = _activation(
        effective_at,
        structurally_hard=structurally_hard,
        irreversible_within_horizon=irreversible,
    )
    spread = _spread(
        ((-4.0, 0.0), (4.0, 0.0)),
        constraints=("capacity-lock",),
        activations=(activation,),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        hard_constraints=(
            _hard_constraint(
                effective_at,
                "not-structurally-hard",
                structurally_hard=structurally_hard,
                irreversible_within_horizon=irreversible,
                source_activation_provenance=activation.provenance,
            ),
        ),
    )

    assert (report.events[0].domain, report.events[0].firmness) == (
        "unknown",
        "unknown",
    )


@pytest.mark.parametrize(
    "mismatch",
    ["scope", "domain", "basis", "provenance"],
)
def test_hard_evidence_mismatch_cannot_join_retained_activation(
    mismatch: str,
) -> None:
    """Every source activation join field is authoritative and exact."""
    effective_at = START + timedelta(hours=2)
    activation = _activation(effective_at)
    spread = _spread(
        ((-4.0, 0.0), (4.0, 0.0)),
        constraints=("capacity-lock",),
        activations=(activation,),
    )
    overrides: dict[str, object] = {
        "source_activation_provenance": activation.provenance
    }
    if mismatch == "scope":
        overrides["component"] = "different-component"
    elif mismatch == "domain":
        overrides["modeled_domain"] = "different modeled domain"
    elif mismatch == "basis":
        overrides["hardness_basis"] = "Different irreversibility basis."
    else:
        overrides["source_activation_provenance"] = _provenance(
            "activation:different-source", observed_at=effective_at
        )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        hard_constraints=(
            _hard_constraint(
                effective_at,
                f"mismatch-{mismatch}",
                **overrides,
            ),
        ),
    )

    assert (report.events[0].domain, report.events[0].firmness) == (
        "unknown",
        "unknown",
    )


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


def test_late_information_declaration_cannot_retroactively_narrow_knowledge() -> None:
    """An epistemic label also requires information available by the event."""
    effective_at = START + timedelta(hours=2)
    spread = _spread(((-4.0, 0.0), (4.0, 0.0)))
    evidence = InformationUpdateEvidence(
        evidence_id="late-information",
        effective_at=effective_at,
        variable="x",
        component=None,
        description="The information was declared only after the forecast.",
        provenance=_provenance(
            "information:late",
            observed_at=effective_at + timedelta(days=30),
        ),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        information_updates=(evidence,),
    )

    assert (report.events[0].domain, report.events[0].firmness) == (
        "unknown",
        "unknown",
    )
    assert report.events[0].classification_basis == (
        "ambiguous_or_inconsistent"
    )


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
        effective_at = START + timedelta(hours=2)
        activation = _activation(effective_at)
        spread = _spread(
            ((-4.0, -2.0), (4.0, 2.0)),
            constraints=("capacity-lock",),
            activations=(activation,),
        )
        information = (
            _information(effective_at, "measurement-conflict"),
        )
        constraints = (
            _hard_constraint(
                effective_at,
                "constraint-conflict",
                source_activation_provenance=activation.provenance,
            ),
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


def test_std_contraction_with_bounding_support_expansion_is_not_hard() -> None:
    """A narrower weighted std cannot hide expansion of positive-weight support."""
    effective_at = START + timedelta(hours=2)
    activation = _activation(effective_at)
    spread = _spread(
        ((-1.0, 0.0), (1.0, 100.0), (0.0, 0.0)),
        weights=(0.499995, 0.00001, 0.499995),
        constraints=("capacity-lock",),
        activations=(activation,),
    )
    assert spread.points[1].spread_range.value == 2.0
    assert spread.points[2].spread_range.value == 100.0
    assert spread.points[2].population_std < spread.points[1].population_std

    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        hard_constraints=(
            _hard_constraint(
                effective_at,
                "width-expanded",
                source_activation_provenance=activation.provenance,
            ),
        ),
    )

    assert len(report.events) == 1
    assert (report.events[0].domain, report.events[0].firmness) == (
        "unknown",
        "unknown",
    )
    assert report.events[0].classification_basis == (
        "ambiguous_or_inconsistent"
    )


def test_information_contract_does_not_borrow_hard_support_width_rule() -> None:
    """Information-only narrowing remains epistemic even if support width expands."""
    spread = _spread(
        ((-1.0, 0.0), (1.0, 100.0), (0.0, 0.0)),
        weights=(0.499995, 0.00001, 0.499995),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        information_updates=(
            _information(START + timedelta(hours=2), "weighted-update"),
        ),
    )

    assert (report.events[0].domain, report.events[0].firmness) == (
        "epistemic",
        "provisional",
    )


def test_zero_weight_outlier_does_not_block_proven_support_contraction() -> None:
    """Only positive-weight support determines the hard-closure width condition."""
    maximum = sys.float_info.max
    effective_at = START + timedelta(hours=2)
    activation = _activation(effective_at)
    spread = _spread(
        ((-4.0, -2.0), (4.0, 2.0), (maximum, -maximum)),
        weights=(0.5, 0.5, 0.0),
        constraints=("capacity-lock",),
        activations=(activation,),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        hard_constraints=(
            _hard_constraint(
                effective_at,
                "supported-width-contracts",
                source_activation_provenance=activation.provenance,
            ),
        ),
    )

    event = report.events[0]
    assert event.before.values[-1] == maximum
    assert event.after.values[-1] == -maximum
    assert event.before.spread_range.value == 8.0
    assert event.after.spread_range.value == 4.0
    assert (event.domain, event.firmness) == ("causal", "hard")


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
    activation = _activation(second_fold)
    spread = _spread(
        ((-4.0, -2.0), (4.0, 2.0)),
        start_at=start_at,
        point_times=(first_fold, second_fold),
        constraints=("capacity-lock",),
        activations=(activation,),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        hard_constraints=(
            _hard_constraint(
                second_fold,
                "fold-lock",
                source_activation_provenance=activation.provenance,
            ),
        ),
    )

    window = report.events[0].window
    assert window.from_at.fold == 0
    assert window.to_at.fold == 1
    assert window.from_at.utcoffset() == -timedelta(hours=4)
    assert window.to_at.utcoffset() == -timedelta(hours=5)
    assert window.from_utc_instant_key != window.to_utc_instant_key
    assert (report.events[0].domain, report.events[0].firmness) == (
        "causal",
        "hard",
    )


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
    activation = _activation(point_times[1])
    spread = _spread(
        ((-4.0, -2.0), (4.0, 2.0)),
        start_at=start_at,
        point_times=point_times,
        constraints=("capacity-lock",),
        activations=(activation,),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        hard_constraints=(
            _hard_constraint(
                point_times[1],
                "boundary-year-lock",
                source_activation_provenance=activation.provenance,
            ),
        ),
    )

    event = report.events[0]
    assert event.window.from_at == point_times[0]
    assert event.window.to_at == point_times[1]
    assert int(event.window.to_utc_instant_key) > int(
        event.window.from_utc_instant_key
    )
    assert (event.domain, event.firmness) == ("causal", "hard")


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


def test_event_ids_hash_unambiguous_scope_time_and_source_identity() -> None:
    """Delimiter-equivalent scope strings must produce different event identities."""
    report = analyze_closure(
        _scope_collision_spread(),
        thresholds=_thresholds(),
    )

    assert [(event.variable, event.component) for event in report.events] == [
        ("a:b", "c"),
        ("a", "b:c"),
    ]
    assert len({event.event_id for event in report.events}) == 2
    assert all(
        re.fullmatch(r"closure-sha256:[0-9a-f]{64}", event.event_id)
        for event in report.events
    )


def test_duplicate_constructed_event_id_is_rejected_before_recomputation() -> None:
    """No analysis may contain two records with one canonical event identity."""
    report = analyze_closure(
        _scope_collision_spread(),
        thresholds=_thresholds(),
    )
    duplicate = ClosureEvent.model_construct(
        **{
            **report.events[1].__dict__,
            "event_id": report.events[0].event_id,
        }
    )
    candidate = ClosureAnalysis.model_construct(
        source=report.source,
        configuration=report.configuration,
        events=(report.events[0], duplicate),
    )

    with pytest.raises(ValidationError, match="event identities must be unique"):
        ClosureAnalysis.model_validate(candidate)


@pytest.mark.parametrize(
    "tamper_target",
    [
        "nested_raw_source",
        "retained_spread_point",
        "derived_event",
        "event_id",
    ],
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
    elif tamper_target == "derived_event":
        payload["events"][0]["domain"] = "causal"
    else:
        payload["events"][0]["event_id"] = "closure-sha256:" + "0" * 64

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


def test_serialized_activation_tamper_changes_hard_derivation_and_is_rejected() -> None:
    """A valid-looking activation edit cannot leave an old causal event authoritative."""
    effective_at = START + timedelta(hours=2)
    activation = _activation(effective_at)
    spread = _spread(
        ((-4.0, 0.0), (4.0, 0.0)),
        constraints=("capacity-lock",),
        activations=(activation,),
    )
    report = analyze_closure(
        spread,
        thresholds=_thresholds(),
        hard_constraints=(
            _hard_constraint(
                effective_at,
                "activation-tamper",
                source_activation_provenance=activation.provenance,
            ),
        ),
    )
    payload = report.model_dump(warnings=False)
    for member in payload["source"]["source"]["trajectories"]:
        member["constraint_activations"][0]["structurally_hard"] = False

    with pytest.raises(ValidationError, match="exactly recompute"):
        ClosureAnalysis.model_validate(payload)


def test_constructed_nested_activation_tamper_is_rejected_through_spread_source() -> None:
    """A construct-bypassed activation cannot survive nested closure source validation."""
    effective_at = START + timedelta(hours=2)
    activation = _activation(effective_at)
    spread = _spread(
        ((-4.0, 0.0), (4.0, 0.0)),
        constraints=("capacity-lock",),
        activations=(activation,),
    )
    invalid = ConstraintActivation.model_construct(
        **{**activation.__dict__, "hardness_basis": ""}
    )
    raw_source = spread.source
    assert isinstance(raw_source, TrajectoryEnsemble)
    changed_members = tuple(
        member.model_copy(update={"constraint_activations": (invalid,)})
        for member in raw_source.trajectories
    )
    changed_ensemble = TrajectoryEnsemble.model_construct(
        **{**raw_source.__dict__, "trajectories": changed_members}
    )
    changed_spread = SpreadAnalysis.model_construct(
        **{**spread.__dict__, "source": changed_ensemble}
    )

    with pytest.raises(ClosureAnalysisError, match="deeply valid"):
        analyze_closure(changed_spread, thresholds=_thresholds())
