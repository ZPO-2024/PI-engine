from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from pi_engine.schemas.common import Provenance
from pi_engine.schemas.outcome import ComparisonWindow, Outcome
from pi_engine.schemas.residual import (
    Residual,
    ResidualCategory,
    ResidualClassification,
)
from pi_engine.schemas.state import StateEstimate
from pi_engine.schemas.trajectory import (
    ScenarioWeight,
    Trajectory,
    TrajectoryEnsemble,
    TrajectoryHorizon,
    TrajectoryPoint,
    summarize_trajectories,
)


START = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
END = START + timedelta(hours=6)


def provenance(source: str = "PI-engine deterministic runner") -> Provenance:
    return Provenance(
        source=source,
        observed_at=START,
        reference="run:river-linear-decay:case-river-1",
    )


def trajectory_payload(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "trajectory_id": "trajectory-river-1",
        "model_id": "river-linear-decay",
        "model_version": "1.2.0",
        "case_id": "case-river-1",
        "initial_state": StateEstimate(
            at=START,
            observed={"flow": 12.5},
            latent={"rainfall": 2.0},
            uncertainty={"flow": 0.4},
            boundary={"rainfall": 2.0},
        ),
        "horizon": TrajectoryHorizon(start_at=START, end_at=END),
        "points": (
            TrajectoryPoint(at=START, values={"flow": 12.5}),
            TrajectoryPoint(at=END, values={"flow": 8.25}),
        ),
        "scenario_weight": ScenarioWeight(
            kind="probability",
            value=0.65,
            justification="normalized branching probabilities from the model",
        ),
        "constraints_encountered": ("flow >= 0",),
        "provenance": provenance(),
    }
    values.update(overrides)
    return values


def outcome_payload(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "outcome_id": "outcome-flow-window-1",
        "case_id": "case-river-1",
        "variable": "flow",
        "unit": "m^3/s",
        "value": 8.0,
        "event_time": END,
        "available_at": END + timedelta(minutes=15),
        "comparison_window": ComparisonWindow(
            start_at=END - timedelta(minutes=30),
            end_at=END + timedelta(minutes=30),
        ),
        "provenance": provenance("USGS held-out stream gauge export"),
    }
    values.update(overrides)
    return values


def residual_payload(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "residual_id": "residual-flow-1",
        "trajectory_id": "trajectory-river-1",
        "prediction_time": END,
        "model_id": "river-linear-decay",
        "model_version": "1.2.0",
        "case_id": "case-river-1",
        "variable": "flow",
        "unit": "m^3/s",
        "predicted_value": 8.25,
        "predicted_distribution_ref": None,
        "observed_outcome": Outcome.model_validate(outcome_payload()),
        "error": 0.25,
        "error_convention": "predicted_minus_observed",
        "classification": ResidualClassification(
            category=ResidualCategory.MODEL_DISCREPANCY,
            basis="persistent signed error outside the declared model envelope",
        ),
        "provenance": provenance("PI-engine holdout comparison"),
    }
    values.update(overrides)
    return values


def test_trajectory_round_trip_preserves_runner_inputs_and_audit_fields() -> None:
    """Dropping runner identity, state, horizon, or constraints breaks reproducibility."""
    trajectory = Trajectory.model_validate(trajectory_payload())
    dumped = trajectory.model_dump(mode="json")

    assert dumped == {
        "trajectory_id": "trajectory-river-1",
        "model_id": "river-linear-decay",
        "model_version": "1.2.0",
        "case_id": "case-river-1",
        "initial_state": {
            "at": "2026-08-08T12:00:00Z",
            "observed": {"flow": 12.5},
            "latent": {"rainfall": 2.0},
            "uncertainty": {"flow": 0.4},
            "boundary": {"rainfall": 2.0},
        },
        "horizon": {
            "start_at": "2026-08-08T12:00:00Z",
            "end_at": "2026-08-08T18:00:00Z",
        },
        "points": [
            {"at": "2026-08-08T12:00:00Z", "values": {"flow": 12.5}},
            {"at": "2026-08-08T18:00:00Z", "values": {"flow": 8.25}},
        ],
        "scenario_weight": {
            "kind": "probability",
            "value": 0.65,
            "justification": "normalized branching probabilities from the model",
        },
        "constraints_encountered": ["flow >= 0"],
        "provenance": {
            "source": "PI-engine deterministic runner",
            "observed_at": "2026-08-08T12:00:00Z",
            "reference": "run:river-linear-decay:case-river-1",
        },
    }
    assert Trajectory.model_validate_json(trajectory.model_dump_json()) == trajectory


def test_trajectory_supports_unweighted_deterministic_scenario() -> None:
    """Requiring a fabricated probability would misrepresent deterministic output."""
    trajectory = Trajectory.model_validate(
        trajectory_payload(scenario_weight=None)
    )

    assert trajectory.scenario_weight is None


@pytest.mark.parametrize(
    ("kind", "value"),
    [("probability", -0.01), ("probability", 1.01), ("relative_weight", -0.01)],
)
def test_scenario_weight_rejects_invalid_numeric_bounds(
    kind: str, value: float
) -> None:
    """Invalid probability or weight bounds would corrupt ensemble semantics."""
    with pytest.raises(ValidationError, match="value"):
        ScenarioWeight(kind=kind, value=value, justification="model branch weight")


def test_trajectory_rejects_misaligned_or_unordered_points() -> None:
    """Points outside or out of order within the horizon make a path ambiguous."""
    with pytest.raises(ValidationError, match="strictly ordered"):
        Trajectory.model_validate(
            trajectory_payload(
                points=(
                    TrajectoryPoint(at=END, values={"flow": 8.25}),
                    TrajectoryPoint(at=START, values={"flow": 12.5}),
                )
            )
        )


def test_trajectory_orders_same_cached_zoneinfo_folds_by_absolute_instant() -> None:
    """Repeated wall time across a DST fold still has an unambiguous order."""
    new_york = ZoneInfo("America/New_York")
    horizon_start = datetime(2026, 11, 1, 0, 30, tzinfo=new_york)
    first_fold = datetime(2026, 11, 1, 1, 30, tzinfo=new_york, fold=0)
    second_fold = datetime(2026, 11, 1, 1, 30, tzinfo=new_york, fold=1)
    horizon_end = datetime(2026, 11, 1, 2, 30, tzinfo=new_york)
    assert TrajectoryHorizon(start_at=first_fold, end_at=second_fold).end_at.fold == 1
    with pytest.raises(ValidationError, match="after start_at"):
        TrajectoryHorizon(start_at=second_fold, end_at=first_fold)
    state = trajectory_payload()["initial_state"].model_copy(
        update={"at": horizon_start}
    )
    horizon = TrajectoryHorizon(start_at=horizon_start, end_at=horizon_end)

    trajectory = Trajectory.model_validate(
        trajectory_payload(
            initial_state=state,
            horizon=horizon,
            points=(
                TrajectoryPoint(at=first_fold, values={"flow": 11.0}),
                TrajectoryPoint(at=second_fold, values={"flow": 10.0}),
                TrajectoryPoint(at=horizon_end, values={"flow": 9.0}),
            ),
        )
    )

    assert trajectory.points[:2] == (
        TrajectoryPoint(at=first_fold, values={"flow": 11.0}),
        TrajectoryPoint(at=second_fold, values={"flow": 10.0}),
    )
    with pytest.raises(ValidationError, match="strictly ordered"):
        Trajectory.model_validate(
            trajectory_payload(
                initial_state=state,
                horizon=horizon,
                points=(
                    TrajectoryPoint(at=second_fold, values={"flow": 10.0}),
                    TrajectoryPoint(at=first_fold, values={"flow": 11.0}),
                    TrajectoryPoint(at=horizon_end, values={"flow": 9.0}),
                ),
            )
        )


@pytest.mark.parametrize(
    ("horizon_start", "horizon_end"),
    [
        (
            datetime(1, 1, 1, 0, tzinfo=timezone(timedelta(hours=14))),
            datetime(1, 1, 1, 1, tzinfo=timezone(timedelta(hours=14))),
        ),
        (
            datetime(9999, 12, 31, 22, tzinfo=timezone(-timedelta(hours=12))),
            datetime(9999, 12, 31, 23, tzinfo=timezone(-timedelta(hours=12))),
        ),
    ],
    ids=("year-1-positive-offset", "year-9999-negative-offset"),
)
def test_trajectory_orders_extreme_offset_instants_without_constructing_utc(
    horizon_start: datetime, horizon_end: datetime
) -> None:
    """Valid boundary-year instants need not be constructible as UTC datetimes."""
    state = trajectory_payload()["initial_state"].model_copy(
        update={"at": horizon_start}
    )

    trajectory = Trajectory.model_validate(
        trajectory_payload(
            initial_state=state,
            horizon=TrajectoryHorizon(
                start_at=horizon_start,
                end_at=horizon_end,
            ),
            points=(
                TrajectoryPoint(at=horizon_end, values={"flow": 8.25}),
            ),
        )
    )

    assert trajectory.points[0].at == horizon_end


def test_trajectory_nested_values_are_deeply_immutable() -> None:
    """Mutating validated state or points would silently change a prediction record."""
    trajectory = Trajectory.model_validate(trajectory_payload())

    with pytest.raises(TypeError):
        trajectory.points[0].values["flow"] = 99.0
    with pytest.raises(TypeError):
        trajectory.initial_state.observed["flow"] = 99.0
    with pytest.raises(ValidationError):
        trajectory.model_version = "2.0.0"


def test_trajectory_rejects_nonfinite_point_and_preconstructed_nested_values() -> None:
    """NaN or construct-bypassed nested objects must not enter a trajectory."""
    invalid_point = TrajectoryPoint.model_construct(
        at=datetime(2026, 8, 8, 13, 0), values={"flow": float("nan")}
    )
    invalid_state = StateEstimate.model_construct(
        at=START,
        observed={"flow": float("inf")},
        latent={},
        uncertainty={},
        boundary={},
    )

    with pytest.raises(ValidationError):
        Trajectory.model_validate(
            trajectory_payload(points=(invalid_point,), initial_state=invalid_state)
        )


def test_trajectory_ensemble_round_trip_retains_raw_samples_and_seed() -> None:
    """Collapsing samples or omitting the seed would prevent stochastic replay."""
    first = Trajectory.model_validate(trajectory_payload())
    second = Trajectory.model_validate(
        trajectory_payload(
            trajectory_id="trajectory-river-2",
            points=(
                TrajectoryPoint(at=START, values={"flow": 12.5}),
                TrajectoryPoint(at=END, values={"flow": 7.75}),
            ),
            scenario_weight=ScenarioWeight(
                kind="probability",
                value=0.35,
                justification="normalized branching probabilities from the model",
            ),
        )
    )
    ensemble = TrajectoryEnsemble(
        ensemble_id="ensemble-river-1",
        model_id="river-linear-decay",
        model_version="1.2.0",
        case_id="case-river-1",
        trajectories=(first, second),
        seed=9173,
        provenance=provenance("PI-engine stochastic runner"),
    )

    assert [item["trajectory_id"] for item in ensemble.model_dump(mode="json")["trajectories"]] == [
        "trajectory-river-1",
        "trajectory-river-2",
    ]
    assert ensemble.seed == 9173
    assert TrajectoryEnsemble.model_validate_json(ensemble.model_dump_json()) == ensemble


def test_scalar_summary_handles_equal_large_finite_values() -> None:
    """A finite equal population must not overflow while computing its mean."""
    trajectories = tuple(
        Trajectory.model_validate(
            trajectory_payload(
                trajectory_id=f"trajectory-large-{index}",
                points=(TrajectoryPoint(at=END, values={"flow": 1e308}),),
                scenario_weight=None,
            )
        )
        for index in range(2)
    )

    summary = summarize_trajectories(trajectories)

    statistics = summary.points[0].statistics["flow"]
    assert statistics.count == 2
    assert statistics.mean == 1e308
    assert statistics.population_variance == 0.0
    assert statistics.population_std == 0.0
    assert statistics.minimum == 1e308
    assert statistics.maximum == 1e308


def test_scalar_summary_rejects_unrepresentable_extreme_variance_deliberately() -> None:
    """An extreme finite spread must raise a domain error, not raw overflow."""
    trajectories = tuple(
        Trajectory.model_validate(
            trajectory_payload(
                trajectory_id=f"trajectory-extreme-{index}",
                points=(TrajectoryPoint(at=END, values={"flow": value}),),
                scenario_weight=None,
            )
        )
        for index, value in enumerate((1e308, -1e308))
    )

    with pytest.raises(
        ValueError, match="population variance.*not representable"
    ):
        summarize_trajectories(trajectories)


@pytest.mark.parametrize("representation", ("serialized", "constructed"))
def test_ensemble_summary_revalidation_rejects_fold_blind_time_tampering(
    representation: str,
) -> None:
    """A derived fold-0 summary point cannot be rebound to fold 1."""
    new_york = ZoneInfo("America/New_York")
    start_at = datetime(2026, 11, 1, 0, 30, tzinfo=new_york)
    first_fold = datetime(2026, 11, 1, 1, 30, tzinfo=new_york, fold=0)
    second_fold = datetime(2026, 11, 1, 1, 30, tzinfo=new_york, fold=1)
    state = trajectory_payload()["initial_state"].model_copy(
        update={"at": start_at}
    )
    trajectories = tuple(
        Trajectory.model_validate(
            trajectory_payload(
                trajectory_id=f"trajectory-fold-summary-{index}",
                initial_state=state,
                horizon=TrajectoryHorizon(
                    start_at=start_at, end_at=second_fold
                ),
                points=(
                    TrajectoryPoint(at=first_fold, values={"flow": value}),
                    TrajectoryPoint(at=second_fold, values={"flow": value - 1.0}),
                ),
                scenario_weight=None,
            )
        )
        for index, value in enumerate((12.0, 10.0))
    )
    summary = summarize_trajectories(trajectories)
    ensemble = TrajectoryEnsemble(
        ensemble_id="ensemble-fold-summary",
        model_id=trajectories[0].model_id,
        model_version=trajectories[0].model_version,
        case_id=trajectories[0].case_id,
        trajectories=trajectories,
        summary=summary,
        provenance=provenance("Task 11 fold summary regression"),
    )
    if representation == "serialized":
        candidate: object = ensemble.model_dump()
        candidate["summary"]["points"][0]["at"] = second_fold
    else:
        tampered_point = type(summary.points[0]).model_construct(
            at=second_fold,
            statistics=summary.points[0].statistics,
        )
        tampered_summary = type(summary).model_construct(
            points=(tampered_point, *summary.points[1:])
        )
        candidate = TrajectoryEnsemble.model_construct(
            ensemble_id=ensemble.ensemble_id,
            model_id=ensemble.model_id,
            model_version=ensemble.model_version,
            case_id=ensemble.case_id,
            trajectories=ensemble.trajectories,
            seed=ensemble.seed,
            rng_scheme=ensemble.rng_scheme,
            summary=tampered_summary,
            provenance=ensemble.provenance,
        )

    with pytest.raises(ValidationError, match="summary must match"):
        TrajectoryEnsemble.model_validate(candidate)


def test_trajectory_ensemble_rejects_mixed_model_or_case_identity() -> None:
    """Mixing unrelated trajectories would make ensemble summaries invalid."""
    mismatched = Trajectory.model_validate(
        trajectory_payload(case_id="case-river-2")
    )

    with pytest.raises(ValidationError, match="identity"):
        TrajectoryEnsemble(
            ensemble_id="ensemble-river-1",
            model_id="river-linear-decay",
            model_version="1.2.0",
            case_id="case-river-1",
            trajectories=(mismatched,),
            seed=9173,
            provenance=provenance("PI-engine stochastic runner"),
        )


@pytest.mark.parametrize(
    "weights",
    [
        (
            ScenarioWeight(
                kind="probability",
                value=0.6,
                justification="normalized model branch probability",
            ),
            None,
        ),
        (
            ScenarioWeight(
                kind="probability",
                value=0.6,
                justification="normalized model branch probability",
            ),
            ScenarioWeight(
                kind="relative_weight",
                value=0.4,
                justification="unnormalized sampling weight",
            ),
        ),
        (
            ScenarioWeight(
                kind="probability",
                value=0.6,
                justification="normalized model branch probability",
            ),
            ScenarioWeight(
                kind="probability",
                value=0.3,
                justification="normalized model branch probability",
            ),
        ),
        (
            ScenarioWeight(
                kind="relative_weight",
                value=0.0,
                justification="unnormalized sampling weight",
            ),
            ScenarioWeight(
                kind="relative_weight",
                value=0.0,
                justification="unnormalized sampling weight",
            ),
        ),
    ],
    ids=("partial", "mixed", "probability-total", "zero-relative-total"),
)
def test_trajectory_ensemble_rejects_incoherent_weight_scheme(
    weights: tuple[ScenarioWeight | None, ScenarioWeight | None],
) -> None:
    """Partial, mixed, unnormalized, or zero-total weights are not one ensemble."""
    trajectories = tuple(
        Trajectory.model_validate(
            trajectory_payload(
                trajectory_id=f"trajectory-river-{index}",
                scenario_weight=weight,
            )
        )
        for index, weight in enumerate(weights, start=1)
    )

    with pytest.raises(ValidationError, match="weight"):
        TrajectoryEnsemble(
            ensemble_id="ensemble-river-1",
            model_id="river-linear-decay",
            model_version="1.2.0",
            case_id="case-river-1",
            trajectories=trajectories,
            seed=9173,
            provenance=provenance("PI-engine stochastic runner"),
        )


@pytest.mark.parametrize("weight_kind", [None, "probability", "relative_weight"])
def test_trajectory_ensemble_accepts_each_coherent_weight_scheme(
    weight_kind: str | None,
) -> None:
    """An ensemble may be unweighted, probabilistic, or relatively weighted."""
    values = (0.4, 0.6) if weight_kind == "probability" else (2.0, 3.0)
    weights = (
        (None, None)
        if weight_kind is None
        else tuple(
            ScenarioWeight(
                kind=weight_kind,
                value=value,
                justification="declared model sampling scheme",
            )
            for value in values
        )
    )
    trajectories = tuple(
        Trajectory.model_validate(
            trajectory_payload(
                trajectory_id=f"trajectory-river-{index}",
                scenario_weight=weight,
            )
        )
        for index, weight in enumerate(weights, start=1)
    )

    ensemble = TrajectoryEnsemble(
        ensemble_id="ensemble-river-1",
        model_id="river-linear-decay",
        model_version="1.2.0",
        case_id="case-river-1",
        trajectories=trajectories,
        seed=9173,
        provenance=provenance("PI-engine stochastic runner"),
    )

    assert ensemble.trajectories == trajectories


def test_trajectory_ensemble_accepts_probability_roundoff_within_tolerance() -> None:
    """Ordinary floating-point roundoff must not invalidate normalized weights."""
    weights = (0.6, 0.4000000005)
    trajectories = tuple(
        Trajectory.model_validate(
            trajectory_payload(
                trajectory_id=f"trajectory-river-{index}",
                scenario_weight=ScenarioWeight(
                    kind="probability",
                    value=value,
                    justification="normalized model branch probability",
                ),
            )
        )
        for index, value in enumerate(weights, start=1)
    )

    ensemble = TrajectoryEnsemble(
        ensemble_id="ensemble-river-1",
        model_id="river-linear-decay",
        model_version="1.2.0",
        case_id="case-river-1",
        trajectories=trajectories,
        seed=9173,
        provenance=provenance("PI-engine stochastic runner"),
    )

    assert ensemble.trajectories == trajectories


def test_trajectory_ensemble_members_share_initial_state() -> None:
    """Samples conditioned on different initial states cannot share an ensemble."""
    baseline = Trajectory.model_validate(
        trajectory_payload(scenario_weight=None)
    )
    changed = Trajectory.model_validate(
        trajectory_payload(
            trajectory_id="trajectory-river-2",
            scenario_weight=None,
            initial_state=StateEstimate(
                at=START,
                observed={"flow": 13.0},
                latent={"rainfall": 2.0},
                uncertainty={"flow": 0.4},
                boundary={"rainfall": 2.0},
            ),
        )
    )

    with pytest.raises(ValidationError, match="initial state"):
        TrajectoryEnsemble(
            ensemble_id="ensemble-river-1",
            model_id="river-linear-decay",
            model_version="1.2.0",
            case_id="case-river-1",
            trajectories=(baseline, changed),
            seed=9173,
            provenance=provenance("PI-engine stochastic runner"),
        )


def test_trajectory_ensemble_members_share_requested_horizon() -> None:
    """Samples over different requested horizons cannot share an ensemble."""
    baseline = Trajectory.model_validate(
        trajectory_payload(scenario_weight=None)
    )
    changed = Trajectory.model_validate(
        trajectory_payload(
            trajectory_id="trajectory-river-2",
            scenario_weight=None,
            horizon=TrajectoryHorizon(
                start_at=START,
                end_at=END + timedelta(hours=1),
            ),
        )
    )

    with pytest.raises(ValidationError, match="horizon"):
        TrajectoryEnsemble(
            ensemble_id="ensemble-river-1",
            model_id="river-linear-decay",
            model_version="1.2.0",
            case_id="case-river-1",
            trajectories=(baseline, changed),
            seed=9173,
            provenance=provenance("PI-engine stochastic runner"),
        )


def test_outcome_round_trip_preserves_withheld_value_timing_window_and_source() -> None:
    """Losing withheld timing, units, or source makes forecast comparison unauditable."""
    outcome = Outcome.model_validate(outcome_payload())

    assert outcome.model_dump(mode="json") == {
        "outcome_id": "outcome-flow-window-1",
        "case_id": "case-river-1",
        "variable": "flow",
        "unit": "m^3/s",
        "value": 8.0,
        "event_time": "2026-08-08T18:00:00Z",
        "available_at": "2026-08-08T18:15:00Z",
        "comparison_window": {
            "start_at": "2026-08-08T17:30:00Z",
            "end_at": "2026-08-08T18:30:00Z",
        },
        "provenance": {
            "source": "USGS held-out stream gauge export",
            "observed_at": "2026-08-08T12:00:00Z",
            "reference": "run:river-linear-decay:case-river-1",
        },
    }
    assert Outcome.model_validate_json(outcome.model_dump_json()) == outcome


def test_outcome_allows_point_comparison_without_fabricated_window() -> None:
    """Point outcomes should not require an inapplicable comparison interval."""
    outcome = Outcome.model_validate(outcome_payload(comparison_window=None))

    assert outcome.comparison_window is None


def test_outcome_rejects_availability_before_the_event() -> None:
    """An outcome cannot be available before the event it records occurs."""
    with pytest.raises(ValidationError, match="available_at"):
        Outcome.model_validate(
            outcome_payload(available_at=END - timedelta(seconds=1))
        )


def test_outcome_comparison_window_must_contain_event_time() -> None:
    """A comparison interval excluding its outcome event compares the wrong period."""
    with pytest.raises(ValidationError, match="contain event_time"):
        Outcome.model_validate(
            outcome_payload(
                comparison_window=ComparisonWindow(
                    start_at=END + timedelta(minutes=1),
                    end_at=END + timedelta(minutes=30),
                )
            )
        )


@pytest.mark.parametrize("field", ["event_time", "available_at"])
def test_outcome_requires_timezone_aware_timing(field: str) -> None:
    """Naive held-out times make cutoff and comparison ordering ambiguous."""
    with pytest.raises(ValidationError, match=field):
        Outcome.model_validate(
            outcome_payload(**{field: datetime(2026, 8, 8, 18, 0)})
        )


def test_outcome_rejects_nonfinite_values_and_nested_validation_bypass() -> None:
    """Non-finite values or invalid constructed provenance must fail structurally."""
    invalid_provenance = Provenance.model_construct(source="", observed_at=START)

    with pytest.raises(ValidationError):
        Outcome.model_validate(
            outcome_payload(value=float("inf"), provenance=invalid_provenance)
        )


def test_residual_rejects_nonnumeric_observed_outcome_value() -> None:
    """A numeric residual cannot embed a categorical observed outcome."""
    categorical_outcome = Outcome.model_validate(
        outcome_payload(value="high flow")
    )

    with pytest.raises(ValidationError, match="finite numeric scalar or vector"):
        Residual.model_validate(
            residual_payload(observed_outcome=categorical_outcome)
        )


@pytest.mark.parametrize(
    "category",
    [
        ResidualCategory.PROCESS_NOISE,
        ResidualCategory.PARAMETER_UNCERTAINTY,
        ResidualCategory.MODEL_DISCREPANCY,
        ResidualCategory.PHASE_TIMING,
        ResidualCategory.TOPOLOGY_COUPLING,
        ResidualCategory.MISSING_VARIABLE,
        ResidualCategory.REGIME_CHANGE,
        ResidualCategory.STRUCTURED_UNKNOWN,
    ],
)
def test_residual_round_trip_preserves_each_distinct_provisional_category(
    category: ResidualCategory,
) -> None:
    """Collapsing distinct residual causes would erase evidence for later analysis."""
    payload = residual_payload(
        classification=ResidualClassification(
            category=category,
            basis=f"provisional evidence for {category.value}",
        )
    )
    residual = Residual.model_validate(payload)

    assert residual.model_dump(mode="json")["classification"] == {
        "category": category.value,
        "basis": f"provisional evidence for {category.value}",
    }
    assert Residual.model_validate_json(residual.model_dump_json()) == residual


def test_residual_round_trip_preserves_trajectory_point_trace() -> None:
    """A residual must retain the trajectory and prediction time it evaluates."""
    residual = Residual.model_validate(residual_payload())
    dumped = residual.model_dump(mode="json")

    assert dumped["trajectory_id"] == "trajectory-river-1"
    assert dumped["prediction_time"] == "2026-08-08T18:00:00Z"
    assert Residual.model_validate_json(residual.model_dump_json()) == residual


def test_residual_requires_prediction_time() -> None:
    """Every residual needs the time coordinate of the evaluated prediction."""
    payload = residual_payload()
    del payload["prediction_time"]

    with pytest.raises(ValidationError) as exc_info:
        Residual.model_validate(payload)

    assert exc_info.value.errors()[0]["loc"] == ("prediction_time",)
    assert exc_info.value.errors()[0]["type"] == "missing"


def test_point_prediction_residual_requires_trajectory_id() -> None:
    """A point prediction must retain the concrete trajectory that produced it."""
    payload = residual_payload()
    del payload["trajectory_id"]

    with pytest.raises(ValidationError, match="trajectory_id"):
        Residual.model_validate(payload)


def test_residual_requires_timezone_aware_prediction_time() -> None:
    """A naive prediction time cannot durably identify a forecast coordinate."""
    with pytest.raises(ValidationError, match="prediction_time"):
        Residual.model_validate(
            residual_payload(prediction_time=datetime(2026, 8, 8, 18, 0))
        )


def test_residual_requires_explicit_classification_instead_of_defaulting_to_noise() -> None:
    """An unexplained residual must never be silently classified as process noise."""
    payload = residual_payload()
    del payload["classification"]

    with pytest.raises(ValidationError) as exc_info:
        Residual.model_validate(payload)

    assert exc_info.value.errors()[0]["loc"] == ("classification",)
    assert exc_info.value.errors()[0]["type"] == "missing"


def test_residual_preserves_distribution_reference_without_fabricated_point_value() -> None:
    """Distribution predictions must remain referentially intact when no point exists."""
    payload = residual_payload(
        predicted_value=None,
        predicted_distribution_ref="ensemble:river-flow:quantiles:v1",
    )
    del payload["trajectory_id"]

    residual = Residual.model_validate(payload)

    assert residual.trajectory_id is None
    assert residual.predicted_value is None
    assert residual.predicted_distribution_ref == "ensemble:river-flow:quantiles:v1"


def test_residual_requires_a_predicted_value_or_distribution_reference() -> None:
    """A residual without a prediction cannot represent forecast error."""
    with pytest.raises(ValidationError, match="predicted value or distribution"):
        Residual.model_validate(
            residual_payload(predicted_value=None, predicted_distribution_ref=None)
        )


@pytest.mark.parametrize(
    ("predicted_value", "observed_value", "error"),
    [
        (8.25, (8.0, 9.0), 0.25),
        ((8.25, 9.25), (8.0, 9.0), 0.25),
        ((8.25, 9.25), (8.0,), (0.25, 0.25)),
    ],
    ids=("scalar-vector", "vector-error-scalar", "vector-length"),
)
def test_residual_rejects_incompatible_numeric_shapes(
    predicted_value: object,
    observed_value: object,
    error: object,
) -> None:
    """Prediction, observation, and error must describe the same numeric shape."""
    observed_outcome = Outcome.model_validate(
        outcome_payload(value=observed_value)
    )

    with pytest.raises(ValidationError, match="shape"):
        Residual.model_validate(
            residual_payload(
                predicted_value=predicted_value,
                observed_outcome=observed_outcome,
                error=error,
            )
        )


def test_residual_accepts_compatible_numeric_vectors() -> None:
    """Vector forecasts retain componentwise observed values and errors."""
    observed_outcome = Outcome.model_validate(
        outcome_payload(value=(8.0, 9.0))
    )
    residual = Residual.model_validate(
        residual_payload(
            predicted_value=(8.25, 9.1),
            observed_outcome=observed_outcome,
            error=(0.25, 0.1),
        )
    )

    assert residual.predicted_value == (8.25, 9.1)
    assert residual.observed_outcome.value == (8.0, 9.0)
    assert residual.error == (0.25, 0.1)


def test_residual_rejects_outcome_identity_mismatch() -> None:
    """Comparing a prediction to another case or variable would corrupt evaluation."""
    other_outcome = Outcome.model_validate(outcome_payload(variable="rainfall", unit="mm/h"))

    with pytest.raises(ValidationError, match="identity"):
        Residual.model_validate(residual_payload(observed_outcome=other_outcome))


def test_residual_revalidates_constructed_outcome_and_classification() -> None:
    """Constructed nested instances must not bypass outcome or classification rules."""
    invalid_outcome = Outcome.model_construct(**outcome_payload(value=float("nan")))
    invalid_classification = ResidualClassification.model_construct(
        category=ResidualCategory.PROCESS_NOISE,
        basis="",
    )

    with pytest.raises(ValidationError):
        Residual.model_validate(
            residual_payload(
                observed_outcome=invalid_outcome,
                classification=invalid_classification,
            )
        )


def test_prediction_schemas_forbid_extra_fields() -> None:
    """Unrecognized fields could conceal behavior or mutable prediction state."""
    samples = (
        (TrajectoryHorizon, {"start_at": START, "end_at": END}),
        (TrajectoryPoint, {"at": START, "values": {"flow": 12.5}}),
        (
            ScenarioWeight,
            {
                "kind": "probability",
                "value": 0.65,
                "justification": "model branch probability",
            },
        ),
        (Trajectory, trajectory_payload()),
        (
            TrajectoryEnsemble,
            {
                "ensemble_id": "ensemble-river-1",
                "model_id": "river-linear-decay",
                "model_version": "1.2.0",
                "case_id": "case-river-1",
                "trajectories": (Trajectory.model_validate(trajectory_payload()),),
                "seed": 9173,
                "provenance": provenance("PI-engine stochastic runner"),
            },
        ),
        (
            ComparisonWindow,
            {"start_at": END - timedelta(minutes=30), "end_at": END},
        ),
        (Outcome, outcome_payload()),
        (
            ResidualClassification,
            {"category": "structured_unknown", "basis": "unexplained structure"},
        ),
        (Residual, residual_payload()),
    )

    for schema, payload in samples:
        with pytest.raises(ValidationError) as exc_info:
            schema.model_validate({**payload, "analysis_hook": "run_me"})
        assert exc_info.value.errors()[0]["loc"] == ("analysis_hook",)
        assert exc_info.value.errors()[0]["type"] == "extra_forbidden"
