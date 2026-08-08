"""Trajectory analysis reports retain the exact geometry they describe."""

from datetime import UTC, datetime, timedelta, timezone
import math
import sys
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from pi_engine.analysis.convergence import (
    ConvergenceAnalysis,
    ConvergenceAnalysisError,
    analyze_trajectory_convergence,
)
from pi_engine.analysis.divergence import (
    SpreadAnalysis,
    SpreadAnalysisError,
    analyze_trajectory_spread,
)
from pi_engine.analysis.sensitivity import (
    LocalSensitivityAnalysis,
    LocalSensitivityError,
    ParameterPerturbation,
    analyze_local_sensitivity,
)
from pi_engine.schemas.common import Provenance
from pi_engine.schemas.state import StateEstimate
from pi_engine.schemas.trajectory import (
    ScenarioWeight,
    Trajectory,
    TrajectoryEnsemble,
    TrajectoryHorizon,
    TrajectoryPoint,
)
from pi_engine.simulation.runner import simulate_deterministic
from pi_engine.simulation.stochastic import simulate_stochastic
from pi_engine.synthetic.controls import (
    no_paired_structure_control,
    random_graph_control,
    shuffled_time_series_control,
)
from pi_engine.synthetic.systems import (
    deterministic_divergence,
    hierarchical_nested_dynamics,
    linear_convergence,
    stochastic_branching,
)


def _copy_with_values(
    source: Trajectory,
    values: tuple[dict[str, object], ...],
    *,
    trajectory_id: str | None = None,
    point_times: tuple[datetime, ...] | None = None,
    **updates: object,
) -> Trajectory:
    times = point_times or tuple(point.at for point in source.points)
    points = tuple(
        point.model_copy(update={"at": at, "values": point_values})
        for point, at, point_values in zip(
            source.points, times, values, strict=True
        )
    )
    return source.model_copy(
        update={
            "trajectory_id": trajectory_id or f"{source.trajectory_id}-copy",
            "points": points,
            **updates,
        }
    )


def _control_trajectory(
    fixture: object,
    point_values: tuple[dict[str, object], ...],
    *,
    stochastic: bool,
) -> Trajectory:
    case = fixture.case
    model = fixture.model
    points = tuple(
        TrajectoryPoint(
            at=case.prediction_cutoff + timedelta(hours=index),
            values=values,
        )
        for index, values in enumerate(point_values, start=1)
    )
    return Trajectory(
        trajectory_id=f"trajectory-control-{fixture.name}",
        model_id=model.model_id,
        model_version=model.version,
        case_id=case.case_id,
        sample_seed=fixture.seed if stochastic else None,
        rng_scheme="synthetic-control-stream-v1" if stochastic else None,
        initial_state=case.state,
        horizon=TrajectoryHorizon(
            start_at=case.prediction_cutoff,
            end_at=points[-1].at,
        ),
        points=points,
        constraints_encountered=case.constraints,
        provenance=Provenance(
            source="Task 11 known control path",
            observed_at=case.prediction_cutoff,
            reference=f"control:{fixture.name}",
        ),
    )


def _time_edge_trajectory(
    *,
    trajectory_id: str,
    start_at: datetime,
    point_times: tuple[datetime, ...],
    point_values: tuple[float, ...] | None = None,
) -> Trajectory:
    values = point_values or tuple(
        float(index) for index in range(1, len(point_times) + 1)
    )
    return Trajectory(
        trajectory_id=trajectory_id,
        model_id="time-edge-model",
        model_version="1",
        case_id="time-edge-case",
        initial_state=StateEstimate(
            at=start_at,
            observed={"x": 0.0},
            latent={},
            uncertainty={},
            boundary={},
        ),
        horizon=TrajectoryHorizon(start_at=start_at, end_at=point_times[-1]),
        points=tuple(
            TrajectoryPoint(at=at, values={"x": value})
            for at, value in zip(point_times, values, strict=True)
        ),
        constraints_encountered=(),
        provenance=Provenance(
            source="Task 11 time edge regression",
            observed_at=start_at,
            reference=f"time-edge:{trajectory_id}",
        ),
    )


def test_linear_and_divergent_paths_retain_literal_ordered_distance_evidence() -> None:
    """Wrong pair order or cross-variable aggregation would change these literals."""
    convergent_fixture = linear_convergence()
    convergent = simulate_deterministic(
        convergent_fixture.case, convergent_fixture.model, horizon=3
    )
    divergent_fixture = deterministic_divergence()
    divergent = simulate_deterministic(
        divergent_fixture.case, divergent_fixture.model, horizon=3
    )

    contraction = analyze_trajectory_convergence(
        convergent, normalization_scales={"x": 2.0}
    )
    expansion = analyze_trajectory_convergence(
        divergent, normalization_scales={"x": 1.0}
    )

    assert contraction.source == convergent
    assert contraction.normalization[0].model_dump() == {
        "variable": "x",
        "scale": 2.0,
        "components": (),
    }
    assert [
        evidence.normalized_distance
        for pair in contraction.pairs
        for evidence in pair.distances
    ] == [0.5, 0.25, 0.125]
    assert [
        (pair.from_at, pair.to_at) for pair in contraction.pairs
    ] == [
        (convergent.horizon.start_at, convergent.points[0].at),
        (convergent.points[0].at, convergent.points[1].at),
        (convergent.points[1].at, convergent.points[2].at),
    ]
    assert contraction.patterns[0].observed_pattern == (
        "strictly_contracting_step_distance"
    )
    assert [
        evidence.normalized_distance
        for pair in expansion.pairs
        for evidence in pair.distances
    ] == [1.0, 2.0, 4.0]
    assert expansion.patterns[0].observed_pattern == (
        "strictly_expanding_step_distance"
    )
    assert "master_score" not in contraction.model_dump()


def test_convergence_report_is_immutable_and_json_round_trippable() -> None:
    """A copied report must preserve its source and derived geometry exactly."""
    fixture = linear_convergence()
    trajectory = simulate_deterministic(fixture.case, fixture.model, horizon=3)

    report = analyze_trajectory_convergence(
        trajectory, normalization_scales={"x": 2.0}
    )

    assert ConvergenceAnalysis.model_validate_json(report.model_dump_json()) == report
    with pytest.raises(ValidationError, match="frozen"):
        report.pairs = ()  # type: ignore[misc]


def test_expanding_ensemble_spread_retains_raw_stable_per_time_evidence() -> None:
    """Spread growth must remain visible without collapsing it into one score."""
    fixture = stochastic_branching(seed=7)
    source = simulate_stochastic(
        fixture.case, fixture.model, horizon=3, samples=4, seed=7
    )
    paths = (
        ((-1.0, -2.0, -4.0)),
        ((1.0, 2.0, 4.0)),
        ((-1.0, -2.0, -4.0)),
        ((1.0, 2.0, 4.0)),
    )
    members = tuple(
        _copy_with_values(
            member,
            tuple({"x": value} for value in path),
            trajectory_id=member.trajectory_id,
        )
        for member, path in zip(source.trajectories, paths, strict=True)
    )
    ensemble = source.model_copy(
        update={"trajectories": members, "summary": None}
    )

    report = analyze_trajectory_spread(
        ensemble, normalization_scales={"x": 1.0}
    )

    assert report.source == TrajectoryEnsemble.model_validate(
        ensemble.model_dump(warnings=False)
    )
    assert report.source_kind == "ensemble"
    assert report.member_count == 4
    evidence = [item for item in report.points if item.variable == "x"]
    assert [item.count for item in evidence] == [4, 4, 4, 4]
    assert [item.values for item in evidence] == [
        (0.0, 0.0, 0.0, 0.0),
        (-1.0, 1.0, -1.0, 1.0),
        (-2.0, 2.0, -2.0, 2.0),
        (-4.0, 4.0, -4.0, 4.0),
    ]
    assert [item.mean for item in evidence] == [0.0, 0.0, 0.0, 0.0]
    assert [item.population_std for item in evidence] == [0.0, 1.0, 2.0, 4.0]
    assert [item.spread_range.value for item in evidence] == [0.0, 2.0, 4.0, 8.0]
    assert [item.normalized_population_std for item in evidence] == [
        0.0,
        1.0,
        2.0,
        4.0,
    ]
    assert report.patterns[0].observed_pattern == "strictly_expanding_spread"
    assert SpreadAnalysis.model_validate_json(report.model_dump_json()) == report
    assert "master_score" not in report.model_dump()


def test_deterministic_path_is_a_singleton_not_evidence_of_ensemble_spread() -> None:
    """A growing path is not the same geometry as dispersion among members."""
    fixture = deterministic_divergence()
    trajectory = simulate_deterministic(fixture.case, fixture.model, horizon=3)

    report = analyze_trajectory_spread(
        trajectory, normalization_scales={"x": 1.0}
    )

    assert report.source_kind == "deterministic_singleton"
    assert report.member_count == 1
    assert [item.population_std for item in report.points] == [0.0] * 4
    assert report.patterns[0].observed_pattern == (
        "deterministic_singleton_no_spread"
    )


def test_one_at_a_time_sensitivity_retains_literal_per_time_slopes() -> None:
    """Parameter deltas and per-variable slopes must remain separately inspectable."""
    fixture = linear_convergence()
    baseline = simulate_deterministic(fixture.case, fixture.model, horizon=3)
    perturbed = _copy_with_values(
        baseline,
        ({"x": 1.2}, {"x": 0.7}, {"x": 0.45}),
        trajectory_id=f"{baseline.trajectory_id}-multiplier-plus-point-one",
    )
    perturbation = ParameterPerturbation(
        parameter_name="multiplier",
        delta=0.1,
        trajectory=perturbed,
    )

    report = analyze_local_sensitivity(
        baseline,
        (perturbation,),
        normalization_scales={"x": 2.0},
    )

    assert report.baseline == baseline
    assert report.perturbations == (perturbation,)
    assert report.analysis_kind == "deterministic_oat"
    assert [item.raw_difference for item in report.point_slopes] == pytest.approx(
        [0.2, 0.2, 0.2]
    )
    assert [item.slope for item in report.point_slopes] == pytest.approx(
        [2.0, 2.0, 2.0]
    )
    assert [item.normalized_slope for item in report.point_slopes] == pytest.approx(
        [1.0, 1.0, 1.0]
    )
    summary = report.summaries[0]
    assert (
        summary.parameter_name,
        summary.variable,
        summary.component,
        summary.point_count,
        summary.mean_absolute_normalized_slope,
        summary.maximum_absolute_normalized_slope,
        summary.final_absolute_normalized_slope,
        summary.observed_pattern,
    ) == (
        "multiplier",
        "x",
        None,
        3,
        pytest.approx(1.0),
        pytest.approx(1.0),
        pytest.approx(1.0),
        "observed_local_response",
    )
    assert LocalSensitivityAnalysis.model_validate_json(
        report.model_dump_json()
    ) == report
    assert "master_score" not in report.model_dump()


def test_negative_controls_do_not_gain_deterministic_convergence_labels() -> None:
    """Gaussian, permuted, and unpaired controls cannot imply deterministic convergence."""
    gaussian = random_graph_control(seed=7)
    gaussian_path = _control_trajectory(
        gaussian,
        (
            {"values": gaussian.outcomes[0].value},
            {"values": (0.5, -0.5, 1.0, -1.0, 0.25)},
        ),
        stochastic=True,
    )
    shuffled = shuffled_time_series_control(seed=7)
    shuffled_path = _control_trajectory(
        shuffled,
        (
            {"signal": 7.0, "draw_index": 7.0},
            {"signal": 1.0, "draw_index": 8.0},
            {"signal": 6.0, "draw_index": 9.0},
        ),
        stochastic=True,
    )
    unpaired = no_paired_structure_control(seed=7)
    unpaired_path = _control_trajectory(
        unpaired,
        ({"values": unpaired.outcomes[0].value},),
        stochastic=False,
    )

    reports = (
        analyze_trajectory_convergence(
            gaussian_path,
            normalization_scales={"values": 1.0},
            vector_components={
                "values": ("node-0", "node-1", "node-2", "node-3", "node-4")
            },
        ),
        analyze_trajectory_convergence(
            shuffled_path,
            normalization_scales={"signal": 1.0, "draw_index": 1.0},
        ),
        analyze_trajectory_convergence(
            unpaired_path,
            normalization_scales={"values": 1.0},
            vector_components={
                "values": ("node-0", "node-1", "node-2", "node-3", "node-4")
            },
        ),
    )

    assert reports[0].trajectory_kind == "stochastic_sample"
    assert reports[1].trajectory_kind == "stochastic_sample"
    assert reports[2].trajectory_kind == "deterministic"
    assert all(
        "deterministic_convergence" not in report.model_dump(mode="json").values()
        for report in reports
    )
    assert all(
        pattern.observed_pattern == "insufficient_step_pairs"
        for pattern in reports[2].patterns
    )


def test_unpaired_stochastic_runs_are_rejected_as_local_sensitivity() -> None:
    """Independent noise streams cannot be treated as a paired parameter response."""
    fixture = stochastic_branching(seed=7)
    ensemble = simulate_stochastic(
        fixture.case, fixture.model, horizon=3, samples=2, seed=7
    )
    perturbation = ParameterPerturbation(
        parameter_name="up_probability",
        delta=0.1,
        trajectory=ensemble.trajectories[1],
    )

    with pytest.raises(LocalSensitivityError, match="paired sample seed"):
        analyze_local_sensitivity(
            ensemble.trajectories[0],
            (perturbation,),
            normalization_scales={"x": 1.0},
        )


def test_vector_analysis_requires_explicit_unique_component_semantics() -> None:
    """Flattening a vector would invent component identities and summaries."""
    fixture = hierarchical_nested_dynamics()
    trajectory = simulate_deterministic(fixture.case, fixture.model, horizon=2)

    with pytest.raises(ConvergenceAnalysisError, match="component semantics"):
        analyze_trajectory_convergence(
            trajectory, normalization_scales={"levels": 1.0}
        )

    report = analyze_trajectory_convergence(
        trajectory,
        normalization_scales={"levels": 1.0},
        vector_components={"levels": ("root", "middle", "leaf")},
    )
    assert [
        (item.variable, item.component) for item in report.patterns
    ] == [
        ("levels", "root"),
        ("levels", "middle"),
        ("levels", "leaf"),
    ]


@pytest.mark.parametrize(
    "scales",
    [
        {},
        {"x": 0.0},
        {"x": -1.0},
        {"x": True},
        {"x": math.nan},
        {"x": 1.0, "extra": 1.0},
    ],
    ids=("missing", "zero", "negative", "bool", "nonfinite", "extra"),
)
def test_normalized_metrics_require_exact_positive_finite_variable_scales(
    scales: dict[str, object],
) -> None:
    """Implicit defaults or unused scales would make normalized evidence opaque."""
    fixture = linear_convergence()
    trajectory = simulate_deterministic(fixture.case, fixture.model, horizon=3)

    with pytest.raises(ConvergenceAnalysisError, match="normalization scale"):
        analyze_trajectory_convergence(trajectory, normalization_scales=scales)


@pytest.mark.parametrize("delta", [0.0, -0.0, True, math.inf, math.nan])
def test_parameter_delta_must_be_explicit_finite_and_nonzero(delta: object) -> None:
    """Zero, coerced, or nonfinite deltas cannot define a local slope."""
    fixture = linear_convergence()
    trajectory = simulate_deterministic(fixture.case, fixture.model, horizon=1)

    with pytest.raises((ValidationError, ValueError), match="delta"):
        ParameterPerturbation(
            parameter_name="multiplier", delta=delta, trajectory=trajectory
        )


def test_stochastic_sample_spread_is_not_labeled_deterministic_singleton() -> None:
    """One sampled path carries no population spread and no deterministic identity."""
    fixture = stochastic_branching(seed=7)
    ensemble = simulate_stochastic(
        fixture.case, fixture.model, horizon=3, samples=2, seed=7
    )

    report = analyze_trajectory_spread(
        ensemble.trajectories[0], normalization_scales={"x": 1.0}
    )

    assert report.source_kind == "stochastic_sample_singleton"
    assert report.patterns[0].observed_pattern == (
        "stochastic_sample_singleton_no_spread"
    )


def test_extreme_finite_distances_and_spread_keep_structural_overflow() -> None:
    """Finite endpoints remain analyzable when their raw range exceeds float max."""
    fixture = linear_convergence()
    source = simulate_deterministic(fixture.case, fixture.model, horizon=1)
    maximum = sys.float_info.max
    initial = source.initial_state.model_copy(
        update={"observed": {"x": maximum}}
    )
    path = _copy_with_values(
        source,
        ({"x": -maximum},),
        trajectory_id=f"{source.trajectory_id}-extreme",
        initial_state=initial,
    )

    distance = analyze_trajectory_convergence(
        path, normalization_scales={"x": maximum}
    ).pairs[0].distances[0]
    assert distance.absolute_distance.kind == "above_float_range"
    assert distance.absolute_distance.value is None
    assert distance.normalized_distance == 2.0

    stochastic = stochastic_branching(seed=7)
    ensemble = simulate_stochastic(
        stochastic.case, stochastic.model, horizon=1, samples=2, seed=7
    )
    members = tuple(
        _copy_with_values(
            member,
            ({"x": value},),
            trajectory_id=member.trajectory_id,
        )
        for member, value in zip(
            ensemble.trajectories, (-maximum, maximum), strict=True
        )
    )
    extreme_ensemble = ensemble.model_copy(
        update={"trajectories": members, "summary": None}
    )

    spread = analyze_trajectory_spread(
        extreme_ensemble, normalization_scales={"x": maximum}
    ).points[-1]
    assert spread.mean == 0.0
    assert spread.population_std == maximum
    assert spread.population_variance.kind == "above_float_range"
    assert spread.spread_range.kind == "above_float_range"
    assert spread.normalized_population_std == 1.0
    assert spread.normalized_range == 2.0


def test_large_finite_sensitivity_uses_scaled_difference_and_summary_mean() -> None:
    """Repeated maximum-scale finite slopes must not overflow their mean."""
    fixture = linear_convergence()
    baseline_source = simulate_deterministic(
        fixture.case, fixture.model, horizon=3
    )
    baseline = _copy_with_values(
        baseline_source,
        ({"x": 0.0}, {"x": 0.0}, {"x": 0.0}),
        trajectory_id=f"{baseline_source.trajectory_id}-zero-baseline",
    )
    maximum = sys.float_info.max
    perturbed = _copy_with_values(
        baseline,
        ({"x": maximum}, {"x": maximum}, {"x": maximum}),
        trajectory_id=f"{baseline.trajectory_id}-maximum-perturbation",
    )

    report = analyze_local_sensitivity(
        baseline,
        (
            ParameterPerturbation(
                parameter_name="offset", delta=1.0, trajectory=perturbed
            ),
        ),
        normalization_scales={"x": maximum},
    )

    assert [item.slope for item in report.point_slopes] == [maximum] * 3
    assert [item.normalized_slope for item in report.point_slopes] == [1.0] * 3
    assert report.summaries[0].mean_absolute_normalized_slope == 1.0


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"model_id": "other-model"}, "model identity"),
        ({"case_id": "other-case"}, "case identity"),
        ({"points": "misaligned-time"}, "time.*baseline"),
        ({"points": "misaligned-variable"}, "variable"),
        ({"points": "misaligned-shape"}, "shape|component semantics"),
        ({"horizon": "misaligned-horizon"}, "horizon"),
    ],
    ids=("model", "case", "time", "variable", "shape", "horizon"),
)
def test_local_sensitivity_requires_exact_baseline_alignment(
    update: dict[str, object], message: str
) -> None:
    """A slope between non-comparable runs has no one-at-a-time meaning."""
    fixture = linear_convergence()
    baseline = simulate_deterministic(fixture.case, fixture.model, horizon=3)
    actual_update = dict(update)
    marker = actual_update.get("points")
    if marker == "misaligned-time":
        points = list(baseline.points)
        points[1] = points[1].model_copy(
            update={"at": points[1].at + timedelta(minutes=30)}
        )
        actual_update["points"] = tuple(points)
    elif marker == "misaligned-variable":
        points = list(baseline.points)
        points[0] = points[0].model_copy(update={"values": {"y": 1.0}})
        actual_update["points"] = tuple(points)
    elif marker == "misaligned-shape":
        points = list(baseline.points)
        points[0] = points[0].model_copy(update={"values": {"x": (1.0,)}})
        actual_update["points"] = tuple(points)
    if actual_update.get("horizon") == "misaligned-horizon":
        end = baseline.horizon.end_at + timedelta(hours=1)
        actual_update["horizon"] = baseline.horizon.model_copy(
            update={"end_at": end}
        )
        points = list(baseline.points)
        points[-1] = points[-1].model_copy(update={"at": end})
        actual_update["points"] = tuple(points)
    perturbed = baseline.model_copy(
        update={
            "trajectory_id": f"{baseline.trajectory_id}-misaligned",
            **actual_update,
        }
    )

    with pytest.raises(LocalSensitivityError, match=message):
        analyze_local_sensitivity(
            baseline,
            (
                ParameterPerturbation(
                    parameter_name="multiplier",
                    delta=0.1,
                    trajectory=perturbed,
                ),
            ),
            normalization_scales={"x": 1.0},
        )


def test_local_sensitivity_requires_unique_parameter_and_trajectory_identities() -> None:
    """Duplicate runs cannot be counted as distinct OAT evidence."""
    fixture = linear_convergence()
    baseline = simulate_deterministic(fixture.case, fixture.model, horizon=1)
    first = _copy_with_values(
        baseline, ({"x": 1.1},), trajectory_id=f"{baseline.trajectory_id}-one"
    )
    second = _copy_with_values(
        baseline, ({"x": 1.2},), trajectory_id=f"{baseline.trajectory_id}-two"
    )

    with pytest.raises(LocalSensitivityError, match="parameter names.*unique"):
        analyze_local_sensitivity(
            baseline,
            (
                ParameterPerturbation(
                    parameter_name="multiplier", delta=0.1, trajectory=first
                ),
                ParameterPerturbation(
                    parameter_name="multiplier", delta=0.2, trajectory=second
                ),
            ),
            normalization_scales={"x": 1.0},
        )
    with pytest.raises(LocalSensitivityError, match="trajectory identities.*unique"):
        analyze_local_sensitivity(
            baseline,
            (
                ParameterPerturbation(
                    parameter_name="multiplier", delta=0.1, trajectory=baseline
                ),
            ),
            normalization_scales={"x": 1.0},
        )


def test_analysis_aligns_repeated_dst_hour_by_utc_instant() -> None:
    """Fold-distinct local timestamps are two ordered instants, not one wall time."""
    fixture = linear_convergence()
    source = simulate_deterministic(fixture.case, fixture.model, horizon=2)
    start = datetime(2026, 11, 1, 4, 30, tzinfo=UTC)
    first_utc = datetime(2026, 11, 1, 5, 30, tzinfo=UTC)
    second_utc = datetime(2026, 11, 1, 6, 30, tzinfo=UTC)
    baseline = source.model_copy(
        update={
            "trajectory_id": f"{source.trajectory_id}-dst-baseline",
            "initial_state": source.initial_state.model_copy(update={"at": start}),
            "horizon": TrajectoryHorizon(start_at=start, end_at=second_utc),
            "points": (
                source.points[0].model_copy(
                    update={"at": first_utc, "values": {"x": 1.0}}
                ),
                source.points[1].model_copy(
                    update={"at": second_utc, "values": {"x": 0.5}}
                ),
            ),
        }
    )
    fold_zero_zone = ZoneInfo.no_cache("America/New_York")
    fold_one_zone = ZoneInfo.no_cache("America/New_York")
    perturbed = _copy_with_values(
        baseline,
        ({"x": 1.1}, {"x": 0.6}),
        trajectory_id=f"{baseline.trajectory_id}-dst-perturbed",
        point_times=(
            datetime(2026, 11, 1, 1, 30, fold=0, tzinfo=fold_zero_zone),
            datetime(2026, 11, 1, 1, 30, fold=1, tzinfo=fold_one_zone),
        ),
    )

    report = analyze_local_sensitivity(
        baseline,
        (
            ParameterPerturbation(
                parameter_name="multiplier", delta=0.1, trajectory=perturbed
            ),
        ),
        normalization_scales={"x": 1.0},
    )

    assert [item.at.astimezone(UTC) for item in report.point_slopes] == [
        first_utc,
        second_utc,
    ]


def test_gaussian_permutation_and_no_pair_controls_do_not_gain_sensitivity() -> None:
    """Unpaired randomness is rejected and unchanged no-pair geometry has no response."""
    gaussian = random_graph_control(seed=7)
    gaussian_path = _control_trajectory(
        gaussian,
        (
            {"values": gaussian.outcomes[0].value},
            {"values": (0.5, -0.5, 1.0, -1.0, 0.25)},
        ),
        stochastic=True,
    )
    gaussian_other_seed = gaussian_path.model_copy(
        update={
            "trajectory_id": f"{gaussian_path.trajectory_id}-other-seed",
            "sample_seed": gaussian_path.sample_seed + 1,
        }
    )
    with pytest.raises(LocalSensitivityError, match="paired sample seed"):
        analyze_local_sensitivity(
            gaussian_path,
            (
                ParameterPerturbation(
                    parameter_name="mean",
                    delta=0.1,
                    trajectory=gaussian_other_seed,
                ),
            ),
            normalization_scales={"values": 1.0},
            vector_components={
                "values": ("node-0", "node-1", "node-2", "node-3", "node-4")
            },
        )

    permutation = shuffled_time_series_control(seed=7)
    permutation_path = _control_trajectory(
        permutation,
        (
            {"signal": 7.0, "draw_index": 7.0},
            {"signal": 1.0, "draw_index": 8.0},
        ),
        stochastic=True,
    )
    permutation_other_seed = permutation_path.model_copy(
        update={
            "trajectory_id": f"{permutation_path.trajectory_id}-other-seed",
            "sample_seed": permutation_path.sample_seed + 1,
        }
    )
    with pytest.raises(LocalSensitivityError, match="paired sample seed"):
        analyze_local_sensitivity(
            permutation_path,
            (
                ParameterPerturbation(
                    parameter_name="population_size",
                    delta=1.0,
                    trajectory=permutation_other_seed,
                ),
            ),
            normalization_scales={"signal": 1.0, "draw_index": 1.0},
        )

    no_pair = no_paired_structure_control(seed=7)
    no_pair_path = _control_trajectory(
        no_pair, ({"values": no_pair.outcomes[0].value},), stochastic=False
    )
    unchanged = no_pair_path.model_copy(
        update={"trajectory_id": f"{no_pair_path.trajectory_id}-unchanged"}
    )
    report = analyze_local_sensitivity(
        no_pair_path,
        (
            ParameterPerturbation(
                parameter_name="declared_unused_parameter",
                delta=1.0,
                trajectory=unchanged,
            ),
        ),
        normalization_scales={"values": 1.0},
        vector_components={
            "values": ("node-0", "node-1", "node-2", "node-3", "node-4")
        },
    )
    assert all(
        item.observed_pattern == "no_observed_response"
        for item in report.summaries
    )


def test_serialized_reports_reject_derived_tampering_and_duplicates() -> None:
    """Plausible copied evidence cannot detach from the retained raw source."""
    fixture = linear_convergence()
    baseline = simulate_deterministic(fixture.case, fixture.model, horizon=3)
    convergence = analyze_trajectory_convergence(
        baseline, normalization_scales={"x": 2.0}
    )
    payload = convergence.model_dump()
    payload["patterns"][0]["normalized_distances"] = (9.0, 9.0, 9.0)
    payload["patterns"][0]["observed_pattern"] = "constant_step_distance"
    with pytest.raises(ValidationError, match="recompute from its source"):
        ConvergenceAnalysis.model_validate(payload)

    spread = analyze_trajectory_spread(
        baseline, normalization_scales={"x": 2.0}
    )
    spread_payload = spread.model_dump()
    spread_payload["points"] = (
        spread_payload["points"][0],
        spread_payload["points"][0],
        *spread_payload["points"][2:],
    )
    with pytest.raises(ValidationError, match="recompute from its source"):
        SpreadAnalysis.model_validate(spread_payload)

    perturbed = _copy_with_values(
        baseline,
        ({"x": 1.2}, {"x": 0.7}, {"x": 0.45}),
        trajectory_id=f"{baseline.trajectory_id}-serialized-perturbation",
    )
    sensitivity = analyze_local_sensitivity(
        baseline,
        (
            ParameterPerturbation(
                parameter_name="multiplier", delta=0.1, trajectory=perturbed
            ),
        ),
        normalization_scales={"x": 2.0},
    )
    sensitivity_payload = sensitivity.model_dump()
    sensitivity_payload["summaries"][0][
        "mean_absolute_normalized_slope"
    ] = 999.0
    with pytest.raises(ValidationError, match="recompute from its sources"):
        LocalSensitivityAnalysis.model_validate(sensitivity_payload)


def test_nested_constructed_analysis_records_are_deeply_revalidated() -> None:
    """model_construct cannot bypass arithmetic at a report authority boundary."""
    fixture = linear_convergence()
    baseline = simulate_deterministic(fixture.case, fixture.model, horizon=1)
    perturbed = _copy_with_values(
        baseline,
        ({"x": 1.2},),
        trajectory_id=f"{baseline.trajectory_id}-constructed-perturbation",
    )
    report = analyze_local_sensitivity(
        baseline,
        (
            ParameterPerturbation(
                parameter_name="multiplier", delta=0.1, trajectory=perturbed
            ),
        ),
        normalization_scales={"x": 2.0},
    )
    point = report.point_slopes[0]
    point_fields = {
        name: getattr(point, name) for name in type(point).model_fields
    }
    point_fields["raw_difference"] = 999.0
    forged_point = type(point).model_construct(**point_fields)
    report_fields = {
        name: getattr(report, name) for name in type(report).model_fields
    }
    report_fields["point_slopes"] = (forged_point,)
    forged_report = LocalSensitivityAnalysis.model_construct(**report_fields)

    with pytest.raises(ValidationError, match="slope arithmetic"):
        LocalSensitivityAnalysis.model_validate(forged_report)


@pytest.mark.parametrize("invalid", [True, math.nan, math.inf, (1.0, 2.0)])
def test_scalar_analysis_rejects_bool_nonfinite_and_undeclared_vectors(
    invalid: object,
) -> None:
    """Coercion and implicit vector flattening would fabricate numeric evidence."""
    fixture = linear_convergence()
    valid = simulate_deterministic(fixture.case, fixture.model, horizon=1)
    first = valid.points[0]
    point_fields = {
        name: getattr(first, name) for name in type(first).model_fields
    }
    point_fields["values"] = {"x": invalid}
    forged_point = type(first).model_construct(**point_fields)
    trajectory_fields = {
        name: getattr(valid, name) for name in type(valid).model_fields
    }
    trajectory_fields["points"] = (forged_point,)
    forged = type(valid).model_construct(**trajectory_fields)

    with pytest.raises(
        (ConvergenceAnalysisError, ValidationError),
        match="valid Trajectory|component semantics",
    ):
        analyze_trajectory_convergence(
            forged, normalization_scales={"x": 1.0}
        )


def test_nonzero_subnormal_geometry_never_collapses_to_false_zero() -> None:
    """Below-resolution results must fail closed instead of gaining no-spread labels."""
    fixture = linear_convergence()
    deterministic = simulate_deterministic(
        fixture.case, fixture.model, horizon=1
    )
    minimum = math.ulp(0.0)
    zero_initial = deterministic.initial_state.model_copy(
        update={"observed": {"x": 0.0}}
    )
    tiny_path = _copy_with_values(
        deterministic,
        ({"x": minimum},),
        trajectory_id=f"{deterministic.trajectory_id}-subnormal",
        initial_state=zero_initial,
    )
    with pytest.raises(ConvergenceAnalysisError, match="not representable"):
        analyze_trajectory_convergence(
            tiny_path, normalization_scales={"x": sys.float_info.max}
        )

    stochastic = stochastic_branching(seed=7)
    ensemble = simulate_stochastic(
        stochastic.case, stochastic.model, horizon=1, samples=2, seed=7
    )
    members = tuple(
        _copy_with_values(
            member,
            ({"x": value},),
            trajectory_id=member.trajectory_id,
        )
        for member, value in zip(
            ensemble.trajectories, (0.0, minimum), strict=True
        )
    )
    tiny_spread = ensemble.model_copy(
        update={"trajectories": members, "summary": None}
    )
    with pytest.raises(SpreadAnalysisError, match="not representable"):
        analyze_trajectory_spread(
            tiny_spread, normalization_scales={"x": 1.0}
        )

    tiny_perturbation = _copy_with_values(
        tiny_path,
        ({"x": minimum},),
        trajectory_id=f"{tiny_path.trajectory_id}-perturbation",
    )
    baseline = _copy_with_values(
        tiny_path,
        ({"x": 0.0},),
        trajectory_id=f"{tiny_path.trajectory_id}-baseline",
    )
    with pytest.raises(LocalSensitivityError, match="not representable"):
        analyze_local_sensitivity(
            baseline,
            (
                ParameterPerturbation(
                    parameter_name="offset",
                    delta=sys.float_info.max,
                    trajectory=tiny_perturbation,
                ),
            ),
            normalization_scales={"x": 1.0},
        )


def test_direct_finite_subtraction_preserves_constant_unit_steps() -> None:
    """Scaling before subtraction must not turn exact unit steps into a mixed pattern."""
    fixture = linear_convergence()
    source = simulate_deterministic(fixture.case, fixture.model, horizon=3)
    initial = source.initial_state.model_copy(update={"observed": {"x": 6.0}})
    trajectory = _copy_with_values(
        source,
        ({"x": 7.0}, {"x": 8.0}, {"x": 9.0}),
        trajectory_id=f"{source.trajectory_id}-constant-unit-steps",
        initial_state=initial,
    )

    report = analyze_trajectory_convergence(
        trajectory, normalization_scales={"x": 1.0}
    )

    assert report.patterns[0].normalized_distances == (1.0, 1.0, 1.0)
    assert report.patterns[0].observed_pattern == "constant_step_distance"


def test_direct_finite_subtraction_preserves_large_offsets_and_adjacent_floats() -> None:
    """Finite differences are exact float evidence even when operands are large."""
    fixture = linear_convergence()
    source = simulate_deterministic(fixture.case, fixture.model, horizon=1)
    left = 1e16
    right = left + 2.0
    initial = source.initial_state.model_copy(update={"observed": {"x": left}})
    trajectory = _copy_with_values(
        source,
        ({"x": right},),
        trajectory_id=f"{source.trajectory_id}-large-offset",
        initial_state=initial,
    )

    distance = analyze_trajectory_convergence(
        trajectory, normalization_scales={"x": 1.0}
    ).pairs[0].distances[0]
    assert distance.absolute_distance.value == 2.0
    assert distance.normalized_distance == 2.0

    ensemble_fixture = stochastic_branching(seed=7)
    ensemble = simulate_stochastic(
        ensemble_fixture.case,
        ensemble_fixture.model,
        horizon=1,
        samples=2,
        seed=7,
    )
    members = tuple(
        _copy_with_values(
            member,
            ({"x": value},),
            trajectory_id=member.trajectory_id,
        )
        for member, value in zip(
            ensemble.trajectories, (left, right), strict=True
        )
    )
    offset_ensemble = ensemble.model_copy(
        update={"trajectories": members, "summary": None}
    )
    spread = analyze_trajectory_spread(
        offset_ensemble, normalization_scales={"x": 1.0}
    ).points[-1]
    assert spread.spread_range.value == 2.0
    assert spread.normalized_range == 2.0

    baseline = _copy_with_values(
        source,
        ({"x": left},),
        trajectory_id=f"{source.trajectory_id}-offset-baseline",
    )
    perturbed = _copy_with_values(
        baseline,
        ({"x": right},),
        trajectory_id=f"{source.trajectory_id}-offset-perturbed",
    )
    sensitivity = analyze_local_sensitivity(
        baseline,
        (
            ParameterPerturbation(
                parameter_name="offset", delta=1.0, trajectory=perturbed
            ),
        ),
        normalization_scales={"x": 1.0},
    ).point_slopes[0]
    assert sensitivity.raw_difference == 2.0
    assert sensitivity.slope == 2.0

    adjacent = math.nextafter(1.0, math.inf)
    adjacent_perturbed = _copy_with_values(
        baseline,
        ({"x": adjacent},),
        trajectory_id=f"{source.trajectory_id}-adjacent-perturbed",
    )
    adjacent_baseline = _copy_with_values(
        baseline,
        ({"x": 1.0},),
        trajectory_id=f"{source.trajectory_id}-adjacent-baseline",
    )
    adjacent_slope = analyze_local_sensitivity(
        adjacent_baseline,
        (
            ParameterPerturbation(
                parameter_name="adjacent",
                delta=1.0,
                trajectory=adjacent_perturbed,
            ),
        ),
        normalization_scales={"x": 1.0},
    ).point_slopes[0]
    assert adjacent_slope.raw_difference == math.ulp(1.0)


@pytest.mark.parametrize(
    ("values", "expected_mean", "expected_std"),
    [
        ((math.nextafter(1.0, 0.0), 1.0), 1.0, 2.0**-54),
        ((1e16, 1e16 + 2.0), 1e16, 1.0),
    ],
    ids=("adjacent-one", "large-offset"),
)
def test_population_variance_centers_in_translated_coordinates(
    values: tuple[float, float], expected_mean: float, expected_std: float
) -> None:
    """Rounding the raw mean must not inflate two-point population variance."""
    fixture = stochastic_branching(seed=7)
    ensemble = simulate_stochastic(
        fixture.case, fixture.model, horizon=1, samples=2, seed=7
    )
    members = tuple(
        _copy_with_values(
            member,
            ({"x": value},),
            trajectory_id=member.trajectory_id,
        )
        for member, value in zip(ensemble.trajectories, values, strict=True)
    )
    source = ensemble.model_copy(update={"trajectories": members, "summary": None})

    point = analyze_trajectory_spread(
        source, normalization_scales={"x": 1.0}
    ).points[-1]

    assert point.mean == expected_mean
    assert point.population_std == expected_std
    assert point.population_variance.kind == "finite"
    assert point.population_variance.value == expected_std * expected_std


@pytest.mark.parametrize(
    ("kind", "raw_weights", "expected_policy"),
    [
        ("probability", (0.99, 0.01), "probability"),
        ("relative_weight", (99.0, 1.0), "normalized_relative_weight"),
    ],
    ids=("probability", "relative"),
)
def test_spread_applies_and_retains_scenario_weight_policy(
    kind: str,
    raw_weights: tuple[float, float],
    expected_policy: str,
) -> None:
    """Weighted population evidence must not silently become equal-member spread."""
    fixture = stochastic_branching(seed=7)
    ensemble = simulate_stochastic(
        fixture.case, fixture.model, horizon=1, samples=2, seed=7
    )
    members = tuple(
        _copy_with_values(
            member,
            ({"x": value},),
            trajectory_id=member.trajectory_id,
            scenario_weight=ScenarioWeight(
                kind=kind,
                value=weight,
                justification="known Task 11 weighted spread",
            ),
        )
        for member, value, weight in zip(
            ensemble.trajectories,
            (0.0, 10.0),
            raw_weights,
            strict=True,
        )
    )
    weighted = ensemble.model_copy(
        update={"trajectories": members, "summary": None}
    )

    report = analyze_trajectory_spread(
        weighted, normalization_scales={"x": 1.0}
    )
    point = report.points[-1]

    assert report.weighting.policy == expected_policy
    assert report.weighting.weights == pytest.approx((0.99, 0.01))
    assert point.weighting_policy == expected_policy
    assert point.weights == pytest.approx((0.99, 0.01))
    assert point.values == (0.0, 10.0)
    assert point.count == 2
    assert point.mean == pytest.approx(0.1)
    assert point.population_std == pytest.approx(math.sqrt(0.99))
    assert point.population_variance.value == pytest.approx(0.99)


def test_zero_weight_member_is_retained_but_excluded_from_weighted_support() -> None:
    """A zero-weight member remains auditable without changing weighted spread."""
    fixture = stochastic_branching(seed=7)
    ensemble = simulate_stochastic(
        fixture.case, fixture.model, horizon=1, samples=2, seed=7
    )
    maximum = sys.float_info.max
    members = tuple(
        _copy_with_values(
            member,
            ({"x": value},),
            trajectory_id=member.trajectory_id,
            scenario_weight=ScenarioWeight(
                kind="probability",
                value=weight,
                justification="zero-weight support regression",
            ),
        )
        for member, value, weight in zip(
            ensemble.trajectories,
            (maximum, -maximum),
            (1.0, 0.0),
            strict=True,
        )
    )
    weighted = ensemble.model_copy(
        update={"trajectories": members, "summary": None}
    )

    point = analyze_trajectory_spread(
        weighted, normalization_scales={"x": maximum}
    ).points[-1]

    assert point.values == (maximum, -maximum)
    assert point.weights == (1.0, 0.0)
    assert (point.mean, point.minimum, point.maximum) == (
        maximum,
        maximum,
        maximum,
    )
    assert point.population_std == 0.0
    assert point.spread_range.value == 0.0


def test_maximum_relative_weights_normalize_before_weighted_extreme_spread() -> None:
    """Finite relative weights and endpoints must not overflow intermediate sums."""
    fixture = stochastic_branching(seed=7)
    ensemble = simulate_stochastic(
        fixture.case, fixture.model, horizon=1, samples=2, seed=7
    )
    maximum = sys.float_info.max
    members = tuple(
        _copy_with_values(
            member,
            ({"x": value},),
            trajectory_id=member.trajectory_id,
            scenario_weight=ScenarioWeight(
                kind="relative_weight",
                value=maximum,
                justification="equal maximum finite relative weights",
            ),
        )
        for member, value in zip(
            ensemble.trajectories, (-maximum, maximum), strict=True
        )
    )
    weighted = ensemble.model_copy(
        update={"trajectories": members, "summary": None}
    )

    point = analyze_trajectory_spread(
        weighted, normalization_scales={"x": maximum}
    ).points[-1]

    assert point.weights == (0.5, 0.5)
    assert point.mean == 0.0
    assert point.population_std == maximum
    assert point.population_variance.kind == "above_float_range"
    assert point.spread_range.kind == "above_float_range"


def test_positive_relative_weight_underflow_fails_closed() -> None:
    """A positive declared member weight must not silently normalize to zero."""
    fixture = stochastic_branching(seed=7)
    ensemble = simulate_stochastic(
        fixture.case, fixture.model, horizon=1, samples=2, seed=7
    )
    members = tuple(
        member.model_copy(
            update={
                "scenario_weight": ScenarioWeight(
                    kind="relative_weight",
                    value=weight,
                    justification="positive normalization-underflow control",
                )
            }
        )
        for member, weight in zip(
            ensemble.trajectories,
            (sys.float_info.max, math.ulp(0.0)),
            strict=True,
        )
    )
    weighted = ensemble.model_copy(
        update={"trajectories": members, "summary": None}
    )

    with pytest.raises(SpreadAnalysisError, match="weight.*representable"):
        analyze_trajectory_spread(
            weighted, normalization_scales={"x": 1.0}
        )


def test_spread_weighting_and_point_weights_reject_serialized_tampering() -> None:
    """Copied weighting metadata must recompute from retained member weights."""
    fixture = stochastic_branching(seed=7)
    ensemble = simulate_stochastic(
        fixture.case, fixture.model, horizon=1, samples=2, seed=7
    )
    report = analyze_trajectory_spread(
        ensemble, normalization_scales={"x": 1.0}
    )

    policy_payload = report.model_dump()
    policy_payload["weighting"]["policy"] = "probability"
    with pytest.raises(ValidationError, match="recompute from its source"):
        SpreadAnalysis.model_validate(policy_payload)

    point_payload = report.model_dump()
    point_payload["points"][-1]["weights"] = (0.99, 0.01)
    with pytest.raises(ValidationError, match="raw values|recompute from its source"):
        SpreadAnalysis.model_validate(point_payload)


@pytest.mark.parametrize(
    ("paths", "expected_pattern", "expected_window_values"),
    [
        (
            ((-4.0, -2.0, -1.0), (4.0, 2.0, 1.0)),
            "strictly_contracting_spread",
            (4.0, 2.0, 1.0),
        ),
        (
            ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0)),
            "constant_positive_spread",
            (2.0, 2.0, 2.0),
        ),
    ],
    ids=("contracting", "constant-positive"),
)
def test_spread_pattern_classifies_forecast_after_shared_initial_evidence(
    paths: tuple[tuple[float, ...], tuple[float, ...]],
    expected_pattern: str,
    expected_window_values: tuple[float, ...],
) -> None:
    """A retained zero-spread initial state must not force a mixed trend label."""
    fixture = stochastic_branching(seed=7)
    ensemble = simulate_stochastic(
        fixture.case, fixture.model, horizon=3, samples=2, seed=7
    )
    members = tuple(
        _copy_with_values(
            member,
            tuple({"x": value} for value in path),
            trajectory_id=member.trajectory_id,
        )
        for member, path in zip(ensemble.trajectories, paths, strict=True)
    )
    source = ensemble.model_copy(update={"trajectories": members, "summary": None})

    report = analyze_trajectory_spread(
        source, normalization_scales={"x": 1.0}
    )
    pattern = report.patterns[0]

    assert pattern.normalized_population_std == (
        0.0,
        *expected_window_values,
    )
    assert pattern.classified_normalized_population_std == expected_window_values
    assert pattern.observed_pattern == expected_pattern
    assert report.classification_window.basis == (
        "forecast_after_shared_horizon_start"
    )
    assert report.classification_window.initial_evidence_source == (
        "injected_initial_state"
    )
    assert report.classification_window.excluded_horizon_start is True
    assert report.classification_window.start_point_index == 1
    assert report.classification_window.classified_point_count == 3
    assert report.classification_window.start_at == source.trajectories[0].points[0].at
    assert report.classification_window.end_at == source.trajectories[0].points[-1].at


def test_spread_classification_window_rejects_serialized_rebinding() -> None:
    """A copied report cannot silently put shared initial evidence back in the trend."""
    fixture = stochastic_branching(seed=7)
    ensemble = simulate_stochastic(
        fixture.case, fixture.model, horizon=2, samples=2, seed=7
    )
    report = analyze_trajectory_spread(
        ensemble, normalization_scales={"x": 1.0}
    )
    payload = report.model_dump()
    payload["classification_window"]["start_point_index"] = 0
    payload["classification_window"]["excluded_horizon_start"] = False

    with pytest.raises(ValidationError, match="classification"):
        SpreadAnalysis.model_validate(payload)


@pytest.mark.parametrize(
    ("start_at", "point_times"),
    [
        (
            datetime(2026, 11, 1, 0, 30, tzinfo=ZoneInfo("America/New_York")),
            (
                datetime(
                    2026,
                    11,
                    1,
                    1,
                    30,
                    tzinfo=ZoneInfo("America/New_York"),
                    fold=0,
                ),
                datetime(
                    2026,
                    11,
                    1,
                    1,
                    30,
                    tzinfo=ZoneInfo("America/New_York"),
                    fold=1,
                ),
            ),
        ),
        (
            datetime(1, 1, 1, 0, tzinfo=timezone(timedelta(hours=14))),
            (datetime(1, 1, 1, 1, tzinfo=timezone(timedelta(hours=14))),),
        ),
        (
            datetime(9999, 12, 31, 22, tzinfo=timezone(-timedelta(hours=12))),
            (
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
    ids=("same-cache-fold", "year-1-positive-offset", "year-9999-negative-offset"),
)
def test_analysis_orders_valid_boundary_instants_without_utc_construction(
    start_at: datetime, point_times: tuple[datetime, ...]
) -> None:
    """Analysis alignment uses integer instant keys at DST and year boundaries."""
    trajectory = _time_edge_trajectory(
        trajectory_id="trajectory-time-edge",
        start_at=start_at,
        point_times=point_times,
    )

    report = analyze_trajectory_convergence(
        trajectory, normalization_scales={"x": 1.0}
    )

    assert len(report.pairs) == len(point_times)
    assert report.pairs[-1].to_at == point_times[-1]


def test_spread_and_sensitivity_align_same_cached_zoneinfo_folds() -> None:
    """All trajectory analyses share the same absolute-instant alignment rule."""
    new_york = ZoneInfo("America/New_York")
    start_at = datetime(2026, 11, 1, 0, 30, tzinfo=new_york)
    point_times = (
        datetime(2026, 11, 1, 1, 30, tzinfo=new_york, fold=0),
        datetime(2026, 11, 1, 1, 30, tzinfo=new_york, fold=1),
    )
    baseline = _time_edge_trajectory(
        trajectory_id="trajectory-fold-baseline",
        start_at=start_at,
        point_times=point_times,
        point_values=(1.0, 2.0),
    )
    perturbed = _time_edge_trajectory(
        trajectory_id="trajectory-fold-perturbed",
        start_at=start_at,
        point_times=point_times,
        point_values=(2.0, 4.0),
    )
    ensemble = TrajectoryEnsemble(
        ensemble_id="ensemble-fold",
        model_id=baseline.model_id,
        model_version=baseline.model_version,
        case_id=baseline.case_id,
        trajectories=(baseline, perturbed),
        provenance=baseline.provenance,
    )

    spread = analyze_trajectory_spread(
        ensemble, normalization_scales={"x": 1.0}
    )
    sensitivity = analyze_local_sensitivity(
        baseline,
        perturbations=(
            ParameterPerturbation(
                parameter_name="gain",
                delta=1.0,
                trajectory=perturbed,
            ),
        ),
        normalization_scales={"x": 1.0},
    )

    assert [point.at.fold for point in spread.points if point.variable == "x"] == [
        0,
        0,
        1,
    ]
    assert tuple(point.at.fold for point in sensitivity.point_slopes) == (0, 1)


def test_all_analyses_wrap_deep_time_reversal_in_typed_errors() -> None:
    """Construct-bypassed timeline corruption must never leak raw datetime errors."""
    new_york = ZoneInfo("America/New_York")
    start_at = datetime(2026, 11, 1, 0, 30, tzinfo=new_york)
    first_fold = datetime(2026, 11, 1, 1, 30, tzinfo=new_york, fold=0)
    second_fold = datetime(2026, 11, 1, 1, 30, tzinfo=new_york, fold=1)
    valid = _time_edge_trajectory(
        trajectory_id="trajectory-fold-valid",
        start_at=start_at,
        point_times=(first_fold, second_fold),
    )
    invalid = valid.model_copy(update={"points": tuple(reversed(valid.points))})

    with pytest.raises(ConvergenceAnalysisError):
        analyze_trajectory_convergence(invalid, normalization_scales={"x": 1.0})
    with pytest.raises(SpreadAnalysisError):
        analyze_trajectory_spread(invalid, normalization_scales={"x": 1.0})
    with pytest.raises(LocalSensitivityError):
        analyze_local_sensitivity(
            invalid,
            perturbations=(),
            normalization_scales={"x": 1.0},
        )


def test_explicit_horizon_start_is_retained_once_and_excluded_from_label() -> None:
    """An explicit initial sample is evidence, not a duplicate zero-duration step."""
    start_at = datetime(2026, 8, 8, 12, tzinfo=UTC)
    point_times = (start_at, start_at + timedelta(hours=1))
    first = _time_edge_trajectory(
        trajectory_id="trajectory-explicit-start-1",
        start_at=start_at,
        point_times=point_times,
        point_values=(0.0, -2.0),
    )
    second = _time_edge_trajectory(
        trajectory_id="trajectory-explicit-start-2",
        start_at=start_at,
        point_times=point_times,
        point_values=(0.0, 2.0),
    )
    ensemble = TrajectoryEnsemble(
        ensemble_id="ensemble-explicit-start",
        model_id=first.model_id,
        model_version=first.model_version,
        case_id=first.case_id,
        trajectories=(first, second),
        provenance=first.provenance,
    )

    report = analyze_trajectory_spread(
        ensemble, normalization_scales={"x": 1.0}
    )

    assert report.classification_window.initial_evidence_source == (
        "explicit_horizon_start"
    )
    assert report.classification_window.classified_point_count == 1
    assert report.patterns[0].normalized_population_std == (0.0, 2.0)
    assert report.patterns[0].observed_pattern == "insufficient_spread_points"
