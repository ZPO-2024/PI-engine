from datetime import UTC, datetime, timedelta
from importlib import import_module
from itertools import pairwise
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from pi_engine.schemas.case import VariableDefinition
from pi_engine.synthetic.systems import (
    coupled_oscillators,
    deterministic_divergence,
    feedback_instability,
    hierarchical_nested_dynamics,
    linear_convergence,
    oscillation,
    stochastic_branching,
)


def test_linear_convergence_produces_literal_future_points() -> None:
    """Using the wrong affine update or time step would corrupt the path."""
    fixture = linear_convergence(seed=7)
    runner = import_module("pi_engine.simulation.runner")

    trajectory = runner.simulate_deterministic(
        fixture.case, fixture.model, horizon=3
    )

    assert [
        (point.at, dict(point.values)) for point in trajectory.points
    ] == [
        (fixture.case.prediction_cutoff + timedelta(hours=1), {"x": 1.0}),
        (fixture.case.prediction_cutoff + timedelta(hours=2), {"x": 0.5}),
        (fixture.case.prediction_cutoff + timedelta(hours=3), {"x": 0.25}),
    ]
    assert trajectory.horizon.start_at == fixture.case.prediction_cutoff
    assert trajectory.horizon.end_at == (
        fixture.case.prediction_cutoff + timedelta(hours=3)
    )


def test_planar_rotation_updates_both_values_from_the_prior_state() -> None:
    """Sequential writes would turn the quarter rotation into the wrong orbit."""
    fixture = oscillation(seed=7)
    runner = import_module("pi_engine.simulation.runner")

    trajectory = runner.simulate_deterministic(
        fixture.case, fixture.model, horizon=3
    )

    assert [dict(point.values) for point in trajectory.points] == [
        {"position": 0.0, "velocity": -1.0},
        {"position": -1.0, "velocity": 0.0},
        {"position": 0.0, "velocity": 1.0},
    ]


@pytest.mark.parametrize(
    ("month", "day", "hour", "fold", "expected_utc_points"),
    [
        (
            3,
            8,
            1,
            0,
            (
                datetime(2026, 3, 8, 7, 30, tzinfo=UTC),
                datetime(2026, 3, 8, 8, 30, tzinfo=UTC),
                datetime(2026, 3, 8, 9, 30, tzinfo=UTC),
            ),
        ),
        (
            11,
            1,
            1,
            0,
            (
                datetime(2026, 11, 1, 6, 30, tzinfo=UTC),
                datetime(2026, 11, 1, 7, 30, tzinfo=UTC),
                datetime(2026, 11, 1, 8, 30, tzinfo=UTC),
            ),
        ),
    ],
    ids=("spring-forward", "fall-back"),
)
def test_fixed_steps_advance_in_absolute_time_across_dst_transitions(
    month: int,
    day: int,
    hour: int,
    fold: int,
    expected_utc_points: tuple[datetime, ...],
) -> None:
    """Wall-time arithmetic would duplicate or skip an absolute-time step."""
    fixture = linear_convergence(seed=7)
    cutoff = datetime(
        2026, month, day, hour, 30,
        tzinfo=ZoneInfo("America/New_York"),
        fold=fold,
    )
    state = fixture.case.state.model_copy(update={"at": cutoff})
    case = fixture.case.model_copy(
        update={
            "prediction_cutoff": cutoff,
            "observations": (),
            "state": state,
        }
    )
    runner = import_module("pi_engine.simulation.runner")

    trajectory = runner.simulate_deterministic(case, fixture.model, horizon=3)

    assert trajectory.initial_state == case.state
    assert tuple(point.at for point in trajectory.points) == expected_utc_points
    timeline = (trajectory.horizon.start_at,) + tuple(
        point.at for point in trajectory.points
    )
    assert [
        later.timestamp() - earlier.timestamp()
        for earlier, later in pairwise(timeline)
    ] == [3600.0, 3600.0, 3600.0]
    assert all(
        earlier < later
        for earlier, later in pairwise(timeline)
    )
    assert all(value.tzinfo is UTC for value in timeline)


def test_cutoff_alignment_compares_absolute_instants_in_repeated_hour() -> None:
    """Equivalent UTC/local state times must align during an ambiguous hour."""
    fixture = linear_convergence(seed=7)
    cutoff = datetime(
        2026,
        11,
        1,
        1,
        30,
        tzinfo=ZoneInfo("America/New_York"),
        fold=0,
    )
    state = fixture.case.state.model_copy(update={"at": cutoff.astimezone(UTC)})
    case = fixture.case.model_copy(
        update={
            "prediction_cutoff": cutoff,
            "observations": (),
            "state": state,
        }
    )
    runner = import_module("pi_engine.simulation.runner")

    trajectory = runner.simulate_deterministic(case, fixture.model, horizon=1)

    assert trajectory.initial_state == state
    assert trajectory.horizon.start_at == datetime(
        2026, 11, 1, 5, 30, tzinfo=UTC
    )
    assert trajectory.points[0].at == datetime(
        2026, 11, 1, 6, 30, tzinfo=UTC
    )


@pytest.mark.parametrize(
    ("factory", "horizon", "expected_points"),
    [
        (
            coupled_oscillators,
            1,
            [{"phase_a": 0.25, "phase_b": 1.3207963267948966}],
        ),
        (
            feedback_instability,
            3,
            [{"x": 1.5}, {"x": 2.25}, {"x": 3.375}],
        ),
        (
            deterministic_divergence,
            3,
            [{"x": 2.0}, {"x": 4.0}, {"x": 8.0}],
        ),
        (
            hierarchical_nested_dynamics,
            2,
            [
                {"levels": (0.5, 0.5, 0.0)},
                {"levels": (0.25, 0.375, 0.25)},
            ],
        ),
    ],
    ids=(
        "coupled-phase",
        "linear-feedback",
        "linear-divergence",
        "nested-linear",
    ),
)
def test_deterministic_catalog_executors_produce_literal_points(
    factory: object,
    horizon: int,
    expected_points: list[dict[str, object]],
) -> None:
    """Missing or miswired trusted executors would skip supported dynamics."""
    fixture = factory(seed=7)
    runner = import_module("pi_engine.simulation.runner")

    trajectory = runner.simulate_deterministic(
        fixture.case, fixture.model, horizon=horizon
    )

    assert len(trajectory.points) == len(expected_points)
    for point, expected in zip(
        trajectory.points, expected_points, strict=True
    ):
        assert set(point.values) == set(expected)
        for variable, expected_value in expected.items():
            assert point.values[variable] == pytest.approx(expected_value)


def test_runner_preserves_identity_state_constraints_and_audit_provenance() -> None:
    """Dropping an input identity or constraint would make a run unauditable."""
    fixture = linear_convergence(seed=7)
    case = fixture.case.model_copy(
        update={"constraints": ("x remains dimensionless",)}
    )
    case_before = case.model_dump(mode="json")
    model_before = fixture.model.model_dump(mode="json")
    runner = import_module("pi_engine.simulation.runner")

    trajectory = runner.simulate_deterministic(case, fixture.model, horizon=2)

    assert trajectory.model_id == fixture.model.model_id
    assert trajectory.model_version == fixture.model.version
    assert trajectory.case_id == case.case_id
    assert trajectory.initial_state == case.state
    assert trajectory.constraints_encountered == (
        "x remains dimensionless",
    )
    assert trajectory.provenance.source == "PI-engine deterministic runner"
    assert trajectory.provenance.observed_at == case.prediction_cutoff
    assert trajectory.provenance.reference == f"run:{trajectory.trajectory_id}"
    assert case.model_dump(mode="json") == case_before
    assert fixture.model.model_dump(mode="json") == model_before


def test_trajectory_identity_is_stable_and_bound_to_requested_horizon() -> None:
    """Random or horizon-blind IDs would collide across reproducible runs."""
    fixture = linear_convergence(seed=7)
    runner = import_module("pi_engine.simulation.runner")

    first = runner.simulate_deterministic(fixture.case, fixture.model, horizon=1)
    same = runner.simulate_deterministic(fixture.case, fixture.model, horizon=1)
    longer = runner.simulate_deterministic(
        fixture.case, fixture.model, horizon=2
    )

    assert first.trajectory_id == same.trajectory_id
    assert first.trajectory_id != longer.trajectory_id
    assert first.trajectory_id.startswith("trajectory-")
    assert len(first.trajectory_id.removeprefix("trajectory-")) == 64


@pytest.mark.parametrize("horizon", [None, True, 0, -1, 1.5])
def test_runner_rejects_nonpositive_or_implicit_horizon(horizon: object) -> None:
    """Coercing an absent or invalid horizon would make run scope ambiguous."""
    fixture = linear_convergence(seed=7)
    runner = import_module("pi_engine.simulation.runner")

    with pytest.raises(
        runner.DeterministicSimulationError, match="positive integer"
    ):
        runner.simulate_deterministic(
            fixture.case, fixture.model, horizon=horizon
        )


def test_inapplicable_model_is_rejected_with_routing_causes_first() -> None:
    """Executing before routing would hide the structural reason for rejection."""
    fixture = linear_convergence(seed=7)
    applicability = fixture.model.applicability.model_copy(
        update={"required_variables": ("x", "missing_input")}
    )
    dynamics = fixture.model.dynamics.model_copy(
        update={"executor_ref": "untrusted_ref"}
    )
    model = fixture.model.model_copy(
        update={"applicability": applicability, "dynamics": dynamics}
    )
    runner = import_module("pi_engine.simulation.runner")

    with pytest.raises(runner.InapplicableModelError) as exc_info:
        runner.simulate_deterministic(fixture.case, model, horizon=1)

    assert exc_info.value.rejection_causes == (
        "missing required variable: missing_input",
    )
    assert "missing required variable: missing_input" in str(exc_info.value)


@pytest.mark.parametrize(
    ("dynamics_update", "expected_message"),
    [
        ({"executor_ref": "untrusted_ref"}, "unknown deterministic executor_ref"),
        ({"executor_version": "2"}, "executor version mismatch"),
        ({"code_sha256": "00" * 32}, "executor code identity mismatch"),
    ],
    ids=("unknown-ref", "version-mismatch", "code-mismatch"),
)
def test_runner_rejects_untrusted_or_tampered_executor_identity(
    dynamics_update: dict[str, str], expected_message: str
) -> None:
    """Approximate executor resolution could execute a different transition."""
    fixture = linear_convergence(seed=7)
    dynamics = fixture.model.dynamics.model_copy(update=dynamics_update)
    model = fixture.model.model_copy(update={"dynamics": dynamics})
    runner = import_module("pi_engine.simulation.runner")

    with pytest.raises(
        runner.DeterministicSimulationError, match=expected_message
    ):
        runner.simulate_deterministic(fixture.case, model, horizon=1)


def test_runner_rejects_stochastic_classification_without_branching() -> None:
    """A deterministic path must never silently stand in for stochastic samples."""
    fixture = stochastic_branching(seed=7)
    runner = import_module("pi_engine.simulation.runner")

    with pytest.raises(
        runner.DeterministicSimulationError,
        match="stochastic-classified model",
    ):
        runner.simulate_deterministic(fixture.case, fixture.model, horizon=1)


def test_runner_rejects_nonfinite_state_produced_by_a_trusted_executor() -> None:
    """Floating-point overflow must fail instead of entering a trajectory."""
    fixture = linear_convergence(seed=7)
    state = fixture.case.state.model_copy(update={"observed": {"x": 1e308}})
    case = fixture.case.model_copy(update={"state": state})
    metadata = dict(fixture.model.dynamics.transition_metadata)
    metadata["multiplier"] = 1e308
    dynamics = fixture.model.dynamics.model_copy(
        update={"transition_metadata": metadata}
    )
    model = fixture.model.model_copy(update={"dynamics": dynamics})
    runner = import_module("pi_engine.simulation.runner")

    with pytest.raises(
        runner.DeterministicSimulationError, match="nonfinite"
    ):
        runner.simulate_deterministic(case, model, horizon=1)


def test_runner_rejects_outputs_not_declared_by_the_model() -> None:
    """Executor variables outside predicted_outputs must not leak into points."""
    fixture = oscillation(seed=7)
    model = fixture.model.model_copy(update={"predicted_outputs": ("position",)})
    runner = import_module("pi_engine.simulation.runner")

    with pytest.raises(
        runner.DeterministicSimulationError, match="undeclared outputs: velocity"
    ):
        runner.simulate_deterministic(fixture.case, model, horizon=1)


@pytest.mark.parametrize(
    ("factory", "variable_updates", "predicted_outputs"),
    [
        (
            oscillation,
            {
                "position_variable": "position",
                "velocity_variable": "position",
            },
            ("position",),
        ),
        (
            coupled_oscillators,
            {
                "phase_a_variable": "phase_a",
                "phase_b_variable": "phase_a",
            },
            ("phase_a",),
        ),
    ],
    ids=("planar-rotation", "coupled-phase"),
)
def test_runner_rejects_aliased_multi_output_variable_metadata(
    factory: object,
    variable_updates: dict[str, str],
    predicted_outputs: tuple[str, ...],
) -> None:
    """Aliased output names must not collapse two transitions into one key."""
    fixture = factory(seed=7)
    metadata = dict(fixture.model.dynamics.transition_metadata)
    metadata.update(variable_updates)
    dynamics = fixture.model.dynamics.model_copy(
        update={"transition_metadata": metadata}
    )
    model = fixture.model.model_copy(
        update={
            "dynamics": dynamics,
            "predicted_outputs": predicted_outputs,
        }
    )
    runner = import_module("pi_engine.simulation.runner")

    with pytest.raises(
        runner.DeterministicSimulationError,
        match="output variables must be distinct",
    ):
        runner.simulate_deterministic(fixture.case, model, horizon=1)


@pytest.mark.parametrize(
    ("factory", "retained_output"),
    [(oscillation, "position"), (coupled_oscillators, "phase_a")],
    ids=("planar-rotation", "coupled-phase"),
)
def test_multi_output_arity_is_checked_before_transition_state_reads(
    factory: object, retained_output: str
) -> None:
    """Reduced declarations must fail before an executor reads its full state."""
    fixture = factory(seed=7)
    state = fixture.case.state.model_copy(
        update={
            "observed": {
                retained_output: fixture.case.state.observed[retained_output]
            },
            "uncertainty": {retained_output: 0.0},
        }
    )
    case = fixture.case.model_copy(update={"state": state})
    model = fixture.model.model_copy(
        update={"predicted_outputs": (retained_output,)}
    )
    runner = import_module("pi_engine.simulation.runner")

    with pytest.raises(
        runner.DeterministicSimulationError,
        match="requires exactly 2 predicted outputs",
    ):
        runner.simulate_deterministic(case, model, horizon=1)


def test_nested_linear_arity_is_checked_before_vector_state_transition() -> None:
    """Extra declarations must fail before nested-linear reads vector shape."""
    fixture = hierarchical_nested_dynamics(seed=7)
    state = fixture.case.state.model_copy(
        update={
            "observed": {"levels": 1.0, "extra": 0.0},
            "uncertainty": {"levels": 0.0, "extra": 0.0},
        }
    )
    case = fixture.case.model_copy(
        update={
            "canonical_variables": fixture.case.canonical_variables
            + (VariableDefinition(name="extra", unit="a.u."),),
            "state": state,
        }
    )
    model = fixture.model.model_copy(
        update={"predicted_outputs": ("levels", "extra")}
    )
    runner = import_module("pi_engine.simulation.runner")

    with pytest.raises(
        runner.DeterministicSimulationError,
        match="requires exactly 1 predicted output",
    ):
        runner.simulate_deterministic(case, model, horizon=1)


def test_runner_rejects_duplicate_declared_outputs() -> None:
    """Duplicate output declarations must not collapse silently into one value."""
    fixture = linear_convergence(seed=7)
    model = fixture.model.model_copy(update={"predicted_outputs": ("x", "x")})
    runner = import_module("pi_engine.simulation.runner")

    with pytest.raises(
        runner.DeterministicSimulationError, match="duplicate predicted outputs"
    ):
        runner.simulate_deterministic(fixture.case, model, horizon=1)


def test_runner_rejects_undeclared_transition_metadata_including_units() -> None:
    """A metadata unit override must not silently drift a canonical variable."""
    fixture = linear_convergence(seed=7)
    metadata = dict(fixture.model.dynamics.transition_metadata)
    metadata["output_unit"] = "different-unit"
    dynamics = fixture.model.dynamics.model_copy(
        update={"transition_metadata": metadata}
    )
    model = fixture.model.model_copy(update={"dynamics": dynamics})
    runner = import_module("pi_engine.simulation.runner")

    with pytest.raises(
        runner.DeterministicSimulationError,
        match="undeclared: output_unit",
    ):
        runner.simulate_deterministic(fixture.case, model, horizon=1)


def test_runner_revalidates_unchecked_case_state_at_the_boundary() -> None:
    """Unchecked NaN state must not bypass the public simulation boundary."""
    fixture = linear_convergence(seed=7)
    state = fixture.case.state.model_construct(
        at=fixture.case.state.at,
        observed={"x": float("nan")},
        latent={},
        uncertainty={"x": 0.0},
        boundary={},
    )
    case = fixture.case.model_copy(update={"state": state})
    runner = import_module("pi_engine.simulation.runner")

    with pytest.raises(ValidationError):
        runner.simulate_deterministic(case, fixture.model, horizon=1)


def test_runner_revalidates_unchecked_model_metadata_at_the_boundary() -> None:
    """Unchecked NaN metadata must not reach trusted executor arithmetic."""
    fixture = linear_convergence(seed=7)
    metadata = dict(fixture.model.dynamics.transition_metadata)
    metadata["multiplier"] = float("nan")
    payload = fixture.model.dynamics.model_dump()
    payload["transition_metadata"] = metadata
    dynamics = fixture.model.dynamics.model_construct(**payload)
    model = fixture.model.model_copy(update={"dynamics": dynamics})
    runner = import_module("pi_engine.simulation.runner")

    with pytest.raises(ValidationError):
        runner.simulate_deterministic(fixture.case, model, horizon=1)


def test_runner_rejects_scalar_vector_executor_shape_mismatch() -> None:
    """Resolving a scalar executor for vector state must fail before a point exists."""
    fixture = hierarchical_nested_dynamics(seed=7)
    metadata = {
        "state_variable": "levels",
        "multiplier": 0.5,
        "intercept": 0.0,
        "step_seconds": 3600,
    }
    dynamics = fixture.model.dynamics.model_copy(
        update={
            "executor_ref": "linear_affine",
            "executor_version": "1",
            "code_sha256": (
                "20d2ac1b70f95a3492439992f268e6070d85a71f6af59a5e4e05d7b46d7c6384"
            ),
            "transition_metadata": metadata,
        }
    )
    model = fixture.model.model_copy(update={"dynamics": dynamics})
    runner = import_module("pi_engine.simulation.runner")

    with pytest.raises(
        runner.DeterministicSimulationError, match="shape mismatch"
    ):
        runner.simulate_deterministic(fixture.case, model, horizon=1)


def test_runner_rejects_unrepresentable_point_times() -> None:
    """Timestamp overflow must fail explicitly instead of escaping as arithmetic."""
    fixture = linear_convergence(seed=7)
    metadata = dict(fixture.model.dynamics.transition_metadata)
    metadata["step_seconds"] = 10**20
    dynamics = fixture.model.dynamics.model_copy(
        update={"transition_metadata": metadata}
    )
    model = fixture.model.model_copy(update={"dynamics": dynamics})
    runner = import_module("pi_engine.simulation.runner")

    with pytest.raises(
        runner.DeterministicSimulationError, match="datetime range"
    ):
        runner.simulate_deterministic(fixture.case, model, horizon=1)


def test_runner_accepts_only_prediction_case_not_withheld_fixture() -> None:
    """Passing the outcome-bearing fixture must fail at the Case boundary."""
    fixture = linear_convergence(seed=7)
    runner = import_module("pi_engine.simulation.runner")

    with pytest.raises(TypeError, match="case must be a Case"):
        runner.simulate_deterministic(fixture, fixture.model, horizon=1)
