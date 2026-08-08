from datetime import UTC, datetime
from importlib import import_module
from itertools import pairwise
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from pydantic import ValidationError

from pi_engine.schemas.trajectory import TrajectoryEnsemble
from pi_engine.synthetic.controls import (
    random_graph_control,
    shuffled_time_series_control,
)
from pi_engine.synthetic.systems import linear_convergence, stochastic_branching


def test_same_seed_reproduces_byte_equivalent_ensemble() -> None:
    """Unseeded streams or random IDs would make exact replay impossible."""
    fixture = stochastic_branching(seed=7)
    stochastic = import_module("pi_engine.simulation.stochastic")

    first = stochastic.simulate_stochastic(
        fixture.case,
        fixture.model,
        horizon=4,
        samples=4,
        seed=7,
    )
    replay = stochastic.simulate_stochastic(
        fixture.case,
        fixture.model,
        horizon=4,
        samples=4,
        seed=7,
    )

    assert first.model_dump_json() == replay.model_dump_json()


def test_summary_matches_hand_derived_paths_without_replacing_raw_samples() -> None:
    """Dropping paths or using sample variance would misstate Monte Carlo output."""
    fixture = stochastic_branching(seed=7)
    stochastic = import_module("pi_engine.simulation.stochastic")

    ensemble = stochastic.simulate_stochastic(
        fixture.case,
        fixture.model,
        horizon=3,
        samples=3,
        seed=7,
    )

    assert [
        [point.values["x"] for point in trajectory.points]
        for trajectory in ensemble.trajectories
    ] == [
        [-1.0, 0.0, 1.0],
        [1.0, 2.0, 3.0],
        [-1.0, 0.0, 1.0],
    ]
    assert ensemble.summary is not None
    assert [point.at for point in ensemble.summary.points] == [
        fixture.case.prediction_cutoff.replace(hour=13),
        fixture.case.prediction_cutoff.replace(hour=14),
        fixture.case.prediction_cutoff.replace(hour=15),
    ]
    statistics = [point.statistics["x"] for point in ensemble.summary.points]
    assert [item.count for item in statistics] == [3, 3, 3]
    assert [item.mean for item in statistics] == pytest.approx(
        [-1.0 / 3.0, 2.0 / 3.0, 5.0 / 3.0]
    )
    assert [item.population_variance for item in statistics] == pytest.approx(
        [8.0 / 9.0] * 3
    )
    assert [item.population_std for item in statistics] == pytest.approx(
        [0.9428090415820634] * 3
    )
    assert [item.minimum for item in statistics] == [-1.0, 0.0, 1.0]
    assert [item.maximum for item in statistics] == [1.0, 2.0, 3.0]


def test_ensemble_rejects_summary_inconsistent_with_raw_samples() -> None:
    """A plausible but fabricated summary must not override retained samples."""
    fixture = stochastic_branching(seed=7)
    stochastic = import_module("pi_engine.simulation.stochastic")
    ensemble = stochastic.simulate_stochastic(
        fixture.case,
        fixture.model,
        horizon=2,
        samples=3,
        seed=7,
    )
    payload = ensemble.model_dump()
    payload["summary"]["points"][0]["statistics"]["x"]["mean"] = -0.25

    with pytest.raises(ValidationError, match="summary must match"):
        TrajectoryEnsemble.model_validate(payload)


def test_members_retain_spawned_seed_identity_and_model_provenance() -> None:
    """Losing each spawned stream or conditioning identity would block replay."""
    fixture = stochastic_branching(seed=7)
    case = fixture.case.model_copy(
        update={"constraints": ("x remains dimensionless",)}
    )
    case_before = case.model_dump(mode="json")
    model_before = fixture.model.model_dump(mode="json")
    stochastic = import_module("pi_engine.simulation.stochastic")

    ensemble = stochastic.simulate_stochastic(
        case,
        fixture.model,
        horizon=4,
        samples=4,
        seed=7,
    )

    expected_sample_seeds = [
        3386250816931739734,
        4042502035264064771,
        17559002276220262541,
        6823953754371609207,
    ]
    assert [
        item.sample_seed for item in ensemble.trajectories
    ] == expected_sample_seeds
    assert len({item.trajectory_id for item in ensemble.trajectories}) == 4
    for sample_index, (trajectory, sample_seed) in enumerate(
        zip(ensemble.trajectories, expected_sample_seeds, strict=True)
    ):
        assert trajectory.model_id == fixture.model.model_id
        assert trajectory.model_version == fixture.model.version
        assert trajectory.case_id == case.case_id
        assert trajectory.initial_state == case.state
        assert trajectory.horizon == ensemble.trajectories[0].horizon
        assert trajectory.constraints_encountered == (
            "x remains dimensionless",
        )
        assert trajectory.provenance.source == "PI-engine stochastic runner"
        assert f"sample_seed:{sample_seed}" in trajectory.provenance.reference
        assert f"spawn_index:{sample_index}" in trajectory.provenance.reference
    assert case.model_dump(mode="json") == case_before
    assert fixture.model.model_dump(mode="json") == model_before
    assert TrajectoryEnsemble.model_validate_json(
        ensemble.model_dump_json()
    ) == ensemble


def test_different_root_seed_changes_streams_and_identities() -> None:
    """Ignoring the requested root seed would silently replay the same ensemble."""
    fixture = stochastic_branching(seed=7)
    stochastic = import_module("pi_engine.simulation.stochastic")

    first = stochastic.simulate_stochastic(
        fixture.case, fixture.model, horizon=4, samples=4, seed=7
    )
    changed = stochastic.simulate_stochastic(
        fixture.case, fixture.model, horizon=4, samples=4, seed=8
    )

    assert first.ensemble_id != changed.ensemble_id
    assert [item.sample_seed for item in first.trajectories] != [
        item.sample_seed for item in changed.trajectories
    ]
    assert [
        [point.values["x"] for point in item.points]
        for item in first.trajectories
    ] != [
        [point.values["x"] for point in item.points]
        for item in changed.trajectories
    ]


def test_runner_does_not_read_or_advance_numpy_global_rng() -> None:
    """Using numpy's process-global RNG would couple otherwise isolated runs."""
    fixture = stochastic_branching(seed=7)
    stochastic = import_module("pi_engine.simulation.stochastic")
    prior_state = np.random.get_state()
    try:
        np.random.seed(12345)
        expected = np.random.random(2)
        np.random.seed(12345)
        first = np.random.random()

        stochastic.simulate_stochastic(
            fixture.case, fixture.model, horizon=4, samples=4, seed=7
        )

        second = np.random.random()
        assert [first, second] == pytest.approx(expected)
    finally:
        np.random.set_state(prior_state)


def test_fixed_steps_advance_in_absolute_utc_time_across_dst() -> None:
    """Wall-time arithmetic would duplicate or skip an absolute stochastic step."""
    fixture = stochastic_branching(seed=7)
    cutoff = datetime(
        2026,
        11,
        1,
        1,
        30,
        tzinfo=ZoneInfo("America/New_York"),
        fold=0,
    )
    state = fixture.case.state.model_copy(update={"at": cutoff})
    case = fixture.case.model_copy(
        update={
            "prediction_cutoff": cutoff,
            "observations": (),
            "state": state,
        }
    )
    stochastic = import_module("pi_engine.simulation.stochastic")

    ensemble = stochastic.simulate_stochastic(
        case, fixture.model, horizon=3, samples=2, seed=7
    )

    expected_times = (
        datetime(2026, 11, 1, 6, 30, tzinfo=UTC),
        datetime(2026, 11, 1, 7, 30, tzinfo=UTC),
        datetime(2026, 11, 1, 8, 30, tzinfo=UTC),
    )
    assert tuple(point.at for point in ensemble.trajectories[0].points) == (
        expected_times
    )
    timeline = (
        ensemble.trajectories[0].horizon.start_at,
        *expected_times,
    )
    assert [
        later.timestamp() - earlier.timestamp()
        for earlier, later in pairwise(timeline)
    ] == [3600.0, 3600.0, 3600.0]
    assert all(value.tzinfo is UTC for value in timeline)


@pytest.mark.parametrize("field_name", ["horizon", "samples"])
@pytest.mark.parametrize("invalid", [None, True, 0, -1, 1.5])
def test_runner_rejects_nonpositive_or_implicit_counts(
    field_name: str, invalid: object
) -> None:
    """Coercing a run count would make sampling scope ambiguous."""
    fixture = stochastic_branching(seed=7)
    stochastic = import_module("pi_engine.simulation.stochastic")
    arguments: dict[str, object] = {
        "horizon": 1,
        "samples": 1,
        "seed": 7,
    }
    arguments[field_name] = invalid

    with pytest.raises(
        stochastic.StochasticSimulationError,
        match=f"{field_name} must be an explicit positive integer",
    ):
        stochastic.simulate_stochastic(
            fixture.case, fixture.model, **arguments
        )


@pytest.mark.parametrize("invalid", [None, True, -1, 1.5])
def test_runner_rejects_implicit_or_invalid_seed(invalid: object) -> None:
    """Coercing a root seed would make stream derivation ambiguous."""
    fixture = stochastic_branching(seed=7)
    stochastic = import_module("pi_engine.simulation.stochastic")

    with pytest.raises(
        stochastic.StochasticSimulationError,
        match="seed must be an explicit nonnegative integer",
    ):
        stochastic.simulate_stochastic(
            fixture.case,
            fixture.model,
            horizon=1,
            samples=1,
            seed=invalid,
        )


def test_inapplicable_model_is_rejected_before_tampered_executor() -> None:
    """Resolving first would hide the structural routing rejection cause."""
    fixture = stochastic_branching(seed=7)
    applicability = fixture.model.applicability.model_copy(
        update={"required_variables": ("x", "missing_input")}
    )
    dynamics = fixture.model.dynamics.model_copy(
        update={"executor_ref": "untrusted_ref"}
    )
    model = fixture.model.model_copy(
        update={"applicability": applicability, "dynamics": dynamics}
    )
    stochastic = import_module("pi_engine.simulation.stochastic")

    with pytest.raises(stochastic.InapplicableStochasticModelError) as exc_info:
        stochastic.simulate_stochastic(
            fixture.case, model, horizon=1, samples=1, seed=7
        )

    assert exc_info.value.rejection_causes == (
        "missing required variable: missing_input",
    )


@pytest.mark.parametrize(
    "dynamics_update",
    [
        {"executor_ref": "untrusted_ref"},
        {"executor_version": "2"},
        {"code_sha256": "00" * 32},
    ],
    ids=("unknown-ref", "version-mismatch", "code-mismatch"),
)
def test_runner_rejects_tampered_executor_identity(
    dynamics_update: dict[str, str],
) -> None:
    """Approximate registry matching could execute untrusted stochastic code."""
    fixture = stochastic_branching(seed=7)
    dynamics = fixture.model.dynamics.model_copy(update=dynamics_update)
    model = fixture.model.model_copy(update={"dynamics": dynamics})
    stochastic = import_module("pi_engine.simulation.stochastic")

    with pytest.raises(
        stochastic.StochasticSimulationError,
        match="exact trusted stochastic executor identity",
    ):
        stochastic.simulate_stochastic(
            fixture.case, model, horizon=1, samples=1, seed=7
        )


def test_runner_rejects_deterministic_classification() -> None:
    """Sampling a deterministic contract would invent stochastic semantics."""
    fixture = linear_convergence(seed=7)
    stochastic = import_module("pi_engine.simulation.stochastic")

    with pytest.raises(
        stochastic.StochasticSimulationError,
        match="deterministic-classified model",
    ):
        stochastic.simulate_stochastic(
            fixture.case, fixture.model, horizon=1, samples=1, seed=7
        )


@pytest.mark.parametrize(
    "factory", [random_graph_control, shuffled_time_series_control]
)
def test_runner_does_not_invent_unsupported_control_semantics(
    factory: object,
) -> None:
    """Unknown stochastic metadata must not be interpreted heuristically."""
    fixture = factory(seed=7)
    stochastic = import_module("pi_engine.simulation.stochastic")

    with pytest.raises(
        stochastic.StochasticSimulationError,
        match="exact trusted stochastic executor identity",
    ):
        stochastic.simulate_stochastic(
            fixture.case, fixture.model, horizon=1, samples=1, seed=7
        )


def test_runner_rejects_vector_output_instead_of_flattening_components() -> None:
    """Generic vector components cannot acquire invented summary identities."""
    fixture = random_graph_control(seed=7)
    metadata = {
        "state_variable": "values",
        "up_probability": 0.5,
        "up_step": 1.0,
        "down_step": -1.0,
        "step_seconds": 3600,
    }
    dynamics = fixture.model.dynamics.model_copy(
        update={
            "executor_ref": "bernoulli_step",
            "executor_version": "1",
            "code_sha256": (
                "b02f51c53d15904a6758faadd0bed53ba89b321d01a54211d97d025c577ef0ad"
            ),
            "transition_metadata": metadata,
        }
    )
    model = fixture.model.model_copy(
        update={"dynamics": dynamics, "predicted_outputs": ("values",)}
    )
    stochastic = import_module("pi_engine.simulation.stochastic")

    with pytest.raises(
        stochastic.StochasticSimulationError,
        match="shape mismatch.*expected scalar",
    ):
        stochastic.simulate_stochastic(
            fixture.case, model, horizon=1, samples=1, seed=7
        )


def test_runner_rejects_nonfinite_state_produced_by_executor() -> None:
    """Overflow must fail before a nonfinite value enters a raw sample."""
    fixture = stochastic_branching(seed=7)
    state = fixture.case.state.model_copy(update={"observed": {"x": 1e308}})
    case = fixture.case.model_copy(update={"state": state})
    metadata = dict(fixture.model.dynamics.transition_metadata)
    metadata.update({"up_step": 1e308, "down_step": 1e308})
    dynamics = fixture.model.dynamics.model_copy(
        update={"transition_metadata": metadata}
    )
    model = fixture.model.model_copy(update={"dynamics": dynamics})
    stochastic = import_module("pi_engine.simulation.stochastic")

    with pytest.raises(
        stochastic.StochasticSimulationError, match="nonfinite state"
    ):
        stochastic.simulate_stochastic(
            case, model, horizon=1, samples=1, seed=7
        )


def test_ensemble_rejects_duplicate_sample_identity() -> None:
    """Duplicate member IDs could make two streams indistinguishable."""
    fixture = stochastic_branching(seed=7)
    stochastic = import_module("pi_engine.simulation.stochastic")
    ensemble = stochastic.simulate_stochastic(
        fixture.case, fixture.model, horizon=1, samples=2, seed=7
    )
    payload = ensemble.model_dump()
    payload["trajectories"][1]["trajectory_id"] = payload["trajectories"][0][
        "trajectory_id"
    ]

    with pytest.raises(ValidationError, match="identities must be distinct"):
        TrajectoryEnsemble.model_validate(payload)


def test_runner_accepts_only_prediction_case_not_withheld_fixture() -> None:
    """The outcome-bearing synthetic wrapper must fail at the Case boundary."""
    fixture = stochastic_branching(seed=7)
    stochastic = import_module("pi_engine.simulation.stochastic")

    with pytest.raises(TypeError, match="case must be a Case"):
        stochastic.simulate_stochastic(
            fixture, fixture.model, horizon=1, samples=1, seed=7
        )
