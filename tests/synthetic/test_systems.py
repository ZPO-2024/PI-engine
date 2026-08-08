from importlib import import_module

import numpy as np
import pytest


def test_linear_convergence_fixture_is_cutoff_safe_and_round_trippable() -> None:
    """Leaking future values or dynamics into Case would invalidate holdout tests."""
    try:
        systems = import_module("pi_engine.synthetic.systems")
    except ModuleNotFoundError:
        pytest.fail("synthetic systems contract is missing")

    fixture = systems.linear_convergence(seed=7)
    dumped = fixture.model_dump(mode="json")

    assert fixture.name == "linear_convergence"
    assert fixture.regime == "convergence"
    assert fixture.seed == 7
    assert fixture.is_control is False
    assert [observation.value for observation in fixture.case.observations] == [
        8.0,
        4.0,
        2.0,
    ]
    assert [outcome.value for outcome in fixture.outcomes] == [1.0, 0.5, 0.25]
    assert all(
        observation.event_time
        <= observation.available_at
        <= fixture.case.prediction_cutoff
        for observation in fixture.case.observations
    )
    assert all(
        fixture.case.prediction_cutoff < outcome.event_time <= outcome.available_at
        for outcome in fixture.outcomes
    )
    assert "outcomes" not in dumped["case"]
    assert "equations" not in dumped["case"]
    assert dumped["ground_truth"] == {"horizon_steps": 3}
    assert (
        systems.SyntheticSystem.model_validate_json(fixture.model_dump_json())
        == fixture
    )

    with pytest.raises(TypeError):
        fixture.ground_truth["horizon_steps"] = 99


@pytest.mark.parametrize(
    ("factory_name", "regime", "expected_values"),
    [
        (
            "oscillation",
            "oscillation",
            [
                ("position", 0.0),
                ("velocity", -1.0),
                ("position", -1.0),
                ("velocity", 0.0),
                ("position", 0.0),
                ("velocity", 1.0),
            ],
        ),
        (
            "deterministic_divergence",
            "divergence",
            [("x", 2.0), ("x", 4.0), ("x", 8.0)],
        ),
        (
            "coupled_oscillators",
            "coupled_oscillation",
            [("phase_a", 0.25), ("phase_b", 1.3207963267948966)],
        ),
        (
            "feedback_instability",
            "feedback_instability",
            [("x", 1.5), ("x", 2.25), ("x", 3.375)],
        ),
        (
            "hierarchical_nested_dynamics",
            "nested_dynamics",
            [("levels", (0.5, 0.5, 0.0)), ("levels", (0.25, 0.375, 0.25))],
        ),
    ],
)
def test_deterministic_factories_expose_hand_checked_withheld_truth(
    factory_name: str,
    regime: str,
    expected_values: list[tuple[str, object]],
) -> None:
    """Wrong transitions or embedded future values would corrupt simulation tests."""
    systems = import_module("pi_engine.synthetic.systems")
    fixture = getattr(systems, factory_name)(seed=7)

    assert fixture.name == factory_name
    assert fixture.regime == regime
    actual_values = [
        (outcome.variable, outcome.value) for outcome in fixture.outcomes
    ]
    assert [variable for variable, _ in actual_values] == [
        variable for variable, _ in expected_values
    ]
    for (_, actual), (_, expected) in zip(actual_values, expected_values, strict=True):
        assert actual == pytest.approx(expected)
    assert all(
        outcome.event_time > fixture.case.prediction_cutoff
        for outcome in fixture.outcomes
    )
    assert fixture.model.dynamics.classification == "deterministic"


def test_stochastic_branching_uses_literal_seeded_path_and_local_rng() -> None:
    """Seed drift or global RNG use would make Monte Carlo tests nonreproducible."""
    systems = import_module("pi_engine.synthetic.systems")
    np.random.seed(1234)
    first_global = np.random.random()

    first = systems.stochastic_branching(seed=17)
    second_global = np.random.random()
    same = systems.stochastic_branching(seed=17)
    different = systems.stochastic_branching(seed=18)

    assert [outcome.value for outcome in first.outcomes] == [-1.0, 0.0, 1.0, 2.0]
    assert first.model_dump_json() == same.model_dump_json()
    assert first.model_dump_json() != different.model_dump_json()
    assert first.model.dynamics.classification == "stochastic"
    assert first.model.dynamics.executor_ref == "bernoulli_step"
    np.random.seed(1234)
    assert [first_global, second_global] == pytest.approx(
        [0.1915194503788923, 0.6221087710398319]
    )


@pytest.mark.parametrize(
    ("factory_name", "executor_ref", "transition_metadata"),
    [
        (
            "linear_convergence",
            "linear_affine",
            {
                "state_variable": "x",
                "multiplier": 0.5,
                "intercept": 0.0,
                "step_seconds": 3600,
            },
        ),
        (
            "oscillation",
            "planar_rotation",
            {
                "position_variable": "position",
                "velocity_variable": "velocity",
                "cosine": 0.0,
                "sine": 1.0,
                "step_seconds": 3600,
            },
        ),
        (
            "deterministic_divergence",
            "linear_affine",
            {
                "state_variable": "x",
                "multiplier": 2.0,
                "intercept": 0.0,
                "step_seconds": 3600,
            },
        ),
        (
            "stochastic_branching",
            "bernoulli_step",
            {
                "state_variable": "x",
                "up_probability": 0.6,
                "up_step": 1.0,
                "down_step": -1.0,
                "step_seconds": 3600,
            },
        ),
        (
            "coupled_oscillators",
            "coupled_phase",
            {
                "phase_a_variable": "phase_a",
                "phase_b_variable": "phase_b",
                "intrinsic_step_a": 0.0,
                "intrinsic_step_b": 0.0,
                "coupling": 0.25,
                "step_seconds": 3600,
            },
        ),
        (
            "feedback_instability",
            "linear_feedback",
            {
                "state_variable": "x",
                "plant_multiplier": 1.0,
                "feedback_gain": 0.5,
                "reference": 0.0,
                "step_seconds": 3600,
            },
        ),
        (
            "hierarchical_nested_dynamics",
            "nested_linear",
            {
                "state_variable": "levels",
                "root_multiplier": 0.5,
                "level_multiplier": 0.25,
                "parent_coupling": 0.5,
                "step_seconds": 3600,
            },
        ),
    ],
)
def test_positive_factory_transition_contracts_are_unambiguous(
    factory_name: str, executor_ref: str, transition_metadata: dict[str, object]
) -> None:
    """Missing transition keys would force simulation executors to guess semantics."""
    systems = import_module("pi_engine.synthetic.systems")
    fixture = getattr(systems, factory_name)(seed=17)

    assert fixture.model.dynamics.executor_ref == executor_ref
    assert fixture.model.dynamics.transition_metadata == transition_metadata
    assert fixture.model.dynamics.equations or fixture.model.dynamics.rules


def test_system_catalog_names_every_required_positive_factory() -> None:
    """Omitting a regime would make comprehensive fixture runs silently incomplete."""
    systems = import_module("pi_engine.synthetic.systems")

    assert tuple(systems.SYSTEM_CATALOG) == (
        "linear_convergence",
        "oscillation",
        "deterministic_divergence",
        "stochastic_branching",
        "coupled_oscillators",
        "feedback_instability",
        "hierarchical_nested_dynamics",
    )
    assert [
        factory(seed=3).name for factory in systems.SYSTEM_CATALOG.values()
    ] == list(systems.SYSTEM_CATALOG)
    with pytest.raises(TypeError):
        systems.SYSTEM_CATALOG["missing"] = systems.linear_convergence


def test_random_graph_control_has_literal_seeded_topology_and_vector_outcome() -> None:
    """Shared graph/outcome RNG state would confound the negative control."""
    controls = import_module("pi_engine.synthetic.controls")
    fixture = controls.random_graph_control(seed=7)
    edges = fixture.case.graph.edges

    assert [(edge.source, edge.target) for edge in edges] == [
        ("node-0", "node-2"),
        ("node-0", "node-4"),
        ("node-1", "node-0"),
        ("node-3", "node-1"),
        ("node-3", "node-4"),
    ]
    assert edges[0].strength == pytest.approx(0.1827022348597933)
    assert edges[0].effective_proximity == pytest.approx(4.357243202416154)
    assert fixture.case.state.observed["values"] == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert fixture.outcomes[0].value == pytest.approx(
        (
            1.4019101206317888,
            0.8534203299688002,
            3.05630239701777,
            -0.057023513331478363,
            1.2870073210024566,
        )
    )
    assert fixture.case.graph.topology_metadata == {
        "randomized": True,
        "rng_spawn_count": 2,
        "rng_spawn_index": 0,
    }
    assert fixture.model.dynamics.transition_metadata == {
        "state_variable": "values",
        "mean": 0.0,
        "standard_deviation": 1.0,
        "rng_spawn_count": 2,
        "rng_spawn_index": 1,
        "step_seconds": 3600,
    }


@pytest.mark.parametrize(
    ("module_name", "factory_name", "expected"),
    [
        ("systems", "linear_convergence", ()),
        ("systems", "oscillation", ()),
        ("systems", "deterministic_divergence", ()),
        ("systems", "stochastic_branching", ("seeded Bernoulli branch",)),
        ("systems", "coupled_oscillators", ()),
        ("systems", "feedback_instability", ()),
        ("systems", "hierarchical_nested_dynamics", ()),
        (
            "controls",
            "random_graph_control",
            ("independent Gaussian component draws",),
        ),
        (
            "controls",
            "shuffled_time_series_control",
            ("seeded permutation without replacement",),
        ),
        ("controls", "irrelevant_proximity_control", ()),
        ("controls", "no_paired_structure_control", ()),
    ],
)
def test_fixture_process_disturbance_matches_declared_transition(
    module_name: str, factory_name: str, expected: tuple[str, ...]
) -> None:
    """Classification-only disturbance labels would misstate model assumptions."""
    module = import_module(f"pi_engine.synthetic.{module_name}")
    fixture = getattr(module, factory_name)(seed=7)

    assert fixture.model.uncertainty.process_disturbance == expected


def test_shuffled_series_control_uses_literal_seeded_order() -> None:
    """Keeping source order would preserve the temporal structure under test."""
    controls = import_module("pi_engine.synthetic.controls")
    fixture = controls.shuffled_time_series_control(seed=7)

    assert [
        observation.value
        for observation in fixture.case.observations
        if observation.variable == "signal"
    ] == [0.0, 6.0, 7.0, 2.0, 4.0, 5.0, 1.0]
    assert fixture.case.state.observed == {"signal": 1.0, "draw_index": 6.0}
    assert fixture.outcomes[0].value == 3.0
    assert fixture.model.dynamics.executor_ref == "permutation_series"
    assert fixture.model.dynamics.transition_metadata == {
        "state_variable": "signal",
        "index_variable": "draw_index",
        "population_size": 8,
        "step_seconds": 3600,
    }


def test_irrelevant_proximity_changes_distances_without_changing_truth() -> None:
    """If proximity changes truth, the control has baked in the tested effect."""
    controls = import_module("pi_engine.synthetic.controls")
    first = controls.irrelevant_proximity_control(seed=7)
    second = controls.irrelevant_proximity_control(seed=8)

    first_proximities = [
        edge.effective_proximity for edge in first.case.graph.edges
    ]
    second_proximities = [
        edge.effective_proximity for edge in second.case.graph.edges
    ]
    assert first_proximities[0] == pytest.approx(3.1442225596931013)
    assert first_proximities[-1] == pytest.approx(4.374089554711496)
    assert first_proximities != second_proximities
    assert first.case.state == second.case.state
    assert [outcome.value for outcome in first.outcomes] == [
        outcome.value for outcome in second.outcomes
    ]
    assert first.model.effective_proximity_dependencies == ()


def test_no_paired_structure_control_has_unique_directed_node_roles() -> None:
    """Duplicate graph roles would accidentally supply a paired-structure signal."""
    controls = import_module("pi_engine.synthetic.controls")
    fixture = controls.no_paired_structure_control(seed=7)
    node_ids = [node.node_id for node in fixture.case.graph.nodes]
    degree_signatures = []
    for node_id in node_ids:
        incoming = sum(edge.target == node_id for edge in fixture.case.graph.edges)
        outgoing = sum(edge.source == node_id for edge in fixture.case.graph.edges)
        degree_signatures.append((incoming, outgoing))

    assert degree_signatures == [(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)]
    assert len(set(degree_signatures)) == len(node_ids)


@pytest.mark.parametrize(
    "factory_name",
    [
        "random_graph_control",
        "shuffled_time_series_control",
        "irrelevant_proximity_control",
        "no_paired_structure_control",
    ],
)
def test_controls_are_reproducible_explicit_and_free_of_favored_labels(
    factory_name: str,
) -> None:
    """Implicit labels or seed drift would bias later negative-control analysis."""
    controls = import_module("pi_engine.synthetic.controls")
    systems = import_module("pi_engine.synthetic.systems")
    factory = getattr(controls, factory_name)
    first = factory(seed=7)
    same = factory(seed=7)
    different = factory(seed=8)

    assert first.is_control is True
    assert first.regime == "negative_control"
    assert first.ground_truth == {"horizon_steps": 1}
    assert first.model.expected_regimes == ("negative_control",)
    assert first.model_dump_json() == same.model_dump_json()
    assert first.model_dump_json() != different.model_dump_json()
    assert systems.SyntheticSystem.model_validate_json(first.model_dump_json()) == first


def test_control_catalog_names_every_negative_control() -> None:
    """An incomplete control catalog would let comprehensive runs skip a falsifier."""
    controls = import_module("pi_engine.synthetic.controls")

    assert tuple(controls.CONTROL_CATALOG) == (
        "random_graph_control",
        "shuffled_time_series_control",
        "irrelevant_proximity_control",
        "no_paired_structure_control",
    )
    assert [
        factory(seed=3).name for factory in controls.CONTROL_CATALOG.values()
    ] == list(controls.CONTROL_CATALOG)
    with pytest.raises(TypeError):
        controls.CONTROL_CATALOG["missing"] = controls.random_graph_control
