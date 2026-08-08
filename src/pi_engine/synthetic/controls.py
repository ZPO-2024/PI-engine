"""Seeded negative controls with no favored invariant ground-truth labels."""

from collections.abc import Callable
from types import MappingProxyType
from typing import Mapping

import numpy as np

from pi_engine.schemas.graph import GraphEdge, GraphNode, SystemGraph
from pi_engine.synthetic.systems import SyntheticSystem, _build_fixture


_INDEPENDENT_GAUSSIAN_SHA256 = (
    "d2ef318f3b9ba16d3769f637b66532982a514fbd5ed556741dffeb26d047d4b2"
)
_PERMUTATION_SERIES_SHA256 = (
    "f62240a0b0eccd18acd114f33244747aa4c3cb78f1ba3e0579e4980a1fc064f4"
)
_IDENTITY_VECTOR_SHA256 = (
    "da4cffb3625ed0ddb8be7c82be72842380c0727952958a1e4a1e9993f1c3aa68"
)


def random_graph_control(seed: int = 0) -> SyntheticSystem:
    """Return random directed topology with independent Gaussian vector truth.

    ``independent_gaussian`` ignores graph edges and draws each vector component
    independently from ``Normal(mean, standard_deviation)`` using the run seed.
    """
    graph_rng = np.random.default_rng(seed)
    edges: list[GraphEdge] = []
    for source in range(5):
        for target in range(5):
            if source == target:
                continue
            if graph_rng.random() < 0.3:
                edges.append(
                    GraphEdge(
                        source=f"node-{source}",
                        target=f"node-{target}",
                        coupling_type="random_control_edge",
                        strength=float(graph_rng.uniform(-1.0, 1.0)),
                        effective_proximity=float(graph_rng.uniform(0.1, 5.0)),
                    )
                )
    outcome_rng = np.random.default_rng(seed)
    outcome = tuple(float(value) for value in outcome_rng.normal(0.0, 1.0, 5))
    zero_vector = (0.0, 0.0, 0.0, 0.0, 0.0)
    return _build_fixture(
        name="random_graph_control",
        regime="negative_control",
        seed=seed,
        variables=(("values", "a.u."),),
        observations=((1, "values", zero_vector),),
        state={"values": zero_vector},
        graph=SystemGraph(
            nodes=tuple(
                GraphNode(node_id=f"node-{index}", variable_refs=("values",))
                for index in range(5)
            ),
            edges=tuple(edges),
            topology_metadata={"randomized": True},
        ),
        executor_ref="independent_gaussian",
        code_sha256=_INDEPENDENT_GAUSSIAN_SHA256,
        equations=("next[i] ~ Normal(mean, standard_deviation) independently",),
        rules=("draw one value per vector component in index order",),
        transition_metadata={
            "state_variable": "values",
            "mean": 0.0,
            "standard_deviation": 1.0,
            "step_seconds": 3600,
        },
        classification="stochastic",
        outcomes=((1, "values", outcome),),
        model_family="independent_noise_control",
        is_control=True,
    )


def shuffled_time_series_control(seed: int = 0) -> SyntheticSystem:
    """Return a permutation of 0..7 with temporal order deliberately destroyed.

    ``permutation_series`` recreates ``default_rng(seed).permutation(population_size)``;
    it increments ``index_variable`` and exposes the corresponding element.
    """
    rng = np.random.default_rng(seed)
    shuffled = tuple(float(value) for value in rng.permutation(8))
    observations = tuple(
        (7 - index, "signal", value)
        for index, value in enumerate(shuffled[:-1])
    )
    return _build_fixture(
        name="shuffled_time_series_control",
        regime="negative_control",
        seed=seed,
        variables=(("signal", "a.u."), ("draw_index", "index")),
        observations=observations,
        state={"signal": shuffled[-2], "draw_index": 6.0},
        graph=SystemGraph(
            nodes=(
                GraphNode(
                    node_id="permuted-series",
                    variable_refs=("signal", "draw_index"),
                ),
            ),
            edges=(),
            topology_metadata={"shuffled": True},
        ),
        executor_ref="permutation_series",
        code_sha256=_PERMUTATION_SERIES_SHA256,
        equations=(
            "index_next = index + 1",
            "signal_next = default_rng(seed).permutation(population_size)[index_next]",
        ),
        rules=("permutation indices are zero-based and never wrap",),
        transition_metadata={
            "state_variable": "signal",
            "index_variable": "draw_index",
            "population_size": 8,
            "step_seconds": 3600,
        },
        classification="stochastic",
        outcomes=((1, "signal", shuffled[-1]),),
        model_family="permutation_control",
        is_control=True,
    )


def irrelevant_proximity_control(seed: int = 0) -> SyntheticSystem:
    """Return varying edge proximities and a transition that ignores all of them."""
    rng = np.random.default_rng(seed)
    pairs = ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3))
    proximities = rng.uniform(0.05, 5.0, len(pairs))
    edges = tuple(
        GraphEdge(
            source=f"node-{source}",
            target=f"node-{target}",
            coupling_type="declared_but_unused_proximity",
            strength=None,
            effective_proximity=float(proximity),
        )
        for (source, target), proximity in zip(pairs, proximities, strict=True)
    )
    values = (2.0, -1.0, 4.0, 0.5)
    return _build_fixture(
        name="irrelevant_proximity_control",
        regime="negative_control",
        seed=seed,
        variables=(("values", "a.u."),),
        observations=((1, "values", values),),
        state={"values": values},
        graph=SystemGraph(
            nodes=tuple(
                GraphNode(node_id=f"node-{index}", variable_refs=("values",))
                for index in range(4)
            ),
            edges=edges,
            topology_metadata={"proximity_irrelevant": True},
        ),
        executor_ref="identity_vector",
        code_sha256=_IDENTITY_VECTOR_SHA256,
        equations=("next = current",),
        rules=("copy the full vector without reading graph edge metadata",),
        transition_metadata={
            "state_variable": "values",
            "step_seconds": 3600,
        },
        classification="deterministic",
        outcomes=((1, "values", values),),
        model_family="identity_control",
        effective_proximity_dependencies=(),
        is_control=True,
    )


def no_paired_structure_control(seed: int = 0) -> SyntheticSystem:
    """Return a transitive tournament: every node has a unique directed role."""
    edges = tuple(
        GraphEdge(
            source=f"node-{source}",
            target=f"node-{target}",
            coupling_type="asymmetric_control_edge",
        )
        for source in range(5)
        for target in range(source + 1, 5)
    )
    values = (2.0, -1.0, 4.0, 0.5, -3.0)
    return _build_fixture(
        name="no_paired_structure_control",
        regime="negative_control",
        seed=seed,
        variables=(("values", "a.u."),),
        observations=((1, "values", values),),
        state={"values": values},
        graph=SystemGraph(
            nodes=tuple(
                GraphNode(node_id=f"node-{index}", variable_refs=("values",))
                for index in range(5)
            ),
            edges=edges,
            topology_metadata={"unique_directed_roles": True},
        ),
        executor_ref="identity_vector",
        code_sha256=_IDENTITY_VECTOR_SHA256,
        equations=("next = current",),
        rules=("copy the full vector without pairing components",),
        transition_metadata={
            "state_variable": "values",
            "step_seconds": 3600,
        },
        classification="deterministic",
        outcomes=((1, "values", values),),
        model_family="identity_control",
        is_control=True,
    )


CONTROL_CATALOG: Mapping[str, Callable[[int], SyntheticSystem]] = MappingProxyType(
    {
        "random_graph_control": random_graph_control,
        "shuffled_time_series_control": shuffled_time_series_control,
        "irrelevant_proximity_control": irrelevant_proximity_control,
        "no_paired_structure_control": no_paired_structure_control,
    }
)
