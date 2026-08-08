from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from pi_engine.schemas.case import Case, VariableDefinition
from pi_engine.schemas.common import Confidence, Provenance, UncertaintyClass
from pi_engine.schemas.graph import GraphEdge, GraphNode, SystemGraph
from pi_engine.schemas.observation import Observation
from pi_engine.schemas.state import StateEstimate


CUTOFF = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)


def make_observation(**overrides: object) -> Observation:
    values: dict[str, object] = {
        "observation_id": "obs-flow-1",
        "case_id": "case-river-1",
        "variable": "flow",
        "value": 12.5,
        "unit": "m^3/s",
        "event_time": CUTOFF - timedelta(hours=2),
        "available_at": CUTOFF - timedelta(hours=1),
        "uncertainty_class": UncertaintyClass.MEASUREMENT,
        "confidence": Confidence(score=0.92, basis="calibrated gauge"),
        "provenance": Provenance(
            source="USGS stream gauge 04249000",
            observed_at=CUTOFF - timedelta(hours=2),
            reference="https://waterdata.usgs.gov/monitoring-location/04249000/",
        ),
    }
    values.update(overrides)
    return Observation.model_validate(values)


def make_case(**overrides: object) -> Case:
    values: dict[str, object] = {
        "case_id": "case-river-1",
        "title": "River response",
        "domain": "hydrology",
        "provenance": Provenance(
            source="PI-engine fixture",
            observed_at=CUTOFF - timedelta(days=1),
        ),
        "prediction_cutoff": CUTOFF,
        "canonical_variables": (
            VariableDefinition(name="flow", unit="m^3/s"),
            VariableDefinition(name="rainfall", unit="mm/h"),
        ),
        "observations": (make_observation(),),
        "state": StateEstimate(
            at=CUTOFF,
            observed={"flow": 12.5},
            latent={"rainfall": 2.0},
            uncertainty={"flow": 0.4, "rainfall": 0.8},
            boundary={"rainfall": 2.0},
        ),
        "graph": SystemGraph(
            nodes=(
                GraphNode(node_id="rain", variable_refs=("rainfall",)),
                GraphNode(node_id="river", variable_refs=("flow",)),
            ),
            edges=(
                GraphEdge(
                    source="rain",
                    target="river",
                    coupling_type="runoff",
                    strength=0.7,
                    effective_proximity=1.0,
                ),
            ),
        ),
        "constraints": ("flow >= 0",),
    }
    values.update(overrides)
    return Case.model_validate(values)


def test_case_preserves_observation_units_provenance_and_uncertainty() -> None:
    """Dropping measurement context would make a normalized case unauditable."""
    dumped = make_case().model_dump(mode="json")

    assert dumped["canonical_variables"] == [
        {"name": "flow", "unit": "m^3/s", "description": None},
        {"name": "rainfall", "unit": "mm/h", "description": None},
    ]
    assert dumped["observations"][0]["unit"] == "m^3/s"
    assert dumped["observations"][0]["uncertainty_class"] == "measurement"
    assert dumped["observations"][0]["confidence"] == {
        "score": 0.92,
        "basis": "calibrated gauge",
    }
    assert dumped["observations"][0]["provenance"]["source"] == (
        "USGS stream gauge 04249000"
    )


def test_observation_accepts_finite_numeric_vector_and_round_trips_as_json() -> None:
    """Rejecting vector measurements would break hybrid scalar/vector cases."""
    observation = make_observation(value=[12.5, 13.25])

    assert observation.value == (12.5, 13.25)
    assert Observation.model_validate_json(observation.model_dump_json()) == observation
    assert observation.model_dump(mode="json")["value"] == [12.5, 13.25]


@pytest.mark.parametrize("nonfinite", [float("inf"), float("-inf"), float("nan")])
def test_observation_rejects_nonfinite_numeric_vector(nonfinite: float) -> None:
    """A nonfinite vector element would not survive a standards-compliant JSON round trip."""
    with pytest.raises(ValidationError) as exc_info:
        make_observation(value=[1.0, nonfinite])

    assert "finite_number" in {error["type"] for error in exc_info.value.errors()}


@pytest.mark.parametrize("time_field", ["event_time", "available_at"])
def test_observation_rejects_naive_event_or_availability_time(time_field: str) -> None:
    """A naïve event or availability time would make cutoff enforcement ambiguous."""
    with pytest.raises(ValidationError):
        make_observation(**{time_field: datetime(2026, 8, 8, 10, 0)})


def test_case_rejects_observation_available_after_cutoff_despite_earlier_event() -> None:
    """Checking event time alone would leak information unavailable at prediction time."""
    late_observation = make_observation(
        event_time=CUTOFF - timedelta(days=2),
        available_at=CUTOFF + timedelta(seconds=1),
    )

    with pytest.raises(ValidationError, match="available_at"):
        make_case(observations=(late_observation,))


def test_case_rejects_observation_from_a_different_case() -> None:
    """Ignoring observation case IDs would mix evidence across prediction cases."""
    with pytest.raises(ValidationError, match="case_id"):
        make_case(observations=(make_observation(case_id="case-river-2"),))


def test_case_rejects_observation_with_unknown_variable() -> None:
    """An unresolved observation variable would bypass the canonical variable registry."""
    with pytest.raises(ValidationError, match="variable"):
        make_case(
            observations=(
                make_observation(variable="temperature", unit="degC"),
            )
        )


def test_case_rejects_observation_unit_that_disagrees_with_canonical_unit() -> None:
    """Accepting a unit mismatch would silently corrupt normalized numeric values."""
    with pytest.raises(ValidationError, match="unit"):
        make_case(observations=(make_observation(unit="ft^3/s"),))


def test_state_rejects_negative_uncertainty() -> None:
    """Negative uncertainty magnitudes are not physically meaningful."""
    with pytest.raises(ValidationError, match="uncertainty"):
        StateEstimate(
            at=CUTOFF,
            observed={"flow": 12.5},
            latent={},
            uncertainty={"flow": -0.1},
            boundary={},
        )


def test_state_rejects_negative_uncertainty_vector() -> None:
    """Skipping vector elements would allow invalid negative uncertainty magnitudes."""
    with pytest.raises(ValidationError, match="uncertainty"):
        StateEstimate(
            at=CUTOFF,
            observed={"flow": [12.5, 13.25]},
            latent={},
            uncertainty={"flow": [0.4, -0.1]},
            boundary={},
        )


def test_state_rejects_naive_estimate_time() -> None:
    """A naïve state time would make its relationship to the cutoff ambiguous."""
    with pytest.raises(ValidationError):
        StateEstimate(
            at=datetime(2026, 8, 8, 12, 0),
            observed={"flow": 12.5},
            latent={},
            uncertainty={"flow": 0.4},
            boundary={},
        )


def test_state_accepts_numeric_vectors_and_round_trips_as_json() -> None:
    """Collapsing vector state components to scalars would lose hybrid state structure."""
    state = StateEstimate(
        at=CUTOFF,
        observed={"flow": [12.5, 13.25]},
        latent={"rainfall": [2.0, 2.5]},
        uncertainty={"flow": [0.4, 0.5]},
        boundary={"rainfall": [1.0, 1.5]},
    )

    assert state.observed["flow"] == (12.5, 13.25)
    assert StateEstimate.model_validate_json(state.model_dump_json()) == state
    assert state.model_dump(mode="json") == {
        "at": "2026-08-08T12:00:00Z",
        "observed": {"flow": [12.5, 13.25]},
        "latent": {"rainfall": [2.0, 2.5]},
        "uncertainty": {"flow": [0.4, 0.5]},
        "boundary": {"rainfall": [1.0, 1.5]},
    }


def test_state_accepts_integer_scalar_and_vector_inputs_as_floats() -> None:
    """Making finite floats strict must not reject ordinary integer measurements."""
    state = StateEstimate(
        at=CUTOFF,
        observed={"flow": 12},
        latent={"rainfall": [2, 3]},
        uncertainty={},
        boundary={},
    )

    assert state.observed["flow"] == 12.0
    assert state.latent["rainfall"] == (2.0, 3.0)


@pytest.mark.parametrize("value", [True, [1.0, True]])
def test_state_rejects_boolean_scalar_or_vector_elements(value: object) -> None:
    """Coercing booleans to 0/1 would admit nonnumeric current-state values."""
    with pytest.raises(ValidationError):
        StateEstimate(
            at=CUTOFF,
            observed={"flow": value},
            latent={},
            uncertainty={},
            boundary={},
        )


@pytest.mark.parametrize("component", ["observed", "latent", "uncertainty", "boundary"])
def test_state_rejects_nonfinite_vector_in_each_component(component: str) -> None:
    """A nonfinite state vector would make simulation inputs non-serializable."""
    values: dict[str, object] = {
        "at": CUTOFF,
        "observed": {},
        "latent": {},
        "uncertainty": {},
        "boundary": {},
    }
    values[component] = {"flow": [1.0, float("inf")]}

    with pytest.raises(ValidationError) as exc_info:
        StateEstimate.model_validate(values)

    assert "finite_number" in {error["type"] for error in exc_info.value.errors()}


@pytest.mark.parametrize("component", ["observed", "latent", "uncertainty", "boundary"])
def test_state_component_mappings_cannot_be_mutated_after_validation(
    component: str,
) -> None:
    """Mutable state maps would allow callers to bypass canonical and uncertainty validation."""
    state = make_case().state

    with pytest.raises(TypeError):
        getattr(state, component)["not_canonical"] = -1.0


@pytest.mark.parametrize("component", ["observed", "latent", "uncertainty", "boundary"])
def test_case_rejects_state_component_with_unknown_variable(component: str) -> None:
    """Every state component must resolve to the case's canonical variables."""
    state_values: dict[str, object] = {
        "at": CUTOFF,
        "observed": {"flow": 12.5},
        "latent": {"rainfall": 2.0},
        "uncertainty": {"flow": 0.4},
        "boundary": {"rainfall": 2.0},
    }
    state_values[component] = {"unknown": 1.0}

    with pytest.raises(ValidationError, match=component):
        make_case(state=StateEstimate.model_validate(state_values))


def test_case_rejects_state_estimated_after_prediction_cutoff() -> None:
    """A post-cutoff state estimate would structurally leak future information."""
    with pytest.raises(ValidationError, match="state"):
        make_case(
            state=StateEstimate(
                at=CUTOFF + timedelta(seconds=1),
                observed={"flow": 12.5},
                latent={},
                uncertainty={"flow": 0.4},
                boundary={},
            )
        )


def test_case_rejects_naive_prediction_cutoff() -> None:
    """A naïve cutoff would make availability comparisons timezone-dependent."""
    with pytest.raises(ValidationError):
        make_case(
            prediction_cutoff=datetime(2026, 8, 8, 12, 0),
            observations=(),
            state=StateEstimate(
                at=datetime(2026, 8, 8, 12, 0),
                observed={"flow": 12.5},
                latent={},
                uncertainty={"flow": 0.4},
                boundary={},
            ),
        )


def test_system_graph_rejects_edge_with_missing_endpoint() -> None:
    """An edge pointing outside the node set would leave topology unresolved."""
    with pytest.raises(ValidationError, match="endpoint"):
        SystemGraph(
            nodes=(GraphNode(node_id="river", variable_refs=("flow",)),),
            edges=(
                GraphEdge(
                    source="rain",
                    target="river",
                    coupling_type="runoff",
                ),
            ),
        )


def test_system_graph_rejects_duplicate_node_ids() -> None:
    """Duplicate node IDs would make edge references ambiguous."""
    with pytest.raises(ValidationError, match="node_id"):
        SystemGraph(
            nodes=(
                GraphNode(node_id="river", variable_refs=("flow",)),
                GraphNode(node_id="river", variable_refs=("rainfall",)),
            ),
            edges=(),
        )


def test_system_graph_round_trips_explicit_boundary_and_geometry_context() -> None:
    """Dropping boundary or geometry/topology context would make the graph incomplete."""
    graph = SystemGraph.model_validate(
        {
            "nodes": ({"node_id": "river", "variable_refs": ("flow",)},),
            "edges": (),
            "boundary_relationships": (
                {
                    "node_id": "river",
                    "variable_ref": "rainfall",
                    "relationship_type": "inflow_boundary",
                },
            ),
            "geometry_metadata": {"coordinate_system": "local", "dimensions": 2},
            "topology_metadata": {"network_type": "directed", "acyclic": True},
        }
    )

    assert SystemGraph.model_validate_json(graph.model_dump_json()) == graph
    assert graph.model_dump(mode="json") == {
        "nodes": [{"node_id": "river", "variable_refs": ["flow"]}],
        "edges": [],
        "boundary_relationships": [
            {
                "node_id": "river",
                "variable_ref": "rainfall",
                "relationship_type": "inflow_boundary",
            }
        ],
        "geometry_metadata": {"coordinate_system": "local", "dimensions": 2},
        "topology_metadata": {"network_type": "directed", "acyclic": True},
    }


@pytest.mark.parametrize("metadata_field", ["geometry_metadata", "topology_metadata"])
def test_system_graph_metadata_cannot_be_mutated_after_validation(
    metadata_field: str,
) -> None:
    """Mutable graph metadata would allow the validated system definition to drift."""
    graph = SystemGraph.model_validate(
        {
            "nodes": (),
            "edges": (),
            metadata_field: {"kind": "declared"},
        }
    )

    with pytest.raises(TypeError):
        getattr(graph, metadata_field)["kind"] = "mutated"


@pytest.mark.parametrize("metadata_field", ["geometry_metadata", "topology_metadata"])
def test_system_graph_omitted_metadata_defaults_are_immutable_and_round_trip(
    metadata_field: str,
) -> None:
    """Skipping default validation would expose mutable graph metadata maps."""
    graph = SystemGraph(nodes=(), edges=())

    with pytest.raises(TypeError):
        getattr(graph, metadata_field)["kind"] = "mutated"

    assert SystemGraph.model_validate_json(graph.model_dump_json()) == graph


def test_system_graph_rejects_boundary_relationship_with_unknown_node() -> None:
    """A boundary relationship pointing outside the node set would be unresolved."""
    with pytest.raises(ValidationError, match="unknown node"):
        SystemGraph.model_validate(
            {
                "nodes": (),
                "edges": (),
                "boundary_relationships": (
                    {
                        "node_id": "missing",
                        "variable_ref": "flow",
                        "relationship_type": "external_input",
                    },
                ),
            }
        )


def test_case_rejects_graph_node_with_unknown_variable_reference() -> None:
    """Graph nodes must refer to the same canonical variables as state and observations."""
    with pytest.raises(ValidationError, match="variable_refs"):
        make_case(
            graph=SystemGraph(
                nodes=(
                    GraphNode(node_id="weather", variable_refs=("temperature",)),
                ),
                edges=(),
            )
        )


def test_case_rejects_boundary_relationship_with_unknown_variable_reference() -> None:
    """Boundary variable references must share the case's canonical variable registry."""
    graph = SystemGraph.model_validate(
        {
            "nodes": ({"node_id": "river", "variable_refs": ("flow",)},),
            "edges": (),
            "boundary_relationships": (
                {
                    "node_id": "river",
                    "variable_ref": "not_canonical",
                    "relationship_type": "external_input",
                },
            ),
        }
    )

    with pytest.raises(ValidationError, match="unknown variables"):
        make_case(graph=graph)


def test_case_rejects_duplicate_canonical_variable_names() -> None:
    """Duplicate variable names would make unit and reference resolution ambiguous."""
    with pytest.raises(ValidationError, match="canonical_variables"):
        make_case(
            canonical_variables=(
                VariableDefinition(name="flow", unit="m^3/s"),
                VariableDefinition(name="flow", unit="ft^3/s"),
            )
        )


def test_case_rejects_withheld_outcomes_in_prediction_input() -> None:
    """Embedding withheld outcomes would defeat cutoff-safe prediction evaluation."""
    payload = make_case().model_dump()
    payload["observations"][0]["withheld_outcomes"] = (
        {"variable": "flow", "value": 14.0},
    )

    with pytest.raises(ValidationError) as exc_info:
        Case.model_validate(payload)

    assert exc_info.value.errors()[0]["loc"] == (
        "observations",
        0,
        "withheld_outcomes",
    )
    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"


def test_case_rejects_outcome_bearing_observation_subclass_instance() -> None:
    """Trusting a validated subclass would retain outcome data across the cutoff boundary."""

    class OutcomeBearingObservation(Observation):
        withheld_outcomes: tuple[dict[str, object], ...]

    values = make_observation().model_dump()
    values["withheld_outcomes"] = ({"variable": "flow", "value": 14.0},)
    outcome_bearing = OutcomeBearingObservation.model_validate(values)

    with pytest.raises(ValidationError) as exc_info:
        make_case(observations=(outcome_bearing,))

    assert exc_info.value.errors()[0]["loc"] == (
        "observations",
        0,
        "withheld_outcomes",
    )
    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"
