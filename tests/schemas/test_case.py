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
    payload["withheld_outcomes"] = ({"variable": "flow", "value": 14.0},)

    with pytest.raises(ValidationError) as exc_info:
        Case.model_validate(payload)

    assert exc_info.value.errors()[0]["loc"] == ("withheld_outcomes",)
    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"
