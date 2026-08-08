import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from pi_engine.registry import ModelRegistry
from pi_engine.schemas.case import Case, VariableDefinition
from pi_engine.schemas.common import Confidence, Provenance
from pi_engine.schemas.graph import GraphNode, SystemGraph
from pi_engine.schemas.model import ExplicitModel
from pi_engine.schemas.state import StateEstimate


NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)


def make_model(
    *,
    model_id: str = "river-flow",
    version: str = "1.0.0",
    required_variables: tuple[str, ...] = ("flow",),
    optional_variables: tuple[str, ...] = (),
    valid_ranges: dict[str, tuple[float | None, float | None]] | None = None,
    topology_requirements: tuple[str, ...] = (),
    boundary_conditions: tuple[str, ...] = (),
    exclusion_rules: tuple[str, ...] = (),
) -> ExplicitModel:
    return ExplicitModel.model_validate(
        {
            "model_id": model_id,
            "version": version,
            "name": f"{model_id} {version}",
            "domain": "hydrology",
            "model_family": "state-space",
            "provenance": Provenance(source="test catalog", observed_at=NOW),
            "initial_confidence": Confidence(score=0.5, basis="fixture"),
            "applicability": {
                "required_variables": required_variables,
                "optional_variables": optional_variables,
                "valid_ranges": (
                    {"flow": (0.0, 100.0)}
                    if valid_ranges is None
                    else valid_ranges
                ),
                "topology_requirements": topology_requirements,
                "boundary_conditions": boundary_conditions,
                "exclusion_rules": exclusion_rules,
            },
            "dynamics": {
                "executor_ref": "river_flow",
                "executor_version": "1",
                "code_sha256": "2f" * 32,
                "equations": ("d(flow)/dt = -k * flow",),
                "rules": (),
                "transition_metadata": {},
                "time_behavior": "continuous",
                "classification": "deterministic",
            },
            "uncertainty": {
                "parameter_uncertainty": (),
                "process_disturbance": (),
                "model_discrepancy": (),
                "structured_unknowns": (),
            },
            "assumptions": (),
            "relationships_used": (),
            "phase_dependencies": (),
            "effective_proximity_dependencies": (),
            "information_dependencies": (),
            "predicted_outputs": ("flow",),
            "prediction_horizon": "PT1H",
            "expected_regimes": (),
            "falsification": {
                "falsifiers": ("flow increases without forcing",),
                "contradictory_evidence_conditions": (),
                "failure_conditions": (),
            },
        }
    )


def make_case(
    *,
    canonical_variables: tuple[str, ...] = ("flow", "rainfall"),
    observed: dict[str, float | tuple[float, ...]] | None = None,
    latent: dict[str, float | tuple[float, ...]] | None = None,
    boundary: dict[str, float | tuple[float, ...]] | None = None,
    topology_metadata: dict[str, str | int | float | bool | None] | None = None,
    constraints: tuple[str, ...] = (),
) -> Case:
    return Case(
        case_id="case-river",
        title="River fixture",
        domain="hydrology",
        provenance=Provenance(source="test case", observed_at=NOW),
        prediction_cutoff=NOW,
        canonical_variables=tuple(
            VariableDefinition(name=name, unit="unit")
            for name in canonical_variables
        ),
        observations=(),
        state=StateEstimate(
            at=NOW,
            observed={"flow": 10.0} if observed is None else observed,
            latent={} if latent is None else latent,
            uncertainty={},
            boundary={} if boundary is None else boundary,
        ),
        graph=SystemGraph(
            nodes=(GraphNode(node_id="river", variable_refs=("flow",)),),
            edges=(),
            topology_metadata=(
                {} if topology_metadata is None else topology_metadata
            ),
        ),
        constraints=constraints,
    )


def test_registry_gets_only_the_exact_registered_model_version() -> None:
    """Returning a latest or approximate version would make routing nondeterministic."""
    registry = ModelRegistry()
    version_one = make_model(version="1.0.0")
    version_two = make_model(version="2.0.0")

    registry.register(version_two)
    registry.register(version_one)

    assert registry.get("river-flow", "1.0.0") == version_one
    assert registry.get("river-flow", "2.0.0") == version_two
    with pytest.raises(KeyError):
        registry.get("river-flow", "1")


def test_registry_rejects_duplicate_exact_model_key() -> None:
    """Replacing an existing model in place would make prior routing unauditable."""
    registry = ModelRegistry()
    registry.register(make_model())

    with pytest.raises(ValueError, match="already registered"):
        registry.register(make_model())


def test_routing_retains_model_rejected_for_missing_required_variable() -> None:
    """Dropping a failed model would erase the reason it was not selected."""
    registry = ModelRegistry()
    registry.register(
        make_model(
            required_variables=("flow", "temperature"),
            valid_ranges={"flow": (0.0, 100.0)},
        )
    )

    results = registry.rank_applicable_models(make_case())

    assert len(results) == 1
    assert results[0].model_id == "river-flow"
    assert results[0].model_version == "1.0.0"
    assert results[0].applicable is False
    assert results[0].rank is None
    assert "missing required variable: temperature" in results[0].rejection_causes


@pytest.mark.parametrize(
    ("observed", "expected_cause"),
    [
        ({"flow": 101.0}, "outside valid range"),
        ({}, "unavailable for range check"),
    ],
)
def test_routing_rejects_out_of_range_or_unavailable_current_state_value(
    observed: dict[str, float], expected_cause: str
) -> None:
    """Assuming a missing or invalid current value is valid would bypass a hard gate."""
    registry = ModelRegistry()
    registry.register(make_model(valid_ranges={"flow": (0.0, 100.0)}))

    result = registry.rank_applicable_models(make_case(observed=observed))[0]

    assert result.applicable is False
    assert any(expected_cause in cause for cause in result.rejection_causes)


def test_routing_checks_every_vector_element_against_declared_range() -> None:
    """Checking only one vector element would admit an invalid current state."""
    registry = ModelRegistry()
    registry.register(make_model(valid_ranges={"flow": (0.0, 100.0)}))

    result = registry.rank_applicable_models(
        make_case(observed={"flow": (10.0, 101.0, 20.0)})
    )[0]

    assert result.applicable is False
    assert any("flow[1]" in cause for cause in result.rejection_causes)


def test_routing_requires_each_topology_metadata_token() -> None:
    """Ignoring an absent topology token would route structurally incompatible cases."""
    registry = ModelRegistry()
    registry.register(
        make_model(
            topology_requirements=("directed", "acyclic"),
            valid_ranges={"flow": (0.0, 100.0)},
        )
    )

    result = registry.rank_applicable_models(
        make_case(topology_metadata={"directed": True})
    )[0]

    assert result.applicable is False
    assert "missing topology requirement: acyclic" in result.rejection_causes
    assert "topology requirement satisfied: directed" in result.reasons


def test_routing_requires_boundary_tokens_and_rejects_exclusion_tokens() -> None:
    """Skipping constraint gates would apply a model in an excluded boundary regime."""
    registry = ModelRegistry()
    registry.register(
        make_model(
            boundary_conditions=("fixed inlet", "open outlet"),
            exclusion_rules=("tidal backflow",),
            valid_ranges={"flow": (0.0, 100.0)},
        )
    )

    result = registry.rank_applicable_models(
        make_case(constraints=("fixed inlet", "tidal backflow"))
    )[0]

    assert result.applicable is False
    assert "missing boundary condition: open outlet" in result.rejection_causes
    assert "exclusion rule present: tidal backflow" in result.rejection_causes
    assert "boundary condition satisfied: fixed inlet" in result.reasons


def test_optional_presence_scores_transparently_without_forcing_tie_winner() -> None:
    """Breaking ties or using hidden evidence would make structural routing opaque."""
    registry = ModelRegistry()
    registry.register(
        make_model(
            model_id="zeta",
            optional_variables=("rainfall",),
            valid_ranges={"flow": (0.0, 100.0)},
        )
    )
    registry.register(
        make_model(
            model_id="best",
            optional_variables=("rainfall", "temperature"),
            valid_ranges={"flow": (0.0, 100.0)},
        )
    )
    registry.register(
        make_model(
            model_id="alpha",
            optional_variables=("rainfall",),
            valid_ranges={"flow": (0.0, 100.0)},
        )
    )

    results = registry.rank_applicable_models(
        make_case(canonical_variables=("flow", "rainfall", "temperature"))
    )

    assert [(item.model_id, item.structural_score, item.rank) for item in results] == [
        ("best", 2, 1),
        ("alpha", 1, 2),
        ("zeta", 1, 2),
    ]
    assert "optional variable present: rainfall" in results[1].reasons
    assert results[1].rejection_causes == ()


def test_routing_returns_deeply_immutable_json_audit_records() -> None:
    """Mutable or non-JSON audit records could change after a routing decision."""
    registry = ModelRegistry()
    registry.register(make_model(optional_variables=("rainfall",)))

    result = registry.rank_applicable_models(make_case())[0]
    dumped = result.model_dump(mode="json")

    assert json.loads(json.dumps(dumped)) == dumped
    assert dumped == {
        "model_id": "river-flow",
        "model_version": "1.0.0",
        "applicable": True,
        "structural_score": 1,
        "structural_score_max": 1,
        "rank": 1,
        "reasons": [
            "required variable present: flow",
            "range satisfied: flow",
            "optional variable present: rainfall",
        ],
        "rejection_causes": [],
    }
    with pytest.raises(ValidationError):
        result.rank = 2
    with pytest.raises(TypeError):
        result.reasons[0] = "changed"
