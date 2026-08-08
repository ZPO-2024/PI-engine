from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from pi_engine.schemas.common import Confidence, ModelStatus, Provenance
from pi_engine.schemas.model import (
    ApplicabilitySpec,
    DynamicsSpec,
    ExplicitModel,
    FalsificationSpec,
    ModelPerformance,
    UncertaintySpec,
)


AS_OF = datetime(2026, 8, 8, 15, 0, tzinfo=UTC)
CODE_SHA256 = "2f" * 32


def model_payload() -> dict[str, object]:
    return {
        "model_id": "river-linear-decay",
        "version": "1.2.0",
        "name": "Linear river-flow relaxation",
        "domain": "hydrology",
        "model_family": "linear-state-space",
        "provenance": Provenance(
            source="Hydrology model catalog",
            observed_at=AS_OF,
            reference="catalog:model/river-linear-decay/1.2.0",
        ),
        "initial_confidence": Confidence(
            score=0.72,
            basis="benchmarked against documented watersheds",
        ),
        "applicability": {
            "required_variables": ("flow", "rainfall"),
            "optional_variables": ("temperature",),
            "valid_ranges": {"flow": (0.0, 10_000.0), "rainfall": (0.0, None)},
            "topology_requirements": ("directed runoff path to river",),
            "boundary_conditions": ("upstream inflow is specified",),
            "exclusion_rules": ("tidal backflow dominates",),
        },
        "dynamics": {
            "executor_ref": "linear_decay",
            "executor_version": "1",
            "code_sha256": CODE_SHA256,
            "equations": ("d(flow)/dt = rainfall - decay_rate * flow",),
            "rules": ("flow remains nonnegative",),
            "transition_metadata": {"integration": "euler", "step_seconds": 60},
            "time_behavior": "continuous-time sampled at fixed intervals",
            "classification": "deterministic",
        },
        "uncertainty": {
            "parameter_uncertainty": ("decay_rate interval",),
            "process_disturbance": ("unmeasured tributary inflow",),
            "model_discrepancy": ("linearization error at flood stage",),
            "structured_unknowns": ("unknown upstream release schedule",),
        },
        "assumptions": ("rainfall is spatially uniform",),
        "relationships_used": ("rainfall -> river",),
        "phase_dependencies": (),
        "effective_proximity_dependencies": (),
        "information_dependencies": ("stream gauge availability",),
        "predicted_outputs": ("flow",),
        "prediction_horizon": "PT6H",
        "expected_regimes": ("relaxation", "forced response"),
        "falsification": {
            "falsifiers": ("flow rises with zero inflow after disturbances cease",),
            "contradictory_evidence_conditions": (
                "persistent negative fitted decay rate",
            ),
            "failure_conditions": ("predicted flow becomes negative",),
        },
    }


def test_explicit_model_round_trip_preserves_inspectable_definition() -> None:
    """Dropping model inputs or audit metadata would make execution irreproducible."""
    model = ExplicitModel.model_validate(model_payload())
    dumped = model.model_dump(mode="json")

    assert dumped["version"] == "1.2.0"
    assert dumped["provenance"]["source"] == "Hydrology model catalog"
    assert dumped["applicability"] == {
        "required_variables": ["flow", "rainfall"],
        "optional_variables": ["temperature"],
        "valid_ranges": {"flow": [0.0, 10_000.0], "rainfall": [0.0, None]},
        "topology_requirements": ["directed runoff path to river"],
        "boundary_conditions": ["upstream inflow is specified"],
        "exclusion_rules": ["tidal backflow dominates"],
    }
    assert dumped["dynamics"] == {
        "executor_ref": "linear_decay",
        "executor_version": "1",
        "code_sha256": CODE_SHA256,
        "equations": ["d(flow)/dt = rainfall - decay_rate * flow"],
        "rules": ["flow remains nonnegative"],
        "transition_metadata": {"integration": "euler", "step_seconds": 60},
        "time_behavior": "continuous-time sampled at fixed intervals",
        "classification": "deterministic",
    }
    assert dumped["uncertainty"] == {
        "parameter_uncertainty": ["decay_rate interval"],
        "process_disturbance": ["unmeasured tributary inflow"],
        "model_discrepancy": ["linearization error at flood stage"],
        "structured_unknowns": ["unknown upstream release schedule"],
    }
    assert dumped["falsification"] == {
        "falsifiers": ["flow rises with zero inflow after disturbances cease"],
        "contradictory_evidence_conditions": [
            "persistent negative fitted decay rate"
        ],
        "failure_conditions": ["predicted flow becomes negative"],
    }
    assert dumped["predicted_outputs"] == ["flow"]
    assert ExplicitModel.model_validate_json(model.model_dump_json()) == model


@pytest.mark.parametrize(
    "missing_field",
    ["version", "provenance", "applicability", "falsification", "predicted_outputs"],
)
def test_explicit_model_requires_core_audit_fields(missing_field: str) -> None:
    """An omitted core field would create an incomplete model definition."""
    payload = model_payload()
    del payload[missing_field]

    with pytest.raises(ValidationError) as exc_info:
        ExplicitModel.model_validate(payload)

    assert exc_info.value.errors()[0]["loc"] == (missing_field,)
    assert exc_info.value.errors()[0]["type"] == "missing"


def test_explicit_model_requires_at_least_one_falsifier_and_predicted_output() -> None:
    """Empty falsifiers or outputs would make the model untestable or non-predictive."""
    payload = model_payload()
    payload["predicted_outputs"] = ()
    payload["falsification"] = {
        "falsifiers": (),
        "contradictory_evidence_conditions": (),
        "failure_conditions": (),
    }

    with pytest.raises(ValidationError) as exc_info:
        ExplicitModel.model_validate(payload)

    assert {error["loc"] for error in exc_info.value.errors()} == {
        ("predicted_outputs",),
        ("falsification", "falsifiers"),
    }


@pytest.mark.parametrize("executor_ref", ["os.system", "pkg:handler", "../handler"])
def test_dynamics_rejects_import_or_path_like_executor_references(
    executor_ref: str,
) -> None:
    """An import or path reference could bypass registry-only executor resolution."""
    payload = model_payload()
    payload["dynamics"] = {**payload["dynamics"], "executor_ref": executor_ref}

    with pytest.raises(ValidationError, match="executor_ref"):
        ExplicitModel.model_validate(payload)


@pytest.mark.parametrize(
    "code_sha256",
    ["short", "G" * 64, "a" * 63, "a" * 65],
)
def test_dynamics_requires_exact_lowercase_sha256_identity(code_sha256: str) -> None:
    """An ambiguous code identity would break executor auditability."""
    payload = model_payload()
    payload["dynamics"] = {**payload["dynamics"], "code_sha256": code_sha256}

    with pytest.raises(ValidationError, match="code_sha256"):
        ExplicitModel.model_validate(payload)


def test_dynamics_requires_at_least_one_equation_or_rule() -> None:
    """Metadata without a declared equation or rule would hide transition behavior."""
    payload = model_payload()
    payload["dynamics"] = {
        **payload["dynamics"],
        "equations": (),
        "rules": (),
    }

    with pytest.raises(ValidationError, match="equation or rule"):
        ExplicitModel.model_validate(payload)


def test_applicability_rejects_overlapping_required_and_optional_variables() -> None:
    """A variable cannot have both required and optional applicability semantics."""
    payload = model_payload()
    payload["applicability"] = {
        **payload["applicability"],
        "optional_variables": ("flow",),
    }

    with pytest.raises(ValidationError, match="required and optional"):
        ExplicitModel.model_validate(payload)


def test_applicability_rejects_inverted_valid_range() -> None:
    """An inverted range could never match a valid case value."""
    payload = model_payload()
    payload["applicability"] = {
        **payload["applicability"],
        "valid_ranges": {"flow": (10.0, 1.0)},
    }

    with pytest.raises(ValidationError, match="valid range"):
        ExplicitModel.model_validate(payload)


def test_explicit_model_nested_containers_are_immutable() -> None:
    """Mutable nested metadata could silently change a versioned model after validation."""
    model = ExplicitModel.model_validate(model_payload())

    with pytest.raises(TypeError):
        model.applicability.valid_ranges["flow"] = (1.0, 2.0)
    with pytest.raises(TypeError):
        model.dynamics.transition_metadata["integration"] = "rk4"
    with pytest.raises(ValidationError):
        model.version = "2.0.0"


@pytest.mark.parametrize("status", [ModelStatus.DEGRADED, ModelStatus.REJECTED])
def test_nonusable_model_performance_remains_serializable_and_auditable(
    status: ModelStatus,
) -> None:
    """Degraded or rejected evidence must remain available instead of being discarded."""
    performance = ModelPerformance(
        model_id="river-linear-decay",
        model_version="1.2.0",
        as_of=AS_OF,
        status=status,
        cases_tested=("case-river-1", "case-river-2"),
        calibration_metrics={"brier_score": 0.28},
        prediction_errors={"rmse": 3.4},
        known_failure_regimes=("tidal backflow",),
        evidence=("eval:hydrology:2026-08-08",),
        provenance=Provenance(
            source="PI-engine holdout evaluation",
            observed_at=AS_OF,
            reference="evaluation:river-linear-decay:1.2.0",
        ),
    )

    dumped = performance.model_dump(mode="json")

    assert dumped["model_id"] == "river-linear-decay"
    assert dumped["model_version"] == "1.2.0"
    assert dumped["as_of"] == "2026-08-08T15:00:00Z"
    assert dumped["status"] == status.value
    assert dumped["known_failure_regimes"] == ["tidal backflow"]
    assert dumped["evidence"] == ["eval:hydrology:2026-08-08"]
    assert (
        ModelPerformance.model_validate_json(performance.model_dump_json())
        == performance
    )


def test_performance_is_separate_from_versioned_model_definition() -> None:
    """Embedding performance history would make an immutable model definition drift."""
    payload = model_payload()
    payload["performance"] = {"status": "degraded"}

    with pytest.raises(ValidationError) as exc_info:
        ExplicitModel.model_validate(payload)

    assert exc_info.value.errors()[0]["loc"] == ("performance",)
    assert exc_info.value.errors()[0]["type"] == "extra_forbidden"


def test_model_performance_metric_maps_are_immutable() -> None:
    """Mutable metrics would change the meaning of an as-of performance record."""
    performance = ModelPerformance(
        model_id="river-linear-decay",
        model_version="1.2.0",
        as_of=AS_OF,
        status=ModelStatus.VALIDATED,
        cases_tested=("case-river-1",),
        calibration_metrics={"coverage": 0.91},
        prediction_errors={"rmse": 1.2},
        known_failure_regimes=(),
        evidence=("eval:hydrology:2026-08-08",),
        provenance=Provenance(
            source="PI-engine holdout evaluation",
            observed_at=AS_OF,
        ),
    )

    with pytest.raises(TypeError):
        performance.calibration_metrics["coverage"] = 0.0
    with pytest.raises(TypeError):
        performance.prediction_errors["rmse"] = 99.0


def test_model_performance_rejects_naive_as_of_time() -> None:
    """A naive as-of time would make historical performance ordering ambiguous."""
    values = model_payload()

    with pytest.raises(ValidationError, match="as_of"):
        ModelPerformance(
            model_id="river-linear-decay",
            model_version="1.2.0",
            as_of=datetime(2026, 8, 8, 15, 0),
            status=ModelStatus.EXPERIMENTAL,
            cases_tested=(),
            calibration_metrics={},
            prediction_errors={},
            known_failure_regimes=(),
            evidence=(),
            provenance=values["provenance"],
        )


def test_all_model_schemas_forbid_extra_fields() -> None:
    """Unrecognized fields could hide unreviewed behavior or mutable state."""
    values = model_payload()
    samples = (
        (ExplicitModel, values),
        (ApplicabilitySpec, values["applicability"]),
        (DynamicsSpec, values["dynamics"]),
        (UncertaintySpec, values["uncertainty"]),
        (FalsificationSpec, values["falsification"]),
        (
            ModelPerformance,
            {
                "model_id": "river-linear-decay",
                "model_version": "1.2.0",
                "as_of": AS_OF,
                "status": "experimental",
                "cases_tested": (),
                "calibration_metrics": {},
                "prediction_errors": {},
                "known_failure_regimes": (),
                "evidence": (),
                "provenance": values["provenance"],
            },
        ),
    )

    for schema, payload in samples:
        with pytest.raises(ValidationError) as exc_info:
            schema.model_validate({**payload, "opaque_callable": "run_me"})
        assert exc_info.value.errors()[0]["loc"] == ("opaque_callable",)
        assert exc_info.value.errors()[0]["type"] == "extra_forbidden"
