"""Tests for explicit, provisional residual classification."""

from datetime import UTC, datetime, timedelta
from importlib import import_module
from typing import Any

import pytest

from pi_engine.schemas.common import Provenance
from pi_engine.schemas.outcome import Outcome
from pi_engine.schemas.residual import (
    Residual,
    ResidualCategory,
    ResidualClassification,
)


START = datetime(2026, 8, 12, 12, tzinfo=UTC)


def _provenance(source: str) -> Provenance:
    return Provenance(
        source=source,
        observed_at=START,
        reference="residual-analysis-fixture",
    )


def _residual() -> Residual:
    outcome = Outcome(
        outcome_id="outcome-flow-1",
        case_id="case-flow-1",
        variable="flow",
        unit="m^3/s",
        value=8.0,
        event_time=START + timedelta(hours=1),
        available_at=START + timedelta(hours=1, minutes=5),
        provenance=_provenance("held-out stream gauge"),
    )
    return Residual(
        residual_id="residual-flow-1",
        trajectory_id="trajectory-flow-1",
        prediction_time=START,
        model_id="flow-model",
        model_version="1.0.0",
        case_id="case-flow-1",
        variable="flow",
        unit="m^3/s",
        predicted_value=8.5,
        observed_outcome=outcome,
        error=0.5,
        error_convention="predicted_minus_observed",
        classification=ResidualClassification(
            category=ResidualCategory.STRUCTURED_UNKNOWN,
            basis="awaiting explicit residual evidence",
        ),
        provenance=_provenance("holdout comparison"),
    )


def _analyze(*evidence_kinds: str) -> Any:
    """Construct an analysis from real residual and provenance-bearing evidence."""
    residuals = import_module("pi_engine.analysis.residuals")
    evidence = tuple(
        residuals.ResidualEvidence(
            evidence_id=f"evidence-{index}",
            kind=kind,
            basis=f"explicit {kind.replace('_', ' ')} evidence",
            provenance=_provenance(f"residual monitor {index}"),
        )
        for index, kind in enumerate(evidence_kinds, start=1)
    )
    return residuals.analyze_residual(_residual(), evidence=evidence)


@pytest.mark.parametrize(
    ("evidence_kind", "expected_category"),
    [
        ("known_process_noise", ResidualCategory.PROCESS_NOISE),
        ("parameter_mismatch", ResidualCategory.PARAMETER_UNCERTAINTY),
        ("model_mismatch", ResidualCategory.MODEL_DISCREPANCY),
        ("phase_error", ResidualCategory.PHASE_TIMING),
        ("topology_error", ResidualCategory.TOPOLOGY_COUPLING),
    ],
    ids=(
        "known-noise",
        "parameter-mismatch",
        "model-mismatch",
        "phase-error",
        "topology-error",
    ),
)
def test_explicit_evidence_classifies_one_known_residual_cause(
    evidence_kind: str, expected_category: ResidualCategory
) -> None:
    """Removing a named rule must not silently erase its corresponding cause."""
    analysis = _analyze(evidence_kind)

    assert analysis.classification.category is expected_category
    assert analysis.residual.classification == analysis.classification
    assert analysis.residual.residual_id == "residual-flow-1"
    assert analysis.evidence[0].evidence_id == "evidence-1"
    assert "evidence-1" in analysis.classification.basis


def test_missing_evidence_retains_a_structured_unknown_residual() -> None:
    """Defaulting an unexplained residual to process noise would hide uncertainty."""
    analysis = _analyze()

    assert analysis.classification.category is ResidualCategory.STRUCTURED_UNKNOWN
    assert analysis.residual.classification == analysis.classification
    assert analysis.evidence == ()
    assert "no inspectable residual evidence" in analysis.classification.basis


def test_competing_evidence_retains_a_structured_unknown_residual() -> None:
    """Choosing one cause from conflicting evidence would create opaque routing."""
    analysis = _analyze("parameter_mismatch", "model_mismatch")

    assert analysis.classification.category is ResidualCategory.STRUCTURED_UNKNOWN
    assert analysis.evidence[0].evidence_id == "evidence-1"
    assert analysis.evidence[1].evidence_id == "evidence-2"
    assert "evidence-1" in analysis.classification.basis
    assert "evidence-2" in analysis.classification.basis
