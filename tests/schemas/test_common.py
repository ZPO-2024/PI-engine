from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from pi_engine.schemas.common import (
    ClosureType,
    Confidence,
    ModelStatus,
    Provenance,
    UncertaintyClass,
)


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_confidence_rejects_scores_outside_probability_bounds(score: float) -> None:
    """A missing 0..1 bound would permit invalid confidence claims."""
    with pytest.raises(ValidationError):
        Confidence(score=score)


def test_provenance_serializes_its_source_and_observation_time() -> None:
    """Dropping source or timestamp would make model inputs unauditable."""
    provenance = Provenance(
        source="USGS stream gauge 04249000",
        observed_at=datetime(2026, 8, 8, 12, 30, tzinfo=UTC),
        reference="https://waterdata.usgs.gov/monitoring-location/04249000/",
    )

    assert provenance.model_dump(mode="json") == {
        "source": "USGS stream gauge 04249000",
        "observed_at": "2026-08-08T12:30:00Z",
        "reference": "https://waterdata.usgs.gov/monitoring-location/04249000/",
    }


def test_provenance_rejects_a_naive_observation_time() -> None:
    """Accepting a naïve time would make prediction-cutoff comparisons ambiguous."""
    with pytest.raises(ValidationError):
        Provenance(
            source="USGS stream gauge 04249000",
            observed_at=datetime(2026, 8, 8, 12, 30),
        )


def test_shared_enums_serialize_canonical_values() -> None:
    """Changing these values would blur uncertainty or closure semantics."""
    assert UncertaintyClass.MEASUREMENT.value == "measurement"
    assert UncertaintyClass.PARAMETER.value == "parameter"
    assert UncertaintyClass.PROCESS_NOISE.value == "process_noise"
    assert UncertaintyClass.MODEL_DISCREPANCY.value == "model_discrepancy"
    assert UncertaintyClass.STRUCTURED_UNKNOWN.value == "structured_unknown"
    assert ModelStatus.EXPERIMENTAL.value == "experimental"
    assert ModelStatus.VALIDATED.value == "validated"
    assert ModelStatus.DEGRADED.value == "degraded"
    assert ModelStatus.REJECTED.value == "rejected"
    assert ClosureType.EPISTEMIC.value == "epistemic"
    assert ClosureType.CAUSAL.value == "causal"
