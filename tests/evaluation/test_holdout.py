"""Temporal holdout tests exercise the public no-leakage boundary."""

from datetime import UTC, datetime, timedelta
import json
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from pi_engine.synthetic.systems import linear_convergence


def test_prepare_revalidates_case_to_exclude_late_published_observation() -> None:
    """Filtering on event time alone would leak a post-cutoff revision."""
    from pi_engine.evaluation.holdout import prepare_holdout

    fixture = linear_convergence()
    original = fixture.case.observations[0]
    late_revision = original.model_copy(
        update={
            "observation_id": "late-revision",
            "event_time": fixture.case.prediction_cutoff - timedelta(hours=2),
            "available_at": fixture.case.prediction_cutoff + timedelta(minutes=1),
            "value": 999_999.0,
        }
    )
    unvalidated_case = fixture.case.model_copy(
        update={"observations": (*fixture.case.observations, late_revision)}
    )

    with pytest.raises(ValidationError, match="available_at"):
        prepare_holdout(unvalidated_case, fixture.outcomes)


def test_prepared_holdout_dump_exposes_case_and_commitments_never_values() -> None:
    """Serializing a prepared split must not disclose withheld raw values."""
    from pi_engine.evaluation.holdout import prepare_holdout

    fixture = linear_convergence()
    sentinel = fixture.outcomes[0].model_copy(update={"value": 123_456.789})

    prepared = prepare_holdout(fixture.case, (sentinel, *fixture.outcomes[1:]))
    dumped = prepared.model_dump(mode="json")
    serialized = json.dumps(dumped, sort_keys=True)

    assert set(dumped) == {"case", "outcome_commitments"}
    assert dumped["case"]["case_id"] == fixture.case.case_id
    assert len(dumped["outcome_commitments"]) == 3
    assert "value" not in dumped["outcome_commitments"][0]
    assert "123456.789" not in serialized
    assert "123456.789" not in repr(prepared)
    assert not hasattr(prepared, "outcomes")
    with pytest.raises(ValidationError, match="frozen"):
        prepared.case = fixture.case  # type: ignore[misc]
    with pytest.raises(TypeError, match="sealed outcomes are immutable"):
        prepared._sealed_outcomes = ()


def test_commitment_is_deterministic_and_binds_full_canonical_outcome() -> None:
    """Ignoring a payload field would let a changed withheld record keep its digest."""
    from pi_engine.evaluation.holdout import prepare_holdout

    fixture = linear_convergence()
    original = prepare_holdout(fixture.case, fixture.outcomes)
    replay = prepare_holdout(fixture.case, fixture.outcomes)
    changed_outcome = fixture.outcomes[0].model_copy(update={"value": 1.25})
    changed = prepare_holdout(
        fixture.case, (changed_outcome, *fixture.outcomes[1:])
    )

    assert original.outcome_commitments[0].sha256 == (
        "5caf95ced616ecbdb7b202cbfa49765efd34432353dce538626f7a50779b23b4"
    )
    assert replay.outcome_commitments == original.outcome_commitments
    assert changed.outcome_commitments[0].sha256 != (
        original.outcome_commitments[0].sha256
    )


@pytest.mark.parametrize(
    ("outcome_update", "message"),
    [
        ({"case_id": "different-case"}, "case_id"),
        ({"variable": "unknown"}, "canonical"),
        ({"unit": "wrong-unit"}, "unit"),
    ],
    ids=("case", "variable", "unit"),
)
def test_prepare_rejects_outcomes_outside_case_contract(
    outcome_update: dict[str, object], message: str
) -> None:
    """Cross-case or noncanonical truth cannot evaluate this prediction view."""
    from pi_engine.evaluation.holdout import prepare_holdout

    fixture = linear_convergence()
    invalid = fixture.outcomes[0].model_copy(update=outcome_update)

    with pytest.raises(ValueError, match=message):
        prepare_holdout(fixture.case, (invalid,))


def test_prepare_rejects_duplicate_outcome_ids() -> None:
    """Duplicate identifiers would make a commitment-to-record audit ambiguous."""
    from pi_engine.evaluation.holdout import prepare_holdout

    fixture = linear_convergence()
    duplicate = fixture.outcomes[1].model_copy(
        update={"outcome_id": fixture.outcomes[0].outcome_id}
    )

    with pytest.raises(ValueError, match="outcome_id.*unique"):
        prepare_holdout(fixture.case, (fixture.outcomes[0], duplicate))


def test_prepare_requires_outcomes_strictly_after_cutoff() -> None:
    """An event at the forecast origin is not withheld future truth."""
    from pi_engine.evaluation.holdout import prepare_holdout

    fixture = linear_convergence()
    cutoff = fixture.case.prediction_cutoff
    at_cutoff = fixture.outcomes[0].model_copy(
        update={"event_time": cutoff, "available_at": cutoff + timedelta(hours=1)}
    )

    with pytest.raises(ValueError, match="event_time.*follow.*cutoff"):
        prepare_holdout(fixture.case, (at_cutoff,))


def test_prepare_allows_window_from_cutoff_to_future_outcome() -> None:
    """A forecast-origin window remains valid when its outcome event is future."""
    from pi_engine.evaluation.holdout import prepare_holdout
    from pi_engine.schemas.outcome import ComparisonWindow

    fixture = linear_convergence()
    cutoff = fixture.case.prediction_cutoff
    windowed = fixture.outcomes[0].model_copy(
        update={
            "comparison_window": ComparisonWindow(
                start_at=cutoff,
                end_at=fixture.outcomes[0].event_time,
            )
        }
    )

    prepared = prepare_holdout(fixture.case, (windowed,))

    assert prepared.outcome_commitments[0].comparison_window.start_at == cutoff


def test_prepare_rejects_impossible_case_observation_chronology() -> None:
    """Treating data as available before its event would corrupt a cutoff view."""
    from pi_engine.evaluation.holdout import prepare_holdout

    fixture = linear_convergence()
    observation = fixture.case.observations[0]
    impossible = observation.model_copy(
        update={"event_time": observation.available_at + timedelta(minutes=1)}
    )
    unvalidated_case = fixture.case.model_copy(
        update={"observations": (impossible, *fixture.case.observations[1:])}
    )

    with pytest.raises(ValueError, match="event_time.*available_at"):
        prepare_holdout(unvalidated_case, fixture.outcomes)


def test_prepare_uses_absolute_availability_across_dst_fold() -> None:
    """Wall-clock ordering at a fold must not admit an actually late publication."""
    from pi_engine.evaluation.holdout import prepare_holdout

    fixture = linear_convergence()
    eastern = ZoneInfo("America/New_York")
    cutoff = datetime(2026, 11, 1, 1, 30, tzinfo=eastern, fold=0)
    event_time = datetime(2026, 11, 1, 1, 0, tzinfo=eastern, fold=0)
    actually_late = datetime(2026, 11, 1, 1, 15, tzinfo=eastern, fold=1)
    observation = fixture.case.observations[0].model_copy(
        update={"event_time": event_time, "available_at": actually_late}
    )
    state = fixture.case.state.model_copy(update={"at": cutoff})
    case = fixture.case.model_copy(
        update={
            "prediction_cutoff": cutoff,
            "observations": (observation,),
            "state": state,
        }
    )
    future = cutoff.astimezone(UTC) + timedelta(hours=2)
    outcome = fixture.outcomes[0].model_copy(
        update={"event_time": future, "available_at": future}
    )

    with pytest.raises(ValueError, match="available_at.*prediction_cutoff"):
        prepare_holdout(case, (outcome,))


def test_prepare_rejects_absolutely_later_state_across_dst_fold() -> None:
    """A later fold state cannot be reused for an earlier absolute cutoff."""
    from pi_engine.evaluation.holdout import prepare_holdout

    fixture = linear_convergence()
    eastern = ZoneInfo("America/New_York")
    cutoff = datetime(2026, 11, 1, 1, 30, tzinfo=eastern, fold=0)
    later_state_time = datetime(2026, 11, 1, 1, 15, tzinfo=eastern, fold=1)
    state = fixture.case.state.model_copy(update={"at": later_state_time})
    case = fixture.case.model_copy(
        update={"prediction_cutoff": cutoff, "observations": (), "state": state}
    )
    future = cutoff.astimezone(UTC) + timedelta(hours=2)
    outcome = fixture.outcomes[0].model_copy(
        update={"event_time": future, "available_at": future}
    )

    with pytest.raises(ValueError, match="state time.*prediction_cutoff"):
        prepare_holdout(case, (outcome,))


def test_reveal_requires_completed_prediction_and_round_trips() -> None:
    """A matching completed forecast is the only route to raw outcomes."""
    from pi_engine.evaluation.holdout import (
        RevealedEvaluation,
        prepare_holdout,
        reveal_holdout,
    )
    from pi_engine.simulation.runner import simulate_deterministic

    fixture = linear_convergence()
    prepared = prepare_holdout(fixture.case, fixture.outcomes)
    prediction = simulate_deterministic(fixture.case, fixture.model, horizon=3)

    with pytest.raises(TypeError, match="prediction"):
        reveal_holdout(prepared)  # type: ignore[call-arg]

    revealed = reveal_holdout(prepared, prediction)

    assert revealed.case_id == fixture.case.case_id
    assert revealed.outcomes == fixture.outcomes
    assert revealed.outcome_commitments == prepared.outcome_commitments
    assert len(revealed.prediction_references) == 1
    assert revealed.prediction_references[0].artifact_type == "trajectory"
    assert revealed.prediction_references[0].artifact_id == prediction.trajectory_id
    assert RevealedEvaluation.model_validate_json(
        revealed.model_dump_json()
    ) == revealed
    with pytest.raises(ValidationError, match="frozen"):
        revealed.case_id = "changed"  # type: ignore[misc]


def test_reveal_rejects_wrong_case_short_and_incomplete_trajectory() -> None:
    """Case identity or nominal horizon alone cannot prove a completed forecast."""
    from pi_engine.evaluation.holdout import prepare_holdout, reveal_holdout
    from pi_engine.simulation.runner import simulate_deterministic

    fixture = linear_convergence()
    prepared = prepare_holdout(fixture.case, fixture.outcomes)
    short = simulate_deterministic(fixture.case, fixture.model, horizon=2)
    complete = simulate_deterministic(fixture.case, fixture.model, horizon=3)
    wrong_case = complete.model_copy(update={"case_id": "different-case"})
    incomplete = complete.model_copy(update={"points": complete.points[:-1]})

    with pytest.raises(ValueError, match="case_id"):
        reveal_holdout(prepared, wrong_case)
    with pytest.raises(ValueError, match="horizon.*cover"):
        reveal_holdout(prepared, short)
    with pytest.raises(ValueError, match="completed"):
        reveal_holdout(prepared, incomplete)


def test_reveal_horizon_must_cover_full_comparison_window() -> None:
    """Covering only the event instant cannot complete a windowed comparison."""
    from pi_engine.evaluation.holdout import prepare_holdout, reveal_holdout
    from pi_engine.schemas.outcome import ComparisonWindow
    from pi_engine.simulation.runner import simulate_deterministic

    fixture = linear_convergence()
    cutoff = fixture.case.prediction_cutoff
    windowed = fixture.outcomes[0].model_copy(
        update={
            "comparison_window": ComparisonWindow(
                start_at=cutoff + timedelta(minutes=30),
                end_at=cutoff + timedelta(hours=3),
            )
        }
    )
    prepared = prepare_holdout(fixture.case, (windowed,))
    prediction = simulate_deterministic(fixture.case, fixture.model, horizon=2)

    with pytest.raises(ValueError, match="horizon.*cover"):
        reveal_holdout(prepared, prediction)


def test_reveal_revalidates_and_rejects_outcome_bearing_prediction() -> None:
    """Bypassed schema validation cannot smuggle truth through the prediction gate."""
    from pi_engine.evaluation.holdout import prepare_holdout, reveal_holdout
    from pi_engine.simulation.runner import simulate_deterministic

    fixture = linear_convergence()
    prepared = prepare_holdout(fixture.case, fixture.outcomes)
    complete = simulate_deterministic(fixture.case, fixture.model, horizon=3)
    empty = complete.model_copy(update={"points": ()})
    outcome_bearing = complete.model_copy(
        update={"outcomes": fixture.outcomes}
    )

    with pytest.raises(ValidationError, match="points"):
        reveal_holdout(prepared, empty)
    with pytest.raises(ValueError, match="must not carry outcomes"):
        reveal_holdout(prepared, outcome_bearing)


def test_reveal_rejects_tampered_public_commitment() -> None:
    """Changing public audit metadata must not unlock a differently sealed record."""
    from pi_engine.evaluation.holdout import prepare_holdout, reveal_holdout
    from pi_engine.simulation.runner import simulate_deterministic

    fixture = linear_convergence()
    prepared = prepare_holdout(fixture.case, fixture.outcomes)
    prediction = simulate_deterministic(fixture.case, fixture.model, horizon=3)
    changed = prepared.outcome_commitments[0].model_copy(
        update={"sha256": "0" * 64}
    )
    tampered = prepared.model_copy(
        update={
            "outcome_commitments": (changed, *prepared.outcome_commitments[1:])
        }
    )

    with pytest.raises(ValueError, match="commitments.*sealed outcomes"):
        reveal_holdout(tampered, prediction)


def test_reveal_accepts_matching_completed_trajectory_ensemble() -> None:
    """A completed raw ensemble is a valid prediction artifact, not a score."""
    from pi_engine.evaluation.holdout import prepare_holdout, reveal_holdout
    from pi_engine.simulation.stochastic import simulate_stochastic
    from pi_engine.synthetic.systems import stochastic_branching

    fixture = stochastic_branching(seed=7)
    prepared = prepare_holdout(fixture.case, fixture.outcomes)
    ensemble = simulate_stochastic(
        fixture.case, fixture.model, horizon=4, samples=2, seed=7
    )

    revealed = reveal_holdout(prepared, ensemble)

    assert revealed.outcomes == fixture.outcomes
    assert revealed.prediction_references[0].artifact_type == "trajectory_ensemble"
    assert revealed.prediction_references[0].artifact_id == ensemble.ensemble_id
    assert revealed.prediction_references[0].trajectory_ids == tuple(
        item.trajectory_id for item in ensemble.trajectories
    )


def _rolling_cases() -> tuple[object, object, object]:
    fixture = linear_convergence()
    later_cutoff = fixture.case.prediction_cutoff + timedelta(hours=1)
    source = fixture.case.observations[0]
    late_revision = source.model_copy(
        update={
            "observation_id": "revision-published-after-first-cutoff",
            "event_time": fixture.case.prediction_cutoff - timedelta(hours=2),
            "available_at": fixture.case.prediction_cutoff + timedelta(minutes=30),
            "value": 3.5,
        }
    )
    later_state = fixture.case.state.model_copy(
        update={"at": later_cutoff, "observed": {"x": 1.0}}
    )
    later_case = fixture.case.model_copy(
        update={
            "prediction_cutoff": later_cutoff,
            "observations": (*fixture.case.observations, late_revision),
            "state": later_state,
        }
    )
    return fixture, later_case, late_revision


def test_prepare_rolling_holdouts_preserves_order_and_revision_isolation() -> None:
    """A later publication must not be backfilled into an earlier prediction view."""
    from pi_engine.evaluation.holdout import prepare_rolling_holdouts

    fixture, later_case, late_revision = _rolling_cases()

    rolling = prepare_rolling_holdouts(
        (fixture.case, later_case),
        (fixture.outcomes, fixture.outcomes[1:]),
    )

    assert isinstance(rolling, tuple)
    assert [item.case.prediction_cutoff for item in rolling] == [
        fixture.case.prediction_cutoff,
        later_case.prediction_cutoff,
    ]
    assert late_revision.observation_id not in {
        item.observation_id for item in rolling[0].case.observations
    }
    assert late_revision.observation_id in {
        item.observation_id for item in rolling[1].case.observations
    }
    assert rolling[0].case.state == fixture.case.state
    assert rolling[1].case.state == later_case.state


def test_prepare_rolling_holdouts_rejects_unsorted_or_reused_later_state() -> None:
    """Sorting or state inference would hide invalid caller-supplied snapshots."""
    from pi_engine.evaluation.holdout import prepare_rolling_holdouts

    fixture, later_case, _ = _rolling_cases()

    with pytest.raises(ValueError, match="strictly ordered"):
        prepare_rolling_holdouts(
            (later_case, fixture.case),
            (fixture.outcomes[1:], fixture.outcomes),
        )

    reused_later_state = fixture.case.model_copy(
        update={"state": later_case.state}
    )
    with pytest.raises(ValidationError, match="state time"):
        prepare_rolling_holdouts(
            (reused_later_state, later_case),
            (fixture.outcomes, fixture.outcomes[1:]),
        )


def test_simulation_boundary_accepts_prepared_case_not_holdout_holder() -> None:
    """Passing the sealed holder itself would expose a mixed prediction boundary."""
    from pi_engine.evaluation.holdout import prepare_holdout
    from pi_engine.simulation.runner import simulate_deterministic

    fixture = linear_convergence()
    prepared = prepare_holdout(fixture.case, fixture.outcomes)

    with pytest.raises(TypeError, match="case must be a Case"):
        simulate_deterministic(  # type: ignore[arg-type]
            prepared, fixture.model, horizon=3
        )
    prediction = simulate_deterministic(
        prepared.case, fixture.model, horizon=3
    )
    assert prediction.case_id == fixture.case.case_id
