"""Temporal holdout tests exercise the public no-leakage capability boundary."""

from copy import copy, deepcopy
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import pickle
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, ValidationError

from pi_engine.schemas.case import Case
from pi_engine.schemas.observation import Observation
from pi_engine.schemas.outcome import ComparisonWindow
from pi_engine.synthetic.systems import SyntheticSystem, linear_convergence


def _case_digest(case: Case, version: str) -> str:
    canonical = json.dumps(
        case.model_dump(mode="json", warnings=False),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(version.encode("utf-8") + b"\0" + canonical).hexdigest()


def _rolling_cases() -> tuple[SyntheticSystem, Case, Observation]:
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


def test_prediction_view_rejects_duplicate_observation_ids() -> None:
    """Duplicate observation identity would make the cutoff snapshot ambiguous."""
    from pi_engine.evaluation.holdout import prepare_holdout

    fixture = linear_convergence()
    duplicate = fixture.case.observations[0].model_copy(
        update={"value": 7.0}
    )
    case = fixture.case.model_copy(
        update={"observations": (*fixture.case.observations, duplicate)}
    )

    with pytest.raises(ValueError, match="observation_id.*unique"):
        prepare_holdout(case, fixture.outcomes)


def test_capability_holder_exposes_only_value_free_public_view() -> None:
    """Ordinary reads or Pydantic forgery must not reach the sealed payload."""
    from pi_engine.evaluation.holdout import PreparedHoldout, prepare_holdout

    fixture = linear_convergence()
    sentinel = fixture.outcomes[0].model_copy(update={"value": 123_456.789})
    prepared = prepare_holdout(fixture.case, (sentinel, *fixture.outcomes[1:]))
    serialized = prepared.model_dump_json()

    assert not isinstance(prepared, BaseModel)
    assert set(prepared.model_dump(mode="json")) == {
        "cutoff",
        "case",
        "case_digest_version",
        "case_sha256",
        "outcome_commitments",
    }
    assert "123456.789" not in serialized
    assert "123456.789" not in repr(prepared)
    assert "nonce" not in serialized
    for attribute in (
        "_sealed_outcomes",
        "__pydantic_private__",
        "model_copy",
        "model_construct",
        "_PreparedHoldout__capability",
    ):
        with pytest.raises(AttributeError):
            getattr(prepared, attribute)
    assert not hasattr(PreparedHoldout, "model_construct")


def test_capability_holder_blocks_copy_pickle_and_private_mutation() -> None:
    """Copy protocols or normal slot assignment must not clone/replace authority."""
    from pi_engine.evaluation.holdout import prepare_holdout

    fixture = linear_convergence()
    prepared = prepare_holdout(fixture.case, fixture.outcomes)

    with pytest.raises(TypeError, match="cannot be copied"):
        copy(prepared)
    with pytest.raises(TypeError, match="cannot be copied"):
        deepcopy(prepared)
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(prepared)
    with pytest.raises(TypeError, match="immutable"):
        prepared._PreparedHoldout__capability = object()  # type: ignore[attr-defined]


def test_commitments_are_salted_and_case_digest_binds_full_case() -> None:
    """Low-entropy truth and same-ID Case edits must not preserve public digests."""
    from pi_engine.evaluation.holdout import prepare_holdout

    fixture = linear_convergence()
    first = prepare_holdout(fixture.case, fixture.outcomes)
    second = prepare_holdout(fixture.case, fixture.outcomes)
    altered_case = fixture.case.model_copy(update={"title": "Altered same-ID case"})
    altered = prepare_holdout(altered_case, fixture.outcomes)

    assert first.outcome_commitments[0].sha256 != (
        second.outcome_commitments[0].sha256
    )
    assert first.case_sha256 == _case_digest(
        fixture.case, first.case_digest_version
    )
    assert altered.case_sha256 != first.case_sha256
    assert first.outcome_commitments[0].commitment_version.startswith(
        "pi-engine.outcome-commitment."
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
    """Duplicate identifiers would make commitment verification ambiguous."""
    from pi_engine.evaluation.holdout import prepare_holdout

    fixture = linear_convergence()
    duplicate = fixture.outcomes[1].model_copy(
        update={"outcome_id": fixture.outcomes[0].outcome_id}
    )

    with pytest.raises(ValueError, match="outcome_id.*unique"):
        prepare_holdout(fixture.case, (fixture.outcomes[0], duplicate))


def test_prepare_uses_absolute_outcome_chronology_across_dst_fold() -> None:
    """Folded wall time must not reverse availability or comparison chronology."""
    from pi_engine.evaluation.holdout import prepare_holdout

    fixture = linear_convergence()
    eastern = ZoneInfo("America/New_York")
    cutoff = datetime(2026, 11, 1, 1, 0, tzinfo=eastern, fold=0)
    case = fixture.case.model_copy(
        update={
            "prediction_cutoff": cutoff,
            "observations": (),
            "state": fixture.case.state.model_copy(update={"at": cutoff}),
        }
    )
    valid_window = ComparisonWindow(
        start_at=datetime(2026, 11, 1, 1, 30, tzinfo=eastern, fold=0),
        end_at=datetime(2026, 11, 1, 1, 15, tzinfo=eastern, fold=1),
    )
    valid = fixture.outcomes[0].model_copy(
        update={
            "event_time": datetime(
                2026, 11, 1, 1, 0, tzinfo=eastern, fold=1
            ),
            "available_at": datetime(
                2026, 11, 1, 1, 15, tzinfo=eastern, fold=1
            ),
            "comparison_window": valid_window,
        }
    )
    prepared = prepare_holdout(case, (valid,))
    assert prepared.outcome_commitments[0].outcome_id == valid.outcome_id

    unavailable = valid.model_copy(
        update={
            "event_time": datetime(
                2026, 11, 1, 1, 15, tzinfo=eastern, fold=1
            ),
            "available_at": datetime(
                2026, 11, 1, 1, 30, tzinfo=eastern, fold=0
            ),
            "comparison_window": None,
        }
    )
    with pytest.raises(ValueError, match="available_at.*event_time"):
        prepare_holdout(case, (unavailable,))

    reversed_window = ComparisonWindow.model_construct(
        start_at=datetime(2026, 11, 1, 1, 15, tzinfo=eastern, fold=1),
        end_at=datetime(2026, 11, 1, 1, 30, tzinfo=eastern, fold=0),
    )
    invalid_window = valid.model_copy(update={"comparison_window": reversed_window})
    with pytest.raises(ValueError, match="window end_at.*start_at"):
        prepare_holdout(case, (invalid_window,))


def test_prepare_uses_absolute_case_availability_and_state_across_fold() -> None:
    """Actually later data/state cannot enter an earlier repeated-hour cutoff."""
    from pi_engine.evaluation.holdout import prepare_holdout

    fixture = linear_convergence()
    eastern = ZoneInfo("America/New_York")
    cutoff = datetime(2026, 11, 1, 1, 30, tzinfo=eastern, fold=0)
    later = datetime(2026, 11, 1, 1, 15, tzinfo=eastern, fold=1)
    future = cutoff.astimezone(UTC) + timedelta(hours=2)
    outcome = fixture.outcomes[0].model_copy(
        update={"event_time": future, "available_at": future}
    )
    observation = fixture.case.observations[0].model_copy(
        update={
            "event_time": datetime(
                2026, 11, 1, 1, 0, tzinfo=eastern, fold=0
            ),
            "available_at": later,
        }
    )
    late_observation_case = fixture.case.model_copy(
        update={
            "prediction_cutoff": cutoff,
            "observations": (observation,),
            "state": fixture.case.state.model_copy(update={"at": cutoff}),
        }
    )
    with pytest.raises(ValueError, match="available_at.*prediction_cutoff"):
        prepare_holdout(late_observation_case, (outcome,))

    late_state_case = late_observation_case.model_copy(
        update={
            "observations": (),
            "state": fixture.case.state.model_copy(update={"at": later}),
        }
    )
    with pytest.raises(ValueError, match="state time.*prediction_cutoff"):
        prepare_holdout(late_state_case, (outcome,))


def test_prepare_accepts_inverse_fold_case_that_is_absolutely_cutoff_safe() -> None:
    """Case revalidation must retain valid earlier-fold evidence and state."""
    from pi_engine.evaluation.holdout import prepare_holdout

    fixture = linear_convergence()
    eastern = ZoneInfo("America/New_York")
    cutoff = datetime(2026, 11, 1, 1, 15, tzinfo=eastern, fold=1)
    earlier = datetime(2026, 11, 1, 1, 30, tzinfo=eastern, fold=0)
    observation = fixture.case.observations[0].model_copy(
        update={
            "event_time": datetime(
                2026, 11, 1, 1, 0, tzinfo=eastern, fold=0
            ),
            "available_at": earlier,
        }
    )
    case = fixture.case.model_copy(
        update={
            "prediction_cutoff": cutoff,
            "observations": (observation,),
            "state": fixture.case.state.model_copy(update={"at": earlier}),
        }
    )
    future = cutoff.astimezone(UTC) + timedelta(hours=1)
    outcome = fixture.outcomes[0].model_copy(
        update={"event_time": future, "available_at": future}
    )

    prepared = prepare_holdout(case, (outcome,))

    assert prepared.case.prediction_cutoff == cutoff


def test_reveal_returns_coherent_verifiable_round_trip() -> None:
    """Reveal must disclose enough nonce/case identity to verify every commitment."""
    from pi_engine.evaluation.holdout import (
        RevealedEvaluation,
        prepare_holdout,
        reveal_holdout,
        verify_outcome_commitment,
    )
    from pi_engine.simulation.runner import simulate_deterministic

    fixture = linear_convergence()
    prepared = prepare_holdout(fixture.case, fixture.outcomes)
    prediction = simulate_deterministic(fixture.case, fixture.model, horizon=3)
    revealed = reveal_holdout(prepared, prediction)

    assert revealed.case == fixture.case
    assert revealed.case_sha256 == prepared.case_sha256
    assert revealed.prediction_references[0].case_sha256 == prepared.case_sha256
    assert revealed.outcomes == fixture.outcomes
    assert len(revealed.outcome_nonces) == len(fixture.outcomes)
    assert all(
        verify_outcome_commitment(outcome, nonce, commitment)
        for outcome, nonce, commitment in zip(
            revealed.outcomes,
            revealed.outcome_nonces,
            revealed.outcome_commitments,
            strict=True,
        )
    )
    assert RevealedEvaluation.model_validate_json(
        revealed.model_dump_json()
    ) == revealed


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


def test_reveal_requires_exact_outcome_time_points_without_interpolation() -> None:
    """A terminal-only path cannot stand in for unrecorded earlier predictions."""
    from pi_engine.evaluation.holdout import prepare_holdout, reveal_holdout
    from pi_engine.simulation.runner import simulate_deterministic

    fixture = linear_convergence()
    prepared = prepare_holdout(fixture.case, fixture.outcomes)
    complete = simulate_deterministic(fixture.case, fixture.model, horizon=3)
    terminal_only = complete.model_copy(update={"points": (complete.points[-1],)})

    with pytest.raises(ValueError, match="point exactly at outcome event_time"):
        reveal_holdout(prepared, terminal_only)


def test_reveal_requires_outcome_variable_at_exact_prediction_point() -> None:
    """A timestamp match without the evaluated variable is not prediction coverage."""
    from pi_engine.evaluation.holdout import prepare_holdout, reveal_holdout
    from pi_engine.simulation.runner import simulate_deterministic

    fixture = linear_convergence()
    prepared = prepare_holdout(fixture.case, fixture.outcomes)
    complete = simulate_deterministic(fixture.case, fixture.model, horizon=3)
    wrong_values = complete.points[0].model_copy(update={"values": {"other": 1.0}})
    missing_variable = complete.model_copy(
        update={"points": (wrong_values, *complete.points[1:])}
    )

    with pytest.raises(ValueError, match="outcome variable.*exact prediction point"):
        reveal_holdout(prepared, missing_variable)


def test_every_ensemble_member_must_cover_exact_outcome_points() -> None:
    """One sparse ensemble member cannot borrow coverage from another member."""
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

    sparse = ensemble.trajectories[0].model_copy(
        update={"points": (ensemble.trajectories[0].points[-1],)}
    )
    incomplete_ensemble = ensemble.model_copy(
        update={
            "trajectories": (sparse, *ensemble.trajectories[1:]),
            "summary": None,
        }
    )
    with pytest.raises(ValueError, match="point exactly at outcome event_time"):
        reveal_holdout(prepared, incomplete_ensemble)


def test_reveal_revalidates_and_rejects_outcome_bearing_prediction() -> None:
    """Bypassed schema validation cannot smuggle truth through prediction input."""
    from pi_engine.evaluation.holdout import prepare_holdout, reveal_holdout
    from pi_engine.simulation.runner import simulate_deterministic

    fixture = linear_convergence()
    prepared = prepare_holdout(fixture.case, fixture.outcomes)
    complete = simulate_deterministic(fixture.case, fixture.model, horizon=3)
    empty = complete.model_copy(update={"points": ()})
    outcome_bearing = complete.model_copy(update={"outcomes": fixture.outcomes})

    with pytest.raises(ValidationError, match="points"):
        reveal_holdout(prepared, empty)
    with pytest.raises(ValueError, match="must not carry outcomes"):
        reveal_holdout(prepared, outcome_bearing)


def test_revealed_json_rejects_case_outcome_nonce_and_reference_tampering() -> None:
    """Revealed audit JSON must be self-coherent, not merely immutable in memory."""
    from pi_engine.evaluation.holdout import (
        RevealedEvaluation,
        prepare_holdout,
        reveal_holdout,
    )
    from pi_engine.simulation.runner import simulate_deterministic

    fixture = linear_convergence()
    prepared = prepare_holdout(fixture.case, fixture.outcomes)
    prediction = simulate_deterministic(fixture.case, fixture.model, horizon=3)
    payload = reveal_holdout(prepared, prediction).model_dump(mode="json")

    changed_case = deepcopy(payload)
    changed_case["case"]["title"] = "forged same-ID case"
    with pytest.raises(ValidationError, match="Case digest"):
        RevealedEvaluation.model_validate_json(json.dumps(changed_case))

    changed_unit = deepcopy(payload)
    changed_unit["case"]["canonical_variables"][0]["unit"] = "forged-unit"
    for observation in changed_unit["case"]["observations"]:
        observation["unit"] = "forged-unit"
    changed_unit["case_sha256"] = _case_digest(
        Case.model_validate(changed_unit["case"]),
        changed_unit["case_digest_version"],
    )
    changed_unit["prediction_references"][0]["case_sha256"] = changed_unit[
        "case_sha256"
    ]
    with pytest.raises(ValidationError, match="outcome unit"):
        RevealedEvaluation.model_validate_json(json.dumps(changed_unit))

    changed_outcome = deepcopy(payload)
    changed_outcome["outcomes"][0]["value"] = 999.0
    with pytest.raises(ValidationError, match="commitment"):
        RevealedEvaluation.model_validate_json(json.dumps(changed_outcome))

    changed_nonce = deepcopy(payload)
    changed_nonce["outcome_nonces"][0]["nonce_hex"] = "0" * 64
    with pytest.raises(ValidationError, match="commitment"):
        RevealedEvaluation.model_validate_json(json.dumps(changed_nonce))

    changed_reference = deepcopy(payload)
    changed_reference["prediction_references"][0]["case_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="prediction reference"):
        RevealedEvaluation.model_validate_json(json.dumps(changed_reference))

    empty_reference = deepcopy(payload)
    empty_reference["prediction_references"][0]["trajectory_ids"] = []
    with pytest.raises(ValidationError, match="prediction reference"):
        RevealedEvaluation.model_validate_json(json.dumps(empty_reference))

    short_reference = deepcopy(payload)
    short_reference["prediction_references"][0]["horizon"]["end_at"] = (
        fixture.outcomes[0].event_time.isoformat()
    )
    with pytest.raises(ValidationError, match="prediction reference.*cover"):
        RevealedEvaluation.model_validate_json(json.dumps(short_reference))


def test_rolling_specs_preserve_cutoff_and_revision_isolation() -> None:
    """Outcomes and late revisions stay attached to an explicit cutoff item."""
    from pi_engine.evaluation.holdout import (
        RollingHoldoutSpec,
        prepare_rolling_holdouts,
    )

    fixture, later_case, late_revision = _rolling_cases()
    items = (
        RollingHoldoutSpec(
            cutoff=fixture.case.prediction_cutoff,
            case=fixture.case,
            outcomes=fixture.outcomes,
        ),
        RollingHoldoutSpec(
            cutoff=later_case.prediction_cutoff,
            case=later_case,
            outcomes=fixture.outcomes[1:],
        ),
    )

    rolling = prepare_rolling_holdouts(items)

    assert [item.cutoff for item in rolling] == [spec.cutoff for spec in items]
    assert late_revision.observation_id not in {
        item.observation_id for item in rolling[0].case.observations
    }
    assert late_revision.observation_id in {
        item.observation_id for item in rolling[1].case.observations
    }
    with pytest.raises(ValidationError, match="frozen"):
        items[0].cutoff = later_case.prediction_cutoff  # type: ignore[misc]


def test_rolling_rejects_order_cutoff_state_and_outcome_swaps() -> None:
    """Rolling preparation cannot sort, infer state, or realign positional truth."""
    from pi_engine.evaluation.holdout import (
        RollingHoldoutSpec,
        prepare_rolling_holdouts,
    )

    fixture, later_case, _ = _rolling_cases()
    early = RollingHoldoutSpec(
        cutoff=fixture.case.prediction_cutoff,
        case=fixture.case,
        outcomes=fixture.outcomes,
    )
    later = RollingHoldoutSpec(
        cutoff=later_case.prediction_cutoff,
        case=later_case,
        outcomes=fixture.outcomes[1:],
    )
    with pytest.raises(ValueError, match="strictly ordered"):
        prepare_rolling_holdouts((later, early))

    swapped_cutoff = RollingHoldoutSpec(
        cutoff=fixture.case.prediction_cutoff,
        case=later_case,
        outcomes=fixture.outcomes[1:],
    )
    with pytest.raises(ValueError, match="Case cutoff.*item cutoff"):
        prepare_rolling_holdouts((swapped_cutoff,))

    stale_state = later_case.model_copy(
        update={"state": fixture.case.state}
    )
    stale = RollingHoldoutSpec(
        cutoff=later_case.prediction_cutoff,
        case=stale_state,
        outcomes=fixture.outcomes[1:],
    )
    with pytest.raises(ValueError, match="state time.*own cutoff"):
        prepare_rolling_holdouts((stale,))

    swapped_truth = (
        RollingHoldoutSpec(
            cutoff=fixture.case.prediction_cutoff,
            case=fixture.case,
            outcomes=fixture.outcomes[1:],
        ),
        RollingHoldoutSpec(
            cutoff=later_case.prediction_cutoff,
            case=later_case,
            outcomes=fixture.outcomes,
        ),
    )
    with pytest.raises(ValueError, match="event_time.*follow.*cutoff"):
        prepare_rolling_holdouts(swapped_truth)


def test_simulation_accepts_public_case_not_capability_holder() -> None:
    """The sealed capability itself must remain outside simulation input."""
    from pi_engine.evaluation.holdout import prepare_holdout
    from pi_engine.simulation.runner import simulate_deterministic

    fixture = linear_convergence()
    prepared = prepare_holdout(fixture.case, fixture.outcomes)

    with pytest.raises(TypeError, match="case must be a Case"):
        simulate_deterministic(  # type: ignore[arg-type]
            prepared, fixture.model, horizon=3
        )
    prediction = simulate_deterministic(prepared.case, fixture.model, horizon=3)
    assert prediction.case_id == fixture.case.case_id
