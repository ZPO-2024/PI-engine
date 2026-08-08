"""Cutoff-safe temporal holdouts with outcome values sealed until prediction.

The seal in this module is an in-process API boundary, not a cryptographic
secret store.  Public models contain SHA-256 commitments, while the prepared
holder privately retains immutable outcomes so a matching completed prediction
can reveal them later.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from pi_engine.schemas.case import Case
from pi_engine.schemas.common import Provenance
from pi_engine.schemas.outcome import ComparisonWindow, Outcome
from pi_engine.schemas.trajectory import (
    Trajectory,
    TrajectoryEnsemble,
    TrajectoryHorizon,
)


NonEmptyString = Annotated[str, Field(min_length=1)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _ImmutableHoldoutSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


class OutcomeCommitment(_ImmutableHoldoutSchema):
    """Public audit metadata bound to one full canonical Outcome payload."""

    outcome_id: NonEmptyString
    case_id: NonEmptyString
    variable: NonEmptyString
    unit: NonEmptyString
    event_time: datetime
    available_at: datetime
    comparison_window: ComparisonWindow | None = None
    provenance: Provenance
    sha256: Sha256Hex


class PreparedHoldout(_ImmutableHoldoutSchema):
    """A cutoff-safe prediction case plus non-revealing outcome commitments."""

    case: Case
    outcome_commitments: Annotated[
        tuple[OutcomeCommitment, ...], Field(min_length=1)
    ]
    _sealed_outcomes: tuple[Outcome, ...] = PrivateAttr(default=())

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_sealed_outcomes":
            raise TypeError("sealed outcomes are immutable")
        super().__setattr__(name, value)

    @classmethod
    def _seal(
        cls,
        case: Case,
        commitments: tuple[OutcomeCommitment, ...],
        outcomes: tuple[Outcome, ...],
    ) -> PreparedHoldout:
        prepared = cls(case=case, outcome_commitments=commitments)
        prepared.__pydantic_private__["_sealed_outcomes"] = outcomes
        return prepared


class PredictionReference(_ImmutableHoldoutSchema):
    """Stable audit identity for the prediction artifact authorizing reveal."""

    artifact_type: Literal["trajectory", "trajectory_ensemble"]
    artifact_id: NonEmptyString
    artifact_sha256: Sha256Hex
    model_id: NonEmptyString
    model_version: NonEmptyString
    case_id: NonEmptyString
    horizon: TrajectoryHorizon
    trajectory_ids: tuple[NonEmptyString, ...] = ()


class RevealedEvaluation(_ImmutableHoldoutSchema):
    """Post-prediction audit record; this is deliberately not a score."""

    case_id: NonEmptyString
    prediction_references: Annotated[
        tuple[PredictionReference, ...], Field(min_length=1)
    ]
    outcome_commitments: Annotated[
        tuple[OutcomeCommitment, ...], Field(min_length=1)
    ]
    outcomes: Annotated[tuple[Outcome, ...], Field(min_length=1)]


def _canonical_outcome_sha256(outcome: Outcome) -> str:
    payload = outcome.model_dump(mode="json", warnings=False)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _canonical_model_sha256(model: BaseModel) -> str:
    payload = model.model_dump(mode="json", warnings=False)
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _commitment(outcome: Outcome) -> OutcomeCommitment:
    return OutcomeCommitment(
        outcome_id=outcome.outcome_id,
        case_id=outcome.case_id,
        variable=outcome.variable,
        unit=outcome.unit,
        event_time=outcome.event_time,
        available_at=outcome.available_at,
        comparison_window=outcome.comparison_window,
        provenance=outcome.provenance,
        sha256=_canonical_outcome_sha256(outcome),
    )


def _utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _validate_case_temporal_view(case: Case) -> None:
    cutoff = _utc(case.prediction_cutoff)
    if _utc(case.state.at) > cutoff:
        raise ValueError("state time must not exceed prediction_cutoff")
    for observation in case.observations:
        if _utc(observation.event_time) > _utc(observation.available_at):
            raise ValueError(
                "observation event_time must not follow available_at"
            )
        if _utc(observation.available_at) > cutoff:
            raise ValueError(
                "observation available_at must not exceed prediction_cutoff"
            )


def _validate_outcomes_against_case(
    case: Case, outcomes: tuple[Outcome, ...]
) -> None:
    variable_units = {
        definition.name: definition.unit
        for definition in case.canonical_variables
    }
    cutoff = _utc(case.prediction_cutoff)
    seen_ids: set[str] = set()
    for outcome in outcomes:
        if outcome.outcome_id in seen_ids:
            raise ValueError("outcome_id values must be unique")
        seen_ids.add(outcome.outcome_id)
        if outcome.case_id != case.case_id:
            raise ValueError("outcome case_id must match the case")
        expected_unit = variable_units.get(outcome.variable)
        if expected_unit is None:
            raise ValueError("outcome variable must be canonical")
        if outcome.unit != expected_unit:
            raise ValueError("outcome unit must match its canonical variable")
        if _utc(outcome.event_time) <= cutoff:
            raise ValueError("outcome event_time must strictly follow the cutoff")
        if _utc(outcome.available_at) <= cutoff:
            raise ValueError(
                "outcome available_at must strictly follow the cutoff"
            )


def prepare_holdout(
    case: Case, outcomes: Sequence[Outcome]
) -> PreparedHoldout:
    """Revalidate and seal outcomes behind a cutoff-safe prediction view."""
    if not isinstance(case, Case):
        raise TypeError("case must be a Case prediction input")
    validated_case = Case.model_validate(case.model_dump(warnings=False))
    _validate_case_temporal_view(validated_case)
    if not isinstance(outcomes, Sequence) or isinstance(outcomes, (str, bytes)):
        raise TypeError("outcomes must be a sequence of Outcome records")
    validated_outcomes = tuple(
        Outcome.model_validate(outcome.model_dump(warnings=False))
        if isinstance(outcome, Outcome)
        else Outcome.model_validate(outcome)
        for outcome in outcomes
    )
    _validate_outcomes_against_case(validated_case, validated_outcomes)
    commitments = tuple(_commitment(outcome) for outcome in validated_outcomes)
    return PreparedHoldout._seal(
        validated_case, commitments, validated_outcomes
    )


def prepare_rolling_holdouts(
    cases: Sequence[Case],
    outcomes_by_cutoff: Sequence[Sequence[Outcome]],
) -> tuple[PreparedHoldout, ...]:
    """Prepare independently supplied cases in caller-declared cutoff order.

    This function never derives, backfills, or reuses state estimates.  Each
    supplied Case is revalidated exactly as provided for its own cutoff.
    """
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise TypeError("cases must be a sequence of independently built Cases")
    if not isinstance(outcomes_by_cutoff, Sequence) or isinstance(
        outcomes_by_cutoff, (str, bytes)
    ):
        raise TypeError("outcomes_by_cutoff must be a sequence")
    if not cases:
        raise ValueError("rolling holdouts require at least one Case")
    if len(cases) != len(outcomes_by_cutoff):
        raise ValueError("cases and outcomes_by_cutoff must have equal length")

    prepared: list[PreparedHoldout] = []
    previous_cutoff: datetime | None = None
    for case, outcomes in zip(cases, outcomes_by_cutoff, strict=True):
        holdout = prepare_holdout(case, outcomes)
        cutoff = _utc(holdout.case.prediction_cutoff)
        if previous_cutoff is not None and cutoff <= previous_cutoff:
            raise ValueError(
                "rolling holdout cutoffs must be strictly ordered"
            )
        prepared.append(holdout)
        previous_cutoff = cutoff
    return tuple(prepared)


PredictionArtifact = Trajectory | TrajectoryEnsemble


def _carries_outcomes(prediction: PredictionArtifact) -> bool:
    if hasattr(prediction, "outcomes"):
        return True
    if isinstance(prediction, TrajectoryEnsemble):
        return any(hasattr(item, "outcomes") for item in prediction.trajectories)
    return False


def _revalidate_prediction(prediction: object) -> PredictionArtifact:
    if not isinstance(prediction, (Trajectory, TrajectoryEnsemble)):
        raise TypeError("prediction must be a Trajectory or TrajectoryEnsemble")
    if _carries_outcomes(prediction):
        raise ValueError("prediction artifacts must not carry outcomes")
    if isinstance(prediction, TrajectoryEnsemble):
        return TrajectoryEnsemble.model_validate(
            prediction.model_dump(warnings=False)
        )
    return Trajectory.model_validate(prediction.model_dump(warnings=False))


def _artifact_trajectories(
    prediction: PredictionArtifact,
) -> tuple[Trajectory, ...]:
    if isinstance(prediction, TrajectoryEnsemble):
        return prediction.trajectories
    return (prediction,)


def _validate_completed_matching_prediction(
    prepared: PreparedHoldout,
    prediction: PredictionArtifact,
    outcomes: tuple[Outcome, ...],
) -> None:
    trajectories = _artifact_trajectories(prediction)
    if prediction.case_id != prepared.case.case_id:
        raise ValueError("prediction case_id must match the prepared holdout")

    cutoff = _utc(prepared.case.prediction_cutoff)
    for trajectory in trajectories:
        if trajectory.case_id != prepared.case.case_id:
            raise ValueError("prediction case_id must match the prepared holdout")
        if _utc(trajectory.horizon.start_at) != cutoff:
            raise ValueError(
                "prediction horizon must start at the prepared case cutoff"
            )
        if trajectory.initial_state != prepared.case.state:
            raise ValueError(
                "prediction initial state must match the prepared case state"
            )
        if _utc(trajectory.points[-1].at) != _utc(trajectory.horizon.end_at):
            raise ValueError("prediction trajectory must be completed")

    horizon = trajectories[0].horizon
    horizon_start = _utc(horizon.start_at)
    horizon_end = _utc(horizon.end_at)
    for outcome in outcomes:
        required_start = _utc(outcome.event_time)
        required_end = required_start
        if outcome.comparison_window is not None:
            required_start = _utc(outcome.comparison_window.start_at)
            required_end = _utc(outcome.comparison_window.end_at)
        if horizon_start > required_start or horizon_end < required_end:
            raise ValueError(
                "prediction horizon must cover every comparison/outcome time"
            )


def _prediction_reference(
    prediction: PredictionArtifact,
) -> PredictionReference:
    if isinstance(prediction, TrajectoryEnsemble):
        first = prediction.trajectories[0]
        return PredictionReference(
            artifact_type="trajectory_ensemble",
            artifact_id=prediction.ensemble_id,
            artifact_sha256=_canonical_model_sha256(prediction),
            model_id=prediction.model_id,
            model_version=prediction.model_version,
            case_id=prediction.case_id,
            horizon=first.horizon,
            trajectory_ids=tuple(
                trajectory.trajectory_id for trajectory in prediction.trajectories
            ),
        )
    return PredictionReference(
        artifact_type="trajectory",
        artifact_id=prediction.trajectory_id,
        artifact_sha256=_canonical_model_sha256(prediction),
        model_id=prediction.model_id,
        model_version=prediction.model_version,
        case_id=prediction.case_id,
        horizon=prediction.horizon,
        trajectory_ids=(prediction.trajectory_id,),
    )


def reveal_holdout(
    prepared: PreparedHoldout,
    prediction: PredictionArtifact,
) -> RevealedEvaluation:
    """Reveal sealed outcomes only for a matching completed prediction artifact."""
    if not isinstance(prepared, PreparedHoldout):
        raise TypeError("prepared must be a PreparedHoldout")
    public = PreparedHoldout.model_validate(prepared.model_dump(warnings=False))
    sealed = prepared._sealed_outcomes
    if not sealed:
        raise ValueError("prepared holdout has no in-process sealed outcomes")
    outcomes = tuple(
        Outcome.model_validate(outcome.model_dump(warnings=False))
        for outcome in sealed
    )
    expected_commitments = tuple(_commitment(outcome) for outcome in outcomes)
    if public.outcome_commitments != expected_commitments:
        raise ValueError("prepared outcome commitments do not match sealed outcomes")

    validated_prediction = _revalidate_prediction(prediction)
    _validate_completed_matching_prediction(
        public, validated_prediction, outcomes
    )
    return RevealedEvaluation(
        case_id=public.case.case_id,
        prediction_references=(_prediction_reference(validated_prediction),),
        outcome_commitments=public.outcome_commitments,
        outcomes=outcomes,
    )
