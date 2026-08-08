"""Cutoff-safe temporal holdouts with capability-gated outcome reveal.

``PreparedHoldout`` is deliberately not a Pydantic model.  It is a final,
non-copyable in-process capability whose ordinary readable surface delegates to
an immutable, value-free ``PreparedHoldoutView``.  Its seal retains canonical
JSON bytes and random nonces in a closure reachable only by module reveal logic.

This is an API boundary, not a cryptographic secret store: hostile process
introspection can bypass Python access conventions.  Ordinary attribute reads,
Pydantic construction/copy paths, copying, pickling, repr, and public model
serialization do not expose raw outcomes or commitment nonces.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from hmac import compare_digest
import json
from secrets import token_bytes
from typing import Annotated, Any, Callable, Literal, final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from pi_engine.schemas.case import Case
from pi_engine.schemas.common import Provenance
from pi_engine.schemas.outcome import ComparisonWindow, Outcome
from pi_engine.schemas.trajectory import (
    Trajectory,
    TrajectoryEnsemble,
    TrajectoryHorizon,
)


CASE_DIGEST_VERSION = "pi-engine.case.canonical-json.sha256.v1"
OUTCOME_COMMITMENT_VERSION = (
    "pi-engine.outcome-commitment.canonical-json.sha256.v1"
)
NONCE_BYTES = 32

NonEmptyString = Annotated[str, Field(min_length=1)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonceHex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class _ImmutableHoldoutSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )


def _revalidate_case(value: object) -> object:
    if isinstance(value, Case):
        return Case.model_validate(value.model_dump(warnings=False))
    return value


def _revalidate_outcome(value: object) -> object:
    if isinstance(value, Outcome):
        return Outcome.model_validate(value.model_dump(warnings=False))
    return value


class OutcomeCommitment(_ImmutableHoldoutSchema):
    """Value-free audit metadata plus one salted Outcome commitment."""

    commitment_version: Literal[
        "pi-engine.outcome-commitment.canonical-json.sha256.v1"
    ]
    outcome_id: NonEmptyString
    case_id: NonEmptyString
    variable: NonEmptyString
    unit: NonEmptyString
    event_time: datetime
    available_at: datetime
    comparison_window: ComparisonWindow | None = None
    provenance: Provenance
    sha256: Sha256Hex


class OutcomeNonce(_ImmutableHoldoutSchema):
    """Post-reveal nonce needed to verify one salted commitment."""

    outcome_id: NonEmptyString
    nonce_hex: NonceHex


class PreparedHoldoutView(_ImmutableHoldoutSchema):
    """Immutable, serializable, value-free public view of a prepared holdout."""

    cutoff: datetime
    case: Case
    case_digest_version: Literal[
        "pi-engine.case.canonical-json.sha256.v1"
    ]
    case_sha256: Sha256Hex
    outcome_commitments: Annotated[
        tuple[OutcomeCommitment, ...], Field(min_length=1)
    ]

    @field_validator("cutoff")
    @classmethod
    def cutoff_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("cutoff must include a timezone")
        return value

    @field_validator("case", mode="before")
    @classmethod
    def revalidate_case(cls, value: object) -> object:
        return _revalidate_case(value)

    @model_validator(mode="after")
    def validate_public_coherence(self) -> PreparedHoldoutView:
        if _utc(self.cutoff) != _utc(self.case.prediction_cutoff):
            raise ValueError("public cutoff must match Case prediction_cutoff")
        if not compare_digest(
            self.case_sha256,
            _case_sha256(self.case, self.case_digest_version),
        ):
            raise ValueError("public Case digest does not match Case")
        _validate_public_commitments(self.case, self.outcome_commitments)
        return self


class PredictionReference(_ImmutableHoldoutSchema):
    """Stable audit identity for the prediction artifact authorizing reveal."""

    artifact_type: Literal["trajectory", "trajectory_ensemble"]
    artifact_id: NonEmptyString
    artifact_sha256: Sha256Hex
    model_id: NonEmptyString
    model_version: NonEmptyString
    case_id: NonEmptyString
    case_sha256: Sha256Hex
    horizon: TrajectoryHorizon
    trajectory_ids: tuple[NonEmptyString, ...] = ()

    @model_validator(mode="after")
    def validate_reference_identity(self) -> PredictionReference:
        if not self.trajectory_ids:
            raise ValueError("prediction reference requires trajectory identities")
        if len(set(self.trajectory_ids)) != len(self.trajectory_ids):
            raise ValueError(
                "prediction reference trajectory identities must be unique"
            )
        if self.artifact_type == "trajectory" and self.trajectory_ids != (
            self.artifact_id,
        ):
            raise ValueError(
                "trajectory prediction reference must name its artifact identity"
            )
        return self


class RevealedEvaluation(_ImmutableHoldoutSchema):
    """Self-validating post-prediction audit record; deliberately not a score."""

    case_id: NonEmptyString
    case: Case
    case_digest_version: Literal[
        "pi-engine.case.canonical-json.sha256.v1"
    ]
    case_sha256: Sha256Hex
    prediction_references: Annotated[
        tuple[PredictionReference, ...], Field(min_length=1)
    ]
    outcome_commitments: Annotated[
        tuple[OutcomeCommitment, ...], Field(min_length=1)
    ]
    outcome_nonces: Annotated[tuple[OutcomeNonce, ...], Field(min_length=1)]
    outcomes: Annotated[tuple[Outcome, ...], Field(min_length=1)]

    @field_validator("case", mode="before")
    @classmethod
    def revalidate_case(cls, value: object) -> object:
        return _revalidate_case(value)

    @field_validator("outcomes", mode="before")
    @classmethod
    def revalidate_outcomes(cls, value: object) -> object:
        if isinstance(value, (tuple, list)):
            return tuple(_revalidate_outcome(item) for item in value)
        return value

    @model_validator(mode="after")
    def validate_audit_coherence(self) -> RevealedEvaluation:
        _validate_case_temporal_view(self.case)
        if self.case_id != self.case.case_id:
            raise ValueError("revealed case_id must match Case")
        if not compare_digest(
            self.case_sha256,
            _case_sha256(self.case, self.case_digest_version),
        ):
            raise ValueError("revealed Case digest does not match Case")
        _validate_outcomes_against_case(self.case, self.outcomes)
        if not (
            len(self.outcomes)
            == len(self.outcome_commitments)
            == len(self.outcome_nonces)
        ):
            raise ValueError(
                "outcomes, commitments, and nonces must have equal length"
            )
        seen_nonces: set[str] = set()
        for outcome, nonce, commitment in zip(
            self.outcomes,
            self.outcome_nonces,
            self.outcome_commitments,
            strict=True,
        ):
            if nonce.nonce_hex in seen_nonces:
                raise ValueError("revealed outcome nonces must be unique")
            seen_nonces.add(nonce.nonce_hex)
            if not verify_outcome_commitment(outcome, nonce, commitment):
                raise ValueError(
                    "revealed outcome commitment verification failed"
                )
        cutoff = _utc(self.case.prediction_cutoff)
        for reference in self.prediction_references:
            if (
                reference.case_id != self.case_id
                or not compare_digest(reference.case_sha256, self.case_sha256)
                or _utc(reference.horizon.start_at) != cutoff
            ):
                raise ValueError(
                    "prediction reference does not match revealed Case identity"
                )
            horizon_start = _utc(reference.horizon.start_at)
            horizon_end = _utc(reference.horizon.end_at)
            for outcome in self.outcomes:
                required_start = _utc(outcome.event_time)
                required_end = required_start
                if outcome.comparison_window is not None:
                    required_start = _utc(outcome.comparison_window.start_at)
                    required_end = _utc(outcome.comparison_window.end_at)
                if horizon_start > required_start or horizon_end < required_end:
                    raise ValueError(
                        "prediction reference horizon must cover revealed outcomes"
                    )
        return self


class RollingHoldoutSpec(_ImmutableHoldoutSchema):
    """One explicit Case/outcome split keyed to a caller-declared cutoff."""

    cutoff: datetime
    case: Case
    outcomes: Annotated[tuple[Outcome, ...], Field(min_length=1)]

    @field_validator("cutoff")
    @classmethod
    def cutoff_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("cutoff must include a timezone")
        return value

    @field_validator("case", mode="before")
    @classmethod
    def revalidate_case(cls, value: object) -> object:
        return _revalidate_case(value)

    @field_validator("outcomes", mode="before")
    @classmethod
    def revalidate_outcomes(cls, value: object) -> object:
        if isinstance(value, (tuple, list)):
            return tuple(_revalidate_outcome(item) for item in value)
        return value


_SealedPayload = tuple[
    bytes,
    str,
    str,
    tuple[bytes, ...],
    tuple[bytes, ...],
]
_Capability = Callable[[], _SealedPayload]


@final
class PreparedHoldout:
    """Non-Pydantic, immutable, non-copyable reveal capability."""

    __slots__ = ("__view", "__capability")

    def __new__(cls, *args: object, **kwargs: object) -> PreparedHoldout:
        raise TypeError("PreparedHoldout capabilities are created by prepare_holdout")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("PreparedHoldout cannot be subclassed")

    def __getattribute__(self, name: str) -> Any:
        if name in {
            "_PreparedHoldout__view",
            "_PreparedHoldout__capability",
            "_sealed_outcomes",
            "__pydantic_private__",
            "__dict__",
        }:
            raise AttributeError(name)
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("PreparedHoldout is immutable")

    def __copy__(self) -> PreparedHoldout:
        raise TypeError("PreparedHoldout cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> PreparedHoldout:
        raise TypeError("PreparedHoldout cannot be copied")

    def __reduce__(self) -> object:
        raise TypeError("PreparedHoldout cannot be pickled")

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("PreparedHoldout cannot be pickled")

    @property
    def view(self) -> PreparedHoldoutView:
        return _holder_view(self)

    @property
    def cutoff(self) -> datetime:
        return _holder_view(self).cutoff

    @property
    def case(self) -> Case:
        return _holder_view(self).case

    @property
    def case_digest_version(self) -> str:
        return _holder_view(self).case_digest_version

    @property
    def case_sha256(self) -> str:
        return _holder_view(self).case_sha256

    @property
    def outcome_commitments(self) -> tuple[OutcomeCommitment, ...]:
        return _holder_view(self).outcome_commitments

    def model_dump(self, *args: object, **kwargs: object) -> dict[str, object]:
        return _holder_view(self).model_dump(*args, **kwargs)

    def model_dump_json(self, *args: object, **kwargs: object) -> str:
        return _holder_view(self).model_dump_json(*args, **kwargs)

    def __repr__(self) -> str:
        return f"PreparedHoldout(view={_holder_view(self)!r})"


def _new_holder(
    view: PreparedHoldoutView, capability: _Capability
) -> PreparedHoldout:
    holder = object.__new__(PreparedHoldout)
    object.__setattr__(holder, "_PreparedHoldout__view", view)
    object.__setattr__(holder, "_PreparedHoldout__capability", capability)
    return holder


def _holder_view(holder: PreparedHoldout) -> PreparedHoldoutView:
    return object.__getattribute__(holder, "_PreparedHoldout__view")


def _utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _canonical_json_bytes(value: BaseModel | dict[str, object]) -> bytes:
    payload: object
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json", warnings=False)
    else:
        payload = value
    return json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _case_sha256(case: Case, version: str = CASE_DIGEST_VERSION) -> str:
    encoded = version.encode("utf-8") + b"\0" + _canonical_json_bytes(case)
    return sha256(encoded).hexdigest()


def _outcome_commitment_sha256(
    outcome: Outcome,
    nonce: bytes,
    version: str = OUTCOME_COMMITMENT_VERSION,
) -> str:
    encoded = (
        version.encode("utf-8")
        + b"\0"
        + nonce
        + b"\0"
        + _canonical_json_bytes(outcome)
    )
    return sha256(encoded).hexdigest()


def _outcome_metadata(outcome: Outcome) -> dict[str, object]:
    payload = outcome.model_dump(mode="json", warnings=False)
    payload.pop("value")
    return payload


def _commitment_metadata(commitment: OutcomeCommitment) -> dict[str, object]:
    payload = commitment.model_dump(mode="json", warnings=False)
    payload.pop("commitment_version")
    payload.pop("sha256")
    return payload


def _metadata_matches(
    outcome: Outcome, commitment: OutcomeCommitment
) -> bool:
    expected = sha256(_canonical_json_bytes(_outcome_metadata(outcome))).digest()
    actual = sha256(
        _canonical_json_bytes(_commitment_metadata(commitment))
    ).digest()
    return compare_digest(expected, actual)


def _commitment(outcome: Outcome, nonce: bytes) -> OutcomeCommitment:
    return OutcomeCommitment(
        commitment_version=OUTCOME_COMMITMENT_VERSION,
        outcome_id=outcome.outcome_id,
        case_id=outcome.case_id,
        variable=outcome.variable,
        unit=outcome.unit,
        event_time=outcome.event_time,
        available_at=outcome.available_at,
        comparison_window=outcome.comparison_window,
        provenance=outcome.provenance,
        sha256=_outcome_commitment_sha256(outcome, nonce),
    )


def verify_outcome_commitment(
    outcome: Outcome,
    nonce: OutcomeNonce | str,
    commitment: OutcomeCommitment,
) -> bool:
    """Verify revealed nonce, metadata, and full Outcome JSON against a digest."""
    try:
        validated_outcome = Outcome.model_validate(
            outcome.model_dump(warnings=False)
            if isinstance(outcome, Outcome)
            else outcome
        )
        validated_commitment = OutcomeCommitment.model_validate(
            commitment.model_dump(warnings=False)
            if isinstance(commitment, OutcomeCommitment)
            else commitment
        )
        if isinstance(nonce, OutcomeNonce):
            validated_nonce = OutcomeNonce.model_validate(
                nonce.model_dump(warnings=False)
            )
        else:
            validated_nonce = OutcomeNonce(
                outcome_id=validated_outcome.outcome_id,
                nonce_hex=nonce,
            )
        if (
            validated_nonce.outcome_id != validated_outcome.outcome_id
            or validated_commitment.outcome_id != validated_outcome.outcome_id
            or validated_commitment.commitment_version
            != OUTCOME_COMMITMENT_VERSION
            or not _metadata_matches(validated_outcome, validated_commitment)
        ):
            return False
        expected = _outcome_commitment_sha256(
            validated_outcome,
            bytes.fromhex(validated_nonce.nonce_hex),
            validated_commitment.commitment_version,
        )
        return compare_digest(expected, validated_commitment.sha256)
    except (AttributeError, TypeError, ValueError):
        return False


def _validate_case_temporal_view(case: Case) -> None:
    cutoff = _utc(case.prediction_cutoff)
    if _utc(case.state.at) > cutoff:
        raise ValueError("state time must not exceed prediction_cutoff")
    seen_observation_ids: set[str] = set()
    for observation in case.observations:
        if observation.observation_id in seen_observation_ids:
            raise ValueError("observation_id values must be unique")
        seen_observation_ids.add(observation.observation_id)
        if _utc(observation.event_time) > _utc(observation.available_at):
            raise ValueError(
                "observation event_time must not follow available_at"
            )
        if _utc(observation.available_at) > cutoff:
            raise ValueError(
                "observation available_at must not exceed prediction_cutoff"
            )


def _validate_outcome_absolute(outcome: Outcome) -> None:
    event_time = _utc(outcome.event_time)
    available_at = _utc(outcome.available_at)
    if available_at < event_time:
        raise ValueError("outcome available_at must not precede event_time")
    if outcome.comparison_window is None:
        return
    start_at = _utc(outcome.comparison_window.start_at)
    end_at = _utc(outcome.comparison_window.end_at)
    if end_at < start_at:
        raise ValueError("comparison window end_at must not precede start_at")
    if not start_at <= event_time <= end_at:
        raise ValueError("comparison window must contain event_time")


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
        _validate_outcome_absolute(outcome)
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


def _validate_public_commitments(
    case: Case, commitments: tuple[OutcomeCommitment, ...]
) -> None:
    variable_units = {
        definition.name: definition.unit
        for definition in case.canonical_variables
    }
    cutoff = _utc(case.prediction_cutoff)
    seen_ids: set[str] = set()
    for commitment in commitments:
        if commitment.outcome_id in seen_ids:
            raise ValueError("commitment outcome_id values must be unique")
        seen_ids.add(commitment.outcome_id)
        if commitment.case_id != case.case_id:
            raise ValueError("commitment case_id must match Case")
        expected_unit = variable_units.get(commitment.variable)
        if expected_unit is None or commitment.unit != expected_unit:
            raise ValueError("commitment variable/unit must match Case")
        event_time = _utc(commitment.event_time)
        available_at = _utc(commitment.available_at)
        if event_time <= cutoff or available_at <= cutoff:
            raise ValueError("commitment times must strictly follow cutoff")
        if available_at < event_time:
            raise ValueError("commitment available_at must follow event_time")
        if commitment.comparison_window is not None:
            start = _utc(commitment.comparison_window.start_at)
            end = _utc(commitment.comparison_window.end_at)
            if end < start or not start <= event_time <= end:
                raise ValueError("commitment comparison window is incoherent")


def prepare_holdout(
    case: Case, outcomes: Sequence[Outcome]
) -> PreparedHoldout:
    """Revalidate and seal raw outcomes behind a value-free public capability."""
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
    if not validated_outcomes:
        raise ValueError("holdout requires at least one Outcome")
    _validate_outcomes_against_case(validated_case, validated_outcomes)

    nonces: list[bytes] = []
    seen_nonces: set[bytes] = set()
    for _ in validated_outcomes:
        nonce = token_bytes(NONCE_BYTES)
        while nonce in seen_nonces:
            nonce = token_bytes(NONCE_BYTES)
        seen_nonces.add(nonce)
        nonces.append(nonce)
    commitments = tuple(
        _commitment(outcome, nonce)
        for outcome, nonce in zip(validated_outcomes, nonces, strict=True)
    )
    case_json = _canonical_json_bytes(validated_case)
    case_digest = _case_sha256(validated_case)
    view = PreparedHoldoutView(
        cutoff=validated_case.prediction_cutoff,
        case=validated_case,
        case_digest_version=CASE_DIGEST_VERSION,
        case_sha256=case_digest,
        outcome_commitments=commitments,
    )
    sealed: _SealedPayload = (
        case_json,
        CASE_DIGEST_VERSION,
        case_digest,
        tuple(_canonical_json_bytes(outcome) for outcome in validated_outcomes),
        tuple(nonces),
    )

    def capability() -> _SealedPayload:
        return sealed

    return _new_holder(view, capability)


def prepare_rolling_holdouts(
    items: Sequence[RollingHoldoutSpec],
) -> tuple[PreparedHoldout, ...]:
    """Prepare explicit cutoff-keyed specs without sorting or state inference."""
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise TypeError("items must be a sequence of RollingHoldoutSpec")
    if not items:
        raise ValueError("rolling holdouts require at least one item")
    prepared: list[PreparedHoldout] = []
    previous_cutoff: datetime | None = None
    for item in items:
        if not isinstance(item, RollingHoldoutSpec):
            raise TypeError("each rolling item must be a RollingHoldoutSpec")
        validated = RollingHoldoutSpec.model_validate(
            item.model_dump(warnings=False)
        )
        cutoff = _utc(validated.cutoff)
        if previous_cutoff is not None and cutoff <= previous_cutoff:
            raise ValueError("rolling holdout cutoffs must be strictly ordered")
        if _utc(validated.case.prediction_cutoff) != cutoff:
            raise ValueError("Case cutoff must equal its item cutoff")
        if _utc(validated.case.state.at) != cutoff:
            raise ValueError("state time must equal its own cutoff")
        prepared.append(prepare_holdout(validated.case, validated.outcomes))
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
    view: PreparedHoldoutView,
    prediction: PredictionArtifact,
    outcomes: tuple[Outcome, ...],
) -> None:
    trajectories = _artifact_trajectories(prediction)
    if prediction.case_id != view.case.case_id:
        raise ValueError("prediction case_id must match the prepared holdout")
    cutoff = _utc(view.case.prediction_cutoff)
    expected_state = _canonical_json_bytes(view.case.state)
    for trajectory in trajectories:
        if trajectory.case_id != view.case.case_id:
            raise ValueError("prediction case_id must match the prepared holdout")
        if _utc(trajectory.horizon.start_at) != cutoff:
            raise ValueError(
                "prediction horizon must start at the prepared case cutoff"
            )
        if not compare_digest(
            _canonical_json_bytes(trajectory.initial_state), expected_state
        ):
            raise ValueError(
                "prediction initial state must match the prepared case state"
            )
        if _utc(trajectory.points[-1].at) != _utc(trajectory.horizon.end_at):
            raise ValueError("prediction trajectory must be completed")

        horizon_start = _utc(trajectory.horizon.start_at)
        horizon_end = _utc(trajectory.horizon.end_at)
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
            matching_points = tuple(
                point
                for point in trajectory.points
                if _utc(point.at) == _utc(outcome.event_time)
            )
            if not matching_points:
                raise ValueError(
                    "prediction must contain a point exactly at outcome event_time"
                )
            if outcome.variable not in matching_points[0].values:
                raise ValueError(
                    "outcome variable must exist at the exact prediction point"
                )


def _canonical_model_sha256(model: BaseModel) -> str:
    return sha256(_canonical_json_bytes(model)).hexdigest()


def _prediction_reference(
    prediction: PredictionArtifact,
    case_sha256: str,
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
            case_sha256=case_sha256,
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
        case_sha256=case_sha256,
        horizon=prediction.horizon,
        trajectory_ids=(prediction.trajectory_id,),
    )


def reveal_holdout(
    prepared: PreparedHoldout,
    prediction: PredictionArtifact,
) -> RevealedEvaluation:
    """Reveal sealed outcomes only for a matching completed prediction artifact."""
    if not isinstance(prepared, PreparedHoldout):
        raise TypeError("prepared must be a PreparedHoldout capability")
    view = PreparedHoldoutView.model_validate(
        _holder_view(prepared).model_dump(warnings=False)
    )
    capability = object.__getattribute__(
        prepared, "_PreparedHoldout__capability"
    )
    (
        sealed_case_json,
        sealed_case_version,
        sealed_case_digest,
        sealed_outcome_json,
        sealed_nonces,
    ) = capability()
    sealed_case = Case.model_validate_json(sealed_case_json)
    if (
        sealed_case_version != CASE_DIGEST_VERSION
        or not compare_digest(
            sealed_case_digest,
            _case_sha256(sealed_case, sealed_case_version),
        )
        or not compare_digest(sealed_case_digest, view.case_sha256)
        or not compare_digest(
            sealed_case_json, _canonical_json_bytes(view.case)
        )
    ):
        raise ValueError("public Case does not match sealed Case identity")
    _validate_case_temporal_view(sealed_case)
    outcomes = tuple(
        Outcome.model_validate_json(payload) for payload in sealed_outcome_json
    )
    _validate_outcomes_against_case(sealed_case, outcomes)
    if len(outcomes) != len(sealed_nonces):
        raise ValueError("sealed outcomes and nonces are incoherent")
    expected_commitments = tuple(
        _commitment(outcome, nonce)
        for outcome, nonce in zip(outcomes, sealed_nonces, strict=True)
    )
    if len(expected_commitments) != len(view.outcome_commitments) or any(
        not (
            _metadata_matches(outcome, public)
            and compare_digest(expected.sha256, public.sha256)
            and expected.commitment_version == public.commitment_version
        )
        for outcome, expected, public in zip(
            outcomes,
            expected_commitments,
            view.outcome_commitments,
            strict=True,
        )
    ):
        raise ValueError("public commitments do not match sealed outcomes")

    validated_prediction = _revalidate_prediction(prediction)
    _validate_completed_matching_prediction(view, validated_prediction, outcomes)
    nonces = tuple(
        OutcomeNonce(outcome_id=outcome.outcome_id, nonce_hex=nonce.hex())
        for outcome, nonce in zip(outcomes, sealed_nonces, strict=True)
    )
    return RevealedEvaluation(
        case_id=sealed_case.case_id,
        case=sealed_case,
        case_digest_version=sealed_case_version,
        case_sha256=sealed_case_digest,
        prediction_references=(
            _prediction_reference(validated_prediction, sealed_case_digest),
        ),
        outcome_commitments=view.outcome_commitments,
        outcome_nonces=nonces,
        outcomes=outcomes,
    )
