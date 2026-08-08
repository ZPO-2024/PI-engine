"""Public seeded stochastic simulation boundary."""

from collections.abc import Mapping
from datetime import UTC, timedelta
from hashlib import sha256
import json
import math

import numpy as np

from pi_engine.registry.applicability import rank_applicable_models
from pi_engine.schemas.case import Case
from pi_engine.schemas.common import NumericValue, Provenance
from pi_engine.schemas.model import ExplicitModel
from pi_engine.schemas.trajectory import (
    Trajectory,
    TrajectoryEnsemble,
    TrajectoryHorizon,
    TrajectoryPoint,
    summarize_trajectories,
)


_BERNOULLI_STEP_IDENTITY = (
    "bernoulli_step",
    "1",
    "b02f51c53d15904a6758faadd0bed53ba89b321d01a54211d97d025c577ef0ad",
)
_BERNOULLI_STEP_METADATA_KEYS = frozenset(
    {
        "state_variable",
        "up_probability",
        "up_step",
        "down_step",
        "step_seconds",
    }
)


class StochasticSimulationError(ValueError):
    """A stochastic simulation contract failed closed."""


class InapplicableStochasticModelError(StochasticSimulationError):
    """A structurally inapplicable model was rejected before execution."""

    def __init__(self, rejection_causes: tuple[str, ...]) -> None:
        self.rejection_causes = rejection_causes
        super().__init__(
            "model is not applicable to case: " + "; ".join(rejection_causes)
        )


def _positive_integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StochasticSimulationError(
            f"{field_name} must be an explicit positive integer"
        )
    return value


def _root_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StochasticSimulationError(
            "seed must be an explicit nonnegative integer"
        )
    return value


def _number(metadata: Mapping[str, object], key: str) -> float:
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StochasticSimulationError(
            f"transition metadata {key!r} must be numeric"
        )
    result = float(value)
    if not math.isfinite(result):
        raise StochasticSimulationError(
            f"transition metadata {key!r} must be finite"
        )
    return result


def _current_state_values(case: Case) -> dict[str, NumericValue]:
    values: dict[str, NumericValue] = {}
    for component in (case.state.observed, case.state.latent, case.state.boundary):
        overlap = set(values) & set(component)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise StochasticSimulationError(
                f"current-state variables are ambiguous across components: {names}"
            )
        values.update(component)
    return values


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def simulate_stochastic(
    case: Case,
    model: ExplicitModel,
    horizon: int,
    samples: int,
    seed: int,
) -> TrajectoryEnsemble:
    """Simulate raw Monte Carlo paths through one trusted stochastic model."""
    if not isinstance(case, Case):
        raise TypeError("case must be a Case prediction input")
    if not isinstance(model, ExplicitModel):
        raise TypeError("model must be an ExplicitModel")
    validated_case = Case.model_validate(case.model_dump(warnings=False))
    validated_model = ExplicitModel.model_validate(
        model.model_dump(warnings=False)
    )
    step_count = _positive_integer(horizon, "horizon")
    sample_count = _positive_integer(samples, "samples")
    root_seed = _root_seed(seed)

    applicability = rank_applicable_models(
        validated_case, (validated_model,)
    )[0]
    if not applicability.applicable:
        raise InapplicableStochasticModelError(applicability.rejection_causes)

    dynamics = validated_model.dynamics
    if dynamics.classification != "stochastic":
        raise StochasticSimulationError(
            "deterministic-classified model cannot use the stochastic runner"
        )
    identity = (
        dynamics.executor_ref,
        dynamics.executor_version,
        dynamics.code_sha256,
    )
    if identity != _BERNOULLI_STEP_IDENTITY:
        raise StochasticSimulationError(
            "model does not resolve to an exact trusted stochastic executor identity"
        )
    metadata = dynamics.transition_metadata
    if set(metadata) != _BERNOULLI_STEP_METADATA_KEYS:
        raise StochasticSimulationError(
            "invalid transition metadata for 'bernoulli_step'"
        )
    variable = metadata["state_variable"]
    if not isinstance(variable, str) or not variable:
        raise StochasticSimulationError(
            "transition metadata 'state_variable' must be a nonempty variable name"
        )
    if validated_model.predicted_outputs != (variable,):
        raise StochasticSimulationError(
            "bernoulli_step requires exactly its state variable as predicted output"
        )
    canonical_variables = {
        definition.name for definition in validated_case.canonical_variables
    }
    if variable not in canonical_variables:
        raise StochasticSimulationError(
            f"model predicts a variable not declared by the case: {variable}"
        )

    try:
        cutoff_utc = validated_case.prediction_cutoff.astimezone(UTC)
        state_at_utc = validated_case.state.at.astimezone(UTC)
    except OverflowError as exc:
        raise StochasticSimulationError(
            "UTC normalization exceeds the supported datetime range"
        ) from exc
    if state_at_utc != cutoff_utc:
        raise StochasticSimulationError(
            "case current state must be estimated at the prediction cutoff"
        )
    step_seconds = metadata["step_seconds"]
    if (
        isinstance(step_seconds, bool)
        or not isinstance(step_seconds, int)
        or step_seconds <= 0
    ):
        raise StochasticSimulationError(
            "transition metadata 'step_seconds' must be a positive integer"
        )
    up_probability = _number(metadata, "up_probability")
    if not 0.0 <= up_probability <= 1.0:
        raise StochasticSimulationError(
            "transition metadata 'up_probability' must be within [0, 1]"
        )
    up_step = _number(metadata, "up_step")
    down_step = _number(metadata, "down_step")
    try:
        step_delta = timedelta(seconds=step_seconds)
        end_at = cutoff_utc + step_delta * step_count
    except OverflowError as exc:
        raise StochasticSimulationError(
            "requested horizon exceeds the supported datetime range"
        ) from exc

    identity_payload = {
        "case": validated_case.model_dump(mode="json", warnings=False),
        "model": validated_model.model_dump(mode="json", warnings=False),
        "horizon_steps": step_count,
        "samples": sample_count,
        "seed": root_seed,
    }
    ensemble_id = f"ensemble-{_canonical_digest(identity_payload)}"
    child_sequences = np.random.SeedSequence(root_seed).spawn(sample_count)
    trajectories: list[Trajectory] = []
    for sample_index, child_sequence in enumerate(child_sequences):
        sample_seed = int(
            child_sequence.generate_state(1, dtype=np.uint64)[0]
        )
        rng = np.random.default_rng(sample_seed)
        current_state = _current_state_values(validated_case)
        current_value = current_state.get(variable)
        if isinstance(current_value, tuple) or not isinstance(
            current_value, (int, float)
        ):
            raise StochasticSimulationError(
                f"current-state shape mismatch for {variable}: expected scalar"
            )
        current = float(current_value)
        points: list[TrajectoryPoint] = []
        for step in range(1, step_count + 1):
            current += up_step if rng.random() < up_probability else down_step
            if not math.isfinite(current):
                raise StochasticSimulationError(
                    f"executor produced nonfinite state for {variable}"
                )
            try:
                point_at = cutoff_utc + step_delta * step
            except OverflowError as exc:
                raise StochasticSimulationError(
                    "requested point exceeds the supported datetime range"
                ) from exc
            points.append(
                TrajectoryPoint(
                    at=point_at,
                    values={variable: current},
                )
            )

        trajectory_id = "trajectory-" + _canonical_digest(
            {
                **identity_payload,
                "sample_index": sample_index,
                "sample_seed": sample_seed,
            }
        )
        trajectories.append(
            Trajectory(
                trajectory_id=trajectory_id,
                model_id=validated_model.model_id,
                model_version=validated_model.version,
                case_id=validated_case.case_id,
                sample_seed=sample_seed,
                initial_state=validated_case.state,
                horizon=TrajectoryHorizon(
                    start_at=cutoff_utc,
                    end_at=end_at,
                ),
                points=tuple(points),
                scenario_weight=None,
                constraints_encountered=validated_case.constraints,
                provenance=Provenance(
                    source="PI-engine stochastic runner",
                    observed_at=cutoff_utc,
                    reference=(
                        f"run:{trajectory_id};sample_seed:{sample_seed};"
                        f"spawn_index:{sample_index}"
                    ),
                ),
            )
        )

    return TrajectoryEnsemble(
        ensemble_id=ensemble_id,
        model_id=validated_model.model_id,
        model_version=validated_model.version,
        case_id=validated_case.case_id,
        trajectories=tuple(trajectories),
        seed=root_seed,
        summary=summarize_trajectories(tuple(trajectories)),
        provenance=Provenance(
            source="PI-engine stochastic runner",
            observed_at=cutoff_utc,
            reference=f"run:{ensemble_id}",
        ),
    )


__all__ = [
    "InapplicableStochasticModelError",
    "StochasticSimulationError",
    "simulate_stochastic",
]
