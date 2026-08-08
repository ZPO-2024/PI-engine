"""Public deterministic simulation boundary."""

from datetime import UTC, timedelta
from hashlib import sha256
import json
import math
from collections.abc import Mapping

from pi_engine.registry.applicability import rank_applicable_models
from pi_engine.schemas.case import Case
from pi_engine.schemas.common import NumericValue, Provenance
from pi_engine.schemas.model import ExplicitModel
from pi_engine.schemas.trajectory import (
    Trajectory,
    TrajectoryHorizon,
    TrajectoryPoint,
)
from pi_engine.simulation.deterministic import (
    DeterministicSimulationError,
    resolve_deterministic_executor,
    validate_deterministic_output_contract,
)


class InapplicableModelError(DeterministicSimulationError):
    """A structurally inapplicable model was rejected before simulation."""

    def __init__(self, rejection_causes: tuple[str, ...]) -> None:
        self.rejection_causes = rejection_causes
        causes = "; ".join(rejection_causes)
        super().__init__(f"model is not applicable to case: {causes}")


def _validate_horizon(horizon: int) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise DeterministicSimulationError(
            "horizon must be an explicit positive integer step count"
        )
    return horizon


def _step_seconds(metadata: Mapping[str, object]) -> int:
    value = metadata.get("step_seconds")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DeterministicSimulationError(
            "transition metadata 'step_seconds' must be a positive integer"
        )
    return value


def _current_state_values(case: Case) -> dict[str, NumericValue]:
    values: dict[str, NumericValue] = {}
    for component in (case.state.observed, case.state.latent, case.state.boundary):
        overlap = set(values) & set(component)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise DeterministicSimulationError(
                f"current-state variables are ambiguous across components: {names}"
            )
        values.update(component)
    return values


def _shape(value: NumericValue) -> tuple[str, int | None]:
    if isinstance(value, tuple):
        return ("vector", len(value))
    return ("scalar", None)


def _validated_value(variable: str, value: object) -> NumericValue:
    elements = value if isinstance(value, tuple) else (value,)
    if not elements or any(
        isinstance(element, bool)
        or not isinstance(element, (int, float))
        or not math.isfinite(element)
        for element in elements
    ):
        raise DeterministicSimulationError(
            f"executor produced nonfinite or nonnumeric state for {variable}"
        )
    if isinstance(value, tuple):
        return tuple(float(element) for element in elements)
    return float(elements[0])


def _trajectory_id(
    case: Case, model: ExplicitModel, horizon: int
) -> str:
    identity = {
        "case": case.model_dump(mode="json", warnings=False),
        "model": model.model_dump(mode="json", warnings=False),
        "executor": {
            "ref": model.dynamics.executor_ref,
            "version": model.dynamics.executor_version,
            "code_sha256": model.dynamics.code_sha256,
        },
        "horizon_steps": horizon,
    }
    canonical = json.dumps(
        identity, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return f"trajectory-{sha256(canonical).hexdigest()}"


def simulate_deterministic(
    case: Case, model: ExplicitModel, horizon: int
) -> Trajectory:
    """Simulate a cutoff-safe case with one exact trusted deterministic model."""
    if not isinstance(case, Case):
        raise TypeError("case must be a Case prediction input")
    if not isinstance(model, ExplicitModel):
        raise TypeError("model must be an ExplicitModel")
    validated_case = Case.model_validate(case.model_dump(warnings=False))
    validated_model = ExplicitModel.model_validate(
        model.model_dump(warnings=False)
    )

    step_count = _validate_horizon(horizon)
    applicability = rank_applicable_models(
        validated_case, (validated_model,)
    )[0]
    if not applicability.applicable:
        raise InapplicableModelError(applicability.rejection_causes)

    contract = resolve_deterministic_executor(validated_model.dynamics)
    cutoff_utc = validated_case.prediction_cutoff.astimezone(UTC)
    if validated_case.state.at.astimezone(UTC) != cutoff_utc:
        raise DeterministicSimulationError(
            "case current state must be estimated at the prediction cutoff"
        )

    canonical_variables = {
        definition.name for definition in validated_case.canonical_variables
    }
    if len(validated_model.predicted_outputs) != len(
        set(validated_model.predicted_outputs)
    ):
        raise DeterministicSimulationError(
            "model contains duplicate predicted outputs"
        )
    predicted_outputs = set(validated_model.predicted_outputs)
    undeclared_outputs = predicted_outputs - canonical_variables
    if undeclared_outputs:
        names = ", ".join(sorted(undeclared_outputs))
        raise DeterministicSimulationError(
            f"model predicts variables not declared by the case: {names}"
        )
    validate_deterministic_output_contract(
        contract,
        validated_model.dynamics,
        validated_model.predicted_outputs,
    )

    step_seconds = _step_seconds(validated_model.dynamics.transition_metadata)
    try:
        step_delta = timedelta(seconds=step_seconds)
        end_at = cutoff_utc + step_delta * step_count
    except OverflowError as exc:
        raise DeterministicSimulationError(
            "requested horizon exceeds the supported datetime range"
        ) from exc
    current = _current_state_values(validated_case)
    initial_shapes: dict[str, tuple[str, int | None]] = {}
    for variable in validated_model.predicted_outputs:
        if variable not in current:
            raise DeterministicSimulationError(
                f"predicted output lacks current-state value: {variable}"
            )
        initial_shapes[variable] = _shape(current[variable])

    points: list[TrajectoryPoint] = []
    for step in range(1, step_count + 1):
        updates = contract.transition(
            current, validated_model.dynamics.transition_metadata
        )
        actual_outputs = set(updates)
        if actual_outputs != predicted_outputs:
            undeclared = actual_outputs - predicted_outputs
            missing = predicted_outputs - actual_outputs
            details: list[str] = []
            if undeclared:
                details.append(
                    "undeclared outputs: " + ", ".join(sorted(undeclared))
                )
            if missing:
                details.append("missing outputs: " + ", ".join(sorted(missing)))
            raise DeterministicSimulationError(
                "executor output contract mismatch; " + "; ".join(details)
            )

        point_values: dict[str, NumericValue] = {}
        for variable in validated_model.predicted_outputs:
            value = _validated_value(variable, updates[variable])
            if _shape(value) != initial_shapes[variable]:
                raise DeterministicSimulationError(
                    f"executor produced shape drift for {variable}"
                )
            current[variable] = value
            point_values[variable] = value
        points.append(
            TrajectoryPoint(
                at=cutoff_utc + step_delta * step,
                values=point_values,
            )
        )

    trajectory_id = _trajectory_id(
        validated_case, validated_model, step_count
    )
    return Trajectory(
        trajectory_id=trajectory_id,
        model_id=validated_model.model_id,
        model_version=validated_model.version,
        case_id=validated_case.case_id,
        initial_state=validated_case.state,
        horizon=TrajectoryHorizon(
            start_at=cutoff_utc,
            end_at=end_at,
        ),
        points=tuple(points),
        scenario_weight=None,
        constraints_encountered=validated_case.constraints,
        provenance=Provenance(
            source="PI-engine deterministic runner",
            observed_at=cutoff_utc,
            reference=f"run:{trajectory_id}",
        ),
    )


__all__ = [
    "DeterministicSimulationError",
    "InapplicableModelError",
    "simulate_deterministic",
]
