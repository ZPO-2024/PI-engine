"""Trusted deterministic transition executors for inspectable model contracts."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import TypeAlias

from pi_engine.schemas.common import JsonScalar, NumericValue
from pi_engine.schemas.model import DynamicsSpec


StateValues: TypeAlias = Mapping[str, NumericValue]
Transition: TypeAlias = Callable[
    [StateValues, Mapping[str, JsonScalar]], dict[str, NumericValue]
]


class DeterministicSimulationError(ValueError):
    """A deterministic simulation contract failed closed."""


class ExecutorResolutionError(DeterministicSimulationError):
    """A model did not resolve to an exact trusted executor identity."""


@dataclass(frozen=True)
class _ExecutorContract:
    executor_ref: str
    executor_version: str
    code_sha256: str
    transition: Transition


def _require_exact_metadata(
    metadata: Mapping[str, JsonScalar],
    expected: frozenset[str],
    executor_ref: str,
) -> None:
    actual = set(metadata)
    if actual != expected:
        missing = ", ".join(sorted(expected - actual)) or "none"
        undeclared = ", ".join(sorted(actual - expected)) or "none"
        raise DeterministicSimulationError(
            f"invalid transition metadata for {executor_ref!r}; "
            f"missing: {missing}; undeclared: {undeclared}"
        )


def _require_variable(
    metadata: Mapping[str, JsonScalar], key: str
) -> str:
    value = metadata[key]
    if not isinstance(value, str) or not value:
        raise DeterministicSimulationError(
            f"transition metadata {key!r} must be a nonempty variable name"
        )
    return value


def _require_number(
    metadata: Mapping[str, JsonScalar], key: str
) -> float:
    value = metadata[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeterministicSimulationError(
            f"transition metadata {key!r} must be numeric"
        )
    return float(value)


def _require_scalar(state: StateValues, variable: str) -> float:
    if variable not in state:
        raise DeterministicSimulationError(
            f"required current-state variable is unavailable: {variable}"
        )
    value = state[variable]
    if isinstance(value, tuple):
        raise DeterministicSimulationError(
            f"current-state shape mismatch for {variable}: expected scalar"
        )
    return float(value)


def _require_vector(state: StateValues, variable: str) -> tuple[float, ...]:
    if variable not in state:
        raise DeterministicSimulationError(
            f"required current-state variable is unavailable: {variable}"
        )
    value = state[variable]
    if not isinstance(value, tuple):
        raise DeterministicSimulationError(
            f"current-state shape mismatch for {variable}: expected vector"
        )
    return tuple(float(element) for element in value)


def _linear_affine(
    state: StateValues, metadata: Mapping[str, JsonScalar]
) -> dict[str, NumericValue]:
    _require_exact_metadata(
        metadata,
        frozenset(
            {"state_variable", "multiplier", "intercept", "step_seconds"}
        ),
        "linear_affine",
    )
    variable = _require_variable(metadata, "state_variable")
    current = _require_scalar(state, variable)
    multiplier = _require_number(metadata, "multiplier")
    intercept = _require_number(metadata, "intercept")
    return {variable: multiplier * current + intercept}


def _planar_rotation(
    state: StateValues, metadata: Mapping[str, JsonScalar]
) -> dict[str, NumericValue]:
    _require_exact_metadata(
        metadata,
        frozenset(
            {
                "position_variable",
                "velocity_variable",
                "cosine",
                "sine",
                "step_seconds",
            }
        ),
        "planar_rotation",
    )
    position_variable = _require_variable(metadata, "position_variable")
    velocity_variable = _require_variable(metadata, "velocity_variable")
    position = _require_scalar(state, position_variable)
    velocity = _require_scalar(state, velocity_variable)
    cosine = _require_number(metadata, "cosine")
    sine = _require_number(metadata, "sine")
    return {
        position_variable: cosine * position + sine * velocity,
        velocity_variable: -sine * position + cosine * velocity,
    }


def _coupled_phase(
    state: StateValues, metadata: Mapping[str, JsonScalar]
) -> dict[str, NumericValue]:
    _require_exact_metadata(
        metadata,
        frozenset(
            {
                "phase_a_variable",
                "phase_b_variable",
                "intrinsic_step_a",
                "intrinsic_step_b",
                "coupling",
                "step_seconds",
            }
        ),
        "coupled_phase",
    )
    phase_a_variable = _require_variable(metadata, "phase_a_variable")
    phase_b_variable = _require_variable(metadata, "phase_b_variable")
    phase_a = _require_scalar(state, phase_a_variable)
    phase_b = _require_scalar(state, phase_b_variable)
    intrinsic_step_a = _require_number(metadata, "intrinsic_step_a")
    intrinsic_step_b = _require_number(metadata, "intrinsic_step_b")
    coupling = _require_number(metadata, "coupling")
    return {
        phase_a_variable: phase_a
        + intrinsic_step_a
        + coupling * math.sin(phase_b - phase_a),
        phase_b_variable: phase_b
        + intrinsic_step_b
        + coupling * math.sin(phase_a - phase_b),
    }


def _linear_feedback(
    state: StateValues, metadata: Mapping[str, JsonScalar]
) -> dict[str, NumericValue]:
    _require_exact_metadata(
        metadata,
        frozenset(
            {
                "state_variable",
                "plant_multiplier",
                "feedback_gain",
                "reference",
                "step_seconds",
            }
        ),
        "linear_feedback",
    )
    variable = _require_variable(metadata, "state_variable")
    current = _require_scalar(state, variable)
    plant_multiplier = _require_number(metadata, "plant_multiplier")
    feedback_gain = _require_number(metadata, "feedback_gain")
    reference = _require_number(metadata, "reference")
    return {
        variable: plant_multiplier * current
        + feedback_gain * (current - reference)
    }


def _nested_linear(
    state: StateValues, metadata: Mapping[str, JsonScalar]
) -> dict[str, NumericValue]:
    _require_exact_metadata(
        metadata,
        frozenset(
            {
                "state_variable",
                "root_multiplier",
                "level_multiplier",
                "parent_coupling",
                "step_seconds",
            }
        ),
        "nested_linear",
    )
    variable = _require_variable(metadata, "state_variable")
    current = _require_vector(state, variable)
    root_multiplier = _require_number(metadata, "root_multiplier")
    level_multiplier = _require_number(metadata, "level_multiplier")
    parent_coupling = _require_number(metadata, "parent_coupling")
    next_values = [root_multiplier * current[0]]
    next_values.extend(
        level_multiplier * current[index]
        + parent_coupling * current[index - 1]
        for index in range(1, len(current))
    )
    return {variable: tuple(next_values)}


_EXECUTOR_CONTRACTS = (
    _ExecutorContract(
        executor_ref="linear_affine",
        executor_version="1",
        code_sha256=(
            "20d2ac1b70f95a3492439992f268e6070d85a71f6af59a5e4e05d7b46d7c6384"
        ),
        transition=_linear_affine,
    ),
    _ExecutorContract(
        executor_ref="planar_rotation",
        executor_version="1",
        code_sha256=(
            "52d60c2ea883458cf5ebd90a5e75b68a02f4e40f8abd3e1b2155807d9ec9176e"
        ),
        transition=_planar_rotation,
    ),
    _ExecutorContract(
        executor_ref="coupled_phase",
        executor_version="1",
        code_sha256=(
            "7294decfc2c72c7d020d3264262b8389a6f324bbb75f59b79afc12fe3ab8cc75"
        ),
        transition=_coupled_phase,
    ),
    _ExecutorContract(
        executor_ref="linear_feedback",
        executor_version="1",
        code_sha256=(
            "32a62827dc0f89e6eb14f4bc8f5112336239e0fec3301d3a3fd77cd9373b6409"
        ),
        transition=_linear_feedback,
    ),
    _ExecutorContract(
        executor_ref="nested_linear",
        executor_version="1",
        code_sha256=(
            "d2f227a4ac4dbf519810562b1e3c6984acb1bf877b06dd63c2397ea73e0a631e"
        ),
        transition=_nested_linear,
    ),
)
_EXECUTORS_BY_IDENTITY = MappingProxyType(
    {
        (
            contract.executor_ref,
            contract.executor_version,
            contract.code_sha256,
        ): contract
        for contract in _EXECUTOR_CONTRACTS
    }
)


def resolve_deterministic_executor(dynamics: DynamicsSpec) -> _ExecutorContract:
    """Resolve a model only through an exact trusted in-code identity."""
    if dynamics.classification != "deterministic":
        raise ExecutorResolutionError(
            "stochastic-classified model cannot use the deterministic runner"
        )

    identity = (
        dynamics.executor_ref,
        dynamics.executor_version,
        dynamics.code_sha256,
    )
    contract = _EXECUTORS_BY_IDENTITY.get(identity)
    if contract is not None:
        return contract

    matching_ref = tuple(
        registered
        for registered_identity, registered in _EXECUTORS_BY_IDENTITY.items()
        if registered_identity[0] == dynamics.executor_ref
    )
    if not matching_ref:
        raise ExecutorResolutionError(
            f"unknown deterministic executor_ref: {dynamics.executor_ref}"
        )
    matching_version = tuple(
        registered
        for registered in matching_ref
        if registered.executor_version == dynamics.executor_version
    )
    if not matching_version:
        trusted_versions = ", ".join(
            sorted({registered.executor_version for registered in matching_ref})
        )
        raise ExecutorResolutionError(
            f"executor version mismatch for {dynamics.executor_ref!r}: "
            f"declared {dynamics.executor_version!r}, "
            f"trusted {trusted_versions!r}"
        )
    raise ExecutorResolutionError(
        f"executor code identity mismatch for {dynamics.executor_ref!r}"
    )
