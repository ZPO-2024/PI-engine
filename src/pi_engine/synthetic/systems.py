"""Auditable synthetic case/model/outcome fixtures with known ground truth."""

from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from collections.abc import Callable
from typing import Annotated, Mapping

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_serializer,
    field_validator,
    model_validator,
)

from pi_engine.schemas.case import Case, VariableDefinition
from pi_engine.schemas.common import (
    Confidence,
    JsonScalar,
    Provenance,
    UncertaintyClass,
)
from pi_engine.schemas.graph import GraphEdge, GraphNode, SystemGraph
from pi_engine.schemas.model import ExplicitModel
from pi_engine.schemas.observation import Observation
from pi_engine.schemas.outcome import Outcome
from pi_engine.schemas.state import StateEstimate


NonEmptyString = Annotated[str, Field(min_length=1)]
SYNTHETIC_CUTOFF = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
_LINEAR_AFFINE_SHA256 = (
    "20d2ac1b70f95a3492439992f268e6070d85a71f6af59a5e4e05d7b46d7c6384"
)
_PLANAR_ROTATION_SHA256 = (
    "52d60c2ea883458cf5ebd90a5e75b68a02f4e40f8abd3e1b2155807d9ec9176e"
)
_BERNOULLI_STEP_SHA256 = (
    "b02f51c53d15904a6758faadd0bed53ba89b321d01a54211d97d025c577ef0ad"
)
_COUPLED_PHASE_SHA256 = (
    "7294decfc2c72c7d020d3264262b8389a6f324bbb75f59b79afc12fe3ab8cc75"
)
_LINEAR_FEEDBACK_SHA256 = (
    "32a62827dc0f89e6eb14f4bc8f5112336239e0fec3301d3a3fd77cd9373b6409"
)
_NESTED_LINEAR_SHA256 = (
    "d2f227a4ac4dbf519810562b1e3c6984acb1bf877b06dd63c2397ea73e0a631e"
)


class SyntheticSystem(BaseModel):
    """Immutable boundary between cutoff-safe input and withheld truth."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )

    name: NonEmptyString
    regime: NonEmptyString
    case: Case
    model: ExplicitModel
    outcomes: tuple[Outcome, ...] = Field(min_length=1)
    seed: StrictInt = Field(ge=0)
    is_control: bool = False
    ground_truth: Mapping[NonEmptyString, JsonScalar]

    @field_validator("case", mode="before")
    @classmethod
    def revalidate_case(cls, value: object) -> object:
        if isinstance(value, Case):
            return Case.model_validate(value.model_dump())
        return value

    @field_validator("model", mode="before")
    @classmethod
    def revalidate_model(cls, value: object) -> object:
        if isinstance(value, ExplicitModel):
            return ExplicitModel.model_validate(value.model_dump())
        return value

    @field_validator("outcomes", mode="before")
    @classmethod
    def revalidate_outcomes(cls, value: object) -> object:
        if isinstance(value, (tuple, list)):
            return tuple(
                Outcome.model_validate(item.model_dump())
                if isinstance(item, Outcome)
                else item
                for item in value
            )
        return value

    @field_validator("ground_truth")
    @classmethod
    def freeze_ground_truth(
        cls, value: Mapping[str, JsonScalar]
    ) -> Mapping[str, JsonScalar]:
        return MappingProxyType(dict(value))

    @field_serializer("ground_truth")
    def serialize_ground_truth(
        self, value: Mapping[str, JsonScalar]
    ) -> dict[str, JsonScalar]:
        return dict(value)

    @model_validator(mode="after")
    def validate_holdout_boundary(self) -> "SyntheticSystem":
        variables = {
            definition.name: definition.unit
            for definition in self.case.canonical_variables
        }
        for observation in self.case.observations:
            if observation.event_time > observation.available_at:
                raise ValueError("observation event_time must not follow available_at")
        for outcome in self.outcomes:
            if outcome.case_id != self.case.case_id:
                raise ValueError("outcome case_id must match the case")
            if outcome.event_time <= self.case.prediction_cutoff:
                raise ValueError("withheld outcome event_time must follow the cutoff")
            if outcome.available_at <= self.case.prediction_cutoff:
                raise ValueError("withheld outcome available_at must follow the cutoff")
            expected_unit = variables.get(outcome.variable)
            if expected_unit is None:
                raise ValueError("outcome variable must be canonical")
            if outcome.unit != expected_unit:
                raise ValueError("outcome unit must match its canonical variable")
        return self


def _fixture_provenance(reference: str, at: datetime) -> Provenance:
    return Provenance(
        source="PI-engine synthetic fixture",
        observed_at=at,
        reference=reference,
    )


def linear_convergence(seed: int = 0) -> SyntheticSystem:
    """Return x[k+1] = 0.5*x[k], converging monotonically to zero."""
    case_id = "synthetic-linear-convergence"
    observation_values = (8.0, 4.0, 2.0)
    observations = tuple(
        Observation(
            observation_id=f"{case_id}-obs-{index}",
            case_id=case_id,
            variable="x",
            value=value,
            unit="a.u.",
            event_time=SYNTHETIC_CUTOFF - timedelta(hours=3 - index),
            available_at=SYNTHETIC_CUTOFF - timedelta(hours=3 - index),
            uncertainty_class=UncertaintyClass.MEASUREMENT,
            confidence=Confidence(score=1.0, basis="synthetic exact value"),
            provenance=_fixture_provenance(
                f"synthetic:{case_id}:observation:{index}",
                SYNTHETIC_CUTOFF - timedelta(hours=3 - index),
            ),
        )
        for index, value in enumerate(observation_values)
    )
    case = Case(
        case_id=case_id,
        title="Linear convergence",
        domain="synthetic_dynamics",
        provenance=_fixture_provenance(
            f"synthetic:{case_id}", SYNTHETIC_CUTOFF - timedelta(hours=3)
        ),
        prediction_cutoff=SYNTHETIC_CUTOFF,
        canonical_variables=(VariableDefinition(name="x", unit="a.u."),),
        observations=observations,
        state=StateEstimate(
            at=SYNTHETIC_CUTOFF,
            observed={"x": 2.0},
            latent={},
            uncertainty={"x": 0.0},
            boundary={},
        ),
        graph=SystemGraph(
            nodes=(GraphNode(node_id="state-x", variable_refs=("x",)),),
            edges=(),
            topology_metadata={"single_state": True},
        ),
    )
    model = ExplicitModel.model_validate(
        {
            "model_id": "synthetic-linear-affine-convergence",
            "version": "1.0.0",
            "name": "Synthetic linear affine convergence",
            "domain": "synthetic_dynamics",
            "model_family": "linear_affine",
            "provenance": _fixture_provenance(
                "synthetic-transition-contract:linear_affine:v1",
                SYNTHETIC_CUTOFF - timedelta(days=1),
            ),
            "initial_confidence": Confidence(
                score=1.0, basis="known synthetic transition"
            ),
            "applicability": {
                "required_variables": ("x",),
                "optional_variables": (),
                "valid_ranges": {},
                "topology_requirements": ("single_state",),
                "boundary_conditions": (),
                "exclusion_rules": (),
            },
            "dynamics": {
                "executor_ref": "linear_affine",
                "executor_version": "1",
                "code_sha256": _LINEAR_AFFINE_SHA256,
                "equations": (
                    "next = multiplier * current + intercept",
                ),
                "rules": (
                    "read and write the scalar named by state_variable once per step",
                ),
                "transition_metadata": {
                    "state_variable": "x",
                    "multiplier": 0.5,
                    "intercept": 0.0,
                    "step_seconds": 3600,
                },
                "time_behavior": "discrete fixed one-hour steps",
                "classification": "deterministic",
            },
            "uncertainty": {
                "parameter_uncertainty": (),
                "process_disturbance": (),
                "model_discrepancy": (),
                "structured_unknowns": (),
            },
            "assumptions": ("exact scalar arithmetic",),
            "relationships_used": (),
            "phase_dependencies": (),
            "effective_proximity_dependencies": (),
            "information_dependencies": ("current x",),
            "predicted_outputs": ("x",),
            "prediction_horizon": "PT3H",
            "expected_regimes": ("convergence",),
            "falsification": {
                "falsifiers": ("absolute x fails to halve on a step",),
                "contradictory_evidence_conditions": (),
                "failure_conditions": ("x is unavailable",),
            },
        }
    )
    outcomes = tuple(
        Outcome(
            outcome_id=f"{case_id}-outcome-{index}",
            case_id=case_id,
            variable="x",
            unit="a.u.",
            value=value,
            event_time=SYNTHETIC_CUTOFF + timedelta(hours=index),
            available_at=SYNTHETIC_CUTOFF + timedelta(hours=index),
            provenance=_fixture_provenance(
                f"synthetic:{case_id}:outcome:{index}",
                SYNTHETIC_CUTOFF + timedelta(hours=index),
            ),
        )
        for index, value in enumerate((1.0, 0.5, 0.25), start=1)
    )
    return SyntheticSystem(
        name="linear_convergence",
        regime="convergence",
        case=case,
        model=model,
        outcomes=outcomes,
        seed=seed,
        is_control=False,
        ground_truth={"horizon_steps": 3},
    )


def _zero_uncertainty(value: object) -> float | tuple[float, ...]:
    if isinstance(value, tuple):
        return tuple(0.0 for _ in value)
    return 0.0


def _build_fixture(
    *,
    name: str,
    regime: str,
    seed: int,
    variables: tuple[tuple[str, str], ...],
    observations: tuple[tuple[int, str, object], ...],
    state: Mapping[str, object],
    graph: SystemGraph,
    executor_ref: str,
    code_sha256: str,
    equations: tuple[str, ...],
    rules: tuple[str, ...],
    transition_metadata: Mapping[str, JsonScalar],
    classification: str,
    outcomes: tuple[tuple[int, str, object], ...],
    model_family: str,
    relationships_used: tuple[str, ...] = (),
    phase_dependencies: tuple[str, ...] = (),
    effective_proximity_dependencies: tuple[str, ...] = (),
    is_control: bool = False,
) -> SyntheticSystem:
    """Assemble schemas; transition equations remain exclusively in ExplicitModel."""
    case_id = f"synthetic-{name.replace('_', '-')}"
    units = dict(variables)
    case_observations = tuple(
        Observation(
            observation_id=f"{case_id}-obs-{index}",
            case_id=case_id,
            variable=variable,
            value=value,
            unit=units[variable],
            event_time=SYNTHETIC_CUTOFF - timedelta(hours=hours_before),
            available_at=SYNTHETIC_CUTOFF - timedelta(hours=hours_before),
            uncertainty_class=UncertaintyClass.MEASUREMENT,
            confidence=Confidence(score=1.0, basis="synthetic exact value"),
            provenance=_fixture_provenance(
                f"synthetic:{case_id}:observation:{index}",
                SYNTHETIC_CUTOFF - timedelta(hours=hours_before),
            ),
        )
        for index, (hours_before, variable, value) in enumerate(observations)
    )
    case = Case(
        case_id=case_id,
        title=name.replace("_", " ").title(),
        domain="synthetic_dynamics",
        provenance=_fixture_provenance(
            f"synthetic:{case_id}", SYNTHETIC_CUTOFF - timedelta(days=1)
        ),
        prediction_cutoff=SYNTHETIC_CUTOFF,
        canonical_variables=tuple(
            VariableDefinition(name=variable, unit=unit)
            for variable, unit in variables
        ),
        observations=case_observations,
        state=StateEstimate(
            at=SYNTHETIC_CUTOFF,
            observed=state,
            latent={},
            uncertainty={
                variable: _zero_uncertainty(value)
                for variable, value in state.items()
            },
            boundary={},
        ),
        graph=graph,
    )
    predicted_outputs = tuple(dict.fromkeys(variable for _, variable, _ in outcomes))
    topology_requirements = tuple(
        key
        for key, value in graph.topology_metadata.items()
        if value is True
    )
    model = ExplicitModel.model_validate(
        {
            "model_id": f"synthetic-{name.replace('_', '-')}-model",
            "version": "1.0.0",
            "name": f"Synthetic {name.replace('_', ' ')} transition",
            "domain": "synthetic_dynamics",
            "model_family": model_family,
            "provenance": _fixture_provenance(
                f"synthetic-transition-contract:{executor_ref}:v1",
                SYNTHETIC_CUTOFF - timedelta(days=1),
            ),
            "initial_confidence": Confidence(
                score=1.0, basis="known synthetic transition"
            ),
            "applicability": {
                "required_variables": tuple(variable for variable, _ in variables),
                "optional_variables": (),
                "valid_ranges": {},
                "topology_requirements": topology_requirements,
                "boundary_conditions": (),
                "exclusion_rules": (),
            },
            "dynamics": {
                "executor_ref": executor_ref,
                "executor_version": "1",
                "code_sha256": code_sha256,
                "equations": equations,
                "rules": rules,
                "transition_metadata": transition_metadata,
                "time_behavior": "discrete fixed one-hour steps",
                "classification": classification,
            },
            "uncertainty": {
                "parameter_uncertainty": (),
                "process_disturbance": (
                    ("seeded Bernoulli branch",)
                    if classification == "stochastic"
                    else ()
                ),
                "model_discrepancy": (),
                "structured_unknowns": (),
            },
            "assumptions": ("transition is evaluated simultaneously once per step",),
            "relationships_used": relationships_used,
            "phase_dependencies": phase_dependencies,
            "effective_proximity_dependencies": effective_proximity_dependencies,
            "information_dependencies": tuple(
                f"current {variable}" for variable, _ in variables
            ),
            "predicted_outputs": predicted_outputs,
            "prediction_horizon": f"PT{max(step for step, _, _ in outcomes)}H",
            "expected_regimes": (regime,),
            "falsification": {
                "falsifiers": (
                    "a transition disagrees with the declared equation or rule",
                ),
                "contradictory_evidence_conditions": (),
                "failure_conditions": ("a required state variable is unavailable",),
            },
        }
    )
    withheld = tuple(
        Outcome(
            outcome_id=f"{case_id}-outcome-{index}",
            case_id=case_id,
            variable=variable,
            unit=units[variable],
            value=value,
            event_time=SYNTHETIC_CUTOFF + timedelta(hours=step),
            available_at=SYNTHETIC_CUTOFF + timedelta(hours=step),
            provenance=_fixture_provenance(
                f"synthetic:{case_id}:outcome:{index}",
                SYNTHETIC_CUTOFF + timedelta(hours=step),
            ),
        )
        for index, (step, variable, value) in enumerate(outcomes)
    )
    return SyntheticSystem(
        name=name,
        regime=regime,
        case=case,
        model=model,
        outcomes=withheld,
        seed=seed,
        is_control=is_control,
        ground_truth={"horizon_steps": max(step for step, _, _ in outcomes)},
    )


def oscillation(seed: int = 0) -> SyntheticSystem:
    """Return a quarter-turn oscillator.

    ``planar_rotation`` updates both values simultaneously as
    ``p' = cosine*p + sine*v`` and ``v' = -sine*p + cosine*v``.
    """
    return _build_fixture(
        name="oscillation",
        regime="oscillation",
        seed=seed,
        variables=(("position", "a.u."), ("velocity", "a.u./h")),
        observations=((1, "position", 1.0), (1, "velocity", 0.0)),
        state={"position": 1.0, "velocity": 0.0},
        graph=SystemGraph(
            nodes=(
                GraphNode(
                    node_id="oscillator",
                    variable_refs=("position", "velocity"),
                ),
            ),
            edges=(),
            topology_metadata={"single_oscillator": True},
        ),
        executor_ref="planar_rotation",
        code_sha256=_PLANAR_ROTATION_SHA256,
        equations=(
            "position_next = cosine * position + sine * velocity",
            "velocity_next = -sine * position + cosine * velocity",
        ),
        rules=("read both current values before writing either next value",),
        transition_metadata={
            "position_variable": "position",
            "velocity_variable": "velocity",
            "cosine": 0.0,
            "sine": 1.0,
            "step_seconds": 3600,
        },
        classification="deterministic",
        outcomes=(
            (1, "position", 0.0),
            (1, "velocity", -1.0),
            (2, "position", -1.0),
            (2, "velocity", 0.0),
            (3, "position", 0.0),
            (3, "velocity", 1.0),
        ),
        model_family="linear_state_space",
        phase_dependencies=("position-velocity phase",),
    )


def deterministic_divergence(seed: int = 0) -> SyntheticSystem:
    """Return x[k+1] = 2*x[k], diverging from the unstable zero point."""
    return _build_fixture(
        name="deterministic_divergence",
        regime="divergence",
        seed=seed,
        variables=(("x", "a.u."),),
        observations=((1, "x", 1.0),),
        state={"x": 1.0},
        graph=SystemGraph(
            nodes=(GraphNode(node_id="state-x", variable_refs=("x",)),),
            edges=(),
            topology_metadata={"single_state": True},
        ),
        executor_ref="linear_affine",
        code_sha256=_LINEAR_AFFINE_SHA256,
        equations=("next = multiplier * current + intercept",),
        rules=("read and write the scalar named by state_variable once per step",),
        transition_metadata={
            "state_variable": "x",
            "multiplier": 2.0,
            "intercept": 0.0,
            "step_seconds": 3600,
        },
        classification="deterministic",
        outcomes=((1, "x", 2.0), (2, "x", 4.0), (3, "x", 8.0)),
        model_family="linear_affine",
    )


def stochastic_branching(seed: int = 0) -> SyntheticSystem:
    """Return a seeded Bernoulli walk using only ``default_rng(seed)`` locally.

    ``bernoulli_step`` draws one uniform value per step. A draw below
    ``up_probability`` adds ``up_step``; otherwise it adds ``down_step``.
    """
    rng = np.random.default_rng(seed)
    current = 0.0
    outcome_values: list[tuple[int, str, object]] = []
    for step in range(1, 5):
        current += 1.0 if rng.random() < 0.6 else -1.0
        outcome_values.append((step, "x", current))
    return _build_fixture(
        name="stochastic_branching",
        regime="stochastic_branching",
        seed=seed,
        variables=(("x", "a.u."),),
        observations=((1, "x", 0.0),),
        state={"x": 0.0},
        graph=SystemGraph(
            nodes=(GraphNode(node_id="state-x", variable_refs=("x",)),),
            edges=(),
            topology_metadata={"single_state": True},
        ),
        executor_ref="bernoulli_step",
        code_sha256=_BERNOULLI_STEP_SHA256,
        equations=(
            "next = current + up_step if draw < up_probability "
            "else current + down_step",
        ),
        rules=("consume exactly one default_rng uniform draw per state step",),
        transition_metadata={
            "state_variable": "x",
            "up_probability": 0.6,
            "up_step": 1.0,
            "down_step": -1.0,
            "step_seconds": 3600,
        },
        classification="stochastic",
        outcomes=tuple(outcome_values),
        model_family="seeded_random_walk",
    )


def coupled_oscillators(seed: int = 0) -> SyntheticSystem:
    """Return one simultaneous sine-coupled phase step.

    Each phase advances by its intrinsic step plus ``coupling*sin(other-self)``
    computed from the two pre-transition phases.
    """
    return _build_fixture(
        name="coupled_oscillators",
        regime="coupled_oscillation",
        seed=seed,
        variables=(("phase_a", "rad"), ("phase_b", "rad")),
        observations=((1, "phase_a", 0.0), (1, "phase_b", 1.5707963267948966)),
        state={"phase_a": 0.0, "phase_b": 1.5707963267948966},
        graph=SystemGraph(
            nodes=(
                GraphNode(node_id="oscillator-a", variable_refs=("phase_a",)),
                GraphNode(node_id="oscillator-b", variable_refs=("phase_b",)),
            ),
            edges=(
                GraphEdge(
                    source="oscillator-a",
                    target="oscillator-b",
                    coupling_type="sine_phase_coupling",
                    strength=0.25,
                    effective_proximity=1.0,
                ),
                GraphEdge(
                    source="oscillator-b",
                    target="oscillator-a",
                    coupling_type="sine_phase_coupling",
                    strength=0.25,
                    effective_proximity=1.0,
                ),
            ),
            topology_metadata={"bidirectional_coupling": True},
        ),
        executor_ref="coupled_phase",
        code_sha256=_COUPLED_PHASE_SHA256,
        equations=(
            "phase_a_next = phase_a + intrinsic_step_a + "
            "coupling * sin(phase_b - phase_a)",
            "phase_b_next = phase_b + intrinsic_step_b + "
            "coupling * sin(phase_a - phase_b)",
        ),
        rules=("compute both next phases from the same pre-transition state",),
        transition_metadata={
            "phase_a_variable": "phase_a",
            "phase_b_variable": "phase_b",
            "intrinsic_step_a": 0.0,
            "intrinsic_step_b": 0.0,
            "coupling": 0.25,
            "step_seconds": 3600,
        },
        classification="deterministic",
        outcomes=(
            (1, "phase_a", 0.25),
            (1, "phase_b", 1.3207963267948966),
        ),
        model_family="coupled_phase",
        relationships_used=("oscillator-a <-> oscillator-b",),
        phase_dependencies=("phase_b - phase_a",),
        effective_proximity_dependencies=(),
    )


def feedback_instability(seed: int = 0) -> SyntheticSystem:
    """Return positive feedback: x' = plant*x + gain*(x-reference)."""
    return _build_fixture(
        name="feedback_instability",
        regime="feedback_instability",
        seed=seed,
        variables=(("x", "a.u."),),
        observations=((1, "x", 1.0),),
        state={"x": 1.0},
        graph=SystemGraph(
            nodes=(GraphNode(node_id="state-x", variable_refs=("x",)),),
            edges=(
                GraphEdge(
                    source="state-x",
                    target="state-x",
                    coupling_type="positive_feedback",
                    strength=0.5,
                ),
            ),
            topology_metadata={"feedback_loop": True},
        ),
        executor_ref="linear_feedback",
        code_sha256=_LINEAR_FEEDBACK_SHA256,
        equations=(
            "next = plant_multiplier * current + feedback_gain * (current - reference)",
        ),
        rules=("positive feedback_gain amplifies displacement from reference",),
        transition_metadata={
            "state_variable": "x",
            "plant_multiplier": 1.0,
            "feedback_gain": 0.5,
            "reference": 0.0,
            "step_seconds": 3600,
        },
        classification="deterministic",
        outcomes=((1, "x", 1.5), (2, "x", 2.25), (3, "x", 3.375)),
        model_family="linear_feedback",
        relationships_used=("state-x -> state-x",),
    )


def hierarchical_nested_dynamics(seed: int = 0) -> SyntheticSystem:
    """Return a three-level vector hierarchy with parent-to-child transfer.

    The root is multiplied by ``root_multiplier``. Every later component is
    updated simultaneously as ``level_multiplier*current + parent_coupling*parent``.
    """
    return _build_fixture(
        name="hierarchical_nested_dynamics",
        regime="nested_dynamics",
        seed=seed,
        variables=(("levels", "a.u."),),
        observations=((1, "levels", (1.0, 0.0, 0.0)),),
        state={"levels": (1.0, 0.0, 0.0)},
        graph=SystemGraph(
            nodes=(
                GraphNode(node_id="root", variable_refs=("levels",)),
                GraphNode(node_id="middle", variable_refs=("levels",)),
                GraphNode(node_id="leaf", variable_refs=("levels",)),
            ),
            edges=(
                GraphEdge(
                    source="root",
                    target="middle",
                    coupling_type="parent_to_child",
                    strength=0.5,
                ),
                GraphEdge(
                    source="middle",
                    target="leaf",
                    coupling_type="parent_to_child",
                    strength=0.5,
                ),
            ),
            geometry_metadata={"levels": 3},
            topology_metadata={"hierarchical": True},
        ),
        executor_ref="nested_linear",
        code_sha256=_NESTED_LINEAR_SHA256,
        equations=(
            "next[0] = root_multiplier * current[0]",
            "next[i] = level_multiplier * current[i] + "
            "parent_coupling * current[i-1] for i >= 1",
        ),
        rules=("compute every vector component from the same pre-transition vector",),
        transition_metadata={
            "state_variable": "levels",
            "root_multiplier": 0.5,
            "level_multiplier": 0.25,
            "parent_coupling": 0.5,
            "step_seconds": 3600,
        },
        classification="deterministic",
        outcomes=(
            (1, "levels", (0.5, 0.5, 0.0)),
            (2, "levels", (0.25, 0.375, 0.25)),
        ),
        model_family="nested_linear",
        relationships_used=("root -> middle", "middle -> leaf"),
    )


SYSTEM_CATALOG: Mapping[str, Callable[[int], SyntheticSystem]] = MappingProxyType(
    {
        "linear_convergence": linear_convergence,
        "oscillation": oscillation,
        "deterministic_divergence": deterministic_divergence,
        "stochastic_branching": stochastic_branching,
        "coupled_oscillators": coupled_oscillators,
        "feedback_instability": feedback_instability,
        "hierarchical_nested_dynamics": hierarchical_nested_dynamics,
    }
)
