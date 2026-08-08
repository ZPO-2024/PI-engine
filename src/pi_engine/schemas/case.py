"""Canonical, cutoff-safe prediction input for PI-engine."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pi_engine.schemas.common import Provenance
from pi_engine.schemas.graph import SystemGraph
from pi_engine.schemas.observation import Observation
from pi_engine.schemas.state import StateEstimate


class VariableDefinition(BaseModel):
    """A canonical variable identity and unit within a case."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    description: str | None = None


class Case(BaseModel):
    """All information available to a prediction at its declared cutoff."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    provenance: Provenance
    prediction_cutoff: datetime
    canonical_variables: tuple[VariableDefinition, ...] = Field(min_length=1)
    observations: tuple[Observation, ...]
    state: StateEstimate
    graph: SystemGraph
    constraints: tuple[str, ...] = ()

    @field_validator("prediction_cutoff")
    @classmethod
    def cutoff_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("prediction_cutoff must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_canonical_references_and_cutoff(self) -> "Case":
        variable_units: dict[str, str] = {}
        for definition in self.canonical_variables:
            if definition.name in variable_units:
                raise ValueError("canonical_variables names must be unique")
            variable_units[definition.name] = definition.unit

        for observation in self.observations:
            if observation.case_id != self.case_id:
                raise ValueError("observation case_id must match the case")
            if observation.available_at > self.prediction_cutoff:
                raise ValueError("observation available_at must not exceed prediction_cutoff")
            expected_unit = variable_units.get(observation.variable)
            if expected_unit is None:
                raise ValueError(
                    f"observation variable is not canonical: {observation.variable}"
                )
            if observation.unit != expected_unit:
                raise ValueError(
                    f"observation unit must be {expected_unit!r} for {observation.variable!r}"
                )

        if self.state.at > self.prediction_cutoff:
            raise ValueError("state time must not exceed prediction_cutoff")

        known_variables = set(variable_units)
        state_components = {
            "observed": self.state.observed,
            "latent": self.state.latent,
            "uncertainty": self.state.uncertainty,
            "boundary": self.state.boundary,
        }
        for component_name, component in state_components.items():
            unknown = set(component) - known_variables
            if unknown:
                unknown_text = ", ".join(sorted(unknown))
                raise ValueError(
                    f"state {component_name} references unknown variables: {unknown_text}"
                )

        for node in self.graph.nodes:
            unknown = set(node.variable_refs) - known_variables
            if unknown:
                unknown_text = ", ".join(sorted(unknown))
                raise ValueError(
                    f"graph variable_refs contain unknown variables: {unknown_text}"
                )

        unknown_boundary_variables = {
            relationship.variable_ref
            for relationship in self.graph.boundary_relationships
            if relationship.variable_ref not in known_variables
        }
        if unknown_boundary_variables:
            unknown_text = ", ".join(sorted(unknown_boundary_variables))
            raise ValueError(
                "boundary relationships reference unknown variables: "
                f"{unknown_text}"
            )
        return self
