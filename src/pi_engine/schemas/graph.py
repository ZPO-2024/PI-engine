"""Inspectable topology schemas for PI-engine cases."""

from types import MappingProxyType
from typing import Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from pi_engine.schemas.common import JsonScalar


class GraphNode(BaseModel):
    """A graph identity tied to one or more canonical case variables."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(min_length=1)
    variable_refs: tuple[str, ...] = Field(min_length=1)


class GraphEdge(BaseModel):
    """An explicit coupling between two graph nodes."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    source: str = Field(min_length=1)
    target: str = Field(min_length=1)
    directed: bool = True
    coupling_type: str = Field(min_length=1)
    strength: float | None = None
    effective_proximity: float | None = None


class BoundaryRelationship(BaseModel):
    """An explicit relationship between a graph node and a boundary variable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str = Field(min_length=1)
    variable_ref: str = Field(min_length=1)
    relationship_type: str = Field(min_length=1)


class SystemGraph(BaseModel):
    """A graph with unambiguous node identities and resolved edge endpoints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    boundary_relationships: tuple[BoundaryRelationship, ...] = ()
    geometry_metadata: Mapping[str, JsonScalar] = Field(
        default_factory=dict, validate_default=True
    )
    topology_metadata: Mapping[str, JsonScalar] = Field(
        default_factory=dict, validate_default=True
    )

    @field_validator("geometry_metadata", "topology_metadata")
    @classmethod
    def freeze_metadata(
        cls, value: Mapping[str, JsonScalar]
    ) -> Mapping[str, JsonScalar]:
        return MappingProxyType(dict(value))

    @field_serializer("geometry_metadata", "topology_metadata")
    def serialize_metadata(
        self, value: Mapping[str, JsonScalar]
    ) -> dict[str, JsonScalar]:
        return dict(value)

    @model_validator(mode="after")
    def validate_references(self) -> "SystemGraph":
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph node_id values must be unique")

        known_nodes = set(node_ids)
        for edge in self.edges:
            missing = {edge.source, edge.target} - known_nodes
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ValueError(f"graph edge endpoint is unknown: {missing_text}")

        for relationship in self.boundary_relationships:
            if relationship.node_id not in known_nodes:
                raise ValueError(
                    "boundary relationship references unknown node: "
                    f"{relationship.node_id}"
                )
        return self
