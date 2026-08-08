"""Inspectable topology schemas for PI-engine cases."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class SystemGraph(BaseModel):
    """A graph with unambiguous node identities and resolved edge endpoints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]

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
        return self
