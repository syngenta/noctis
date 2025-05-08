from pydantic import BaseModel, field_validator, Field, ConfigDict, model_validator

from typing import Annotated
from pydantic import StringConstraints

from typing import Optional


class Node(BaseModel):
    """Represents a node in a graph with a unique identifier and optional properties."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)
    node_label: Annotated[str, StringConstraints(pattern=r"^[A-Z][\w\-]*$")]
    uid: Annotated[str, StringConstraints(pattern=r"^[A-Z]{0,3}{1,2}\d+$")]
    properties: Annotated[Optional[dict], Field(default={})]

    def __hash__(self):
        return self.uid.__hash__()

    def __eq__(self, other):
        return isinstance(other, Node) and self.uid == other.uid

    def get(self, attribute_name: str) -> any:
        return getattr(self, attribute_name)

    def __str__(self) -> str:
        properties_preview = ", ".join(f"{k}: {v}" for k, v in self.properties.items())
        return f"Node {{label={self.node_label}, uid={self.uid}, properties={properties_preview} }})"


class Relationship(BaseModel):
    """Represents a relationship between two nodes in a graph."""

    model_config = ConfigDict(populate_by_name=True, frozen=True)
    relationship_type: Annotated[str, StringConstraints(pattern=r"^[A-Z\d_-]+$")]

    start_node: Node
    end_node: Node
    properties: Annotated[Optional[dict], Field(default={})]

    @property
    def start_end_uids(self) -> tuple[str, str]:
        return self.start_node.uid, self.end_node.uid

    def __hash__(self):
        return self.start_end_uids.__hash__()

    def __eq__(self, other):
        return (
            isinstance(other, Relationship)
            and self.start_end_uids == other.start_end_uids
        )

    def get(self, attribute_name: str) -> any:
        return getattr(self, attribute_name)

    def __str__(self) -> str:
        properties_preview = ", ".join(f"{k}: {v}" for k, v in self.properties.items())
        return (
            f"Relationship {{type={self.relationship_type}, "
            f"start={self.start_node.uid}, end={self.end_node.uid}, "
            f"properties= {properties_preview} }})"
        )


class GraphRecord(BaseModel):
    """Represents a record of nodes and relationships forming a graph."""

    nodes: list[Node] = Field(default=[])
    relationships: list[Relationship] = Field(default=[])

    @field_validator("nodes", mode="after")
    def ensure_unique_nodes(cls, v):
        unique_nodes = set(v)
        return list(unique_nodes)

    @field_validator("relationships", mode="after")
    def ensure_unique_relationships(cls, v):
        unique_relationships = set(v)
        return list(unique_relationships)

    @model_validator(mode="after")
    def ensure_relationship_nodes_present(self) -> "GraphRecord":
        if not self.relationships:
            return self

        unique_nodes_uids = {node.uid for node in self.nodes}
        new_nodes = []

        unique_nodes_from_relationships = {
            node.uid: node
            for r in self.relationships
            for node in (r.start_node, r.end_node)
        }
        for node_uid in unique_nodes_from_relationships.keys():
            if node_uid not in unique_nodes_uids:
                new_nodes.append(unique_nodes_from_relationships[node_uid])

        if new_nodes:
            self.nodes.extend(new_nodes)

        return self

    @property
    def nodes_uids(self) -> set[str]:
        return {node.uid for node in self.nodes}

    @property
    def relationships_nodes_uids(self) -> set[str]:
        return {node_uid for r in self.relationships for node_uid in r.start_end_uids}

    def get_nodes_with_label(self, input_label: str) -> list[Node]:
        return [node for node in self.nodes if node.node_label == input_label]

    def get_all_relationships_start_end_uids(self) -> list[tuple[str, str]]:
        return [rel.start_end_uids for rel in self.relationships]

    def __eq__(self, other: "GraphRecord") -> bool:
        if not isinstance(other, GraphRecord):
            raise TypeError(
                f"Cannot compare {type(self).__name__} with {type(other).__name__}"
            )

        return (
            self.nodes_uids == other.nodes_uids
            and self.get_all_relationships_start_end_uids()
            == other.get_all_relationships_start_end_uids()
        )

    def __str__(self) -> str:
        nodes_preview = ", ".join(
            f"\nNode {{label: {node.node_label}, uid: {node.uid}, properties: {node.properties}}}"
            for node in self.nodes
        )
        relationships_preview = ", ".join(
            f"\nRelationship {{type: {rel.relationship_type}, start: {rel.start_node.uid}, end: {rel.end_node.uid}}}"
            for rel in self.relationships
        )

        return (
            f"\nRecord("
            f"\nnodes = [{nodes_preview}],"
            f"\nrelationships = [{relationships_preview}])"
        )
