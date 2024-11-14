from pydantic import BaseModel, field_validator, Field, ConfigDict, model_validator

from typing_extensions import Annotated
from pydantic import StringConstraints

from typing import List, Optional


# TODO: load_from_dict methods to be revised


class DataModelAttributes:
    RELATIONSHIP_TYPE = "relationship_type"
    START_NODE = "start_node"
    END_NODE = "end_node"
    RELATIONSHIP_PROPERTIES = "properties"
    NODE_LABEL = "node_label"
    UID = "uid"
    NODE_PROPERTIES = "properties"


class Node(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    node_label: Annotated[
        str, StringConstraints(pattern=r"^[A-Z][\w\-.]*$"), Field(alias="label")
    ]
    uid: Annotated[
        str, StringConstraints(pattern=r"^[A-Z]{1,2}\d+$"), Field(alias="gid")
    ]
    properties: Annotated[Optional[dict], Field(default={}, alias="properties")]

    def __hash__(self):
        return self.uid.__hash__()

    def __eq__(self, other):
        return isinstance(other, Node) and self.uid == other.uid

    @classmethod
    def get_alias(cls, attribute_name: str) -> str:
        return cls.model_fields[attribute_name].alias

    def get(self, attribute_name: str) -> any:
        return getattr(self, attribute_name)


class Relationship(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)
    relationship_type: Annotated[
        str, StringConstraints(pattern=r"^[A-Z0-9_.-]+$"), Field(alias="type")
    ]
    start_node: Annotated[Node, Field(alias="startnode")]
    end_node: Annotated[Node, Field(alias="endnode")]
    properties: Annotated[Optional[dict], Field(default={}, alias="properties")]

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

    @classmethod
    def get_attribute_alias(cls, attribute_name: str) -> str:
        return cls.model_fields[attribute_name].alias

    def get(self, attribute_name: str) -> any:
        return getattr(self, attribute_name)


class GraphRecord(BaseModel):
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


class DataContainer(BaseModel):
    records: dict[int, GraphRecord] = Field(default={})

    def __eq__(self, other):
        if isinstance(other, DataContainer):
            if set(self.records.keys()) == set(other.records.keys()):
                return all(
                    self.records[key] == other.records[key]
                    for key in self.records.keys()
                )
        return False

    def add_record(self, record: GraphRecord, record_key: Optional[int] = None) -> None:
        """To add a GraphRecord to a DataContainer"""
        max_key = max(self.records.keys(), default=-1)
        record_key = record_key or max_key + 1
        self.records[record_key] = record

    def get_record(self, record_key: int) -> GraphRecord:
        return self.records[record_key]

    def get_records(self, record_keys: List[int]) -> List[GraphRecord]:
        return [self.records[key] for key in record_keys]

    def get_subcontainer_with_records(self, record_keys: List[int]) -> "DataContainer":
        subcontainer = DataContainer()
        missing_keys: set[int] = set()

        for key in record_keys:
            if key in self.records:
                subcontainer.add_record(
                    self.records[key].__deepcopy__(), record_key=key
                )
            else:
                missing_keys.add(key)

        if missing_keys:
            missing_keys_str = ", ".join(map(str, missing_keys))
            raise KeyError(
                f"Record keys {missing_keys_str} not found in DataContainer."
            )

        return subcontainer
