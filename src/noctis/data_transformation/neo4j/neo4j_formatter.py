from typing import Union, List, Any

import neo4j

# from neo4j._sync.work.result import Result
from neo4j import Result
from neo4j import Record
from neo4j.graph import Node as neo4jNode
from neo4j.graph import Relationship as neo4jRelationship
from neo4j.graph import Path as neo4jPath
from typing import Union

from noctis.data_architecture.datamodel import (
    Node,
    Relationship,
    GraphRecord,
    DataContainer,
)
from noctis.utilities import console_logger

logger = console_logger(__name__)


class Neo4jResultFormatter:
    TYPE_METHOD_MAP: dict[type, str] = {
        neo4jNode: "_format_node",
        neo4jRelationship: "_format_relationship",
        neo4jPath: "_format_path",
        list: "_format_list",
    }

    @classmethod
    def extract_nodes_and_relationships(
        cls, record: Record
    ) -> tuple[list[Node], list[Relationship]]:
        return cls._process_values(record.values())

    @classmethod
    def _process_values(
        cls, values: List[Any]
    ) -> tuple[List[Node], List[Relationship]]:
        nodes = []
        relationships = []
        for value in values:
            method_name = cls._get_method_name(value)
            if method_name:
                method = getattr(cls, method_name)
                new_nodes, new_relationships = method(value)
                nodes.extend(new_nodes)
                relationships.extend(new_relationships)
            else:
                logger.warning(
                    f"Non-graph object of type {type(value)} encountered. It will be ignored"
                )
        return nodes, relationships

    @classmethod
    def _get_method_name(cls, value: Any) -> Union[str, None]:
        return next(
            (
                method
                for type_, method in cls.TYPE_METHOD_MAP.items()
                if isinstance(value, type_)
            ),
            None,
        )

    @classmethod
    def _format_node(cls, node: neo4jNode) -> tuple[List[Node], List[Relationship]]:
        return [cls._create_node(node)], []

    @classmethod
    def _format_relationship(
        cls, relationships: neo4jRelationship
    ) -> tuple[List[Node], List[Relationship]]:
        return [], [cls._create_relationship(relationships)]

    @classmethod
    def _format_path(cls, path: neo4jPath) -> tuple[list[Node], list[Relationship]]:
        nodes = [cls._create_node(node) for node in path.nodes]
        relationships = [
            cls._create_relationship(relationship)
            for relationship in path.relationships
        ]
        return nodes, relationships

    @classmethod
    def _format_list(cls, input_list: list) -> tuple[list[Node], list[Relationship]]:
        return cls._process_values(input_list)

    @staticmethod
    def _create_node(node: neo4jNode) -> Node:
        _label = list(node.labels)[0] if node.labels else None
        uid_key = "uid"
        _uid = node.get(uid_key)
        _properties = {key: value for key, value in node.items() if key != uid_key}
        return Node(node_label=_label, uid=_uid, properties=_properties)

    @classmethod
    def _create_relationship(cls, relationship: neo4jRelationship) -> Relationship:
        _type = relationship.type
        _start_node = cls._create_node(relationship.nodes[0])
        _end_node = cls._create_node(relationship.nodes[1])
        return Relationship(
            relationship_type=_type, start_node=_start_node, end_node=_end_node
        )


def format_result(result: Union[list[Record], Result]):
    # print("Formatting started")
    formatter = Neo4jResultFormatter()
    formatted_result = DataContainer()
    for record in result:
        nodes, relationships = formatter.extract_nodes_and_relationships(record)
        formatted_record = GraphRecord(
            nodes=nodes,
            relationships=relationships,
        )
        formatted_result.add_record(formatted_record)
    return formatted_result
