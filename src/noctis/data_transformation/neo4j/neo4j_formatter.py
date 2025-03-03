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
)
from noctis.data_architecture.datacontainer import DataContainer
from noctis.utilities import console_logger

from noctis import settings

logger = console_logger(__name__)


class Neo4jResultFormatter:
    """
    A class to format Neo4j query results into nodes and relationships.

    Attributes:
        TYPE_METHOD_MAP (dict[type, str]): Mapping of types to formatting method names.
    """

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
        """
        Extract nodes and relationships from a Neo4j record.

        Args:
            record (Record): A Neo4j record containing graph data.

        Returns:
            tuple[list[Node], list[Relationship]]: Lists of Node and Relationship objects.
        """
        return cls._process_values(record.values())

    @classmethod
    def _process_values(
        cls, values: List[Any]
    ) -> tuple[List[Node], List[Relationship]]:
        """
        Process values from a Neo4j record to extract nodes and relationships.

        Args:
            values (list[Any]): Values from a Neo4j record.

        Returns:
            tuple[list[Node], list[Relationship]]: Lists of Node and Relationship objects.
        """
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
        """
        Get the method name for formatting based on the type of value.

        Args:
            value (Any): A value from a Neo4j record.

        Returns:
            Union[str, None]: The name of the method to format the value, or None if no method is found.
        """
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
        """
        Format a Neo4j node into a Node object.

        Args:
            node (neo4jNode): A Neo4j node.

        Returns:
            tuple[list[Node], list[Relationship]]: A list containing one Node object and an empty list for relationships.
        """
        return [cls._create_node(node)], []

    @classmethod
    def _format_relationship(
        cls, relationships: neo4jRelationship
    ) -> tuple[List[Node], List[Relationship]]:
        """
        Format a Neo4j relationship into a Relationship object.

        Args:
            relationships (neo4jRelationship): A Neo4j relationship.

        Returns:
            tuple[list[Node], list[Relationship]]: An empty list for nodes and a list containing one Relationship object.
        """
        return [], [cls._create_relationship(relationships)]

    @classmethod
    def _format_path(cls, path: neo4jPath) -> tuple[list[Node], list[Relationship]]:
        """
        Format a Neo4j path into lists of Node and Relationship objects.

        Args:
            path (neo4jPath): A Neo4j path.

        Returns:
            tuple[list[Node], list[Relationship]]: Lists of Node and Relationship objects extracted from the path.
        """
        nodes = [cls._create_node(node) for node in path.nodes]
        relationships = [
            cls._create_relationship(relationship)
            for relationship in path.relationships
        ]
        return nodes, relationships

    @classmethod
    def _format_list(cls, input_list: list) -> tuple[list[Node], list[Relationship]]:
        """
        Format a list of values into nodes and relationships.

        Args:
            input_list (list): A list of values from a Neo4j record.

        Returns:
            tuple[list[Node], list[Relationship]]: Lists of Node and Relationship objects.
        """
        return cls._process_values(input_list)

    @staticmethod
    def _create_node(node: neo4jNode) -> Node:
        """
        Create a Node object from a Neo4j node.

        Args:
            node (neo4jNode): A Neo4j node.

        Returns:
            Node: A Node object representing the Neo4j node.
        """
        _label = list(node.labels)[0] if node.labels else None
        uid_key = "uid"
        _uid = node.get(uid_key)
        _properties = {key: value for key, value in node.items() if key != uid_key}
        return Node(node_label=_label, uid=_uid, properties=_properties)

    @classmethod
    def _create_relationship(cls, relationship: neo4jRelationship) -> Relationship:
        """
        Create a Relationship object from a Neo4j relationship.

        Args:
            relationship (neo4jRelationship): A Neo4j relationship.

        Returns:
            Relationship: A Relationship object representing the Neo4j relationship.
        """
        _type = relationship.type
        _start_node = cls._create_node(relationship.nodes[0])
        _end_node = cls._create_node(relationship.nodes[1])
        return Relationship(
            relationship_type=_type, start_node=_start_node, end_node=_end_node
        )


def format_result(result: Union[list[Record], Result], ce_label: str) -> DataContainer:
    """
    Format a Neo4j result into a DataContainer of GraphRecords.

    Args:
        result (Union[list[Record], Result]): A list of Neo4j records or a Result object.
        ce_label (str): Label for chemical equations.

    Returns:
        DataContainer: A DataContainer containing formatted GraphRecords.
    """

    formatter = Neo4jResultFormatter()
    formatted_result = DataContainer()
    formatted_result.set_ce_label(ce_label)
    for record in result:
        nodes, relationships = formatter.extract_nodes_and_relationships(record)
        formatted_record = GraphRecord(
            nodes=nodes,
            relationships=relationships,
        )
        formatted_result.add_record(formatted_record)
    return formatted_result
