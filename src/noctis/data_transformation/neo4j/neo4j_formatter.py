import neo4j
from neo4j._sync.work.result import Result
from tqdm import tqdm

from noctis.data_architecture.datamodel import (
    Node,
    Relationship,
    GraphRecord,
    DataContainer,
)


class Neo4jResultFormatter:
    @staticmethod
    def _format_node(node) -> Node:
        _label = list(node.labels)[0] if node.labels else None
        uid_key = "uid"
        _uid = node.get(uid_key)
        _properties = {key: value for key, value in node.items() if key != uid_key}
        return Node(node_label=_label, uid=_uid, properties=_properties)

    @classmethod
    def format_nodes(cls, record):
        return [cls._format_node(node) for node in record["nodes"]]

    @classmethod
    def _format_relationship(cls, relationship) -> Relationship:
        _type = relationship.type
        _start_node = cls._format_node(relationship.nodes[0])
        _end_node = cls._format_node(relationship.nodes[1])
        return Relationship(
            relationship_type=_type, start_node=_start_node, end_node=_end_node
        )

    @classmethod
    def format_relationships(cls, record):
        if "relationships" not in record.keys():
            return []
        return [
            cls._format_relationship(relationship)
            for relationship in record["relationships"]
        ]


def select_formatter(result):
    if isinstance(result, neo4j._sync.work.result.Result):
        return Neo4jResultFormatter()
    else:
        raise ValueError("Unsupported result class")


def format_result(result):
    # print("Formatting started")
    formatter = select_formatter(result)
    formatted_result = DataContainer()
    for record in result:
        formatted_record = GraphRecord(
            nodes=formatter.format_nodes(record),
            relationships=formatter.format_relationships(record),
        )
        formatted_result.add_record(formatted_record)
    return formatted_result
