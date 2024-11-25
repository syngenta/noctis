import pytest
from pydantic import ValidationError

from noctis.data_architecture.datamodel import (
    Node,
    Relationship,
    GraphRecord,
    DataContainer,
)


def test_node_creation_valid():
    # Test valid node creation
    node = Node(node_label="MyNode", uid="AB123", properties={"key": "value"})
    assert node.node_label == "MyNode"
    assert node.uid == "AB123"
    assert node.properties == {"key": "value"}


def test_node_creation_invalid_label():
    # Test invalid node label
    with pytest.raises(ValidationError):
        Node(node_label="myNode", uid="AB123")


def test_node_creation_invalid_uid():
    # Test invalid uid
    with pytest.raises(ValidationError):
        Node(node_label="MyNode", uid="invalid")


def test_node_hash():
    # Test node hash
    node1 = Node(node_label="MyNode", uid="AB123")
    node2 = Node(node_label="MyNode", uid="AB123")
    assert hash(node1) == hash(node2)


def test_get_alias_nodes():
    # Test get_alias method
    assert Node.get_alias("node_label") == "label"
    assert Node.get_alias("uid") == "gid"
    assert Node.get_alias("properties") == "properties"


def test_get_attribute():
    # Test get method
    node = Node(node_label="MyNode", uid="AB123", properties={"key": "value"})
    assert node.get("node_label") == "MyNode"
    assert node.get("uid") == "AB123"
    assert node.get("properties") == {"key": "value"}


def test_relationship_creation_valid():
    # Test valid relationship creation
    start_node = Node(node_label="StartNode", uid="AB123")
    end_node = Node(node_label="EndNode", uid="CD456")
    relationship = Relationship(
        relationship_type="RELATED_TO",
        start_node=start_node,
        end_node=end_node,
        properties={"key": "value"},
    )
    assert relationship.relationship_type == "RELATED_TO"
    assert relationship.start_node == start_node
    assert relationship.end_node == end_node
    assert relationship.properties == {"key": "value"}


def test_relationship_creation_invalid_type():
    # Test invalid relationship type
    start_node = Node(node_label="StartNode", uid="AB123")
    end_node = Node(node_label="EndNode", uid="CD456")
    with pytest.raises(ValidationError):
        Relationship(
            relationship_type="Invalid Type", start_node=start_node, end_node=end_node
        )


def test_start_end_uids():
    # Test start_end_uids property
    start_node = Node(node_label="StartNode", uid="AB123")
    end_node = Node(node_label="EndNode", uid="CD456")
    relationship = Relationship(
        relationship_type="RELATED_TO", start_node=start_node, end_node=end_node
    )
    assert relationship.start_end_uids == ("AB123", "CD456")


def test_relationship_hash():
    # Test relationship hash
    start_node1 = Node(node_label="StartNode", uid="AB123")
    end_node1 = Node(node_label="EndNode", uid="CD456")
    relationship1 = Relationship(
        relationship_type="RELATED_TO", start_node=start_node1, end_node=end_node1
    )

    start_node2 = Node(node_label="StartNode", uid="AB123")
    end_node2 = Node(node_label="EndNode", uid="CD456")
    relationship2 = Relationship(
        relationship_type="RELATED_TO", start_node=start_node2, end_node=end_node2
    )

    start_node3 = Node(node_label="StartNode", uid="AB123")
    end_node3 = Node(node_label="EndNode", uid="CD789")
    relationship3 = Relationship(
        relationship_type="RELATED_TO", start_node=start_node3, end_node=end_node3
    )

    assert hash(relationship1) == hash(relationship2)
    assert hash(relationship1) != hash(relationship3)


def test_get_attribute_relationship():
    # Test get method
    start_node = Node(node_label="StartNode", uid="AB123")
    end_node = Node(node_label="EndNode", uid="CD456")
    relationship = Relationship(
        relationship_type="RELATED_TO",
        start_node=start_node,
        end_node=end_node,
        properties={"key": "value"},
    )
    assert relationship.get("relationship_type") == "RELATED_TO"
    assert relationship.get("start_node") == start_node
    assert relationship.get("end_node") == end_node
    assert relationship.get("properties") == {"key": "value"}


def test_ensure_unique_nodes():
    # Test ensure_unique_nodes validator
    node1 = Node(node_label="Node1", uid="AB123")
    node2 = Node(node_label="Node2", uid="CD456")
    graph_record = GraphRecord(nodes=[node1, node2, node1])
    assert len(graph_record.nodes) == 2
    assert node1 in graph_record.nodes
    assert node2 in graph_record.nodes


def test_ensure_unique_relationships():
    # Test ensure_unique_relationships validator
    start_node = Node(node_label="StartNode", uid="AB123")
    end_node = Node(node_label="EndNode", uid="CD456")
    rel1 = Relationship(
        relationship_type="RELATED_TO", start_node=start_node, end_node=end_node
    )
    rel2 = Relationship(
        relationship_type="RELATED_TO", start_node=start_node, end_node=end_node
    )
    graph_record = GraphRecord(relationships=[rel1, rel2])
    assert len(graph_record.relationships) == 1
    assert rel1 in graph_record.relationships


def test_ensure_relationship_nodes_present():
    # Test ensure_relationship_nodes_present validator
    start_node = Node(node_label="StartNode", uid="AB123")
    end_node = Node(node_label="EndNode", uid="CD456")
    rel = Relationship(
        relationship_type="RELATED_TO", start_node=start_node, end_node=end_node
    )
    graph_record = GraphRecord(nodes=[start_node], relationships=[rel])
    assert len(graph_record.nodes) == 2
    assert start_node in graph_record.nodes
    assert end_node in graph_record.nodes


def test_nodes_uids():
    # Test nodes_uids property
    node1 = Node(node_label="Node1", uid="AB123")
    node2 = Node(node_label="Node2", uid="CD456")
    graph_record = GraphRecord(nodes=[node1, node2])
    assert graph_record.nodes_uids == {"AB123", "CD456"}


def test_relationships_nodes_uids():
    # Test relationships_nodes_uids property
    start_node = Node(node_label="StartNode", uid="AB123")
    end_node = Node(node_label="EndNode", uid="CD456")
    rel = Relationship(
        relationship_type="RELATED_TO", start_node=start_node, end_node=end_node
    )
    graph_record = GraphRecord(relationships=[rel])
    assert graph_record.relationships_nodes_uids == {"AB123", "CD456"}


def test_get_nodes_with_label():
    # Test get_nodes_with_label method
    node1 = Node(node_label="Node1", uid="AB123")
    node2 = Node(node_label="Node2", uid="CD456")
    node3 = Node(node_label="Node1", uid="EF789")
    graph_record = GraphRecord(nodes=[node1, node2, node3])
    nodes_with_label = graph_record.get_nodes_with_label("Node1")
    assert len(nodes_with_label) == 2
    assert node1 in nodes_with_label
    assert node3 in nodes_with_label


def test_get_all_relationships_start_end_uids():
    # Test get_all_relationships_start_end_uids method
    start_node1 = Node(node_label="StartNode1", uid="AB123")
    end_node1 = Node(node_label="EndNode1", uid="CD456")
    rel1 = Relationship(
        relationship_type="RELATED_TO", start_node=start_node1, end_node=end_node1
    )

    start_node2 = Node(node_label="StartNode2", uid="EF789")
    end_node2 = Node(node_label="EndNode2", uid="GH012")
    rel2 = Relationship(
        relationship_type="RELATED_TO", start_node=start_node2, end_node=end_node2
    )

    graph_record = GraphRecord(relationships=[rel1, rel2])
    start_end_uids = graph_record.get_all_relationships_start_end_uids()
    assert len(start_end_uids) == 2
    assert ("AB123", "CD456") in start_end_uids
    assert ("EF789", "GH012") in start_end_uids


def test_equality():
    # Test __eq__ method
    node1 = Node(node_label="Node1", uid="AB123")
    node2 = Node(node_label="Node2", uid="CD456")
    start_node = Node(node_label="StartNode", uid="EF789")
    end_node = Node(node_label="EndNode", uid="GH012")
    rel = Relationship(
        relationship_type="RELATED_TO", start_node=start_node, end_node=end_node
    )

    graph_record1 = GraphRecord(nodes=[node1, node2], relationships=[rel])
    graph_record2 = GraphRecord(nodes=[node2, node1], relationships=[rel])
    assert graph_record1 == graph_record2

    graph_record3 = GraphRecord(nodes=[node1], relationships=[])
    assert graph_record1 != graph_record3

    with pytest.raises(TypeError):
        graph_record1 == "not a GraphRecord"


def test_data_container_equality():
    # Test __eq__ method
    node1 = Node(node_label="Node1", uid="AB123")
    node2 = Node(node_label="Node2", uid="CD456")
    rel = Relationship(relationship_type="RELATED_TO", start_node=node1, end_node=node2)
    record1 = GraphRecord(nodes=[node1, node2], relationships=[rel])
    record2 = GraphRecord(nodes=[node2, node1], relationships=[rel])

    container1 = DataContainer(records={0: record1, 1: record2})
    container2 = DataContainer(records={1: record2, 0: record1})
    assert container1 == container2

    container3 = DataContainer(records={0: record1})
    assert container1 != container3

    assert container1 != "not a DataContainer"


def test_add_record():
    # Test add_record method
    node1 = Node(node_label="Node1", uid="AB123")
    record1 = GraphRecord(nodes=[node1])
    container = DataContainer()
    container.add_record(record1)
    assert container.records[0] == record1

    record2 = GraphRecord(nodes=[node1])
    container.add_record(record2, record_key=2)
    assert container.records[2] == record2


def test_get_record():
    # Test get_record method
    node1 = Node(node_label="Node1", uid="AB123")
    record1 = GraphRecord(nodes=[node1])

    container = DataContainer(records={0: record1})
    assert container.get_record(0) == record1

    with pytest.raises(KeyError):
        container.get_record(1)


def test_get_records():
    # Test get_records method
    node1 = Node(node_label="Node1", uid="AB123")
    record1 = GraphRecord(nodes=[node1])
    record2 = GraphRecord(nodes=[node1])

    container = DataContainer(records={0: record1, 1: record2})
    records = container.get_records([0, 1])
    assert len(records) == 2
    assert record1 in records
    assert record2 in records


def test_get_subcontainer_with_records():
    # Test get_subcontainer_with_records method
    node1 = Node(node_label="Node1", uid="AB123")
    record1 = GraphRecord(nodes=[node1])
    record2 = GraphRecord(nodes=[node1])

    container = DataContainer(records={0: record1, 1: record2})
    subcontainer = container.get_subcontainer_with_records([0, 1])
    assert len(subcontainer.records) == 2
    assert subcontainer.records[0] == record1
    assert subcontainer.records[1] == record2
    assert subcontainer.records[0] is not record1  # Deep copy check

    with pytest.raises(KeyError):
        container.get_subcontainer_with_records([0, 2])
