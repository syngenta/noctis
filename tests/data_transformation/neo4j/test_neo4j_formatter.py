import unittest
from unittest.mock import Mock, patch
from noctis.data_transformation.neo4j.neo4j_formatter import (
    Neo4jResultFormatter,
    format_result,
    Node,
    Relationship,
    GraphRecord,
    DataContainer,
)

from neo4j import Result
from neo4j import Record
from neo4j.graph import Node as neo4jNode
from neo4j.graph import Relationship as neo4jRelationship
from neo4j.graph import Path as neo4jPath


class TestNeo4jResultFormatter(unittest.TestCase):
    def setUp(self):
        self.formatter = Neo4jResultFormatter()

    def test_extract_nodes_and_relationships(self):
        mock_record = Mock(spec=Record)
        mock_record.values.return_value = [
            Mock(spec=neo4jNode),
            Mock(spec=neo4jRelationship),
        ]

        with patch.object(
            Neo4jResultFormatter, "_format_node"
        ) as mock_format_node, patch.object(
            Neo4jResultFormatter, "_format_relationship"
        ) as mock_format_relationship:
            mock_format_node.return_value = ([Node(node_label="Test", uid="T123")], [])
            mock_format_relationship.return_value = (
                [],
                [
                    Relationship(
                        relationship_type="TEST",
                        start_node=Node(node_label="Start", uid="T1"),
                        end_node=Node(node_label="End", uid="T2"),
                    )
                ],
            )

            nodes, relationships = Neo4jResultFormatter.extract_nodes_and_relationships(
                mock_record
            )

        self.assertEqual(len(nodes), 1)
        self.assertEqual(len(relationships), 1)
        self.assertIsInstance(nodes[0], Node)
        self.assertIsInstance(relationships[0], Relationship)

    def test_create_node(self):
        mock_node = Mock()
        mock_node.labels = ["M1"]
        mock_node.get.return_value = "M123"
        mock_node.items.return_value = [("uid", "M123"), ("prop1", "value1")]

        result = self.formatter._create_node(mock_node)

        self.assertIsInstance(result, Node)
        self.assertEqual(result.node_label, "M1")
        self.assertEqual(result.uid, "M123")
        self.assertEqual(result.properties, {"prop1": "value1"})

    def test_format_node(self):
        mock_record = Mock()
        with patch.object(Neo4jResultFormatter, "_create_node") as mock_create_node:
            mock_create_node.side_effect = [Node(node_label="M1", uid="M111")]
            result = self.formatter._format_node(mock_record)

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], list)
        self.assertIsInstance(result[1], list)
        self.assertIsInstance(result[0][0], Node)
        self.assertEqual(result[1], [])

    def test_format_relationship(self):
        mock_rel = Mock()
        mock_rel.type = "REL"
        mock_rel.nodes = [Mock(), Mock()]

        with patch.object(Neo4jResultFormatter, "_create_node") as mock_create_node:
            mock_create_node.side_effect = [
                Node(node_label="M1", uid="M111"),
                Node(node_label="M2", uid="M222"),
            ]
            result = self.formatter._format_relationship(mock_rel)

        self.assertIsInstance(result[1][0], Relationship)
        self.assertEqual(result[1][0].relationship_type, "REL")
        self.assertIsInstance(result[1][0].start_node, Node)
        self.assertIsInstance(result[1][0].end_node, Node)

    def test_process_values(self):
        mock_values = [Mock(spec=neo4jNode), Mock(spec=neo4jRelationship), "string"]

        with patch.object(
            Neo4jResultFormatter, "_get_method_name"
        ) as mock_get_method_name, patch.object(
            Neo4jResultFormatter, "_format_node"
        ) as mock_format_node, patch.object(
            Neo4jResultFormatter, "_format_relationship"
        ) as mock_format_relationship:
            mock_get_method_name.side_effect = [
                "_format_node",
                "_format_relationship",
                None,
            ]
            mock_format_node.return_value = ([Node(node_label="Test", uid="T123")], [])
            mock_format_relationship.return_value = (
                [],
                [
                    Relationship(
                        relationship_type="TEST",
                        start_node=Node(node_label="Start", uid="T1"),
                        end_node=Node(node_label="End", uid="T2"),
                    )
                ],
            )

            nodes, relationships = Neo4jResultFormatter._process_values(mock_values)

        self.assertEqual(len(nodes), 1)
        self.assertEqual(len(relationships), 1)

    def test_get_method_name(self):
        self.assertEqual(
            Neo4jResultFormatter._get_method_name(Mock(spec=neo4jNode)), "_format_node"
        )
        self.assertEqual(
            Neo4jResultFormatter._get_method_name(Mock(spec=neo4jRelationship)),
            "_format_relationship",
        )
        self.assertEqual(
            Neo4jResultFormatter._get_method_name(Mock(spec=neo4jPath)), "_format_path"
        )
        self.assertEqual(Neo4jResultFormatter._get_method_name([]), "_format_list")
        self.assertIsNone(Neo4jResultFormatter._get_method_name("string"))

    def test_format_path(self):
        mock_path = Mock(spec=neo4jPath)
        mock_path.nodes = [Mock(spec=neo4jNode), Mock(spec=neo4jNode)]
        mock_path.relationships = [Mock(spec=neo4jRelationship)]

        with patch.object(
            Neo4jResultFormatter, "_create_node"
        ) as mock_create_node, patch.object(
            Neo4jResultFormatter, "_create_relationship"
        ) as mock_create_relationship:
            mock_create_node.side_effect = [
                Node(node_label="Node1", uid="T1"),
                Node(node_label="Node2", uid="T2"),
            ]
            mock_create_relationship.return_value = Relationship(
                relationship_type="REL",
                start_node=Node(node_label="Node1", uid="T1"),
                end_node=Node(node_label="Node2", uid="T2"),
            )

            nodes, relationships = Neo4jResultFormatter._format_path(mock_path)

        self.assertEqual(len(nodes), 2)
        self.assertEqual(len(relationships), 1)
        self.assertIsInstance(nodes[0], Node)
        self.assertIsInstance(relationships[0], Relationship)

    def test_format_list(self):
        mock_list = [Mock(spec=neo4jNode), Mock(spec=neo4jRelationship)]

        with patch.object(
            Neo4jResultFormatter, "_process_values"
        ) as mock_process_values:
            mock_process_values.return_value = (
                [Node(node_label="Test", uid="T123")],
                [
                    Relationship(
                        relationship_type="TEST",
                        start_node=Node(node_label="Start", uid="T1"),
                        end_node=Node(node_label="End", uid="T2"),
                    )
                ],
            )

            nodes, relationships = Neo4jResultFormatter._format_list(mock_list)

        self.assertEqual(len(nodes), 1)
        self.assertEqual(len(relationships), 1)
        self.assertIsInstance(nodes[0], Node)
        self.assertIsInstance(relationships[0], Relationship)

    def test_create_relationship(self):
        mock_relationship = Mock(spec=neo4jRelationship)
        mock_relationship.type = "TEST_REL"
        mock_relationship.nodes = [Mock(spec=neo4jNode), Mock(spec=neo4jNode)]

        with patch.object(Neo4jResultFormatter, "_create_node") as mock_create_node:
            mock_create_node.side_effect = [
                Node(node_label="Start", uid="T1"),
                Node(node_label="End", uid="T2"),
            ]

            result = Neo4jResultFormatter._create_relationship(mock_relationship)

        self.assertIsInstance(result, Relationship)
        self.assertEqual(result.relationship_type, "TEST_REL")
        self.assertEqual(result.start_node.uid, "T1")
        self.assertEqual(result.end_node.uid, "T2")


class TestFormatResult(unittest.TestCase):
    @patch("noctis.data_transformation.neo4j.neo4j_formatter.Neo4jResultFormatter")
    def test_format_result(self, mock_neo4j_formatter):
        # Create a mock formatter
        mock_formatter = Mock()

        # Set up the mock formatter to return specific nodes and relationships
        mock_formatter.extract_nodes_and_relationships.return_value = (
            [Node(node_label="M1", uid="M111")],
            [
                Relationship(
                    relationship_type="R1",
                    start_node=Node(node_label="M1", uid="M111"),
                    end_node=Node(node_label="M2", uid="M222"),
                )
            ],
        )

        mock_neo4j_formatter.return_value = mock_formatter

        mock_result = [Mock(), Mock()]

        result = format_result(mock_result)

        self.assertIsInstance(result, DataContainer)
        self.assertEqual(len(result.records), 2)

        for key, record in result.records.items():
            self.assertIsInstance(key, int)
            self.assertIsInstance(record, GraphRecord)
            self.assertEqual(len(record.nodes), 2)  # Now expecting 2 nodes
            self.assertEqual(len(record.relationships), 1)

            self.assertEqual(record.nodes[0].node_label, "M1")
            self.assertEqual(record.nodes[0].uid, "M111")

            self.assertEqual(record.nodes[1].node_label, "M2")
            self.assertEqual(record.nodes[1].uid, "M222")

            self.assertEqual(record.relationships[0].relationship_type, "R1")
            self.assertEqual(record.relationships[0].start_node.uid, "M111")
            self.assertEqual(record.relationships[0].end_node.uid, "M222")
