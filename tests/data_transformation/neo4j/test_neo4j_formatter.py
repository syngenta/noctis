import unittest
from unittest.mock import Mock, patch
from neo4j._sync.work.result import Result

from noctis.data_transformation.neo4j.neo4j_formatter import (
    Neo4jResultFormatter,
    select_formatter,
    format_result,
    Node,
    Relationship,
    GraphRecord,
    DataContainer,
)


class TestNeo4jResultFormatter(unittest.TestCase):
    def setUp(self):
        self.formatter = Neo4jResultFormatter()

    def test_format_node(self):
        mock_node = Mock()
        mock_node.labels = ["M1"]
        mock_node.get.return_value = "M123"
        mock_node.items.return_value = [("uid", "M123"), ("prop1", "value1")]

        result = self.formatter._format_node(mock_node)

        self.assertIsInstance(result, Node)
        self.assertEqual(result.node_label, "M1")
        self.assertEqual(result.uid, "M123")
        self.assertEqual(result.properties, {"prop1": "value1"})

    def test_format_nodes(self):
        mock_record = {"nodes": [Mock(), Mock()]}
        with patch.object(Neo4jResultFormatter, "_format_node") as mock_format_node:
            mock_format_node.side_effect = [
                Node(node_label="M1", uid="M111"),
                Node(node_label="M2", uid="M222"),
            ]
            result = self.formatter.format_nodes(mock_record)

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Node)
        self.assertIsInstance(result[1], Node)

    def test_format_relationship(self):
        mock_rel = Mock()
        mock_rel.type = "REL"
        mock_rel.nodes = [Mock(), Mock()]

        with patch.object(Neo4jResultFormatter, "_format_node") as mock_format_node:
            mock_format_node.side_effect = [
                Node(node_label="M1", uid="M111"),
                Node(node_label="M2", uid="M222"),
            ]
            result = self.formatter._format_relationship(mock_rel)

        self.assertIsInstance(result, Relationship)
        self.assertEqual(result.relationship_type, "REL")
        self.assertIsInstance(result.start_node, Node)
        self.assertIsInstance(result.end_node, Node)

    def test_format_relationships(self):
        mock_record = {"relationships": [Mock(), Mock()]}
        with patch.object(
            Neo4jResultFormatter, "_format_relationship"
        ) as mock_format_rel:
            mock_format_rel.side_effect = [
                Relationship(
                    relationship_type="R1",
                    start_node=Node(node_label="M1", uid="M111"),
                    end_node=Node(node_label="M2", uid="M222"),
                ),
                Relationship(
                    relationship_type="R2",
                    start_node=Node(node_label="M2", uid="M222"),
                    end_node=Node(node_label="M1", uid="M111"),
                ),
            ]
            result = self.formatter.format_relationships(mock_record)

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], Relationship)
        self.assertIsInstance(result[1], Relationship)

    def test_format_relationships_empty(self):
        mock_record = {}
        result = self.formatter.format_relationships(mock_record)
        self.assertEqual(result, [])


class TestSelectFormatter(unittest.TestCase):
    def test_select_formatter_with_neo4j_result(self):
        result = Mock(spec=Result)
        formatter = select_formatter(result)
        self.assertIsInstance(formatter, Neo4jResultFormatter)

    def test_select_formatter_with_unsupported_result(self):
        result = object()  # Creating an unsupported result object
        with self.assertRaises(ValueError):
            select_formatter(result)


class TestFormatResult(unittest.TestCase):
    @patch("noctis.data_transformation.neo4j.neo4j_formatter.select_formatter")
    def test_format_result(self, mock_select_formatter):
        # Create a mock formatter
        mock_formatter = Mock()

        # Set up the mock formatter to return specific nodes and relationships
        mock_formatter.format_nodes.return_value = [Node(node_label="M1", uid="M111")]
        mock_formatter.format_relationships.return_value = [
            Relationship(
                relationship_type="R1",
                start_node=Node(node_label="M1", uid="M111"),
                end_node=Node(node_label="M2", uid="M222"),
            )
        ]

        mock_select_formatter.return_value = mock_formatter

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
