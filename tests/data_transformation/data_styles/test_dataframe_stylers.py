import unittest
import pandas as pd

from noctis.data_transformation.data_styles.dataframe_stylers import (
    NodesRelationshipsStyle,
)
from noctis.data_architecture.datamodel import Node, Relationship


class TestNeo4jStyle(unittest.TestCase):
    def setUp(self):
        self.sample_nodes = [
            Node(uid="M1", node_label="M", properties={"p1": "1", "p2": "2"}),
            Node(uid="M2", node_label="CE", properties={"p1": "1", "p2": "2"}),
        ]
        self.sample_relationships = [
            Relationship(
                start_node=self.sample_nodes[0],
                end_node=self.sample_nodes[1],
                relationship_type="P",
                properties={},
            ),
            Relationship(
                start_node=self.sample_nodes[1],
                end_node=self.sample_nodes[0],
                relationship_type="R",
                properties={},
            ),
        ]

    def test_export_nodes(self):
        result = NodesRelationshipsStyle.export_nodes(
            {"M": [self.sample_nodes[0]], "CE": [self.sample_nodes[1]]}
        )
        self.assertIn("M", result)
        self.assertIn("CE", result)
        self.assertIsInstance(result["M"], pd.DataFrame)
        self.assertIsInstance(result["CE"], pd.DataFrame)
        self.assertEqual(len(result["M"]), 1)
        self.assertEqual(len(result["CE"]), 1)

    def test_export_relationships(self):
        result = NodesRelationshipsStyle.export_relationships(
            {"P": [self.sample_relationships[0]], "R": [self.sample_relationships[1]]}
        )
        self.assertIn("P", result)
        self.assertIn("R", result)
        self.assertIsInstance(result["P"], pd.DataFrame)
        self.assertIsInstance(result["R"], pd.DataFrame)
        self.assertEqual(len(result["P"]), 1)
        self.assertEqual(len(result["R"]), 1)

    def test_process_nodes_dataframe(self):
        result = NodesRelationshipsStyle._process_nodes_dataframe(self.sample_nodes)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("uid", result.columns)
        self.assertIn("node_label", result.columns)
        self.assertIn("p1", result.columns)
        self.assertIn("p2", result.columns)

    def test_process_relationships_dataframe(self):
        result = NodesRelationshipsStyle._process_relationships_dataframe(
            self.sample_relationships
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("start_node", result.columns)
        self.assertIn("end_node", result.columns)
        self.assertIn("relationship_type", result.columns)

    def test_reorder_columns(self):
        df = pd.DataFrame({"c": [1, 2], "a": [3, 4], "b": [5, 6]})
        result = NodesRelationshipsStyle._reorder_columns(df, ["a"], ["c"])
        self.assertEqual(list(result.columns), ["a", "b", "c"])

    def test_build_nodes_dataframe(self):
        result = NodesRelationshipsStyle._build_nodes_dataframe(self.sample_nodes)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("uid", result.columns)
        self.assertIn("node_label", result.columns)
        self.assertIn("p1", result.columns)
        self.assertIn("p2", result.columns)

    def test_build_relationships_dataframe(self):
        result = NodesRelationshipsStyle._build_relationships_dataframe(
            self.sample_relationships
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("start_node", result.columns)
        self.assertIn("end_node", result.columns)
        self.assertIn("relationship_type", result.columns)

    def test_expand_properties_in_dataframe(self):
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "properties": [{"p1": "1", "p2": "2"}, {"p1": "3", "p2": "4"}],
            }
        )
        result = NodesRelationshipsStyle._expand_properties_in_dataframe(df)
        self.assertIn("id", result.columns)
        self.assertIn("p1", result.columns)
        self.assertIn("p2", result.columns)
        self.assertNotIn("properties", result.columns)

    def test_rename_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        rename_dict = {"a": "x", "b": "y"}
        result = NodesRelationshipsStyle._rename_columns(df, rename_dict)
        self.assertEqual(list(result.columns), ["x", "y"])

    def test_expand_properties_flag(self):
        class TestStyle(NodesRelationshipsStyle):
            EXPAND_PROPERTIES = False

        result = TestStyle._process_nodes_dataframe(self.sample_nodes)
        self.assertIn("properties", result.columns)
        self.assertNotIn("p1", result.columns)
        self.assertNotIn("p2", result.columns)

    def test_column_names_override(self):
        class TestStyle(NodesRelationshipsStyle):
            COLUMN_NAMES_NODES = {"uid": "ID", "node_label": "Label"}

        result = TestStyle._process_nodes_dataframe(self.sample_nodes)
        self.assertIn("ID", result.columns)
        self.assertIn("Label", result.columns)
        self.assertNotIn("uid", result.columns)
        self.assertNotIn("node_label", result.columns)
