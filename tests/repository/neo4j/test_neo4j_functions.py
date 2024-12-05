from unittest.mock import Mock, patch, mock_open
import unittest
import pandas as pd
import warnings

from noctis.repository.neo4j.neo4j_functions import (
    _generate_nodes_files_string,
    _generate_properties_assignment,
    _generate_relationships_files_string,
    _convert_datacontainer_to_query,
    _convert_record_to_query_neo4j,
    _create_node_queries,
    _create_relationship_queries,
    _get_dict_keys_from_csv,
)
from noctis.data_architecture.datamodel import (
    Node,
    Relationship,
    DataContainer,
    GraphRecord,
)


class TestNeo4jFunctions(unittest.TestCase):
    def setUp(self):
        self.sample_nodes = [
            Node(
                uid="M1",
                node_label="M",
                properties={"p1": "1", "p2": "2", "smiles": "C1=CC=CC=C1"},
            ),
            Node(
                uid="M2",
                node_label="CE",
                properties={"p1": "1", "p2": "2", "smiles": "C1=CC=CC=C1"},
            ),
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
        self.sample_record = GraphRecord(
            nodes=self.sample_nodes, relationships=self.sample_relationships
        )
        self.sample_data_container = DataContainer(
            records={0: self.sample_record, 1: self.sample_record}
        )

    def test_create_node_queries(self):
        result = _create_node_queries(self.sample_nodes)

        expected = [
            'MERGE (:M {uid: "M1",smiles: "C1=CC=CC=C1"})\n',
            'MERGE (:CE {uid: "M2",smiles: "C1=CC=CC=C1"})\n',
        ]
        self.assertEqual(result, expected)

    def test_create_relationship_queries(self):
        result = _create_relationship_queries(self.sample_relationships)

        expected = [
            'MATCH (sn:M {uid: "M1"})\nMATCH (en:CE {uid: "M2"})\nMERGE (sn)-[:P]->(en)\n',
            'MATCH (sn:CE {uid: "M2"})\nMATCH (en:M {uid: "M1"})\nMERGE (sn)-[:R]->(en)\n',
        ]
        self.assertEqual(result, expected)

    def test_convert_datacontainer_to_query(self):
        result = _convert_datacontainer_to_query(self.sample_data_container)

        expected = [
            'MERGE (:M {uid: "M1",smiles: "C1=CC=CC=C1"})\n',
            'MERGE (:CE {uid: "M2",smiles: "C1=CC=CC=C1"})\n',
            'MATCH (sn:M {uid: "M1"})\nMATCH (en:CE {uid: "M2"})\nMERGE (sn)-[:P]->(en)\n',
            'MATCH (sn:CE {uid: "M2"})\nMATCH (en:M {uid: "M1"})\nMERGE (sn)-[:R]->(en)\n',
            'MERGE (:M {uid: "M1",smiles: "C1=CC=CC=C1"})\n',
            'MERGE (:CE {uid: "M2",smiles: "C1=CC=CC=C1"})\n',
            'MATCH (sn:M {uid: "M1"})\nMATCH (en:CE {uid: "M2"})\nMERGE (sn)-[:P]->(en)\n',
            'MATCH (sn:CE {uid: "M2"})\nMATCH (en:M {uid: "M1"})\nMERGE (sn)-[:R]->(en)\n',
        ]
        self.assertEqual(set(result), set(expected))

    def test_convert_record_to_query_neo4j(self):
        result = _convert_record_to_query_neo4j(self.sample_record)

        expected = [
            'MERGE (:M {uid: "M1",smiles: "C1=CC=CC=C1"})\n',
            'MERGE (:CE {uid: "M2",smiles: "C1=CC=CC=C1"})\n',
            'MATCH (sn:M {uid: "M1"})\nMATCH (en:CE {uid: "M2"})\nMERGE (sn)-[:P]->(en)\n',
            'MATCH (sn:CE {uid: "M2"})\nMATCH (en:M {uid: "M1"})\nMERGE (sn)-[:R]->(en)\n',
        ]
        self.assertEqual(set(result), set(expected))

    def test_generate_properties_assignment(self):
        properties = ["prop1", "prop2", "prop3"]
        result = _generate_properties_assignment(properties)
        expected = "prop1: apoc.convert.fromJsonMap(row.properties).prop1, prop2: apoc.convert.fromJsonMap(row.properties).prop2, prop3: apoc.convert.fromJsonMap(row.properties).prop3"
        self.assertEqual(result, expected)

    @patch("pandas.read_csv")
    def test_get_dict_keys_from_csv(self, mock_read_csv):
        # Arrange
        mock_read_csv.return_value = pd.DataFrame(
            {
                "column1": [1, 3, 5],
                "column2": [2, 4, 6],
                "properties": [
                    {"property1": "value1"},
                    {"property2": "value2"},
                    {"property1": "value3", "property2": "value4"},
                ],
            }
        )

        # Act
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _get_dict_keys_from_csv("test.csv")

        # Assert
        self.assertEqual(set(result), {"property1", "property2"})
        self.assertEqual(len(w), 1)
        self.assertIsInstance(w[0].message, UserWarning)
        self.assertEqual(
            str(w[0].message), "Some dictionaries are missing keys: {'property2'}"
        )

        # Verify that open and read_csv were called correctly
        mock_read_csv.assert_called_once_with("test.csv")

    def test_generate_nodes_files_string(self):
        prefix = "prefix"
        labels = ["Label1", "Label2"]
        result = _generate_nodes_files_string(prefix, labels)
        expected = "{fileName:'file:/prefix_LABEL1.csv', labels:[]}, {fileName:'file:/prefix_LABEL2.csv', labels:[]}"
        self.assertEqual(result, expected)

    def test_generate_relationships_files_string(self):
        prefix = "prefix"
        types = ["Type1", "Type2"]
        result = _generate_relationships_files_string(prefix, types)
        expected = "{fileName:'file:/prefix_TYPE1.csv', types:[]}, {fileName:'file:/prefix_TYPE2.csv', types:[]}"
        self.assertEqual(result, expected)
