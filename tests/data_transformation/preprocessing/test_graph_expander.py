import unittest
from unittest.mock import Mock, patch, MagicMock

from noctis.data_architecture.graph_schema import GraphSchema
from noctis.data_transformation.preprocessing.graph_expander import GraphExpander
from noctis.data_transformation.preprocessing.core_graph_builder import (
    ValidatedStringBuilder,
    UnvalidatedStringBuilder,
    build_core_graph,
)
from noctis.data_architecture.datamodel import Node, Relationship


class TestGraphExpander(unittest.TestCase):
    def setUp(self):
        self.sample_schema = GraphSchema.build_from_dict(
            {
                "base_nodes": {"chemical_equation": "CE", "molecule": "M"},
                "extra_nodes": {"node_1": "LABEL1", "node_2": "LABEL2"},
                "base_relationships": {
                    "product": {
                        "type": "P",
                        "start_node": "node_1",
                        "end_node": "node_2",
                    },
                    "reactant": {
                        "type": "R",
                        "start_node": "node_1",
                        "end_node": "node_2",
                    },
                },
                "extra_relationships": {
                    "rel1": {
                        "type": "REL1",
                        "start_node": "node_1",
                        "end_node": "node_2",
                    },
                    "rel2": {
                        "type": "REL2",
                        "start_node": "node_2",
                        "end_node": "node_1",
                    },
                },
            }
        )
        self.graph_expander = GraphExpander(self.sample_schema)

    def test_init(self):
        self.assertEqual(self.graph_expander.schema, self.sample_schema)
        self.assertEqual(self.graph_expander.nodes, {})
        self.assertEqual(self.graph_expander.relationships, {})

    @patch(
        "noctis.data_transformation.preprocessing.graph_expander.ValidatedStringBuilder"
    )
    @patch(
        "noctis.data_transformation.preprocessing.graph_expander.build_core_graph",
        side_effect=build_core_graph,
    )
    @patch.object(GraphExpander, "_expand_extra_nodes")
    @patch.object(GraphExpander, "_expand_extra_relationships")
    def test_expand_from_csv_with_validation(
        self,
        mock_expand_relationships,
        mock_expand_nodes,
        mock_build_core_graph,
        MockValidatedStringBuilder,
    ):
        # Arrange
        mock_processor = Mock(spec=ValidatedStringBuilder)

        # Create mock Node and Relationship objects
        mock_ce_node = Mock(spec=Node)
        mock_molecule_node = Mock(spec=Node)
        mock_relationship = Mock(spec=Relationship)

        # Set up the return value for the process method
        mock_processor.process.return_value = (
            {"chemical_equation": [mock_ce_node], "molecule": [mock_molecule_node]},
            {"reactant": [mock_relationship], "product": []},
        )

        MockValidatedStringBuilder.return_value = mock_processor

        step_dict = {
            "CE": {
                "smiles": "C>>O",
                "properties": {"prop1": "value1", "prop2": "value2"},
            },
            "extra_data": {"prop": "value"},
        }

        # Act
        nodes, relationships = self.graph_expander.expand_from_csv(
            step_dict, "smiles", "smarts", True
        )

        # Assert
        MockValidatedStringBuilder.assert_called_once_with(
            input_format="smiles", output_format="smarts"
        )
        mock_processor.process.assert_called_once_with(
            {"smiles": "C>>O", "properties": {"prop1": "value1", "prop2": "value2"}}
        )
        mock_expand_nodes.assert_called_once_with(step_dict)
        mock_expand_relationships.assert_called_once()

        # Check that the returned nodes and relationships match what we expect
        self.assertEqual(nodes["chemical_equation"], [mock_ce_node])
        self.assertEqual(nodes["molecule"], [mock_molecule_node])
        self.assertEqual(relationships["reactant"], [mock_relationship])
        self.assertEqual(relationships["product"], [])

    @patch(
        "noctis.data_transformation.preprocessing.graph_expander.UnvalidatedStringBuilder"
    )
    @patch(
        "noctis.data_transformation.preprocessing.graph_expander.build_core_graph",
        side_effect=build_core_graph,
    )
    @patch.object(GraphExpander, "_expand_extra_nodes")
    @patch.object(GraphExpander, "_expand_extra_relationships")
    def test_expand_from_csv_without_validation(
        self,
        mock_expand_relationships,
        mock_expand_nodes,
        mock_build_core_graph,
        MockUnvalidatedStringBuilder,
    ):
        # Arrange
        mock_processor = Mock(spec=UnvalidatedStringBuilder)

        # Create mock Node and Relationship objects
        mock_ce_node = Mock(spec=Node)
        mock_molecule_node = Mock(spec=Node)
        mock_relationship = Mock(spec=Relationship)

        # Set up the return value for the process method
        mock_processor.process.return_value = (
            {"chemical_equation": [mock_ce_node], "molecule": [mock_molecule_node]},
            {"reactant": [mock_relationship], "product": []},
        )

        MockUnvalidatedStringBuilder.return_value = mock_processor

        step_dict = {
            "CE": {
                "smiles": "C>>O",
                "properties": {"prop1": "value1", "prop2": "value2"},
            },
            "extra_data": {"prop": "value"},
        }

        # Act
        nodes, relationships = self.graph_expander.expand_from_csv(
            step_dict, "smiles", "smarts", False
        )

        # Assert
        MockUnvalidatedStringBuilder.assert_called_once_with(input_format="smiles")
        mock_processor.process.assert_called_once_with(
            {"smiles": "C>>O", "properties": {"prop1": "value1", "prop2": "value2"}}
        )
        mock_expand_nodes.assert_called_once_with(step_dict)
        mock_expand_relationships.assert_called_once()

        # Check that the returned nodes and relationships match what we expect
        self.assertEqual(nodes["chemical_equation"], [mock_ce_node])
        self.assertEqual(nodes["molecule"], [mock_molecule_node])
        self.assertEqual(relationships["reactant"], [mock_relationship])
        self.assertEqual(relationships["product"], [])

    def test_expand_extra_nodes(self):
        # Setup
        self.graph_expander.schema.extra_nodes = {
            "node_1": "LABEL1",
            "node_2": "LABEL2",
        }
        step_dict = {
            "LABEL1": {"uid": "A123", "properties": {"attr": "value"}},
            "LABEL2": {"uid": "B123", "properties": {"attr": "value"}},
        }

        # Execute
        self.graph_expander._expand_extra_nodes(step_dict)

        # Assert
        expected_nodes = {
            "node_1": [
                Node(uid="A123", node_label="LABEL1", properties={"attr": "value"})
            ],
            "node_2": [
                Node(uid="B123", node_label="LABEL2", properties={"attr": "value"})
            ],
        }
        self.assertEqual(self.graph_expander.nodes, expected_nodes)

    def test_expand_extra_relationships(self):
        # Setup
        self.graph_expander.nodes = {
            "node_1": [
                Node(uid="A123", node_label="LABEL1", properties={"attr": "value1"}),
                Node(uid="A124", node_label="LABEL1", properties={"attr": "value2"}),
            ],
            "node_2": [
                Node(uid="B123", node_label="LABEL2", properties={"attr": "value3"})
            ],
        }

        # Execute
        self.graph_expander._expand_extra_relationships()

        # Assert
        expected_relationships = {
            "rel1": [
                Relationship(
                    relationship_type="REL1",
                    start_node=Node(
                        uid="A123", node_label="LABEL1", properties={"attr": "value1"}
                    ),
                    end_node=Node(
                        uid="B123", node_label="LABEL2", properties={"attr": "value3"}
                    ),
                ),
                Relationship(
                    relationship_type="REL1",
                    start_node=Node(
                        uid="A124", node_label="LABEL1", properties={"attr": "value2"}
                    ),
                    end_node=Node(
                        uid="B123", node_label="LABEL2", properties={"attr": "value3"}
                    ),
                ),
            ],
            "rel2": [
                Relationship(
                    relationship_type="REL2",
                    start_node=Node(
                        uid="B123", node_label="LABEL2", properties={"attr": "value3"}
                    ),
                    end_node=Node(
                        uid="A123", node_label="LABEL1", properties={"attr": "value1"}
                    ),
                ),
                Relationship(
                    relationship_type="REL2",
                    start_node=Node(
                        uid="B123", node_label="LABEL2", properties={"attr": "value3"}
                    ),
                    end_node=Node(
                        uid="A124", node_label="LABEL1", properties={"attr": "value2"}
                    ),
                ),
            ],
        }

        self.assertEqual(self.graph_expander.relationships, expected_relationships)
