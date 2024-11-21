import unittest
from unittest.mock import Mock, patch
import pandas as pd

import noctis.data_transformation.preprocessing.graph_expander
from noctis.data_transformation.preprocessing.graph_expander import GraphExpander


class TestGraphExpander(unittest.TestCase):
    def setUp(self):
        self.sample_schema = {
            'nodes': {
                'base': {'chemical_equation': 'CE'},
                'extra': {'node_1': 'LABEL_1', 'node_2': 'LABEL_2'},
            },
            'relationships': {
                'base': {},
                'extra': {'rel1':{'type': 'REL1', 'start_node':'node_1', 'end_node':'node_2'},
                          'rel2': {'type': 'REL2', 'start_node':'node_2', 'end_node':'node_1'}},
            }
        }
        self.graph_expander = GraphExpander(self.sample_schema)

    def test_init(self):
        self.assertEqual(self.graph_expander.schema, self.sample_schema)
        self.assertEqual(self.graph_expander.nodes, {})
        self.assertEqual(self.graph_expander.relationships, {})

    @patch.object(GraphExpander, '_expand_extra_nodes')
    @patch.object(GraphExpander, '_expand_extra_relationships')
    @patch.object(noctis.data_transformation.preprocessing.graph_expander.ReactionPreProcessor, 'build_from_string_w_validation')
    def test_expand_from_csv_with_validation(self, mock_build, mock_expand_relationships, mock_expand_nodes):
        mock_build.return_value = ({'node1': {'attr': 'val'}}, {'rel1': {'type': 'REL'}})
        step_dict = {'CE': 'A + B -> C', 'extra_data': {'prop': 'value'}}

        nodes, relationships = self.graph_expander.expand_from_csv(step_dict, 'smiles', 'inchi', True)

        mock_build.assert_called_once_with('A + B -> C', 'smiles', 'inchi')
        mock_expand_nodes.assert_called_once_with(step_dict)
        mock_expand_relationships.assert_called_once()
        self.assertIn('node1', nodes)
        self.assertIn('rel1', relationships)

    @patch.object(GraphExpander, '_expand_extra_nodes')
    @patch.object(GraphExpander, '_expand_extra_relationships')
    @patch.object(noctis.data_transformation.preprocessing.graph_expander.ReactionPreProcessor, 'build_from_string')
    def test_expand_from_csv_without_validation(self, mock_build, mock_expand_relationships, mock_expand_nodes):
        mock_build.return_value = ({'node1': {'attr': 'val'}}, {'rel1': {'type': 'REL'}})
        step_dict = {'CE': 'A + B -> C', 'extra_data': {'prop': 'value'}}

        nodes, relationships = self.graph_expander.expand_from_csv(step_dict, 'smiles', 'inchi', False)

        mock_build.assert_called_once_with('A + B -> C', 'smiles')  # Update this line
        mock_expand_nodes.assert_called_once_with(step_dict)
        mock_expand_relationships.assert_called_once()
        self.assertIn('node1', nodes)
        self.assertIn('rel1', relationships)

    def test_expand_extra_nodes(self):
        # Setup
        step_dict = {
            'CE': ['A + B -> C'],
            'LABEL_1': {'attr': 'value'},
            'LABEL_2': {'attr': 'value2'}
        }

        # Execute
        self.graph_expander._expand_extra_nodes(step_dict)

        # Assert
        expected_nodes = {
            'node_1': [{'attr': 'value', 'label': 'LABEL_1'}],
            'node_2': [{'attr': 'value2', 'label': 'LABEL_2'}]}
        self.assertEqual(self.graph_expander.nodes, expected_nodes)

    def test_expand_extra_relationships(self):
        # Setup
        self.graph_expander.nodes = {
            'node_1': [{'attr': 'value1', 'label': 'LABEL_1'}, {'attr': 'value2', 'label': 'LABEL_1'}],
            'node_2': [{'attr': 'value3', 'label': 'LABEL_2'}]
        }

        # Execute
        self.graph_expander._expand_extra_relationships()

        # Assert
        expected_relationships = {'rel1': [{'end_node': {'attr': 'value3', 'label': 'LABEL_2'},
           'start_node': {'attr': 'value1', 'label': 'LABEL_1'},
           'type': 'REL1'},
          {'end_node': {'attr': 'value3', 'label': 'LABEL_2'},
           'start_node': {'attr': 'value2', 'label': 'LABEL_1'},
           'type': 'REL1'}],
 'rel2': [{'end_node': {'attr': 'value1', 'label': 'LABEL_1'},
           'start_node': {'attr': 'value3', 'label': 'LABEL_2'},
           'type': 'REL2'},
          {'end_node': {'attr': 'value2', 'label': 'LABEL_1'},
           'start_node': {'attr': 'value3', 'label': 'LABEL_2'},
           'type': 'REL2'}]}

        self.assertEqual(self.graph_expander.relationships, expected_relationships)
