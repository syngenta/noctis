import unittest
import pandas as pd
import os
import tempfile
from unittest.mock import patch

import pytest
from noctis import settings
from noctis.data_transformation.preprocessing.utilities import (
    _update_partition_dict_with_row,
    _save_dataframes_to_partition_csv,
    explode_smiles_like_reaction_string,
    create_noctis_node,
    create_noctis_relationship,
)
from pydantic import ValidationError
from noctis.data_architecture.datamodel import Node, Relationship
from noctis.data_architecture.graph_schema import GraphSchema


class TestDataProcessingFunctions(unittest.TestCase):
    def test_update_partition_dict_with_row(self):
        target_dict = {"A": [1, 2], "B": [3, 4]}
        source_dict = {"A": [5], "C": [6]}

        _update_partition_dict_with_row(target_dict, source_dict)

        self.assertEqual(target_dict, {"A": [1, 2, 5], "B": [3, 4], "C": [6]})

    @patch("pandas.DataFrame.to_csv")
    def test_save_dataframes_to_partition_csv(self, mock_to_csv):
        with tempfile.TemporaryDirectory() as tmpdir:
            dict_nodes = {
                "node1": pd.DataFrame({"col1": [1, 2]}),
                "node2": pd.DataFrame({"col2": [3, 4]}),
            }
            dict_relationships = {
                "rel1": pd.DataFrame({"col3": [5, 6]}),
                "rel2": pd.DataFrame({"col4": [7, 8]}),
            }

            gs = GraphSchema()
            gs.extra_nodes = {"node1": "N1", "node2": "N2"}
            gs.extra_relationships = {"rel1": {"type": "R1"}, "rel2": {"type": "R2"}}
            _save_dataframes_to_partition_csv(
                dict_nodes, dict_relationships, gs, tmpdir, 1
            )

            # Check if directory was created
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "partition_1")))

            # Check if to_csv was called the correct number of times
            self.assertEqual(mock_to_csv.call_count, 4)

            # Check if to_csv was called with the correct filenames
            expected_calls = [
                os.path.join(tmpdir, "partition_1", "N1.csv"),
                os.path.join(tmpdir, "partition_1", "N2.csv"),
                os.path.join(tmpdir, "partition_1", "R1.csv"),
                os.path.join(tmpdir, "partition_1", "R2.csv"),
            ]
            actual_calls = [call[0][0] for call in mock_to_csv.call_args_list]
            self.assertEqual(sorted(actual_calls), sorted(expected_calls))


def test_explode_reaction_smiles():
    reactions_smiles = "CNOC(O)=O.Cl>>CNCl"
    reactants, products = explode_smiles_like_reaction_string(reactions_smiles)
    assert sorted(reactants) == sorted(["CNOC(O)=O", "Cl"])
    assert sorted(products) == sorted(["CNCl"])

    general_string = "reactant1.reactant2>reagent1>product1.product2"
    reactants, products = explode_smiles_like_reaction_string(general_string)
    assert sorted(reactants) == sorted(["reactant1", "reactant2"])
    assert sorted(products) == sorted(["product1", "product2"])


def test_create_noctis_node():
    node1 = create_noctis_node(node_uid="AB123", node_label="MyNode", properties={})
    assert isinstance(node1, Node)

    with pytest.raises(ValidationError):
        Node(uid=122, node_label="Node1", properties={})


def test_create_noctis_relationship():
    # Test valid relationship creation
    mol_node = Node(node_label="StartNode", uid="AB123")
    ce_node = Node(node_label="EndNode", uid="CD456")

    relationship = create_noctis_relationship(
        mol_node=mol_node, ce_node=ce_node, role="reactants"
    )
    assert isinstance(relationship, Relationship)
    assert relationship.start_node == mol_node
    assert (
        relationship.relationship_type == settings.relationships.relationship_reactant
    )
    relationship = create_noctis_relationship(
        mol_node=mol_node, ce_node=ce_node, role="products"
    )
    assert isinstance(relationship, Relationship)
    assert relationship.start_node == ce_node
    assert relationship.relationship_type == settings.relationships.relationship_product


if __name__ == "__main__":
    unittest.main()
