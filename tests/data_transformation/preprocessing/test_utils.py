import unittest
import pandas as pd
import os
import tempfile
from unittest.mock import patch
from noctis.data_transformation.preprocessing.utils import (
    _update_partition_dict_with_row,
    _build_dataframes_from_dict,
    _save_dataframes_to_partition_csv,
    explode_smiles_like_reaction_string,
    explode_v3000_reaction_string,
)


class TestDataProcessingFunctions(unittest.TestCase):
    def test_update_partition_dict_with_row(self):
        target_dict = {"A": [1, 2], "B": [3, 4]}
        source_dict = {"A": [5], "C": [6]}

        _update_partition_dict_with_row(target_dict, source_dict)

        self.assertEqual(target_dict, {"A": [1, 2, 5], "B": [3, 4], "C": [6]})

    def test_build_dataframes_from_dict(self):
        input_dict = {
            "df1": {"col1": [1, 2], "col2": ["a", "b"]},
            "df2": {"col3": [3, 4], "col4": ["c", "d"]},
        }

        result = _build_dataframes_from_dict(input_dict)

        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result["df1"], pd.DataFrame)
        self.assertIsInstance(result["df2"], pd.DataFrame)
        pd.testing.assert_frame_equal(
            result["df1"], pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        )
        pd.testing.assert_frame_equal(
            result["df2"], pd.DataFrame({"col3": [3, 4], "col4": ["c", "d"]})
        )

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

            _save_dataframes_to_partition_csv(dict_nodes, dict_relationships, tmpdir, 1)

            # Check if directory was created
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "partition_1")))

            # Check if to_csv was called the correct number of times
            self.assertEqual(mock_to_csv.call_count, 4)

            # Check if to_csv was called with the correct filenames
            expected_calls = [
                os.path.join(tmpdir, "partition_1", "NODE1.csv"),
                os.path.join(tmpdir, "partition_1", "NODE2.csv"),
                os.path.join(tmpdir, "partition_1", "REL1.csv"),
                os.path.join(tmpdir, "partition_1", "REL2.csv"),
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


if __name__ == "__main__":
    unittest.main()
