import unittest
from unittest.mock import Mock, patch, call
import pandas as pd

import noctis.data_transformation.preprocessing.data_preprocessing
from noctis.data_transformation.preprocessing.data_preprocessing import (
    Preprocessor,
    FilePreprocessorConfig,
    FilePreprocessor,
)

# from noctis.data_transformation.preprocessing.graph_expander import GraphExpander, ReactionPreProcessor


class TestPreprocessor(unittest.TestCase):
    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.FilePreprocessor"
    )
    def test_preprocess_csv_for_neo4j(self, mock_csv_preprocessor):
        schema = {"some": "schema"}
        config = FilePreprocessorConfig(
            input_file="test.csv",
            output_folder="/output",
            tmp_folder="/tmp",
            validation=True,
            parallel=False,
            prefix="test_",
            blocksize=64,
            chunksize=1000,
            inp_chem_format="smiles",
            out_chem_format="smiles",
        )
        preprocessor = Preprocessor(schema)

        # Create a mock instance for FilePreprocessor
        mock_csv_instance = Mock()
        mock_csv_preprocessor.return_value = mock_csv_instance

        preprocessor.preprocess_csv_for_neo4j(config)

        mock_csv_preprocessor.assert_called_once_with(schema, config)
        mock_csv_instance.run.assert_called_once()

    def test_preprocessor_initialization(self):
        schema = {"some": "schema"}

        preprocessor = Preprocessor(schema)

        self.assertEqual(preprocessor.schema, schema)


class TestFilePreprocessor(unittest.TestCase):
    def setUp(self):
        self.schema = {"nodes": {"tag": "label"}, "relationships": {"tag": "type"}}
        self.config = FilePreprocessorConfig(
            input_file="test.csv",
            output_folder="/output",
            tmp_folder="/tmp",
            validation=True,
            parallel=False,
            prefix="test_",
            blocksize=64,
            chunksize=1000,
            inp_chem_format="smiles",
            out_chem_format="smiles",
        )
        self.preprocessor = FilePreprocessor(self.schema, self.config)

    def test_init(self):
        self.assertEqual(self.preprocessor.schema, self.schema)
        self.assertEqual(self.preprocessor.config, self.config)

    @patch.object(FilePreprocessor, "_run_parallel")
    @patch.object(FilePreprocessor, "_run_serial")
    def test_run_parallel(self, mock_run_serial, mock_run_parallel):
        self.config.parallel = True
        self.preprocessor.run()
        mock_run_parallel.assert_called_once()
        mock_run_serial.assert_not_called()

    @patch.object(FilePreprocessor, "_run_parallel")
    @patch.object(FilePreprocessor, "_run_serial")
    def test_run_serial(self, mock_run_serial, mock_run_parallel):
        self.config.parallel = False
        self.preprocessor.run()
        mock_run_serial.assert_called_once()
        mock_run_parallel.assert_not_called()

    @patch("noctis.data_transformation.preprocessing.data_preprocessing.Client")
    @patch("noctis.data_transformation.preprocessing.data_preprocessing.dd.read_csv")
    def test_run_parallel_execution(self, mock_read_csv, mock_client):
        mock_ddf = Mock()
        mock_read_csv.return_value = mock_ddf

        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance

        self.config.parallel = True
        self.preprocessor._run_parallel()

        mock_client.assert_called_once()
        mock_read_csv.assert_called_once_with(
            self.config.input_file, blocksize=self.config.blocksize
        )
        mock_ddf.map_partitions.assert_called_once()
        mock_ddf.map_partitions.return_value.compute.assert_called_once()

    @patch("pandas.read_csv")
    @patch.object(FilePreprocessor, "_process_partition")
    def test_run_serial_execution(self, mock_process_partition, mock_read_csv):
        mock_df = Mock()
        mock_read_csv.return_value = [mock_df, mock_df]  # Simulate two partitions

        self.preprocessor._run_serial()

        self.assertEqual(mock_process_partition.call_count, 2)
        mock_process_partition.assert_has_calls([call(mock_df, 0), call(mock_df, 1)])

    @patch("pandas.read_csv")
    def test_serial_partition_generator_without_chunksize(self, mock_read_csv):
        mock_df = Mock()
        mock_read_csv.return_value = mock_df
        self.config.chunksize = None

        result = list(self.preprocessor._serial_partition_generator())

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], mock_df)
        mock_read_csv.assert_called_once_with(self.config.input_file)

    @patch("pandas.read_csv")
    def test_serial_partition_generator_with_chunksize(self, mock_read_csv):
        mock_df1, mock_df2 = Mock(), Mock()
        mock_read_csv.return_value = [mock_df1, mock_df2]
        self.config.chunksize = 1000

        result = list(self.preprocessor._serial_partition_generator())

        self.assertEqual(len(result), 2)
        self.assertEqual(result, [mock_df1, mock_df2])
        mock_read_csv.assert_called_once_with(
            self.config.input_file, chunksize=self.config.chunksize
        )

    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing._update_partition_dict_with_row"
    )
    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing._build_dataframes_from_dict"
    )
    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing._save_dataframes_to_partition_csv"
    )
    def test_process_partition(self, mock_save, mock_build, mock_update):
        mock_partition_data = pd.DataFrame(
            {"header1": ["value1"], "header2": ["value2"]}
        )
        mock_build.side_effect = [
            Mock(),
            Mock(),
        ]  # One for nodes, one for relationships

        # Mock the _process_row method
        mock_nodes = {"node1": {"attr1": "value1"}}
        mock_relationships = {"rel1": {"attr2": "value2"}}
        mock_process_row = Mock(return_value=(mock_nodes, mock_relationships))

        with patch.object(self.preprocessor, "_process_row", mock_process_row):
            self.preprocessor._process_partition(mock_partition_data, 0)

        # Check if _process_row was called for each row in the DataFrame
        self.assertEqual(mock_process_row.call_count, len(mock_partition_data))

        # Check if _update_partition_dict_with_row was called with the mock data
        expected_calls = [call({}, mock_nodes), call({}, mock_relationships)]
        mock_update.assert_has_calls(expected_calls, any_order=True)

        self.assertEqual(
            mock_update.call_count, 2
        )  # Called for both nodes and relationships
        self.assertEqual(
            mock_build.call_count, 2
        )  # Called for both nodes and relationships
        mock_save.assert_called_once()

    @patch("noctis.data_transformation.preprocessing.data_preprocessing.GraphExpander")
    def test_process_row(self, MockGraphExpander):
        # Prepare test data
        print(
            "hey",
            noctis.data_transformation.preprocessing.data_preprocessing.GraphExpander,
        )
        row = pd.Series({"header1": "value1", "header2": "value2"})

        # Mock the _split_row_by_node_types method
        self.preprocessor._split_row_by_node_types = Mock(return_value={"split": "row"})

        # Set up the mock for GraphExpander
        mock_expander_instance = MockGraphExpander.return_value
        mock_expander_instance.expand_from_csv = Mock(
            return_value=({"node1": {"attr1": "val1"}}, {"rel1": {"attr2": "val2"}})
        )

        # Call the method
        result_nodes, result_relationships = self.preprocessor._process_row(row)

        # Assert the mocks were called correctly
        self.preprocessor._split_row_by_node_types.assert_called_once_with(row)
        MockGraphExpander.assert_called_once_with(self.schema)
        mock_expander_instance.expand_from_csv.assert_called_once_with(
            {"split": "row"},
            self.config.inp_chem_format,
            self.config.out_chem_format,
            self.config.validation,
        )

        # Assert the results
        self.assertEqual(result_nodes, {"node1": {"attr1": "val1"}})
        self.assertEqual(result_relationships, {"rel1": {"attr2": "val2"}})

    def test_split_row_by_node_types(self):
        # Create a sample pandas Series
        data = {
            "Person.name": "John Doe",
            "Person.age": "30",
            "Address.street": "Main St",
            "Address.city": "New York",
            "Job.title": "Engineer",
            "InvalidColumn": "This should be ignored",
        }
        row = pd.Series(data)

        # Expected output
        expected_output = {
            "Person": {"name": "John Doe", "age": "30"},
            "Address": {"street": "Main St", "city": "New York"},
            "Job": {"title": "Engineer"},
        }

        # Call the function
        result = self.preprocessor._split_row_by_node_types(row)

        # Assert the result matches the expected output
        self.assertEqual(result, expected_output)

    def test_empty_row(self):
        # Test with an empty Series
        row = pd.Series({})
        result = self.preprocessor._split_row_by_node_types(row)
        self.assertEqual(result, {})

    def test_no_valid_columns(self):
        # Test with a Series containing no valid columns
        data = {"Column1": "Value1", "Column2": "Value2"}
        row = pd.Series(data)
        result = self.preprocessor._split_row_by_node_types(row)
        self.assertEqual(result, {})
