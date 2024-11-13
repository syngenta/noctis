import unittest
from unittest.mock import Mock, patch, call
import pandas as pd
from noctis.data_transformation.preprocessing.data_preprocessing import (
    Preprocessor,
    FilePreprocessorConfig,
    CSVPreprocessor,
)


class TestPreprocessor(unittest.TestCase):
    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.CSVPreprocessor"
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
        )
        preprocessor = Preprocessor(schema)

        # Create a mock instance for CSVPreprocessor
        mock_csv_instance = Mock()
        mock_csv_preprocessor.return_value = mock_csv_instance

        preprocessor.preprocess_csv_for_neo4j(config)

        mock_csv_preprocessor.assert_called_once_with(schema, config)
        mock_csv_instance.run.assert_called_once()

    def test_preprocessor_initialization(self):
        schema = {"some": "schema"}

        preprocessor = Preprocessor(schema)

        self.assertEqual(preprocessor.schema, schema)


class TestCSVPreprocessor(unittest.TestCase):
    def setUp(self):
        self.schema = {"some": "schema"}
        self.config = FilePreprocessorConfig(
            input_file="test.csv",
            output_folder="/output",
            tmp_folder="/tmp",
            validation=True,
            parallel=False,
            prefix="test_",
            blocksize=64,
            chunksize=1000,
        )
        self.preprocessor = CSVPreprocessor(self.schema, self.config)

    def test_init(self):
        self.assertEqual(self.preprocessor.schema, self.schema)
        self.assertEqual(self.preprocessor.config, self.config)

    @patch.object(CSVPreprocessor, "_run_parallel")
    @patch.object(CSVPreprocessor, "_run_serial")
    def test_run_parallel(self, mock_run_serial, mock_run_parallel):
        self.config.parallel = True
        self.preprocessor.run()
        mock_run_parallel.assert_called_once()
        mock_run_serial.assert_not_called()

    @patch.object(CSVPreprocessor, "_run_parallel")
    @patch.object(CSVPreprocessor, "_run_serial")
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
    @patch.object(CSVPreprocessor, "_process_partition")
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
        # failed
        mock_partition_data = pd.DataFrame(
            {"header1": ["value1"], "header2": ["value2"]}
        )
        mock_build.side_effect = [
            Mock(),
            Mock(),
        ]  # One for nodes, one for relationships

        self.preprocessor._process_partition(mock_partition_data, 0)

        self.assertEqual(
            mock_update.call_count, 2
        )  # Called for both nodes and relationships
        self.assertEqual(
            mock_build.call_count, 2
        )  # Called for both nodes and relationships
        mock_save.assert_called_once()

    def test_process_row(self):
        row = pd.Series({"header1": "value1", "header2": "value2"})
        nodes, relationships = self.preprocessor._process_row(row)

        self.assertEqual(nodes, {"node_label": [{"h1": "value1"}]})
        self.assertEqual(relationships, {"relationship_type": [{"h1": "value2"}]})
