import unittest
from unittest.mock import Mock, MagicMock, patch, call, mock_open
import pandas as pd
import numpy as np

import noctis.data_transformation.preprocessing.data_preprocessing
from noctis.data_transformation.preprocessing.data_preprocessing import (
    Preprocessor,
    PreprocessorConfig,
    CSVPreprocessor,
    PandasRowPreprocessorBase,
    ChemicalStringPreprocessorBase,
    ReactionStringsPreprocessor,
    SynGraphPreprocessor,
    PythonObjectPreprocessorFactory,
    PythonObjectPreprocessorInterface,
    DataFramePreprocessor,
    NoPreprocessorError,
    MissingUIDError,
    NoChemicalStringError,
    EmptyHeaderError,
)
from noctis.data_architecture.graph_schema import GraphSchema
from noctis.data_architecture.datamodel import Node, Relationship
from noctis.data_architecture.datacontainer import DataContainer
from abc import ABC
from linchemin.cgu.syngraph import (
    SynGraph,
    MonopartiteReacSynGraph,
    MonopartiteMolSynGraph,
    BipartiteSynGraph,
)
from dask.distributed import Client, as_completed
from tqdm import tqdm


class TestPreprocessor(unittest.TestCase):
    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.CSVPreprocessor"
    )
    def test_preprocess_csv_for_neo4j_serial(self, mock_csv_preprocessor):
        schema = {
            "base_nodes": {"chemical_equation": "CE", "molecule": "MM"},
            "base_relationships": {
                "product": {
                    "type": "PRODUCT",
                    "start_node": "chemical_equation",
                    "end_node": "molecule",
                },
                "reactant": {
                    "type": "REACTANT",
                    "start_node": "molecule",
                    "end_node": "chemical_equation",
                },
            },
        }

        preprocessor = Preprocessor(schema=GraphSchema.build_from_dict(schema))
        input_file = "test.csv"
        # Create a mock instance for CSVPreprocessor
        mock_csv_instance = Mock()
        mock_csv_preprocessor.return_value = mock_csv_instance

        preprocessor.preprocess_csv_for_neo4j_serial(
            input_file,
            output_folder="output",
            tmp_folder="tmp",
            validation=True,
            prefix="K",
            chunksize=1000,
            inp_chem_format="smiles",
            out_chem_format="smiles",
        )

        # Assert that CSVPreprocessor was called once
        mock_csv_preprocessor.assert_called_once()

        # Get the arguments of the call
        call_args = mock_csv_preprocessor.call_args

        # Assert the first argument is the schema
        self.assertIsInstance(call_args[0][0], GraphSchema)

        # Assert the second argument is a PreprocessorConfig instance
        self.assertIsInstance(call_args[0][1], PreprocessorConfig)

        # Assert specific attributes of the PreprocessorConfig
        config = call_args[0][1]
        graph_schema = call_args[0][0]
        self.assertEqual(config.output_folder, "output")
        self.assertEqual(config.tmp_folder, "tmp")
        self.assertTrue(config.validation)
        self.assertEqual(config.prefix, "K")
        self.assertEqual(config.chunksize, 1000)
        self.assertEqual(config.inp_chem_format, "smiles")
        self.assertEqual(config.out_chem_format, "smiles")
        self.assertEqual(graph_schema.base_nodes["chemical_equation"], "CE")
        self.assertEqual(graph_schema.base_nodes["molecule"], "MM")

        # You can add more assertions for other attributes as needed

        # If you want to check that no other unexpected attributes were set:
        expected_attributes = {
            "output_folder",
            "tmp_folder",
            "validation",
            "prefix",
            "chunksize",
            "inp_chem_format",
            "out_chem_format",
            "delete_tmp",
            "delimiter",
            "lineterminator",
            "quotechar",
            "blocksize",
            "nrows",
        }
        self.assertEqual(set(vars(config).keys()), expected_attributes)

    def test_preprocessor_initialization(self):
        schema = GraphSchema()
        schema.base_nodes["molecule"] = "TESTMOLECULE"

        preprocessor = Preprocessor(schema=schema)

        self.assertEqual(preprocessor.schema, schema)

    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.PythonObjectPreprocessorFactory"
    )
    def test_preprocess_object_for_neo4j(self, mock_factory):
        preprocessor = Preprocessor()
        mock_preprocessor = Mock(spec=PythonObjectPreprocessorInterface)
        mock_factory.get_preprocessor.return_value = mock_preprocessor
        mock_data_container = Mock(spec=DataContainer)
        mock_preprocessor.run.return_value = mock_data_container

        # Test with DataFrame
        df_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        data_type = "pandas"
        result = preprocessor.preprocess_object_for_neo4j(df_data, data_type)

        self._assert_get_preprocessor_call(mock_factory, data_type)
        mock_preprocessor.run.assert_called_once_with(df_data)
        self.assertEqual(result, mock_data_container)

        # Reset mocks
        mock_factory.reset_mock()
        mock_preprocessor.reset_mock()

        # Test with list of strings
        string_data = ["CCO>>CC(=O)O", "C=C>>CC"]
        data_type = "reaction_strings"
        result = preprocessor.preprocess_object_for_neo4j(string_data, data_type)

        self._assert_get_preprocessor_call(mock_factory, data_type)
        mock_preprocessor.run.assert_called_once_with(string_data)
        self.assertEqual(result, mock_data_container)

        # Reset mocks
        mock_factory.reset_mock()
        mock_preprocessor.reset_mock()

        # Test with list of SynGraphs
        mock_syngraph = Mock(spec=SynGraph)
        syngraph_data = [mock_syngraph, mock_syngraph]
        data_type = "syngraph"
        result = preprocessor.preprocess_object_for_neo4j(syngraph_data, data_type)

        self._assert_get_preprocessor_call(mock_factory, data_type)
        mock_preprocessor.run.assert_called_once_with(syngraph_data)
        self.assertEqual(result, mock_data_container)

    def _assert_get_preprocessor_call(self, mock_factory, expected_data_type):
        mock_factory.get_preprocessor.assert_called_once()
        call_args = mock_factory.get_preprocessor.call_args[0]
        self.assertEqual(call_args[0], expected_data_type)
        self.assertIsInstance(call_args[1], GraphSchema)
        self.assertIsInstance(call_args[2], PreprocessorConfig)

    def test_get_failed_strings_for_csv_preprocessor(self):
        preprocessor = Preprocessor()
        mock_csv_preprocessor = Mock(spec=CSVPreprocessor)
        mock_config = Mock()
        mock_config.output_folder = "/output"
        mock_csv_preprocessor.config = mock_config
        preprocessor.preprocessor = mock_csv_preprocessor

        with patch("builtins.print") as mock_print:
            result = preprocessor.get_failed_strings()

        self.assertIsNone(result)

    def test_get_failed_strings_for_python_object_preprocessor(self):
        preprocessor = Preprocessor()
        mock_python_preprocessor = Mock(spec=ReactionStringsPreprocessor)
        mock_python_preprocessor.failed_strings = ["failed1", "failed2"]
        preprocessor.preprocessor = mock_python_preprocessor

        result = preprocessor.get_failed_strings()

        self.assertEqual(result, ["failed1", "failed2"])

    def test_get_failed_strings_no_preprocessor(self):
        preprocessor = Preprocessor()

        with self.assertRaises(NoPreprocessorError):
            preprocessor.get_failed_strings()

    def test_preprocess_csv_for_neo4j_default_config(self):
        preprocessor = Preprocessor()
        with patch(
            "noctis.data_transformation.preprocessing.data_preprocessing.CSVPreprocessor"
        ) as mock_csv_preprocessor:
            preprocessor.preprocess_csv_for_neo4j_serial("input.csv")
            # Assert that CSVPreprocessor was called once
            mock_csv_preprocessor.assert_called_once()

            # Get the arguments of the call
            call_args = mock_csv_preprocessor.call_args

            # Assert the first argument is the schema
            self.assertIsInstance(call_args[0][0], GraphSchema)

            # Assert the second argument is a PreprocessorConfig instance
            self.assertIsInstance(call_args[0][1], PreprocessorConfig)

    def test_preprocess_object_for_neo4j_default_config(self):
        schema = GraphSchema()
        preprocessor = Preprocessor(schema)
        mock_data = pd.DataFrame()
        with patch(
            "noctis.data_transformation.preprocessing.data_preprocessing.PythonObjectPreprocessorFactory"
        ) as mock_factory:
            preprocessor.preprocess_object_for_neo4j(mock_data, "dataframe")
            mock_factory.get_preprocessor.assert_called_once_with(
                "dataframe", schema, PreprocessorConfig()
            )


class TestPandasRowPreprocessor(unittest.TestCase):
    def setUp(self):
        self.schema = Mock(spec=GraphSchema)
        self.schema.base_nodes = {
            "chemical_equation": "ChemicalEquation",
            "molecule": "Molecule",
        }
        self.schema.extra_nodes = {"extra_node": "ExtraNode"}
        self.config = PreprocessorConfig(
            output_folder="/output",
            tmp_folder="/tmp",
            validation=True,
            prefix="test_",
            blocksize=64,
            chunksize=1000,
            inp_chem_format="smiles",
            out_chem_format="smiles",
        )

        class ConcretePreprocessor(PandasRowPreprocessorBase):
            pass

        self.preprocessor = ConcretePreprocessor(self.schema, self.config)

    def test_init(self):
        self.assertEqual(self.preprocessor.schema, self.schema)
        self.assertEqual(self.preprocessor.config, self.config)

    def test_validate_header_valid(self):
        header = ["ChemicalEquation.smiles", "ExtraNode.uid", "ChemicalEquation.name"]
        # This should not raise an exception
        self.preprocessor._validate_the_header(header)

    def test_validate_header_empty(self):
        header = []
        # This should not raise an exception
        with self.assertRaises(EmptyHeaderError):
            self.preprocessor._validate_the_header(header)

    def test_validate_header_missing_chemical_equation(self):
        header = ["ExtraNode.uid", "ChemicalEquation.name"]
        with self.assertRaises(NoChemicalStringError):
            self.preprocessor._validate_the_header(header)

    def test_validate_header_missing_extra_node_uid(self):
        header = ["ChemicalEquation.smiles", "ExtraNode.name"]
        with self.assertRaises(MissingUIDError):
            self.preprocessor._validate_the_header(header)

    def test_validate_header_ignore_molecule_node(self):
        header = ["ChemicalEquation.smiles", "ExtraNode.uid", "Molecule.smiles"]
        with patch(
            "noctis.data_transformation.preprocessing.data_preprocessing.logger.warning"
        ) as mock_warning:
            self.preprocessor._validate_the_header(header)
        mock_warning.assert_called_with(
            "MoleculeNodeWarning: Field 'Molecule.smiles' will be ignored. Molecule nodes are reconstructed from 'ChemicalEquation' nodes."
        )

    def test_validate_header_ignore_invalid_node(self):
        header = ["ChemicalEquation.smiles", "ExtraNode.uid", "InvalidNode.property"]
        with patch(
            "noctis.data_transformation.preprocessing.data_preprocessing.logger.warning"
        ) as mock_warning:
            self.preprocessor._validate_the_header(header)
        mock_warning.assert_called_with(
            "FieldNotInSchemaWarning: Field 'InvalidNode.property' will be ignored during processing. Node 'InvalidNode' is not in the schema."
        )

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
        mock_expander_instance.expand_reaction_step = Mock(
            return_value=({"node1": {"attr1": "val1"}}, {"rel1": {"attr2": "val2"}})
        )

        # Call the method
        (
            result_nodes,
            result_relationships,
            failed_string,
        ) = self.preprocessor._process_row(row)

        # Assert the mocks were called correctly
        self.preprocessor._split_row_by_node_types.assert_called_once_with(row)
        MockGraphExpander.assert_called_once_with(self.schema)
        mock_expander_instance.expand_reaction_step.assert_called_once_with(
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
            "Person.uid": "NNN",
            "Address.uid": "SSS",
            "Job.uid": "BBB",
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
            "Person": {"uid": "NNN", "properties": {"name": "John Doe", "age": "30"}},
            "Address": {
                "uid": "SSS",
                "properties": {"street": "Main St", "city": "New York"},
            },
            "Job": {"uid": "BBB", "properties": {"title": "Engineer"}},
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


class TestCSVPreprocessor(unittest.TestCase):
    def setUp(self):
        self.schema = Mock(spec=GraphSchema)
        self.schema.base_nodes = {"tag": "label"}
        self.schema.base_relationships = {"tag": "type"}
        self.config = PreprocessorConfig(
            output_folder="/output",
            tmp_folder="/tmp",
            validation=True,
            prefix="test_",
            blocksize=64,
            chunksize=1000,
            inp_chem_format="smiles",
            out_chem_format="smiles",
        )
        self.preprocessor = CSVPreprocessor(self.schema, self.config)
        self.input_file = "test.csv"

    def test_init(self):
        self.assertEqual(self.preprocessor.schema, self.schema)
        self.assertEqual(self.preprocessor.config, self.config)

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="header1,header2\nvalue1,value2",
    )
    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing._delete_tmp_folder"
    )
    @patch.object(CSVPreprocessor, "_merge_all_partition_files")
    @patch.object(CSVPreprocessor, "_validate_the_header")
    @patch.object(CSVPreprocessor, "_run_parallel")
    @patch.object(CSVPreprocessor, "_run_serial")
    def test_run_parallel(
        self,
        mock_run_serial,
        mock_run_parallel,
        mock_validate_header,
        mock_merge_all_partition_files,
        mock_delete_tmp_folder,
        mock_file,
    ):
        self.config.parallel = True
        self.preprocessor.run(self.input_file, parallel=True)
        mock_validate_header.assert_called_once()
        mock_run_parallel.assert_called_once()
        mock_run_serial.assert_not_called()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="header1,header2\nvalue1,value2",
    )
    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing._delete_tmp_folder"
    )
    @patch.object(CSVPreprocessor, "_merge_all_partition_files")
    @patch.object(CSVPreprocessor, "_validate_the_header")
    @patch.object(CSVPreprocessor, "_run_parallel")
    @patch.object(CSVPreprocessor, "_run_serial")
    def test_run_serial(
        self,
        mock_run_serial,
        mock_run_parallel,
        mock_validate_header,
        mock_merge_all_partition_files,
        mock_delete_tmp_folder,
        mock_file,
    ):
        self.preprocessor.run(self.input_file, parallel=False)
        mock_validate_header.assert_called_once()
        mock_run_serial.assert_called_once()
        mock_run_parallel.assert_not_called()

    @patch("noctis.data_transformation.preprocessing.data_preprocessing.Client")
    @patch("noctis.data_transformation.preprocessing.data_preprocessing.dd.read_csv")
    @patch("noctis.data_transformation.preprocessing.data_preprocessing.np.concatenate")
    @patch("noctis.data_transformation.preprocessing.data_preprocessing.np.cumsum")
    @patch("noctis.data_transformation.preprocessing.data_preprocessing.as_completed")
    def test_run_parallel_execution(
        self,
        mock_as_completed,
        mock_cumsum,
        mock_concatenate,
        mock_read_csv,
        mock_client,
    ):
        # Mock the Dask DataFrame
        mock_ddf = Mock()
        mock_ddf.npartitions = 3
        mock_read_csv.return_value = mock_ddf

        # Mock partition sizes
        mock_partition_sizes = MagicMock()
        mock_ddf.map_partitions.return_value.compute.return_value = mock_partition_sizes

        # Mock numpy functions
        mock_cumsum.return_value = np.array([100, 250, 450])
        mock_concatenate.return_value = np.array([0, 100, 250])

        # Mock the Dask Client and its compute method
        mock_client_instance = Mock(spec=Client)
        mock_client.return_value = mock_client_instance

        # Mock the futures and their results
        mock_futures = [Mock(), Mock(), Mock()]
        for future in mock_futures:
            future.result.return_value = None
        mock_client_instance.compute.return_value = mock_futures
        mock_as_completed.return_value = mock_futures

        # Run the method
        self.preprocessor._run_parallel(dask_client=None)

        # Check if map_partitions was called twice
        assert (
            mock_ddf.map_partitions.call_count == 2
        ), f"Expected 2 calls to map_partitions, got {mock_ddf.map_partitions.call_count}"

        # Check the first call to map_partitions (for calculating partition sizes)
        first_call = mock_ddf.map_partitions.call_args_list[0]
        assert (
            first_call[0][0] == len
        ), "First call to map_partitions should be with len function"

        # Check the second call to map_partitions (for processing partitions)
        second_call = mock_ddf.map_partitions.call_args_list[1]
        assert callable(
            second_call[0][0]
        ), "Second call to map_partitions should be with a callable"
        assert (
            "meta" in second_call[1]
        ), "Second call to map_partitions should include 'meta' keyword argument"

        # Assertions
        mock_read_csv.assert_called_once()
        mock_client_instance.compute.assert_called_once()
        mock_as_completed.assert_called_once()

        # Check that the client's compute method was called with the result of to_delayed()
        assert (
            mock_ddf.map_partitions.return_value.to_delayed.called
        ), "to_delayed() should have been called on the result of map_partitions"
        mock_client_instance.compute.assert_called_once_with(
            mock_ddf.map_partitions.return_value.to_delayed.return_value
        )

    @patch("pandas.read_csv")
    @patch.object(CSVPreprocessor, "_process_partition")
    @patch("builtins.open", new_callable=mock_open, read_data="header\nrow1\nrow2\n")
    def test_run_serial_execution(
        self, mock_open, mock_process_partition, mock_read_csv
    ):
        # Mock the DataFrame returned by read_csv
        mock_df = Mock()
        mock_read_csv.return_value = [mock_df, mock_df]  # Simulate two partitions

        self.preprocessor._run_serial()

        # Assertions
        self.assertEqual(mock_process_partition.call_count, 2)
        mock_process_partition.assert_has_calls([call(mock_df, 0), call(mock_df, 1)])

    @patch("pandas.read_csv")
    def test_serial_partition_generator_without_chunksize(self, mock_read_csv):
        mock_df = Mock()
        mock_read_csv.return_value = mock_df
        self.config.chunksize = None

        # Assuming these are the default or configured parameters for read_csv
        expected_kwargs = {
            "delimiter": ",",
            "quotechar": '"',
            "lineterminator": None,
            "nrows": None,
        }

        result = list(self.preprocessor._serial_partition_generator())

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], mock_df)
        mock_read_csv.assert_called_once_with(
            self.preprocessor.input_file, **expected_kwargs
        )

    @patch("pandas.read_csv")
    def test_serial_partition_generator_with_chunksize(self, mock_read_csv):
        mock_df1, mock_df2 = Mock(), Mock()
        mock_read_csv.return_value = [mock_df1, mock_df2]
        self.config.chunksize = 1000
        self.schema.base_nodes["chemical_equation"] = "ChemicalEquation"

        result = list(self.preprocessor._serial_partition_generator())

        self.assertEqual(len(result), 2)
        self.assertEqual(result, [mock_df1, mock_df2])

        # Include additional expected keyword arguments
        expected_kwargs = {
            "chunksize": self.config.chunksize,
            "delimiter": ",",
            "quotechar": '"',
            "lineterminator": None,
            "nrows": None,
        }
        mock_read_csv.assert_called_once_with(
            self.preprocessor.input_file, **expected_kwargs
        )

    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing._update_partition_dict_with_row"
    )
    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.Neo4jImportStyle.export_nodes"
    )
    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.Neo4jImportStyle.export_relationships"
    )
    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing._save_dataframes_to_partition_csv"
    )
    def test_process_partition(
        self, mock_save, mock_styler_relationships, mock_styler_nodes, mock_update
    ):
        mock_partition_data = pd.DataFrame(
            {"header1": ["value1"], "header2": ["value2"]}
        )
        mock_styler_relationships.side_effect = [Mock(), Mock()]
        mock_styler_nodes.side_effect = [Mock(), Mock()]

        # Mock the _process_row method
        mock_nodes = {"node1": {"attr1": "value1"}}
        mock_relationships = {"rel1": {"attr2": "value2"}}
        mock_process_row = Mock(
            side_effect=[
                (mock_nodes, mock_relationships, None),  # Successful processing
                ({}, {}, "wrong_smiles"),  # Failed processing
            ]
        )
        self.schema.base_nodes["chemical_equation"] = "ChemicalEquation"

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
        self.assertEqual(mock_styler_relationships.call_count, 1)
        self.assertEqual(mock_styler_nodes.call_count, 1)
        mock_save.assert_called_once()

    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing._save_dataframes_to_partition_csv"
    )
    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing._save_list_to_partition_csv"
    )
    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.Neo4jImportStyle.export_nodes"
    )
    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.Neo4jImportStyle.export_relationships"
    )
    def test_process_partition_with_failed_strings(
        self,
        mock_export_relationships,
        mock_export_nodes,
        mock_save_list,
        mock_save_dataframes,
    ):
        df = pd.DataFrame(
            {
                "ChemicalEquation.smiles": [
                    "CCO>>CC(=O)O",
                    "InvalidSmiles",
                    "C=C>>CC",
                    pd.NA,
                ],
                "ChemicalEquation.name": [
                    "Reaction1",
                    "Reaction2",
                    "Reaction3",
                    "Reaction4",
                ],
            }
        )

        self.schema.base_nodes["chemical_equation"] = "ChemicalEquation"
        self.config.inp_chem_format = "smiles"

        mock_process_row = Mock(
            side_effect=[
                ({"Node1": [Mock()]}, {"Rel1": [Mock()]}, None),
                ({}, {}, "InvalidSmiles"),
                ({"Node2": [Mock()]}, {"Rel2": [Mock()]}, None),
                ({}, {}, pd.NA),
            ]
        )

        with patch.object(self.preprocessor, "_process_row", mock_process_row):
            self.preprocessor._process_partition(df, 0)

        # Check call for failed strings
        mock_save_list.assert_any_call(
            [["InvalidSmiles", 3]],
            header=["ChemicalEquation.smiles", "index"],
            output_dir=self.config.tmp_folder,
            name="failed_strings",
            partition_num=0,
        )

        # Check call for empty strings
        mock_save_list.assert_any_call(
            [[5]],
            header=["index"],
            output_dir=self.config.tmp_folder,
            name="empty_strings",
            partition_num=0,
        )

        # Ensure _save_list_to_partition_csv was called exactly twice
        assert mock_save_list.call_count == 2

        # Check other method calls
        mock_export_nodes.assert_called_once()
        mock_export_relationships.assert_called_once()
        mock_save_dataframes.assert_called_once()


class TestPythonObjectPreprocessorInterface(unittest.TestCase):
    def test_interface_structure(self):
        # Check if PythonObjectPreprocessorInterface is an abstract base class
        self.assertTrue(issubclass(PythonObjectPreprocessorInterface, ABC))

        # Check if the run method is an abstract method
        self.assertTrue(hasattr(PythonObjectPreprocessorInterface, "run"))
        self.assertTrue(
            getattr(PythonObjectPreprocessorInterface, "run").__isabstractmethod__
        )

    def test_concrete_implementation(self):
        # Define a concrete implementation of the interface
        class ConcretePreprocessor(PythonObjectPreprocessorInterface):
            def run(self, data: object) -> DataContainer:
                # Simple implementation for testing
                return DataContainer()

        # Check if we can instantiate the concrete implementation
        try:
            preprocessor = ConcretePreprocessor()
        except TypeError:
            self.fail(
                "Failed to instantiate concrete implementation of PythonObjectPreprocessorInterface"
            )

        # Check if the run method can be called and returns a DataContainer
        result = preprocessor.run(any)
        self.assertIsInstance(result, DataContainer)

    def test_abstract_class_instantiation(self):
        # Attempt to instantiate the abstract base class should raise TypeError
        with self.assertRaises(TypeError):
            PythonObjectPreprocessorInterface()

    def test_incomplete_implementation(self):
        # Define an incomplete implementation (missing run method)
        class IncompletePreprocessor(PythonObjectPreprocessorInterface):
            pass

        # Attempt to instantiate the incomplete implementation should raise TypeError
        with self.assertRaises(TypeError):
            IncompletePreprocessor()

    def test_incorrect_return_type(self):
        # Define an implementation with incorrect return type
        class IncorrectReturnTypePreprocessor(PythonObjectPreprocessorInterface):
            def run(self, data: object) -> str:
                return "Not a DataContainer"

        # This should not raise an error at instantiation time
        preprocessor = IncorrectReturnTypePreprocessor()

        # But it violates the interface contract
        self.assertNotIsInstance(preprocessor.run(any), DataContainer)


class TestDataFramePreprocessor(unittest.TestCase):
    def setUp(self):
        self.schema = Mock(spec=GraphSchema)
        self.schema.base_nodes = {"chemical_equation": "CE"}
        self.config = Mock(spec=PreprocessorConfig)
        self.preprocessor = DataFramePreprocessor(self.schema, self.config)

    def test_initialization(self):
        self.assertIsInstance(self.preprocessor, DataFramePreprocessor)
        self.assertEqual(self.preprocessor.schema, self.schema)
        self.assertEqual(self.preprocessor.config, self.config)

    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.create_data_container"
    )
    def test_run_with_failed_strings(self, mock_create_data_container):
        df = pd.DataFrame(
            {
                "ChemicalEquation.smiles": ["CCO>>CC(=O)O", "InvalidSmiles", "C=C>>CC"],
                "ChemicalEquation.name": ["Reaction1", "Reaction2", "Reaction3"],
            }
        )

        mock_process_row = Mock(
            side_effect=[
                ({"Node1": [Mock()]}, {"Rel1": [Mock()]}, None),
                ({}, {}, "InvalidSmiles"),
                ({"Node2": [Mock()]}, {"Rel2": [Mock()]}, None),
            ]
        )

        with patch.object(self.preprocessor, "_process_row", mock_process_row):
            with patch.object(self.preprocessor, "_validate_the_header"):
                result = self.preprocessor.run(df)

        self.assertEqual(len(self.preprocessor.failed_strings), 1)
        self.assertEqual(self.preprocessor.failed_strings[0], "InvalidSmiles")
        mock_create_data_container.assert_called_once()

    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.create_data_container"
    )
    def test_run_single_row(self, mock_create_data_container):
        df = pd.DataFrame({"col1": [1], "col2": ["a"]})
        mock_nodes = {"Node1": [Mock(spec=Node)]}
        mock_relationships = {"Rel1": [Mock(spec=Relationship)]}

        with patch.object(
            self.preprocessor,
            "_process_row",
            return_value=(mock_nodes, mock_relationships, None),
        ):
            with patch.object(
                self.preprocessor, "_validate_the_header"
            ) as mock_validate_header:
                result = self.preprocessor.run(df)

        mock_validate_header.assert_called_once_with(["col1", "col2"])
        mock_create_data_container.assert_called_once_with(
            [mock_nodes["Node1"][0]],
            [mock_relationships["Rel1"][0]],
            self.schema.base_nodes["chemical_equation"],
        )
        self.assertEqual(result, mock_create_data_container.return_value)

    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.create_data_container"
    )
    def test_run_multiple_rows(self, mock_create_data_container):
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        mock_nodes1 = {"Node1": [Mock(spec=Node)], "Node2": [Mock(spec=Node)]}
        mock_relationships1 = {"Rel1": [Mock(spec=Relationship)]}
        mock_nodes2 = {"Node1": [Mock(spec=Node)]}
        mock_relationships2 = {
            "Rel1": [Mock(spec=Relationship)],
            "Rel2": [Mock(spec=Relationship)],
        }

        with patch.object(
            self.preprocessor,
            "_process_row",
            side_effect=[
                (mock_nodes1, mock_relationships1, None),
                (mock_nodes2, mock_relationships2, None),
            ],
        ):
            with patch.object(
                self.preprocessor, "_validate_the_header"
            ) as mock_validate_header:
                result = self.preprocessor.run(df)

        mock_validate_header.assert_called_once_with(["col1", "col2"])
        expected_nodes = [
            mock_nodes1["Node1"][0],
            mock_nodes1["Node2"][0],
            mock_nodes2["Node1"][0],
        ]
        expected_relationships = [
            mock_relationships1["Rel1"][0],
            mock_relationships2["Rel1"][0],
            mock_relationships2["Rel2"][0],
        ]
        mock_create_data_container.assert_called_once_with(
            expected_nodes,
            expected_relationships,
            self.schema.base_nodes["chemical_equation"],
        )
        self.assertEqual(result, mock_create_data_container.return_value)

    def test_run_integration(self):
        df = pd.DataFrame(
            {
                "Molecule.uid": ["M1", "M2"],
                "Molecule.smiles": ["CCO", "CC"],
                "Reaction.uid": ["reac1", "reac2"],
                "Reaction.name": ["Ethanol synthesis", "Ethane synthesis"],
            }
        )

        mock_node1 = Mock(spec=Node)
        mock_node2 = Mock(spec=Node)
        mock_node3 = Mock(spec=Node)
        mock_node4 = Mock(spec=Node)
        mock_rel1 = Mock(spec=Relationship)
        mock_rel2 = Mock(spec=Relationship)

        mock_process_row_return1 = (
            {"Molecule": [mock_node1], "Reaction": [mock_node2]},
            {"PARTICIPATES_IN": [mock_rel1]},
            None,
        )
        mock_process_row_return2 = (
            {"Molecule": [mock_node3], "Reaction": [mock_node4]},
            {"PARTICIPATES_IN": [mock_rel2]},
            None,
        )

        with patch.object(
            self.preprocessor,
            "_process_row",
            side_effect=[mock_process_row_return1, mock_process_row_return2],
        ):
            with patch.object(
                self.preprocessor, "_validate_the_header"
            ) as mock_validate_header:
                with patch(
                    "noctis.data_transformation.preprocessing.data_preprocessing.create_data_container"
                ) as mock_create_data_container:
                    result = self.preprocessor.run(df)

            # Assert that _validate_the_header was called once with the correct headers
        mock_validate_header.assert_called_once_with(
            ["Molecule.uid", "Molecule.smiles", "Reaction.uid", "Reaction.name"]
        )

        mock_create_data_container.assert_called_once_with(
            [mock_node1, mock_node2, mock_node3, mock_node4],
            [mock_rel1, mock_rel2],
            self.schema.base_nodes["chemical_equation"],
        )
        self.assertEqual(result, mock_create_data_container.return_value)


class TestChemicalStringPreprocessorBase(unittest.TestCase):
    def setUp(self):
        self.schema = Mock(spec=GraphSchema)
        self.schema.base_nodes = {"chemical_equation": "ChemicalEquation"}
        self.config = Mock(spec=PreprocessorConfig)
        self.config.inp_chem_format = "smiles"
        self.config.out_chem_format = "smiles"
        self.config.validation = True

        class ConcreteChemicalStringPreprocessor(ChemicalStringPreprocessorBase):
            pass

        self.preprocessor = ConcreteChemicalStringPreprocessor(self.schema, self.config)

    def test_initialization(self):
        self.assertIsInstance(self.preprocessor, ChemicalStringPreprocessorBase)
        self.assertEqual(self.preprocessor.schema, self.schema)
        self.assertEqual(self.preprocessor.config, self.config)

    def test_build_reaction_string_dict(self):
        reaction_string = "CCO>>CC(=O)O"
        expected_dict = {"ChemicalEquation": {"properties": {"smiles": "CCO>>CC(=O)O"}}}
        result = self.preprocessor._build_reaction_string_dict(reaction_string)
        self.assertEqual(result, expected_dict)

    @patch("noctis.data_transformation.preprocessing.data_preprocessing.GraphExpander")
    @patch("noctis.data_transformation.preprocessing.data_preprocessing.dict_to_list")
    def test_process_reaction_string(self, mock_dict_to_list, MockGraphExpander):
        reaction_string = "CCO>>CC(=O)O"
        mock_nodes = {
            "ChemicalEquation": [{"uid": "EQ1"}],
            "Molecule": [{"uid": "M1"}, {"uid": "M2"}],
        }
        mock_relationships = {
            "CONTAINS": [{"start": "EQ1", "end": "M1"}, {"start": "EQ1", "end": "M2"}]
        }

        mock_expander = MockGraphExpander.return_value
        mock_expander.expand_reaction_step.return_value = (
            mock_nodes,
            mock_relationships,
        )

        mock_dict_to_list.side_effect = [
            [
                Node(node_label="ChemicalEquation", uid="EQ1"),
                Node(node_label="Molecule", uid="M1"),
                Node(node_label="Molecule", uid="M2"),
            ],
            [
                Relationship(
                    start_node=Node(node_label="ChemicalEquation", uid="EQ1"),
                    end_node=Node(node_label="Molecule", uid="M1"),
                    relationship_type="CONTAINS",
                ),
                Relationship(
                    start_node=Node(node_label="ChemicalEquation", uid="EQ1"),
                    end_node=Node(node_label="Molecule", uid="M2"),
                    relationship_type="CONTAINS",
                ),
            ],
        ]

        (
            nodes,
            relationships,
            failed_string,
        ) = self.preprocessor._process_reaction_string(reaction_string)

        MockGraphExpander.assert_called_once_with(self.schema)
        mock_expander.expand_reaction_step.assert_called_once_with(
            {"ChemicalEquation": {"properties": {"smiles": "CCO>>CC(=O)O"}}},
            self.config.inp_chem_format,
            self.config.out_chem_format,
            self.config.validation,
        )

        self.assertEqual(len(mock_dict_to_list.call_args_list), 2)
        mock_dict_to_list.assert_any_call(mock_nodes)
        mock_dict_to_list.assert_any_call(mock_relationships)

        self.assertEqual(len(nodes), 3)
        self.assertEqual(len(relationships), 2)
        self.assertIsInstance(nodes[0], Node)
        self.assertIsInstance(relationships[0], Relationship)

    def test_abstract_base_class(self):
        self.assertTrue(issubclass(ChemicalStringPreprocessorBase, ABC))


class TestReactionStringsPreprocessor(unittest.TestCase):
    def setUp(self):
        self.schema = Mock(spec=GraphSchema)
        self.schema.base_nodes = {"chemical_equation": "CE"}
        self.config = Mock(spec=PreprocessorConfig)
        self.preprocessor = ReactionStringsPreprocessor(self.schema, self.config)

    def test_inheritance(self):
        self.assertIsInstance(self.preprocessor, ChemicalStringPreprocessorBase)
        self.assertTrue(hasattr(self.preprocessor, "run"))

    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.create_data_container"
    )
    def test_run(self, mock_create_data_container):
        # Mock data
        reaction_strings = ["CCO>>CC(=O)O", "C=C>>CC"]

        # Mock _process_reaction_string method
        self.preprocessor._process_reaction_string = Mock()
        self.preprocessor._process_reaction_string.side_effect = [
            (
                [
                    Node(node_label="ChemicalEquation", uid="EQ1"),
                    Node(node_label="Molecule", uid="M1"),
                ],
                [
                    Relationship(
                        start_node=Node(node_label="ChemicalEquation", uid="EQ1"),
                        end_node=Node(node_label="Molecule", uid="M1"),
                        relationship_type="CONTAINS",
                    )
                ],
                None,
            ),
            (
                [
                    Node(node_label="ChemicalEquation", uid="EQ2"),
                    Node(node_label="Molecule", uid="M2"),
                ],
                [
                    Relationship(
                        start_node=Node(node_label="ChemicalEquation", uid="EQ2"),
                        end_node=Node(node_label="Molecule", uid="M2"),
                        relationship_type="CONTAINS",
                    )
                ],
                None,
            ),
        ]

        # Mock create_data_container
        mock_data_container = Mock(spec=DataContainer)
        mock_create_data_container.return_value = mock_data_container

        # Run the method
        result = self.preprocessor.run(reaction_strings)

        # Assertions
        self.assertEqual(self.preprocessor._process_reaction_string.call_count, 2)
        self.preprocessor._process_reaction_string.assert_any_call("CCO>>CC(=O)O")
        self.preprocessor._process_reaction_string.assert_any_call("C=C>>CC")

        expected_nodes = [
            Node(node_label="ChemicalEquation", uid="EQ1"),
            Node(node_label="Molecule", uid="M1"),
            Node(node_label="ChemicalEquation", uid="EQ2"),
            Node(node_label="Molecule", uid="M2"),
        ]
        expected_relationships = [
            Relationship(
                start_node=Node(node_label="ChemicalEquation", uid="EQ1"),
                end_node=Node(node_label="Molecule", uid="M1"),
                relationship_type="CONTAINS",
            ),
            Relationship(
                start_node=Node(node_label="ChemicalEquation", uid="EQ2"),
                end_node=Node(node_label="Molecule", uid="M2"),
                relationship_type="CONTAINS",
            ),
        ]

        mock_create_data_container.assert_called_once_with(
            expected_nodes,
            expected_relationships,
            self.schema.base_nodes["chemical_equation"],
        )
        self.assertEqual(result, mock_data_container)


class TestSynGraphPreprocessor(unittest.TestCase):
    def setUp(self):
        self.schema = Mock(spec=GraphSchema)
        self.schema.base_nodes = {"chemical_equation": "ChemicalEquation"}
        self.config = Mock(spec=PreprocessorConfig)
        self.preprocessor = SynGraphPreprocessor(self.schema, self.config)

    def test_inheritance(self):
        self.assertIsInstance(self.preprocessor, ChemicalStringPreprocessorBase)
        self.assertTrue(hasattr(self.preprocessor, "run"))

    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.create_data_container"
    )
    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.extract_reactions_from_syngraph"
    )
    def test_run(self, mock_extract_reactions, mock_create_data_container):
        # Mock data
        mock_syngraph1 = Mock(spec=MonopartiteReacSynGraph)
        mock_syngraph2 = Mock(spec=BipartiteSynGraph)
        syngraphs = [mock_syngraph1, mock_syngraph2]

        # Mock extract_reactions_from_syngraph
        mock_extract_reactions.side_effect = [
            [{"query_id": 1, "input_string": "CCO>>CC(=O)O"}],
            [
                {"query_id": 2, "input_string": "C=C>>CC"},
                {"query_id": 3, "input_string": "CC>>CCC"},
            ],
        ]

        # Mock _process_reaction_string method
        self.preprocessor._process_reaction_string = Mock()
        self.preprocessor._process_reaction_string.side_effect = [
            (
                [
                    Node(node_label="ChemicalEquation", uid="EQ1"),
                    Node(node_label="Molecule", uid="M1"),
                ],
                [
                    Relationship(
                        start_node=Node(node_label="ChemicalEquation", uid="EQ1"),
                        end_node=Node(node_label="Molecule", uid="M1"),
                        relationship_type="CONTAINS",
                    )
                ],
                None,
            ),
            (
                [
                    Node(node_label="ChemicalEquation", uid="EQ2"),
                    Node(node_label="Molecule", uid="M2"),
                ],
                [
                    Relationship(
                        start_node=Node(node_label="ChemicalEquation", uid="EQ2"),
                        end_node=Node(node_label="Molecule", uid="M2"),
                        relationship_type="CONTAINS",
                    )
                ],
                None,
            ),
            (
                [
                    Node(node_label="ChemicalEquation", uid="EQ3"),
                    Node(node_label="Molecule", uid="M3"),
                ],
                [
                    Relationship(
                        start_node=Node(node_label="ChemicalEquation", uid="EQ3"),
                        end_node=Node(node_label="Molecule", uid="M3"),
                        relationship_type="CONTAINS",
                    )
                ],
                None,
            ),
        ]

        # Mock create_data_container
        mock_data_container = Mock(spec=DataContainer)
        mock_create_data_container.return_value = mock_data_container

        # Run the method
        result = self.preprocessor.run(syngraphs)

        # Assertions
        self.assertEqual(mock_extract_reactions.call_count, 2)
        mock_extract_reactions.assert_any_call(mock_syngraph1)
        mock_extract_reactions.assert_any_call(mock_syngraph2)

        self.assertEqual(self.preprocessor._process_reaction_string.call_count, 3)
        self.preprocessor._process_reaction_string.assert_any_call("CCO>>CC(=O)O")
        self.preprocessor._process_reaction_string.assert_any_call("C=C>>CC")
        self.preprocessor._process_reaction_string.assert_any_call("CC>>CCC")

        expected_nodes = [
            Node(node_label="ChemicalEquation", uid="EQ1"),
            Node(node_label="Molecule", uid="M1"),
            Node(node_label="ChemicalEquation", uid="EQ2"),
            Node(node_label="Molecule", uid="M2"),
            Node(node_label="ChemicalEquation", uid="EQ3"),
            Node(node_label="Molecule", uid="M3"),
        ]
        expected_relationships = [
            Relationship(
                start_node=Node(node_label="ChemicalEquation", uid="EQ1"),
                end_node=Node(node_label="Molecule", uid="M1"),
                relationship_type="CONTAINS",
            ),
            Relationship(
                start_node=Node(node_label="ChemicalEquation", uid="EQ2"),
                end_node=Node(node_label="Molecule", uid="M2"),
                relationship_type="CONTAINS",
            ),
            Relationship(
                start_node=Node(node_label="ChemicalEquation", uid="EQ3"),
                end_node=Node(node_label="Molecule", uid="M3"),
                relationship_type="CONTAINS",
            ),
        ]

        mock_create_data_container.assert_called_once_with(
            expected_nodes,
            expected_relationships,
            self.schema.base_nodes["chemical_equation"],
        )
        self.assertEqual(result, mock_data_container)

    def test_run_with_different_syngraph_types(self):
        mock_monopartite_reac = Mock(spec=MonopartiteReacSynGraph)
        mock_bipartite = Mock(spec=BipartiteSynGraph)
        mock_monopartite_mol = Mock(spec=MonopartiteMolSynGraph)

        with patch(
            "noctis.data_transformation.preprocessing.data_preprocessing.extract_reactions_from_syngraph"
        ) as mock_extract:
            mock_extract.return_value = [{"output_string": "C>>CC"}]
            with patch.object(
                self.preprocessor, "_process_reaction_string"
            ) as mock_process:
                mock_process.return_value = ([], [], None)
                with patch(
                    "noctis.data_transformation.preprocessing.data_preprocessing.create_data_container"
                ) as mock_create:
                    self.preprocessor.run(
                        [mock_monopartite_reac, mock_bipartite, mock_monopartite_mol]
                    )

        self.assertEqual(mock_extract.call_count, 3)
        self.assertEqual(mock_process.call_count, 3)


class TestPythonObjectPreprocessorFactory(unittest.TestCase):
    def setUp(self):
        self.schema = Mock(spec=GraphSchema)
        self.config = Mock(spec=PreprocessorConfig)

    def test_get_preprocessor_dataframe(self):
        data_type = "dataframe"
        preprocessor = PythonObjectPreprocessorFactory.get_preprocessor(
            data_type, self.schema, self.config
        )
        self.assertIsInstance(preprocessor, DataFramePreprocessor)

    def test_get_preprocessor_reaction_strings(self):
        data_type = "reaction_string"
        preprocessor = PythonObjectPreprocessorFactory.get_preprocessor(
            data_type, self.schema, self.config
        )

        self.assertIsInstance(preprocessor, ReactionStringsPreprocessor)

    def test_get_preprocessor_syngraph(self):
        data_type = "syngraph"
        preprocessor = PythonObjectPreprocessorFactory.get_preprocessor(
            data_type, self.schema, self.config
        )

        self.assertIsInstance(preprocessor, SynGraphPreprocessor)

    def test_get_preprocessor_unsupported_type(self):
        data_type = "unsupported_type"
        with self.assertRaises(ValueError) as context:
            PythonObjectPreprocessorFactory.get_preprocessor(
                data_type, self.schema, self.config
            )

        self.assertEqual(str(context.exception), f"Unsupported data type: {data_type}")


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.schema = GraphSchema.build_from_dict(
            {
                "base_nodes": {
                    "chemical_equation": "ChemicalEquation",
                    "molecule": "Molecule",
                },
                "base_relationships": {
                    "product": {
                        "type": "PRODUCT",
                        "start_node": "chemical_equation",
                        "end_node": "molecule",
                    },
                    "reactant": {
                        "type": "REACTANT",
                        "start_node": "molecule",
                        "end_node": "chemical_equation",
                    },
                },
                "extra_nodes": {"extra_node_1": "Extra1", "extra_node_2": "Extra2"},
                "extra_relationships": {
                    "extra_relationship_1": {
                        "type": "EXTRAREL1",
                        "start_node": "extra_node_1",
                        "end_node": "extra_node_2",
                    },
                    "extra_relationship_2": {
                        "type": "EXTRAREL2",
                        "start_node": "extra_node_2",
                        "end_node": "chemical_equation",
                    },
                },
            }
        )
        self.config = PreprocessorConfig(
            output_folder="output",
            tmp_folder="tmp",
            validation=False,
            prefix="test_",
            blocksize=64,
            chunksize=1000,
            inp_chem_format="smiles",
            out_chem_format="smiles",
        )
        self.preprocessor = CSVPreprocessor(self.schema, self.config)

        self.input_file = ("test.csv",)

    def test_end_to_end_process_partition(self):
        # Create sample input dataframe
        data_dict = {
            "ChemicalEquation.smiles": ["C>>O", "WrongSmiles"],
            "ChemicalEquation.uid": ["C123", "a"],
            "ChemicalEquation.random_property": ["hello", "a"],
            "Extra1.uid": ["ID123", "a"],
            "Extra1.value": ["DOGE", "a"],
            "Extra2.uid": ["N123", "a"],
        }
        input_df = pd.DataFrame(data_dict)

        # Create expected output dataframes
        expected_nodes_df = pd.DataFrame(
            {
                "node_id": [1, 2],
                "node_label": ["Label1", "Label2"],
                "property1": ["prop1", "prop2"],
            }
        )

        expected_relationships_df = pd.DataFrame(
            {
                "start_node": [1, 2],
                "end_node": [2, 1],
                "relationship_type": ["TYPE1", "TYPE2"],
            }
        )

        self.preprocessor._process_partition(input_df, 1)
