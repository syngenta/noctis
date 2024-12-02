import unittest
from unittest.mock import Mock, patch, call
import pandas as pd

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
)
from noctis.data_architecture.graph_schema import GraphSchema
from noctis.data_architecture.datamodel import DataContainer, Node, Relationship
from abc import ABC
from linchemin.cgu.syngraph import (
    SynGraph,
    MonopartiteReacSynGraph,
    MonopartiteMolSynGraph,
    BipartiteSynGraph,
)

# from noctis.data_transformation.preprocessing.graph_expander import GraphExpander, ReactionPreProcessor


class TestPreprocessor(unittest.TestCase):
    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.CSVPreprocessor"
    )
    def test_preprocess_csv_for_neo4j(self, mock_csv_preprocessor):
        schema = {"some": "schema"}
        config = PreprocessorConfig(
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

    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.PythonObjectPreprocessorFactory"
    )
    def test_preprocess_object_for_neo4j(self, mock_factory):
        schema = {"some": "schema"}
        config = PreprocessorConfig(
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
        mock_preprocessor = Mock(spec=PythonObjectPreprocessorInterface)
        mock_factory.get_preprocessor.return_value = mock_preprocessor
        mock_data_container = Mock(spec=DataContainer)
        mock_preprocessor.run.return_value = mock_data_container

        # Test with DataFrame
        df_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        data_type = "pandas"
        result = preprocessor.preprocess_object_for_neo4j(df_data, data_type, config)

        mock_factory.get_preprocessor.assert_called_once_with(data_type, schema, config)
        mock_preprocessor.run.assert_called_once_with(df_data)
        self.assertEqual(result, mock_data_container)

        # Reset mocks
        mock_factory.reset_mock()
        mock_preprocessor.reset_mock()

        # Test with list of strings
        string_data = ["CCO>>CC(=O)O", "C=C>>CC"]
        data_type = "reaction_strings"
        result = preprocessor.preprocess_object_for_neo4j(
            string_data, data_type, config
        )

        mock_factory.get_preprocessor.assert_called_once_with(data_type, schema, config)
        mock_preprocessor.run.assert_called_once_with(string_data)
        self.assertEqual(result, mock_data_container)

        # Reset mocks
        mock_factory.reset_mock()
        mock_preprocessor.reset_mock()

        # Test with list of SynGraphs
        mock_syngraph = Mock(spec=SynGraph)
        syngraph_data = [mock_syngraph, mock_syngraph]
        data_type = "syngraph"
        result = preprocessor.preprocess_object_for_neo4j(
            syngraph_data, data_type, config
        )

        mock_factory.get_preprocessor.assert_called_once_with(data_type, schema, config)
        mock_preprocessor.run.assert_called_once_with(syngraph_data)
        self.assertEqual(result, mock_data_container)


class TestPandasRowPreprocessor(unittest.TestCase):
    def setUp(self):
        self.schema = Mock(spec=GraphSchema)
        self.schema.base_nodes = {"tag": "label"}
        self.schema.base_relationships = {"tag": "type"}
        self.config = PreprocessorConfig(
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

        class ConcretePreprocessor(PandasRowPreprocessorBase):
            pass

        self.preprocessor = ConcretePreprocessor(self.schema, self.config)

    def test_init(self):
        self.assertEqual(self.preprocessor.schema, self.schema)
        self.assertEqual(self.preprocessor.config, self.config)

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
        mock_styler_relationships.side_effect = [
            Mock(),
            Mock(),
        ]
        mock_styler_nodes.side_effect = [
            Mock(),
            Mock(),
        ]
        # One for nodes, one for relationships

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
            mock_styler_relationships.call_count, 1
        )  # Called for both nodes and relationships
        self.assertEqual(
            mock_styler_nodes.call_count, 1
        )  # Called for both nodes and relationships
        mock_save.assert_called_once()


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
        self.config = Mock(spec=PreprocessorConfig)
        self.preprocessor = DataFramePreprocessor(self.schema, self.config)

    def test_initialization(self):
        self.assertIsInstance(self.preprocessor, DataFramePreprocessor)
        self.assertEqual(self.preprocessor.schema, self.schema)
        self.assertEqual(self.preprocessor.config, self.config)

    @patch(
        "noctis.data_transformation.preprocessing.data_preprocessing.create_data_container"
    )
    def test_run_empty_dataframe(self, mock_create_data_container):
        df = pd.DataFrame()
        result = self.preprocessor.run(df)

        mock_create_data_container.assert_called_once_with([], [])
        self.assertEqual(result, mock_create_data_container.return_value)

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
            return_value=(mock_nodes, mock_relationships),
        ):
            result = self.preprocessor.run(df)

        mock_create_data_container.assert_called_once_with(
            [mock_nodes["Node1"][0]], [mock_relationships["Rel1"][0]]
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
                (mock_nodes1, mock_relationships1),
                (mock_nodes2, mock_relationships2),
            ],
        ):
            result = self.preprocessor.run(df)

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
            expected_nodes, expected_relationships
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
        )
        mock_process_row_return2 = (
            {"Molecule": [mock_node3], "Reaction": [mock_node4]},
            {"PARTICIPATES_IN": [mock_rel2]},
        )

        with patch.object(
            self.preprocessor,
            "_process_row",
            side_effect=[mock_process_row_return1, mock_process_row_return2],
        ):
            with patch(
                "noctis.data_transformation.preprocessing.data_preprocessing.create_data_container"
            ) as mock_create_data_container:
                result = self.preprocessor.run(df)

        mock_create_data_container.assert_called_once_with(
            [mock_node1, mock_node2, mock_node3, mock_node4], [mock_rel1, mock_rel2]
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
        mock_expander.expand_from_csv.return_value = (mock_nodes, mock_relationships)

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

        nodes, relationships = self.preprocessor._process_reaction_string(
            reaction_string
        )

        MockGraphExpander.assert_called_once_with(self.schema)
        mock_expander.expand_from_csv.assert_called_once_with(
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
            expected_nodes, expected_relationships
        )
        self.assertEqual(result, mock_data_container)


class TestSynGraphPreprocessor(unittest.TestCase):
    def setUp(self):
        self.schema = Mock(spec=GraphSchema)
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
            [{"query_id": 1, "output_string": "CCO>>CC(=O)O"}],
            [
                {"query_id": 2, "output_string": "C=C>>CC"},
                {"query_id": 3, "output_string": "CC>>CCC"},
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
            expected_nodes, expected_relationships
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
                mock_process.return_value = ([], [])
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
            input_file="test.csv",
            output_folder="output",
            tmp_folder="tmp",
            validation=False,
            parallel=False,
            prefix="test_",
            blocksize=64,
            chunksize=1000,
            inp_chem_format="smiles",
            out_chem_format="smiles",
        )
        self.preprocessor = CSVPreprocessor(self.schema, self.config)

    def test_end_to_end_process_partition(self):
        # Create sample input dataframe
        data_dict = {
            "ChemicalEquation.smiles": ["C>>O"],
            "ChemicalEquation.uid": ["C123"],
            "ChemicalEquation.random_property": ["hello"],
            "Extra1.uid": ["ID123"],
            "Extra1.value": ["DOGE"],
            "Extra2.uid": ["N123"],
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
