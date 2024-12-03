import unittest

import pytest
from abc import ABC, abstractmethod
from unittest.mock import Mock, patch, call
from typing import Type
import pandas as pd
import networkx as nx

# Import your classes here
from noctis.data_transformation.postprocessing.chemdata_generators import (
    NoReactionSmilesError,
    ChemDataGeneratorInterface,
    ChemDataGeneratorFactory,
    PandasGenerator,
    SyngraphGenerator,
    ReactionStringGenerator,
    NetworkXGenerator,
)
from noctis.data_transformation.data_styles.dataframe_stylers import PandasExportStyle

from noctis.data_architecture.datamodel import (
    Node,
    Relationship,
    GraphRecord,
    DataContainer,
)
from collections import defaultdict
from noctis import settings
from linchemin.cgu.syngraph import BipartiteSynGraph


class TestChemDataGeneratorInterface(unittest.TestCase):
    def test_interface_structure(self):
        # Check if ChemDataGeneratorInterface is an abstract base class
        self.assertTrue(issubclass(ChemDataGeneratorInterface, ABC))

        # Check if the generate method is an abstract method
        self.assertTrue(hasattr(ChemDataGeneratorInterface, "generate"))
        self.assertTrue(
            getattr(ChemDataGeneratorInterface, "generate").__isabstractmethod__
        )

    def test_reaction_formats(self):
        # Check if reaction_formats is correctly defined
        expected_formats = {"smiles", "smarts", "rxn_blockV3K", "rxn_blockV2K"}
        self.assertEqual(ChemDataGeneratorInterface.reaction_formats, expected_formats)

    def test_concrete_implementation(self):
        # Define a concrete implementation of the interface
        class ConcreteGenerator(ChemDataGeneratorInterface):
            def generate(self, data_container, by_record, ce_label):
                return "Generated data"

        # Check if we can instantiate the concrete implementation
        generator = ConcreteGenerator()
        self.assertIsInstance(generator, ChemDataGeneratorInterface)

        # Check if the generate method can be called
        result = generator.generate(None, False, None)
        self.assertEqual(result, "Generated data")

    def test_abstract_class_instantiation(self):
        # Attempt to instantiate the abstract base class should raise TypeError
        with self.assertRaises(TypeError):
            ChemDataGeneratorInterface()


class TestChemDataGeneratorFactory(unittest.TestCase):
    def setUp(self):
        # Clear the generators dictionary before each test
        ChemDataGeneratorFactory.generators = {}

    def test_register_and_get_generator(self):
        # Define a mock generator
        mock_generator = Mock(spec=ChemDataGeneratorInterface)

        # Register the mock generator
        @ChemDataGeneratorFactory.register_generator("mock")
        class MockGenerator:
            def __new__(cls):
                return mock_generator

        # Get the registered generator
        retrieved_generator = ChemDataGeneratorFactory.get_generator("mock")

        # Check if the retrieved generator is the same as the mock
        self.assertEqual(retrieved_generator, mock_generator)

    def test_get_unknown_generator(self):
        # Attempt to get an unregistered generator should raise ValueError
        with self.assertRaises(ValueError):
            ChemDataGeneratorFactory.get_generator("unknown")

    def test_register_multiple_generators(self):
        # Register multiple generators
        @ChemDataGeneratorFactory.register_generator("gen1")
        class Generator1(ChemDataGeneratorInterface):
            def generate(self, data_container, by_record, ce_label):
                return "Gen1"

        @ChemDataGeneratorFactory.register_generator("gen2")
        class Generator2(ChemDataGeneratorInterface):
            def generate(self, data_container, by_record, ce_label):
                return "Gen2"

        # Get and check both generators
        gen1 = ChemDataGeneratorFactory.get_generator("gen1")
        gen2 = ChemDataGeneratorFactory.get_generator("gen2")

        self.assertIsInstance(gen1, Generator1)
        self.assertIsInstance(gen2, Generator2)
        self.assertEqual(gen1.generate(None, False, None), "Gen1")
        self.assertEqual(gen2.generate(None, False, None), "Gen2")

    def test_register_decorator_returns_class(self):
        # The register_generator decorator should return the original class
        @ChemDataGeneratorFactory.register_generator("test")
        class TestGenerator(ChemDataGeneratorInterface):
            def generate(self, data_container, by_record, ce_label):
                pass

        self.assertTrue(issubclass(TestGenerator, ChemDataGeneratorInterface))


def create_mock_data_container():
    node1 = Node(uid="N1", node_label="Molecule", properties={"smiles": "CC"})
    node2 = Node(uid="N2", node_label="Reaction", properties={"smiles": "CC>>CCO"})
    rel = Relationship(start_node=node1, end_node=node2, relationship_type="RELATION")
    record = GraphRecord(nodes=[node1, node2], relationships=[rel])
    return DataContainer(records={0: record, 1: record})


class TestPandasGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = PandasGenerator()
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
        self.sample_record = GraphRecord(
            nodes=self.sample_nodes, relationships=self.sample_relationships
        )
        self.sample_data_container = DataContainer(
            records={0: self.sample_record, 1: self.sample_record}
        )

    def test_generate(self):
        nodes_df, relationships_df = self.generator.generate(
            self.sample_data_container, by_record=False
        )
        self.assertIsInstance(nodes_df, pd.DataFrame)
        self.assertIsInstance(relationships_df, pd.DataFrame)
        self.assertEqual(len(nodes_df), 4)  # 2 nodes per record, 2 records
        self.assertEqual(
            len(relationships_df), 4
        )  # 2 relationships per record, 2 records

    def test_generate_by_record(self):
        nodes_df, relationships_df = self.generator.generate(
            self.sample_data_container, by_record=True
        )
        self.assertIsInstance(nodes_df, pd.DataFrame)
        self.assertIsInstance(relationships_df, pd.DataFrame)
        self.assertTrue("record_id" in nodes_df.columns)
        self.assertTrue("record_id" in relationships_df.columns)
        self.assertEqual(set(nodes_df["record_id"]), {0, 1})
        self.assertEqual(set(relationships_df["record_id"]), {0, 1})

    def test_process_records(self):
        nodes_data, relationships_data = self.generator._process_records(
            self.sample_data_container, by_record=False
        )
        self.assertEqual(len(nodes_data), 2)
        self.assertEqual(len(relationships_data), 2)
        self.assertIsInstance(nodes_data[0], pd.DataFrame)
        self.assertIsInstance(relationships_data[0], pd.DataFrame)

    def test_process_nodes(self):
        nodes_df = self.generator._process_nodes(
            self.sample_record, record_id=0, by_record=False
        )
        self.assertIsInstance(nodes_df, pd.DataFrame)
        self.assertEqual(len(nodes_df), 2)
        self.assertFalse("record_id" in nodes_df.columns)

    def test_process_nodes_by_record(self):
        nodes_df = self.generator._process_nodes(
            self.sample_record, record_id=0, by_record=True
        )
        self.assertIsInstance(nodes_df, pd.DataFrame)
        self.assertEqual(len(nodes_df), 2)
        self.assertTrue("record_id" in nodes_df.columns)
        self.assertTrue(all(nodes_df["record_id"] == 0))

    def test_process_relationships(self):
        relationships_df = self.generator._process_relationships(
            self.sample_record, record_id=0, by_record=False
        )
        self.assertIsInstance(relationships_df, pd.DataFrame)
        self.assertEqual(len(relationships_df), 2)
        self.assertFalse("record_id" in relationships_df.columns)

    def test_process_relationships_by_record(self):
        relationships_df = self.generator._process_relationships(
            self.sample_record, record_id=0, by_record=True
        )
        self.assertIsInstance(relationships_df, pd.DataFrame)
        self.assertEqual(len(relationships_df), 2)
        self.assertTrue("record_id" in relationships_df.columns)
        self.assertTrue(all(relationships_df["record_id"] == 0))

    def test_add_record_id_if_needed(self):
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        result_df = self.generator._add_record_id_if_needed(
            df, record_id=0, by_record=True
        )
        self.assertTrue("record_id" in result_df.columns)
        self.assertTrue(all(result_df["record_id"] == 0))

    def test_add_record_id_if_not_needed(self):
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        result_df = self.generator._add_record_id_if_needed(
            df, record_id=0, by_record=False
        )
        self.assertFalse("record_id" in result_df.columns)

    @patch.object(PandasExportStyle, "export_nodes")
    def test_style_nodes(self, mock_export_nodes):
        mock_export_nodes.return_value = {"all_nodes": pd.DataFrame({"col1": [1, 2]})}
        result = self.generator._style_nodes(self.sample_nodes)
        self.assertIsInstance(result, pd.DataFrame)
        mock_export_nodes.assert_called_once_with({"all_nodes": self.sample_nodes})

    @patch.object(PandasExportStyle, "export_relationships")
    def test_style_relationships(self, mock_export_relationships):
        mock_export_relationships.return_value = {
            "all_relationships": pd.DataFrame({"col1": [1, 2]})
        }
        result = self.generator._style_relationships(self.sample_relationships)
        self.assertIsInstance(result, pd.DataFrame)
        mock_export_relationships.assert_called_once_with(
            {"all_relationships": self.sample_relationships}
        )

    def test_concatenate_dataframes(self):
        df1 = pd.DataFrame({"col1": [1, 2]})
        df2 = pd.DataFrame({"col1": [3, 4]})
        nodes_result, relationships_result = self.generator._concatenate_dataframes(
            [df1, df2], [df2, df1]
        )
        self.assertEqual(len(nodes_result), 4)
        self.assertEqual(len(relationships_result), 4)


class TestNetworkXGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = NetworkXGenerator()
        self.sample_nodes = [
            Node(uid="M1", node_label="M", properties={"p1": "1", "p2": "2"}),
            Node(uid="M2", node_label="CE", properties={"p1": "1", "p2": "2"}),
        ]
        self.sample_relationships = [
            Relationship(
                start_node=self.sample_nodes[0],
                end_node=self.sample_nodes[1],
                relationship_type="P",
                properties={"r1": "a"},
            ),
            Relationship(
                start_node=self.sample_nodes[1],
                end_node=self.sample_nodes[0],
                relationship_type="R",
                properties={"r2": "b"},
            ),
        ]
        self.sample_record = GraphRecord(
            nodes=self.sample_nodes, relationships=self.sample_relationships
        )
        self.sample_data_container = DataContainer(
            records={0: self.sample_record, 1: self.sample_record}
        )

    def test_generate_by_record(self):
        result = self.generator.generate(self.sample_data_container, by_record=True)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for graph in result:
            self.assertIsInstance(graph, nx.Graph)
            self.assertEqual(len(graph.nodes), 2)
            self.assertEqual(len(graph.edges), 2)

    def test_generate_not_by_record(self):
        result = self.generator.generate(self.sample_data_container, by_record=False)
        self.assertIsInstance(result, nx.Graph)
        self.assertEqual(len(result.nodes), 2)  # Nodes are unique across records
        self.assertEqual(len(result.edges), 2)  # Edges are unique across records

    def test_process_records(self):
        result = self.generator._process_records(self.sample_data_container)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for graph in result:
            self.assertIsInstance(graph, nx.Graph)

    def test_process_record(self):
        result = self.generator._process_record(self.sample_record)
        self.assertIsInstance(result, nx.Graph)
        self.assertEqual(len(result.nodes), 2)
        self.assertEqual(len(result.edges), 2)

    def test_add_nodes_to_graph(self):
        G = nx.DiGraph()
        self.generator._add_nodes_to_graph(G, self.sample_nodes)
        self.assertEqual(len(G.nodes), 2)
        for node in G.nodes(data=True):
            self.assertIn("node_label", node[1])
            self.assertIn("properties", node[1])

    def test_add_relationships_to_graph(self):
        G = nx.DiGraph()
        self.generator._add_nodes_to_graph(G, self.sample_nodes)
        self.generator._add_relationships_to_graph(G, self.sample_relationships)
        self.assertEqual(len(G.edges), 2)
        for _, _, data in G.edges(data=True):
            self.assertIn("relationship_type", data)
            self.assertIn("properties", data)

    def test_create_node_attributes(self):
        node = self.sample_nodes[0]
        attributes = self.generator._create_node_attributes(node)
        self.assertEqual(attributes["node_label"], node.node_label)
        self.assertEqual(attributes["properties"], node.properties)

    def test_create_relationship_attributes(self):
        relationship = self.sample_relationships[0]
        attributes = self.generator._create_relationship_attributes(relationship)
        self.assertEqual(
            attributes["relationship_type"], relationship.relationship_type
        )
        self.assertEqual(attributes["properties"], relationship.properties)

    def test_compose_all_graphs(self):
        G1 = nx.Graph()
        G1.add_node(1, attr1="A")
        G2 = nx.Graph()
        G2.add_node(2, attr2="B")
        result = self.generator._compose_all_graphs([G1, G2])
        self.assertIsInstance(result, nx.Graph)
        self.assertEqual(len(result.nodes), 2)
        self.assertEqual(result.nodes[1]["attr1"], "A")
        self.assertEqual(result.nodes[2]["attr2"], "B")

    @patch("networkx.compose_all")
    def test_compose_all_graphs_calls_nx_compose_all(self, mock_compose_all):
        G1, G2 = nx.Graph(), nx.Graph()
        self.generator._compose_all_graphs([G1, G2])
        mock_compose_all.assert_called_once_with([G1, G2])


class TestReactionStringGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = ReactionStringGenerator()
        self.sample_nodes = [
            Node(uid="M1", node_label="M", properties={"p1": "1", "p2": "2"}),
            Node(
                uid="CE1",
                node_label="CE",
                properties={
                    "p1": "1",
                    "p2": "2",
                    "smiles": "C>>CC",
                    "smarts": "[C:1]>>[C:1][C]",
                },
            ),
            Node(
                uid="CE2",
                node_label="CE",
                properties={"smiles": "CC>>CCC", "smarts": "[C:1][C:2]>>[C:1][C:2][C]"},
            ),
        ]
        self.sample_record = GraphRecord(nodes=self.sample_nodes, relationships=[])
        self.sample_data_container = DataContainer(
            records={0: self.sample_record, 1: self.sample_record}
        )

    def test_generate_by_record(self):
        result = self.generator.generate(
            self.sample_data_container, by_record=True, ce_label="CE"
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for record_dict in result:
            self.assertIn("smiles", record_dict)
            self.assertIn("smarts", record_dict)

    def test_generate_not_by_record(self):
        result = self.generator.generate(
            self.sample_data_container, by_record=False, ce_label="CE"
        )
        self.assertIsInstance(result, dict)
        self.assertIn("smiles", result)
        self.assertIn("smarts", result)
        self.assertEqual(len(result["smiles"]), 4)  # 2 smiles per record, 2 records
        self.assertEqual(len(result["smarts"]), 4)  # 2 smarts per record, 2 records

    @patch("noctis.data_transformation.postprocessing.chemdata_generators.logger")
    def test_get_ce_label_with_none(self, mock_logger):
        settings.nodes.chemical_equation = "TEST_CE"
        result = self.generator._get_ce_label(None)
        self.assertEqual(result, "TEST_CE")
        mock_logger.warning.assert_called_once()

    def test_get_ce_label_with_value(self):
        result = self.generator._get_ce_label("CUSTOM_CE")
        self.assertEqual(result, "CUSTOM_CE")

    def test_process_records(self):
        result = self.generator._process_records(self.sample_data_container, "CE")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for record_dict in result:
            self.assertIn("smiles", record_dict)
            self.assertIn("smarts", record_dict)

    @patch("noctis.data_transformation.postprocessing.chemdata_generators.logger")
    def test_process_records_no_ce_nodes(self, mock_logger):
        data_container = DataContainer(
            records={0: GraphRecord(nodes=[self.sample_nodes[0]], relationships=[])}
        )
        result = self.generator._process_records(data_container, "CE")
        self.assertEqual(result, [{}])
        mock_logger.warning.assert_called_once()

    def test_process_record(self):
        result = self.generator._process_record(
            [self.sample_nodes[1], self.sample_nodes[2]]
        )
        self.assertIn("smiles", result)
        self.assertIn("smarts", result)
        self.assertEqual(len(result["smiles"]), 2)
        self.assertEqual(len(result["smarts"]), 2)

    def test_add_node_reactions_to_record(self):
        record_dict = defaultdict(list)
        self.generator._add_node_reactions_to_record(self.sample_nodes[1], record_dict)
        self.assertIn("smiles", record_dict)
        self.assertIn("smarts", record_dict)
        self.assertEqual(len(record_dict["smiles"]), 1)
        self.assertEqual(len(record_dict["smarts"]), 1)

    def test_process_node(self):
        result = self.generator._process_node(self.sample_nodes[1])
        self.assertIn("smiles", result)
        self.assertIn("smarts", result)
        self.assertEqual(result["smiles"], "C>>CC")
        self.assertEqual(result["smarts"], "[C:1]>>[C:1][C]")

    @patch("noctis.data_transformation.postprocessing.chemdata_generators.logger")
    def test_process_node_no_reaction_strings(self, mock_logger):
        result = self.generator._process_node(self.sample_nodes[0])
        self.assertEqual(result, {})
        mock_logger.warning.assert_called_once()

    def test_merge_dict_lists(self):
        dict_list = [
            {"smiles": ["C>>CC"], "smarts": ["[C:1]>>[C:1][C]"]},
            {"smiles": ["CC>>CCC"], "smarts": ["[C:1][C:2]>>[C:1][C:2][C]"]},
        ]
        result = self.generator._merge_dict_lists(dict_list)
        self.assertIn("smiles", result)
        self.assertIn("smarts", result)
        self.assertEqual(len(result["smiles"]), 2)
        self.assertEqual(len(result["smarts"]), 2)


class TestSyngraphGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = SyngraphGenerator()
        self.sample_nodes = [
            Node(uid="CE1", node_label="CE", properties={"smiles": "C>>CC"}),
            Node(uid="CE2", node_label="CE", properties={"smiles": "CC>>CCC"}),
            Node(uid="M1", node_label="M", properties={"p1": "1", "p2": "2"}),
        ]
        self.sample_record = GraphRecord(nodes=self.sample_nodes, relationships=[])
        self.sample_data_container = DataContainer(
            records={0: self.sample_record, 1: self.sample_record}
        )

    @patch(
        "noctis.data_transformation.postprocessing.chemdata_generators.merge_syngraph"
    )
    def test_generate_by_record(self, mock_merge_syngraph):
        result = self.generator.generate(
            self.sample_data_container, by_record=True, ce_label="CE"
        )
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for syngraph in result:
            self.assertIsInstance(syngraph, BipartiteSynGraph)
        mock_merge_syngraph.assert_not_called()

    @patch(
        "noctis.data_transformation.postprocessing.chemdata_generators.merge_syngraph"
    )
    def test_generate_not_by_record(self, mock_merge_syngraph):
        mock_merge_syngraph.return_value = BipartiteSynGraph([])
        result = self.generator.generate(
            self.sample_data_container, by_record=False, ce_label="CE"
        )
        self.assertIsInstance(result, BipartiteSynGraph)
        mock_merge_syngraph.assert_called_once()

    @patch("noctis.data_transformation.postprocessing.chemdata_generators.logger")
    def test_get_ce_label_with_none(self, mock_logger):
        settings.nodes.chemical_equation = "TEST_CE"
        result = self.generator._get_ce_label(None)
        self.assertEqual(result, "TEST_CE")
        mock_logger.warning.assert_called_once()

    def test_get_ce_label_with_value(self):
        result = self.generator._get_ce_label("CUSTOM_CE")
        self.assertEqual(result, "CUSTOM_CE")

    def test_process_records(self):
        result = self.generator._process_records(self.sample_data_container, "CE")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        for syngraph in result:
            self.assertIsInstance(syngraph, BipartiteSynGraph)

    @patch("noctis.data_transformation.postprocessing.chemdata_generators.logger")
    def test_process_records_no_ce_nodes(self, mock_logger):
        data_container = DataContainer(
            records={0: GraphRecord(nodes=[self.sample_nodes[2]], relationships=[])}
        )
        result = self.generator._process_records(data_container, "CE")
        self.assertEqual(result, [])
        mock_logger.warning.assert_called_once()

    def test_process_record(self):
        result = self.generator._process_record(
            [self.sample_nodes[0], self.sample_nodes[1]]
        )
        self.assertIsInstance(result, BipartiteSynGraph)

    def test_process_record_no_smiles(self):
        with self.assertRaises(NoReactionSmilesError):
            self.generator._process_record([self.sample_nodes[2]])

    def test_extract_reaction_smiles(self):
        result = self.generator._extract_reaction_smiles(
            [self.sample_nodes[0], self.sample_nodes[1]]
        )
        expected = [
            {"query_id": 0, "output_string": "C>>CC"},
            {"query_id": 1, "output_string": "CC>>CCC"},
        ]
        self.assertEqual(result, expected)

    @patch("noctis.data_transformation.postprocessing.chemdata_generators.logger")
    def test_extract_reaction_smiles_no_smiles(self, mock_logger):
        result = self.generator._extract_reaction_smiles([self.sample_nodes[2]])
        self.assertEqual(result, [])
        mock_logger.warning.assert_called_once()


class TestChemDataGeneratorFactory:
    def test_register_and_get_generator(self):
        print(ChemDataGeneratorFactory.generators)
        assert isinstance(
            ChemDataGeneratorFactory.get_generator("pandas"), PandasGenerator
        )
        assert isinstance(
            ChemDataGeneratorFactory.get_generator("networkx"), NetworkXGenerator
        )
        assert isinstance(
            ChemDataGeneratorFactory.get_generator("reaction_strings"),
            ReactionStringGenerator,
        )
        assert isinstance(
            ChemDataGeneratorFactory.get_generator("syngraph"), SyngraphGenerator
        )

    def test_get_unknown_generator(self):
        with pytest.raises(ValueError):
            ChemDataGeneratorFactory.get_generator("unknown")
