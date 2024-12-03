from abc import ABC, abstractmethod
import pandas as pd
import networkx as nx

from noctis.data_architecture.datamodel import (
    DataContainer,
    GraphRecord,
    Node,
    Relationship,
)
from noctis.data_transformation.data_styles.dataframe_stylers import PandasExportStyle
from linchemin.cgu.syngraph import SynGraph, BipartiteSynGraph
from typing import Union
from noctis import settings

from collections import defaultdict
from typing import Optional, Callable

from linchemin.cgu.syngraph_operations import merge_syngraph
from noctis.utilities import console_logger

logger = console_logger(__name__)


class NoReactionSmilesError(Exception):
    """Raised when no reaction SMILES are found in chemical equation nodes"""

    pass


class ChemDataGeneratorInterface(ABC):
    reaction_formats = {"smiles", "smarts", "rxn_blockV3K", "rxn_blockV2K"}

    @abstractmethod
    def generate(
        self, data_container: DataContainer, by_record: bool, ce_label: Optional[str]
    ):
        pass


class ChemDataGeneratorFactory:
    generators = {}

    @classmethod
    def get_generator(cls, generator_type: str) -> ChemDataGeneratorInterface:
        generator_class = cls.generators.get(generator_type)
        if generator_class:
            return generator_class()
        raise ValueError(f"Unknown generator type: {generator_type}")

    @classmethod
    def register_generator(
        cls, generator_type: str
    ) -> Callable[[type[ChemDataGeneratorInterface]], type[ChemDataGeneratorInterface]]:
        def decorator(
            generator: type[ChemDataGeneratorInterface],
        ) -> type[ChemDataGeneratorInterface]:
            cls.generators[generator_type] = generator
            return generator

        return decorator


@ChemDataGeneratorFactory.register_generator("pandas")
class PandasGenerator(ChemDataGeneratorInterface):
    def generate(
        self, data_container, by_record, ce_label: str = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        nodes_data, relationships_data = self._process_records(
            data_container, by_record
        )
        return self._concatenate_dataframes(nodes_data, relationships_data)

    def _process_records(self, data_container, by_record):
        nodes_data = []
        relationships_data = []
        for record_id, record in data_container.records.items():
            nodes_df = self._process_nodes(record, record_id, by_record)
            relationships_df = self._process_relationships(record, record_id, by_record)
            nodes_data.append(nodes_df)
            relationships_data.append(relationships_df)
        return nodes_data, relationships_data

    def _process_nodes(self, record, record_id, by_record):
        nodes_df = self._style_nodes(record.nodes)
        return self._add_record_id_if_needed(nodes_df, record_id, by_record)

    def _process_relationships(self, record, record_id, by_record):
        relationships_df = self._style_relationships(record.relationships)
        return self._add_record_id_if_needed(relationships_df, record_id, by_record)

    @staticmethod
    def _add_record_id_if_needed(df, record_id, by_record):
        if by_record:
            df["record_id"] = record_id
        return df

    @staticmethod
    def _style_nodes(nodes: list[Node]) -> pd.DataFrame:
        styled_df = PandasExportStyle.export_nodes({"all_nodes": nodes})
        return styled_df["all_nodes"]

    @staticmethod
    def _style_relationships(relationships: list[Relationship]) -> pd.DataFrame:
        styled_df = PandasExportStyle.export_relationships(
            {"all_relationships": relationships}
        )
        return styled_df["all_relationships"]

    @staticmethod
    def _concatenate_dataframes(nodes_data, relationships_data):
        return pd.concat(nodes_data), pd.concat(relationships_data)


@ChemDataGeneratorFactory.register_generator("networkx")
class NetworkXGenerator(ChemDataGeneratorInterface):
    def generate(
        self, data_container, by_record, ce_label: str = None
    ) -> Union[nx.Graph, list[nx.Graph]]:
        graphs = self._process_records(data_container)
        return graphs if by_record else self._compose_all_graphs(graphs)

    def _process_records(self, data_container):
        return [
            self._process_record(record) for record in data_container.records.values()
        ]

    def _process_record(self, record: GraphRecord) -> nx.Graph:
        G = nx.DiGraph()
        self._add_nodes_to_graph(G, record.nodes)
        self._add_relationships_to_graph(G, record.relationships)
        return G

    def _add_nodes_to_graph(self, G, nodes):
        for node in nodes:
            G.add_node(node.uid, **self._create_node_attributes(node))

    def _add_relationships_to_graph(self, G, relationships):
        for relationship in relationships:
            G.add_edge(
                relationship.start_node.uid,
                relationship.end_node.uid,
                **self._create_relationship_attributes(relationship),
            )

    @staticmethod
    def _create_node_attributes(node: Node):
        return {
            "node_label": node.node_label,
            "properties": node.properties,
        }

    @staticmethod
    def _create_relationship_attributes(relationship: Relationship):
        return {
            "relationship_type": relationship.relationship_type,
            "properties": relationship.properties,
        }

    @staticmethod
    def _compose_all_graphs(graphs: list[nx.Graph]) -> nx.Graph:
        return nx.compose_all(graphs)


@ChemDataGeneratorFactory.register_generator("reaction_strings")
class ReactionStringGenerator(ChemDataGeneratorInterface):
    def generate(
        self, data_container, by_record, ce_label: str = None
    ) -> Union[dict[str, list[str]], list[dict]]:
        ce_label = self._get_ce_label(ce_label)
        reaction_strings = self._process_records(data_container, ce_label)
        return (
            reaction_strings if by_record else self._merge_dict_lists(reaction_strings)
        )

    @staticmethod
    def _get_ce_label(ce_label):
        if ce_label is None:
            ce_label = settings.nodes.chemical_equation
            logger.warning(
                f"Label from settings {settings.nodes.chemical_equation} has been used"
            )
        return ce_label

    def _process_records(self, data_container, ce_label):
        reaction_strings = []
        for record_id, record in data_container.records.items():
            chem_equation_nodes = record.get_nodes_with_label(ce_label)
            if not chem_equation_nodes:
                logger.warning(
                    f"No chemical equation with label found in record {record_id}"
                )
            reaction_strings.append(self._process_record(chem_equation_nodes))
        return reaction_strings

    def _process_record(self, chem_equation_nodes: list[Node]) -> dict[str : list[str]]:
        record_dict = defaultdict(list)
        for node in chem_equation_nodes:
            self._add_node_reactions_to_record(node, record_dict)
        return dict(record_dict)

    def _add_node_reactions_to_record(self, node: Node, record_dict: defaultdict):
        node_dict = self._process_node(node)
        for key, value_list in node_dict.items():
            record_dict[key].append(value_list)

    def _process_node(self, node: Node) -> dict[str, str]:
        reaction_strings = {
            key: value
            for key, value in node.properties.items()
            if key in self.reaction_formats
        }
        if not reaction_strings:
            logger.warning(
                f"No reaction string found for node {node.uid}. Available properties: {list(node.properties.keys())}"
            )
        return reaction_strings

    @staticmethod
    def _merge_dict_lists(
        dict_list: list[dict[str, list[any]]]
    ) -> dict[str, list[any]]:
        merged = defaultdict(list)
        for d in dict_list:
            for key, value_list in d.items():
                merged[key].extend(value_list)
        return dict(merged)


@ChemDataGeneratorFactory.register_generator("syngraph")
class SyngraphGenerator(ChemDataGeneratorInterface):
    def generate(
        self, data_container, by_record, ce_label
    ) -> Union[BipartiteSynGraph, list[BipartiteSynGraph]]:
        ce_label = self._get_ce_label(ce_label)
        syngraphs = self._process_records(data_container, ce_label)
        return syngraphs if by_record else merge_syngraph(syngraphs)

    @staticmethod
    def _get_ce_label(ce_label):
        if ce_label is None:
            ce_label = settings.nodes.chemical_equation
            logger.warning(
                f"Label from settings {settings.nodes.chemical_equation} has been used"
            )
        return ce_label

    def _process_records(self, data_container, ce_label):
        syngraphs = []
        for record_id, record in data_container.records.items():
            chem_equation_nodes = record.get_nodes_with_label(ce_label)
            if not chem_equation_nodes:
                logger.warning(
                    f"No chemical equations with label {ce_label} found in record {record_id}"
                )
            else:
                syngraphs.append(self._process_record(chem_equation_nodes))
        return syngraphs

    def _process_record(self, chem_equation_nodes: list[Node]) -> BipartiteSynGraph:
        reaction_smiles = self._extract_reaction_smiles(chem_equation_nodes)
        if not reaction_smiles:
            raise NoReactionSmilesError(
                "No reaction SMILES found in chemical equation nodes"
            )
        return BipartiteSynGraph(reaction_smiles)

    @staticmethod
    def _extract_reaction_smiles(chem_equation_nodes: list[Node]) -> list[dict]:
        reaction_smiles = []
        for node_id, node in enumerate(chem_equation_nodes):
            smiles = node.properties.get("smiles")
            if smiles:
                reaction_smiles.append({"query_id": node_id, "output_string": smiles})
            else:
                logger.warning(f"No smiles found for node {node.uid}.")
        return reaction_smiles


class ChemDataGenerator:
    factory = ChemDataGeneratorFactory()

    def generate_data(
        self,
        data_container,
        format_type: str,
        by_record: bool = False,
        ce_label: Optional[str] = None,
    ):
        generator = self.factory.get_generator(format_type)
        return generator.generate(data_container, by_record, ce_label)
