from abc import ABC, abstractmethod
import pandas as pd
import networkx as nx

from noctis.data_architecture.datamodel import (
    GraphRecord,
    Node,
    Relationship,
)
from noctis.data_transformation.data_styles.dataframe_stylers import PandasExportStyle

from linchemin.cgu.syngraph import BipartiteSynGraph

from typing import Union, Any
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
    """
    Abstract base class for chemical data generators.

    Attributes:
        reaction_formats (set[str]): Set of supported reaction formats.
    """

    reaction_formats = {"smiles", "smarts", "rxn_blockV3K", "rxn_blockV2K"}

    @abstractmethod
    def generate(
        self,
        records: list[GraphRecord],
        with_record_id: bool,
        ce_label: Optional[str],
    ) -> Any:
        """
        Abstract method to generate chemical data.

        Args:
            records (list[GraphRecord]): List of graph records indexed by an integer ID.
            with_record_id (bool): Flag to include record IDs in the output.
            ce_label (Optional[str]): Label for chemical equations.

        Returns:
            Any: The generated chemical data.
        """
        pass


class ChemDataGeneratorFactory:
    """
    Factory class for registering and retrieving chemical data generators.

    Attributes:
        generators (dict[str, type[ChemDataGeneratorInterface]]): Dictionary of registered generator classes.
    """

    generators = {}

    @classmethod
    def get_generator(cls, generator_type: str) -> ChemDataGeneratorInterface:
        """
        Retrieve a generator instance by type.

        Args:
            generator_type (str): Type of the generator to retrieve.

        Returns:
            ChemDataGeneratorInterface: Instance of the requested generator type.

        Raises:
            ValueError: If the generator type is unknown.
        """
        generator_class = cls.generators.get(generator_type)
        if generator_class:
            return generator_class()
        raise ValueError(f"Unknown generator type: {generator_type}")

    @classmethod
    def register_generator(
        cls, generator_type: str
    ) -> Callable[[type[ChemDataGeneratorInterface]], type[ChemDataGeneratorInterface]]:
        """
        Register a new generator class.

        Args:
            generator_type (str): Type of the generator to register.

        Returns:
            Callable: A decorator for registering the generator class.
        """

        def decorator(
            generator: type[ChemDataGeneratorInterface],
        ) -> type[ChemDataGeneratorInterface]:
            cls.generators[generator_type] = generator
            return generator

        return decorator

    @classmethod
    def get_available_formats(cls) -> list[str]:
        """Return a list of all registered generator types."""
        return list(cls.generators.keys())

    @classmethod
    def get_available_reaction_formats(cls) -> list[str]:
        """Return a list of available reaction formats from the first registered generator."""
        if not cls.generators:
            return []

        first_generator_type = next(iter(cls.generators))
        first_generator = cls.generators[first_generator_type]()

        return list(first_generator.reaction_formats)


@ChemDataGeneratorFactory.register_generator("pandas")
class PandasGenerator(ChemDataGeneratorInterface):
    """
    Generator class for exporting chemical data to pandas DataFrames.
    """

    def generate(
        self, records: list[GraphRecord], with_record_id: bool, ce_label: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate pandas DataFrames from graph records.

        Args:
            records (list[GraphRecord]): List of graph records indexed by an integer ID.
            with_record_id (bool): Flag to include record IDs in the output.
            ce_label (str): Label for chemical equations.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Tuple containing nodes and relationships DataFrames.
        """
        nodes_data, relationships_data = self._process_records(records, with_record_id)
        return self._concatenate_dataframes(nodes_data, relationships_data)

    def _process_records(
        self, records: list[GraphRecord], with_record_id: bool
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """
        Process records to extract nodes and relationships data.

        Args:
            records (list[GraphRecord]): List of graph records indexed by an integer ID.
            with_record_id (bool): Flag to include record IDs in the output.

        Returns:
            tuple[list[pd.DataFrame], list[pd.DataFrame]]: Lists of nodes and relationships DataFrames.
        """
        nodes_data = []
        relationships_data = []
        for record_id, record in enumerate(records):
            nodes_df = self._process_nodes(record, record_id, with_record_id)
            relationships_df = self._process_relationships(
                record, record_id, with_record_id
            )
            nodes_data.append(nodes_df)
            relationships_data.append(relationships_df)
        return nodes_data, relationships_data

    def _process_nodes(
        self, record: GraphRecord, record_id: int, with_record_id: bool
    ) -> pd.DataFrame:
        """
        Process nodes from a graph record.

        Args:
            record (GraphRecord): A graph record containing nodes.
            record_id (int): ID of the record.
            with_record_id (bool): Flag to include record IDs in the output.

        Returns:
            pd.DataFrame: DataFrame containing nodes data.
        """
        nodes_df = self._style_nodes(record.nodes)
        return self._add_record_id_if_needed(nodes_df, record_id, with_record_id)

    def _process_relationships(
        self, record: GraphRecord, record_id: int, with_record_id: bool
    ) -> pd.DataFrame:
        """
        Process relationships from a graph record.

        Args:
            record (GraphRecord): A graph record containing relationships.
            record_id (int): ID of the record.
            with_record_id (bool): Flag to include record IDs in the output.

        Returns:
            pd.DataFrame: DataFrame containing relationships data.
        """
        relationships_df = self._style_relationships(record.relationships)
        return self._add_record_id_if_needed(
            relationships_df, record_id, with_record_id
        )

    @staticmethod
    def _add_record_id_if_needed(
        df: pd.DataFrame, record_id: int, with_record_id: bool
    ) -> pd.DataFrame:
        """
        Add record ID to a DataFrame if needed.

        Args:
            df (pd.DataFrame): DataFrame to modify.
            record_id (int): ID of the record.
            with_record_id (bool): Flag to include record IDs in the output.

        Returns:
            pd.DataFrame: Modified DataFrame with record ID.
        """
        if with_record_id:
            df["record_id"] = record_id
        return df

    @staticmethod
    def _style_nodes(nodes: list[Node]) -> pd.DataFrame:
        """
        Style nodes data into a DataFrame.

        Args:
            nodes (list[Node]): List of Node objects.

        Returns:
            pd.DataFrame: Styled DataFrame containing nodes data.
        """
        styled_df = PandasExportStyle.export_nodes({"all_nodes": nodes})
        return styled_df["all_nodes"]

    @staticmethod
    def _style_relationships(relationships: list[Relationship]) -> pd.DataFrame:
        """
        Style relationships data into a DataFrame.

        Args:
            relationships (list[Relationship]): List of Relationship objects.

        Returns:
            pd.DataFrame: Styled DataFrame containing relationships data.
        """
        styled_df = PandasExportStyle.export_relationships(
            {"all_relationships": relationships}
        )
        return styled_df["all_relationships"]

    @staticmethod
    def _concatenate_dataframes(
        nodes_data: list[pd.DataFrame], relationships_data: list[pd.DataFrame]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Concatenate lists of DataFrames into single DataFrames.

        Args:
            nodes_data (list[pd.DataFrame]): List of nodes DataFrames.
            relationships_data (list[pd.DataFrame]): List of relationships DataFrames.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: Concatenated nodes and relationships DataFrames.
        """
        return pd.concat(nodes_data), pd.concat(relationships_data)


@ChemDataGeneratorFactory.register_generator("networkx")
class NetworkXGenerator(ChemDataGeneratorInterface):
    """
    Generator class for exporting chemical data to NetworkX graphs.
    """

    def generate(
        self,
        records: list[GraphRecord],
        with_record_id: bool,
        ce_label: Optional[str] = None,
    ) -> Union[nx.Graph, list[nx.Graph]]:
        """
        Generate NetworkX graphs from graph records.

        Args:
            records (list[GraphRecord]): List of graph records indexed by an integer ID.
            with_record_id (bool): Flag to include record IDs in the output.
            ce_label (Optional[str]): Label for chemical equations.

        Returns:
            Union[nx.Graph, list[nx.Graph]]: NetworkX graph or a list of graphs.
        """
        graphs = self._process_records(records)
        return graphs if with_record_id else self._compose_all_graphs(graphs)

    def _process_records(self, records: list[GraphRecord]) -> list[nx.Graph]:
        """
        Process records to create NetworkX graphs.

        Args:
            records (list[GraphRecord]): List of graph records indexed by an integer ID.

        Returns:
            list[nx.Graph]: List of NetworkX graphs.
        """
        return [self._process_record(record) for record in records]

    def _process_record(self, record: GraphRecord) -> nx.Graph:
        """
        Process a single graph record into a NetworkX graph.

        Args:
            record (GraphRecord): A graph record.

        Returns:
            nx.Graph: NetworkX graph representing the record.
        """
        G = nx.DiGraph()
        self._add_nodes_to_graph(G, record.nodes)
        if record.relationships:
            self._add_relationships_to_graph(G, record.relationships)
        return G

    def _add_nodes_to_graph(self, G: nx.Graph, nodes: list[Node]) -> None:
        """
        Add nodes to a NetworkX graph.

        Args:
            G (nx.Graph): NetworkX graph.
            nodes (list[Node]): List of Node objects to add.
        """
        for node in nodes:
            G.add_node(node.uid, **self._create_node_attributes(node))

    def _add_relationships_to_graph(
        self, G: nx.Graph, relationships: list[Relationship]
    ) -> None:
        """
        Add relationships to a NetworkX graph.

        Args:
            G (nx.Graph): NetworkX graph.
            relationships (list[Relationship]): List of Relationship objects to add.
        """
        for relationship in relationships:
            G.add_edge(
                relationship.start_node.uid,
                relationship.end_node.uid,
                **self._create_relationship_attributes(relationship),
            )

    @staticmethod
    def _create_node_attributes(node: Node) -> dict[str, Any]:
        """
        Create attributes for a node in a NetworkX graph.

        Args:
            node (Node): Node object.

        Returns:
            dict[str, Any]: Dictionary of node attributes.
        """
        return {
            "node_label": node.node_label,
            "properties": node.properties,
        }

    @staticmethod
    def _create_relationship_attributes(relationship: Relationship) -> dict[str, Any]:
        """
        Create attributes for a relationship in a NetworkX graph.

        Args:
            relationship (Relationship): Relationship object.

        Returns:
            dict[str, Any]: Dictionary of relationship attributes.
        """
        return {
            "relationship_type": relationship.relationship_type,
            "properties": relationship.properties,
        }

    @staticmethod
    def _compose_all_graphs(graphs: list[nx.Graph]) -> nx.Graph:
        """
        Compose multiple NetworkX graphs into a single graph.

        Args:
            graphs (list[nx.Graph]): List of NetworkX graphs.

        Returns:
            nx.Graph: Composed NetworkX graph.
        """
        return nx.compose_all(graphs)


@ChemDataGeneratorFactory.register_generator("reaction_strings")
class ReactionStringGenerator(ChemDataGeneratorInterface):
    """
    Generator class for exporting chemical data to reaction strings.
    """

    def generate(
        self,
        records: list[GraphRecord],
        with_record_id: bool,
        ce_label: Optional[str] = None,
    ) -> Union[dict[str, list[str]], list[dict]]:
        """
        Generate reaction strings from graph records.

        Args:
            records (list[GraphRecord]): List of graph records indexed by an integer ID.
            with_record_id (bool): Flag to include record IDs in the output.
            ce_label (Optional[str]): Label for chemical equations.

        Returns:
            Union[dict[str, list[str]], list[dict[str, list[str]]]]: Reaction strings or a list of dictionaries.
        """
        ce_label = self._get_ce_label(ce_label)
        reaction_strings = self._process_records(records, ce_label)
        return (
            reaction_strings
            if with_record_id
            else self._merge_dict_lists(reaction_strings)
        )

    @staticmethod
    def _get_ce_label(ce_label: Optional[str]) -> str:
        """
        Get the chemical equation label, using settings if necessary.

        Args:
            ce_label (Optional[str]): Label for chemical equations.

        Returns:
            str: Chemical equation label.
        """
        if ce_label is None:
            ce_label = settings.nodes.chemical_equation
            logger.warning(
                f"Label from settings {settings.nodes.chemical_equation} has been used"
            )
        return ce_label

    def _process_records(
        self, records: list[GraphRecord], ce_label: str
    ) -> list[dict[str, list[str]]]:
        """
        Process records to extract reaction strings.

        Args:
            records (list[GraphRecord]): List of graph records indexed by an integer ID.
            ce_label (str): Label for chemical equations.

        Returns:
            list[dict[str, list[str]]]: List of dictionaries containing reaction strings.
        """
        reaction_strings = []
        for record_id, record in enumerate(records):
            chem_equation_nodes = record.get_nodes_with_label(ce_label)
            if not chem_equation_nodes:
                logger.warning(
                    f"No chemical equation with label found in record {record_id}"
                )
            reaction_strings.append(self._process_record(chem_equation_nodes))
        return reaction_strings

    def _process_record(self, chem_equation_nodes: list[Node]) -> dict[str : list[str]]:
        """
        Process chemical equation nodes to extract reaction strings.

        Args:
            chem_equation_nodes (list[Node]): List of chemical equation nodes.

        Returns:
            dict[str, list[str]]: Dictionary containing reaction strings.
        """
        record_dict = defaultdict(list)
        for node in chem_equation_nodes:
            self._add_node_reactions_to_record(node, record_dict)
        return dict(record_dict)

    def _add_node_reactions_to_record(self, node: Node, record_dict: defaultdict):
        """
        Add node reactions to the record dictionary.

        Args:
            node (Node): Node object.
            record_dict (defaultdict[str, list[str]]): Dictionary to store reaction strings.
        """
        node_dict = self._process_node(node)
        for key, value_list in node_dict.items():
            record_dict[key].append(value_list)

    def _process_node(self, node: Node) -> dict[str, str]:
        """
        Process a node to extract reaction strings.

        Args:
            node (Node): Node object.

        Returns:
            dict[str, str]: Dictionary containing reaction strings.
        """
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
        """
        Merge a list of dictionaries into a single dictionary.

        Args:
            dict_list (list[dict[str, list[Any]]]): List of dictionaries to merge.

        Returns:
            dict[str, list[Any]]: Merged dictionary.
        """
        merged = defaultdict(list)
        for d in dict_list:
            for key, value_list in d.items():
                merged[key].extend(value_list)
        return dict(merged)


@ChemDataGeneratorFactory.register_generator("syngraph")
class SyngraphGenerator(ChemDataGeneratorInterface):
    """
    Generator class for exporting chemical data to synthetic graphs.
    """

    def generate(
        self, records: list[GraphRecord], with_record_id: bool, ce_label: Optional[str]
    ) -> Union[BipartiteSynGraph, list[BipartiteSynGraph]]:
        """
        Generate synthetic graphs from graph records.

        Args:
            records (dict[int, GraphRecord]): Dictionary of graph records indexed by an integer ID.
            with_record_id (bool): Flag to include record IDs in the output.
            ce_label (Optional[str]): Label for chemical equations.

        Returns:
            Union[BipartiteSynGraph, list[BipartiteSynGraph]]: Synthetic graph or a list of synthetic graphs.
        """
        ce_label = self._get_ce_label(ce_label)
        syngraphs = self._process_records(records, ce_label)
        return syngraphs if with_record_id else merge_syngraph(syngraphs)

    @staticmethod
    def _get_ce_label(ce_label: Optional[str]) -> str:
        """
        Get the chemical equation label, using settings if necessary.

        Args:
            ce_label (Optional[str]): Label for chemical equations.

        Returns:
            str: Chemical equation label.
        """
        if ce_label is None:
            ce_label = settings.nodes.chemical_equation
            logger.warning(
                f"Label from settings {settings.nodes.chemical_equation} has been used"
            )
        return ce_label

    def _process_records(
        self, records: list[GraphRecord], ce_label: str
    ) -> list[BipartiteSynGraph]:
        """
        Process records to create synthetic graphs.

        Args:
            records (dict[int, GraphRecord]): Dictionary of graph records indexed by an integer ID.
            ce_label (str): Label for chemical equations.

        Returns:
            list[BipartiteSynGraph]: List of synthetic graphs.
        """
        syngraphs = []
        for record_id, record in enumerate(records):
            chem_equation_nodes = record.get_nodes_with_label(ce_label)
            if not chem_equation_nodes:
                logger.warning(
                    f"No chemical equations with label {ce_label} found in record {record_id}"
                )
            else:
                syngraphs.append(self._process_record(chem_equation_nodes))
        return syngraphs

    def _process_record(self, chem_equation_nodes: list[Node]) -> BipartiteSynGraph:
        """
        Process chemical equation nodes to create a synthetic graph.

        Args:
            chem_equation_nodes (list[Node]): List of chemical equation nodes.

        Returns:
            BipartiteSynGraph: Synthetic graph.
        """
        reaction_smiles = self._extract_reaction_smiles(chem_equation_nodes)
        if not reaction_smiles:
            raise NoReactionSmilesError(
                "No reaction SMILES found in chemical equation nodes"
            )
        return BipartiteSynGraph(reaction_smiles)

    @staticmethod
    def _extract_reaction_smiles(chem_equation_nodes: list[Node]) -> list[dict]:
        """
        Extract reaction SMILES from chemical equation nodes.

        Args:
            chem_equation_nodes (list[Node]): List of chemical equation nodes.

        Returns:
            list[dict]: List of dictionaries containing reaction SMILES.
        """
        reaction_smiles = []
        for node_id, node in enumerate(chem_equation_nodes):
            smiles = node.properties.get("smiles")
            if smiles:
                reaction_smiles.append({"query_id": node_id, "output_string": smiles})
            else:
                logger.warning(f"No smiles found for node {node.uid}.")
        return reaction_smiles
