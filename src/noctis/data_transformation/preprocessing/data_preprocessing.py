from collections.abc import Iterator
from dataclasses import dataclass, fields, asdict
from dask.distributed import Client, as_completed
from typing import Type, Callable, Optional
import yaml
import pandas as pd
import dask.dataframe as dd
import os
import numpy as np
from linchemin.cgu.syngraph import (
    SynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
    BipartiteSynGraph,
)
from noctis.data_architecture.datacontainer import DataContainer
from noctis.data_architecture.datamodel import (
    Relationship,
    Node,
)
from noctis.data_transformation.preprocessing.utilities import (
    create_data_container,
)
from linchemin.cgu.syngraph_operations import extract_reactions_from_syngraph
from abc import ABC, abstractmethod

from noctis.data_transformation.neo4j.stylers import Neo4jImportStyle
from pathlib import Path

from noctis.utilities import console_logger
from typing import Union

from tqdm import tqdm
import textwrap

from noctis.data_transformation.preprocessing.utilities import (
    _save_dataframes_to_partition_csv,
    _save_list_to_partition_csv,
    _update_partition_dict_with_row,
    _merge_partition_files,
    _delete_tmp_folder,
    dict_to_list,
)

from noctis.data_transformation.preprocessing.graph_expander import GraphExpander
from noctis.data_architecture.graph_schema import GraphSchema

logger = console_logger(__name__)


class EmptyHeaderError(Exception):
    pass


class NoPreprocessorError(Exception):
    pass


class NoChemicalStringError(Exception):
    pass


class MissingUIDError(Exception):
    """Custom error for when a UID is missing for a node."""

    def __init__(self, label):
        self.message = f"Missing 'uid' for node with label '{label}'"
        super().__init__(self.message)


def validate_single_char(value):
    if value is not None and len(value) != 1:
        raise ValueError(
            "Invalid lineterminator: must be a single character or None. "
            "For Windows-style line endings ('\\r\\n'), keep lineterminator as None. "
            f"Received: {repr(value)}"
        )
    return value


@dataclass
class PreprocessorConfig:
    """Configuration for preprocessing chemical data."""

    inp_chem_format: Optional[str] = "smiles"
    out_chem_format: Optional[str] = None
    validation: Optional[bool] = True
    output_folder: Optional[str] = "output"
    tmp_folder: Optional[str] = None
    delete_tmp: Optional[bool] = True
    prefix: Optional[str] = None
    delimiter: Optional[str] = ","
    lineterminator: Optional[str] = None
    quotechar: Optional[str] = '"'
    blocksize: Optional[int] = 600000  # Kb; used by dask in run_parallel
    chunksize: Optional[int] = 10000  # number of lines; used in run_serial
    nrows: Optional[int] = None  # only for serial run

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None:
                setattr(self, f.name, f.default)

        if self.tmp_folder is None:
            self.tmp_folder = os.path.join(self.output_folder, "tmp")

        if self.out_chem_format is None:
            self.out_chem_format = self.inp_chem_format

        validate_single_char(self.lineterminator)
        if self.nrows is not None and self.parallel:
            logger.warning(
                "The 'nrows' parameter is only applicable for serial processing. "
                "It will be ignored in parallel mode."
            )

    @classmethod
    def build_from_yaml(cls, file_path: str) -> "PreprocessorConfig":
        return _load_config_from_yaml(file_path)

    def save_to_yaml(self, file_path: str) -> None:
        try:
            with open(file_path, "w") as file:
                yaml.safe_dump(asdict(self), file, sort_keys=False)
                logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving configuration to YAML: {e}")
            raise NoPreprocessorError(f"Error saving configuration to YAML: {e}")


def _load_config_from_yaml(yaml_file_path: str) -> PreprocessorConfig:
    try:
        with open(yaml_file_path) as file:
            config_data = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {yaml_file_path}")
        raise NoPreprocessorError(f"Configuration file not found: {yaml_file_path}")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        raise NoPreprocessorError(f"Unexpected error loading configuration: {e}")
    if not isinstance(config_data, dict):
        logger.error("Configuration file does not contain a dictionary")
        raise NoPreprocessorError("Configuration file does not contain a dictionary")

    return PreprocessorConfig(**config_data)


class PandasRowPreprocessorBase(ABC):
    """
    Abstract base class for preprocessing rows of a pandas DataFrame
    according to a specified graph schema and configuration.

    Attributes:
        schema (GraphSchema): The schema defining the structure of nodes and relationships.
        config (PreprocessorConfig): Configuration settings for preprocessing.
    """

    def __init__(self, schema: GraphSchema, config: PreprocessorConfig):
        """
        Initialize the preprocessor with a graph schema and configuration.

        Args:
            schema (GraphSchema): The schema defining the nodes and relationships.
            config (PreprocessorConfig): Configuration settings for preprocessing.
        """
        self.schema = schema
        self.config = config

    def _process_row(self, row: pd.Series) -> tuple[dict, dict, Union[str, None]]:
        """
        Process a single row from a DataFrame, extracting nodes and relationships.

        Args:
            row (pd.Series): A row of data from a pandas DataFrame.

        Returns:
            tuple: A tuple containing dictionaries of nodes and relationships, and
                   a string indicating a failed chemical string, or None if successful.
        """
        split_row = self._split_row_by_node_types(row)
        ge = GraphExpander(self.schema)
        try:
            nodes, relationships = ge.expand_reaction_step(
                split_row,
                self.config.inp_chem_format,
                self.config.out_chem_format,
                self.config.validation,
            )
            return nodes, relationships, None
        except Exception as e:
            reaction_string = split_row[self.schema.base_nodes["chemical_equation"]][
                "properties"
            ][self.config.inp_chem_format]
            logger.error(
                f"Row with reaction string {reaction_string} cannot be expanded into nodes and relationships. Error: {str(e)}"
            )
            return {}, {}, reaction_string

    def _split_row_by_node_types(self, row: pd.Series) -> dict[str, dict]:
        """
        Split a row into a dictionary organized by node types.

        Args:
            row (pd.Series): A row of data from a pandas DataFrame.

        Returns:
            dict: A dictionary mapping node labels to their properties and UID.
        """
        node_data = {}

        for column, value in row.items():
            parts = column.split(".", 1)

            if len(parts) == 2:
                label, property_name = parts
                if label not in node_data:
                    node_data[label] = {"properties": {}}

                if property_name == "uid":
                    node_data[label]["uid"] = value
                else:
                    node_data[label]["properties"][property_name] = value

        return node_data

    def _validate_the_header(self, header: list[str]) -> None:
        """
        Validate the header of a DataFrame to ensure required fields are present.

        Args:
            header (list): List of column names from the DataFrame.

        Raises:
            EmptyHeaderError: If the header is empty.
            NoChemicalStringError: If required chemical string fields are missing.
            MissingUIDError: If UID fields are missing for extra schema nodes.
        """
        chemical_equation_label = self.schema.base_nodes["chemical_equation"]
        chemical_equation_field = (
            f"{chemical_equation_label}.{self.config.inp_chem_format}"
        )
        if not header:
            error_msg = "Header is empty"
            logger.error(f"EmptyHeaderError: {error_msg}")
            raise EmptyHeaderError(error_msg)

        if chemical_equation_field not in header:
            error_msg = f"Missing required field: {chemical_equation_field}. Current header: {header}"
            logger.error(f"NoChemicalStringError: {error_msg}")
            raise NoChemicalStringError(error_msg)

        for header_field in header:
            parts = header_field.split(".", 1)
            if len(parts) != 2:
                logger.warning(
                    f"IncorrectHeaderFieldFormatWarning: Field '{header_field}' will be ignored during processing. Allowed format is LABEL.property"
                )
                continue

            label, property_name = parts

            if label == self.schema.base_nodes["molecule"]:
                logger.warning(
                    f"MoleculeNodeWarning: Field '{header_field}' will be ignored. Molecule nodes are reconstructed from '{chemical_equation_label}' nodes."
                )
                continue

            if (
                label not in self.schema.base_nodes.values()
                and label not in self.schema.extra_nodes.values()
            ):
                logger.warning(
                    f"FieldNotInSchemaWarning: Field '{header_field}' will be ignored during processing. Node '{label}' is not in the schema."
                )
                continue

        for node_label in self.schema.extra_nodes.values():
            uid_field = f"{node_label}.uid"
            if uid_field not in header:
                error_msg = f"Missing uid for node {node_label}. All extra schema nodes require uid."
                logger.error(f"MissingUIDError: {error_msg}")
                raise MissingUIDError(error_msg)


class CSVPreprocessor(PandasRowPreprocessorBase):
    """
    Preprocessor for handling CSV files, capable of processing data
    either in parallel or serially using Dask or pandas.

    Attributes:
        input_file (str): Path to the input CSV file.
        max_partitions (int): Maximum number of partitions for processing.
    """

    input_file: Optional[str] = None
    max_partitions: Optional[int] = None

    def run(
        self, input_file: str, parallel: bool, dask_client: Optional[Client] = None
    ) -> None:
        """
        Execute the preprocessing of a CSV file.

        Args:
            input_file (str): Path to the input CSV file.
            parallel (bool): Flag indicating whether to run in parallel mode.
            dask_client (Optional[Client]): Optional Dask client for parallel processing.
        """
        self.input_file = input_file

        header = self._read_header()

        self._validate_the_header(header)

        if parallel:
            self._run_parallel(dask_client)
        else:
            self._run_serial()

        self._merge_all_partition_files()

        if self.config.delete_tmp:
            _delete_tmp_folder(self.config.tmp_folder)

    def _merge_all_partition_files(self) -> None:
        """
        Merge all partition files into final output files.
        """
        combined_templates = []

        combined_templates.extend(
            [
                value.upper()
                for value in (
                    self.schema.get_nodes_labels()
                    + self.schema.get_relationships_types()
                )
            ]
        )

        combined_templates.append("failed_strings")
        combined_templates.append("empty_strings")

        for template in combined_templates:
            filename = f"{template}.csv"
            _merge_partition_files(
                filename,
                self.config.tmp_folder,
                self.config.output_folder,
                self.max_partitions,
                self.config.prefix,
            )

    def _read_header(self) -> list[str]:
        """
        Read the header from the input CSV file.

        Returns:
            list[str]: A list of column names from the CSV file.
        """
        df = pd.read_csv(
            self.input_file,
            delimiter=self.config.delimiter,
            quotechar=self.config.quotechar,
            lineterminator=self.config.lineterminator,
            nrows=1,
        )
        return df.columns.tolist()

    def _run_parallel(self, dask_client: Optional[Client]) -> None:
        """
        Run the preprocessing in parallel using Dask.

        Args:
            dask_client (Optional[Client]): Optional Dask client for parallel processing.
        """
        client_provided = dask_client is not None
        client = dask_client if client_provided else Client()

        ddf = dd.read_csv(
            self.input_file,
            blocksize=self.config.blocksize,
            delimiter=self.config.delimiter,
            lineterminator=self.config.lineterminator,
            quotechar=self.config.quotechar,
        )
        self.max_partitions = ddf.npartitions
        partition_sizes = ddf.map_partitions(len).compute()
        offsets = np.concatenate(([0], np.cumsum(partition_sizes)[:-1]))
        ddf_proc = ddf.map_partitions(
            lambda df, partition_info: self._process_partition(
                df, partition_info["number"], offsets
            ),
            meta=pd.DataFrame(),
        )

        delayed_list = ddf_proc.to_delayed()

        futures = client.compute(delayed_list)
        with tqdm(
            total=self.max_partitions, desc="Processing partitions in parallel"
        ) as pbar:
            for fut in as_completed(futures):
                fut.result()
                pbar.update(1)
        if client_provided is False:
            client.close()

    def _run_serial(self) -> None:
        """
        Run the preprocessing serially using pandas.
        """
        self.max_partitions = self._calculate_max_partitions_for_serial()
        with tqdm(
            total=self.max_partitions, desc="Processing partitions serially"
        ) as pbar:
            for partition_number, df_partition in enumerate(
                self._serial_partition_generator()
            ):
                self._process_partition(df_partition, partition_number)
                pbar.update(1)

    def _serial_partition_generator(self) -> Iterator[pd.DataFrame]:
        """
        Generate partitions of the CSV file for serial processing.

        Yields:
            pd.DataFrame: A DataFrame representing a partition of the CSV file.
        """
        if self.config.chunksize is None:
            yield pd.read_csv(
                self.input_file,
                delimiter=self.config.delimiter,
                quotechar=self.config.quotechar,
                lineterminator=self.config.lineterminator,
                nrows=self.config.nrows,
            )
        else:
            yield from pd.read_csv(
                self.input_file,
                chunksize=self.config.chunksize,
                delimiter=self.config.delimiter,
                quotechar=self.config.quotechar,
                lineterminator=self.config.lineterminator,
                nrows=self.config.nrows,
            )

    def _process_partition(
        self,
        df: pd.DataFrame,
        partition_number: int = None,
        offsets: Optional[list[int]] = None,
    ) -> None:
        """
        Process a single partition of the CSV file.

        Args:
            df (pd.DataFrame): The DataFrame representing a partition.
            partition_number (int): The partition number.
            offsets (Optional[list[int]]): List of offsets for partition indices.
        """
        nodes_partition = {}
        relationships_partition = {}
        failed_strings = []
        empty_reaction_strings = []
        for index, row in df.iterrows():
            nodes_row, relationships_row, failed_chem_string = self._process_row(row)

            # Calculate global_index
            if offsets is None:
                global_index = index + 2
            else:
                global_index = index + offsets[partition_number] + 2
            if failed_chem_string is None:
                # Successful processing
                _update_partition_dict_with_row(nodes_partition, nodes_row)
                _update_partition_dict_with_row(
                    relationships_partition, relationships_row
                )
            elif pd.isna(failed_chem_string):
                # Empty or invalid reaction
                empty_reaction_strings.append([global_index])
            else:
                # Failed processing
                failed_strings.append([failed_chem_string, global_index])

        df_nodes = Neo4jImportStyle.export_nodes(nodes_partition)
        df_relationships = Neo4jImportStyle.export_relationships(
            relationships_partition
        )
        _save_dataframes_to_partition_csv(
            df_nodes,
            df_relationships,
            graph_schema=self.schema,
            output_dir=self.config.tmp_folder,
            partition_num=partition_number,
        )
        _save_list_to_partition_csv(
            failed_strings,
            header=[
                f'{self.schema.base_nodes["chemical_equation"]}.{self.config.inp_chem_format}',
                "index",
            ],
            output_dir=self.config.tmp_folder,
            name="failed_strings",
            partition_num=partition_number,
        )
        _save_list_to_partition_csv(
            empty_reaction_strings,
            header=[
                "index",
            ],
            output_dir=self.config.tmp_folder,
            name="empty_strings",
            partition_num=partition_number,
        )

    def _calculate_max_partitions_for_serial(self) -> int:
        """
        Calculate the maximum number of partitions for serial processing.

        Returns:
            int: The maximum number of partitions for serial processing.
        """
        total_rows = sum(1 for _ in open(self.input_file)) - 1
        effective_rows = (
            min(total_rows, self.config.nrows - 1) if self.config.nrows else total_rows
        )
        return (effective_rows + self.config.chunksize - 1) // self.config.chunksize


class PythonObjectPreprocessorInterface(ABC):
    """
    Abstract base class for preprocessors that handle Python objects.
    Defines a common interface for running preprocessing tasks.

    Attributes:
        failed_strings (list): A list to store strings that failed during processing.
    """

    failed_strings: list[str] = []

    @abstractmethod
    def run(self, data: object) -> DataContainer:
        """
        Abstract method to run the preprocessing task on the given data.

        Args:
            data (object): The data to be processed.

        Returns:
            DataContainer: A container holding processed nodes and relationships.
        """
        pass


class ChemicalStringPreprocessorBase(ABC):
    """
    Base class for preprocessors that handle chemical strings.
    Provides methods for processing chemical reaction strings.

    Attributes:
        schema (GraphSchema): The schema defining the nodes and relationships.
        config (PreprocessorConfig): Configuration settings for preprocessing.
    """

    def __init__(self, schema: GraphSchema, config: PreprocessorConfig):
        """
        Initialize the preprocessor with a graph schema and configuration.

        Args:
            schema (GraphSchema): The schema defining nodes and relationships.
            config (PreprocessorConfig): Configuration settings for preprocessing.
        """
        self.schema = schema
        self.config = config

    def _process_reaction_string(
        self, reaction_string: str
    ) -> tuple[list[Node], list[Relationship], Union[None, str]]:
        """
        Process a chemical reaction string into nodes and relationships.

        Args:
            reaction_string (str): The chemical reaction string to be processed.

        Returns:
            tuple: A tuple containing lists of nodes and relationships, and
                   a string indicating a failed reaction string, or None if successful.
        """
        reaction_dict = self._build_reaction_string_dict(reaction_string)
        ge = GraphExpander(self.schema)
        try:
            nodes, relationships = ge.expand_reaction_step(
                reaction_dict,
                self.config.inp_chem_format,
                self.config.out_chem_format,
                self.config.validation,
            )
            return dict_to_list(nodes), dict_to_list(relationships), None
        except Exception as e:
            logger.error(
                f"Row with reaction string {reaction_string} cannot be expanded into nodes and relationships. Error: {str(e)}"
            )
            return [], [], reaction_string

    def _build_reaction_string_dict(self, reaction_string: str) -> dict:
        """
        Build a dictionary representation of a chemical reaction string.

        Args:
            reaction_string (str): The chemical reaction string.

        Returns:
            dict: A dictionary mapping the chemical equation label to its properties.
        """
        label = self.schema.base_nodes["chemical_equation"]
        return {label: {"properties": {self.config.inp_chem_format: reaction_string}}}


class PythonObjectPreprocessorFactory:
    """
    Factory class for creating preprocessors based on data type.
    Manages registration and retrieval of preprocessor classes.

    Attributes:
        preprocessors (dict): A dictionary mapping data types to preprocessor classes.
    """

    preprocessors: dict[str, Type[PythonObjectPreprocessorInterface]] = {}

    @classmethod
    def get_preprocessor(
        cls,
        data_type: str,
        schema: GraphSchema,
        config: PreprocessorConfig,
    ) -> PythonObjectPreprocessorInterface:
        """
        Retrieve a preprocessor class based on the data type.

        Args:
            data_type (str): The type of data to be processed.
            schema (GraphSchema): The schema defining nodes and relationships.
            config (PreprocessorConfig): Configuration settings for preprocessing.

        Returns:
            PythonObjectPreprocessorInterface: An instance of the preprocessor class.

        Raises:
            ValueError: If the data type is not supported.
        """
        preprocessor_class = cls.preprocessors.get(data_type)

        if preprocessor_class:
            return preprocessor_class(schema, config)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

    @classmethod
    def register_preprocessor(
        cls, data_type: str
    ) -> Callable[
        [Type[PythonObjectPreprocessorInterface]],
        Type[PythonObjectPreprocessorInterface],
    ]:
        """
        Decorator to register a preprocessor class for a specific data type.

        Args:
            data_type (str): The type of data the preprocessor handles.

        Returns:
            Callable: A decorator function to register the preprocessor class.
        """

        def decorator(
            preprocessor: type[PythonObjectPreprocessorInterface],
        ) -> type[PythonObjectPreprocessorInterface]:
            cls.preprocessors[data_type] = preprocessor
            return preprocessor

        return decorator


@PythonObjectPreprocessorFactory.register_preprocessor("dataframe")
class DataFramePreprocessor(
    PandasRowPreprocessorBase, PythonObjectPreprocessorInterface
):
    """
    Preprocessor for handling pandas DataFrames, extracting nodes and relationships
    based on a predefined schema and configuration.

    Inherits from:
        PandasRowPreprocessorBase
        PythonObjectPreprocessorInterface
    """

    def run(self, df: pd.DataFrame) -> DataContainer:
        """
        Process the DataFrame to extract nodes and relationships.

        Args:
            df (pd.DataFrame): The DataFrame containing the data to be processed.

        Returns:
            DataContainer: A container holding processed nodes and relationships.
        """
        nodes: list[Node] = []
        relationships: list[Relationship] = []
        header = df.columns.tolist()
        self._validate_the_header(header)

        for _, row in df.iterrows():
            nodes_row, relationships_row, failed_chem_string = self._process_row(row)
            if failed_chem_string:
                self.failed_strings.append(failed_chem_string)
            else:
                nodes.extend(
                    [node for node_list in nodes_row.values() for node in node_list]
                )
                relationships.extend(
                    [rel for rel_list in relationships_row.values() for rel in rel_list]
                )

        return create_data_container(
            nodes, relationships, self.schema.base_nodes["chemical_equation"]
        )


@PythonObjectPreprocessorFactory.register_preprocessor("reaction_string")
class ReactionStringsPreprocessor(
    ChemicalStringPreprocessorBase, PythonObjectPreprocessorInterface
):
    """
    Preprocessor for handling lists of chemical reaction strings, extracting nodes
    and relationships based on a predefined schema and configuration.

    Inherits from:
        ChemicalStringPreprocessorBase
        PythonObjectPreprocessorInterface
    """

    def run(self, data: list[str]) -> DataContainer:
        """
        Process a list of chemical reaction strings to extract nodes and relationships.

        Args:
            data (list[str]): A list of chemical reaction strings.

        Returns:
            DataContainer: A container holding processed nodes and relationships.
        """
        nodes: list[Node] = []
        relationships: list[Relationship] = []
        for reaction_string in data:
            (
                nodes_reaction,
                relationships_reaction,
                failed_string,
            ) = self._process_reaction_string(reaction_string)
            if failed_string:
                self.failed_strings.append(failed_string)
            else:
                nodes.extend(nodes_reaction)
                relationships.extend(relationships_reaction)

        return create_data_container(
            nodes, relationships, self.schema.base_nodes["chemical_equation"]
        )


@PythonObjectPreprocessorFactory.register_preprocessor("syngraph")
class SynGraphPreprocessor(
    ChemicalStringPreprocessorBase, PythonObjectPreprocessorInterface
):
    """
    Preprocessor for handling synthetic graph objects, extracting nodes and relationships
    based on a predefined schema and configuration.

    Inherits from:
        ChemicalStringPreprocessorBase
        PythonObjectPreprocessorInterface
    """

    def __init__(self, schema: GraphSchema, config: PreprocessorConfig):
        """
        Initialize the preprocessor with a graph schema and configuration.
        Sets validation to False for synthetic graph processing.

        Args:
            schema (GraphSchema): The schema defining nodes and relationships.
            config (PreprocessorConfig): Configuration settings for preprocessing.
        """
        super().__init__(schema, config)
        self.config.validation = False

    def run(
        self,
        data: list[
            Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        ],
    ) -> DataContainer:
        """
        Process a list of synthetic graph objects to extract nodes and relationships.

        Args:
            data (list[Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]]):
                A list of synthetic graph objects.

        Returns:
            DataContainer: A container holding processed nodes and relationships.
        """
        nodes: list[Node] = []
        relationships: list[Relationship] = []
        for one_syngraph in data:
            reactions = extract_reactions_from_syngraph(one_syngraph)
            for reaction in reactions:
                reaction_string = reaction.get("input_string")
                (
                    nodes_reaction,
                    relationships_reaction,
                    _,
                ) = self._process_reaction_string(reaction_string)
                nodes.extend(nodes_reaction)
                relationships.extend(relationships_reaction)
        return create_data_container(
            nodes, relationships, self.schema.base_nodes["chemical_equation"]
        )


class Preprocessor:
    """
    A class to handle preprocessing tasks for various data formats,
    including CSV files and Python objects, for Neo4j integration.

    Attributes:
        preprocessor (Optional[PythonObjectPreprocessorInterface]): The preprocessor instance.
        schema (Optional[GraphSchema]): The graph schema for processing.
        config (PreprocessorConfig): Configuration settings for preprocessing.
    """

    def __init__(self, schema: Optional[GraphSchema] = GraphSchema()):
        """
        Initialize the Preprocessor with a graph schema and default configuration.

        Args:
            schema (Optional[GraphSchema]): The graph schema for processing.
        """
        self.preprocessor: Optional[PythonObjectPreprocessorInterface] = None
        self.schema = schema
        self.config = PreprocessorConfig()

    def set_config_from_yaml(self, file_path: Optional[str] = None) -> None:
        """
        Set the preprocessor configuration from a YAML file.

        Args:
            file_path (Optional[str]): Path to the YAML configuration file.

        Raises:
            ValueError: If no file path is provided or if the file doesn't exist.
            FileNotFoundError: If the configuration file does not exist.
            YAMLError: If there's an error parsing the YAML file.
        """
        if not file_path:
            logger.error("No file path provided for YAML configuration.")
            raise ValueError("A file path must be provided to set the configuration.")

        yaml_path = Path(file_path)
        if not yaml_path.exists():
            logger.error(f"Configuration file not found: {file_path}")
            raise FileNotFoundError(
                f"The configuration file does not exist: {file_path}"
            )

        try:
            self.config = PreprocessorConfig.build_from_yaml(yaml_path)
            logger.info(f"Successfully loaded configuration from {file_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {file_path}: {str(e)}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error while loading configuration from {file_path}: {str(e)}"
            )
            raise

    def preprocess_csv_for_neo4j_parallel(
        self,
        input_file: str,
        dask_client: Optional[Client] = None,
        output_folder: Optional[str] = None,
        tmp_folder: Optional[str] = None,
        inp_chem_format: Optional[str] = None,
        out_chem_format: Optional[str] = None,
        validation: Optional[bool] = None,
        prefix: Optional[str] = None,
        blocksize: Optional[int] = None,
        delimiter: Optional[str] = None,
        lineterminator: Optional[str] = None,
        delete_tmp: Optional[bool] = None,
        quotechar: Optional[str] = None,
    ) -> None:
        """
        Preprocess a CSV file for Neo4j integration using parallel processing.

        Args:
            input_file (str): Path to the input CSV file.
            dask_client (Optional[Client]): Optional Dask client for parallel processing.
            output_folder (Optional[str]): Folder to store output files.
            tmp_folder (Optional[str]): Temporary folder for intermediate files.
            inp_chem_format (Optional[str]): Input chemical format.
            out_chem_format (Optional[str]): Output chemical format.
            validation (Optional[bool]): Whether to validate the data.
            prefix (Optional[str]): Prefix for output files.
            blocksize (Optional[int]): Block size for Dask processing.
            delimiter (Optional[str]): Delimiter used in the CSV file.
            lineterminator (Optional[str]): Line terminator used in the CSV file.
            delete_tmp (Optional[bool]): Whether to delete temporary files.
            quotechar (Optional[str]): Quote character used in the CSV file.
        """
        config_dict = asdict(self.config)

        new_config = {
            "output_folder": output_folder,
            "tmp_folder": tmp_folder,
            "inp_chem_format": inp_chem_format,
            "out_chem_format": out_chem_format,
            "validation": validation,
            "prefix": prefix,
            "blocksize": blocksize,
            "delimiter": delimiter,
            "lineterminator": lineterminator,
            "delete_tmp": delete_tmp,
            "quotechar": quotechar,
        }
        config_dict.update({k: v for k, v in new_config.items() if v is not None})

        config = PreprocessorConfig(**config_dict)

        self.preprocessor = CSVPreprocessor(self.schema, config)
        self.preprocessor.run(input_file, parallel=True, dask_client=dask_client)

    def preprocess_csv_for_neo4j_serial(
        self,
        input_file: str,
        output_folder: Optional[str] = None,
        tmp_folder: Optional[str] = None,
        inp_chem_format: Optional[str] = None,
        out_chem_format: Optional[str] = None,
        validation: Optional[bool] = None,
        prefix: Optional[str] = None,
        delimiter: Optional[str] = None,
        lineterminator: Optional[str] = None,
        delete_tmp: Optional[bool] = None,
        quotechar: Optional[str] = None,
        chunksize: Optional[int] = None,
        nrows: Optional[int] = None,
    ) -> None:
        """
        Preprocess a CSV file for Neo4j integration using serial processing.

        Args:
            input_file (str): Path to the input CSV file.
            output_folder (Optional[str]): Folder to store output files.
            tmp_folder (Optional[str]): Temporary folder for intermediate files.
            inp_chem_format (Optional[str]): Input chemical format.
            out_chem_format (Optional[str]): Output chemical format.
            validation (Optional[bool]): Whether to validate the data.
            prefix (Optional[str]): Prefix for output files.
            delimiter (Optional[str]): Delimiter used in the CSV file.
            lineterminator (Optional[str]): Line terminator used in the CSV file.
            delete_tmp (Optional[bool]): Whether to delete temporary files.
            quotechar (Optional[str]): Quote character used in the CSV file.
            chunksize (Optional[int]): Chunk size for reading the CSV file.
            nrows (Optional[int]): Number of rows to read from the CSV file.
        """
        config_dict = asdict(self.config)

        new_config = {
            "output_folder": output_folder,
            "tmp_folder": tmp_folder,
            "inp_chem_format": inp_chem_format,
            "out_chem_format": out_chem_format,
            "validation": validation,
            "prefix": prefix,
            "chunksize": chunksize,
            "nrows": nrows,
            "delimiter": delimiter,
            "lineterminator": lineterminator,
            "delete_tmp": delete_tmp,
            "quotechar": quotechar,
        }
        config_dict.update({k: v for k, v in new_config.items() if v is not None})

        config = PreprocessorConfig(**config_dict)

        self.preprocessor = CSVPreprocessor(self.schema, config)
        self.preprocessor.run(input_file, parallel=False)

    def preprocess_object_for_neo4j(
        self,
        data: Union[pd.DataFrame, list[Union[str, "SynGraph"]]],
        data_type: str,
        inp_chem_format: Optional[str] = None,
        out_chem_format: Optional[str] = None,
        validation: Optional[bool] = None,
    ) -> DataContainer:
        """
        Preprocess Python objects for Neo4j integration.

        Args:
            data (Union[pd.DataFrame, list[Union[str, "SynGraph"]]]): The data to be processed.
            data_type (str): The type of data to be processed.
            inp_chem_format (Optional[str]): Input chemical format.
            out_chem_format (Optional[str]): Output chemical format.
            validation (Optional[bool]): Whether to validate the data.

        Returns:
            DataContainer: A container holding processed nodes and relationships.
        """
        config_dict = asdict(self.config)

        new_config = {
            "inp_chem_format": inp_chem_format,
            "out_chem_format": out_chem_format,
            "validation": validation,
        }
        config_dict.update({k: v for k, v in new_config.items() if v is not None})

        config = PreprocessorConfig(**config_dict)

        self.preprocessor = PythonObjectPreprocessorFactory.get_preprocessor(
            data_type, self.schema, config
        )

        data_container = self.preprocessor.run(data)
        return data_container

    def get_failed_strings(self):
        """
        Retrieve strings that failed during preprocessing.

        Returns:
            Optional[list[str]]: A list of failed strings, or None if stored in a file.

        Raises:
            NoPreprocessorError: If no preprocessor has been defined.
        """
        if self.preprocessor is None:
            logger.error("No preprocessor has been defined")
            raise NoPreprocessorError

        if isinstance(self.preprocessor, PythonObjectPreprocessorInterface):
            return self.preprocessor.failed_strings
        else:
            file_name = os.path.join(
                self.preprocessor.config.output_folder, "failed_strings.csv"
            )
            logger.info(f"Failed strings are in a file {file_name}")
            return None

    @classmethod
    def info(cls) -> None:
        """
        Display information about the Preprocessor capabilities, including
        the types of objects it can transform and the reaction string formats it supports.
        Provides a small usage example.

        Prints:
            - Supported data types for transformation.
            - Reaction string formats it can process.
            - Usage example.
        """
        # Supported formats
        supported_formats = {"smiles", "smarts", "rxn_blockV3K", "rxn_blockV2K"}

        # Supported data types
        supported_data_types = list(
            PythonObjectPreprocessorFactory.preprocessors.keys()
        )

        # Information display
        print("Available Preprocessing Capabilities:")
        print("=====================================")
        print(
            "Name                                           Required Args                      Optional Args                      "
        )
        print(
            "------------------------------------------------------------------------------------------------------------------------"
        )
        print(
            "preprocess_csv_for_neo4j_serial                 input_file                         output_folder, tmp_folder,          "
        )
        print(
            "                                                                                       inp_chem_format, out_chem_format,  "
        )
        print(
            "                                                                                       validation, prefix, delimiter,     "
        )
        print(
            "                                                                                       lineterminator, delete_tmp,        "
        )
        print(
            "                                                                                       quotechar, chunksize, nrows        "
        )
        print(
            "------------------------------------------------------------------------------------------------------------------------"
        )
        print(
            "preprocess_csv_for_neo4j_parallel               input_file                         output_folder, tmp_folder,          "
        )
        print(
            "                                                                                       inp_chem_format, out_chem_format,  "
        )
        print(
            "                                                                                       validation, prefix, blocksize,     "
        )
        print(
            "                                                                                       delimiter, lineterminator,         "
        )
        print(
            "                                                                                       delete_tmp, quotechar,             "
        )
        print(
            "                                                                                       dask_client                        "
        )
        print(
            "------------------------------------------------------------------------------------------------------------------------"
        )
        print(
            "preprocess_object_for_neo4j                     data, data_type                    inp_chem_format, out_chem_format,  "
        )
        print(
            "                                                                                       validation                         "
        )
        print(
            "------------------------------------------------------------------------------------------------------------------------"
        )

        print("\nSupported Data Types:")
        print("=====================")
        for data_type in supported_data_types:
            print(f"- {data_type}")

        print("\nSupported Reaction String Formats:")
        print("===================================")
        for format_name in supported_formats:
            print(f"- {format_name}")

        print("\nUsage Example:")
        print("--------------")
        print(
            textwrap.dedent(
                """
        preprocessor = Preprocessor(schema = GraphSchema(...))

        # Optional set configuration from a YAML file
        preprocessor.set_config_from_yaml('path/to/config.yaml')

        # Preprocess a CSV file for Neo4j integration
        preprocessor.preprocess_csv_for_neo4j_serial(
            input_file='path/to/input.csv',
            output_folder='path/to/output',
            inp_chem_format='smiles',
            out_chem_format='rxn_blockV3K',
            validation=True
        )

        # Preprocess Python objects
        data = ['N.O>>C', 'T.I>>S']
        data_container = preprocessor.preprocess_object_for_neo4j(
            data=data,
            data_type='reaction_string',
            inp_chem_format='smiles',
            out_chem_format='smarts'
        )
        """
            )
        )
