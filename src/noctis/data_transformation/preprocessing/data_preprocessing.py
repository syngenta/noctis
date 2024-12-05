from collections.abc import Iterator
from dataclasses import dataclass
from dask.distributed import Client
from typing import Type, Callable, Optional
import pandas as pd
import dask.dataframe as dd
import os
from linchemin.cgu.syngraph import (
    SynGraph,
    MonopartiteMolSynGraph,
    MonopartiteReacSynGraph,
    BipartiteSynGraph,
)

from noctis.data_architecture.datamodel import (
    DataContainer,
    Relationship,
    Node,
)
from noctis.data_transformation.preprocessing.utils import create_data_container
from linchemin.cgu.syngraph_operations import extract_reactions_from_syngraph
from abc import ABC, abstractmethod

from noctis.data_transformation.neo4j.stylers import Neo4jImportStyle


from noctis.utilities import console_logger
from typing import Union
import csv

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


from noctis.data_transformation.preprocessing.utils import (
    _save_dataframes_to_partition_csv,
    _save_list_to_partition_csv,
    _update_partition_dict_with_row,
    dict_to_list,
)

from noctis.data_transformation.preprocessing.graph_expander import GraphExpander
from noctis.data_architecture.graph_schema import GraphSchema


@dataclass
class PreprocessorConfig:
    output_folder: Optional[str] = "output"
    tmp_folder: Optional[str] = "tmp"
    inp_chem_format: Optional[str] = "smiles"
    out_chem_format: Optional[str] = "smiles"
    validation: Optional[bool] = True
    parallel: Optional[bool] = False
    prefix: Optional[str] = "N"
    blocksize: Optional[int] = 10  # MB
    chunksize: Optional[int] = 1000  # number of lines


class PandasRowPreprocessorBase(ABC):
    def __init__(self, schema: GraphSchema, config: PreprocessorConfig):
        self.schema = schema
        self.config = config

    def _process_row(self, row: pd.Series) -> tuple[dict, dict, Union[str, None]]:
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

    def _validate_the_header(self, header):
        chemical_equation_label = self.schema.base_nodes["chemical_equation"]
        chemical_equation_field = (
            f"{chemical_equation_label}.{self.config.inp_chem_format}"
        )
        if not header:
            error_msg = "Header is empty"
            logger.error(f"EmptyHeaderError: {error_msg}")
            raise EmptyHeaderError(error_msg)

        if chemical_equation_field not in header:
            error_msg = f"Missing required field: {chemical_equation_field}"
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
    input_file = None

    def run(self, input_file) -> None:
        self.input_file = input_file

        with open(self.input_file) as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)

        self._validate_the_header(header)

        if self.config.parallel:
            self._run_parallel()
        else:
            self._run_serial()
        # here will be building a final file from partition files

    def _run_parallel(self) -> None:
        Client()
        ddf = dd.read_csv(self.input_file, blocksize=self.config.blocksize)

        ddf.map_partitions(
            lambda df, partition_info: self._process_partition(
                df, partition_info["number"]
            ),
            meta=pd.DataFrame(),
        ).compute()

    def _run_serial(self) -> None:
        for partition_number, df_partition in enumerate(
            self._serial_partition_generator()
        ):
            self._process_partition(df_partition, partition_number)

    def _serial_partition_generator(self) -> Iterator[pd.DataFrame]:
        if self.config.chunksize is None:
            yield pd.read_csv(self.input_file)
        else:
            yield from pd.read_csv(self.input_file, chunksize=self.config.chunksize)

    def _process_partition(
        self, df: pd.DataFrame, partition_number: int = None
    ) -> None:
        nodes_partition = {}
        relationships_partition = {}
        failed_strings = []

        for index, row in df.iterrows():
            nodes_row, relationships_row, failed_chem_string = self._process_row(row)
            if failed_chem_string:
                failed_strings.append(failed_chem_string)
            else:
                _update_partition_dict_with_row(nodes_partition, nodes_row)
                _update_partition_dict_with_row(
                    relationships_partition, relationships_row
                )

        df_nodes = Neo4jImportStyle.export_nodes(nodes_partition)
        df_relationships = Neo4jImportStyle.export_relationships(
            relationships_partition
        )
        _save_dataframes_to_partition_csv(
            df_nodes,
            df_relationships,
            output_dir=self.config.tmp_folder,
            partition_num=partition_number,
        )
        _save_list_to_partition_csv(
            failed_strings,
            header=self.config.inp_chem_format,
            output_dir=self.config.tmp_folder,
            name="failed_strings",
            partition_num=partition_number,
        )


class PythonObjectPreprocessorInterface(ABC):
    failed_strings = []

    @abstractmethod
    def run(self, data: object) -> DataContainer:
        pass


class ChemicalStringPreprocessorBase(ABC):
    def __init__(self, schema: GraphSchema, config: PreprocessorConfig):
        self.schema = schema
        self.config = config

    def _process_reaction_string(
        self, reaction_string: str
    ) -> tuple[list[Node], list[Relationship], Union[None, str]]:
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
        label = self.schema.base_nodes["chemical_equation"]
        return {label: {"properties": {self.config.inp_chem_format: reaction_string}}}


class PythonObjectPreprocessorFactory:
    preprocessors = {}

    @classmethod
    def get_preprocessor(
        cls,
        data_type: str,
        schema: GraphSchema,
        config: PreprocessorConfig,
    ) -> PythonObjectPreprocessorInterface:
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
    def run(self, df: pd.DataFrame) -> DataContainer:
        nodes = []
        relationships = []
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

        return create_data_container(nodes, relationships)


@PythonObjectPreprocessorFactory.register_preprocessor("reaction_string")
class ReactionStringsPreprocessor(
    ChemicalStringPreprocessorBase, PythonObjectPreprocessorInterface
):
    def run(self, data: list[str]) -> DataContainer:
        nodes = []
        relationships = []
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

        return create_data_container(nodes, relationships)


@PythonObjectPreprocessorFactory.register_preprocessor("syngraph")
class SynGraphPreprocessor(
    ChemicalStringPreprocessorBase, PythonObjectPreprocessorInterface
):
    def __init__(self, schema: GraphSchema, config: PreprocessorConfig):
        super().__init__(schema, config)
        self.config.validation = False

    def run(
        self,
        data: list[
            Union[MonopartiteReacSynGraph, BipartiteSynGraph, MonopartiteMolSynGraph]
        ],
    ) -> DataContainer:
        nodes = []
        relationships = []
        for one_syngraph in data:
            reactions = extract_reactions_from_syngraph(one_syngraph)
            for reaction in reactions:
                (
                    nodes_reaction,
                    relationships_reaction,
                    _,
                ) = self._process_reaction_string(reaction["output_string"])
                nodes.extend(nodes_reaction)
                relationships.extend(relationships_reaction)
        return create_data_container(nodes, relationships)


class Preprocessor:
    def __init__(self, schema: GraphSchema = None):
        if schema is None:
            self.schema = GraphSchema()
        else:
            self.schema = schema
        self.preprocessor = None

    def preprocess_csv_for_neo4j(
        self, input_file: str, config: PreprocessorConfig = None
    ) -> None:
        if config is None:
            self.preprocessor = CSVPreprocessor(self.schema, PreprocessorConfig())
        else:
            self.preprocessor = CSVPreprocessor(self.schema, config)
        self.preprocessor.run(input_file)

    def preprocess_object_for_neo4j(
        self,
        data: Union[pd.DataFrame, list[Union[str, "SynGraph"]]],
        data_type: str,
        config: PreprocessorConfig = None,
    ) -> DataContainer:
        if config is None:
            self.preprocessor = PythonObjectPreprocessorFactory.get_preprocessor(
                data_type, self.schema, PreprocessorConfig()
            )
        else:
            self.preprocessor = PythonObjectPreprocessorFactory.get_preprocessor(
                data_type, self.schema, config
            )
        data_container = self.preprocessor.run(data)
        return data_container

    def get_failed_strings(self):
        if self.preprocessor is None:
            logger.error("No preprocessor has been defined")
            raise NoPreprocessorError

        if isinstance(self.preprocessor, PythonObjectPreprocessorInterface):
            return self.preprocessor.failed_strings
        else:
            file_name = os.path.join(
                self.preprocessor.config.output_folder, "failed_strings.csv"
            )
            print(f"Failed strings are in a file {file_name}")
            return None
