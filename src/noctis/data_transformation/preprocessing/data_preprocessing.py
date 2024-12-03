from collections.abc import Iterator
from dataclasses import dataclass
from dask.distributed import Client
from typing import Type, Callable
import pandas as pd
import dask.dataframe as dd
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

logger = console_logger(__name__)


class MissingUIDError(Exception):
    """Custom error for when a UID is missing for a node."""

    def __init__(self, label):
        self.message = f"Missing 'uid' for node with label '{label}'"
        super().__init__(self.message)


from noctis.data_transformation.preprocessing.utils import (
    _save_dataframes_to_partition_csv,
    _update_partition_dict_with_row,
    dict_to_list,
)

from noctis.data_transformation.preprocessing.graph_expander import GraphExpander
from noctis.data_architecture.graph_schema import GraphSchema


@dataclass
class PreprocessorConfig:
    input_file: str
    output_folder: str
    tmp_folder: str
    inp_chem_format: str
    out_chem_format: str
    validation: bool
    parallel: bool
    prefix: str
    blocksize: int  # MB
    chunksize: int  # number of lines


class PandasRowPreprocessorBase(ABC):
    def __init__(self, schema: GraphSchema, config: PreprocessorConfig):
        self.schema = schema
        self.config = config

    def _process_row(self, row: pd.Series) -> tuple[dict, dict]:
        split_row = self._split_row_by_node_types(row)
        ge = GraphExpander(self.schema)
        nodes, relationships = ge.expand_from_csv(
            split_row,
            self.config.inp_chem_format,
            self.config.out_chem_format,
            self.config.validation,
        )

        return nodes, relationships

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

        for label, data in node_data.items():
            if label not in self.schema.base_nodes.values() and "uid" not in data:
                logger.error(f"Missing uid column in node {label}")
                raise MissingUIDError(label)

        return node_data


class CSVPreprocessor(PandasRowPreprocessorBase):
    def run(self) -> None:
        if self.config.parallel:
            self._run_parallel()
        else:
            self._run_serial()

    def _run_parallel(self) -> None:
        Client()
        ddf = dd.read_csv(self.config.input_file, blocksize=self.config.blocksize)

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
            yield pd.read_csv(self.config.input_file)
        else:
            yield from pd.read_csv(
                self.config.input_file, chunksize=self.config.chunksize
            )

    def _process_partition(
        self, df: pd.DataFrame, partition_number: int = None
    ) -> None:
        nodes_partition = {}
        relationships_partition = {}

        for index, row in df.iterrows():
            nodes_row, relationships_row = self._process_row(row)
            _update_partition_dict_with_row(nodes_partition, nodes_row)
            _update_partition_dict_with_row(relationships_partition, relationships_row)

        df_nodes = Neo4jImportStyle.export_nodes(nodes_partition)
        df_relationships = Neo4jImportStyle.export_relationships(
            relationships_partition
        )
        _save_dataframes_to_partition_csv(
            df_nodes,
            df_relationships,
            output_dir=self.config.output_folder,
            partition_num=partition_number,
        )


class PythonObjectPreprocessorInterface(ABC):
    @abstractmethod
    def run(self, data: object) -> DataContainer:
        pass


class ChemicalStringPreprocessorBase(ABC):
    def __init__(self, schema: GraphSchema, config: PreprocessorConfig):
        self.schema = schema
        self.config = config

    def _process_reaction_string(
        self, reaction_string: str
    ) -> tuple[list[Node], list[Relationship]]:
        reaction_dict = self._build_reaction_string_dict(reaction_string)
        ge = GraphExpander(self.schema)
        nodes, relationships = ge.expand_from_csv(
            reaction_dict,
            self.config.inp_chem_format,
            self.config.out_chem_format,
            self.config.validation,
        )

        return dict_to_list(nodes), dict_to_list(relationships)

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

        for _, row in df.iterrows():
            nodes_row, relationships_row = self._process_row(row)
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
            nodes_reaction, relationships_reaction = self._process_reaction_string(
                reaction_string
            )
            nodes.extend(nodes_reaction)
            relationships.extend(relationships_reaction)

        return create_data_container(nodes, relationships)


@PythonObjectPreprocessorFactory.register_preprocessor("syngraph")
class SynGraphPreprocessor(
    ChemicalStringPreprocessorBase, PythonObjectPreprocessorInterface
):
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
                nodes_reaction, relationships_reaction = self._process_reaction_string(
                    reaction["output_string"]
                )
                nodes.extend(nodes_reaction)
                relationships.extend(relationships_reaction)
        return create_data_container(nodes, relationships)


class Preprocessor:
    def __init__(self, schema: GraphSchema):
        self.schema = schema

    def preprocess_csv_for_neo4j(self, config: PreprocessorConfig) -> None:
        csv_preprocessor = CSVPreprocessor(self.schema, config)
        csv_preprocessor.run()

    def preprocess_object_for_neo4j(
        self,
        data: Union[pd.DataFrame, list[Union[str, "SynGraph"]]],
        data_type: str,
        config: PreprocessorConfig,
    ) -> DataContainer:
        preprocessor = PythonObjectPreprocessorFactory.get_preprocessor(
            data_type, self.schema, config
        )
        data_container = preprocessor.run(data)
        return data_container
