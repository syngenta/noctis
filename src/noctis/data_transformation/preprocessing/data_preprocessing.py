import json
from typing import Iterator
from dataclasses import dataclass
from dask.distributed import Client, wait
import pandas as pd
import dask.dataframe as dd
import os
import csv

from noctis.data_transformation.preprocessing.utils import (
    _save_dataframes_to_partition_csv,
    _update_partition_dict_with_row,
    _build_dataframes_from_dict,
)

from noctis.data_transformation.preprocessing.graph_expander import GraphExpander
from noctis.data_architecture.graph_schema import GraphSchema


@dataclass
class FilePreprocessorConfig:
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


class Preprocessor:
    def __init__(self, schema: GraphSchema):
        self.schema = schema

    def preprocess_csv_for_neo4j(self, config: FilePreprocessorConfig) -> None:
        csv_preprocessor = FilePreprocessor(self.schema, config)
        csv_preprocessor.run()

    def preprocess_object_for_neo4j(self, config: FilePreprocessorConfig) -> None:
        # TODO: a wrapper on DC Generator
        pass


class FilePreprocessor:
    def __init__(self, schema: GraphSchema, config: FilePreprocessorConfig):
        self.schema = schema
        self.config = config

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

        df_nodes = _build_dataframes_from_dict(nodes_partition)
        df_relationships = _build_dataframes_from_dict(relationships_partition)
        _save_dataframes_to_partition_csv(
            df_nodes,
            df_relationships,
            output_dir=self.config.output_folder,
            partition_num=partition_number,
        )
        return

    def _process_row(self, row: pd.Series) -> tuple[dict, dict]:
        splitted_row = self._split_row_by_node_types(row)
        ge = GraphExpander(self.schema)
        nodes, relationships = ge.expand_from_csv(
            splitted_row,
            self.config.inp_chem_format,
            self.config.out_chem_format,
            self.config.validation,
        )

        return nodes, relationships

    def _split_row_by_node_types(self, row: pd.Series) -> dict[str, dict[str, str]]:
        node_data = {}

        for column, value in row.items():
            parts = column.split(".", 1)

            if len(parts) == 2:
                label, property_name = parts
                if label not in node_data:
                    node_data[label] = {}
                node_data[label][property_name] = value

        return node_data
