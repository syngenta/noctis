from pydantic import BaseModel, Field
from typing import Optional
from noctis.data_architecture.datamodel import GraphRecord
from noctis.data_transformation.postprocessing.chemdata_generators import (
    ChemDataGeneratorFactory,
)
from noctis import settings


class DataContainer(BaseModel):
    """
    A container for managing and transforming chemical equation data.

    DataContainer is a collection of GraphRecord objects with methods for
    managing records and transforming data into various formats.

    Attributes:
        records (list[GraphRecord]): List of GraphRecord objects.
        ce_label (str): Label for chemical equations (default: settings.nodes.node_chemequation).

    Methods:
        add_record(record): Add a GraphRecord to the container.
        get_record(record_key): Retrieve a specific GraphRecord.
        get_records(record_keys): Retrieve multiple GraphRecords.
        get_subcontainer_with_records(record_keys): Create a new DataContainer with specified records.
        transform_to(format_type, with_record_id, ce_label): Transform data to specified format.

    Note:
        DataContainer objects can be compared for equality using the == operator.
    """

    records: list[GraphRecord] = Field(default_factory=list)
    ce_label: str = Field(default=settings.nodes.node_chemequation)

    def __eq__(self, other):
        if isinstance(other, DataContainer):
            return self.records == other.records
        return False

    def set_ce_label(self, ce_label: str) -> None:
        self.ce_label = ce_label

    def add_record(self, record: GraphRecord) -> None:
        """To add a GraphRecord to a DataContainer"""
        self.records.append(record)

    def get_record(self, record_key: int) -> GraphRecord:
        return self.records[record_key]

    def get_records(self, record_keys: list[int]) -> list[GraphRecord]:
        return [self.records[key] for key in record_keys]

    def get_subcontainer_with_records(self, record_keys: list[int]) -> "DataContainer":
        subcontainer = DataContainer()
        missing_keys: set[int] = set()

        for key in record_keys:
            if key < len(self.records):
                subcontainer.add_record(self.records[key].__deepcopy__())
            else:
                missing_keys.add(key)

        if missing_keys:
            missing_keys_str = ", ".join(map(str, missing_keys))
            raise KeyError(
                f"Record keys {missing_keys_str} not found in DataContainer."
            )

        return subcontainer

    def __str__(self) -> str:
        num_records = len(self.records)
        records_preview = ", ".join(
            str(record) for record in self.records[:10]
        )  # Preview first 3 records
        return (
            f"DataContainer with {num_records} records\n"
            f"Chemical Equation Label: {self.ce_label}\n"
            f"Records Preview: [{records_preview}]"
        )

    def transform_to(
        self,
        format_type: str,
        with_record_id: Optional[bool] = True,
        ce_label: Optional[str] = None,
    ):
        if ce_label is None:
            ce_label = self.ce_label
        generator = ChemDataGeneratorFactory().get_generator(format_type)
        return generator.generate(self.records, with_record_id, ce_label)

    @classmethod
    def info(cls) -> str:
        """Return detailed information about registered generators, reaction formats, and usage."""

        # Get available format types and reaction string formats from ChemDataGeneratorFactory
        available_format_types = ChemDataGeneratorFactory.get_available_formats()
        available_reaction_formats = (
            ChemDataGeneratorFactory.get_available_reaction_formats()
        )

        info_lines = [
            "DataContainer Class Information:",
            "================================",
            "Attributes:",
            "-----------",
            "records (list[GraphRecord]): List of GraphRecord objects.",
            "ce_label (str): Label for chemical equations (default: settings.nodes.node_chemequation).",
            "",
            "Methods:",
            "--------",
            "add_record(record): Add a GraphRecord to the container.",
            "get_record(record_key): Retrieve a specific GraphRecord.",
            "get_records(record_keys): Retrieve multiple GraphRecords.",
            "get_subcontainer_with_records(record_keys): Create a new DataContainer with specified records.",
            "transform_to(format_type, with_record_id, ce_label): Transform data to specified format.",
            "",
            "Available Format Types for transform_to:",
            "----------------------------------------",
            ", ".join(available_format_types),
            "",
            "Available Reaction String Formats:",
            "-----------------------------------",
            ", ".join(available_reaction_formats),
            "",
            "Usage Example:",
            "--------------",
            "data_container = DataContainer()\n"
            "data_container.add_record(GraphRecord(...))\n"
            "record = data_container.get_record(0)\n"
            "subcontainer = data_container.get_subcontainer_with_records([0, 1, 2])\n"
            "dataframe_nodes, dataframe_relationships = data_container.transform_to(format_type='pandas')\n",
        ]

        return print("\n".join(info_lines))
