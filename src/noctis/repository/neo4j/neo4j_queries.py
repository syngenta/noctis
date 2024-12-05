from pathlib import Path
from typing import ClassVar, Type, Union, Optional

import pandas as pd
from linchemin.cgu.syngraph import SynGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from noctis.data_architecture.graph_schema import GraphSchema
from noctis.data_transformation.neo4j.stylers import Neo4jLoadStyle
from noctis.data_transformation.preprocessing.data_preprocessing import (
    Preprocessor,
    PreprocessorConfig,
)

from noctis import settings
from noctis.data_architecture.datamodel import DataContainer
from noctis.repository.neo4j.neo4j_functions import (
    _convert_datacontainer_to_query,
    _generate_nodes_files_string,
    _generate_properties_assignment,
    _generate_relationships_files_string,
    _get_dict_keys_from_csv,
)

import yaml

from noctis.utilities import console_logger

logger = console_logger(__name__)


class QueryAlreadyExists(Exception):
    """Exception raised when attempting to register a query that already exists."""


class Neo4jError(Exception):
    """Base exception raised ba neo4j queries"""


class Neo4jQueryValidationError(Neo4jError):
    """Error to be raised when the validation of a query fails"""


class AbstractQuery(BaseModel):
    """Abstract class representing a query"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query_name: ClassVar[str]
    query_type: ClassVar[str]
    is_parameterized: ClassVar[bool]

    query_args_required: ClassVar[list[str]]
    query_args_optional: ClassVar[list[str]] = []
    query: Union[list[str], str]

    @model_validator(mode="before")
    @classmethod
    def validate_query_kwargs(cls, values: dict[str, any]) -> dict[str, any]:
        """Class method to validate the required and optional arguments"""
        required_args = cls.query_args_required
        optional_args = cls.query_args_optional

        if required_args:
            missing_required_args = [arg for arg in required_args if arg not in values]
            if missing_required_args:
                raise Neo4jQueryValidationError(
                    f"Missing required arguments: {', '.join(missing_required_args)}"
                )
        invalid_args = [
            arg
            for arg in values
            if arg not in required_args and arg not in optional_args
        ]
        if invalid_args:
            raise Neo4jQueryValidationError(
                f"Invalid arguments provided: {', '.join(invalid_args)}"
            )
        return values

    @classmethod
    def list_arguments(cls) -> dict:
        """Helper method to list the required and optional arguments of a query"""
        return {
            "required": cls.query_args_required,
            "optional": cls.query_args_optional,
        }

    def get_query(self) -> Union[list[str], str]:
        return self.query


class Neo4jQueryRegistry:
    queries: dict = {}

    @classmethod
    def register_query(cls) -> callable:
        """Class decorator for automatic registration of new queries"""

        def decorator(registered_class: AbstractQuery) -> AbstractQuery:
            query_name = registered_class.query_name
            if query_name not in cls.queries:
                cls.queries[query_name] = registered_class
            else:
                raise QueryAlreadyExists
            return registered_class

        return decorator

    @classmethod
    def get_query_object(cls, query_name: str) -> Type[AbstractQuery]:
        """To retrieve a specific Query class based in its type and name"""
        if query_name not in cls.queries:
            available_queries = ", ".join(cls.queries.keys())
            raise ValueError(
                f"Query '{query_name}' not found. Available queries: {available_queries}"
            )

        return cls.queries[query_name]

    @classmethod
    def get_all_queries(cls):
        """To return a dictionary with all the registered query types and names"""
        return set(cls.queries.keys())


# Constraints queries
@Neo4jQueryRegistry.register_query()
class CreateUniquenessConstraints(AbstractQuery):
    """Query to constraint uniqueness of Molecule and ChemicalEquation nodes"""

    query_name: ClassVar[str] = "create_uniqueness_constraints"
    query_type: ClassVar[str] = "constraints"
    query: ClassVar[list[str]] = [
        f"CREATE CONSTRAINT Molecule_gid_unique IF NOT EXISTS FOR (a:{settings.nodes.node_molecule}) REQUIRE a.uid IS UNIQUE;",
        f"CREATE CONSTRAINT ChemicalEquation_gid_unique IF NOT EXISTS FOR (a:{settings.nodes.node_chemequation}) REQUIRE a.uid IS UNIQUE;",
    ]
    is_parameterized = False
    query_args_required: ClassVar[list[str]] = []
    query_args_optional: ClassVar[list[str]] = []

    def get_query(self) -> list[str]:
        return self.query


@Neo4jQueryRegistry.register_query()
class DropUniquenessConstraints(AbstractQuery):
    """Query to remove the uniqueness constraint for Molecule and ChemicalEquation nodes"""

    query_name: ClassVar[str] = "drop_uniqueness_constraints"
    query_type: ClassVar[str] = "constraints"
    is_parameterized = False
    query: ClassVar[list[str]] = [
        "DROP CONSTRAINT Molecule_gid_unique",
        "DROP CONSTRAINT ChemicalEquation_gid_unique",
    ]
    query_args_required: ClassVar[list[str]] = []
    query_args_optional: ClassVar[list[str]] = []


# Read queries
@Neo4jQueryRegistry.register_query()
class GetNode(AbstractQuery):
    """Query to retrieve a node based on its uid"""

    query_name: ClassVar[str] = "get_node"
    query_type: ClassVar[str] = "retrieve_graph"
    is_parameterized = False
    query: ClassVar[
        str
    ] = f"MATCH result = (n {{uid:$node_uid}}) RETURN nodes(result) as nodes"
    query_args_required: ClassVar[list[str]] = ["node_uid"]
    query_args_optional: ClassVar[list[str]] = []


@Neo4jQueryRegistry.register_query()
class GetTree(AbstractQuery):
    """Query to retrieve a tree for the given Molecule root"""

    query_name: ClassVar[str] = "get_tree"
    query_type: ClassVar[str] = "retrieve_graph"
    is_parameterized = False
    query: ClassVar[str] = (
        f"MATCH (start {{uid:$root_node_uid}}) "
        f"CALL apoc.path.subgraphAll(start, {{ "
        f"   relationshipFilter: '<PRODUCT,<REACTANT', "
        f"   minLevel: 0, "
        f"   maxLevel: $max_level "
        f"}}) "
        f"YIELD nodes, relationships "
        f"RETURN nodes,relationships"
    )
    query_args_required: ClassVar[list[str]] = ["root_node_uid", "max_level"]
    query_args_optional: ClassVar[list[str]] = []


@Neo4jQueryRegistry.register_query()
class GetRoute(AbstractQuery):
    """Query to retrieve the list of routes for a given Molecule root"""

    query_name: ClassVar[str] = "get_route"
    query_type: ClassVar[str] = "retrieve_graph"
    is_parameterized = False
    query: ClassVar[str] = (
        f"MATCH (n {{uid:$root_node_uid}}) "
        f"CALL syngenta.routes.find(n, 'Molecule', 'ChemicalEquation', '<REACTANT', '<PRODUCT', {{maxR:$max_level}}) "
        f"YIELD relationships "
        f"WITH relationships, "
        f"     [ rel in relationships | startNode(rel)] AS startNodes, "
        f"     [ rel in relationships | endNode(rel)] AS endNodes "
        f"WITH relationships, startNodes + endNodes AS allNodes "
        f"RETURN "
        f"    relationships, "
        f"    apoc.coll.toSet(allNodes) AS nodes"
    )
    query_args_required: ClassVar[list[str]] = ["root_node_uid", "max_level"]
    query_args_optional: ClassVar[list[str]] = []


@Neo4jQueryRegistry.register_query()
class CreateAnonymizedUidsOnTree(AbstractQuery):
    """Query to anonymize a tree"""

    query_name: ClassVar[str] = "create_anonymized_uid_on_tree"
    query_type: ClassVar[str] = "modify_graph"
    is_parameterized = False
    query: ClassVar[str] = (
        f"MATCH (start {{uid:$root_node_uid}}) "
        f"CALL apoc.path.subgraphAll(start, {{ "
        f"   relationshipFilter: '<PRODUCT,<REACTANT', "
        f"   minLevel: 0, "
        f"   maxLevel: $max_level "
        f"}}) "
        f"YIELD nodes, relationships "
        f"WITH nodes as nds"
        f"unwind apoc.coll.zip(range(0, size(nds)),nds) as pair"
        f"with pair[0] as counter, pair[1] as n"
        f"set n.nid = counter"
        f"return n"
    )
    query_args_required: ClassVar[list[str]] = ["root_node_uid", "max_level"]
    query_args_optional: ClassVar[list[str]] = []


# Write queries
@Neo4jQueryRegistry.register_query()
class AddNodesAndRelationships(AbstractQuery):
    """Query to add nodes and relationships from a python object. Suitable for small to medium size uploads"""

    query_name: ClassVar[str] = "load_nodes_and_relationships"
    query_type: ClassVar[str] = "modify_graph"
    query_args_required: ClassVar[list[str]] = [
        "data",
        "data_type",
    ]
    query_args_optional: ClassVar[list[str]] = [
        "input_chem_format",
        "output_chem_format",
        "validation",
        "graph_schema",
    ]
    is_parameterized = True

    query: list[str] = Field(default=None)
    data: Union[list[str], list[SynGraph], DataContainer, pd.DataFrame] = Field(
        default=None
    )
    data_type: str = Field(default=None)
    input_chem_format: str = Field(default=None)
    output_chem_format: str = Field(default=None)
    validation: bool = Field(default=None)
    graph_schema: GraphSchema = Field(default=None)

    def _build_query(self) -> None:
        data_container = self.data
        if self.data_type != "data_container":
            config = PreprocessorConfig(
                inp_chem_format=self.input_chem_format,
                out_chem_format=self.output_chem_format,
                validation=self.validation,
            )
            data_container = Preprocessor(
                self.graph_schema
            ).preprocess_object_for_neo4j(
                data=self.data,
                data_type=self.data_type,
                config=config,
            )
        self.query = _convert_datacontainer_to_query(data_container)

    def get_query(self) -> list[str]:
        if self.query is None:
            self._build_query()
        return self.query


@Neo4jQueryRegistry.register_query()
class LoadNodesFromCsv(AbstractQuery):
    """Query to load nodes from a csv file. Suitable for big bulk uploads"""

    query_name: ClassVar[str] = "load_nodes_from_csv"
    query_type: ClassVar[str] = "modify_graph"
    is_parameterized = True
    query_args_required: ClassVar[list[str]] = ["file_path"]
    query_args_optional: ClassVar[list[str]] = ["batch_size", "field_terminator"]

    query: list[str] = Field(default=None)
    file_path: Union[str, Path] = Field(default=None)
    batch_size: int = Field(default=100)
    field_terminator: str = Field(default=",")

    def _build_query(self) -> None:
        list_of_properties = _get_dict_keys_from_csv(self.file_path)
        properties_part = _generate_properties_assignment(list_of_properties)

        query = f"""
        CALL apoc.periodic.iterate(
            'LOAD CSV WITH HEADERS FROM "{self.file_path}" AS row FIELDTERMINATOR "{self.field_terminator}" RETURN row',
            'CALL apoc.merge.node([row.{Neo4jLoadStyle.COLUMN_NAMES_NODES['node_label']}], {{{Neo4jLoadStyle.COLUMN_NAMES_NODES['uid']}:row.{Neo4jLoadStyle.COLUMN_NAMES_NODES['uid']}, {properties_part} }}) YIELD node RETURN count(node) as cn',
            {{batchSize: {self.batch_size}, parallel: false}}
        )
        """
        self.query = [query]

    def get_query(self) -> list[str]:
        if self.query is None:
            self._build_query()
        return self.query


@Neo4jQueryRegistry.register_query()
class LoadRelationshipsFromCsv(AbstractQuery):
    """Query to load relationships from CSV file"""

    query_name: ClassVar[str] = "load_relationships_from_csv"
    query_type: ClassVar[str] = "modify_graph"
    is_parameterized = True
    query_args_required: ClassVar[list[str]] = ["file_path"]
    query_args_optional: ClassVar[list[str]] = ["batch_size", "field_terminator"]

    query: list[str] = Field(default=None)
    file_path: Union[str, Path] = Field(default=None)
    batch_size: int = Field(default=100)
    field_terminator: str = Field(default=",")

    def _build_query(self) -> None:
        query = f"""
         CALL apoc.periodic.iterate(
            'LOAD CSV WITH HEADERS FROM "{self.file_path}" AS row FIELDTERMINATOR "{self.field_terminator}" RETURN row',
            'MATCH (sourceNode {{{Neo4jLoadStyle.COLUMN_NAMES_NODES['uid']}: row.{Neo4jLoadStyle.COLUMN_NAMES_RELATIONSHIPS['start_node']}}})
             MATCH (destinationNode {{{Neo4jLoadStyle.COLUMN_NAMES_NODES['uid']}: row.{Neo4jLoadStyle.COLUMN_NAMES_RELATIONSHIPS['end_node']}}})
             CALL apoc.merge.relationship(sourceNode, row.{Neo4jLoadStyle.COLUMN_NAMES_RELATIONSHIPS['relationship_type']}, {{}},{{}} ,destinationNode) YIELD rel
             RETURN rel',
            {{batchSize: {self.batch_size}, parallel: false}}
        )
        """

        self.query = [query]

    def get_query(self) -> list[str]:
        if self.query is None:
            self._build_query()
        return self.query


@Neo4jQueryRegistry.register_query()
class ImportDbFromCsv(AbstractQuery):
    """Query to import nodes and relationships from CSV files"""

    query_name: ClassVar[str] = "import_db_from_csv"
    query_type: ClassVar[str] = "modify_graph"
    is_parameterized = True
    query_args_required: ClassVar[list[str]] = [
        "prefix_nodes",
        "nodes_labels",
        "prefix_relationships",
        "relationships_types",
    ]
    query_args_optional: ClassVar[list[str]] = ["delimiter"]

    query: list[str] = Field(default=None)
    prefix_nodes: str = Field(default=None)
    nodes_labels: list[str] = Field(default_factory=list)
    prefix_relationships: str = Field(default=None)
    relationships_types: list[str] = Field(default_factory=list)
    delimiter: str = Field(default=",")

    def _build_query(self) -> None:
        nodes_files = _generate_nodes_files_string(self.prefix_nodes, self.nodes_labels)
        relationships_files = _generate_relationships_files_string(
            self.prefix_relationships, self.relationships_types
        )

        query_import = f"""CALL apoc.import.csv([{nodes_files}],[{relationships_files}], {{delimiter:{self.delimiter}, stringIds: true,
                ignoreDuplicateNodes: true}})"""
        query_refactor_relationships = "MATCH (a)-[r]->(b)\n"
        query_refactor_relationships += "with a, b, collect(r) as rels\n"
        query_refactor_relationships += "where size(rels) > 1\n"
        query_refactor_relationships += (
            'CALL apoc.refactor.mergeRelationships(rels, {properties:"combine"})\n'
        )
        query_refactor_relationships += "YIELD rel\n"
        query_refactor_relationships += "RETURN rel"
        self.query = [query_import, query_refactor_relationships]

    def get_query(self) -> list[str]:
        if self.query is None:
            self._build_query()
        return self.query


@Neo4jQueryRegistry.register_query()
class DeleteChemicalEquationNodes(AbstractQuery):
    """Query to delete ChemicalEquation nodes based on the number of a particular retlationship type"""

    query_name: ClassVar[str] = "remove_ce_nodes"
    query_type: ClassVar[str] = "modify_graph"
    is_parameterized = False
    query_args_required: ClassVar[list[str]] = [
        "relationship_type",
        "max_relationships",
    ]
    query_args_optional: ClassVar[list[str]] = []
    query: ClassVar[str] = (
        f"MATCH (c:{settings.nodes.node_chemequation})<-[r:$relationship_type]-(p) "
        f"WITH c, count(r) AS molCount "
        f"where molCount > $max_relationships "
        f"detach delete c "
    )


class CustomQuery(BaseModel):
    """Custom query class for executing queries from YAML files"""

    query_name: str
    query_type: str = Field(default="retrieve_stats")
    is_parameterized: ClassVar[bool] = False
    query: Union[str, list[str]]
    query_args_required: list[str] = Field(default_factory=list)
    query_args_optional: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def validate_query_structure(cls, values: dict[str, any]) -> dict[str, any]:
        query = values.get("query")
        if isinstance(query, str):
            return values
        elif isinstance(query, list) and all(isinstance(item, str) for item in query):
            return values
        else:
            raise ValueError("Query must be either a string or a list of strings")

    @classmethod
    def build_from_dict(cls, query_dict: dict) -> "CustomQuery":
        """Build a CustomQuery instance from a dictionary"""
        if query_dict.get("is_parameterized") is True:
            logger.warning(
                f"'is_parameterized' is set to True, but it will be ignored and set to False."
            )
            query_dict["is_parameterized"] = False
        return cls(**query_dict)

    @classmethod
    def from_yaml(cls, yaml_file: str, query_name: str) -> "CustomQuery":
        """Load a specific query from a YAML file"""
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        queries = [
            item for item in data if isinstance(item, dict) and "query_name" in item
        ]
        query_dict = next(
            (q for q in queries if q.get("query_name") == query_name), None
        )
        if not query_dict:
            raise ValueError(f"Query '{query_name}' not found in the YAML file")

        return cls.build_from_dict(query_dict)

    def get_query(self) -> Union[str, list[str]]:
        return self.query

    @classmethod
    def list_queries(cls, yaml_file: str) -> list[str]:
        """List all query names available in the YAML file"""
        with open(yaml_file) as f:
            data = yaml.safe_load(f)

        return [
            item["query_name"]
            for item in data
            if isinstance(item, dict) and "query_name" in item
        ]

    def validate_query_kwargs(self, kwargs: dict[str, any]) -> None:
        """Validate the query arguments"""
        missing_required_args = [
            arg for arg in self.query_args_required if arg not in kwargs
        ]
        if missing_required_args:
            raise ValueError(
                f"Missing required arguments: {', '.join(missing_required_args)}"
            )

        invalid_args = [
            arg
            for arg in kwargs
            if arg not in self.query_args_required
            and arg not in self.query_args_optional
        ]
        if invalid_args:
            raise ValueError(f"Invalid arguments provided: {', '.join(invalid_args)}")

    def __call__(self, **kwargs: any) -> any:
        """Make the CustomQuery instance callable"""
        self.validate_query_kwargs(kwargs)
        return self
