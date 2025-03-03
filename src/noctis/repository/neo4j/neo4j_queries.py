from pathlib import Path
from typing import ClassVar, Type, Union, Optional
import os
from urllib.parse import quote

import pandas as pd
from linchemin.cgu.syngraph import SynGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from noctis.data_architecture.graph_schema import GraphSchema
from noctis.data_transformation.neo4j.stylers import Neo4jLoadStyle
from noctis.data_transformation.preprocessing.data_preprocessing import (
    Preprocessor,
)
from noctis.utilities import _wrap_text

from noctis.data_architecture.datacontainer import DataContainer
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
    parameters_embedded: ClassVar[bool]

    query_args_required: ClassVar[list[str]]
    query_args_optional: ClassVar[list[str]] = []
    query: Union[list[str], str]

    graph_schema: Optional[GraphSchema] = GraphSchema()

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

    @classmethod
    def info(cls):
        print("Available Queries:")
        print("==================")

        queries = cls.get_all_queries()

        # Define column widths
        name_width = 30
        type_width = 20
        args_width = 35

        # Print header
        header = f"{'Name':<{name_width}}{'Type':<{type_width}}{'Required Args':<{args_width}}{'Optional Args':<{args_width}}"
        print(header)
        print("-" * len(header))

        # Print each query's information
        for query_name in sorted(queries):
            query = cls.get_query_object(query_name)
            args = query.list_arguments()

            required_args = ", ".join(args["required"]) or "None"
            optional_args = ", ".join(args["optional"]) or "None"

            # Wrap long argument lists
            required_args_lines = _wrap_text(required_args, args_width)
            optional_args_lines = _wrap_text(optional_args, args_width)

            # Print the first line
            print(
                f"{query_name:<{name_width}}"
                f"{query.query_type:<{type_width}}"
                f"{required_args_lines[0]:<{args_width}}"
                f"{optional_args_lines[0]:<{args_width}}"
            )

            # Print any remaining lines
            max_lines = max(len(required_args_lines), len(optional_args_lines))
            for i in range(1, max_lines):
                print(
                    f"{'':<{name_width}}"
                    f"{'':<{type_width}}"
                    f"{required_args_lines[i] if i < len(required_args_lines) else '':<{args_width}}"
                    f"{optional_args_lines[i] if i < len(optional_args_lines) else '':<{args_width}}"
                )

            # Add a separator line between queries
            print("-" * len(header))

        print("\nUsage Example:")
        print("-------------")
        print("repo = Neo4jRepository(")
        print("    uri=<uri>,")
        print("    username=<user>,")
        print("    password=<password>,")
        print("    database=<db>,")
        print("    schema_yaml=<schema.yaml>")
        print(")")
        print()
        print("result = repo.execute_query(")
        print("    'query_name',")
        print("    arg1=<value1>,")
        print("    arg2=<value2>")
        print(")")
        print("result_custom = repo.execute_custom_query_from_yaml(")
        print("    'yaml_file',")
        print("    'query_name',")
        print("    arg1=<value1>,")
        print("    arg2=<value2>")
        print(")")


# Constraints queries
@Neo4jQueryRegistry.register_query()
class CreateUniquenessConstraints(AbstractQuery):
    """Query to constraint uniqueness of Molecule and ChemicalEquation nodes"""

    query_name: ClassVar[str] = "create_uniqueness_constraints"
    query_type: ClassVar[str] = "constraints"
    parameters_embedded = False
    query_args_required: ClassVar[list[str]] = []
    query_args_optional: ClassVar[list[str]] = []
    query: list[str] = Field(default=None)

    def _build_query(self):
        self.query = [
            f"CREATE CONSTRAINT {node_label}_uid_unique IF NOT EXISTS FOR (a:{node_label}) REQUIRE a.uid IS UNIQUE;"
            for node_label in self.graph_schema.base_nodes.values()
        ]

    def get_query(self) -> list[str]:
        if self.query is None:
            self._build_query()
        return self.query


@Neo4jQueryRegistry.register_query()
class DropUniquenessConstraints(AbstractQuery):
    """Query to remove the uniqueness constraint for Molecule and ChemicalEquation nodes"""

    query_name: ClassVar[str] = "drop_uniqueness_constraints"
    query_type: ClassVar[str] = "constraints"
    parameters_embedded = False
    query_args_required: ClassVar[list[str]] = []
    query_args_optional: ClassVar[list[str]] = []
    query: list[str] = Field(default=None)

    def _build_query(self):
        self.query = [
            f"DROP CONSTRAINT {node_label}_uid_unique"
            for node_label in self.graph_schema.base_nodes.values()
        ]

    def get_query(self) -> list[str]:
        if self.query is None:
            self._build_query()
        return self.query


@Neo4jQueryRegistry.register_query()
class ShowUniquenessConstraints(AbstractQuery):
    """Query to show the uniqueness constraints"""

    query_name: ClassVar[str] = "show_uniqueness_constraints"
    query_type: ClassVar[str] = "retrieve_stats"
    parameters_embedded = False
    query_args_required: ClassVar[list[str]] = []
    query_args_optional: ClassVar[list[str]] = []
    query: ClassVar[str] = "SHOW CONSTRAINTS"

    def get_query(self) -> list[str]:
        return self.query


# Read queries
@Neo4jQueryRegistry.register_query()
class GetNode(AbstractQuery):
    """Query to retrieve a node based on its uid"""

    query_name: ClassVar[str] = "get_node"
    query_type: ClassVar[str] = "retrieve_graph"
    parameters_embedded = False
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
    parameters_embedded = False
    query: str = None
    query_args_required: ClassVar[list[str]] = ["root_node_uid", "max_level"]
    query_args_optional: ClassVar[list[str]] = []

    def _build_query(self):
        self.query = (
            f"MATCH (start {{uid:$root_node_uid}}) "
            f"CALL apoc.path.subgraphAll(start, {{ "
            f"   relationshipFilter: '<{self.graph_schema.base_relationships['product']['type']},<{self.graph_schema.base_relationships['reactant']['type']}', "
            f"   minLevel: 0, "
            f"   maxLevel: $max_level "
            f"}}) "
            f"YIELD nodes, relationships "
            f"RETURN nodes, relationships"
        )

    def get_query(self) -> str:
        if self.query is None:
            self._build_query()
        return self.query


@Neo4jQueryRegistry.register_query()
class GetRoutes(AbstractQuery):
    """Query to retrieve the list of routes for a given Molecule root"""

    query_name: ClassVar[str] = "get_routes"
    query_type: ClassVar[str] = "retrieve_graph"
    parameters_embedded = True
    query: str = None
    query_args_required: ClassVar[list[str]] = [
        "root_node_uid",
    ]
    query_args_optional: ClassVar[list[str]] = [
        "max_number_reactions",
        "node_stop_property",
    ]
    root_node_uid: str = Field(default=None)
    max_number_reactions: int = Field(default=None)
    node_stop_property: str = Field(default=None)

    def _build_query(self):
        self.query = (
            f"MATCH (n {{uid:'{self.root_node_uid}'}}) "
            f"CALL noctis.route.miner(n, '{self.graph_schema.base_nodes['molecule']}', '{self.graph_schema.base_nodes['chemical_equation']}', '<{self.graph_schema.base_relationships['reactant']['type']}', '<{self.graph_schema.base_relationships['product']['type']}', {self._build_parameters_map()}) "
            f"YIELD relationships "
            f"WITH relationships, "
            f"     [ rel in relationships | startNode(rel)] AS startNodes, "
            f"     [ rel in relationships | endNode(rel)] AS endNodes "
            f"WITH relationships, startNodes + endNodes AS allNodes "
            f"RETURN "
            f"    relationships, "
            f"    apoc.coll.toSet(allNodes) AS nodes"
        )

    def _build_parameters_map(self):
        parameters = {}

        if self.max_number_reactions is not None:
            parameters["maxNumberReactions"] = self.max_number_reactions

        if self.node_stop_property is not None:
            parameters["nodeStopProperty"] = f'"{self.node_stop_property}"'

        # Custom string representation
        if parameters:
            items = [f"{k}:{v}" for k, v in parameters.items()]
            return "{" + ", ".join(items) + "}"
        else:
            return "{}"

    def get_query(self) -> list[str]:
        if self.query is None:
            self._build_query()
        return [self.query]


@Neo4jQueryRegistry.register_query()
class GetPathsThroughIntermediates(AbstractQuery):
    """Query to get paths which start with start node and go through provided intermediate nodes"""

    query_name: ClassVar[str] = "get_paths"
    query_type: ClassVar[str] = "retrieve_graph"
    query_args_required: ClassVar[list[str]] = [
        "start_node_uid",
    ]
    query_args_optional: ClassVar[list[str]] = [
        "intermediates_uids",
        "max_reactions_between_intermediates",
        "total_n_reactions",
        "min_max_n_reactions",
        "end_at_last_intermediate",
        "limit",
    ]
    parameters_embedded = True

    query: list[str] = Field(default=None)
    start_node_uid: str = Field(default=None)
    intermediates_uids: list[str] = Field(default=None)
    max_reactions_between_intermediates: int = Field(default=None)
    total_n_reactions: int = Field(default=None)
    min_max_n_reactions: tuple[Union[None, int], int] = Field(default=None)
    end_at_last_intermediate: bool = Field(default=True)
    limit: int = Field(default=None)
    between_intr_in_rel: int = Field(default=None)
    total_in_rel: int = Field(default=None)
    min_max_in_rel: tuple[int, int] = Field(default=None)

    def _build_query(self):
        query_parts = []
        where_clauses = []

        # Start node
        query_parts.append(f"MATCH (start {{uid:'{self.start_node_uid}'}})")

        # Validate path length constraints
        self._validate_path_length_constraints()

        # Handle intermediates
        if self.intermediates_uids:
            query_parts.extend(self._build_intermediates_query())
        else:
            query_parts.append(self._build_no_intermediates_query())

        # Add WHERE clauses for path length
        where_clauses.extend(self._build_path_length_clauses())

        # Combine all parts
        built_query = " ".join(part for part in query_parts)

        if where_clauses:
            built_query += f" WHERE {' AND '.join(where_clauses)}"
        built_query += " RETURN p"

        if self.limit is not None:
            built_query += f" LIMIT {self.limit}"

        self.query = built_query

        return

    def _calculate_in_rel_values(self):
        self.between_intr_in_rel = self.max_reactions_between_intermediates * 2
        if self.total_n_reactions is not None:
            self.total_in_rel = self.total_n_reactions * 2
        if self.min_max_n_reactions is not None:
            min_steps, max_steps = self.min_max_n_reactions
            self.min_max_in_rel = (min_steps * 2, max_steps * 2)

    def _validate_path_length_constraints(self):
        if self.total_n_reactions is not None and self.min_max_n_reactions is not None:
            raise ValueError(
                "total_n_reactions and min_max_n_reactions cannot be used together"
            )

        if self.min_max_n_reactions is not None:
            min_steps, max_steps = self.min_max_n_reactions
            if min_steps is not None:
                if min_steps >= max_steps:
                    raise ValueError("min_steps must be less than max_steps")
            else:
                if self.intermediates_uids is not None:
                    min_steps = len(self.intermediates_uids)
                else:
                    min_steps = 1
                self.min_max_n_reactions = (min_steps, max_steps)
        if self.max_reactions_between_intermediates is None:
            if self.total_n_reactions is not None:
                self.max_reactions_between_intermediates = self.total_n_reactions
            else:
                self.max_reactions_between_intermediates = 3

        self._calculate_in_rel_values()

    def _build_intermediates_query(self):
        parts = []
        for i, intermediate in enumerate(self.intermediates_uids):
            parts.append(f"MATCH (intrm{i} {{uid: '{intermediate}'}})")

        path = "MATCH p=(start)"
        for i in range(len(self.intermediates_uids)):
            path += f"<-[*2..{self.between_intr_in_rel}]-(intrm{i})"

        if not self.end_at_last_intermediate:
            path += f"<-[*2..{self.between_intr_in_rel}]-()"

        parts.append(path)
        return parts

    def _build_no_intermediates_query(self) -> str:
        if self.min_max_n_reactions is not None:
            min_steps, max_steps = self.min_max_in_rel
            return f"MATCH p=(start)<-[*{min_steps}..{max_steps}]-(:{self.graph_schema.base_nodes['molecule']})"
        elif self.total_n_reactions is not None:
            return f"MATCH p=(start)<-[*{self.total_in_rel}]-(:{self.graph_schema.base_nodes['molecule']})"
        else:
            return f"MATCH p=(start)<-[*]-(:{self.graph_schema['molecule']})"

    def _build_path_length_clauses(self):
        clauses = []
        if self.total_n_reactions is not None:
            clauses.append(f"length(p) = {self.total_in_rel}")
        elif self.min_max_n_reactions is not None:
            min_steps, max_steps = self.min_max_in_rel
            clauses.append(f"length(p) >= {min_steps}")
            clauses.append(f"length(p) <= {max_steps}")
        return clauses

    def get_query(self) -> list[str]:
        if self.query is None:
            self._build_query()
        return [self.query]


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
    parameters_embedded = True

    query: list[str] = Field(default=None)
    data: Union[list[str], list[SynGraph], DataContainer, pd.DataFrame] = Field(
        default=None
    )
    data_type: str = Field(default=None)
    input_chem_format: str = Field(default=None)
    output_chem_format: str = Field(default=None)
    validation: bool = Field(default=None)

    def _build_query(self) -> None:
        data_container = self.data
        if self.data_type != "data_container":
            data_container = Preprocessor(
                self.graph_schema
            ).preprocess_object_for_neo4j(
                data=self.data,
                data_type=self.data_type,
                inp_chem_format=self.input_chem_format,
                out_chem_format=self.output_chem_format,
                validation=self.validation,
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
    parameters_embedded = True
    query_args_required: ClassVar[list[str]] = ["file_path"]
    query_args_optional: ClassVar[list[str]] = [
        "batch_size",
        "field_terminator",
        "import_from_file_system",
    ]

    query: list[str] = Field(default=None)
    file_path: Union[str, Path] = Field(default=None)
    batch_size: int = Field(default=100)
    field_terminator: str = Field(default=",")
    import_from_file_system: bool = Field(default=True)

    def _build_query(self) -> None:
        abs_path = os.path.abspath(self.file_path)
        if self.import_from_file_system:
            file_url = f"file:///{quote(abs_path.replace(os.sep, '/'))}"
        else:
            file_name = os.path.basename(abs_path)
            file_url = f"file:///{file_name}"

        list_of_properties = _get_dict_keys_from_csv(abs_path)
        properties_part = _generate_properties_assignment(list_of_properties)

        query = f"""
        CALL apoc.periodic.iterate(
            'LOAD CSV WITH HEADERS FROM "{file_url}" AS row FIELDTERMINATOR "{self.field_terminator}" RETURN row',
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
    parameters_embedded = True
    query_args_required: ClassVar[list[str]] = ["file_path"]
    query_args_optional: ClassVar[list[str]] = [
        "batch_size",
        "field_terminator",
        "import_from_file_system",
    ]

    query: list[str] = Field(default=None)
    file_path: Union[str, Path] = Field(default=None)
    batch_size: int = Field(default=100)
    field_terminator: str = Field(default=",")
    import_from_file_system: bool = Field(default=True)

    def _build_query(self) -> None:
        abs_path = os.path.abspath(self.file_path)
        if self.import_from_file_system:
            file_url = f"file:///{quote(abs_path.replace(os.sep, '/'))}"
        else:
            file_name = os.path.basename(abs_path)
            file_url = f"file:///{file_name}"

        query = f"""
         CALL apoc.periodic.iterate(
            'LOAD CSV WITH HEADERS FROM "{file_url}" AS row FIELDTERMINATOR "{self.field_terminator}" RETURN row',
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
    parameters_embedded = True
    query_args_required: ClassVar[list[str]] = []
    query_args_optional: ClassVar[list[str]] = [
        "folder_path",
        "prefix",
        "delimiter",
        "nodes_labels",
        "relationships_types",
    ]

    query: list[str] = Field(default=None)
    prefix: str = Field(default=None)
    nodes_labels: list[str] = Field(default_factory=list)
    relationships_types: list[str] = Field(default_factory=list)
    delimiter: str = Field(default=",")
    folder_path: str = Field(default=None)

    def _build_query(self) -> None:
        if not self.nodes_labels:
            self.nodes_labels = self.graph_schema.get_nodes_labels()
        if not self.relationships_types:
            self.relationships_types = self.graph_schema.get_relationships_types()

        nodes_files = _generate_nodes_files_string(
            self.folder_path, self.prefix, self.nodes_labels
        )
        relationships_files = _generate_relationships_files_string(
            self.folder_path, self.prefix, self.relationships_types
        )

        query_import = f"""CALL apoc.import.csv([{nodes_files}],[{relationships_files}], {{delimiter:'{self.delimiter}', stringIds: true,
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
class DeleteAllNodes(AbstractQuery):
    """Query to delete ChemicalEquation nodes based on the number of a particular relationship type"""

    query_name: ClassVar[str] = "delete_all_nodes"
    query_type: ClassVar[str] = "modify_graph"
    parameters_embedded = False
    query_args_required: ClassVar[list[str]] = []
    query_args_optional: ClassVar[list[str]] = []
    query: ClassVar[str] = f"MATCH (c)" f"detach delete c "


@Neo4jQueryRegistry.register_query()
class GetGDBSchema(AbstractQuery):
    """Query to delete ChemicalEquation nodes based on the number of a particular relationship type"""

    query_name: ClassVar[str] = "get_gdb_schema"
    query_type: ClassVar[str] = "retrieve_stats"
    parameters_embedded = False
    query_args_required: ClassVar[list[str]] = []
    query_args_optional: ClassVar[list[str]] = []
    query: ClassVar[str] = (
        f"call db.labels() yield label "
        f"with collect(label) as nodes "
        f"call db.relationshipTypes() yield relationshipType "
        f"return nodes, collect(relationshipType) as relationships"
    )


class CustomQuery(BaseModel):
    """Custom query class for executing queries from YAML files"""

    query_name: str
    query_type: str = Field(default="retrieve_stats")
    parameters_embedded: ClassVar[bool] = False
    query: Union[str, list[str]]
    query_args_required: list[str] = Field(default_factory=list)
    query_args_optional: list[str] = Field(default_factory=list)
    graph_schema: GraphSchema = Field(default=None)

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
        if query_dict.get("parameters_embedded") is True:
            logger.warning(
                f"'parameters_embedded' is set to True, but it will be ignored and set to False."
            )
            query_dict["parameters_embedded"] = False
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
