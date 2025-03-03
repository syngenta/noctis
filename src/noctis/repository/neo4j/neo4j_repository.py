import pandas as pd
from contextlib import contextmanager
from typing import Optional, Type, Callable, Union

from neo4j import Driver, GraphDatabase, Session
from neo4j.exceptions import Neo4jError

from noctis.data_architecture.graph_schema import GraphSchema
from noctis.data_architecture.datacontainer import DataContainer
from noctis.data_transformation.neo4j.neo4j_formatter import format_result
from noctis.repository.neo4j.neo4j_queries import (
    AbstractQuery,
    Neo4jQueryRegistry,
    CustomQuery,
)

from neo4j import Result, Record
from noctis.utilities import console_logger
from noctis import settings

logger = console_logger(__name__)


class Neo4jRepository:
    """
    Repository class for interacting with a Neo4j database using defined query strategies.

    Attributes:
        schema (GraphSchema): The schema used for graph operations.
        _driver (Driver): Neo4j driver for database connections.
        _query_strategies (dict[str, tuple[Callable, Callable]]): Mapping of query types to strategies and transaction functions.
    """

    def __init__(
        self,
        uri: Optional[str] = settings.NEO4J_DEV_URL,
        username: Optional[str] = settings.NEO4J_DEV_USER,
        password: Optional[str] = settings.NEO4J_DEV_PASSWORD,
        database: Optional[str] = None,
        schema: Optional[GraphSchema] = GraphSchema(),
    ):
        """
        Initialize the Neo4jRepository with connection details and schema.

        Args:
            uri (Optional[str]): URI of the Neo4j database. Defaults to settings.NEO4J_DEV_URL.
            username (Optional[str]): Username for authentication. Defaults to settings.NEO4J_DEV_USER.
            password (Optional[str]): Password for authentication. Defaults to settings.NEO4J_DEV_PASSWORD.
            database (Optional[str]): Database name, if applicable. Defaults to None.
            schema (Optional[GraphSchema]): Schema for graph operations. Defaults to a new GraphSchema instance.
        """
        self.schema = schema
        self._driver = GraphDatabase.driver(
            uri, auth=(username, password), database=database
        )
        self._query_strategies = {
            "retrieve_graph": (self._retrieve_graph_strategy, self._execute_read),
            "modify_graph": (self._modify_graph_strategy, self._execute_write),
            "retrieve_stats": (self._retrieve_stats_strategy, self._execute_read),
        }

    def close(self) -> None:
        """
        Close the connection to the Neo4j database.
        """
        self._driver.close()

    @contextmanager
    def _session_context(self) -> Session:
        """
        Context manager for Neo4j session handling.

        Yields:
            Session: A Neo4j session for executing queries.

        Raises:
            Neo4jError: If a Neo4j-specific error occurs.
            Exception: For other types of errors.
        """
        session = None
        try:
            session = self._driver.session()
            yield session
        except Neo4jError as e:
            print(f"Neo4j error occurred: {e}")
            raise
        except Exception as e:
            print(f"Something went wrong: {e}")
            raise
        finally:
            if session:
                session.close()

    @classmethod
    def info(cls):
        """Returns info about all available queries, their required and optional arguments, and the query templates"""
        return Neo4jQueryRegistry.info()

    def create_constraints(self):
        """
        Create uniqueness constraints in the Neo4j database using predefined queries.
        """
        query = Neo4jQueryRegistry.get_query_object("create_uniqueness_constraints")
        with self._session_context() as session:
            for query in query().get_query():
                session.run(query)

    def drop_constraints(self):
        """
        Drop uniqueness constraints in the Neo4j database using predefined queries.
        """
        query = Neo4jQueryRegistry.get_query_object("drop_uniqueness_constraints")
        with self._session_context() as session:
            for query in query().get_query():
                session.run(query)

    def show_constraints(self):
        """
        Show the current uniqueness constraints in the Neo4j database.

        Returns:
            any: Result of the query showing constraints.
        """
        return self.execute_query("show_uniqueness_constraints")

    def execute_custom_query_from_yaml(
        self, yaml_file: str, query_name: str, **kwargs: any
    ) -> any:
        """
        Execute a custom query defined in a YAML file.

        Args:
            yaml_file (str): Path to the YAML file containing query definitions.
            query_name (str): Name of the query to execute.
            **kwargs: Additional arguments for the query.

        Returns:
            any: Result of the custom query execution.
        """
        query = CustomQuery.from_yaml(yaml_file=yaml_file, query_name=query_name)
        result = self._build_query_execution_strategy(query, **kwargs)

        gdb_schema = self.execute_query("get_gdb_schema")
        self._compare_user_schema_and_gdb_schema(gdb_schema)

        return result

    def _compare_user_schema_and_gdb_schema(self, gdb_schema: pd.DataFrame) -> None:
        """
        Compare the user-defined schema with the graph database schema.

        Args:
            gdb_schema (pd.DataFrame): Schema from the graph database.

        Note:
            Logs warnings if there are discrepancies between the schemas.
        """
        node_labels = self.schema.get_nodes_labels()
        relationship_types = self.schema.get_relationships_types()

        # Check if nodes and relationships columns exist in the DataFrame
        if (
            "nodes" not in gdb_schema.columns
            or "relationships" not in gdb_schema.columns
        ):
            logger.warning(
                "Nodes and/or relationships are not defined in the GDB schema DataFrame."
            )
            return

        gdb_node_labels = gdb_schema["nodes"].iloc[0]
        gdb_relationship_types = gdb_schema["relationships"].iloc[0]
        if set(node_labels) != set(gdb_node_labels) or set(relationship_types) != set(
            gdb_relationship_types
        ):
            logger.warning(
                f"""
            The user schema does not match the graph database schema after running custom query.

            Graph DB Schema:
                Nodes: {gdb_node_labels}
                Relationships: {gdb_relationship_types}

            User Schema:
                Nodes: {node_labels}
                Relationships: {relationship_types}
            """
            )

    def execute_query(self, query_name: str, **kwargs: any) -> any:
        """
        Execute a predefined query by name.

        Args:
            query_name (str): Name of the query to execute.
            **kwargs: Additional arguments for the query.

        Returns:
            any: Result of the query execution.
        """
        query = Neo4jQueryRegistry.get_query_object(query_name)
        return self._build_query_execution_strategy(query, **kwargs)

    def _build_query_execution_strategy(
        self, query: Union[Type[AbstractQuery], CustomQuery], **kwargs: any
    ) -> any:
        """
        Build and execute a query strategy based on the query type.

        Args:
            query (Union[Type[AbstractQuery], CustomQuery]): The query object.
            **kwargs: Additional arguments for the query.

        Returns:
            any: Result of the query execution.

        Raises:
            ValueError: If the query type is unsupported.
        """
        query_type = query.query_type
        strategy, transaction_function = self._query_strategies.get(
            query_type, (None, None)
        )
        if not strategy or not transaction_function:
            raise ValueError(f"Unsupported query type: {query_type}")

        with self._session_context() as session:
            return transaction_function(session, strategy, query, **kwargs)

    @staticmethod
    def _execute_read(
        session: Session, strategy: Callable, query: AbstractQuery, **kwargs: any
    ) -> any:
        """
        Execute a read transaction using the specified strategy.

        Args:
            session (Session): Neo4j session for executing the transaction.
            strategy (Callable): Strategy function for query execution.
            query (AbstractQuery): Query object.
            **kwargs: Additional arguments for the query.

        Returns:
            any: Result of the read transaction.
        """
        return session.execute_read(lambda tx: strategy(tx, query, **kwargs))

    @staticmethod
    def _execute_write(
        session: Session, strategy: Callable, query: AbstractQuery, **kwargs: any
    ) -> any:
        """
        Execute a write transaction using the specified strategy.

        Args:
            session (Session): Neo4j session for executing the transaction.
            strategy (Callable): Strategy function for query execution.
            query (AbstractQuery): Query object.
            **kwargs: Additional arguments for the query.

        Returns:
            any: Result of the write transaction.
        """
        return session.execute_write(lambda tx: strategy(tx, query, **kwargs))

    def _execute_query(
        self, tx, query: Type[AbstractQuery], **kwargs: any
    ) -> Union[Result, list[Record]]:
        """
        Execute a query using a transaction.

        Args:
            tx: Transaction object.
            query (Type[AbstractQuery]): Query object.
            **kwargs: Additional arguments for the query.

        Returns:
            Union[Result, list[Record]]: Result of the query execution.
        """
        query_object = query(**kwargs)
        query_object.graph_schema = self.schema
        query_string = query_object.get_query()
        if query.parameters_embedded:
            return self._execute_parameters_embedded_query(tx, query_string)
        else:
            return self._execute_parameters_not_embedded_query(
                tx, query_string, **kwargs
            )

    @staticmethod
    def _execute_parameters_not_embedded_query(tx, query: str, **kwargs: any) -> Result:
        """ "
        Execute a query with parameters passed at runtime.

        Args:
            tx: Transaction object.
            query (str): Query string.
            **kwargs: Additional arguments for the query.

        Returns:
            Result: Result of the query execution.
        """
        result = tx.run(query, **kwargs)
        return result

    @staticmethod
    def _execute_parameters_embedded_query(tx, query: list[str]) -> list[Record]:
        """
        Execute a query with parameters embedded in the query string.

        Args:
            tx: Transaction object.
            query (list[str]): Query string with embedded parameters.

        Returns:
            list[Record]: Result of the query execution.
        """
        results = []
        for line in query:
            result = tx.run(line)
            results.extend(result.to_eager_result().records)
        return results

    def _retrieve_graph_strategy(
        self, tx, query: Type[AbstractQuery], **kwargs: any
    ) -> DataContainer:
        """
        Strategy for retrieving graph data.

        Args:
            tx: Transaction object.
            query (Type[AbstractQuery]): Query object.
            **kwargs: Additional arguments for the query.

        Returns:
            DataContainer: Formatted result as a data container.
        """
        result = self._execute_query(tx, query, **kwargs)
        return format_result(result, self.schema.base_nodes["chemical_equation"])

    def _modify_graph_strategy(
        self, tx, query: Type[AbstractQuery], **kwargs: any
    ) -> list:
        """
        Strategy for modifying graph data.

        Args:
            tx: Transaction object.
            query (Type[AbstractQuery]): Query object.
            **kwargs: Additional arguments for the query.

        Returns:
            list: List of records resulting from the modification.
        """
        result = self._execute_query(tx, query, **kwargs)
        return [record for record in result]

    def _retrieve_stats_strategy(
        self, tx, query: Type[AbstractQuery], **kwargs: any
    ) -> pd.DataFrame:
        """
        Strategy for retrieving statistical data.

        Args:
            tx: Transaction object.
            query (Type[AbstractQuery]): Query object.
            **kwargs: Additional arguments for the query.

        Returns:
            pd.DataFrame: DataFrame containing statistical results.
        """
        result = self._execute_query(tx, query, **kwargs)
        return pd.DataFrame([dict(record) for record in result])
