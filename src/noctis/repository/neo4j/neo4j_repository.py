from abc import ABC, abstractmethod
import pandas as pd
from contextlib import contextmanager
from typing import Optional, Type, Callable, Union

from neo4j import Driver, GraphDatabase, Session
from neo4j.exceptions import Neo4jError
from pandas.core.interchange.dataframe_protocol import DataFrame

from noctis.data_architecture.datamodel import DataContainer
from noctis.data_transformation.neo4j.neo4j_formatter import format_result
from noctis.repository.neo4j.neo4j_queries import (
    AbstractQuery,
    Neo4jQueryRegistry,
    CustomQuery,
)

from neo4j import Result, Record


class Neo4jRepository:
    def __init__(
        self, uri: str, username: str, password: str, database: Optional[str] = None
    ):
        self._driver = GraphDatabase.driver(
            uri, auth=(username, password), database=database
        )
        self._query_strategies = {
            "retrieve_graph": (self._retrieve_graph_strategy, self._execute_read),
            "modify_graph": (self._modify_graph_strategy, self._execute_write),
            "retrieve_stats": (self._retrieve_stats_strategy, self._execute_read),
        }

    def close(self) -> None:
        self._driver.close()

    @contextmanager
    def _session_context(self) -> Session:
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

    def create_constraints(self):
        query = Neo4jQueryRegistry.get_query_object("create_uniqueness_constraints")
        with self._session_context() as session:
            for query in query().get_query():
                session.run(query)

    def drop_constraints(self):
        query = Neo4jQueryRegistry.get_query_object("drop_uniqueness_constraints")
        with self._session_context() as session:
            for query in query().get_query():
                session.run(query)

    def execute_custom_query_from_yaml(
        self, yaml_file: str, query_name: str, **kwargs: any
    ):
        query = CustomQuery.from_yaml(yaml_file=yaml_file, query_name=query_name)
        return self._build_query_execution_strategy(query, **kwargs)

    def execute_query(self, query_name: str, **kwargs: any):
        query = Neo4jQueryRegistry.get_query_object(query_name)
        return self._build_query_execution_strategy(query, **kwargs)

    def _build_query_execution_strategy(
        self, query: Union[Type[AbstractQuery], CustomQuery], **kwargs: any
    ):
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
        return session.execute_read(lambda tx: strategy(tx, query, **kwargs))

    @staticmethod
    def _execute_write(
        session: Session, strategy: Callable, query: AbstractQuery, **kwargs: any
    ) -> any:
        return session.execute_write(lambda tx: strategy(tx, query, **kwargs))

    def _execute_query(
        self, tx, query: Type[AbstractQuery], **kwargs: any
    ) -> Union[Result, list[Record]]:
        query_object = query(**kwargs)
        query_string = query_object.get_query()
        if query.is_parameterized:
            return self._execute_parametrized_query(tx, query_string)
        else:
            return self._execute_non_parametrized_query(tx, query_string, **kwargs)

    @staticmethod
    def _execute_non_parametrized_query(tx, query: str, **kwargs: any) -> Result:
        """To execute a query whose arguments need to be passed at run time"""
        result = tx.run(query, **kwargs)
        # print("Query is done. Now formatting")
        return result

    @staticmethod
    def _execute_parametrized_query(tx, query: list[str]) -> list[Record]:
        """To execute a query whose arguments are embedded in the query string"""
        results = []
        for line in query:
            result = tx.run(line)
            results.extend(result.to_eager_result().records)
        return results

    def _retrieve_graph_strategy(
        self, tx, query: Type[AbstractQuery], **kwargs: any
    ) -> DataContainer:
        result = self._execute_query(tx, query, **kwargs)
        return format_result(result)

    def _modify_graph_strategy(
        self, tx, query: Type[AbstractQuery], **kwargs: any
    ) -> list:
        result = self._execute_query(tx, query, **kwargs)
        return [record for record in result]

    def _retrieve_stats_strategy(
        self, tx, query: Type[AbstractQuery], **kwargs: any
    ) -> pd.DataFrame:
        result = self._execute_query(tx, query, **kwargs)
        return pd.DataFrame([dict(record) for record in result])
