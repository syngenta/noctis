import pytest
from unittest.mock import Mock, patch, MagicMock
from neo4j import GraphDatabase, Session, Driver
from neo4j.exceptions import Neo4jError
from noctis.repository.neo4j.neo4j_repository import Neo4jRepository
from noctis.repository.neo4j.neo4j_queries import Neo4jQueryRegistry, AbstractQuery
import pandas as pd


@pytest.fixture
def mock_driver():
    return Mock(spec=Driver)


@pytest.fixture
def mock_session():
    return Mock(spec=Session)


@pytest.fixture
def neo4j_repository(mock_driver):
    with patch(
        "noctis.repository.neo4j.neo4j_repository.GraphDatabase.driver",
        return_value=mock_driver,
    ):
        repo = Neo4jRepository("bolt://localhost:7687", "neo4j", "password")
        yield repo
        repo.close()


def test_init(neo4j_repository):
    assert isinstance(neo4j_repository._driver, Mock)
    GraphDatabase.driver.assert_called_once_with(
        "bolt://localhost:7687", auth=("neo4j", "password"), database=None
    )


def test_close(neo4j_repository):
    neo4j_repository.close()
    neo4j_repository._driver.close.assert_called_once()


@pytest.mark.parametrize(
    "query_type, expected_strategy",
    [
        ("retrieve_graph", "_retrieve_graph_strategy"),
        ("modify_graph", "_modify_graph_strategy"),
        ("retrieve_stats", "_retrieve_stats_strategy"),
    ],
)
def test_execute_query(neo4j_repository, mock_session, query_type, expected_strategy):
    mock_query = Mock(spec=AbstractQuery)
    mock_query.query_type = query_type

    with patch.object(Neo4jQueryRegistry, "get_query_object", return_value=mock_query):
        with patch.object(neo4j_repository, "_session_context") as mock_context:
            mock_context.return_value.__enter__.return_value = mock_session
            with patch.object(neo4j_repository, expected_strategy) as mock_strategy:
                neo4j_repository.execute_query("test_query", param1="value1")

    if query_type in ["retrieve_graph", "retrieve_stats"]:
        mock_session.execute_read.assert_called_once()
    else:
        mock_session.execute_write.assert_called_once()


def test_execute_query_unsupported_type(neo4j_repository):
    mock_query = Mock(spec=AbstractQuery)
    mock_query.query_type = "unsupported_type"

    with patch.object(Neo4jQueryRegistry, "get_query_object", return_value=mock_query):
        with pytest.raises(
            ValueError, match="Unsupported query type: unsupported_type"
        ):
            neo4j_repository.execute_query("test_query")


def test_session_context(neo4j_repository, mock_session):
    neo4j_repository._driver.session.return_value = mock_session

    with neo4j_repository._session_context() as session:
        assert session == mock_session

    mock_session.close.assert_called_once()


def test_session_context_neo4j_error(neo4j_repository, mock_session):
    neo4j_repository._driver.session.return_value = mock_session
    mock_session.close.side_effect = Neo4jError("Test error")

    with pytest.raises(Neo4jError):
        with neo4j_repository._session_context():
            raise Neo4jError("Test error")

    mock_session.close.assert_called_once()


def test_session_context_generic_error(neo4j_repository, mock_session):
    neo4j_repository._driver.session.return_value = mock_session

    with pytest.raises(Exception):
        with neo4j_repository._session_context():
            raise Exception("Generic error")

    mock_session.close.assert_called_once()


def test_execute_read(neo4j_repository, mock_session):
    mock_strategy = Mock()
    mock_query = Mock(spec=AbstractQuery)

    neo4j_repository._execute_read(
        mock_session, mock_strategy, mock_query, param1="value1"
    )

    mock_session.execute_read.assert_called_once()


def test_execute_write(neo4j_repository, mock_session):
    mock_strategy = Mock()
    mock_query = Mock(spec=AbstractQuery)

    neo4j_repository._execute_write(
        mock_session, mock_strategy, mock_query, param1="value1"
    )

    mock_session.execute_write.assert_called_once()


@pytest.mark.parametrize("is_parameterized", [True, False])
def test_execute_query_method(neo4j_repository, is_parameterized):
    mock_tx = Mock()
    mock_query = Mock(spec=AbstractQuery)
    mock_query.is_parameterized = is_parameterized
    mock_query_object = Mock()
    mock_query_object.get_query.return_value = "MOCK QUERY"
    mock_query.return_value = mock_query_object

    with patch.object(
        neo4j_repository,
        (
            "_execute_parametrized_query"
            if is_parameterized
            else "_execute_non_parametrized_query"
        ),
    ) as mock_executor:
        neo4j_repository._execute_query(mock_tx, mock_query, param1="value1")

    mock_executor.assert_called_once()


def test_execute_non_parametrized_query():
    mock_tx = Mock()
    result = Neo4jRepository._execute_non_parametrized_query(
        mock_tx, "MOCK QUERY", param1="value1"
    )
    mock_tx.run.assert_called_once_with("MOCK QUERY", param1="value1")


def test_execute_parametrized_query():
    mock_tx = Mock()
    mock_tx.run.return_value.single.return_value = "MOCK RESULT"
    result = Neo4jRepository._execute_parametrized_query(mock_tx, ["QUERY1", "QUERY2"])
    assert mock_tx.run.call_count == 2
    assert result == ["MOCK RESULT", "MOCK RESULT"]


@pytest.mark.parametrize(
    "strategy_name",
    ["_retrieve_graph_strategy", "_modify_graph_strategy", "_retrieve_stats_strategy"],
)
def test_query_strategies(neo4j_repository, strategy_name):
    mock_tx = Mock()
    mock_query = Mock(spec=AbstractQuery)
    mock_result = Mock()  # This will simulate the neo4j.Result object
    mock_df = pd.DataFrame({"column": [1, 2, 3]})  # Mock DataFrame

    # Set up mock_result.to_df() to return the mock DataFrame
    mock_result.to_df.return_value = mock_df

    with patch.object(
        neo4j_repository, "_execute_query", return_value=mock_result
    ) as mock_execute_query:
        if strategy_name == "_retrieve_graph_strategy":
            with patch(
                "noctis.repository.neo4j.neo4j_repository.format_result"
            ) as mock_format_result:
                mock_format_result.return_value = "Formatted Result"
                strategy = getattr(neo4j_repository, strategy_name)
                result = strategy(mock_tx, mock_query, param1="value1")
                mock_format_result.assert_called_once_with(mock_result)
                assert result == "Formatted Result"
        else:
            strategy = getattr(neo4j_repository, strategy_name)
            result = strategy(mock_tx, mock_query, param1="value1")
            assert isinstance(result, pd.DataFrame)
            pd.testing.assert_frame_equal(result, mock_df)

    mock_execute_query.assert_called_once_with(mock_tx, mock_query, param1="value1")

    if strategy_name != "_retrieve_graph_strategy":
        mock_result.to_df.assert_called_once()
