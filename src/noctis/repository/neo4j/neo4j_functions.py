import pandas as pd
import warnings
import ast

from noctis.data_architecture.datamodel import GraphRecord
from noctis.data_architecture.datacontainer import DataContainer
from pathlib import Path
from typing import Union, List


def _convert_datacontainer_to_query(data_container: DataContainer) -> list[str]:
    """To convert a DataContainer into a query string"""
    queries = []
    for record in data_container.records:
        queries.extend(_convert_record_to_query_neo4j(record))
    return queries


def _convert_record_to_query_neo4j(record: GraphRecord) -> list[str]:
    """To convert a GraphRecord into a query string"""
    queries = []
    queries.extend(_create_node_queries(record.nodes))
    queries.extend(_create_relationship_queries(record.relationships))
    return list(queries)


def _create_node_queries(nodes: list) -> list[str]:
    """To create node queries"""
    queries = []
    for node in nodes:
        query = f'MERGE (:{node.node_label} {{uid: "{node.uid}",smiles: "{node.properties["smiles"]}"}})\n'
        queries.append(query)
    return list(queries)


def _create_relationship_queries(relationships: list) -> list[str]:
    """To create relationship query"""
    queries = []
    for relationship in relationships:
        query_relationship = f'MATCH (sn:{relationship.start_node.node_label} {{uid: "{relationship.start_node.uid}"}})\n'
        query_relationship += f'MATCH (en:{relationship.end_node.node_label} {{uid: "{relationship.end_node.uid}"}})\n'
        query_relationship += f"MERGE (sn)-[:{relationship.relationship_type}]->(en)\n"
        queries.append(query_relationship)
    return list(queries)


def _generate_properties_assignment(properties: list[str]) -> str:
    """To generate properties assignment"""
    assignments = []
    for field in properties:
        assignments.append(f"{field}: apoc.convert.fromJsonMap(row.properties).{field}")
    return ", ".join(assignments)


def _get_dict_keys_from_csv(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Extract the "properties" column
    properties_series = df["properties"].apply(ast.literal_eval)

    # Get the keys of the dictionaries
    all_keys = [list(d.keys()) for d in properties_series]
    unique_keys = set().union(*all_keys)

    # Check if any dictionary is missing keys
    for keys in all_keys:
        if set(keys) != unique_keys:
            warnings.warn(
                f"Some dictionaries are missing keys: {unique_keys - set(keys)}",
                UserWarning,
            )
            break

    return list(unique_keys)


def _create_neo4j_import_path(directory: Union[str, Path], file_name: str) -> str:
    """
    Combines the directory and file name into an absolute file path and converts it into
    a file URI suitable for use in a Cypher LOAD CSV command.

    Parameters:
        directory (Union[str, Path]): The directory containing the file.
        file_name (str): The name of the file.

    Returns:
        str: The file URI that can be fed into a Cypher command.
    """
    file_path = Path(directory) / file_name
    return file_path.resolve().as_uri()


def _generate_files_string(
    folder_path: Union[str, Path, None],
    prefix: Union[str, None],
    items: List[str],
    item_type: str,
) -> str:
    """
    Generates a string of file descriptors for nodes or relationships.

    Parameters:
        folder_path (Union[str, Path, None]): The path to the folder containing the CSV files.
        prefix (Union[str, None]): The prefix to be added to the CSV file names.
        items (List[str]): The list of node labels or relationship types.
        item_type (str): Either 'labels' or 'types' to specify whether we're dealing with nodes or relationships.

    Returns:
        str: A string of file descriptors joined by commas.
    """
    query = []
    for item in items:
        csv_name = f"{prefix + '_' if prefix else ''}{item.upper()}.csv"
        if not folder_path:
            file_uri = f"file:/{csv_name}"
        else:
            file_uri = _create_neo4j_import_path(folder_path, csv_name)
        query.append(f"{{fileName:'{file_uri}', {item_type}:[]}}")
    return ", ".join(query)


def _generate_nodes_files_string(
    folder_path: Union[str, Path, None],
    prefix_nodes: Union[str, None],
    nodes_labels: List[str],
) -> str:
    """
    Generates a string of file descriptors for nodes.

    Parameters:
        folder_path (Union[str, Path, None]): The path to the folder containing the CSV files.
        prefix_nodes (Union[str, None]): The prefix to be added to the node CSV file names.
        nodes_labels (List[str]): The list of node labels.

    Returns:
        str: A string of file descriptors for nodes joined by commas.
    """
    return _generate_files_string(folder_path, prefix_nodes, nodes_labels, "labels")


def _generate_relationships_files_string(
    folder_path: Union[str, Path, None],
    prefix_relationships: Union[str, None],
    relationships_types: List[str],
) -> str:
    """
    Generates a string of file descriptors for relationships.

    Parameters:
        folder_path (Union[str, Path, None]): The path to the folder containing the CSV files.
        prefix_relationships (Union[str, None]): The prefix to be added to the relationship CSV file names.
        relationships_types (List[str]): The list of relationship types.

    Returns:
        str: A string of file descriptors for relationships joined by commas.
    """
    return _generate_files_string(
        folder_path, prefix_relationships, relationships_types, "types"
    )
