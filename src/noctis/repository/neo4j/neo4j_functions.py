import pandas as pd
import warnings

from noctis.data_architecture.datamodel import DataContainer, GraphRecord, Node


def _convert_datacontainer_to_query(data_container: DataContainer) -> list[str]:
    """To convert a DataContainer into a query string"""
    queries = []
    for record in data_container.records.values():
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
    properties_series = df["properties"]

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


# problem with apoc.import.csv -- it creates, not merges. if you try to add nodes which are already in the DB
# it will crush. For merging one can use only load csv


def _generate_nodes_files_string(prefix_nodes: str, nodes_labels: list[str]) -> str:
    query = []
    for label in nodes_labels:
        query.append(
            f"{{fileName:'file:/{prefix_nodes +'_'+ label.upper() + '.csv'}', labels:[]}}"
        )
    return ", ".join(query)


def _generate_relationships_files_string(
    prefix_relationships: str, relationships_types: list[str]
) -> str:
    query = []
    for rtype in relationships_types:
        query.append(
            f"{{fileName:'file:/{prefix_relationships +'_' + rtype.upper() + '.csv'}', types:[]}}"
        )
    return ", ".join(query)
