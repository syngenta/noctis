import pandas as pd
import os
import csv
import shutil

from noctis import settings
from noctis.data_architecture.graph_schema import GraphSchema
from noctis.utilities import console_logger, timeit
from typing import Union, Optional
from noctis.data_architecture.datamodel import (
    Node,
    Relationship,
    GraphRecord,
)
from noctis.data_architecture.datacontainer import DataContainer


logger = console_logger(__name__)


def _update_partition_dict_with_row(
    target_dict: dict[str:list], source_dict: dict[str:list]
):
    """
    Update the target dictionary with values from the source dictionary.

    Args:
        target_dict (dict[str, list]): The dictionary to be updated.
        source_dict (dict[str, list]): The dictionary providing new values.

    Note:
        If a key exists in the target dictionary, values are extended; otherwise, the key is added.
    """
    for key, values in source_dict.items():
        if key in target_dict:
            target_dict[key].extend(values)
        else:
            target_dict[key] = values


@timeit(logger)
def _save_dataframes_to_partition_csv(
    dict_nodes: dict[str, pd.DataFrame],
    dict_relationships: dict[str, pd.DataFrame],
    graph_schema: GraphSchema,
    output_dir: str,
    partition_num: int,
) -> None:
    """
    Save node and relationship DataFrames to partitioned CSV files.

    Args:
        dict_nodes (dict[str, pd.DataFrame]): Dictionary of node DataFrames.
        dict_relationships (dict[str, pd.DataFrame]): Dictionary of relationship DataFrames.
        graph_schema (GraphSchema): Schema to determine labels and types.
        output_dir (str): Directory to save the partitioned CSV files.
        partition_num (int): Partition number for file naming.

    Note:
        CSV files are saved in a directory specific to the partition number.
    """
    output_file_dir = os.path.join(output_dir, f"partition_{partition_num}")
    os.makedirs(output_file_dir, exist_ok=True)
    for key, value in dict_nodes.items():
        node_label = graph_schema.get_node_label_by_tag(key)
        filename_nodes = os.path.join(output_file_dir, f"{node_label.upper()}.csv")
        value.to_csv(filename_nodes, index=False)
    for key, value in dict_relationships.items():
        relationship_type = graph_schema.get_relationship_type_by_tag(key)
        filename_relationships = os.path.join(
            output_file_dir, f"{relationship_type.upper()}.csv"
        )
        value.to_csv(filename_relationships, index=False)


def _save_list_to_partition_csv(
    my_list: list, header: list[str], output_dir: str, name: str, partition_num: int
) -> None:
    """
    Save a list to a CSV file with a header in a partition-specific directory.

    Args:
        my_list (list): List of items to save.
        header (list[str]): Header for the CSV file.
        output_dir (str): Directory to save the CSV file.
        name (str): Name of the CSV file.
        partition_num (int): Partition number for directory naming.

    Note:
        Creates the directory if it doesn't exist and writes the header followed by list items.
    """
    output_file_dir = os.path.join(output_dir, f"partition_{partition_num}")
    os.makedirs(output_file_dir, exist_ok=True)  # Create directory if it doesn't exist

    output_file = os.path.join(output_file_dir, f"{name}.csv")
    with open(output_file, "w", newline="\n") as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write header as a single-item list
        for item in my_list:
            writer.writerow(item)


def _merge_partition_files(
    filename: str,
    tmp_dir: str,
    output_dir: str,
    n_partitions: int,
    prefix: Optional[str] = None,
):
    """
    Merge partition files into a single CSV file.

    Args:
        filename (str): Name of the file to merge.
        tmp_dir (str): Temporary directory containing partition files.
        output_dir (str): Directory to save the merged CSV file.
        n_partitions (int): Number of partitions to merge.
        prefix (Optional[str]): Optional prefix for the merged file name.

    Note:
        Writes headers only once and logs warnings if partition files are missing.
    """
    # Determine the final filename with optional prefix
    os.makedirs(output_dir, exist_ok=True)
    final_filename = f"{prefix}_{filename}" if prefix else filename
    final_csv = os.path.join(output_dir, final_filename)

    header_written = False

    with open(final_csv, "w") as outfile:
        for batch_num in range(n_partitions):
            batch_dir = os.path.join(tmp_dir, f"partition_{batch_num}")
            batch_file = os.path.join(batch_dir, filename)

            if os.path.exists(batch_file):
                with open(batch_file) as infile:
                    header = infile.readline()
                    if not header_written:
                        outfile.write(header)
                        header_written = True

                    outfile.write(infile.read())
            else:
                logger.warning(f"Warning: File not found - {batch_file}")

    logger.info(f"Merged CSV file created: {final_csv}")


def create_noctis_relationship(
    mol_node: Node, ce_node: Node, role: str
) -> dict[str : Union[str, dict]]:
    """
    Create a noctis relationship based on its type.

    Args:
        mol_node (Node): Molecule node involved in the relationship.
        ce_node (Node): Chemical equation node involved in the relationship.
        role (str): Role of the molecule in the relationship ("reactants" or "products").

    Returns:
        Relationship: The created relationship object.

    Note:
        Determines the relationship type based on the role and constructs the relationship accordingly.
    """
    if role == "reactants":
        relationship_type = settings.relationships.relationship_reactant
        start_node, end_node = mol_node, ce_node
    else:  # product
        relationship_type = settings.relationships.relationship_product
        start_node, end_node = ce_node, mol_node
    return Relationship(
        relationship_type=relationship_type,
        start_node=start_node,
        end_node=end_node,
        properties={},
    )


def _delete_tmp_folder(tmp_folder: str):
    """
    Delete a temporary folder and log the outcome.

    Args:
        tmp_folder (str): Path to the temporary folder to delete.

    Note:
        Logs success or error messages based on the deletion outcome.
    """
    if os.path.exists(tmp_folder):
        try:
            shutil.rmtree(tmp_folder)
            logger.info(f"Successfully deleted temporary directory: {tmp_folder}")
        except Exception as e:
            logger.error(f"Error deleting temporary directory {tmp_folder}: {str(e)}")
    else:
        logger.warning(f"Temporary directory does not exist: {tmp_folder}")


def create_noctis_node(node_uid: str, node_label: str, properties: dict) -> Node:
    """
    Create a noctis node.

    Args:
        node_uid (str): Unique identifier for the node.
        node_label (str): Label for the node.
        properties (dict): Properties associated with the node.

    Returns:
        Node: The created node object.
    """
    return Node(
        uid=node_uid,
        node_label=node_label,
        properties=properties,
    )


def explode_smiles_like_reaction_string(
    reaction_string: str,
) -> tuple[list[str], list[str]]:
    """
    Explode a SMILES-like reaction string into reactants and products.

    Args:
        reaction_string (str): SMILES-like reaction string to explode.

    Returns:
        tuple[list[str], list[str]]: Lists of reactants and products.
    """
    reactants, _, products = reaction_string.split(">")
    reactants = reactants.split(".")
    products = products.split(".")
    return reactants, products


def explode_v3000_reaction_string(reaction_string: str) -> tuple[list[str], list[str]]:
    """
    Placeholder function for exploding V3000 reaction strings.

    Args:
        reaction_string (str): V3000 reaction string to explode.

    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError


def dict_to_list(d: dict[str, list]) -> list:
    """
    Convert a dictionary of lists into a single list.

    Args:
        d (dict[str, list]): Dictionary of lists to convert.

    Returns:
        list: Flattened list containing all values from the dictionary.
    """
    return [item for sublist in d.values() for item in sublist]


def create_data_container(
    nodes: list[Node], relationships: list[Relationship], ce_label: str
) -> DataContainer:
    """
    Create a DataContainer object from nodes and relationships.

    Args:
        nodes (list[Node]): List of nodes to include in the container.
        relationships (list[Relationship]): List of relationships to include in the container.
        ce_label (str): Label for the chemical equation.

    Returns:
        DataContainer: The created data container object.
    """
    graph_record = GraphRecord(nodes=nodes, relationships=relationships)
    data_container = DataContainer()
    data_container.set_ce_label(ce_label)
    data_container.add_record(graph_record)
    return data_container
