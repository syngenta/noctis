import pandas as pd
import os
import csv
import json
from typing import Union

from rdkit.Chem.QED import properties

from noctis import settings
from noctis.data_architecture.datamodel import Node, Relationship


def _update_partition_dict_with_row(
    target_dict: dict[str:list], source_dict: dict[str:list]
):
    for key, values in source_dict.items():
        if key in target_dict:
            target_dict[key].extend(values)
        else:
            target_dict[key] = values


def _build_dataframes_from_dict(data_dict: dict[str, dict]) -> dict[str, pd.DataFrame]:
    data_df = {}
    for key, values in data_dict.items():
        data_df[key] = pd.DataFrame(values)
    return data_df


def _save_dataframes_to_partition_csv(
    dict_nodes: dict[str, pd.DataFrame],
    dict_relationships: dict[str, pd.DataFrame],
    output_dir: str,
    partition_num: int,
) -> None:
    output_file_dir = os.path.join(output_dir, f"partition_{partition_num}")
    os.makedirs(output_file_dir, exist_ok=True)
    for key, value in dict_nodes.items():
        filename_nodes = os.path.join(output_file_dir, f"{key.upper()}.csv")
        value.to_csv(filename_nodes, index=False)
    for key, value in dict_relationships.items():
        filename_relationships = os.path.join(output_file_dir, f"{key.upper()}.csv")
        value.to_csv(filename_relationships, index=False)


def save_list_to_partition_csv(my_list, output_dir, name, partition_num):
    output_file = os.path.join(output_dir, f"partition_{partition_num}", f"{name}.csv")
    with open(output_file, "w", newline="\n") as file:
        writer = csv.writer(file)
        for item in my_list:
            writer.writerow(item)


def create_noctis_relationship(
    mol_node: Node, ce_node: Node, role: str
) -> dict[str : Union[str, dict]]:
    """To create a noctis relationship based on its type"""
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


def create_noctis_node(node_uid: str, node_label: str, properties: dict) -> Node:
    """To create a noctis node"""
    return Node(
        uid=node_uid,
        node_label=node_label,
        properties=properties,
    )


def explode_smiles_like_reaction_string(
    reaction_string: str,
) -> tuple[list[str], list[str]]:
    reactants, _, products = reaction_string.split(">")
    reactants = reactants.split(".")
    products = products.split(".")
    return reactants, products


def explode_v3000_reaction_string(reaction_string: str) -> tuple[list[str], list[str]]:
    raise NotImplementedError
