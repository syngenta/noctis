import linchemin.cheminfo.functions as cif
from typing import Tuple, Dict, List

from linchemin.cheminfo.constructors import ChemicalEquationConstructor
from linchemin.cheminfo.models import ChemicalEquation, Molecule
from linchemin.utilities import create_hash

from noctis import settings
from noctis.utilities import console_logger
from noctis.data_architecture.datamodel import Node, Relationship
from noctis.data_transformation.preprocessing.utils import (
    explode_smiles_like_reaction_string,
    explode_v3000_reaction_string,
    create_noctis_relationship,
    create_noctis_node,
)

from abc import ABC, abstractmethod

logger = console_logger(__name__)


class NoneChemicalEquation(Exception):
    """To be raised if a ChemicalEquation object is None"""


class UnavailableFormat(ValueError):
    """To be raised if the selected format is not available"""


class FormatValidator:
    reaction_mol_format_map = {
        "smiles": cif.compute_mol_smiles,
        "smarts": cif.compute_mol_smarts,
        "rxn_blockV3K": cif.compute_mol_blockV3k,
        "rxn_blockV2K": cif.compute_mol_block,
    }

    @classmethod
    def validate_format(cls, selected_format: str) -> None:
        if selected_format not in cls.reaction_mol_format_map:
            raise UnavailableFormat(f"{selected_format} is not available.")

    @classmethod
    def get_mol_to_string_function(cls, fmt: str):
        cls.validate_format(fmt)
        return cls.reaction_mol_format_map[fmt]


class CoreGraphBuilder(ABC):
    @abstractmethod
    def process(
        self, reaction_data: dict
    ) -> tuple[dict[str, list[Node]], dict[str, list[Relationship]]]:
        pass


class ValidatedStringBuilder(CoreGraphBuilder):
    """Processor to handle reaction string with a cheminformatics validation"""

    def __init__(self, input_format: str, output_format: str):
        FormatValidator.validate_format(input_format)
        FormatValidator.validate_format(output_format)
        self.input_format = input_format
        self.output_format = output_format
        self.mol_to_string = FormatValidator.get_mol_to_string_function(output_format)

    def process(
        self, reaction_data: dict
    ) -> tuple[dict[str, list[Node]], dict[str, list[Relationship]]]:
        """To process a reaction string with validation"""
        nodes = {"chemical_equation": [], "molecule": []}
        relationships = {"reactant": [], "product": []}

        chemical_equation = ChemicalEquationConstructor().build_from_reaction_string(
            reaction_string=reaction_data[self.input_format], inp_fmt=self.input_format
        )
        if chemical_equation is None:
            logger.error("The returned ChemicalEquation object is None")
            raise NoneChemicalEquation

        chemical_equation_node = self._handle_chemical_equation(
            chemical_equation, reaction_data
        )
        nodes["chemical_equation"].append(chemical_equation_node)

        reactants = chemical_equation.get_reactants()
        reactant_nodes, reactant_relationships = self._expand_molecules(
            reactants, chemical_equation_node, "reactants"
        )
        nodes["molecule"].extend(reactant_nodes)
        relationships["reactant"].extend(reactant_relationships)

        products = chemical_equation.get_products()
        product_nodes, product_relationships = self._expand_molecules(
            products, chemical_equation_node, "products"
        )
        nodes["molecule"].extend(product_nodes)
        relationships["product"].extend(product_relationships)

        return nodes, relationships

    def _handle_chemical_equation(
        self, chemical_equation: ChemicalEquation, reaction_data: dict
    ) -> Node:
        """To create a noctis Node from a ChemicalEquation  object"""
        validated_string = self._ce_to_string(chemical_equation=chemical_equation)
        reaction_uid = "C" + str(chemical_equation.uid)
        properties = reaction_data["properties"]
        if "uid" in reaction_data:
            logger.warning("Found custom uid. It will be included in the properties.")
            properties.update({"custom_uid": reaction_data["uid"]})

        properties.update({self.output_format: validated_string})
        return create_noctis_node(
            reaction_uid, settings.nodes.node_chemequation, properties
        )

    def _expand_molecules(
        self, mol_list: list[Molecule], chemical_equation_node: Node, role: str
    ) -> tuple[list[Node], list[Relationship]]:
        """To expand a list of molecules (LCIN)
        into nodes and relationships based on their role"""
        nodes = []
        relationships = []
        for molecule in mol_list:
            mol_uid = "M" + str(molecule.uid)
            mol_label = settings.nodes.node_molecule
            mol_validated_string = self.mol_to_string(molecule)
            molecule_node = create_noctis_node(
                node_uid=mol_uid,
                node_label=mol_label,
                properties={self.output_format: mol_validated_string},
            )
            nodes.append(molecule_node)
            relationship = create_noctis_relationship(
                mol_node=molecule_node, ce_node=chemical_equation_node, role=role
            )
            relationships.append(relationship)
        return nodes, relationships

    def _ce_to_string(self, chemical_equation: ChemicalEquation) -> str:
        """To convert a ChemicalEquation into a string representation"""
        return cif.rdrxn_to_string(chemical_equation.rdrxn, out_fmt=self.output_format)


class UnvalidatedStringBuilder(CoreGraphBuilder):
    """Processor to handle reaction string"""

    def __init__(self, input_format: str):
        FormatValidator.validate_format(input_format)
        self.input_format = input_format
        if self.input_format in ["smiles", "smarts"]:
            self.reaction_explosion_func = explode_smiles_like_reaction_string
        else:
            self.reaction_explosion_func = explode_v3000_reaction_string

    def process(
        self, reaction_data: dict
    ) -> tuple[dict[str, list[Node]], dict[str, list[Relationship]]]:
        """To process a reaction string"""
        nodes = {"chemical_equation": [], "molecule": []}
        relationships = {"reactant": [], "product": []}

        reaction_string = reaction_data[self.input_format]
        chemical_equation_node = self._handle_chemical_reaction_string(
            reaction_data=reaction_data
        )

        nodes["chemical_equation"].append(chemical_equation_node)
        reactants, products = self.reaction_explosion_func(reaction_string)

        reactant_nodes, reactant_relationships = self._expand_mol_strings(
            reactants, chemical_equation_node, "reactants"
        )
        nodes["molecule"].extend(reactant_nodes)
        relationships["reactant"].extend(reactant_relationships)

        product_nodes, product_relationships = self._expand_mol_strings(
            products, chemical_equation_node, "products"
        )
        nodes["molecule"].extend(product_nodes)
        relationships["product"].extend(product_relationships)

        return nodes, relationships

    def _handle_chemical_reaction_string(self, reaction_data: dict) -> Node:
        """T create a noctis Node from a reaction string"""
        reaction_string = reaction_data[self.input_format]
        reaction_uid = "C" + str(create_hash(reaction_string))
        properties = reaction_data.get("properties", {})
        properties.update({self.input_format: reaction_string})
        return create_noctis_node(
            reaction_uid, settings.nodes.node_chemequation, properties
        )

    def _expand_mol_strings(
        self, mol_list: list[str], chemical_equation_node: Node, role: str
    ):
        """To expand a list of molecule strings
        into a set of nodes and relationships based on their role"""
        nodes = []
        relationships = []
        for molecule in mol_list:
            mol_uid = "M" + str(create_hash(molecule))
            mol_label = settings.nodes.node_molecule
            molecule_node = create_noctis_node(
                node_uid=mol_uid,
                node_label=mol_label,
                properties={self.input_format: molecule},
            )
            nodes.append(molecule_node)
            relationship = create_noctis_relationship(
                mol_node=molecule_node, ce_node=chemical_equation_node, role=role
            )
            relationships.append(relationship)
        return nodes, relationships


class RDLDocBuilder(CoreGraphBuilder):
    """Processor to handle reaction document"""

    def process(
        self, reaction_data: dict
    ) -> tuple[dict[str, list[Node]], dict[str, list[Relationship]]]:
        raise NotImplementedError


def build_core_graph(
    reaction_data: dict, builder: CoreGraphBuilder
) -> tuple[dict[str, list[Node]], dict[str, list[Relationship]]]:
    return builder.process(reaction_data)
