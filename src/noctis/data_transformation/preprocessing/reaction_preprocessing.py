from linchemin.cheminfo.models import ChemicalEquation, Molecule
from linchemin.cheminfo.constructors import ChemicalEquationConstructor
import linchemin.cheminfo.functions as cif
from linchemin.utilities import create_hash
from noctis import settings
from typing import Union, Optional

from noctis.data_architecture.datamodel import DataModelAttributes
from noctis.utilities import console_logger
from noctis.data_transformation.preprocessing.utils import (
    explode_smiles_like_reaction_string,
    explode_v3000_reaction_string,
)

logger = console_logger(__name__)


class NoneChemicalEquation(Exception):
    """To be raised if a ChemicalEquation object is None"""


class UnavailableFormat(ValueError):
    """To be raised if the selected format is not available"""


class ReactionPreProcessor:
    """Class to manipulate reaction strings"""

    input_format: Optional[str] = None
    output_format: Optional[str] = None
    reaction_mol_format_map = {
        "smiles": cif.compute_mol_smiles,
        "smarts": cif.compute_mol_smarts,
        "rxn_blockV3K": cif.compute_mol_blockV3k,
        "rxn_blockV2K": cif.compute_mol_block,
    }

    def reaction_string_to_ce(
        self, reaction_string: str, input_format: str
    ) -> ChemicalEquation:
        """To convert a reaction string into a ChemicalEquation object"""
        self._check_format(input_format)

        ce_constructor = ChemicalEquationConstructor()
        ce = ce_constructor.build_from_reaction_string(
            reaction_string=reaction_string, inp_fmt=input_format
        )
        if ce is None:
            logger.error("The returned ChemicalEquation object is None")
            raise NoneChemicalEquation
        return ce

    def ce_to_noctis(
        self, chemical_equation: ChemicalEquation, output_format: str
    ) -> tuple[list[dict], list[dict]]:
        """To convert a ChemicalEquation object into a tuple of NOCtis dictionaries"""
        self._check_format(output_format)
        self.output_format = output_format
        nodes = []
        relationships = []

        chemical_equation_dict = self._ce_to_dict(chemical_equation)
        nodes.append(chemical_equation_dict)

        reactants = chemical_equation.get_reactants()
        reactant_nodes, reactant_relationships = self._expand_molecules(
            reactants, chemical_equation_dict, "reactants"
        )
        nodes.extend(reactant_nodes)
        relationships.extend(reactant_relationships)

        products = chemical_equation.get_products()
        product_nodes, product_relationships = self._expand_molecules(
            products, chemical_equation_dict, "products"
        )
        nodes.extend(product_nodes)
        relationships.extend(product_relationships)

        return nodes, relationships

    def _check_format(self, selected_format: str) -> None:
        """To check if the specified format is among the supported formats"""
        if selected_format not in self.reaction_mol_format_map:
            logger.error(f"{selected_format} is not available.")
            raise UnavailableFormat

    def _ce_to_dict(
        self, chemical_equation: ChemicalEquation
    ) -> dict[str : Union[str, dict]]:
        """To convert a ChemicalEquation into a noctis dictionary"""
        reaction_string = cif.rdrxn_to_string(
            chemical_equation.rdrxn, out_fmt=self.output_format
        )
        return self._ce_string_to_dict(reaction_string, chemical_equation.uid)

    def _ce_string_to_dict(
        self, reaction_string: str, reaction_uid: int
    ) -> dict[str : Union[str, dict]]:
        """To convert a reaction string into a noctis"""
        reaction_uid = "C" + str(reaction_uid)
        return {
            DataModelAttributes.NODE_LABEL: settings.nodes.node_chemequation,
            DataModelAttributes.UID: reaction_uid,
            DataModelAttributes.NODE_PROPERTIES: {self.output_format: reaction_string},
        }

    def _mol_to_dict(self, molecule: Molecule) -> dict[str : Union[str, dict]]:
        """To convert a Molecule into a noctis dictionary"""
        func = self.reaction_mol_format_map[self.output_format]
        mol_string = func(molecule.rdmol)
        return self._mol_string_to_dict(mol_string, molecule.uid)

    def _mol_string_to_dict(
        self, mol_string: str, mol_uid: int
    ) -> dict[str : Union[str, dict]]:
        """To convert a molecule string into a noctis dictionary"""
        mol_uid = "M" + str(mol_uid)
        return {
            DataModelAttributes.NODE_LABEL: settings.nodes.node_molecule,
            DataModelAttributes.UID: mol_uid,
            DataModelAttributes.NODE_PROPERTIES: {self.output_format: mol_string},
        }

    def _expand_molecules(
        self, mol_list: list[Molecule], chemical_equation_dict: dict, role: str
    ) -> tuple[list[dict], list[dict]]:
        """To expand a list of molecules (LCIN)
        into nodes and relationships based on their role"""
        nodes = []
        relationships = []
        for molecule in mol_list:
            molecule_node = self._mol_to_dict(molecule)
            nodes.append(molecule_node)
            relationship = self._handle_relationship(
                mol_node=molecule_node, ce_node=chemical_equation_dict, role=role
            )
            relationships.append(relationship)
        return nodes, relationships

    @staticmethod
    def _serialize_relationship(
        relationship_type: str,
        start_node: dict,
        end_node: dict,
        properties: Optional[dict] = None,
    ) -> dict[str : Union[str, dict]]:
        """To generate a relationship as noctis dictionary"""
        if properties is None:
            properties = {}
        return {
            DataModelAttributes.RELATIONSHIP_TYPE: relationship_type,
            DataModelAttributes.START_NODE: start_node,
            DataModelAttributes.END_NODE: end_node,
            DataModelAttributes.RELATIONSHIP_PROPERTIES: properties,
        }

    def reaction_string_to_noctis(
        self, reaction_string: str, input_format: str, output_format: str
    ) -> tuple[list[dict], list[dict]]:
        """To convert a reaction string into a tuple of noctis dictionaries"""
        self._check_format(input_format)
        self.input_format = input_format
        self._check_format(output_format)
        self.output_format = output_format

        nodes = []
        relationships = []

        reaction_uid = create_hash(reaction_string)
        chemical_equation_dict = self._ce_string_to_dict(
            reaction_string=reaction_string, reaction_uid=reaction_uid
        )
        nodes.append(chemical_equation_dict)
        reactants, products = self._explode_reaction(reaction_string)

        reactant_nodes, reactant_relationships = self._expand_mol_strings(
            reactants, chemical_equation_dict, "reactants"
        )
        nodes.extend(reactant_nodes)
        relationships.extend(reactant_relationships)

        product_nodes, product_relationships = self._expand_mol_strings(
            products, chemical_equation_dict, "products"
        )
        nodes.extend(product_nodes)
        relationships.extend(product_relationships)

        return nodes, relationships

    def _explode_reaction(self, reaction_string: str) -> tuple[list[str], list[str]]:
        """To explode a reaction string into reactants and products based on the string format"""
        if self.input_format in ["smiles", "smarts"]:
            return explode_smiles_like_reaction_string(reaction_string)
        return explode_v3000_reaction_string(reaction_string)

    def _expand_mol_strings(
        self, mol_list: list[str], chemical_equation_dict: dict, role: str
    ):
        """To expand a list of molecule strings
        into a set of nodes and relationships based on their role"""
        nodes = []
        relationships = []
        for molecule in mol_list:
            molecule_uid = create_hash(molecule)
            molecule_node = self._mol_string_to_dict(
                mol_string=molecule, mol_uid=molecule_uid
            )
            nodes.append(molecule_node)
            relationship = self._handle_relationship(
                molecule_node, chemical_equation_dict, role
            )
            relationships.append(relationship)
        return nodes, relationships

    def _handle_relationship(
        self, mol_node: dict, ce_node: dict, role: str
    ) -> dict[str : Union[str, dict]]:
        """To initialize a relationship based on its type"""
        if role == "reactants":
            relationship_type = settings.relationships.relationship_reactant
            start_node, end_node = mol_node, ce_node
        else:  # product
            relationship_type = settings.relationships.relationship_product
            start_node, end_node = ce_node, mol_node
        return self._serialize_relationship(
            relationship_type=relationship_type,
            start_node=start_node,
            end_node=end_node,
        )
