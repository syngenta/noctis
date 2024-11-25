from noctis.data_transformation.preprocessing.core_graph_builder import (
    FormatValidator,
    UnavailableFormat,
    ValidatedStringBuilder,
    UnvalidatedStringBuilder,
    build_core_graph,
    NoneChemicalEquation,
)
from noctis.data_transformation.preprocessing.utils import (
    explode_smiles_like_reaction_string,
    explode_v3000_reaction_string,
)
import pytest
from unittest.mock import Mock, patch, call, PropertyMock
from noctis.data_architecture.datamodel import Node, Relationship
import linchemin.cheminfo.functions as cif
from noctis import settings
from linchemin.cheminfo.models import ChemicalEquation
from linchemin.cheminfo.constructors import ChemicalEquationConstructor


@pytest.fixture
def mocked_reaction_data():
    return {}


def create_mock_molecule(uid, smiles):
    mock_molecule = Mock()
    mock_molecule.uid = uid
    mock_molecule.rdmol = Mock()
    mock_molecule.smiles = smiles
    return mock_molecule


def test_format_validator_validation():
    FormatValidator.validate_format("smiles")

    with pytest.raises(UnavailableFormat):
        FormatValidator.validate_format("unavailable_ftm")


def test_get_mol_to_string_function_valid_format():
    assert (
        FormatValidator.get_mol_to_string_function("smiles") == cif.compute_mol_smiles
    )
    assert (
        FormatValidator.get_mol_to_string_function("smarts") == cif.compute_mol_smarts
    )
    assert (
        FormatValidator.get_mol_to_string_function("rxn_blockV3K")
        == cif.compute_mol_blockV3k
    )
    assert (
        FormatValidator.get_mol_to_string_function("rxn_blockV2K")
        == cif.compute_mol_block
    )


def test_validated_builder():
    builder = ValidatedStringBuilder(input_format="smiles", output_format="smarts")
    assert builder
    assert builder.input_format == "smiles"
    assert builder.output_format == "smarts"
    assert builder.mol_to_string == cif.compute_mol_smarts


@pytest.fixture
def validated_string_builder():
    return ValidatedStringBuilder(input_format="smiles", output_format="smarts")


@pytest.fixture
def mock_chemical_equation():
    ce = Mock(spec=ChemicalEquation)
    ce.uid = "12345"
    return ce


@pytest.fixture
def mock_reaction_data():
    return {
        "properties": {"prop1": "value1", "prop2": "value2"},
        "smiles": "CC(=O)O.O>>CC(O)=O",
    }


def test_handle_chemical_equation(
    validated_string_builder, mock_chemical_equation, mock_reaction_data
):
    validated_string_builder._ce_to_string = Mock(return_value="C>>O")

    with patch(
        "noctis.data_transformation.preprocessing.core_graph_builder.create_noctis_node"
    ) as mock_create_node:
        mock_node = Mock(spec=Node)
        mock_create_node.return_value = mock_node

        result = validated_string_builder._handle_chemical_equation(
            mock_chemical_equation, mock_reaction_data
        )

        assert result == mock_node
        validated_string_builder._ce_to_string.assert_called_once_with(
            chemical_equation=mock_chemical_equation
        )
        mock_create_node.assert_called_once_with(
            "C12345",
            settings.nodes.node_chemequation,
            {"prop1": "value1", "prop2": "value2", "smarts": "C>>O"},
        )


def test_handle_chemical_equation_with_custom_uid(
    validated_string_builder, mock_chemical_equation, mock_reaction_data
):
    mock_reaction_data["uid"] = "custom_uid_123"
    validated_string_builder._ce_to_string = Mock(return_value="C>>O")

    with patch(
        "noctis.data_transformation.preprocessing.core_graph_builder.create_noctis_node"
    ) as mock_create_node:
        with patch(
            "noctis.data_transformation.preprocessing.core_graph_builder.logger.warning"
        ) as mock_logger_warning:
            mock_node = Mock(spec=Node)
            mock_create_node.return_value = mock_node
            result = validated_string_builder._handle_chemical_equation(
                mock_chemical_equation, mock_reaction_data
            )

            assert result == mock_node
            mock_logger_warning.assert_called_once_with(
                "Found custom uid. It will be included in the properties."
            )
            mock_create_node.assert_called_once_with(
                "C12345",
                settings.nodes.node_chemequation,
                {
                    "prop1": "value1",
                    "prop2": "value2",
                    "custom_uid": "custom_uid_123",
                    "smarts": "C>>O",
                },
            )


@pytest.fixture
def mock_chemical_equation_node():
    return Mock(spec=Node)


@pytest.fixture
def mock_molecules():
    return [
        create_mock_molecule("MOL1", "C1=CC=CC=C1"),
        create_mock_molecule("MOL2", "CCO"),
        create_mock_molecule("MOL3", "CN=C=O"),
    ]


def test_expand_molecules(
    validated_string_builder, mock_chemical_equation_node, mock_molecules
):
    validated_string_builder.mol_to_string = Mock(side_effect=["C", "CC", "CCC"])

    with patch(
        "noctis.data_transformation.preprocessing.core_graph_builder.create_noctis_node"
    ) as mock_create_node:
        with patch(
            "noctis.data_transformation.preprocessing.core_graph_builder.create_noctis_relationship"
        ) as mock_create_relationship:
            mock_nodes = [Mock(spec=Node) for _ in range(3)]
            mock_relationships = [Mock(spec=Relationship) for _ in range(3)]

            mock_create_node.side_effect = mock_nodes
            mock_create_relationship.side_effect = mock_relationships

            nodes, relationships = validated_string_builder._expand_molecules(
                mock_molecules, mock_chemical_equation_node, "reactants"
            )

            assert len(nodes) == 3
            assert len(relationships) == 3
            assert nodes == mock_nodes
            assert relationships == mock_relationships

            assert validated_string_builder.mol_to_string.call_count == 3

            expected_node_calls = [
                call(
                    node_uid=f"MMOL{i + 1}",
                    node_label="Molecule",
                    properties={"smarts": mol_string},
                )
                for i, mol_string in enumerate(["C", "CC", "CCC"])
            ]
            assert mock_create_node.call_args_list == expected_node_calls

            expected_relationship_calls = [
                (
                    {
                        "mol_node": mock_nodes[i],
                        "ce_node": mock_chemical_equation_node,
                        "role": "reactants",
                    },
                )
                for i in range(3)
            ]
            assert (
                mock_create_relationship.call_args_list == expected_relationship_calls
            )


def test_process_method(validated_string_builder):
    reaction_data = {"smiles": "CC(=O)O.O>>CC(O)=O", "temperature": 25}

    mock_constructor = Mock(spec=ChemicalEquationConstructor)
    mock_chemical_equation = Mock()
    mock_constructor.return_value.build_from_reaction_string.return_value = (
        mock_chemical_equation
    )

    # Mock chemical equation methods
    mock_chemical_equation.get_reactants.return_value = ["CC(=O)O", "O"]
    mock_chemical_equation.get_products.return_value = ["CC(O)=O"]

    # Mock internal methods of validatedStringBuilder
    mock_handle_ce = Mock()
    mock_expand_molecules = Mock()

    # Prepare mock return values
    mock_ce_node = Mock(spec=Node)
    mock_handle_ce.return_value = mock_ce_node

    mock_reactant_nodes = [Mock(spec=Node), Mock(spec=Node)]
    mock_reactant_relationships = [Mock(spec=Relationship), Mock(spec=Relationship)]
    mock_product_nodes = [Mock(spec=Node)]
    mock_product_relationships = [Mock(spec=Relationship)]

    mock_expand_molecules.side_effect = [
        (mock_reactant_nodes, mock_reactant_relationships),
        (mock_product_nodes, mock_product_relationships),
    ]

    # Apply patches
    with (
        patch(
            "noctis.data_transformation.preprocessing.core_graph_builder.ChemicalEquationConstructor",
            mock_constructor,
        ),
        patch.object(
            validated_string_builder, "_handle_chemical_equation", mock_handle_ce
        ),
        patch.object(
            validated_string_builder, "_expand_molecules", mock_expand_molecules
        ),
    ):
        # Call the method under test
        nodes, relationships = validated_string_builder.process(reaction_data)

    # Assertions
    mock_constructor.return_value.build_from_reaction_string.assert_called_once_with(
        reaction_string="CC(=O)O.O>>CC(O)=O", inp_fmt="smiles"
    )

    mock_handle_ce.assert_called_once_with(mock_chemical_equation, reaction_data)

    assert mock_expand_molecules.call_count == 2
    mock_expand_molecules.assert_any_call(["CC(=O)O", "O"], mock_ce_node, "reactants")
    mock_expand_molecules.assert_any_call(["CC(O)=O"], mock_ce_node, "products")

    # Check the structure of the returned data
    assert nodes == {
        "chemical_equation": [mock_ce_node],
        "molecule": mock_reactant_nodes + mock_product_nodes,
    }
    assert relationships == {
        "reactant": mock_reactant_relationships,
        "product": mock_product_relationships,
    }


def test_process_method_none_chemical_equation(validated_string_builder):
    reaction_data = {"smiles": "invalid_smiles", "temperature": 25}

    mock_constructor = Mock(spec=ChemicalEquationConstructor)
    mock_constructor.return_value.build_from_reaction_string.return_value = None

    with patch(
        "noctis.data_transformation.preprocessing.core_graph_builder.ChemicalEquationConstructor",
        mock_constructor,
    ):
        with pytest.raises(NoneChemicalEquation):
            validated_string_builder.process(reaction_data)

    mock_constructor.return_value.build_from_reaction_string.assert_called_once_with(
        reaction_string="invalid_smiles", inp_fmt="smiles"
    )


def test_unvalidated_builder():
    builder = UnvalidatedStringBuilder(input_format="smarts")
    assert builder
    assert builder.input_format == "smarts"
    assert builder.reaction_explosion_func == explode_smiles_like_reaction_string

    builder = UnvalidatedStringBuilder(input_format="rxn_blockV3K")
    assert builder
    assert builder.input_format == "rxn_blockV3K"
    assert builder.reaction_explosion_func == explode_v3000_reaction_string


@pytest.fixture
def unvalidated_string_builder():
    return UnvalidatedStringBuilder("smiles")


def test_handle_chemical_reaction_string(unvalidated_string_builder):
    reaction_data = {"smiles": "CC(=O)O.O>>CC(O)=O", "properties": {"temperature": 25}}

    with patch(
        "noctis.data_transformation.preprocessing.core_graph_builder.create_hash"
    ) as mock_hash:
        with patch(
            "noctis.data_transformation.preprocessing.core_graph_builder.create_noctis_node"
        ) as mock_create_node:
            mocked_uid = "C123456"
            mock_hash.return_value = mocked_uid
            final_properties = {"smiles": "CC(=O)O.O>>CC(O)=O", "temperature": 25}
            node = Node(
                node_label=settings.nodes.node_chemequation,
                uid=mocked_uid,
                properties=final_properties,
            )
            mock_create_node.return_value = node

            result = unvalidated_string_builder._handle_chemical_reaction_string(
                reaction_data
            )

            mock_hash.assert_called_once_with("CC(=O)O.O>>CC(O)=O")
            mock_create_node.assert_called_once_with(
                "C" + mocked_uid, settings.nodes.node_chemequation, final_properties
            )
            assert result is node


def test_expand_mol_strings(unvalidated_string_builder):
    mol_list = ["CC(=O)O", "O"]
    chemical_equation_node = Mock(name="ChemicalEquationNode")
    role = "reactant"

    with patch(
        "noctis.data_transformation.preprocessing.core_graph_builder.create_hash"
    ) as mock_hash:
        with patch(
            "noctis.data_transformation.preprocessing.core_graph_builder.create_noctis_node"
        ) as mock_create_node:
            with patch(
                "noctis.data_transformation.preprocessing.core_graph_builder.create_noctis_relationship"
            ) as mock_create_relationship:
                mock_hash.side_effect = ["123", "456"]
                mock_nodes = [Mock(name=f"MoleculeNode{i}") for i in range(2)]
                mock_create_node.side_effect = mock_nodes
                mock_relationships = [Mock(name=f"Relationship{i}") for i in range(2)]
                mock_create_relationship.side_effect = mock_relationships

                nodes, relationships = unvalidated_string_builder._expand_mol_strings(
                    mol_list, chemical_equation_node, role
                )

                mock_hash.assert_any_call("CC(=O)O")
                mock_hash.assert_any_call("O")

                assert mock_create_node.call_count == 2
                mock_create_node.assert_any_call(
                    node_uid="M123",
                    node_label=settings.nodes.node_molecule,
                    properties={"smiles": "CC(=O)O"},
                )
                mock_create_node.assert_any_call(
                    node_uid="M456",
                    node_label=settings.nodes.node_molecule,
                    properties={"smiles": "O"},
                )

                assert mock_create_relationship.call_count == 2
                for mock_node in mock_nodes:
                    mock_create_relationship.assert_any_call(
                        mol_node=mock_node, ce_node=chemical_equation_node, role=role
                    )

                assert nodes == mock_nodes
                assert relationships == mock_relationships


def test_expand_mol_strings_empty_list(unvalidated_string_builder):
    mol_list = []
    chemical_equation_node = Mock(name="ChemicalEquationNode")
    role = "product"

    nodes, relationships = unvalidated_string_builder._expand_mol_strings(
        mol_list, chemical_equation_node, role
    )

    assert nodes == []
    assert relationships == []


@patch("noctis.data_transformation.preprocessing.core_graph_builder.create_hash")
def test_build_core_graph(mock_hash, unvalidated_string_builder):
    mock_hash.side_effect = ["111", "222", "333", "444"]
    reaction_data = {"smiles": "CC(=O)O.O>>CC(O)=O", "properties": {"temperature": 25}}
    nodes, relationships = build_core_graph(
        reaction_data=reaction_data, builder=unvalidated_string_builder
    )
    assert nodes == {
        "chemical_equation": [
            Node(
                node_label="ChemicalEquation",
                uid="C111",
                properties={"temperature": 25, "smiles": "CC(=O)O.O>>CC(O)=O"},
            )
        ],
        "molecule": [
            Node(
                node_label="Molecule",
                uid="M222",
                properties={"smiles": "CC(=O)O"},
            ),
            Node(
                node_label="Molecule",
                uid="M333",
                properties={"smiles": "O"},
            ),
            Node(
                node_label="Molecule",
                uid="M444",
                properties={"smiles": "CC(O)=O"},
            ),
        ],
    }
    assert relationships == {
        "product": [
            Relationship(
                relationship_type="PRODUCT",
                start_node=Node(
                    node_label="ChemicalEquation",
                    uid="C111",
                    properties={"temperature": 25, "smiles": "CC(=O)O.O>>CC(O)=O"},
                ),
                end_node=Node(
                    node_label="Molecule",
                    uid="M444",
                    properties={"smiles": "CC(O)=O"},
                ),
                properties={},
            )
        ],
        "reactant": [
            Relationship(
                relationship_type="REACTANT",
                start_node=Node(
                    node_label="Molecule",
                    uid="M222",
                    properties={"smiles": "CC(=O)O"},
                ),
                end_node=Node(
                    node_label="ChemicalEquation",
                    uid="C111",
                    properties={"temperature": 25, "smiles": "CC(=O)O.O>>CC(O)=O"},
                ),
                properties={},
            ),
            Relationship(
                relationship_type="REACTANT",
                start_node=Node(
                    node_label="Molecule",
                    uid="M333",
                    properties={"smiles": "O"},
                ),
                end_node=Node(
                    node_label="ChemicalEquation",
                    uid="C111",
                    properties={"temperature": 25, "smiles": "CC(=O)O.O>>CC(O)=O"},
                ),
                properties={},
            ),
        ],
    }
