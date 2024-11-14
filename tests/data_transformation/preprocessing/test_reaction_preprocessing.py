from noctis.data_transformation.preprocessing.reaction_preprocessing import (
    ReactionPreProcessor,
    NoneChemicalEquation,
    UnavailableFormat,
)
from linchemin.cheminfo.models import ChemicalEquation
from unittest.mock import patch, Mock
import pytest


def test_string_to_ce():
    preprocessor = ReactionPreProcessor()
    reaction_string = "CC(=O)O.[Na+].[OH-]>>CC(=O)[O-].[Na+].O"
    ce = preprocessor.reaction_string_to_ce(
        reaction_string=reaction_string, input_format="smiles"
    )
    assert isinstance(ce, ChemicalEquation)


def test_check_format():
    with pytest.raises(UnavailableFormat):
        ReactionPreProcessor()._check_format("unavailable_ftm")


@patch(
    "noctis.data_transformation.preprocessing.reaction_preprocessing.ChemicalEquationConstructor.build_from_reaction_string"
)
def test_string_to_ce_invalid(mock_constructor):
    mock_constructor.return_value = None
    reaction_string = "CC(=O)O"
    with pytest.raises(NoneChemicalEquation):
        ReactionPreProcessor().reaction_string_to_ce(
            reaction_string=reaction_string, input_format="smiles"
        )


def test_ce_string_to_dict():
    ce_string = "CNOC(O)=O.Cl>>CNCl"
    ce_uid = 123
    processor = ReactionPreProcessor()
    processor.output_format = "smiles"
    d = processor._ce_string_to_dict(reaction_string=ce_string, reaction_uid=ce_uid)
    expected_uid = "C" + str(ce_uid)
    expected_label = "ChemicalEquation"
    expected_properties = {"smiles": ce_string}
    assert d["node_label"] == expected_label
    assert d["uid"] == expected_uid
    assert d["properties"] == expected_properties


@patch(
    "noctis.data_transformation.preprocessing.reaction_preprocessing.cif.rdrxn_to_string"
)
def test_ce_to_dict(mock_cif):
    mock_equation = Mock()
    mock_equation.uid = "CHEM123"
    mock_equation.smiles = "C1=CC>>CC=C1"
    mock_cif.return_value = mock_equation.smiles
    expected_uid = "C" + mock_equation.uid
    expected_label = "ChemicalEquation"
    expected_properties = {"smiles": mock_equation.smiles}
    processor = ReactionPreProcessor()
    processor.output_format = "smiles"
    d = processor._ce_to_dict(mock_equation)
    assert d["node_label"] == expected_label
    assert d["uid"] == expected_uid
    assert d["properties"] == expected_properties


def test_mol_string_to_dict():
    mol_string = "Cl~[#7]-[#6]"
    mol_uid = 123
    expected_uid = "M" + str(mol_uid)
    expected_label = "Molecule"
    expected_properties = {"smarts": "Cl~[#7]-[#6]"}
    processor = ReactionPreProcessor()
    processor.output_format = "smarts"
    d = processor._mol_string_to_dict(mol_string=mol_string, mol_uid=mol_uid)
    assert d["node_label"] == expected_label
    assert d["uid"] == expected_uid
    assert d["properties"] == expected_properties


def test_mol_to_dict():
    mock_mol = Mock()
    mock_mol.uid = "MOL123"
    mock_mol.rdmol = Mock()
    mock_compute_func = Mock(return_value="Cl~[#7]-[#6]")
    with patch.dict(
        "noctis.data_transformation.preprocessing.reaction_preprocessing.ReactionPreProcessor.reaction_mol_format_map",
        {"smarts": mock_compute_func},
        clear=True,
    ):
        processor = ReactionPreProcessor()
        processor.output_format = "smarts"
        d = processor._mol_to_dict(mock_mol)

        expected_uid = "M" + mock_mol.uid
        expected_label = "Molecule"
        expected_properties = {"smarts": "Cl~[#7]-[#6]"}

        assert d["node_label"] == expected_label
        assert d["uid"] == expected_uid
        assert d["properties"] == expected_properties

        # Verify that our mock function was called with the correct argument
        mock_compute_func.assert_called_once_with(mock_mol.rdmol)


def test_serialize_relationship():
    relationship_type = "test_relationship"
    start_node = {"uid": "123", "node_type": "starting_node"}
    end_node = {"uid": "456", "node_type": "ending_node"}
    processor = ReactionPreProcessor()
    relationship_dict = processor._serialize_relationship(
        relationship_type, start_node, end_node
    )
    assert relationship_dict == {
        "relationship_type": "test_relationship",
        "start_node": {"uid": "123", "node_type": "starting_node"},
        "end_node": {"uid": "456", "node_type": "ending_node"},
        "properties": {},
    }
    properties = {"prop1": "value1", "prop2": "value2"}
    relationship_dict = processor._serialize_relationship(
        relationship_type, start_node, end_node, properties
    )
    assert relationship_dict["properties"] == properties


def test_handle_relationship():
    ce_node = {"node_label": "ChemicalEquation", "uid": "ce_uid", "properties": {}}
    mol_node = {"node_label": "Molecule", "uid": "mol_uid", "properties": {}}
    processor = ReactionPreProcessor()
    relationship_reactant = processor._handle_relationship(
        mol_node=mol_node, ce_node=ce_node, role="reactants"
    )

    assert relationship_reactant == {
        "relationship_type": "REACTANT",
        "start_node": mol_node,
        "end_node": ce_node,
        "properties": {},
    }
    relationship_reactant = processor._handle_relationship(
        mol_node=mol_node, ce_node=ce_node, role="products"
    )

    assert relationship_reactant == {
        "relationship_type": "PRODUCT",
        "start_node": ce_node,
        "end_node": mol_node,
        "properties": {},
    }


def create_mock_molecule(uid, smiles):
    mock_molecule = Mock()
    mock_molecule.uid = uid
    mock_molecule.rdmol = Mock()
    mock_molecule.smiles = smiles
    return mock_molecule


@pytest.mark.parametrize(
    "relationship_type, expected_relationship",
    [
        (
            "reactants",
            {
                "start_key": "start_node",
                "end_key": "end_node",
                "relationship_type": "REACTANT",
            },
        ),
        (
            "products",
            {
                "start_key": "end_node",
                "end_key": "start_node",
                "relationship_type": "PRODUCT",
            },
        ),
    ],
)
def test_expand_molecules(relationship_type, expected_relationship):
    # Create a list of mock Molecule objects
    mock_molecules = [
        create_mock_molecule("MOL1", "C1=CC=CC=C1"),
        create_mock_molecule("MOL2", "CCO"),
        create_mock_molecule("MOL3", "CN=C=O"),
    ]
    ce_dict = {"node_label": "ChemicalEquation", "uid": "ce_uid", "properties": {}}
    processor = ReactionPreProcessor()
    processor.output_format = "smiles"
    mock_compute_func = Mock(side_effect=["C1=CC=CC=C1", "CCO", "CN=C=O"])
    with patch.dict(
        "noctis.data_transformation.preprocessing.reaction_preprocessing.ReactionPreProcessor.reaction_mol_format_map",
        {"smiles": mock_compute_func},
        clear=True,
    ):
        nodes, relationships = processor._expand_molecules(
            mock_molecules, ce_dict, relationship_type
        )

        expected_nodes = [
            {
                "node_label": "Molecule",
                "properties": {"smiles": mol.smiles},
                "uid": "M" + mol.uid,
            }
            for mol in mock_molecules
        ]
        assert len(nodes) == len(expected_nodes)
        for actual_node, expected_node in zip(nodes, expected_nodes):
            assert actual_node == expected_node

        assert len(relationships) == len(mock_molecules)
        for relationship in relationships:
            assert (
                relationship["relationship_type"]
                == expected_relationship["relationship_type"]
            )

        if relationship_type == "reactants":
            assert all(
                relationship["end_node"] == ce_dict for relationship in relationships
            )
            assert all(
                relationship["start_node"] in [node for node in nodes]
                for relationship in relationships
            )
        else:  # products
            assert all(
                relationship["start_node"] == ce_dict for relationship in relationships
            )
            assert all(
                relationship["end_node"] in [node for node in nodes]
                for relationship in relationships
            )


@pytest.mark.parametrize(
    "relationship_type, expected_relationship",
    [
        (
            "reactants",
            {
                "start_key": "start_node",
                "end_key": "end_node",
                "relationship_type": "REACTANT",
            },
        ),
        (
            "products",
            {
                "start_key": "end_node",
                "end_key": "start_node",
                "relationship_type": "PRODUCT",
            },
        ),
    ],
)
def test_expand_string_molecules(relationship_type, expected_relationship):
    # Create a list of mock Molecule objects
    molecules = [
        {"mol_string": "C1=CC=CC=C1", "mol_uid": 1},
        {"mol_string": "CCO", "mol_uid": 2},
        {"mol_string": "CN=C=O", "mol_uid": 3},
    ]

    ce_dict = {"node_label": "ChemicalEquation", "uid": "ce_uid", "properties": {}}
    processor = ReactionPreProcessor()
    processor.output_format = "smiles"
    with patch(
        "noctis.data_transformation.preprocessing.reaction_preprocessing.create_hash",
        side_effect=[d["mol_uid"] for d in molecules],
    ):
        mol_strings = [d["mol_string"] for d in molecules]
        nodes, relationships = processor._expand_mol_strings(
            mol_strings, ce_dict, relationship_type
        )

        expected_nodes = [
            {
                "node_label": "Molecule",
                "properties": {"smiles": d["mol_string"]},
                "uid": "M" + str(d["mol_uid"]),
            }
            for d in molecules
        ]
        assert len(nodes) == len(expected_nodes)
        for actual_node, expected_node in zip(nodes, expected_nodes):
            assert actual_node == expected_node

        assert len(relationships) == len(molecules)
        for relationship in relationships:
            assert (
                relationship["relationship_type"]
                == expected_relationship["relationship_type"]
            )

        if relationship_type == "reactants":
            assert all(
                relationship["end_node"] == ce_dict for relationship in relationships
            )
            assert all(
                relationship["start_node"] in [node for node in nodes]
                for relationship in relationships
            )
        else:  # products
            assert all(
                relationship["start_node"] == ce_dict for relationship in relationships
            )
            assert all(
                relationship["end_node"] in [node for node in nodes]
                for relationship in relationships
            )


def test_ce_to_noctis():
    mock_ce = Mock(spec=ChemicalEquation)
    mock_ce.get_reactants.return_value = ["reactant1", "reactant2"]
    mock_ce.get_products.return_value = ["product1", "product2"]

    processor = ReactionPreProcessor()
    processor._check_format = Mock()
    processor._ce_to_dict = Mock(
        return_value={"node_label": "ChemicalEquation", "uid": "ce_uid"}
    )
    processor._expand_molecules = Mock(
        side_effect=[
            (
                [
                    {"node_label": "Molecule", "uid": "r1"},
                    {"node_label": "Molecule", "uid": "r2"},
                ],
                [
                    {
                        "start_node": {"uid": "r1"},
                        "end_node": {"uid": "ce_uid"},
                        "relationship_type": "REACTANT",
                    },
                    {
                        "start_node": {"uid": "r2"},
                        "end_node": {"uid": "ce_uid"},
                        "relationship_type": "REACTANT",
                    },
                ],
            ),
            (
                [
                    {"node_label": "Molecule", "uid": "p1"},
                    {"node_label": "Molecule", "uid": "p2"},
                ],
                [
                    {
                        "start_node": {"uid": "ce_uid"},
                        "end_node": {"uid": "p1"},
                        "relationship_type": "PRODUCT",
                    },
                    {
                        "start_node": {"uid": "ce_uid"},
                        "end_node": {"uid": "p2"},
                        "relationship_type": "PRODUCT",
                    },
                ],
            ),
        ]
    )
    nodes, relationships = processor.ce_to_noctis(mock_ce, "smiles")

    processor._check_format.assert_called_once_with("smiles")
    assert processor.output_format == "smiles"

    processor._ce_to_dict.assert_called_once_with(mock_ce)

    assert processor._expand_molecules.call_count == 2
    processor._expand_molecules.assert_any_call(
        ["reactant1", "reactant2"],
        {"node_label": "ChemicalEquation", "uid": "ce_uid"},
        "reactants",
    )
    processor._expand_molecules.assert_any_call(
        ["product1", "product2"],
        {"node_label": "ChemicalEquation", "uid": "ce_uid"},
        "products",
    )

    # Check the final results
    expected_nodes = [
        {"node_label": "ChemicalEquation", "uid": "ce_uid"},
        {"node_label": "Molecule", "uid": "r1"},
        {"node_label": "Molecule", "uid": "r2"},
        {"node_label": "Molecule", "uid": "p1"},
        {"node_label": "Molecule", "uid": "p2"},
    ]
    expected_relationships = [
        {
            "start_node": {"uid": "r1"},
            "end_node": {"uid": "ce_uid"},
            "relationship_type": "REACTANT",
        },
        {
            "start_node": {"uid": "r2"},
            "end_node": {"uid": "ce_uid"},
            "relationship_type": "REACTANT",
        },
        {
            "start_node": {"uid": "ce_uid"},
            "end_node": {"uid": "p1"},
            "relationship_type": "PRODUCT",
        },
        {
            "start_node": {"uid": "ce_uid"},
            "end_node": {"uid": "p2"},
            "relationship_type": "PRODUCT",
        },
    ]

    assert nodes == expected_nodes
    assert relationships == expected_relationships


@patch(
    "noctis.data_transformation.preprocessing.reaction_preprocessing.explode_v3000_reaction_string"
)
@patch(
    "noctis.data_transformation.preprocessing.reaction_preprocessing.explode_smiles_like_reaction_string"
)
def test_explode_reaction(mock_smiles_like, mock_v3000):
    processor = ReactionPreProcessor()
    processor.input_format = "smiles"
    ce_string = "CNOC(O)=O.Cl>>CNCl"
    processor._explode_reaction(ce_string)
    mock_smiles_like.assert_called_once_with(ce_string)

    processor.input_format = "rxn_blockV3K"
    ce_string = "CNOC(O)=O.Cl>>CNCl"
    processor._explode_reaction(ce_string)
    mock_v3000.assert_called_once_with(ce_string)


@patch("noctis.data_transformation.preprocessing.reaction_preprocessing.create_hash")
def test_reaction_string_to_noctis(mock_create_hash):
    # Setup
    reaction_string = "CNOC(O)=O.Cl>>CNCl"
    input_format = "smiles"
    output_format = "smiles"

    # Mock the create_hash function
    mock_create_hash.return_value = "mocked_reaction_uid"

    # Create the ReactionPreProcessor instance
    processor = ReactionPreProcessor()

    # Mock the internal methods
    processor._check_format = Mock()
    processor._ce_string_to_dict = Mock(
        return_value={"node_label": "ChemicalEquation", "uid": "ce_uid"}
    )
    processor._explode_reaction = Mock(return_value=(["CNOC(O)=O", "Cl"], ["CNCl"]))
    processor._expand_mol_strings = Mock(
        side_effect=[
            (
                [
                    {"node_label": "Molecule", "uid": "r1"},
                    {"node_label": "Molecule", "uid": "r2"},
                ],
                [
                    {
                        "start_node": {"uid": "r1"},
                        "end_node": {"uid": "ce_uid"},
                        "relationship_type": "REACTANT",
                    },
                    {
                        "start_node": {"uid": "r2"},
                        "end_node": {"uid": "ce_uid"},
                        "relationship_type": "REACTANT",
                    },
                ],
            ),
            (
                [
                    {"node_label": "Molecule", "uid": "p1"},
                ],
                [
                    {
                        "start_node": {"uid": "ce_uid"},
                        "end_node": {"uid": "p1"},
                        "relationship_type": "PRODUCT",
                    },
                ],
            ),
        ]
    )

    # Call the method
    nodes, relationships = processor.reaction_string_to_noctis(
        reaction_string, input_format, output_format
    )

    # Assertions
    processor._check_format.assert_any_call(input_format)
    processor._check_format.assert_any_call(output_format)
    assert processor._check_format.call_count == 2
    assert processor.input_format == input_format
    assert processor.output_format == output_format

    mock_create_hash.assert_called_once_with(reaction_string)

    processor._ce_string_to_dict.assert_called_once_with(
        reaction_string=reaction_string, reaction_uid="mocked_reaction_uid"
    )

    processor._explode_reaction.assert_called_once_with(reaction_string)

    assert processor._expand_mol_strings.call_count == 2
    processor._expand_mol_strings.assert_any_call(
        ["CNOC(O)=O", "Cl"],
        {"node_label": "ChemicalEquation", "uid": "ce_uid"},
        "reactants",
    )
    processor._expand_mol_strings.assert_any_call(
        ["CNCl"], {"node_label": "ChemicalEquation", "uid": "ce_uid"}, "products"
    )

    # Check the final results
    expected_nodes = [
        {"node_label": "ChemicalEquation", "uid": "ce_uid"},
        {"node_label": "Molecule", "uid": "r1"},
        {"node_label": "Molecule", "uid": "r2"},
        {"node_label": "Molecule", "uid": "p1"},
    ]
    expected_relationships = [
        {
            "start_node": {"uid": "r1"},
            "end_node": {"uid": "ce_uid"},
            "relationship_type": "REACTANT",
        },
        {
            "start_node": {"uid": "r2"},
            "end_node": {"uid": "ce_uid"},
            "relationship_type": "REACTANT",
        },
        {
            "start_node": {"uid": "ce_uid"},
            "end_node": {"uid": "p1"},
            "relationship_type": "PRODUCT",
        },
    ]

    assert nodes == expected_nodes
    assert relationships == expected_relationships
