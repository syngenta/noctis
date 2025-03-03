import pytest
from pydantic import ValidationError
from noctis.data_architecture.graph_schema import GraphSchema


def test_graph_schema_default_initialization():
    schema = GraphSchema()
    assert "chemical_equation" in schema.base_nodes
    assert "molecule" in schema.base_nodes
    assert "product" in schema.base_relationships
    assert "reactant" in schema.base_relationships
    assert schema.extra_nodes == {}
    assert schema.extra_relationships == {}


def test_graph_schema_custom_initialization():
    custom_schema = {
        "base_nodes": {"chemical_equation": "ChemicalEquation", "molecule": "Molecule"},
        "base_relationships": {
            "product": {
                "type": "PRODUCT",
                "start_node": "chemical_equation",
                "end_node": "molecule",
            },
            "reactant": {
                "type": "REACTANT",
                "start_node": "molecule",
                "end_node": "chemical_equation",
            },
        },
        "extra_nodes": {"reaction": "Reaction"},
        "extra_relationships": {
            "catalyzes": {
                "type": "CATALYZES",
                "start_node": "molecule",
                "end_node": "chemical_equation",
            }
        },
    }
    schema = GraphSchema.build_from_dict(custom_schema)
    assert schema.base_nodes["chemical_equation"] == "ChemicalEquation"
    assert schema.base_nodes["molecule"] == "Molecule"
    assert schema.extra_nodes["reaction"] == "Reaction"
    assert "catalyzes" in schema.extra_relationships


def test_invalid_base_nodes():
    with pytest.raises(ValidationError):
        GraphSchema(base_nodes={"molecule": "Molecule"})


def test_invalid_base_relationships():
    with pytest.raises(ValidationError):
        GraphSchema(
            base_relationships={
                "product": {
                    "type": "PRODUCT",
                    "start_node": "chemical_equation",
                    "end_node": "molecule",
                }
            }
        )


def test_invalid_extra_relationships():
    with pytest.raises(ValidationError):
        GraphSchema(
            base_nodes={
                "chemical_equation": "ChemicalEquation",
                "molecule": "Molecule",
            },
            base_relationships={
                "product": {
                    "type": "PRODUCT",
                    "start_node": "chemical_equation",
                    "end_node": "molecule",
                },
                "reactant": {
                    "type": "REACTANT",
                    "start_node": "molecule",
                    "end_node": "chemical_equation",
                },
            },
            extra_relationships={
                "invalid": {
                    "type": "INVALID",
                    "start_node": "non_existent",
                    "end_node": "molecule",
                }
            },
        )


def test_is_existing_node_type():
    schema = GraphSchema()
    assert schema._is_existing_node_type("chemical_equation") is True
    assert schema._is_existing_node_type("molecule") is True
    assert schema._is_existing_node_type("non_existent") is False

    schema_with_extra = GraphSchema(extra_nodes={"reaction": "Reaction"})
    assert schema_with_extra._is_existing_node_type("reaction") is True


def test_build_from_dict():
    custom_schema = {
        "base_nodes": {"chemical_equation": "ChemicalEquation", "molecule": "Molecule"},
        "base_relationships": {
            "product": {
                "type": "PRODUCT",
                "start_node": "chemical_equation",
                "end_node": "molecule",
            },
            "reactant": {
                "type": "REACTANT",
                "start_node": "molecule",
                "end_node": "chemical_equation",
            },
        },
    }
    schema = GraphSchema.build_from_dict(custom_schema)
    assert isinstance(schema, GraphSchema)
    assert schema.base_nodes["chemical_equation"] == "ChemicalEquation"
    assert schema.base_nodes["molecule"] == "Molecule"


def test_invalid_build_from_dict():
    invalid_schema = {"base_nodes": {"chemical_equation": "ChemicalEquation"}}
    with pytest.raises(ValidationError):
        GraphSchema.build_from_dict(invalid_schema)
