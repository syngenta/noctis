from typing import Dict
from pydantic import BaseModel, Field, field_validator
from noctis import settings


class GraphSchemaValidationError(ValueError):
    """Error to be raised when a problem with the validation of a GraphSchema object occurs"""


class GraphSchema(BaseModel):
    """Base model representing the schema of a NOCtis graph"""

    base_nodes: dict[str, str] = Field(
        default={
            "chemical_equation": settings.nodes.node_chemequation,
            "molecule": settings.nodes.node_molecule,
        },
        min_length=2,
    )
    base_relationships: dict[str, dict[str, str]] = Field(
        default={
            "product": {
                "type": settings.relationships.relationship_product,
                "start_node": "chemical_equation",
                "end_node": "molecule",
            },
            "reactant": {
                "type": settings.relationships.relationship_reactant,
                "start_node": "molecule",
                "end_node": "chemical_equation",
            },
        },
        min_length=2,
    )
    extra_nodes: dict[str, str] = Field(default_factory=dict)
    extra_relationships: dict[str, dict[str, str]] = Field(default_factory=dict)

    @field_validator("base_nodes")
    @classmethod
    def validate_base_nodes(cls, v):
        """To validate the base nodes"""
        if "chemical_equation" not in v or "molecule" not in v:
            raise GraphSchemaValidationError(
                "'chemical_equation' and 'molecule' node types are mandatory"
            )
        return v

    @field_validator("base_relationships")
    @classmethod
    def validate_base_relationships(cls, v):
        """To validate the base relationships"""
        if "product" not in v or "reactant" not in v:
            raise GraphSchemaValidationError(
                "'product' and 'reactant' relationships are mandatory"
            )
        for rel in v.values():
            if not all(key in rel for key in ["type", "start_node", "end_node"]):
                raise GraphSchemaValidationError(
                    f"Invalid relationship structure: {rel}"
                )
        return v

    @field_validator("extra_relationships")
    @classmethod
    def validate_extra_relationships(cls, v, info):
        """To validate the extra relationships"""
        all_node_types = set(info.data.get("base_nodes", {}).keys()) | set(
            info.data.get("extra_nodes", {}).keys()
        )
        for rel in v.values():
            if not all(key in rel for key in ["type", "start_node", "end_node"]):
                raise GraphSchemaValidationError(
                    f"Invalid relationship structure: {rel}"
                )
            if (
                rel["start_node"] not in all_node_types
                or rel["end_node"] not in all_node_types
            ):
                raise GraphSchemaValidationError(
                    f"Invalid node type in relationship: {rel}"
                )
        return v

    @classmethod
    def build_from_dict(cls, data: dict) -> "GraphSchema":
        """To build a GraphSchema object from a dictionary"""
        return cls.model_validate(data)

    def _is_existing_node_type(self, node_tag: str) -> bool:
        """To check whether a node type is already defined"""
        return node_tag in self.base_nodes or node_tag in self.extra_nodes
