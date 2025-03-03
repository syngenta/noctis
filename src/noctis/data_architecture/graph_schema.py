from pydantic import BaseModel, Field, field_validator, ConfigDict
from noctis import settings
from typing import Literal

import json

import yaml

from noctis.utilities import console_logger

logger = console_logger(__name__)


class GraphSchemaValidationError(ValueError):
    """Error to be raised when a problem with the validation of a GraphSchema object occurs"""


class GraphSchema(BaseModel):
    """Base model representing the schema of a NOCtis graph"""

    model_config = ConfigDict(extra="forbid")
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

    @classmethod
    def _build_from_json(cls, data: str) -> "GraphSchema":
        """To build a GraphSchema object from a json file"""
        try:
            with open(data) as f:
                data = json.load(f)
        except FileNotFoundError:
            raise GraphSchemaValidationError(f"File '{data}' not found")
        except json.JSONDecodeError:
            raise GraphSchemaValidationError(f"Invalid JSON file '{data}'")
        except Exception as e:
            raise GraphSchemaValidationError(f"Error reading file '{data}': {e}")
        if not isinstance(data, dict):
            raise GraphSchemaValidationError(f"Invalid JSON file '{data}'")

        return cls.build_from_dict(data)

    @classmethod
    def _build_from_yaml(cls, data: str) -> "GraphSchema":
        """To build a GraphSchema object from a yaml file"""
        try:
            with open(data) as f:
                data = yaml.safe_load(f)
        except FileNotFoundError:
            raise GraphSchemaValidationError(f"File '{data}' not found")
        except yaml.YAMLError as e:
            raise GraphSchemaValidationError(f"Invalid YAML file '{data}': {e}")
        except Exception as e:
            raise GraphSchemaValidationError(f"Error reading file '{data}': {e}")
        if not isinstance(data, dict):
            raise GraphSchemaValidationError(f"Invalid YAML file '{data}'")

        return cls.build_from_dict(data)

    @classmethod
    def build_from_file(
        cls, file_path: str, file_format: Literal["json", "yaml"] = "json"
    ):
        """To build a GraphSchema object from a file"""
        if file_format == "json":
            return cls._build_from_json(file_path)
        elif file_format == "yaml":
            return cls._build_from_yaml(file_path)
        else:
            raise GraphSchemaValidationError(f"Invalid format '{file_format}'")

    def _is_existing_node_type(self, node_tag: str) -> bool:
        """To check whether a node type is already defined"""
        return node_tag in self.base_nodes or node_tag in self.extra_nodes

    def get_nodes_labels(self) -> list[str]:
        labels = []
        for node in self.base_nodes.values():
            labels.append(node)
        for node in self.extra_nodes.values():
            labels.append(node)
        return labels

    def get_relationships_types(self) -> list[str]:
        types = []
        for tag, relationship_schema in self.base_relationships.items():
            types.append(relationship_schema["type"])
        for tag, relationship_schema in self.extra_relationships.items():
            types.append(relationship_schema["type"])
        return types

    def get_node_label_by_tag(self, node_tag: str) -> str:
        """To get the label of a node type"""
        if node_tag in self.base_nodes:
            return self.base_nodes[node_tag]
        elif node_tag in self.extra_nodes:
            return self.extra_nodes[node_tag]
        else:
            raise GraphSchemaValidationError(f"Node type '{node_tag}' not found")

    def get_relationship_type_by_tag(self, relationship_tag: str) -> str:
        """To get the type of a relationship"""
        if relationship_tag in self.base_relationships:
            return self.base_relationships[relationship_tag]["type"]
        elif relationship_tag in self.extra_relationships:
            return self.extra_relationships[relationship_tag]["type"]
        else:
            raise GraphSchemaValidationError(
                f"Relationship type '{relationship_tag}' not found"
            )

    def save_to_file(
        self, file_path: str, file_format: Literal["json", "yaml"] = "json"
    ) -> None:
        """To save the schema to a file"""
        if file_format == "json":
            self._save_to_json(file_path)
        elif file_format == "yaml":
            self._save_to_yaml(file_path)
        else:
            raise GraphSchemaValidationError(f"Invalid format '{file_format}'")

    def _save_to_json(self, file_path: str) -> None:
        """To save the schema to a json file"""
        with open(file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=4)

            logger.info(f"Schema saved to {file_path}")

    def _save_to_yaml(self, file_path: str) -> None:
        """To save the schema to a yaml file"""
        with open(file_path, "w") as f:
            yaml.dump(self.model_dump(), f)

            logger.info(f"Schema saved to {file_path}")

    def __str__(self) -> str:
        """Returns formatted string representation of the GraphSchema object in a YAML-like format."""
        base_nodes_str = "\n".join(
            [f"  {key}: {value}" for key, value in self.base_nodes.items()]
        )
        extra_nodes_str = "\n".join(
            [f"  {key}: {value}" for key, value in self.extra_nodes.items()]
        )
        base_relationships_str = "\n".join(
            [
                f"  {key}:\n    end_node: {value['end_node']}\n    start_node: {value['start_node']}\n    type: {value['type']}"
                for key, value in self.base_relationships.items()
            ]
        )
        extra_relationships_str = "\n".join(
            [
                f"  {key}:\n    end_node: {value['end_node']}\n    start_node: {value['start_node']}\n    type: {value['type']}"
                for key, value in self.extra_relationships.items()
            ]
        )

        return (
            f"GraphSchema:\n"
            f"base_nodes:\n{base_nodes_str}\n"
            f"base_relationships:\n{base_relationships_str}\n"
            f"extra_nodes:\n{extra_nodes_str}\n"
            f"extra_relationships:\n{extra_relationships_str}\n"
        )
