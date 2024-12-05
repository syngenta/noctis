# from noctis.data_transformation.preprocessing.reaction_preprocessing import (
#     ReactionPreProcessor,
# )
from noctis.data_transformation.preprocessing.core_graph_builder import (
    build_core_graph,
    ValidatedStringBuilder,
    UnvalidatedStringBuilder,
)

from noctis.data_architecture.graph_schema import GraphSchema

from noctis.data_architecture.datamodel import Node, Relationship
from noctis.utilities import console_logger

logger = console_logger(__name__)


class GraphExpander:
    def __init__(self, schema: GraphSchema):
        self.schema = schema
        self.nodes = {}
        self.relationships = {}

    def expand_reaction_step(
        self, step_dict: dict[str, dict], input_format, output_format, validation
    ) -> tuple[dict[str:dict], dict[str:dict]]:
        # expand core schema
        if validation:
            processor = ValidatedStringBuilder(
                input_format=input_format, output_format=output_format
            )

            base_nodes, base_relationships = build_core_graph(
                reaction_data=step_dict[self.schema.base_nodes["chemical_equation"]],
                builder=processor,
            )
        else:
            processor = UnvalidatedStringBuilder(input_format=input_format)

            base_nodes, base_relationships = build_core_graph(
                reaction_data=step_dict[self.schema.base_nodes["chemical_equation"]],
                builder=processor,
            )

        self.nodes.update(base_nodes)
        self.relationships.update(base_relationships)

        # expand extra nodes
        self._expand_extra_nodes(step_dict)

        # expand extra relationships
        self._expand_extra_relationships()

        return self.nodes, self.relationships

    def _expand_extra_nodes(self, step_dict):
        for tag, label in self.schema.extra_nodes.items():
            if label not in step_dict:
                logger.warning(
                    f"Node with label '{label}' is missing in step_dict. Skipping this node."
                )
                continue
            node = step_dict[
                label
            ].copy()  # Create a copy to avoid modifying the original
            node = Node(
                uid=step_dict[label]["uid"],
                node_label=label,
                properties=step_dict[label]["properties"],
            )
            self.nodes.setdefault(tag, []).append(node)

    def _expand_extra_relationships(self):
        for tag, relationship_schema in self.schema.extra_relationships.items():
            start_node = relationship_schema["start_node"]
            end_node = relationship_schema["end_node"]

            if start_node not in self.nodes:
                logger.warning(
                    f"Start node '{start_node}' is missing for relationship '{tag}'. Skipping this relationship."
                )
                continue

            if end_node not in self.nodes:
                logger.warning(
                    f"End node '{end_node}' is missing for relationship '{tag}'. Skipping this relationship."
                )
                continue

            for node in self.nodes[start_node]:
                for another_node in self.nodes[end_node]:
                    relationship = Relationship(
                        relationship_type=relationship_schema["type"],
                        start_node=node,
                        end_node=another_node,
                    )
                    self.relationships.setdefault(tag, []).append(relationship)
