from noctis.data_architecture.graph_schema import GraphSchema
from noctis.data_transformation.preprocessing.reaction_preprocessing import (
    ReactionPreProcessor,
)
from noctis.data_architecture.datamodel import Node, Relationship
from noctis.data_architecture.graph_schema import GraphSchema


class GraphExpander:
    def __init__(self, schema: GraphSchema):
        self.schema = schema
        self.nodes = {}
        self.relationships = {}

    def expand_from_csv(
        self, step_dict: dict, input_format, output_format, validation
    ) -> tuple[dict[str:dict], dict[str:dict]]:
        # expand core schema
        if validation:
            # dict[str:[Node]], dict[str:[Relationships]]
            (
                base_nodes,
                base_relationships,
            ) = ReactionPreProcessor.build_from_string_w_validation(
                step_dict[self.schema.base_nodes["chemical_equation"]],
                input_format,
                output_format,
            )
        else:
            base_nodes, base_relationships = ReactionPreProcessor.build_from_string(
                step_dict[self.schema.base_nodes["chemical_equation"]],
                input_format,
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
            node = step_dict[
                label
            ].copy()  # Create a copy to avoid modifying the original
            node["label"] = label
            self.nodes.setdefault(tag, []).append(node)

    def _expand_extra_relationships(self):
        for tag, relationship_schema in self.schema.extra_relationships.items():
            start_node = relationship_schema["start_node"]
            end_node = relationship_schema["end_node"]
            for node in self.nodes[start_node]:
                for another_node in self.nodes[end_node]:
                    relationship = {
                        "type": relationship_schema["type"],
                        "start_node": node,
                        "end_node": another_node,
                    }
                    self.relationships.setdefault(tag, []).append(relationship)
