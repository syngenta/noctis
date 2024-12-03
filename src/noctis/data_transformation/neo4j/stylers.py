from noctis.data_transformation.data_styles.dataframe_stylers import (
    NodesRelationshipsStyle,
)


class Neo4jImportStyle(NodesRelationshipsStyle):
    COLUMN_NAMES_NODES = {
        "uid": "uid:ID",
        "node_label": ":LABEL",
    }
    COLUMN_NAMES_RELATIONSHIPS = {
        "start_node": "uid:START_ID",
        "end_node": "uid:END_ID",
        "relationship_type": ":TYPE",
    }
    EXPAND_PROPERTIES = True


class Neo4jLoadStyle(NodesRelationshipsStyle):
    COLUMN_NAMES_NODES = {
        "node_label": "label",
    }
    COLUMN_NAMES_RELATIONSHIPS = {
        "start_node": "startnode",
        "end_node": "endnode",
        "relationship_type": "type",
    }
    EXPAND_PROPERTIES = False
