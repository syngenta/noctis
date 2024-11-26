from noctis.data_architecture.datamodel import Node, Relationship
import pandas as pd


class Neo4jStyle:
    COLUMN_NAMES_NODES: dict[str, str] = {}
    COLUMN_NAMES_RELATIONSHIPS: dict[str, str] = {}
    EXPAND_PROPERTIES: bool = True

    @classmethod
    def export_nodes(
        cls, dict_label_nodes: dict[str, list[Node]]
    ) -> dict[str, pd.DataFrame]:
        return {
            label: cls._process_nodes_dataframe(nodes_list)
            for label, nodes_list in dict_label_nodes.items()
        }

    @classmethod
    def export_relationships(
        cls, dict_type_relationships: dict[str, list[Relationship]]
    ) -> dict[str, pd.DataFrame]:
        return {
            rtype: cls._process_relationships_dataframe(relationships_list)
            for rtype, relationships_list in dict_type_relationships.items()
        }

    @classmethod
    def _process_nodes_dataframe(cls, nodes_list: list[Node]) -> pd.DataFrame:
        df = cls._build_nodes_dataframe(
            nodes_list, expand_properties=cls.EXPAND_PROPERTIES
        )
        df = cls._reorder_columns(df, ["uid"], ["node_label"])
        return cls._rename_columns(df, cls.COLUMN_NAMES_NODES)

    @classmethod
    def _process_relationships_dataframe(
        cls, relationships_list: list[Relationship]
    ) -> pd.DataFrame:
        df = cls._build_relationships_dataframe(
            relationships_list, expand_properties=cls.EXPAND_PROPERTIES
        )
        df = cls._reorder_columns(df, ["start_node"], ["end_node", "relationship_type"])
        return cls._rename_columns(df, cls.COLUMN_NAMES_RELATIONSHIPS)

    @staticmethod
    def _reorder_columns(
        df: pd.DataFrame, first_columns: list[str], last_columns: list[str]
    ) -> pd.DataFrame:
        all_columns = df.columns.tolist()
        middle_columns = [
            col
            for col in all_columns
            if col not in first_columns and col not in last_columns
        ]
        return df[first_columns + middle_columns + last_columns]

    @staticmethod
    def _build_nodes_dataframe(
        nodes: list[Node], expand_properties: bool = True
    ) -> pd.DataFrame:
        nodes_dict = [dict(item) for item in nodes]
        df = pd.DataFrame(nodes_dict)
        if expand_properties:
            df = Neo4jStyle._expand_properties_in_dataframe(df)
        return df

    @staticmethod
    def _build_relationships_dataframe(
        relationships: list[Relationship], expand_properties: bool = True
    ) -> pd.DataFrame:
        relationships_dict = [
            {
                "relationship_type": item.relationship_type,
                "start_node": item.start_node.uid,
                "end_node": item.end_node.uid,
                "properties": item.properties,
            }
            for item in relationships
        ]
        df = pd.DataFrame(relationships_dict)
        if expand_properties:
            df = Neo4jStyle._expand_properties_in_dataframe(df)
        return df

    @staticmethod
    def _expand_properties_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        properties_df = pd.json_normalize(df["properties"])
        return pd.concat([df.drop("properties", axis=1), properties_df], axis=1)

    @staticmethod
    def _rename_columns(df: pd.DataFrame, rename_dict: dict[str, str]) -> pd.DataFrame:
        return df.rename(columns={col: rename_dict.get(col, col) for col in df.columns})


class Neo4jImportStyle(Neo4jStyle):
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


class Neo4jLoadStyle(Neo4jStyle):
    COLUMN_NAMES_NODES = {
        "node_label": "label",
    }
    COLUMN_NAMES_RELATIONSHIPS = {
        "start_node": "startnode",
        "end_node": "endnode",
        "relationship_type": "type",
    }
    EXPAND_PROPERTIES = False
