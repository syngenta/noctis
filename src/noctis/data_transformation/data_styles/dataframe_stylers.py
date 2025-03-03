from noctis.data_architecture.datamodel import Node, Relationship
import pandas as pd


class NodesRelationshipsStyle:
    """
    A class to manage the style and export of nodes and relationships data to pandas DataFrames.

    """

    COLUMN_NAMES_NODES: dict[str, str] = {}
    COLUMN_NAMES_RELATIONSHIPS: dict[str, str] = {}
    EXPAND_PROPERTIES: bool = True

    @classmethod
    def export_nodes(
        cls, dict_label_nodes: dict[str, list[Node]]
    ) -> dict[str, pd.DataFrame]:
        """
        Export nodes to DataFrames.

        Args:
            dict_label_nodes (dict[str, list[Node]]): Dictionary mapping labels to lists of Node objects.

        Returns:
            dict[str, pd.DataFrame]: Dictionary mapping labels to DataFrames containing node data.
        """
        return {
            label: cls._process_nodes_dataframe(nodes_list)
            for label, nodes_list in dict_label_nodes.items()
        }

    @classmethod
    def export_relationships(
        cls, dict_type_relationships: dict[str, list[Relationship]]
    ) -> dict[str, pd.DataFrame]:
        """
        Export relationships to DataFrames.

        Args:
            dict_type_relationships (dict[str, list[Relationship]]): Dictionary mapping relationship types to lists of Relationship objects.

        Returns:
            dict[str, pd.DataFrame]: Dictionary mapping relationship types to DataFrames containing relationship data.
        """
        return {
            rtype: cls._process_relationships_dataframe(relationships_list)
            for rtype, relationships_list in dict_type_relationships.items()
        }

    @classmethod
    def _process_nodes_dataframe(cls, nodes_list: list[Node]) -> pd.DataFrame:
        """
        Process a list of nodes into a DataFrame.

        Args:
            nodes_list (list[Node]): List of Node objects.

        Returns:
            pd.DataFrame: DataFrame containing processed node data.
        """
        df = cls._build_nodes_dataframe(
            nodes_list, expand_properties=cls.EXPAND_PROPERTIES
        )
        df = cls._reorder_columns(df, ["uid"], ["node_label"])
        return cls._rename_columns(df, cls.COLUMN_NAMES_NODES)

    @classmethod
    def _process_relationships_dataframe(
        cls, relationships_list: list[Relationship]
    ) -> pd.DataFrame:
        """
        Process a list of relationships into a DataFrame.

        Args:
            relationships_list (list[Relationship]): List of Relationship objects.

        Returns:
            pd.DataFrame: DataFrame containing processed relationship data.
        """
        if relationships_list:
            df = cls._build_relationships_dataframe(
                relationships_list, expand_properties=cls.EXPAND_PROPERTIES
            )
        else:
            df = pd.DataFrame(columns=["start_node", "end_node", "relationship_type"])
        df = cls._reorder_columns(df, ["start_node"], ["end_node", "relationship_type"])
        return cls._rename_columns(df, cls.COLUMN_NAMES_RELATIONSHIPS)

    @staticmethod
    def _reorder_columns(
        df: pd.DataFrame, first_columns: list[str], last_columns: list[str]
    ) -> pd.DataFrame:
        """
        Reorder columns in a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to reorder.
            first_columns (list[str]): List of column names to place first.
            last_columns (list[str]): List of column names to place last.

        Returns:
            pd.DataFrame: DataFrame with reordered columns.
        """
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
        """
        Build a DataFrame from a list of nodes.

        Args:
            nodes (list[Node]): List of Node objects.
            expand_properties (bool): Flag to expand properties into separate columns.

        Returns:
            pd.DataFrame: DataFrame containing node data.
        """
        nodes_dict = [dict(item) for item in nodes]
        df = pd.DataFrame(nodes_dict)
        if expand_properties:
            df = NodesRelationshipsStyle._expand_properties_in_dataframe(df)
        return df

    @staticmethod
    def _build_relationships_dataframe(
        relationships: list[Relationship], expand_properties: bool = True
    ) -> pd.DataFrame:
        """
        Build a DataFrame from a list of relationships.

        Args:
            relationships (list[Relationship]): List of Relationship objects.
            expand_properties (bool): Flag to expand properties into separate columns.

        Returns:
            pd.DataFrame: DataFrame containing relationship data.
        """
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
            df = NodesRelationshipsStyle._expand_properties_in_dataframe(df)
        return df

    @staticmethod
    def _expand_properties_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand properties column into separate columns in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame with a 'properties' column to expand.

        Returns:
            pd.DataFrame: DataFrame with expanded properties.
        """
        properties_df = pd.json_normalize(df["properties"])
        return pd.concat([df.drop("properties", axis=1), properties_df], axis=1)

    @staticmethod
    def _rename_columns(df: pd.DataFrame, rename_dict: dict[str, str]) -> pd.DataFrame:
        """
        Rename columns in the DataFrame according to a given mapping.

        Args:
            df (pd.DataFrame): DataFrame to rename columns.
            rename_dict (dict[str, str]): Dictionary mapping original column names to new names.

        Returns:
            pd.DataFrame: DataFrame with renamed columns.
        """
        return df.rename(columns={col: rename_dict.get(col, col) for col in df.columns})


class PandasExportStyle(NodesRelationshipsStyle):
    """
    A subclass of NodesRelationshipsStyle with default settings for exporting nodes and relationships to DataFrames.
    """

    COLUMN_NAMES_NODES = {}
    COLUMN_NAMES_RELATIONSHIPS = {}
    EXPAND_PROPERTIES = True


class PandasExportStylePropertiesNotExpanded(NodesRelationshipsStyle):
    """
    A subclass of NodesRelationshipsStyle with settings to prevent expansion of properties when exporting to DataFrames.
    """

    COLUMN_NAMES_NODES = {}
    COLUMN_NAMES_RELATIONSHIPS = {}
    EXPAND_PROPERTIES = False
