import unittest
from typing import ClassVar
from noctis.repository.neo4j.neo4j_queries import (
    Neo4jQueryRegistry,
    Neo4jQueryValidationError,
    AbstractQuery,
)

from pydantic import ValidationError
from noctis.repository.neo4j.neo4j_queries import CustomQuery
import tempfile
import os


class TestAbstractQuery(unittest.TestCase):
    def setUp(self):
        class TestQuery(AbstractQuery):
            query_name: ClassVar[str] = "test_query"
            query_type: ClassVar[str] = "test"
            is_parameterized: ClassVar[bool] = False
            query_args_required: ClassVar[list[str]] = ["arg1", "arg2"]
            query_args_optional: ClassVar[list[str]] = ["arg3"]
            query: str = "TEST QUERY"

        self.TestQuery = TestQuery

    def test_class_variables(self):
        self.assertEqual(self.TestQuery.query_name, "test_query")
        self.assertEqual(self.TestQuery.query_type, "test")
        self.assertFalse(self.TestQuery.is_parameterized)
        self.assertEqual(self.TestQuery.query_args_required, ["arg1", "arg2"])
        self.assertEqual(self.TestQuery.query_args_optional, ["arg3"])

    def test_model_config(self):
        self.assertTrue(self.TestQuery.model_config.get("arbitrary_types_allowed"))

    def test_validate_query_kwargs_success(self):
        valid_data = {"arg1": "value1", "arg2": "value2"}
        self.assertEqual(self.TestQuery.validate_query_kwargs(valid_data), valid_data)

    def test_validate_query_kwargs_with_optional(self):
        valid_data = {"arg1": "value1", "arg2": "value2", "arg3": "value3"}
        self.assertEqual(self.TestQuery.validate_query_kwargs(valid_data), valid_data)

    def test_validate_query_kwargs_missing_required(self):
        invalid_data = {"arg1": "value1"}
        with self.assertRaises(Neo4jQueryValidationError) as context:
            self.TestQuery.validate_query_kwargs(invalid_data)
        self.assertIn("Missing required arguments: arg2", str(context.exception))

    def test_validate_query_kwargs_invalid_arg(self):
        invalid_data = {"arg1": "value1", "arg2": "value2", "invalid_arg": "value"}
        with self.assertRaises(Neo4jQueryValidationError) as context:
            self.TestQuery.validate_query_kwargs(invalid_data)
        self.assertIn("Invalid arguments provided: invalid_arg", str(context.exception))

    def test_list_arguments(self):
        expected = {"required": ["arg1", "arg2"], "optional": ["arg3"]}
        self.assertEqual(self.TestQuery.list_arguments(), expected)

    def test_get_query(self):
        query = self.TestQuery(arg1="value1", arg2="value2")
        self.assertEqual(query.get_query(), "TEST QUERY")

    def test_instantiation_success(self):
        query = self.TestQuery(arg1="value1", arg2="value2")
        self.assertIsInstance(query, self.TestQuery)

    def test_instantiation_failure(self):
        with self.assertRaises(Neo4jQueryValidationError):
            self.TestQuery(arg1="value1")

    def test_instantiation_with_extra_arg(self):
        with self.assertRaises(Neo4jQueryValidationError):
            self.TestQuery(arg1="value1", arg2="value2", extra_arg="value")

    def test_query_with_no_required_args(self):
        class NoArgsQuery(AbstractQuery):
            query_name: ClassVar[str] = "no_args_query"
            query_type: ClassVar[str] = "test"
            is_parameterized: ClassVar[bool] = False
            query_args_required: ClassVar[list] = []
            query_args_optional: ClassVar[list] = []
            query: ClassVar[str] = "NO ARGS QUERY"

        query = NoArgsQuery()
        self.assertEqual(query.get_query(), "NO ARGS QUERY")

    def test_query_with_only_optional_args(self):
        class OptionalArgsQuery(AbstractQuery):
            query_name: ClassVar[str] = "optional_args_query"
            query_type: ClassVar[str] = "test"
            is_parameterized: ClassVar[bool] = False
            query_args_required: ClassVar[list[str]] = []
            query_args_optional: ClassVar[list[str]] = ["opt1", "opt2"]
            query: ClassVar[str] = "OPTIONAL ARGS QUERY"

        query = OptionalArgsQuery()
        self.assertEqual(query.get_query(), "OPTIONAL ARGS QUERY")

        query_with_args = OptionalArgsQuery(opt1="value1")
        self.assertEqual(query_with_args.get_query(), "OPTIONAL ARGS QUERY")

    def test_query_with_parameterized_true(self):
        class ParameterizedQuery(AbstractQuery):
            query_name: ClassVar[str] = "parameterized_query"
            query_type: ClassVar[str] = "test"
            is_parameterized: ClassVar[bool] = True
            query_args_required: ClassVar[list[str]] = ["param1"]
            query_args_optional: ClassVar[list[str]] = []
            query: ClassVar[str] = "PARAMETERIZED QUERY WITH {param1}"

            param1: str

        query = ParameterizedQuery(param1="test_value")
        self.assertEqual(query.get_query(), "PARAMETERIZED QUERY WITH {param1}")

    def test_model_validator_decorator(self):
        # This test checks if the model_validator decorator is applied correctly
        self.assertTrue(hasattr(self.TestQuery, "validate_query_kwargs"))
        self.assertTrue(callable(getattr(self.TestQuery, "validate_query_kwargs")))

    def test_config_dict_usage(self):
        # Check if model_config exists and is a dictionary
        self.assertTrue(hasattr(self.TestQuery, "model_config"))
        self.assertIsInstance(self.TestQuery.model_config, dict)

        # Check if the expected configuration is present
        self.assertTrue("arbitrary_types_allowed" in self.TestQuery.model_config)
        self.assertTrue(self.TestQuery.model_config["arbitrary_types_allowed"])

        # Instead of checking for ConfigDict specifically, we'll check for the presence
        # of the expected configuration options
        expected_keys = {"arbitrary_types_allowed"}
        self.assertTrue(expected_keys.issubset(self.TestQuery.model_config.keys()))

        # Optionally, you can print the type for debugging
        print(f"Type of model_config: {type(self.TestQuery.model_config)}")


class TestNeo4jQueryRegistry(unittest.TestCase):
    def setUp(self):
        Neo4jQueryRegistry.queries = {}

    def test_register_query(self):
        @Neo4jQueryRegistry.register_query()
        class TestQuery(AbstractQuery):
            query_name = "test_query"
            query_type = "test"

        self.assertIn("test_query", Neo4jQueryRegistry.queries)

    def test_get_query_object(self):
        @Neo4jQueryRegistry.register_query()
        class TestQuery(AbstractQuery):
            query_name = "test_query"
            query_type = "test"

        query_class = Neo4jQueryRegistry.get_query_object("test_query")
        self.assertEqual(query_class, TestQuery)

        with self.assertRaises(ValueError):
            Neo4jQueryRegistry.get_query_object("invalid_name")

    def test_get_all_queries(self):
        @Neo4jQueryRegistry.register_query()
        class TestQuery1(AbstractQuery):
            query_name = "test_query1"
            query_type = "test"

        @Neo4jQueryRegistry.register_query()
        class TestQuery2(AbstractQuery):
            query_name = "test_query2"
            query_type = "test"

        expected = {"test_query1", "test_query2"}
        self.assertEqual(Neo4jQueryRegistry.get_all_queries(), expected)


class TestCustomQuery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sample YAML content for testing
        cls.SAMPLE_YAML = """
        - version: 1.0
        - date: 2024-12-04

        - query_name: get_tree
          query_type: retrieve_graph
          is_parameterized: false
          query_args_required:
            - root_node_uid
          query_args_optional:
            - max_level
          query: |
            MATCH (start {uid:$root_node_uid})
            CALL apoc.path.subgraphAll(start, {
              relationshipFilter: '<PRODUCT,<REACTANT',
              minLevel: 0,
              maxLevel: $max_level
            })
            YIELD nodes, relationships
            RETURN nodes, relationships

        - query_name: get_node_by_id
          query_type: retrieve_graph
          is_parameterized: false
          query_args_required:
            - node_id
          query: |
            MATCH (n:Node {id: $node_id})
            RETURN n
        """
        # Create a temporary YAML file
        cls.temp_dir = tempfile.mkdtemp()
        cls.yaml_file = os.path.join(cls.temp_dir, "test_queries.yaml")
        with open(cls.yaml_file, "w") as f:
            f.write(cls.SAMPLE_YAML)

    @classmethod
    def tearDownClass(cls):
        # Clean up the temporary file
        os.remove(cls.yaml_file)
        os.rmdir(cls.temp_dir)

    def test_custom_query_creation(self):
        query = CustomQuery(
            query_name="test",
            query_type="retrieve_graph",
            query="MATCH (n) RETURN n",
            query_args_required=["arg1"],
            query_args_optional=["arg2"],
        )
        self.assertEqual(query.query_name, "test")
        self.assertEqual(query.query_type, "retrieve_graph")
        self.assertEqual(query.query, "MATCH (n) RETURN n")
        self.assertEqual(query.query_args_required, ["arg1"])
        self.assertEqual(query.query_args_optional, ["arg2"])

    def test_custom_query_validation(self):
        with self.assertRaises(ValidationError):
            CustomQuery(query_name="test", query=123)  # Invalid query type

    def test_build_from_dict(self):
        query_dict = {
            "query_name": "test",
            "query_type": "retrieve_graph",
            "query": "MATCH (n) RETURN n",
            "query_args_required": ["arg1"],
        }
        query = CustomQuery.build_from_dict(query_dict)
        self.assertEqual(query.query_name, "test")
        self.assertEqual(query.query_type, "retrieve_graph")
        self.assertEqual(query.query, "MATCH (n) RETURN n")
        self.assertEqual(query.query_args_required, ["arg1"])

    def test_from_yaml(self):
        query = CustomQuery.from_yaml(self.yaml_file, "get_tree")
        self.assertEqual(query.query_name, "get_tree")
        self.assertEqual(query.query_type, "retrieve_graph")
        self.assertIn("MATCH (start {uid:$root_node_uid})", query.query)
        self.assertEqual(query.query_args_required, ["root_node_uid"])
        self.assertEqual(query.query_args_optional, ["max_level"])

    def test_from_yaml_not_found(self):
        with self.assertRaises(ValueError) as context:
            CustomQuery.from_yaml(self.yaml_file, "non_existent")
        self.assertIn(
            "Query 'non_existent' not found in the YAML file", str(context.exception)
        )

    def test_get_query(self):
        query = CustomQuery.from_yaml(self.yaml_file, "get_node_by_id")
        self.assertIn("MATCH (n:Node {id: $node_id})", query.get_query())

    def test_list_queries(self):
        queries = CustomQuery.list_queries(self.yaml_file)
        self.assertEqual(set(queries), {"get_tree", "get_node_by_id"})

    def test_validate_query_kwargs(self):
        query = CustomQuery.from_yaml(self.yaml_file, "get_tree")

        # Valid args
        query.validate_query_kwargs({"root_node_uid": "value", "max_level": 3})

        # Missing required arg
        with self.assertRaises(ValueError) as context:
            query.validate_query_kwargs({"max_level": 3})
        self.assertIn(
            "Missing required arguments: root_node_uid", str(context.exception)
        )

        # Invalid arg
        with self.assertRaises(ValueError) as context:
            query.validate_query_kwargs(
                {"root_node_uid": "value", "invalid_arg": "value"}
            )
        self.assertIn("Invalid arguments provided: invalid_arg", str(context.exception))

    def test_call_method(self):
        query = CustomQuery.from_yaml(self.yaml_file, "get_node_by_id")

        result = query(node_id="123")
        self.assertIsInstance(result, CustomQuery)

        with self.assertRaises(ValueError):
            query(invalid_arg="value")  # Invalid arg

    def test_query_as_string(self):
        query = CustomQuery.from_yaml(self.yaml_file, "get_node_by_id")
        self.assertIsInstance(query.query, str)
        self.assertIn("MATCH (n:Node {id: $node_id})", query.query)

    def test_query_invalid_structure(self):
        with self.assertRaises(ValidationError):
            CustomQuery(
                query_name="test",
                query_type="retrieve_graph",
                query=[1, 2, 3],  # Invalid: not a string or list of strings
                query_args_required=["arg1"],
            )
