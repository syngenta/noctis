import unittest
from typing import ClassVar
from noctis.repository.neo4j.neo4j_queries import (
    Neo4jQueryRegistry,
    Neo4jQueryValidationError,
    AbstractQuery,
)


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
