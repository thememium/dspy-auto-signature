"""Tests for the TypeResolver utility."""

from typing import Any

import pytest

from dspy_auto_signature.utils.type_resolver import TypeResolver


class TestTypeResolver:
    """Comprehensive tests for natural-language → Python type resolution."""

    @pytest.mark.parametrize(
        ("description", "expected"),
        [
            ("string", str),
            ("str", str),
            ("text", str),
            ("integer", int),
            ("int", int),
            ("number", float),
            ("float", float),
            ("boolean", bool),
            ("bool", bool),
            ("yes/no", bool),
            ("list", list),
            ("array", list),
            ("dict", dict),
            ("dictionary", dict),
            ("any", Any),
        ],
    )
    def test_primitive_aliases(self, description: str, expected: type) -> None:
        assert TypeResolver.resolve(description) is expected

    @pytest.mark.parametrize(
        ("description", "expected_container", "expected_inner"),
        [
            ("list of strings", list, str),
            ("array of integers", list, int),
            ("sequence of booleans", list, bool),
        ],
    )
    def test_container_types(
        self,
        description: str,
        expected_container: type,
        expected_inner: type,
    ) -> None:
        resolved = TypeResolver.resolve(description)
        origin = getattr(resolved, "__origin__", None)
        args = getattr(resolved, "__args__", ())

        assert origin is expected_container
        assert args and args[0] is expected_inner

    def test_optional_types(self) -> None:
        resolved = TypeResolver.resolve("optional string")
        args = getattr(resolved, "__args__", ())
        assert str in args
        assert type(None) in args

    def test_strip_articles(self) -> None:
        assert TypeResolver.resolve("a string") is str
        assert TypeResolver.resolve("an integer") is int
        assert TypeResolver.resolve("the number") is float

    def test_unknown_type_fallback(self) -> None:
        """Unrecognised descriptions should fall back to ``str``."""
        assert TypeResolver.resolve("some_custom_domain_type") is str

    def test_empty_input(self) -> None:
        assert TypeResolver.resolve("") is str

    def test_custom_registration(self) -> None:
        class MyCustomType:
            pass

        TypeResolver.register("custom_type", MyCustomType)
        assert TypeResolver.resolve("custom_type") is MyCustomType

    # --- Literal types ---

    @pytest.mark.parametrize(
        "description",
        [
            "literal high, medium, low",
            "one of high, medium, low",
            "one of: high, medium, low",
            "enum high, medium, low",
        ],
    )
    def test_literal_basic(self, description: str) -> None:
        resolved = TypeResolver.resolve(description)
        args = getattr(resolved, "__args__", ())
        assert args == ("high", "medium", "low")

    def test_literal_two_values(self) -> None:
        resolved = TypeResolver.resolve("one of: yes, no")
        args = getattr(resolved, "__args__", ())
        assert args == ("yes", "no")

    def test_literal_case_insensitive_prefix(self) -> None:
        resolved = TypeResolver.resolve("Literal a, b, c")
        args = getattr(resolved, "__args__", ())
        assert args == ("a", "b", "c")

    def test_literal_strips_quotes(self) -> None:
        resolved = TypeResolver.resolve('literal "a", "b"')
        args = getattr(resolved, "__args__", ())
        assert args == ("a", "b")

    @pytest.mark.parametrize(
        "description",
        [
            'literal ["low", "medium", "high"]',
            "literal ['low', 'medium', 'high']",
            'one of: ["low", "medium", "high"]',
        ],
    )
    def test_literal_parses_bracketed_lists(self, description: str) -> None:
        resolved = TypeResolver.resolve(description)
        args = getattr(resolved, "__args__", ())
        assert args == ("low", "medium", "high")

    # --- Dict generics ---

    def test_dict_of_string_to_integer(self) -> None:
        resolved = TypeResolver.resolve("dict of string to integer")
        origin = getattr(resolved, "__origin__", None)
        args: tuple = getattr(resolved, "__args__", ())
        assert origin is dict
        assert args[0] is str
        assert args[1] is int

    def test_dict_of_string_to_list_of_strings(self) -> None:
        resolved = TypeResolver.resolve("dict of string to list of strings")
        origin = getattr(resolved, "__origin__", None)
        args: tuple = getattr(resolved, "__args__", ())
        assert origin is dict
        assert args[0] is str
        val_origin = getattr(args[1], "__origin__", None)
        val_args: tuple = getattr(args[1], "__args__", ())
        assert val_origin is list
        assert val_args[0] is str

    def test_mapping_of_string_to_float(self) -> None:
        resolved = TypeResolver.resolve("mapping of string to float")
        origin = getattr(resolved, "__origin__", None)
        args: tuple = getattr(resolved, "__args__", ())
        assert origin is dict
        assert args[0] is str
        assert args[1] is float

    def test_dictionary_of_integer_to_boolean(self) -> None:
        resolved = TypeResolver.resolve("dictionary of integer to boolean")
        origin = getattr(resolved, "__origin__", None)
        args: tuple = getattr(resolved, "__args__", ())
        assert origin is dict
        assert args[0] is int
        assert args[1] is bool

    # --- Union types (3+) ---

    def test_union_three_types(self) -> None:
        resolved = TypeResolver.resolve("string, integer, or float")
        args = getattr(resolved, "__args__", ())
        assert str in args
        assert int in args
        assert float in args

    def test_union_two_types(self) -> None:
        resolved = TypeResolver.resolve("string or integer")
        args = getattr(resolved, "__args__", ())
        assert str in args
        assert int in args

    # --- Optional containers ---

    def test_optional_list_of_strings(self) -> None:
        resolved = TypeResolver.resolve("optional list of strings")
        args = getattr(resolved, "__args__", ())
        assert type(None) in args
        list_type = [a for a in args if a is not type(None)][0]
        origin = getattr(list_type, "__origin__", None)
        inner: tuple = getattr(list_type, "__args__", ())
        assert origin is list
        assert inner[0] is str

    # --- Pydantic model registration ---

    def test_register_pydantic_model(self) -> None:
        class FakeModel:
            __name__ = "FakeModel"

        TypeResolver.register_pydantic_model(FakeModel)
        assert TypeResolver.resolve("FakeModel") is FakeModel
        assert TypeResolver.resolve("fakemodel") is FakeModel

    def test_register_pydantic_model_with_real_class(self) -> None:
        class MemoryOperation:
            __name__ = "MemoryOperation"

        TypeResolver.register_pydantic_model(MemoryOperation)
        assert TypeResolver.resolve("MemoryOperation") is MemoryOperation
        assert TypeResolver.resolve("memoryoperation") is MemoryOperation
