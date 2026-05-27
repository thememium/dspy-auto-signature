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
        # Python generics like list[str] are not simple instances,
        # so we inspect the __class_getitem__ / origin mechanism.
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
