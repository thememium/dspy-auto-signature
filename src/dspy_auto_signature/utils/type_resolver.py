"""Map natural-language type descriptions to concrete Python types."""

from __future__ import annotations

import re
import types
from typing import Any


class TypeResolver:
    """Resolve natural-language type strings to real Python types.

    Supports common primitives, containers, unions, and optional fallbacks.
    """

    # Primitive and container mappings — feel free to extend.
    _ALIASES: dict[str, type[Any]] = {
        # Strings
        "string": str,
        "str": str,
        "text": str,
        # Numbers
        "integer": int,
        "int": int,
        "number": float,
        "float": float,
        "decimal": float,
        # Boolean
        "boolean": bool,
        "bool": bool,
        "yes/no": bool,
        "yes or no": bool,
        # Collections
        "list": list,
        "array": list,
        "sequence": list,
        "tuple": tuple,
        "dict": dict,
        "dictionary": dict,
        "map": dict,
        "object": dict,
        # Special
        "any": Any,
    }

    @classmethod
    def resolve(
        cls, type_desc: str
    ) -> type[Any] | types.UnionType | types.GenericAlias:
        """Convert a natural-language type description to a Python type.

        Examples:
            >>> TypeResolver.resolve("string")
            <class 'str'>
            >>> TypeResolver.resolve("list of strings")
            list[str]
            >>> TypeResolver.resolve("a number")
            <class 'float'>
            >>> TypeResolver.resolve("optional integer")
            int | None

        Args:
            type_desc: A natural-language description of a type.

        Returns:
            The best-matching Python type. Falls back to ``str`` when
            no match is found.

        """
        if not type_desc:
            return str

        cleaned = type_desc.strip().lower()

        # Handle "optional X" → X | None
        if cleaned.startswith("optional "):
            inner = cls.resolve(cleaned[9:])
            return inner | None  # type: ignore[return-value]

        # Handle "nullable X" → X | None
        if cleaned.startswith("nullable "):
            inner = cls.resolve(cleaned[9:])
            return inner | None  # type: ignore[return-value]

        # Handle container generics: "list of X", "array of X", "dict of X to Y"
        container_match = re.match(
            r"^(list|array|sequence|tuple)\s+of\s+(.+)$",
            cleaned,
        )
        if container_match:
            container_name = container_match.group(1)
            inner_desc = container_match.group(2).strip()
            inner_type = cls.resolve(inner_desc)
            container_cls = cls._ALIASES.get(container_name, list)
            return container_cls[inner_type]  # type: ignore[return-value]

        # Handle "X or Y" unions (simple two-type unions)
        if " or " in cleaned and "list of" not in cleaned:
            parts = [p.strip() for p in cleaned.split(" or ")]
            if len(parts) == 2:
                left, right = cls.resolve(parts[0]), cls.resolve(parts[1])
                return left | right  # type: ignore[return-value]

        # Exact match on aliases
        if cleaned in cls._ALIASES:
            return cls._ALIASES[cleaned]

        # Strip leading articles for friendlier matching
        cleaned = re.sub(r"^(a|an|the)\s+", "", cleaned)
        if cleaned in cls._ALIASES:
            return cls._ALIASES[cleaned]

        # Handle common plurals by stripping trailing 's'
        singular = cleaned.rstrip("s")
        if singular in cls._ALIASES:
            return cls._ALIASES[singular]

        # Fallback: assume it's a string field
        return str

    @classmethod
    def register(cls, name: str, py_type: type[Any]) -> None:
        """Register a custom type alias.

        Useful for domain-specific types (e.g. ``email`` → ``pydantic.EmailStr``).

        Args:
            name: The natural-language name for the type.
            py_type: The Python type to map it to.

        """
        cls._ALIASES[name.lower()] = py_type
