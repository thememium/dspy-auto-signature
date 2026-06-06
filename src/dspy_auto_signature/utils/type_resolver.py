"""Map natural-language type descriptions to concrete Python types."""

from __future__ import annotations

import ast
import json
import re
import types
from typing import Any, Literal


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

    # Patterns that signal a Literal type.
    # Matches: "literal X, Y, Z", "one of X, Y, Z", "enum X, Y, Z"
    # Also handles optional colon: "one of: X, Y, Z"
    _LITERAL_PATTERN = re.compile(
        r"^(literal|one\s+of|enum)\s*:?\s*(.+)$",
        re.IGNORECASE,
    )

    @staticmethod
    def _parse_literal_values(raw: str) -> tuple[str, ...]:
        """Parse literal values from comma-separated or list syntax.

        Supports forms such as ``low, medium, high``,
        ``["low", "medium", "high"]``, and ``['low', 'medium', 'high']``.
        """
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            parsed: Any = None
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    pass
            if isinstance(parsed, (list, tuple)):
                return tuple(str(value) for value in parsed if str(value))

        values: list[str] = []
        for part in raw.split(","):
            val = part.strip().strip("[]").strip().strip("\"'")
            if val:
                values.append(val)
        return tuple(values)

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

        # Handle Literal types: "literal X, Y, Z" / "one of X, Y, Z" / "enum X, Y, Z"
        literal_match = cls._LITERAL_PATTERN.match(cleaned)
        if literal_match:
            values = cls._parse_literal_values(literal_match.group(2))
            if values:
                return Literal[values]  # type: ignore

        # Handle "optional X" → X | None
        if cleaned.startswith("optional "):
            inner = cls.resolve(cleaned[9:])
            return inner | None  # type: ignore[return-value]

        # Handle "nullable X" → X | None
        if cleaned.startswith("nullable "):
            inner = cls.resolve(cleaned[9:])
            return inner | None  # type: ignore[return-value]

        # Handle container generics: "list of X", "array of X"
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

        # Handle dict generics: "dict of X to Y", "mapping of X to Y"
        dict_match = re.match(
            r"^(dict|dictionary|map|mapping)\s+of\s+(.+?)\s+to\s+(.+)$",
            cleaned,
        )
        if dict_match:
            key_desc = dict_match.group(2).strip()
            val_desc = dict_match.group(3).strip()
            key_type = cls.resolve(key_desc)
            val_type = cls.resolve(val_desc)
            return dict[key_type, val_type]  # type: ignore

        # Handle "X or Y" unions (supports 3+ types: "X, Y, or Z")
        if " or " in cleaned and "list of" not in cleaned:
            parts = [
                p.strip() for p in re.split(r",?\s*or\s+|,\s+", cleaned) if p.strip()
            ]
            if len(parts) >= 2:
                resolved = [cls.resolve(p) for p in parts]
                result = resolved[0]
                for t in resolved[1:]:
                    result = result | t  # type: ignore[assignment]
                return result  # type: ignore[return-value]

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

    @classmethod
    def register_pydantic_model(cls, model_cls: type[Any]) -> None:
        """Register a Pydantic BaseModel class for use in type resolution.

        Registers the model under its class name lowercased
        (e.g. ``MemoryOperation`` → ``memoryoperation``).

        Args:
            model_cls: A Pydantic ``BaseModel`` subclass.

        """
        name = model_cls.__name__
        cls._ALIASES[name.lower()] = model_cls
