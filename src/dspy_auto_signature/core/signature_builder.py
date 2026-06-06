"""Build live ``dspy.Signature`` classes from a ``SignatureSpec``."""

from __future__ import annotations

import logging
import types
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, cast

import dspy
from pydantic.fields import FieldInfo

if TYPE_CHECKING:
    from dspy_auto_signature.types.signature_spec import FieldSpec, SignatureSpec

logger = logging.getLogger(__name__)


class GeneratedSignature(Protocol):
    """A ``dspy.Signature`` subclass produced by ``SignatureBuilder``."""

    __name__: str
    __doc__: str | None
    instructions: str
    input_fields: dict[str, FieldInfo]
    output_fields: dict[str, FieldInfo]
    _spec: SignatureSpec

    @classmethod
    def to_source(cls) -> str: ...

    @classmethod
    def to_signature(cls) -> type[dspy.Signature]: ...

    def __call__(self, **kwargs: Any) -> dspy.Signature: ...


_BUILTIN_MODULES = frozenset({"builtins"})


def _is_builtin_type(t: type) -> bool:
    """Check if *t* is a Python builtin that does not require an import."""
    return getattr(t, "__module__", "") in _BUILTIN_MODULES


def _type_to_str(t: Any) -> str:
    """Convert a resolved Python type to its source-code string representation.

    Handles:
    - Simple types (str, int, etc.) → ``"str"``, ``"int"``
    - ``typing.Any`` → ``"Any"``
    - ``NoneType`` → ``"None"``
    - ``Ellipsis`` → ``"..."``
    - ``typing.Literal["a", "b"]`` → ``"Literal['a', 'b']"``
    - ``types.GenericAlias`` (list[str], dict[str, Any]) → ``"list[str]"``, ``"dict[str, Any]"``
    - ``types.UnionType`` (str | int | None) → ``"str | int | None"``
    """
    if t is Any:
        return "Any"

    # NoneType → "None"
    if t is type(None):
        return "None"

    # Ellipsis (used in tuple[str, ...])
    if t is Ellipsis:
        return "..."

    origin = getattr(t, "__origin__", None)

    # Literal types (e.g. Literal["a", "b"])
    if origin is Literal:
        args = getattr(t, "__args__", ())
        arg_strs = [repr(a) for a in args]
        return f"Literal[{', '.join(arg_strs)}]"

    # types.GenericAlias (list[str], dict[str, int], etc.)
    if isinstance(t, types.GenericAlias):
        origin_name = _type_to_str(t.__origin__)
        args = getattr(t, "__args__", ())
        if args:
            arg_strs = [_type_to_str(a) for a in args]
            return f"{origin_name}[{', '.join(arg_strs)}]"
        return origin_name

    # types.UnionType (str | int | None)
    if isinstance(t, types.UnionType):
        arg_strs = [_type_to_str(a) for a in t.__args__]
        return " | ".join(arg_strs)

    # typing._GenericAlias (typing.Union, typing.Optional, etc.)
    if origin is not None:
        name = getattr(t, "_name", None)
        if name:
            origin_name = name
        elif isinstance(origin, type):
            origin_name = origin.__name__
        else:
            origin_name = repr(origin)
        args = getattr(t, "__args__", ())
        if args:
            arg_strs = [_type_to_str(a) for a in args]
            return f"{origin_name}[{', '.join(arg_strs)}]"
        return origin_name

    # Simple named type (str, int, MyModel, etc.)
    if isinstance(t, type):
        return t.__name__

    # Fallback
    return repr(t)


def _collect_imports(fields: list[FieldSpec]) -> set[str]:
    """Collect import statements needed for the types used in *fields*.

    Walks through all field types (including nested generic parameters)
    and returns a set of import statement strings such as
    ``"from typing import Any"`` or ``"from pydantic import BaseModel"``.
    """
    imports: set[str] = set()
    _visited: set[int] = set()

    def _walk(t: object) -> None:
        tid = id(t)
        if tid in _visited:
            return
        _visited.add(tid)

        if t is Any:
            imports.add("from typing import Any")
            return

        origin = getattr(t, "__origin__", None)

        # Literal — args are values, not types; just import Literal
        if origin is Literal:
            imports.add("from typing import Literal")
            return

        # GenericAlias / UnionType — walk type parameters
        if isinstance(t, (types.GenericAlias, types.UnionType)):
            for arg in getattr(t, "__args__", ()):
                _walk(arg)
            return

        # typing._GenericAlias — walk type parameters
        if origin is not None:
            for arg in getattr(t, "__args__", ()):
                _walk(arg)
            return

        # Plain class — check if it needs an import
        if isinstance(t, type):
            if _is_builtin_type(t):
                return
            module = getattr(t, "__module__", None)
            if module and module not in ("builtins", "__main__"):
                imports.add(f"from {module} import {t.__name__}")

    for field in fields:
        _walk(field.resolved_type)

    return imports


def _escape_for_docstring(text: str) -> str:
    """Escape *text* so it is safe inside a triple-double-quoted docstring."""
    return text.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')


def _generate_source(spec: SignatureSpec) -> str:
    lines: list[str] = ["import dspy"]

    all_fields: list[FieldSpec] = [*spec.inputs, *spec.outputs]
    for imp in sorted(_collect_imports(all_fields)):
        lines.append(imp)

    lines.extend(["", ""])

    lines.append(f"class {spec.name}(dspy.Signature):")
    instructions = spec.instructions
    if "\n" in instructions or '"""' in instructions:
        escaped = _escape_for_docstring(instructions)
        lines.append('    """')
        for doc_line in escaped.split("\n"):
            lines.append(f"    {doc_line}")
        lines.append('    """')
    else:
        lines.append(f'    """{instructions}"""')

    lines.append("")

    for field in spec.inputs:
        type_str = _type_to_str(field.resolved_type)
        desc = field.description.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    {field.name}: {type_str} = dspy.InputField(desc="{desc}")')

    for field in spec.outputs:
        type_str = _type_to_str(field.resolved_type)
        desc = field.description.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'    {field.name}: {type_str} = dspy.OutputField(desc="{desc}")')

    return "\n".join(lines) + "\n"


class SignatureBuilder:
    """Construct a DSPy Signature class from a specification.

    This is the bridge between our intermediate ``SignatureSpec`` and the
    actual ``dspy.Signature`` subclass that the user can instantiate.
    """

    @classmethod
    def build(cls, spec: SignatureSpec) -> GeneratedSignature:
        """Create a new ``dspy.Signature`` subclass from *spec*.

        The returned value is a **class**, not an instance. It can be passed
        directly to ``dspy.Predict``, ``dspy.ChainOfThought``, etc.

        Args:
            spec: The intermediate signature specification.

        Returns:
            A fresh ``dspy.Signature`` subclass.

        Raises:
            ValueError: If any field is missing a description or type.

        """
        fields: dict[
            str, tuple[type | types.UnionType | types.GenericAlias, FieldInfo]
        ] = {}

        for field in spec.inputs:
            fields[field.name] = cls._make_field_tuple(field, dspy.InputField)

        for field in spec.outputs:
            fields[field.name] = cls._make_field_tuple(field, dspy.OutputField)

        sig_class = dspy.signatures.make_signature(
            cast(dict[str, tuple[type, FieldInfo]], fields),
            instructions=spec.instructions,
            signature_name=spec.name,
        )

        sig_any: Any = sig_class
        sig_any._spec = spec
        sig_any.to_source = classmethod(lambda cls: _generate_source(cls._spec))
        sig_any.to_signature = classmethod(lambda cls: cast(type[dspy.Signature], cls))

        logger.debug(
            "Built signature %s with %d input(s) and %d output(s)",
            spec.name,
            len(spec.inputs),
            len(spec.outputs),
        )

        return cast(GeneratedSignature, sig_class)

    @classmethod
    def to_source(cls, spec: SignatureSpec) -> str:
        """Generate Python source code for a ``dspy.Signature`` subclass.

        Args:
            spec: The intermediate signature specification.

        Returns:
            A string containing valid Python source code.

        """
        return _generate_source(spec)

    @classmethod
    def _make_field_tuple(
        cls,
        field_spec: FieldSpec,
        field_factory: Callable[..., FieldInfo],
    ) -> tuple[type | types.UnionType | types.GenericAlias, FieldInfo]:
        """Create a (type, FieldInfo) tuple compatible with ``make_signature``."""
        if not field_spec.description or not field_spec.description.strip():
            raise ValueError(
                f"Field '{field_spec.name}' is missing a description. "
                "All fields must have a non-empty description."
            )

        resolved_type = field_spec.resolved_type
        kwargs: dict[str, object] = {
            "desc": field_spec.description,
            "description": field_spec.description,
        }
        if field_spec.constraints:
            kwargs["json_schema_extra"] = {"constraints": field_spec.constraints}

        return (resolved_type, field_factory(**kwargs))
