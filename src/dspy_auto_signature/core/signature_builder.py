"""Build live ``dspy.Signature`` classes from a ``SignatureSpec``."""

from __future__ import annotations

import logging
import types
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast

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


def _type_to_str(t: type[Any] | types.UnionType | types.GenericAlias) -> str:
    """Convert a resolved Python type to its source-code string representation."""
    if t is Any:
        return "Any"
    if isinstance(t, type):
        return t.__name__
    return repr(t)


def _generate_source(spec: SignatureSpec) -> str:
    lines: list[str] = [
        "import dspy",
        "",
        "",
        f"class {spec.name}(dspy.Signature):",
        f'    """{spec.instructions}"""',
        "",
    ]

    for field in spec.inputs:
        type_str = _type_to_str(field.resolved_type)
        lines.append(
            f'    {field.name}: {type_str} = dspy.InputField(desc="{field.description}")'
        )

    for field in spec.outputs:
        type_str = _type_to_str(field.resolved_type)
        lines.append(
            f'    {field.name}: {type_str} = dspy.OutputField(desc="{field.description}")'
        )

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
