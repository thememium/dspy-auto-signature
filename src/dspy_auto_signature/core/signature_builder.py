"""Build live ``dspy.Signature`` classes from a ``SignatureSpec``."""

from __future__ import annotations

import logging
import types
from typing import TYPE_CHECKING, Callable

import dspy
from pydantic.fields import FieldInfo

if TYPE_CHECKING:
    from dspy_auto_signature.types.signature_spec import FieldSpec, SignatureSpec

logger = logging.getLogger(__name__)


class SignatureBuilder:
    """Construct a DSPy Signature class from a specification.

    This is the bridge between our intermediate ``SignatureSpec`` and the
    actual ``dspy.Signature`` subclass that the user can instantiate.
    """

    @classmethod
    def build(cls, spec: SignatureSpec) -> type[dspy.Signature]:
        """Create a new ``dspy.Signature`` subclass from *spec*.

        The returned value is a **class**, not an instance. It can be passed
        directly to ``dspy.Predict``, ``dspy.ChainOfThought``, etc.

        Args:
            spec: The intermediate signature specification.

        Returns:
            A fresh ``dspy.Signature`` subclass.

        """
        fields: dict[
            str, tuple[type | types.UnionType | types.GenericAlias, FieldInfo]
        ] = {}

        for field in spec.inputs:
            fields[field.name] = cls._make_field_tuple(field, dspy.InputField)

        for field in spec.outputs:
            fields[field.name] = cls._make_field_tuple(field, dspy.OutputField)

        from typing import cast

        sig_class = dspy.signatures.make_signature(
            cast(dict[str, tuple[type, FieldInfo]], fields),
            instructions=spec.instructions,
            signature_name=spec.name,
        )

        logger.debug(
            "Built signature %s with %d input(s) and %d output(s)",
            spec.name,
            len(spec.inputs),
            len(spec.outputs),
        )

        return sig_class  # type: ignore[return-value]

    @classmethod
    def _make_field_tuple(
        cls,
        field_spec: FieldSpec,
        field_factory: Callable[..., FieldInfo],
    ) -> tuple[type | types.UnionType | types.GenericAlias, FieldInfo]:
        """Create a (type, FieldInfo) tuple compatible with ``make_signature``."""
        resolved_type = field_spec.resolved_type
        kwargs: dict[str, object] = {"desc": field_spec.description}
        if field_spec.constraints:
            # Attach constraints as extra metadata; DSPy may expose them later.
            kwargs["json_schema_extra"] = {"constraints": field_spec.constraints}

        return (resolved_type, field_factory(**kwargs))
