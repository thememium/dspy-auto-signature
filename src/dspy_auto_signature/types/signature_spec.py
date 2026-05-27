"""Pydantic models for intermediate signature representation."""

from __future__ import annotations

import types
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FieldType(str, Enum):
    """Classification of a field's role in a signature."""

    INPUT = "input"
    OUTPUT = "output"


class FieldSpec(BaseModel):
    """Specification for a single field in a DSPy Signature.

    Attributes:
        name: The snake_case identifier for the field.
        description: Natural-language description of what the field represents.
        suggested_type: Natural-language type hint (e.g., "list of strings").
        field_type: Whether this is an input or output field.
        constraints: Optional validation constraints expressed as text.

    """

    name: str = Field(..., description="Snake-case field name")
    description: str = Field(..., description="What this field represents")
    suggested_type: str = Field(default="str", description="Natural-language type hint")
    field_type: FieldType = Field(..., description="input or output")
    constraints: str | None = Field(
        default=None, description="Optional validation constraints"
    )

    @property
    def resolved_type(self) -> type[Any] | types.UnionType | types.GenericAlias:
        """Resolve the natural-language type to an actual Python type.

        This is a convenience delegation to the TypeResolver.
        """
        from dspy_auto_signature.utils.type_resolver import TypeResolver

        return TypeResolver.resolve(self.suggested_type)


class ParsedPrompt(BaseModel):
    """Normalized representation of any prompt input.

    The parser layer converts heterogeneous inputs (raw strings, Vercel SDK
    arrays, Anthropic XML) into this common structure before the DSPy
    meta-program analyses it.

    Attributes:
        instruction_text: The core task description / system instructions.
        examples: Optional few-shot examples extracted from the prompt.
        raw_input: The original input for provenance.

    """

    instruction_text: str = Field(
        ..., description="Core task description / system instructions"
    )
    examples: list[dict[str, str]] = Field(
        default_factory=list, description="Few-shot examples"
    )
    raw_input: Any = Field(default=None, description="Original input for provenance")


class SignatureSpec(BaseModel):
    """Intermediate representation of a DSPy Signature.

    This is the *contract* produced by the generator layer and consumed by
    the builder layer. It is fully serialisable and human-readable.

    Attributes:
        name: A PascalCase class name for the generated Signature.
        instructions: The docstring / task description for the Signature.
        inputs: Ordered list of input field specifications.
        outputs: Ordered list of output field specifications.

    """

    name: str = Field(default="AutoSignature", description="PascalCase class name")
    instructions: str = Field(..., description="Signature docstring / task description")
    inputs: list[FieldSpec] = Field(default_factory=list, description="Input fields")
    outputs: list[FieldSpec] = Field(default_factory=list, description="Output fields")

    @property
    def all_fields(self) -> list[FieldSpec]:
        """Return input fields followed by output fields."""
        return [*self.inputs, *self.outputs]

    def to_signature_string(self) -> str:
        """Render a DSPy shorthand string like 'a, b -> c, d'.

        Useful for debugging or when the user wants a quick textual
        representation of the signature shape.
        """
        input_names = ", ".join(f.name for f in self.inputs)
        output_names = ", ".join(f.name for f in self.outputs)
        return f"{input_names} -> {output_names}"
