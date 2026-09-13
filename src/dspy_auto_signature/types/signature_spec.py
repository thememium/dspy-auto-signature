"""Pydantic models for intermediate signature representation."""

from __future__ import annotations

import keyword
import re
import types
from enum import Enum
from typing import Any, Literal, cast

from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    create_model,
    field_validator,
    model_validator,
)
from pydantic.fields import FieldInfo


class FieldType(str, Enum):
    """Classification of a field's role in a signature."""

    INPUT = "input"
    OUTPUT = "output"


class SchemaFieldType(str, Enum):
    """Concrete value types available inside generated Pydantic models."""

    STRING = "str"
    INTEGER = "int"
    FLOAT = "float"
    BOOLEAN = "bool"
    LIST_STRING = "list[str]"
    LIST_INTEGER = "list[int]"
    LIST_FLOAT = "list[float]"
    LIST_BOOLEAN = "list[bool]"
    DICT_STR_STRING = "dict[str, str]"
    DICT_STR_INTEGER = "dict[str, int]"
    DICT_STR_FLOAT = "dict[str, float]"
    DICT_STR_ANY = "dict[str, Any]"
    LIST_DICT = "list[dict]"
    DICT_ANY = "dict"
    LITERAL = "Literal"
    PYDANTIC_MODEL = "pydantic"


_SCHEMA_ANNOTATIONS: dict[SchemaFieldType, Any] = {
    SchemaFieldType.STRING: str,
    SchemaFieldType.INTEGER: int,
    SchemaFieldType.FLOAT: float,
    SchemaFieldType.BOOLEAN: bool,
    SchemaFieldType.LIST_STRING: list[str],
    SchemaFieldType.LIST_INTEGER: list[int],
    SchemaFieldType.LIST_FLOAT: list[float],
    SchemaFieldType.LIST_BOOLEAN: list[bool],
    SchemaFieldType.DICT_STR_STRING: dict[str, str],
    SchemaFieldType.DICT_STR_INTEGER: dict[str, int],
    SchemaFieldType.DICT_STR_FLOAT: dict[str, float],
    SchemaFieldType.DICT_STR_ANY: dict[str, Any],
    SchemaFieldType.LIST_DICT: list[dict],
    SchemaFieldType.DICT_ANY: dict,
}

_SCHEMA_ALIASES: dict[str, SchemaFieldType] = {
    alias.lower(): field_type
    for field_type in SchemaFieldType
    for alias in (field_type.value, field_type.name)
}
_SCHEMA_ALIASES.update(
    {
        "string": SchemaFieldType.STRING,
        "text": SchemaFieldType.STRING,
        "integer": SchemaFieldType.INTEGER,
        "number": SchemaFieldType.FLOAT,
        "double": SchemaFieldType.FLOAT,
        "boolean": SchemaFieldType.BOOLEAN,
        "list of strings": SchemaFieldType.LIST_STRING,
        "string list": SchemaFieldType.LIST_STRING,
        "list[string]": SchemaFieldType.LIST_STRING,
        "list of integers": SchemaFieldType.LIST_INTEGER,
        "list of floats": SchemaFieldType.LIST_FLOAT,
        "list of booleans": SchemaFieldType.LIST_BOOLEAN,
        "list[bool]": SchemaFieldType.LIST_BOOLEAN,
        "list of dicts": SchemaFieldType.LIST_DICT,
        "dict list": SchemaFieldType.LIST_DICT,
        "array of objects": SchemaFieldType.LIST_DICT,
        "dict[str,string]": SchemaFieldType.DICT_STR_STRING,
        "dict of strings": SchemaFieldType.DICT_STR_STRING,
        "dict[str,int]": SchemaFieldType.DICT_STR_INTEGER,
        "dict[str,float]": SchemaFieldType.DICT_STR_FLOAT,
        "dictionary": SchemaFieldType.DICT_ANY,
        "object": SchemaFieldType.PYDANTIC_MODEL,
        "model": SchemaFieldType.PYDANTIC_MODEL,
        "pydantic model": SchemaFieldType.PYDANTIC_MODEL,
        "enum": SchemaFieldType.LITERAL,
        "list of ints": SchemaFieldType.LIST_INTEGER,
        "list[integer]": SchemaFieldType.LIST_INTEGER,
        "literal": SchemaFieldType.LITERAL,
    }
)

_MAX_SCHEMA_DEPTH = 8


def _sanitize_identifier(name: str) -> str:
    """Return a valid, non-reserved Python identifier."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(name)).strip("_")
    if not normalized:
        normalized = "field"
    if normalized[0].isdigit():
        normalized = f"field_{normalized}"
    if keyword.iskeyword(normalized):
        normalized = f"{normalized}_value"
    return normalized


def _normalize_schema_type(value: Any) -> tuple[str, bool]:
    """Map an LLM-proposed type string to ``(SchemaFieldType value, required)``.

    Optional wrappers (``Optional[str]``, ``str | None``) strip to their inner
    type with ``required=False``; unknown types degrade to ``str`` so one
    malformed proposal never invalidates a complete draft.
    """
    raw = str(value).strip()
    required = True
    lowered = raw.lower()
    if lowered.startswith("optional[") and raw.endswith("]"):
        raw = raw[len("optional[") : -1].strip()
        required = False
    elif "|" in raw:
        parts = [part.strip() for part in raw.split("|")]
        required = not any(part.lower() in ("none", "null") for part in parts)
        parts = [part for part in parts if part.lower() not in ("none", "null")]
        raw = parts[0] if parts else "str"
    return _SCHEMA_ALIASES.get(raw.lower(), SchemaFieldType.STRING).value, required


class PydanticFieldDef(BaseModel):
    """One typed field inside a generated Pydantic model."""

    name: str = Field(description="Snake_case field name")
    type: SchemaFieldType = Field(description="Concrete field type")
    description: str | None = Field(default=None, description="Field description")
    required: bool = Field(default=True, description="Whether the field is required")
    literal_values: list[str | int] | None = Field(
        default=None, description="Allowed values for Literal types"
    )
    nested_model: PydanticModelSchema | None = Field(
        default=None, description="Nested Pydantic model for PYDANTIC_MODEL fields"
    )

    @field_validator("name", mode="before")
    @classmethod
    def _clean_name(cls, value: Any) -> str:
        return _sanitize_identifier(value)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("type") is not None:
            data = dict(data)
            type_value, required = _normalize_schema_type(data["type"])
            data.setdefault("required", required)
            data["type"] = type_value
        return data

    @model_validator(mode="after")
    def _repair_type(self) -> PydanticFieldDef:
        """Downgrade degenerate Literal/Pydantic proposals to plain strings."""
        if self.type is SchemaFieldType.LITERAL and not self.literal_values:
            self.type = SchemaFieldType.STRING
        if self.type is SchemaFieldType.PYDANTIC_MODEL and self.nested_model is None:
            self.type = SchemaFieldType.STRING
        return self

    def annotation(self) -> Any:
        """Return the Python annotation object for this field."""
        if self.type is SchemaFieldType.LITERAL:
            base: Any = Literal[tuple(self.literal_values or ())]  # ty: ignore[invalid-type-form]
        elif self.type is SchemaFieldType.PYDANTIC_MODEL:
            base = self.nested_model.build_model() if self.nested_model else str
        else:
            base = _SCHEMA_ANNOTATIONS[self.type]
        return base if self.required else base | None

    def annotation_source(self) -> str:
        """Return the source-code annotation string for this field."""
        if self.type is SchemaFieldType.LITERAL:
            base = f"Literal[{', '.join(repr(v) for v in self.literal_values or ())}]"
        elif self.type is SchemaFieldType.PYDANTIC_MODEL and self.nested_model:
            base = self.nested_model.model_name
        else:
            base = self.type.value
        return base if self.required else f"{base} | None"

    def field_source(self) -> str:
        """Render this field as Pydantic source code at one indent level."""
        args: list[str] = []
        if self.description:
            args.append(f"description={self.description!r}")
        if not self.required:
            args.append("default=None")
        initializer = f"Field({', '.join(args)})" if args else "Field()"
        return f"    {self.name}: {self.annotation_source()} = {initializer}"


class PydanticModelSchema(BaseModel):
    """Complete Pydantic model definition proposed by the signature architect."""

    model_name: str = Field(description="PascalCase Pydantic model class name")
    description: str | None = Field(default=None, description="Model docstring")
    fields: list[PydanticFieldDef] = Field(
        default_factory=list, description="Typed model fields"
    )
    _built_model: type[BaseModel] | None = PrivateAttr(default=None)

    @field_validator("model_name", mode="before")
    @classmethod
    def _clean_model_name(cls, value: Any) -> str:
        words = re.findall(r"[a-zA-Z0-9]+", str(value))
        name = "".join(word[:1].upper() + word[1:] for word in words)
        if not name:
            name = "GeneratedModel"
        return f"Model{name}" if name[0].isdigit() else name

    @model_validator(mode="after")
    def _dedupe_fields(self) -> PydanticModelSchema:
        """Guarantee unique field names so ``create_model`` cannot collide."""
        used: set[str] = set()
        for field in self.fields:
            if field.name in used:
                suffix = 2
                while f"{field.name}_{suffix}" in used:
                    suffix += 1
                field.name = f"{field.name}_{suffix}"
            used.add(field.name)
        return self

    def ordered_models(self, _depth: int = 0) -> list[PydanticModelSchema]:
        """Return nested models depth-first, then this model last."""
        if _depth > _MAX_SCHEMA_DEPTH:
            raise ValueError(
                f"Pydantic model nesting exceeds {_MAX_SCHEMA_DEPTH} levels"
            )
        models: list[PydanticModelSchema] = []
        for field in self.fields:
            nested = field.nested_model
            if field.type is SchemaFieldType.PYDANTIC_MODEL and nested is not None:
                if not any(m is nested for m in models):
                    models.extend(nested.ordered_models(_depth + 1))
        models.append(self)
        return models

    def to_code(self) -> str:
        """Render this model (not its nested models) as importable Python source."""
        lines = [f"class {self.model_name}(BaseModel):"]
        if self.description:
            escaped = self.description.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
            lines.append(f'    """{escaped}"""')
            lines.append("")
        for field in self.fields:
            lines.append(field.field_source())
        if not self.fields:
            lines.append("    pass")
        return "\n".join(lines)

    def build_model(self) -> type[BaseModel]:
        """Materialize this schema as a real Pydantic model class (cached)."""
        if self._built_model is None:
            definitions: dict[str, tuple[Any, Any]] = {}
            for field in self.fields:
                if field.required:
                    info = (
                        Field(description=field.description)
                        if field.description
                        else FieldInfo()
                    )
                else:
                    info = Field(default=None, description=field.description)
                definitions[field.name] = (field.annotation(), info)
            self._built_model = cast(
                "type[BaseModel]",
                create_model(
                    self.model_name,
                    __doc__=self.description or None,
                    **cast(Any, definitions),
                ),
            )
        return self._built_model


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
    literal_values: list[str | int] | None = Field(
        default=None, description="Allowed values when the type is a Literal"
    )
    model_schema: PydanticModelSchema | None = Field(
        default=None, description="Pydantic model definition for structured fields"
    )

    @property
    def resolved_type(self) -> type[Any] | types.UnionType | types.GenericAlias:
        """Resolve the natural-language type to an actual Python type.

        Structured fields resolve to their generated Pydantic model class,
        literal fields to ``typing.Literal``; everything else delegates to the
        TypeResolver.
        """
        if self.model_schema is not None:
            return self.model_schema.build_model()
        if self.literal_values:
            return Literal[tuple(self.literal_values)]  # ty: ignore[invalid-type-form]
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
    source_kind: Literal["prompt", "dataset"] = Field(
        default="prompt", description="Normalized source category"
    )
    data_profile: dict[str, Any] | None = Field(
        default=None, description="Precomputed profile for dataset sources"
    )
    sample_rows: list[dict[str, Any]] = Field(
        default_factory=list, description="Representative rows for dataset sources"
    )


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

    @field_validator("name")
    @classmethod
    def _ensure_signature_suffix(cls, value: str) -> str:
        """Guarantee the class name ends with ``Signature``.

        Every generated ``dspy.Signature`` subclass must be named like
        ``TicketClassificationSignature``. Names that already carry the
        suffix (any casing) pass through untouched; everything else gets
        ``Signature`` appended.
        """
        value = value.strip() or "AutoSignature"
        if not value.lower().endswith("signature"):
            value = f"{value}Signature"
        return value

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
