"""Tests for the types module."""

import ast
import typing
from typing import Any, cast

import pytest
from pydantic import ValidationError as PydanticValidationError

from dspy_auto_signature.types.signature_spec import (
    FieldSpec,
    FieldType,
    ParsedPrompt,
    PydanticFieldDef,
    PydanticModelSchema,
    SignatureSpec,
    _normalize_schema_type,
    _sanitize_identifier,
)


class TestFieldSpec:
    def test_field_creation(self) -> None:
        field = FieldSpec(
            name="test_field",
            description="A test field",
            field_type=FieldType.INPUT,
        )
        assert field.name == "test_field"
        assert field.description == "A test field"
        assert field.suggested_type == "str"
        assert field.field_type == FieldType.INPUT

    def test_type_resolution(self) -> None:
        field = FieldSpec(
            name="items",
            description="List of items",
            suggested_type="list of strings",
            field_type=FieldType.OUTPUT,
        )
        resolved = field.resolved_type
        assert getattr(resolved, "__origin__", None) is list


class TestParsedPrompt:
    def test_creation(self) -> None:
        prompt = ParsedPrompt(
            instruction_text="Summarize this",
            examples=[{"input": "test", "output": "result"}],
        )
        assert prompt.instruction_text == "Summarize this"
        assert len(prompt.examples) == 1


class TestSignatureSpec:
    def test_to_signature_string(self) -> None:
        spec = SignatureSpec(
            name="Test",
            instructions="Test task",
            inputs=[
                FieldSpec(name="a", description="Input A", field_type=FieldType.INPUT),
                FieldSpec(name="b", description="Input B", field_type=FieldType.INPUT),
            ],
            outputs=[
                FieldSpec(
                    name="c", description="Output C", field_type=FieldType.OUTPUT
                ),
            ],
        )
        assert spec.to_signature_string() == "a, b -> c"

    def test_all_fields_property(self) -> None:
        spec = SignatureSpec(
            name="Test",
            instructions="Test",
            inputs=[FieldSpec(name="x", description="X", field_type=FieldType.INPUT)],
            outputs=[FieldSpec(name="y", description="Y", field_type=FieldType.OUTPUT)],
        )
        assert len(spec.all_fields) == 2
        assert spec.all_fields[0].name == "x"
        assert spec.all_fields[1].name == "y"

    def test_name_appends_signature_suffix(self) -> None:
        spec = SignatureSpec(
            name="TicketClassification", instructions="Classify tickets"
        )
        assert spec.name == "TicketClassificationSignature"

    def test_name_keeps_existing_signature_suffix(self) -> None:
        spec = SignatureSpec(
            name="TicketClassificationSignature", instructions="Classify tickets"
        )
        assert spec.name == "TicketClassificationSignature"

    def test_name_suffix_match_is_case_insensitive(self) -> None:
        spec = SignatureSpec(name="ticketclassifysignature", instructions="Test")
        assert spec.name == "ticketclassifysignature"

    def test_default_name_already_has_suffix(self) -> None:
        spec = SignatureSpec(instructions="Test")
        assert spec.name == "AutoSignature"

    def test_blank_name_falls_back_to_default_with_suffix(self) -> None:
        spec = SignatureSpec(name="   ", instructions="Test")
        assert spec.name == "AutoSignature"


def _contact_schema() -> dict:
    return {
        "model_name": "ContactRecord",
        "description": "A contact record",
        "fields": [
            {"name": "full name", "type": "str", "description": "Full name"},
            {"name": "age", "type": "Optional[int]", "description": "Age"},
            {
                "name": "priority",
                "type": "literal",
                "literal_values": ["low", "high"],
                "description": "Priority",
            },
            {
                "name": "address",
                "type": "pydantic",
                "description": "Mailing address",
                "nested_model": {
                    "model_name": "Address",
                    "fields": [
                        {"name": "street", "type": "string", "description": "Street"},
                        {
                            "name": "kind",
                            "type": "enum",
                            "literal_values": ["home", "work"],
                            "description": "Kind",
                        },
                    ],
                },
            },
            {"name": "weird", "type": "quantum flux", "description": "Unknown"},
        ],
    }


class TestPydanticModelSchema:
    def test_parses_lenient_type_aliases(self) -> None:
        schema = PydanticModelSchema.model_validate(_contact_schema())
        kinds = [field.type.value for field in schema.fields]
        assert kinds == [
            "str",
            "int",
            "Literal",
            "pydantic",
            "str",
        ]
        assert [field.required for field in schema.fields] == [
            True,
            False,
            True,
            True,
            True,
        ]

    def test_build_model_enforces_types(self) -> None:
        schema = PydanticModelSchema.model_validate(_contact_schema())
        Model: Any = schema.build_model()
        record = Model(
            full_name="Ada",
            priority="high",
            weird="w",
            address={"street": "1 Main", "kind": "home"},
        )
        assert record.age is None
        assert record.address.kind == "home"
        with pytest.raises(PydanticValidationError):
            Model(full_name="Ada", priority="bogus", weird="w")

    def test_optional_wraps_annotation(self) -> None:
        schema = PydanticModelSchema.model_validate(_contact_schema())
        age = schema.fields[1]
        assert age.required is False
        assert age.annotation() == int | None

    def test_unknown_type_degrades_to_string(self) -> None:
        schema = PydanticModelSchema.model_validate(_contact_schema())
        assert schema.fields[4].type.value == "str"

    def test_field_names_sanitized(self) -> None:
        schema = PydanticModelSchema.model_validate(_contact_schema())
        assert schema.fields[0].name == "full_name"

    def test_duplicate_fields_deduped(self) -> None:
        schema = PydanticModelSchema.model_validate(
            {
                "model_name": "M",
                "fields": [
                    {"name": "a b", "type": "str", "description": "x"},
                    {"name": "a_b", "type": "int", "description": "y"},
                ],
            }
        )
        assert [field.name for field in schema.fields] == ["a_b", "a_b_2"]

    def test_literal_without_values_downgrades_to_string(self) -> None:
        field = PydanticFieldDef(
            name="status", type=cast(Any, "Literal"), description="Status"
        )
        assert field.type.value == "str"

    def test_ordered_models_nested_first(self) -> None:
        schema = PydanticModelSchema.model_validate(_contact_schema())
        assert [model.model_name for model in schema.ordered_models()] == [
            "Address",
            "ContactRecord",
        ]

    def test_to_code_parses_and_matches_semantics(self) -> None:
        schema = PydanticModelSchema.model_validate(_contact_schema())
        source = (
            "from pydantic import BaseModel, Field\n"
            "from typing import Literal\n\n\n"
            + "\n\n\n".join(model.to_code() for model in schema.ordered_models())
        )
        tree = ast.parse(source)
        names = [node.name for node in tree.body if isinstance(node, ast.ClassDef)]
        assert names == ["Address", "ContactRecord"]
        assert "full_name: str = Field(description='Full name')" in source
        assert "age: int | None" in source
        assert "priority: Literal['low', 'high']" in source

    def test_to_code_empty_model_gets_pass(self) -> None:
        schema = PydanticModelSchema.model_validate(
            {"model_name": "Empty", "fields": []}
        )
        assert "    pass" in schema.to_code()

    def test_build_model_cached(self) -> None:
        schema = PydanticModelSchema.model_validate(_contact_schema())
        assert schema.build_model() is schema.build_model()

    def test_excessive_nesting_rejected(self) -> None:
        model: dict = {"model_name": "Loop", "fields": []}
        inner = model
        for _ in range(12):
            child: dict = {"model_name": "Child", "fields": []}
            inner["fields"] = [
                {
                    "name": "child",
                    "type": "pydantic",
                    "description": "Child",
                    "nested_model": child,
                }
            ]
            inner = child
        schema = PydanticModelSchema.model_validate(model)
        with pytest.raises(ValueError, match="nesting exceeds"):
            schema.ordered_models()

    def test_optional_maps_to_required_flag(self) -> None:
        schema = PydanticModelSchema.model_validate(_contact_schema())
        assert schema.fields[1].required is False
        assert schema.fields[0].required is True


class TestFieldSpecPydantic:
    def test_resolved_type_is_generated_model(self) -> None:
        field = FieldSpec(
            name="contact",
            description="The contact record",
            field_type=FieldType.OUTPUT,
            suggested_type="pydantic",
            model_schema=PydanticModelSchema.model_validate(
                {
                    "model_name": "ContactRecord",
                    "fields": [{"name": "name", "type": "str", "description": "Name"}],
                }
            ),
        )
        model: Any = field.resolved_type
        instance = model(name="Ada")
        assert instance.name == "Ada"
        assert field.resolved_type is model  # cached, stable identity

    def test_literal_values_resolve_to_literal(self) -> None:
        field = FieldSpec(
            name="severity",
            description="Severity level",
            field_type=FieldType.OUTPUT,
            literal_values=["low", "medium", "high"],
        )
        resolved = field.resolved_type
        assert typing.get_origin(resolved) is typing.Literal
        assert resolved.__args__ == ("low", "medium", "high")


class TestSchemaHelpers:
    def test_sanitize_identifier_fallbacks(self) -> None:
        assert _sanitize_identifier("") == "field"
        assert _sanitize_identifier("!!!") == "field"
        assert _sanitize_identifier("123abc") == "field_123abc"
        assert _sanitize_identifier("class") == "class_value"

    def test_normalize_schema_type_union_branch(self) -> None:
        assert _normalize_schema_type("str | None") == ("str", False)
        assert _normalize_schema_type("str | null") == ("str", False)
        assert _normalize_schema_type("str | int") == ("str", True)
        assert _normalize_schema_type("None | str") == ("str", False)
        assert _normalize_schema_type("int") == ("int", True)

    def test_pydantic_type_without_nested_model_degrades(self) -> None:
        field = PydanticFieldDef(
            name="payload", type=cast(Any, "pydantic"), description="Payload"
        )
        assert field.type.value == "str"

    def test_empty_model_name_gets_placeholder(self) -> None:
        schema = PydanticModelSchema.model_validate({"model_name": "!!!", "fields": []})
        assert schema.model_name == "GeneratedModel"

    def test_digit_model_name_gets_prefix(self) -> None:
        schema = PydanticModelSchema.model_validate(
            {"model_name": "9 lives", "fields": []}
        )
        assert schema.model_name == "Model9Lives"

    def test_triple_duplicate_fields_deduped(self) -> None:
        schema = PydanticModelSchema.model_validate(
            {
                "model_name": "M",
                "fields": [
                    {"name": "x", "type": "str", "description": "a"},
                    {"name": "x", "type": "int", "description": "b"},
                    {"name": "x", "type": "float", "description": "c"},
                ],
            }
        )
        assert [field.name for field in schema.fields] == ["x", "x_2", "x_3"]
