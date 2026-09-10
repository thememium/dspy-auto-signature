"""Tests for the types module."""

from dspy_auto_signature.types.signature_spec import (
    FieldSpec,
    FieldType,
    ParsedPrompt,
    SignatureSpec,
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
