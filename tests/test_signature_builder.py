"""Tests for the SignatureBuilder."""

from __future__ import annotations

from typing import cast

import dspy
import pytest
from pydantic.fields import FieldInfo

from dspy_auto_signature.core.signature_builder import SignatureBuilder
from dspy_auto_signature.types.signature_spec import FieldSpec, FieldType, SignatureSpec


def _get_json_schema_extra(field_info: FieldInfo, key: str) -> object:
    """Safely extract a value from a FieldInfo's json_schema_extra dict."""
    extra = field_info.json_schema_extra
    if extra is None or callable(extra):
        return None
    return extra.get(key)


class TestSignatureBuilder:
    def test_build_simple_signature(self) -> None:
        spec = SignatureSpec(
            name="TestSummarizer",
            instructions="Summarize text into bullet points.",
            inputs=[
                FieldSpec(
                    name="text",
                    description="Text to summarize",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="summary",
                    description="Bullet point summary",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        Sig = SignatureBuilder.build(spec)

        assert issubclass(cast(type, Sig), dspy.Signature)
        assert Sig.__name__ == "TestSummarizer"
        assert Sig.instructions == "Summarize text into bullet points."
        assert "text" in Sig.input_fields
        assert "summary" in Sig.output_fields

    def test_build_with_multiple_fields(self) -> None:
        spec = SignatureSpec(
            name="MultiFieldTask",
            instructions="Do something complex.",
            inputs=[
                FieldSpec(
                    name="query", description="Search query", field_type=FieldType.INPUT
                ),
                FieldSpec(
                    name="context",
                    description="Additional context",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="answer", description="The answer", field_type=FieldType.OUTPUT
                ),
                FieldSpec(
                    name="confidence",
                    description="Confidence score",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        Sig = SignatureBuilder.build(spec)

        assert len(Sig.input_fields) == 2
        assert len(Sig.output_fields) == 2
        assert list(Sig.input_fields.keys()) == ["query", "context"]
        assert list(Sig.output_fields.keys()) == ["answer", "confidence"]

    def test_field_descriptions_preserved(self) -> None:
        spec = SignatureSpec(
            name="DescTest",
            instructions="Test descriptions.",
            inputs=[
                FieldSpec(
                    name="input_text",
                    description="The main input text to process",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="result",
                    description="The processing result",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        Sig = SignatureBuilder.build(spec)

        assert (
            _get_json_schema_extra(Sig.input_fields["input_text"], "desc")
            == "The main input text to process"
        )
        assert (
            Sig.input_fields["input_text"].description
            == "The main input text to process"
        )
        assert (
            _get_json_schema_extra(Sig.output_fields["result"], "desc")
            == "The processing result"
        )
        assert Sig.output_fields["result"].description == "The processing result"

    def test_signature_has_docstring(self) -> None:
        """Generated signatures must have their instructions as a class docstring."""
        spec = SignatureSpec(
            name="DocstringTest",
            instructions="This is the task description that should become the docstring.",
            inputs=[
                FieldSpec(
                    name="text",
                    description="Input text",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="output",
                    description="Output text",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        Sig = SignatureBuilder.build(spec)

        assert Sig.__doc__ is not None
        assert "task description" in Sig.__doc__
        assert Sig.instructions == spec.instructions

    def test_all_fields_have_types(self) -> None:
        """Every field in the generated signature must have a type annotation."""
        spec = SignatureSpec(
            name="TypedTest",
            instructions="Test type annotations.",
            inputs=[
                FieldSpec(
                    name="text",
                    description="A string input",
                    suggested_type="string",
                    field_type=FieldType.INPUT,
                ),
                FieldSpec(
                    name="count",
                    description="An integer input",
                    suggested_type="integer",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="result",
                    description="A list output",
                    suggested_type="list of strings",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        Sig = SignatureBuilder.build(spec)

        assert Sig.__annotations__["text"] is str
        assert Sig.__annotations__["count"] is int
        assert Sig.__annotations__["result"] == list[str]

    def test_all_fields_are_dspy_fields(self) -> None:
        """Input fields must use InputField and output fields must use OutputField."""
        spec = SignatureSpec(
            name="FieldKindTest",
            instructions="Test field kinds.",
            inputs=[
                FieldSpec(
                    name="query",
                    description="Search query",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="answer",
                    description="The answer",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        Sig = SignatureBuilder.build(spec)

        assert (
            _get_json_schema_extra(Sig.input_fields["query"], "__dspy_field_type")
            == "input"
        )
        assert (
            _get_json_schema_extra(Sig.output_fields["answer"], "__dspy_field_type")
            == "output"
        )

    def test_missing_description_raises(self) -> None:
        """Building a signature with an empty description must raise ValueError."""
        spec = SignatureSpec(
            name="BadSpec",
            instructions="Test missing description.",
            inputs=[
                FieldSpec(
                    name="text",
                    description="",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[],
        )

        with pytest.raises(ValueError, match="missing a description"):
            SignatureBuilder.build(spec)

    def test_full_class_based_signature(self) -> None:
        """A complete signature must have docstring, typed fields, and descriptions."""
        spec = SignatureSpec(
            name="CompleteSignature",
            instructions="Given a product review, extract the sentiment and key points.",
            inputs=[
                FieldSpec(
                    name="review",
                    description="The customer product review to analyze",
                    suggested_type="string",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="sentiment",
                    description="The overall sentiment: positive, negative, or neutral",
                    suggested_type="string",
                    field_type=FieldType.OUTPUT,
                ),
                FieldSpec(
                    name="key_points",
                    description="A list of key points mentioned in the review",
                    suggested_type="list of strings",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        Sig = SignatureBuilder.build(spec)

        assert Sig.__doc__ is not None
        assert "product review" in Sig.__doc__

        assert Sig.__annotations__["review"] is str
        assert (
            Sig.input_fields["review"].description
            == "The customer product review to analyze"
        )
        assert (
            _get_json_schema_extra(Sig.input_fields["review"], "desc")
            == "The customer product review to analyze"
        )

        assert Sig.__annotations__["sentiment"] is str
        assert Sig.__annotations__["key_points"] == list[str]
        assert (
            Sig.output_fields["sentiment"].description
            == "The overall sentiment: positive, negative, or neutral"
        )
        assert (
            Sig.output_fields["key_points"].description
            == "A list of key points mentioned in the review"
        )

    def test_to_source_generates_valid_python(self) -> None:
        spec = SignatureSpec(
            name="SourceTest",
            instructions="Test the to_source method.",
            inputs=[
                FieldSpec(
                    name="text",
                    description="Input text",
                    suggested_type="string",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="result",
                    description="The result",
                    suggested_type="list of strings",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        Sig = SignatureBuilder.build(spec)
        source = Sig.to_source()

        assert "class SourceTest(dspy.Signature):" in source
        assert 'text: str = dspy.InputField(desc="Input text")' in source
        assert 'result: list[str] = dspy.OutputField(desc="The result")' in source
        assert "import dspy" in source

        compile(source, "<generated>", "exec")

    def test_signature_builder_to_source_classmethod(self) -> None:
        spec = SignatureSpec(
            name="ClassMethodTest",
            instructions="Test the classmethod.",
            inputs=[
                FieldSpec(
                    name="query",
                    description="Search query",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="answer",
                    description="The answer",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        source = SignatureBuilder.to_source(spec)

        assert "class ClassMethodTest(dspy.Signature):" in source
        assert 'query: str = dspy.InputField(desc="Search query")' in source
        assert 'answer: str = dspy.OutputField(desc="The answer")' in source
