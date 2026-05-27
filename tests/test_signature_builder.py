"""Tests for the SignatureBuilder."""

from __future__ import annotations

import dspy

from dspy_auto_signature.core.signature_builder import SignatureBuilder
from dspy_auto_signature.types.signature_spec import FieldSpec, FieldType, SignatureSpec


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

        assert issubclass(Sig, dspy.Signature)
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
            Sig.input_fields["input_text"].description
            == "The main input text to process"
        )
        assert Sig.output_fields["result"].description == "The processing result"
