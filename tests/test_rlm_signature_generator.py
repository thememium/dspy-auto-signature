"""Tests for unified RLM signature generation without LLM calls."""

from __future__ import annotations

from dspy_auto_signature.core.signature_builder import SignatureBuilder
from dspy_auto_signature.generator.rlm_signature_generator import RLMSignatureGenerator
from dspy_auto_signature.generator.rlm_signatures import (
    GenerateSDKSignature,
    GenerateSignature,
    ProposedSignature,
)
from dspy_auto_signature.types.signature_spec import (
    FieldSpec,
    FieldType,
    ParsedPrompt,
    SignatureSpec,
)


class TestUnifiedRLMContract:
    def test_single_signature_has_unified_context_and_draft(self) -> None:
        assert set(GenerateSignature.input_fields) == {
            "source_kind",
            "task_context",
            "examples_json",
            "data_profile_json",
            "sample_rows_json",
        }
        assert set(GenerateSignature.output_fields) == {"draft"}
        assert GenerateSignature.output_fields["draft"].annotation is ProposedSignature

    def test_generator_has_rlm_instances(self) -> None:
        generator = RLMSignatureGenerator(max_iterations=5, max_llm_calls=10)
        assert hasattr(generator, "rlm")
        assert hasattr(generator, "sdk_rlm")

    def test_sdk_signature_has_correct_fields(self) -> None:
        assert set(GenerateSDKSignature.input_fields) == {
            "sdk_format",
            "messages_json",
            "task_hint",
        }
        assert set(GenerateSDKSignature.output_fields) == {"draft"}


class TestUnifiedContext:
    def test_prompt_context_contains_examples(self) -> None:
        prompt = ParsedPrompt(
            instruction_text="Summarize the article.",
            examples=[{"input": "long", "output": "short"}],
            raw_input="Summarize the article.",
        )
        context = RLMSignatureGenerator._build_context(prompt)
        assert context["source_kind"] == "prompt"
        assert "long" in context["examples_json"]
        assert context["data_profile_json"] == "{}"

    def test_dataset_context_contains_profile_and_rows(self) -> None:
        rows = [{"message": "urgent", "label": "high"}]
        prompt = ParsedPrompt(
            instruction_text="Task: Predict label",
            raw_input=rows,
        )
        context = RLMSignatureGenerator._build_context(prompt)
        assert context["source_kind"] == "dataset"
        assert '"message"' in context["data_profile_json"]
        assert '"label": "high"' in context["sample_rows_json"]

    def test_vercel_messages_are_not_treated_as_dataset(self) -> None:
        messages = [{"role": "system", "content": "Summarize {article}"}]
        prompt = ParsedPrompt(
            instruction_text="Summarize {article}", raw_input=messages
        )
        assert not RLMSignatureGenerator._is_dataset(prompt)


class TestDraftNormalization:
    def test_complete_draft_becomes_buildable_spec(self) -> None:
        draft = {
            "name": "TicketClassifier",
            "instructions": "Classify support tickets by urgency.",
            "inputs": [
                {
                    "name": "ticket_message",
                    "description": "Full customer support ticket message",
                    "type": "string",
                }
            ],
            "outputs": [
                {
                    "name": "urgency",
                    "description": "Predicted urgency level",
                    "type": "literal low, high",
                }
            ],
        }
        spec = RLMSignatureGenerator._draft_to_spec(draft)
        assert spec.name == "TicketClassifier"
        assert spec.inputs[0].name == "ticket_message"
        assert spec.outputs[0].suggested_type == "literal low, high"
        SignatureBuilder.build(spec)

    def test_draft_accepts_aliases_and_repairs_fields(self) -> None:
        draft = {
            "name": "ticket classifier!",
            "doctrine": "Classify support tickets.",
            "inputs": {"field_name": "Ticket Message", "desc": "Customer request"},
            "outputs": [
                '{"name": "class", "description": "Predicted ticket class"}',
                {"missing": "name"},
            ],
        }
        spec = RLMSignatureGenerator._draft_to_spec(draft)
        assert spec.name == "TicketClassifier"
        assert spec.inputs[0].name == "ticket_message"
        assert spec.outputs[0].name == "class_value"
        SignatureBuilder.build(spec)

    def test_draft_repairs_input_output_collision(self) -> None:
        draft = {
            "name": "ScoreAnswer",
            "instructions": "Score the answer.",
            "inputs": [{"name": "score", "description": "Existing score"}],
            "outputs": [{"name": "score", "description": "Predicted score"}],
        }
        spec = RLMSignatureGenerator._draft_to_spec(draft)
        assert spec.outputs[0].name == "score_result"
        SignatureBuilder.build(spec)

    def test_bracketed_literal_type_builds_correctly(self) -> None:
        draft = {
            "name": "ClassifyUrgency",
            "instructions": "Classify urgency.",
            "inputs": [{"name": "message", "description": "Ticket message"}],
            "outputs": [
                {
                    "name": "urgency",
                    "description": "Predicted urgency",
                    "type": 'literal ["low", "medium", "high"]',
                }
            ],
        }
        spec = RLMSignatureGenerator._draft_to_spec(draft)
        signature = SignatureBuilder.build(spec)
        annotation = signature.output_fields["urgency"].annotation
        assert getattr(annotation, "__args__", ()) == (
            "low",
            "medium",
            "high",
        )

    def test_incomplete_draft_is_rejected_as_one_unit(self) -> None:
        try:
            RLMSignatureGenerator._draft_to_spec(
                {"name": "Incomplete", "instructions": "Do work.", "inputs": []}
            )
        except ValueError as exc:
            assert "incomplete signature draft" in str(exc)
        else:
            raise AssertionError("Expected incomplete draft to be rejected")


class TestGroundedFallback:
    def test_dataset_fallback_is_grounded_and_buildable(self) -> None:
        profile = {
            "columns": {
                "message": {"dtype": "text", "n_unique": 100, "null_rate": 0},
                "urgency": {
                    "dtype": "categorical",
                    "n_unique": 2,
                    "null_rate": 0,
                    "top_values": [
                        {"value": "high", "count": 2},
                        {"value": "low", "count": 2},
                    ],
                },
            }
        }
        spec = RLMSignatureGenerator._fallback_from_dataset(
            profile,
            "Dataset profile\n\nTask: Classify tickets by urgency",
        )
        assert [field.name for field in spec.inputs] == ["message"]
        assert [field.name for field in spec.outputs] == ["urgency"]
        assert spec.outputs[0].suggested_type == "literal high, low"
        SignatureBuilder.build(spec)

    def test_prompt_fallback_uses_placeholders_and_task_verb(self) -> None:
        spec = RLMSignatureGenerator._fallback_from_prompt(
            "Summarize {article} for {audience}"
        )
        assert [field.name for field in spec.inputs] == ["article", "audience"]
        assert [field.name for field in spec.outputs] == ["summary"]
        SignatureBuilder.build(spec)


class TestSDKDetection:
    def test_detects_openai_sdk_format(self) -> None:
        messages = [
            {"role": "system", "content": "You summarize articles."},
            {"role": "user", "content": "Summarize: {article}"},
        ]
        prompt = ParsedPrompt(
            instruction_text="You summarize articles.",
            raw_input=messages,
        )
        assert RLMSignatureGenerator._is_sdk_format(prompt)

    def test_detects_anthropic_sdk_format(self) -> None:
        messages = [
            {"role": "user", "content": "Analyze sentiment of {review}"},
            {"role": "assistant", "content": "The sentiment is positive."},
        ]
        prompt = ParsedPrompt(
            instruction_text="Analyze sentiment",
            raw_input=messages,
        )
        assert RLMSignatureGenerator._is_sdk_format(prompt)

    def test_detects_gemini_sdk_format(self) -> None:
        messages = [
            {"role": "user", "parts": [{"text": "Extract entities from {text}"}]},
        ]
        prompt = ParsedPrompt(
            instruction_text="Extract entities",
            raw_input=messages,
        )
        assert RLMSignatureGenerator._is_sdk_format(prompt)

    def test_rejects_plain_string(self) -> None:
        prompt = ParsedPrompt(
            instruction_text="Summarize the article.",
            raw_input="Summarize the article.",
        )
        assert not RLMSignatureGenerator._is_sdk_format(prompt)

    def test_rejects_dataset_list(self) -> None:
        rows = [{"message": "urgent", "label": "high"}]
        prompt = ParsedPrompt(
            instruction_text="Task: Predict label",
            raw_input=rows,
        )
        assert not RLMSignatureGenerator._is_sdk_format(prompt)


class TestSDKContext:
    def test_builds_openai_context(self) -> None:
        messages = [
            {"role": "system", "content": "You summarize articles."},
            {"role": "user", "content": "Summarize: {article}"},
        ]
        prompt = ParsedPrompt(
            instruction_text="You summarize articles.",
            raw_input=messages,
        )
        context = RLMSignatureGenerator._build_sdk_context(prompt)
        assert context["sdk_format"] == "openai"
        assert "You summarize articles" in context["messages_json"]
        assert context["task_hint"] == "You summarize articles."

    def test_builds_gemini_context(self) -> None:
        messages = [
            {"role": "user", "parts": [{"text": "Extract entities"}]},
        ]
        prompt = ParsedPrompt(
            instruction_text="Extract entities",
            raw_input=messages,
        )
        context = RLMSignatureGenerator._build_sdk_context(prompt)
        assert context["sdk_format"] == "gemini"

    def test_builds_anthropic_context(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Analyze {review}"}],
            },
        ]
        prompt = ParsedPrompt(
            instruction_text="Analyze reviews",
            raw_input=messages,
        )
        context = RLMSignatureGenerator._build_sdk_context(prompt)
        assert context["sdk_format"] == "anthropic"


class TestSDKSanitization:
    def test_removes_forbidden_field_names(self) -> None:
        spec = SignatureSpec(
            name="TestSpec",
            instructions="Test",
            inputs=[
                FieldSpec(name="role", description="Bad", field_type=FieldType.INPUT),
                FieldSpec(
                    name="article", description="Good", field_type=FieldType.INPUT
                ),
            ],
            outputs=[
                FieldSpec(
                    name="content", description="Bad", field_type=FieldType.OUTPUT
                ),
                FieldSpec(
                    name="summary", description="Good", field_type=FieldType.OUTPUT
                ),
            ],
        )
        clean = RLMSignatureGenerator._sanitize_sdk_spec(spec)
        assert [f.name for f in clean.inputs] == ["article"]
        assert [f.name for f in clean.outputs] == ["summary"]

    def test_raises_when_all_inputs_forbidden(self) -> None:
        spec = SignatureSpec(
            name="TestSpec",
            instructions="Test",
            inputs=[
                FieldSpec(name="role", description="Bad", field_type=FieldType.INPUT)
            ],
            outputs=[
                FieldSpec(
                    name="summary", description="Good", field_type=FieldType.OUTPUT
                )
            ],
        )
        try:
            RLMSignatureGenerator._sanitize_sdk_spec(spec)
        except ValueError as exc:
            assert "forbidden" in str(exc)
        else:
            raise AssertionError("Expected ValueError for forbidden-only inputs")

    def test_raises_when_all_outputs_forbidden(self) -> None:
        spec = SignatureSpec(
            name="TestSpec",
            instructions="Test",
            inputs=[
                FieldSpec(
                    name="article", description="Good", field_type=FieldType.INPUT
                )
            ],
            outputs=[
                FieldSpec(
                    name="content", description="Bad", field_type=FieldType.OUTPUT
                )
            ],
        )
        try:
            RLMSignatureGenerator._sanitize_sdk_spec(spec)
        except ValueError as exc:
            assert "forbidden" in str(exc)
        else:
            raise AssertionError("Expected ValueError for forbidden-only outputs")
