"""Tests for unified RLM signature generation without LLM calls."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import dspy
import pytest

from dspy_auto_signature.core.config import Config
from dspy_auto_signature.core.signature_builder import SignatureBuilder
from dspy_auto_signature.generator.rlm_signature_generator import RLMSignatureGenerator
from dspy_auto_signature.generator.rlm_signatures import (
    GenerateSDKSignature,
    GenerateSignature,
    ProposedField,
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


class TestStructuralFastPath:
    def test_sdk_messages_bypass_rlm_and_build_generic_message_spec(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.sdk_rlm = stub  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text="Analyze the sentiment of customer reviews.",
            raw_input=[
                {
                    "role": "user",
                    "content": "Analyze the sentiment of customer reviews.",
                },
                {
                    "role": "assistant",
                    "content": "I'll classify each review as positive, negative, or neutral.",
                },
            ],
        )

        spec = generator.forward(prompt)

        assert stub.calls == 0
        assert [field.name for field in spec.inputs] == ["message"]
        assert [field.name for field in spec.outputs] == ["response"]
        assert "sentiment" in spec.instructions.lower()
        SignatureBuilder.build(spec)

    def test_sdk_placeholders_become_input_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.sdk_rlm = stub  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text="",
            raw_input=[
                {"role": "system", "content": "You translate documents."},
                {
                    "role": "user",
                    "content": "Translate the {paragraph} into {language}.",
                },
            ],
        )

        spec = generator.forward(prompt)

        assert stub.calls == 0
        assert [field.name for field in spec.inputs] == ["paragraph", "language"]
        SignatureBuilder.build(spec)

    def test_sdk_assistant_json_becomes_typed_outputs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.sdk_rlm = stub  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text="",
            raw_input=[
                {
                    "role": "user",
                    "content": "Extract entities from this legal contract.",
                },
                {
                    "role": "assistant",
                    "content": '{"parties": "Acme Corp", "count": 3, "confidence": 0.9, "verified": true}',
                },
            ],
        )

        spec = generator.forward(prompt)

        assert stub.calls == 0
        assert [field.name for field in spec.outputs] == [
            "parties",
            "count",
            "confidence",
            "verified",
        ]
        assert [field.suggested_type for field in spec.outputs] == [
            "string",
            "integer",
            "float",
            "boolean",
        ]
        SignatureBuilder.build(spec)

    def test_sdk_user_json_payload_becomes_input_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.sdk_rlm = stub  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text="",
            raw_input=[
                {
                    "role": "user",
                    "content": '{"ticket_text": "Server is down", "priority": "high"}',
                },
            ],
        )

        spec = generator.forward(prompt)

        assert stub.calls == 0
        assert [field.name for field in spec.inputs] == ["ticket_text", "priority"]
        SignatureBuilder.build(spec)

    def test_prompt_placeholders_bypass_rlm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.rlm = stub  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text="Summarize {article} for {audience}.",
            raw_input="Summarize {article} for {audience}.",
        )

        spec = generator.forward(prompt)

        assert stub.calls == 0
        assert [field.name for field in spec.inputs] == ["article", "audience"]
        SignatureBuilder.build(spec)


class _StubRLM:
    """Stand-in for a dspy.RLM that returns a canned draft or raises."""

    def __init__(self, draft: Any = None, error: Exception | None = None) -> None:
        self.draft = draft
        self.error = error
        self.kwargs: dict[str, Any] = {}
        self.calls = 0

    def __call__(self, **kwargs: Any) -> Any:
        self.calls += 1
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return SimpleNamespace(draft=self.draft)


_COMPLETE_DRAFT: dict[str, Any] = {
    "name": "ArticleSummarizer",
    "instructions": "Summarize the article.",
    "inputs": [
        {"name": "article", "description": "The article text", "type": "string"}
    ],
    "outputs": [
        {"name": "summary", "description": "The generated summary", "type": "string"}
    ],
}


class TestForwardPromptPath:
    def test_forward_runs_rlm_and_normalizes_draft(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.rlm = stub  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text="Summarize the article.",
            raw_input="Summarize the article.",
        )

        spec = generator.forward(prompt)

        assert stub.kwargs["source_kind"] == "prompt"
        assert spec.name == "ArticleSummarizer"
        assert [field.name for field in spec.inputs] == ["article"]
        SignatureBuilder.build(spec)

    def test_forward_dataset_source_is_deterministic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_dataset_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.rlm = stub  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text="Predict label", raw_input=[{"message": "x", "label": "y"}]
        )

        spec = generator.forward(prompt)

        assert stub.calls == 0
        assert [field.name for field in spec.inputs] == ["message"]
        assert [field.name for field in spec.outputs] == ["label"]
        SignatureBuilder.build(spec)

    def test_forward_falls_back_when_rlm_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        generator.rlm = _StubRLM(error=RuntimeError("boom"))  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text="Summarize {article} for {audience}",
            raw_input="Summarize {article} for {audience}",
        )

        spec = generator.forward(prompt)

        assert [field.name for field in spec.inputs] == ["article", "audience"]
        assert [field.name for field in spec.outputs] == ["summary"]
        SignatureBuilder.build(spec)

    def test_forward_dataset_source_falls_back_to_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_dataset_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        generator.rlm = _StubRLM(error=RuntimeError("boom"))  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text="Do the analysis",
            raw_input=[{"message": "urgent", "label": "high"}],
        )

        spec = generator.forward(prompt)

        assert [field.name for field in spec.inputs] == ["message"]
        assert [field.name for field in spec.outputs] == ["label"]
        SignatureBuilder.build(spec)


class TestForwardSDKPath:
    def test_forward_sdk_runs_sdk_rlm_and_sanitizes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.sdk_rlm = stub  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text="You summarize articles.",
            raw_input=[{"role": "system", "content": "You summarize articles."}],
        )

        spec = generator.forward(prompt)

        assert stub.kwargs["sdk_format"] == "openai"
        assert spec.name == "ArticleSummarizer"
        SignatureBuilder.build(spec)

    def test_forward_sdk_falls_back_to_prompt_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        generator.sdk_rlm = _StubRLM(  # type: ignore[assignment]
            error=RuntimeError("boom")
        )
        prompt = ParsedPrompt(
            instruction_text="Summarize {article}.",
            raw_input=[{"role": "system", "content": "Summarize {article}."}],
        )

        spec = generator.forward(prompt)

        assert [field.name for field in spec.inputs] == ["article"]
        assert [field.name for field in spec.outputs] == ["summary"]
        SignatureBuilder.build(spec)


class TestSDKFormatEdgeCases:
    def test_empty_messages_are_generic(self) -> None:
        assert RLMSignatureGenerator._detect_sdk_format([]) == "generic"

    def test_message_without_content_is_generic(self) -> None:
        assert (
            RLMSignatureGenerator._detect_sdk_format([{"role": "system"}]) == "generic"
        )


class TestDatasetDetectionBranches:
    def test_explicit_source_kind_is_dataset(self) -> None:
        prompt = ParsedPrompt(
            instruction_text="Analyze",
            source_kind="dataset",
            raw_input="not tabular at all",
        )
        assert RLMSignatureGenerator._is_dataset(prompt)

    def test_none_raw_input_is_not_dataset(self) -> None:
        prompt = ParsedPrompt(instruction_text="Analyze", raw_input=None)
        assert not RLMSignatureGenerator._is_dataset(prompt)

    def test_dict_raw_input_is_dataset(self) -> None:
        prompt = ParsedPrompt(instruction_text="Analyze", raw_input={"a": 1})
        assert RLMSignatureGenerator._is_dataset(prompt)


class TestDraftRejection:
    def test_unparseable_draft_is_rejected_as_empty(self) -> None:
        try:
            RLMSignatureGenerator._draft_to_spec("this is not a mapping")
        except ValueError as exc:
            assert "no complete signature draft" in str(exc)
        else:
            raise AssertionError("Expected empty draft to be rejected")

    def test_placeholder_draft_is_rejected(self) -> None:
        draft = {
            "name": "AutoSignature",
            "instructions": "Real instructions.",
            "inputs": [{"name": "text", "description": "Input"}],
            "outputs": [{"name": "result", "description": "Output"}],
        }
        try:
            RLMSignatureGenerator._draft_to_spec(draft)
        except ValueError as exc:
            assert "placeholder" in str(exc)
        else:
            raise AssertionError("Expected placeholder draft to be rejected")


class TestFieldCoercionEdges:
    def test_plain_string_fields_are_named_and_junk_skipped(self) -> None:
        draft = {
            "name": "ReviewScore",
            "instructions": "Score the review.",
            "inputs": ["review quality", 42],
            "outputs": [{"name": "score", "description": "Predicted score"}],
        }
        spec = RLMSignatureGenerator._draft_to_spec(draft)
        assert [field.name for field in spec.inputs] == ["review_quality"]
        SignatureBuilder.build(spec)

    def test_pydantic_field_models_are_dumped(self) -> None:
        draft = {
            "name": "ScoreAnswer",
            "instructions": "Score the answer.",
            "inputs": [
                ProposedField(name="answer", description="The answer", type="string")
            ],
            "outputs": [{"name": "score", "description": "Predicted score"}],
        }
        spec = RLMSignatureGenerator._draft_to_spec(draft)
        assert [field.name for field in spec.inputs] == ["answer"]
        SignatureBuilder.build(spec)

    def test_malformed_json_field_collection_is_single_field(self) -> None:
        draft = {
            "name": "TextProcessor",
            "instructions": "Process the text.",
            "inputs": "not json at all",
            "outputs": [{"name": "result", "description": "The result"}],
        }
        spec = RLMSignatureGenerator._draft_to_spec(draft)
        assert [field.name for field in spec.inputs] == ["not_json_at_all"]
        SignatureBuilder.build(spec)

    def test_tuple_field_collection_is_list(self) -> None:
        draft = {
            "name": "ScoreAnswer",
            "instructions": "Score the answer.",
            "inputs": ({"name": "answer", "description": "The answer"},),
            "outputs": [{"name": "score", "description": "Predicted score"}],
        }
        spec = RLMSignatureGenerator._draft_to_spec(draft)
        assert [field.name for field in spec.inputs] == ["answer"]
        SignatureBuilder.build(spec)

    def test_duplicate_field_names_are_suffixed(self) -> None:
        draft = {
            "name": "ScoreAnswer",
            "instructions": "Score the answer.",
            "inputs": [
                {"name": "score", "description": "First score"},
                {"name": "score", "description": "Second score"},
                {"name": "score", "description": "Third score"},
            ],
            "outputs": [{"name": "verdict", "description": "The verdict"}],
        }
        spec = RLMSignatureGenerator._draft_to_spec(draft)
        assert [field.name for field in spec.inputs] == ["score", "score_2", "score_3"]
        SignatureBuilder.build(spec)

    def test_unusable_names_fall_back_to_valid_identifiers(self) -> None:
        draft = {
            "name": "!!!",
            "instructions": "Do the work.",
            "inputs": [
                {"name": "###", "description": "Symbols only"},
                {"name": "9 lives", "description": "Leading digit"},
            ],
            "outputs": [{"name": "result", "description": "The result"}],
        }
        spec = RLMSignatureGenerator._draft_to_spec(draft)
        assert spec.name == "GeneratedSignature"
        assert [field.name for field in spec.inputs] == ["field", "field_9_lives"]
        SignatureBuilder.build(spec)


class TestFallbackFromContext:
    def test_dataset_context_routes_to_dataset_fallback(self) -> None:
        profile = {
            "columns": {
                "message": {"dtype": "text", "n_unique": 100, "null_rate": 0},
                "urgency": {"dtype": "categorical", "n_unique": 2, "null_rate": 0},
            }
        }
        context = {
            "source_kind": "dataset",
            "task_context": "Classify tickets by urgency",
            "data_profile_json": json.dumps(profile),
        }
        spec = RLMSignatureGenerator._fallback_from_context(context)
        assert [field.name for field in spec.inputs] == ["message"]
        assert [field.name for field in spec.outputs] == ["urgency"]

    def test_empty_profile_falls_back_to_prompt(self) -> None:
        spec = RLMSignatureGenerator._fallback_from_dataset({}, "Classify things")
        assert [field.name for field in spec.inputs] == ["source_content"]
        assert [field.name for field in spec.outputs] == ["classification"]
        SignatureBuilder.build(spec)

    def test_prompt_context_routes_to_prompt_fallback(self) -> None:
        context = {
            "source_kind": "prompt",
            "task_context": "Summarize {article}",
        }
        spec = RLMSignatureGenerator._fallback_from_context(context)
        assert [field.name for field in spec.inputs] == ["article"]
        assert [field.name for field in spec.outputs] == ["summary"]


class TestDatasetFallbackTargetSelection:
    def test_target_column_selected_when_task_hint_has_no_column_name(self) -> None:
        profile = {
            "columns": {
                "message": {"dtype": "text", "n_unique": 100, "null_rate": 0},
                "label": {"dtype": "categorical", "n_unique": 3, "null_rate": 0},
            }
        }
        spec = RLMSignatureGenerator._fallback_from_dataset(profile, "Do the work")
        assert [field.name for field in spec.outputs] == ["label"]

    def test_last_column_is_output_when_nothing_matches(self) -> None:
        profile = {
            "columns": {
                "alpha": {"dtype": "text", "n_unique": 5, "null_rate": 0},
                "beta": {"dtype": "text", "n_unique": 5, "null_rate": 0},
            }
        }
        spec = RLMSignatureGenerator._fallback_from_dataset(profile, "Do the work")
        assert [field.name for field in spec.outputs] == ["beta"]

    def test_output_list_is_trimmed_when_every_column_matches(self) -> None:
        profile = {
            "columns": {
                "message": {"dtype": "text", "n_unique": 100, "null_rate": 0},
                "label": {"dtype": "categorical", "n_unique": 2, "null_rate": 0},
            }
        }
        spec = RLMSignatureGenerator._fallback_from_dataset(
            profile, "Dataset profile\n\nTask: Predict message and label"
        )
        assert [field.name for field in spec.inputs] == ["message"]
        assert [field.name for field in spec.outputs] == ["label"]

    def test_single_output_column_synthesizes_context_input(self) -> None:
        profile = {
            "columns": {
                "label": {"dtype": "categorical", "n_unique": 2, "null_rate": 0}
            }
        }
        spec = RLMSignatureGenerator._fallback_from_dataset(profile, "Do the work")
        assert [field.name for field in spec.outputs] == ["label"]
        assert [field.name for field in spec.inputs] == ["label_context"]
        SignatureBuilder.build(spec)


class TestTargetHeuristic:
    def test_target_column_with_few_unique_values(self) -> None:
        info = {"n_unique": 3}
        assert RLMSignatureGenerator._looks_like_target("sentiment", info)

    def test_high_cardinality_target_column_is_rejected(self) -> None:
        info = {"n_unique": 100}
        assert not RLMSignatureGenerator._looks_like_target("label", info)

    def test_non_target_column_is_rejected(self) -> None:
        info = {"n_unique": 3}
        assert not RLMSignatureGenerator._looks_like_target("message", info)


class TestFieldTypeFromProfile:
    def test_high_null_rate_adds_optional_marker(self) -> None:
        field = RLMSignatureGenerator._field_from_profile(
            "message", {"dtype": "text", "null_rate": 0.5}, FieldType.INPUT
        )
        assert field.suggested_type == "optional string"

    def test_literal_type_is_not_marked_optional(self) -> None:
        field = RLMSignatureGenerator._field_from_profile(
            "urgency",
            {
                "dtype": "categorical",
                "null_rate": 0.5,
                "top_values": [{"value": "high"}, {"value": "low"}],
            },
            FieldType.OUTPUT,
        )
        assert field.suggested_type == "literal high, low"


class TestInferPromptNames:
    def test_fallback_input_name_comes_from_task_noun(self) -> None:
        spec = RLMSignatureGenerator._fallback_from_prompt("Summarize this review")
        assert [field.name for field in spec.inputs] == ["review"]
        assert [field.name for field in spec.outputs] == ["summary"]
        SignatureBuilder.build(spec)

    def test_unrecognized_prompt_uses_generic_names(self) -> None:
        spec = RLMSignatureGenerator._fallback_from_prompt("Do the thing now")
        assert [field.name for field in spec.inputs] == ["source_content"]
        assert [field.name for field in spec.outputs] == ["task_result"]
        SignatureBuilder.build(spec)
