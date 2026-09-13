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
    PydanticModelSchema,
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

    def test_generator_has_llm_architect_instances(self) -> None:
        generator = RLMSignatureGenerator(max_iterations=5, max_llm_calls=10)
        assert hasattr(generator, "rlm")
        assert hasattr(generator, "sdk_rlm")
        assert hasattr(generator, "cot")
        assert hasattr(generator, "cot_sdk")
        # The iteration cap must land on both RLMs regardless of whether the
        # installed dspy names it max_iterations (<3.3) or max_iters (>=3.3).
        for rlm in (generator.rlm, generator.sdk_rlm):
            cap = getattr(rlm, "max_iters", None) or getattr(
                rlm, "max_iterations", None
            )
            assert cap == 5

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

    def test_prompt_fallback_extracts_enumerated_outputs(self) -> None:
        spec = RLMSignatureGenerator._fallback_from_prompt(
            "Given a customer support ticket with {message},"
            "predict the urgency level and sentiment."
        )
        assert [field.name for field in spec.inputs] == ["message"]
        assert [field.name for field in spec.outputs] == ["urgency", "sentiment"]
        SignatureBuilder.build(spec)

    def test_prompt_fallback_uses_extracted_output_without_keyword_signal(
        self,
    ) -> None:
        spec = RLMSignatureGenerator._fallback_from_prompt(
            "Given {ticket}, predict the urgency level."
        )
        assert [field.name for field in spec.inputs] == ["ticket"]
        assert [field.name for field in spec.outputs] == ["urgency"]
        SignatureBuilder.build(spec)

    def test_extract_output_names_skips_unusable_segments(self) -> None:
        names = RLMSignatureGenerator._extract_output_names(
            "Given {text}, predict the urgency level and do the thing very carefully, !!",
            reserved=set(),
        )
        assert names == ["urgency"]

    def test_extract_output_names_skips_reserved_names(self) -> None:
        names = RLMSignatureGenerator._extract_output_names(
            "Given {text}, predict the text and urgency.",
            reserved={"text"},
        )
        assert names == ["urgency"]


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


class TestWarmInterpreter:
    def test_thread_interpreter_is_reused_until_reset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dspy_auto_signature.generator.rlm_signature_generator as gen_mod

        created: list[object] = []

        class _FakeInterpreter:
            def __init__(self) -> None:
                created.append(self)

        monkeypatch.setattr(gen_mod, "PythonInterpreter", _FakeInterpreter)
        gen_mod._INTERPRETERS.interpreter = None
        try:
            first = gen_mod._thread_interpreter()
            second = gen_mod._thread_interpreter()
            assert first is second
            assert len(created) == 1
        finally:
            gen_mod._INTERPRETERS.interpreter = None

    def test_generators_share_the_thread_interpreter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dspy_auto_signature.generator.rlm_signature_generator as gen_mod

        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        first = RLMSignatureGenerator()
        second = RLMSignatureGenerator()
        if first._pass_warm_interpreter:
            # dspy >=3.3 resolves the interpreter per call from the thread-local.
            assert not hasattr(first.rlm, "_interpreter")
            assert gen_mod._thread_interpreter() is gen_mod._thread_interpreter()
        else:
            assert first.rlm._interpreter is second.rlm._interpreter

    def test_close_interpreter_for_thread_shuts_down(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dspy_auto_signature.generator.rlm_signature_generator as gen_mod

        class _FakeInterpreter:
            def __init__(self) -> None:
                self.shutdown_calls = 0

            def shutdown(self) -> None:
                self.shutdown_calls += 1

        fake = _FakeInterpreter()
        gen_mod._INTERPRETERS.interpreter = fake
        try:
            gen_mod.close_interpreter()
            assert fake.shutdown_calls == 1
            assert gen_mod._INTERPRETERS.interpreter is None
        finally:
            gen_mod._INTERPRETERS.interpreter = None

    def test_generator_cache_is_thread_local(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import threading

        import dspy_auto_signature as das

        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        results: dict[str, Any] = {}
        shared_lm = dspy.LM("openai/gpt-4o")

        def worker(key: str) -> None:
            generator = das._get_generator(shared_lm)
            results[key] = generator

        das._get_generator(shared_lm)  # warm main thread
        threads = [threading.Thread(target=worker, args=(key,)) for key in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert results["a"] is not results["b"]
        if results["a"]._pass_warm_interpreter:
            # dspy >=3.3 resolves interpreters from per-thread state at call time.
            assert not hasattr(results["a"].rlm, "_interpreter")
        else:
            assert results["a"].rlm._interpreter is not results["b"].rlm._interpreter
        assert das._get_generator(shared_lm) is das._get_generator(shared_lm)

    def test_interpreter_proxy_delegates_and_cleans_up(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import gc

        import dspy_auto_signature.generator.rlm_signature_generator as gen_mod

        class _FakeInterpreter:
            def __init__(self) -> None:
                self.tools: dict[str, str] = {}
                self.output_fields: list[str] | None = None
                self._tools_registered = False
                self.shutdown_calls = 0
                self.start_calls = 0
                self.executed: list[tuple[str, dict[str, Any] | None]] = []
                self.extra = "delegated"

            def start(self) -> None:
                self.start_calls += 1

            def execute(
                self, code: str, variables: dict[str, Any] | None = None
            ) -> str:
                self.executed.append((code, variables))
                return "ok"

            def shutdown(self) -> None:
                self.shutdown_calls += 1

        monkeypatch.setattr(gen_mod, "PythonInterpreter", _FakeInterpreter)
        gen_mod._INTERPRETERS.interpreter = None
        try:
            proxy = gen_mod._thread_interpreter()
            inner = proxy._interpreter
            proxy.start()
            assert proxy.execute("print(1)", variables={"x": 1}) == "ok"
            assert inner.start_calls == 1
            assert inner.executed == [("print(1)", {"x": 1})]
            assert proxy.extra == "delegated"  # __getattr__ fallback
            proxy.tools["llm_query"] = "fn"  # attribute passthrough
            assert inner.tools == {"llm_query": "fn"}
            proxy.output_fields = ["draft"]
            assert inner.output_fields == ["draft"]
            proxy._tools_registered = False
            assert inner._tools_registered is False

            proxy.shutdown()
            gen_mod._INTERPRETERS.interpreter = None
            del proxy
            gc.collect()
            assert inner.shutdown_calls == 2  # explicit + __del__ cleanup
        finally:
            gen_mod._INTERPRETERS.interpreter = None

    def test_call_architect_hands_warm_interpreter_to_rlm_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import dspy_auto_signature.generator.rlm_signature_generator as gen_mod

        class _SpyModule:
            def __init__(self) -> None:
                self.interpreter: Any = None
                self.kwargs: dict[str, Any] = {}

            def __call__(self, *args: Any, **kwargs: Any) -> None:
                if args:
                    self.interpreter = args[0]
                self.kwargs = kwargs

        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        rlm_stub, cot_stub = _SpyModule(), _SpyModule()
        generator.rlm = rlm_stub  # type: ignore[assignment]
        generator.cot = cot_stub  # ty: ignore[invalid-assignment]
        generator._pass_warm_interpreter = True
        sentinel = object()
        monkeypatch.setattr(gen_mod, "_thread_interpreter", lambda: sentinel)

        context = {"source_kind": "prompt"}
        generator._call_architect(generator.rlm, context)
        generator._call_architect(generator.cot, context)

        assert rlm_stub.interpreter is sentinel
        assert rlm_stub.kwargs == context
        assert cot_stub.interpreter is None
        assert cot_stub.kwargs == context


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

    def test_rlm_mode_forces_rlm_on_structured_sdk_input(
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
            ],
        )

        spec = generator.forward(prompt, mode="rlm")

        assert stub.calls == 1
        assert spec.name == "ArticleSummarizer"
        SignatureBuilder.build(spec)

    def test_cot_mode_routes_sdk_input_through_cot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.cot_sdk = stub  # ty: ignore[invalid-assignment]
        prompt = ParsedPrompt(
            instruction_text="Analyze the sentiment of customer reviews.",
            raw_input=[
                {
                    "role": "user",
                    "content": "Analyze the sentiment of customer reviews.",
                },
            ],
        )

        spec = generator.forward(prompt, mode="cot")

        assert stub.calls == 1
        assert spec.name == "ArticleSummarizer"
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

    def test_sdk_structural_spec_preserves_task_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.sdk_rlm = stub  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text="Write a README.\n\nTask: Target markdown output",
            raw_input=[
                {"role": "system", "content": "You are a technical writer."},
                {"role": "user", "content": "Write a README."},
            ],
        )

        spec = generator.forward(prompt)

        assert stub.calls == 0
        assert "Task: Target markdown output" in spec.instructions
        SignatureBuilder.build(spec)

    def test_hint_drives_output_name_for_prose_sdk_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.sdk_rlm = stub  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text=".\n\nTask: Classify support ticket urgency",
            raw_input=[
                {"role": "user", "content": "Server is on fire."},
                {"role": "assistant", "content": "Got it, I'll take a look."},
            ],
        )

        spec = generator.forward(prompt)

        assert stub.calls == 0
        assert [field.name for field in spec.outputs] == ["classification"]
        SignatureBuilder.build(spec)

    def test_fenced_json_in_assistant_content_is_decoded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.sdk_rlm = stub  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text="",
            raw_input=[
                {"role": "user", "content": "Extract entities."},
                {
                    "role": "assistant",
                    "content": '```json\n{"parties": "Acme", "items": ["a", "b"]}\n```',
                },
            ],
        )

        spec = generator.forward(prompt)

        assert stub.calls == 0
        assert [field.name for field in spec.outputs] == ["parties", "items"]
        assert [field.suggested_type for field in spec.outputs] == [
            "string",
            "list of strings",
        ]
        SignatureBuilder.build(spec)

    def test_empty_and_tool_messages_shape_instructions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.sdk_rlm = stub  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text="",
            raw_input=[
                {"role": "tool", "content": "Tool trace: search executed."},
                {"role": "user", "content": ""},
                {"role": "user", "content": "Summarize the log."},
            ],
        )

        spec = generator.forward(prompt)

        assert stub.calls == 0
        assert "Tool trace" in spec.instructions
        assert "Summarize the log." in spec.instructions
        SignatureBuilder.build(spec)

    def test_fast_mode_falls_back_for_system_only_sdk(
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

        spec = generator.forward(prompt, mode="fast")

        assert stub.calls == 0
        assert [field.name for field in spec.inputs] == ["article"]
        SignatureBuilder.build(spec)

    def test_dataset_structural_failure_falls_back_to_cot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.cot = stub  # ty: ignore[invalid-assignment]

        def _raise(cls: type, profile: dict, task_context: str) -> SignatureSpec:
            raise ValueError("bad profile")

        monkeypatch.setattr(
            RLMSignatureGenerator,
            "_fallback_from_dataset",
            classmethod(_raise),
        )
        prompt = ParsedPrompt(
            instruction_text="Analyze the data",
            raw_input=[{"message": "urgent", "label": "high"}],
        )

        spec = generator.forward(prompt)

        assert stub.calls == 1
        assert spec.name == "ArticleSummarizer"

    def test_rlm_failure_falls_back_for_structureless_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        generator.rlm = _StubRLM(error=RuntimeError("boom"))  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text="Given an article, produce a concise summary.",
            raw_input="Given an article, produce a concise summary.",
        )

        spec = generator.forward(prompt, mode="rlm")

        assert [field.name for field in spec.inputs] == ["article"]
        assert [field.name for field in spec.outputs] == ["summary"]
        SignatureBuilder.build(spec)

    def test_sdk_class_name_fallbacks(self) -> None:
        assert RLMSignatureGenerator._sdk_class_name("Do it.", []) == "TaskSignature"
        assert (
            RLMSignatureGenerator._sdk_class_name(
                "Do it.",
                [
                    FieldSpec(
                        name="task_result",
                        description="The result",
                        field_type=FieldType.OUTPUT,
                    ),
                ],
            )
            == "TaskResult"
        )


class _StubRLM:
    """Stand-in for a dspy.RLM that returns a canned draft or raises."""

    def __init__(self, draft: Any = None, error: Exception | None = None) -> None:
        self.draft = draft
        self.error = error
        self.interpreter: Any = None
        self.kwargs: dict[str, Any] = {}
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        # dspy >=3.3 passes a caller-owned interpreter positionally.
        self.interpreter = args[0] if args else None
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

        spec = generator.forward(prompt, mode="rlm")

        assert stub.kwargs["source_kind"] == "prompt"
        assert spec.name == "ArticleSummarizer"
        assert [field.name for field in spec.inputs] == ["article"]
        SignatureBuilder.build(spec)

    def test_auto_falls_back_to_cot_for_structureless_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.cot = stub  # ty: ignore[invalid-assignment]
        prompt = ParsedPrompt(
            instruction_text="Summarize the article.",
            raw_input="Summarize the article.",
        )

        spec = generator.forward(prompt)

        assert stub.calls == 1
        assert stub.kwargs["source_kind"] == "prompt"
        assert spec.name == "ArticleSummarizer"
        SignatureBuilder.build(spec)

    def test_cot_mode_runs_cot_on_plain_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.cot = stub  # ty: ignore[invalid-assignment]
        prompt = ParsedPrompt(
            instruction_text="Given an article, produce a concise summary.",
            raw_input="Given an article, produce a concise summary.",
        )

        spec = generator.forward(prompt, mode="cot")

        assert stub.calls == 1
        assert stub.kwargs["source_kind"] == "prompt"
        assert spec.name == "ArticleSummarizer"
        SignatureBuilder.build(spec)

    def test_cot_mode_runs_cot_on_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_dataset_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.cot = stub  # ty: ignore[invalid-assignment]
        prompt = ParsedPrompt(
            instruction_text="Predict label", raw_input=[{"message": "x", "label": "y"}]
        )

        spec = generator.forward(prompt, mode="cot")

        assert stub.calls == 1
        assert stub.kwargs["source_kind"] == "dataset"
        assert '"message"' in stub.kwargs["data_profile_json"]
        SignatureBuilder.build(spec)

    def test_cot_failure_falls_back_for_structureless_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        generator.cot = _StubRLM(error=RuntimeError("boom"))  # ty: ignore[invalid-assignment]
        prompt = ParsedPrompt(
            instruction_text="Given an article, produce a concise summary.",
            raw_input="Given an article, produce a concise summary.",
        )

        spec = generator.forward(prompt, mode="cot")

        assert [field.name for field in spec.inputs] == ["article"]
        assert [field.name for field in spec.outputs] == ["summary"]
        SignatureBuilder.build(spec)

    def test_cot_failure_on_dataset_falls_back_to_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_dataset_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        generator.cot = _StubRLM(error=RuntimeError("boom"))  # ty: ignore[invalid-assignment]
        prompt = ParsedPrompt(
            instruction_text="Do the analysis",
            raw_input=[{"message": "urgent", "label": "high"}],
        )

        spec = generator.forward(prompt, mode="cot")

        assert [field.name for field in spec.inputs] == ["message"]
        assert [field.name for field in spec.outputs] == ["label"]
        SignatureBuilder.build(spec)

    def test_fast_mode_plain_prompt_bypasses_rlm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.rlm = stub  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text="Given an article, produce a concise summary.",
            raw_input="Given an article, produce a concise summary.",
        )

        spec = generator.forward(prompt, mode="fast")

        assert stub.calls == 0
        assert [field.name for field in spec.inputs] == ["article"]
        assert [field.name for field in spec.outputs] == ["summary"]
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

    def test_rlm_mode_forces_rlm_on_dataset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_dataset_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.rlm = stub  # type: ignore[assignment]
        prompt = ParsedPrompt(
            instruction_text="Predict label", raw_input=[{"message": "x", "label": "y"}]
        )

        spec = generator.forward(prompt, mode="rlm")

        assert stub.calls == 1
        assert stub.kwargs["source_kind"] == "dataset"
        assert spec.name == "ArticleSummarizer"
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
    def test_forward_sdk_runs_sdk_cot_and_sanitizes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Config, "_lm", dspy.LM("openai/gpt-4o"))
        generator = RLMSignatureGenerator()
        stub = _StubRLM(_COMPLETE_DRAFT)
        generator.cot_sdk = stub  # ty: ignore[invalid-assignment]
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
        generator.cot_sdk = _StubRLM(  # ty: ignore[invalid-assignment]
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


class TestPydanticDraftConversion:
    def test_structured_output_proposal_becomes_typed_signature(self) -> None:
        draft = {
            "name": "ContactExtractor",
            "instructions": "Extract contact records from the message.",
            "inputs": [
                {"name": "message", "description": "The raw message", "type": "string"}
            ],
            "outputs": [
                {
                    "name": "contact",
                    "description": "The extracted contact",
                    "type": "pydantic",
                    "pydantic_model": {
                        "model_name": "ContactRecord",
                        "description": "A contact record",
                        "fields": [
                            {
                                "name": "full name",
                                "type": "str",
                                "description": "Full name",
                            },
                            {
                                "name": "age",
                                "type": "Optional[int]",
                                "description": "Age in years",
                            },
                            {
                                "name": "priority",
                                "type": "literal",
                                "literal_values": ["low", "high"],
                                "description": "Priority",
                            },
                        ],
                    },
                },
                {
                    "name": "summary",
                    "description": "One-line summary",
                    "type": "string",
                },
            ],
        }
        spec = RLMSignatureGenerator._draft_to_spec(draft)
        contact = spec.outputs[0]
        assert contact.model_schema is not None
        assert contact.model_schema.model_name == "ContactRecord"

        signature = SignatureBuilder.build(spec)
        model: Any = signature.output_fields["contact"].annotation
        record = model(full_name="Ada", age=None, priority="high")
        assert record.full_name == "Ada"
        with pytest.raises(Exception):
            model(full_name="Ada", priority="bogus")

        source = signature.to_source()
        assert "class ContactRecord(BaseModel):" in source
        assert "contact: ContactRecord" in source

    def test_literal_values_proposal_resolves_to_literal(self) -> None:
        draft = {
            "name": "UrgencyRouter",
            "instructions": "Route by urgency.",
            "inputs": [
                {"name": "ticket", "description": "The ticket", "type": "string"}
            ],
            "outputs": [
                {
                    "name": "urgency",
                    "description": "Urgency level",
                    "type": "Literal",
                    "literal_values": ["low", "high"],
                }
            ],
        }
        spec = RLMSignatureGenerator._draft_to_spec(draft)
        signature = SignatureBuilder.build(spec)
        annotation = signature.output_fields["urgency"].annotation
        assert getattr(annotation, "__args__", ()) == ("low", "high")

    def test_invalid_model_schema_is_dropped_field_survives(self) -> None:
        draft = {
            "name": "BrokenSchema",
            "instructions": "Produce a record.",
            "inputs": [{"name": "text", "description": "Input text"}],
            "outputs": [
                {
                    "name": "record",
                    "description": "The record",
                    "type": "pydantic",
                    "pydantic_model": {"fields": [{"name": "only", "type": "str"}]},
                },
                {"name": "fallback", "description": "Fallback output"},
            ],
        }
        spec = RLMSignatureGenerator._draft_to_spec(draft)
        assert spec.outputs[0].model_schema is None
        assert spec.outputs[0].suggested_type == "pydantic"
        assert spec.outputs[1].name == "fallback"
        SignatureBuilder.build(spec)

    def test_nested_model_schema_becomes_nested_annotation(self) -> None:
        draft = {
            "name": "OrderReviewer",
            "instructions": "Review the order.",
            "inputs": [{"name": "order", "description": "The order"}],
            "outputs": [
                {
                    "name": "review",
                    "description": "The review",
                    "type": "pydantic",
                    "pydantic_model": {
                        "model_name": "Review",
                        "fields": [
                            {"name": "score", "type": "int", "description": "Score"},
                            {
                                "name": "shipping address",
                                "type": "pydantic",
                                "description": "Shipping",
                                "nested_model": {
                                    "model_name": "ShippingAddress",
                                    "fields": [
                                        {
                                            "name": "city",
                                            "type": "str",
                                            "description": "City",
                                        }
                                    ],
                                },
                            },
                        ],
                    },
                }
            ],
        }
        spec = RLMSignatureGenerator._draft_to_spec(draft)
        signature = SignatureBuilder.build(spec)
        model: Any = signature.output_fields["review"].annotation
        review = model(score=5, shipping_address={"city": "Berlin"})
        assert review.shipping_address.city == "Berlin"
        source = signature.to_source()
        assert "class ShippingAddress(BaseModel):" in source
        assert "shipping_address: ShippingAddress" in source

    def test_proposed_field_defaults_keep_plain_paths_stable(self) -> None:
        field = ProposedField(name="answer", description="The answer")
        assert field.literal_values is None
        assert field.pydantic_model is None

    def test_existing_model_schema_instance_is_passed_through(self) -> None:
        schema = PydanticModelSchema.model_validate(
            {
                "model_name": "ContactRecord",
                "fields": [{"name": "name", "type": "str", "description": "Name"}],
            }
        )
        assert RLMSignatureGenerator._parse_model_schema(schema) is schema

    def test_unparseable_model_schema_payload_is_dropped(self) -> None:
        draft = {
            "name": "JunkSchema",
            "instructions": "Extract things.",
            "inputs": [{"name": "message", "description": "The message"}],
            "outputs": [
                {
                    "name": "result",
                    "description": "The result",
                    "type": "pydantic",
                    "pydantic_model": "not json at all",
                },
                {"name": "fallback", "description": "The fallback"},
            ],
        }
        spec = RLMSignatureGenerator._draft_to_spec(draft)
        assert spec.outputs[0].model_schema is None
        SignatureBuilder.build(spec)
