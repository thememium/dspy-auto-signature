"""Tests for the public API (__init__.py)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import dspy
import pytest

import dspy_auto_signature as das
from dspy_auto_signature.core.config import Config
from dspy_auto_signature.types.signature_spec import (
    FieldSpec,
    FieldType,
    ParsedPrompt,
    SignatureSpec,
)


@pytest.fixture(autouse=True)
def _isolated_config() -> Iterator[None]:
    """Keep package-level Config state out of the global test process."""
    Config.reset()
    yield
    Config.reset()


def _summarizer_spec() -> SignatureSpec:
    """A minimal spec a mocked generator could plausibly produce."""
    return SignatureSpec(
        name="Summarizer",
        instructions="Summarize the article.",
        inputs=[
            FieldSpec(
                name="article",
                description="The article text",
                field_type=FieldType.INPUT,
            )
        ],
        outputs=[
            FieldSpec(
                name="summary",
                description="The summary",
                field_type=FieldType.OUTPUT,
            )
        ],
    )


class _FakeGenerator:
    """Stands in for ``RLMSignatureGenerator`` so no LLM is contacted."""

    def __init__(self, sub_lm: dspy.LM | None = None) -> None:
        self.sub_lm = sub_lm
        self.parsed: ParsedPrompt | None = None

    def __call__(
        self,
        parsed: ParsedPrompt,
        *,
        mode: str = "auto",
    ) -> SignatureSpec:
        self.parsed = parsed
        self.mode = mode
        return _summarizer_spec()


class TestPublicAPI:
    """Tests for the public API surface."""

    def test_imports(self) -> None:
        """Verify all public symbols are importable."""
        assert hasattr(das, "from_prompt")
        assert hasattr(das, "from_dataset")
        assert hasattr(das, "generate")
        assert hasattr(das, "configure")
        assert hasattr(das, "close_interpreter")
        assert hasattr(das, "SignatureSpec")

    def test_configure_sets_lm(self) -> None:
        lm = dspy.LM("openai/gpt-4o")
        das.configure(lm=lm)
        # Should not raise
        assert Config.get_lm() is lm

    def test_from_dataset_rejects_prompt_input(self) -> None:
        with pytest.raises(TypeError, match=r"Use generate\(\)"):
            das.from_dataset("Summarize this")

    def test_generate_rejects_unknown_mode(self) -> None:
        bad_mode: Any = "turbo"
        with pytest.raises(ValueError, match="Unknown mode"):
            das.generate("Summarize this", mode=bad_mode)


class TestGeneratePipeline:
    """End-to-end ``generate`` flow with the RLM generator mocked out."""

    def test_from_prompt_full_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lm = dspy.LM("openai/gpt-4o")
        das.configure(lm=lm)
        created: list[_FakeGenerator] = []
        monkeypatch.setattr(
            das,
            "RLMSignatureGenerator",
            lambda sub_lm=None: created.append(_FakeGenerator(sub_lm)) or created[-1],
        )

        sig = das.from_prompt(
            "Summarize articles",
            input_hints={"article": "The raw article"},
        )

        assert len(created) == 1
        assert created[0].sub_lm is lm
        assert created[0].parsed is not None
        assert "Summarize articles" in created[0].parsed.instruction_text
        assert issubclass(cast("type", sig), dspy.Signature)
        assert "article" in sig.input_fields
        assert "summary" in sig.output_fields

    def test_generate_appends_task_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        das.configure(lm=dspy.LM("openai/gpt-4o"))
        gen = _FakeGenerator()
        monkeypatch.setattr(das, "RLMSignatureGenerator", lambda sub_lm=None: gen)

        das.generate("Infer fields", task_hint="Predict urgency")

        assert gen.parsed is not None
        assert gen.parsed.instruction_text.endswith("\n\nTask: Predict urgency")

    def test_from_dataset_full_pipeline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        das.configure(lm=dspy.LM("openai/gpt-4o"))
        gen = _FakeGenerator()
        monkeypatch.setattr(das, "RLMSignatureGenerator", lambda sub_lm=None: gen)

        sig = das.from_dataset(
            [{"message": "server on fire", "urgency": "high"}],
            task_hint="Classify support tickets",
        )

        assert gen.parsed is not None
        assert gen.parsed.source_kind == "dataset"
        assert gen.parsed.instruction_text.endswith(
            "\n\nTask: Classify support tickets"
        )
        assert issubclass(cast("type", sig), dspy.Signature)

    def test_generate_forwards_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        das.configure(lm=dspy.LM("openai/gpt-4o"))
        gen = _FakeGenerator()
        monkeypatch.setattr(das, "RLMSignatureGenerator", lambda sub_lm=None: gen)

        das.generate("Summarize this", mode="fast")
        assert gen.mode == "fast"

        das.generate("Summarize this", mode="rlm")
        assert gen.mode == "rlm"

        das.generate("Summarize this")
        assert gen.mode == "auto"


class TestApplyHints:
    """Hint merging into a ``SignatureSpec`` copy."""

    def test_updates_existing_field_description(self) -> None:
        spec = _summarizer_spec()

        merged = das._apply_hints(spec, {"article": "  The raw article text  "}, None)

        assert merged.inputs[0].description == "The raw article text"

    def test_appends_new_input_and_output_fields(self) -> None:
        spec = _summarizer_spec()

        merged = das._apply_hints(
            spec, {"topic": "What the article is about"}, {"tags": "Topic tags"}
        )

        assert [f.name for f in merged.inputs] == ["article", "topic"]
        assert merged.inputs[1].field_type is FieldType.INPUT
        assert [f.name for f in merged.outputs] == ["summary", "tags"]
        assert merged.outputs[1].field_type is FieldType.OUTPUT

    def test_merged_spec_remains_buildable(self) -> None:
        from dspy_auto_signature.core.signature_builder import SignatureBuilder

        merged = das._apply_hints(
            _summarizer_spec(), {"topic": "Topic"}, {"tags": "Tags"}
        )

        sig = SignatureBuilder.build(merged)
        assert list(sig.input_fields) == ["article", "topic"]

    def test_returns_a_copy_and_does_not_mutate_original(self) -> None:
        spec = _summarizer_spec()

        merged = das._apply_hints(spec, {"topic": "Topic"}, None)

        assert len(spec.inputs) == 1
        assert len(merged.inputs) == 2
        assert spec.inputs[0].description == "The article text"

    def test_rejects_non_identifier_name(self) -> None:
        with pytest.raises(ValueError, match=r"valid identifier"):
            das._apply_hints(_summarizer_spec(), {"not a name!": "x"}, None)

    def test_rejects_keyword_name(self) -> None:
        with pytest.raises(ValueError, match=r"valid identifier"):
            das._apply_hints(_summarizer_spec(), {"class": "x"}, None)

    def test_rejects_blank_description(self) -> None:
        with pytest.raises(ValueError, match=r"cannot be empty"):
            das._apply_hints(_summarizer_spec(), {"article": "   "}, None)

    def test_rejects_conflict_with_other_field(self) -> None:
        with pytest.raises(ValueError, match=r"conflicts with another field"):
            das._apply_hints(_summarizer_spec(), {"summary": "x"}, None)
