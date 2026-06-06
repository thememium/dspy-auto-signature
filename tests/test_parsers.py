"""Tests for the parser layer."""

from __future__ import annotations

import pytest

from dspy_auto_signature.parser import AutoParser
from dspy_auto_signature.parser.base import PromptParser
from dspy_auto_signature.parser.sdk_parser import SDKParser
from dspy_auto_signature.parser.string_parser import StringParser
from dspy_auto_signature.types.signature_spec import ParsedPrompt


class TestStringParser:
    def test_can_parse_string(self) -> None:
        parser = StringParser()
        assert parser.can_parse("hello") is True
        assert parser.can_parse(123) is False

    def test_simple_string(self) -> None:
        parser = StringParser()
        result = parser.parse("Summarize articles")
        assert isinstance(result, ParsedPrompt)
        assert result.instruction_text == "Summarize articles"
        assert result.examples == []

    def test_extracts_examples(self) -> None:
        text = """
        Summarize articles. Here are examples:
        Input: Long article about AI
        Output: AI is transforming technology
        Input: Article about climate
        Output: Climate change is accelerating
        """
        parser = StringParser()
        result = parser.parse(text)
        assert len(result.examples) == 2
        assert result.examples[0]["input"] == "Long article about AI"
        assert result.examples[0]["output"] == "AI is transforming technology"


class TestSDKParser:
    def test_can_parse_sdk_format(self) -> None:
        parser = SDKParser()
        assert parser.can_parse([{"role": "system", "content": "hi"}]) is True
        assert parser.can_parse("not a list") is False
        assert parser.can_parse([{"no_role": "x"}]) is False

    def test_system_becomes_instruction_user_assistant_become_examples(self) -> None:
        messages = [
            {"role": "system", "content": "You are a summarizer."},
            {"role": "user", "content": "Summarize this article."},
            {"role": "assistant", "content": "Here is the summary."},
        ]
        parser = SDKParser()
        result = parser.parse(messages)
        assert result.instruction_text == "You are a summarizer."
        assert len(result.examples) == 1
        assert result.examples[0]["input"] == "Summarize this article."
        assert result.examples[0]["output"] == "Here is the summary."

    def test_multi_turn_conversation(self) -> None:
        messages = [
            {"role": "system", "content": "You translate."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hola"},
            {"role": "user", "content": "Goodbye"},
            {"role": "assistant", "content": "Adiós"},
        ]
        parser = SDKParser()
        result = parser.parse(messages)
        assert result.instruction_text == "You translate."
        assert len(result.examples) == 2
        assert result.examples[1] == {"input": "Goodbye", "output": "Adiós"}


class TestAutoParser:
    def setup_method(self) -> None:
        self._original_parsers = list(AutoParser._parsers)

    def teardown_method(self) -> None:
        AutoParser._parsers = self._original_parsers

    def test_auto_selects_string_parser(self) -> None:
        result = AutoParser.parse("Just a string")
        assert isinstance(result, ParsedPrompt)
        assert result.instruction_text == "Just a string"

    def test_auto_selects_sdk_parser(self) -> None:
        result = AutoParser.parse([{"role": "system", "content": "test"}])
        assert isinstance(result, ParsedPrompt)
        assert "test" in result.instruction_text

    def test_raises_on_unsupported(self) -> None:
        with pytest.raises(ValueError, match="No parser available"):
            AutoParser.parse(12345)  # type: ignore[arg-type]

    def test_custom_parser_registration(self) -> None:
        class IntParser(PromptParser):
            def can_parse(self, raw: object) -> bool:
                return isinstance(raw, int)

            def parse(self, raw: object) -> ParsedPrompt:
                return ParsedPrompt(
                    instruction_text=str(raw), examples=[], raw_input=raw
                )

        AutoParser.register(IntParser, prepend=True)
        result = AutoParser.parse(42)
        assert result.instruction_text == "42"
