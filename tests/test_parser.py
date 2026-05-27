"""Tests for parser module."""

from __future__ import annotations

import pytest

from dspy_auto_signature.parser import AutoParser
from dspy_auto_signature.parser.base import PromptParser
from dspy_auto_signature.parser.string_parser import StringParser
from dspy_auto_signature.parser.vercel_parser import VercelParser
from dspy_auto_signature.types.signature_spec import ParsedPrompt


class TestStringParser:
    """Tests for StringParser."""

    def test_can_parse_string(self) -> None:
        parser = StringParser()
        assert parser.can_parse("hello") is True
        assert parser.can_parse(123) is False

    def test_parse_simple_string(self) -> None:
        parser = StringParser()
        result = parser.parse("Summarize articles")
        assert isinstance(result, ParsedPrompt)
        assert result.instruction_text == "Summarize articles"
        assert result.examples == []

    def test_parse_with_examples(self) -> None:
        parser = StringParser()
        text = """
        Summarize articles.

        Input: This is a long article about AI.
        Output: AI is transforming industries.

        Input: Another article about space.
        Output: Space exploration advances.
        """
        result = parser.parse(text)
        assert "Summarize articles" in result.instruction_text
        assert len(result.examples) == 2
        assert result.examples[0]["input"] == "This is a long article about AI."
        assert result.examples[0]["output"] == "AI is transforming industries."

    def test_parse_empty_string(self) -> None:
        parser = StringParser()
        result = parser.parse("")
        assert result.instruction_text == ""


class TestVercelParser:
    """Tests for VercelParser."""

    def test_can_parse_vercel_format(self) -> None:
        parser = VercelParser()
        assert parser.can_parse([{"role": "user", "content": "hi"}]) is True
        assert parser.can_parse("hello") is False
        assert parser.can_parse([]) is False  # Empty list should not match
        assert parser.can_parse([{"not_role": "x"}]) is False

    def test_parse_simple_messages(self) -> None:
        parser = VercelParser()
        messages = [
            {"role": "system", "content": "You summarize articles."},
            {"role": "user", "content": "Please summarize this: {article}"},
        ]
        result = parser.parse(messages)
        assert "You summarize articles" in result.instruction_text
        assert "Please summarize this" in result.instruction_text

    def test_parse_with_assistant(self) -> None:
        parser = VercelParser()
        messages = [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "Question: {question}"},
            {"role": "assistant", "content": "Answer: {answer}"},
        ]
        result = parser.parse(messages)
        assert "You are a bot" in result.instruction_text
        # Assistant role should be included as context
        assert "[assistant]" in result.instruction_text


class TestAutoParser:
    """Tests for AutoParser."""

    def setup_method(self) -> None:
        self._original = list(AutoParser._parsers)

    def teardown_method(self) -> None:
        AutoParser._parsers = self._original

    def test_parse_string(self) -> None:
        result = AutoParser.parse("Hello world")
        assert result.instruction_text == "Hello world"

    def test_parse_vercel(self) -> None:
        messages = [
            {"role": "system", "content": "You translate."},
        ]
        result = AutoParser.parse(messages)
        assert "You translate" in result.instruction_text

    def test_parse_invalid_input(self) -> None:
        with pytest.raises(ValueError, match="No parser available"):
            AutoParser.parse(12345)

    def test_register_custom_parser(self) -> None:
        class IntParser(PromptParser):
            def can_parse(self, raw: object) -> bool:
                return isinstance(raw, int)

            def parse(self, raw: object) -> ParsedPrompt:
                return ParsedPrompt(instruction_text=str(raw))

        AutoParser.register(IntParser)
        result = AutoParser.parse(42)
        assert result.instruction_text == "42"
