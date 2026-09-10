"""Tests for parser module."""

from __future__ import annotations

import pytest

from dspy_auto_signature.parser import AutoParser
from dspy_auto_signature.parser.base import PromptParser
from dspy_auto_signature.parser.sdk_parser import SDKParser
from dspy_auto_signature.parser.string_parser import StringParser
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


class TestSDKParser:
    """Tests for SDKParser."""

    def test_can_parse_sdk_format(self) -> None:
        parser = SDKParser()
        assert parser.can_parse([{"role": "user", "content": "hi"}]) is True
        assert parser.can_parse("hello") is False
        assert parser.can_parse([]) is False
        assert parser.can_parse([{"not_role": "x"}]) is False

    def test_system_becomes_instruction(self) -> None:
        parser = SDKParser()
        messages = [
            {"role": "system", "content": "You summarize articles."},
            {"role": "user", "content": "Summarize this article about AI."},
            {"role": "assistant", "content": "AI is transforming industries."},
        ]
        result = parser.parse(messages)
        assert result.instruction_text == "You summarize articles."
        assert len(result.examples) == 1
        assert result.examples[0]["input"] == "Summarize this article about AI."
        assert result.examples[0]["output"] == "AI is transforming industries."

    def test_user_assistant_pairs_become_examples(self) -> None:
        parser = SDKParser()
        messages = [
            {"role": "system", "content": "You are a translator."},
            {"role": "user", "content": "Translate to French: Hello"},
            {"role": "assistant", "content": "Bonjour"},
            {"role": "user", "content": "Translate to French: Goodbye"},
            {"role": "assistant", "content": "Au revoir"},
        ]
        result = parser.parse(messages)
        assert result.instruction_text == "You are a translator."
        assert len(result.examples) == 2
        assert result.examples[0] == {
            "input": "Translate to French: Hello",
            "output": "Bonjour",
        }
        assert result.examples[1] == {
            "input": "Translate to French: Goodbye",
            "output": "Au revoir",
        }

    def test_google_gemini_parts_format(self) -> None:
        parser = SDKParser()
        messages = [
            {"role": "user", "parts": [{"text": "Extract entities."}]},
            {"role": "model", "parts": [{"text": "I'll identify key entities."}]},
        ]
        result = parser.parse(messages)
        assert result.instruction_text == ""
        assert len(result.examples) == 1
        assert result.examples[0]["input"] == "Extract entities."
        assert result.examples[0]["output"] == "I'll identify key entities."

    def test_anthropic_content_blocks(self) -> None:
        parser = SDKParser()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Analyze this."}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Analysis complete."}],
            },
        ]
        result = parser.parse(messages)
        assert len(result.examples) == 1
        assert result.examples[0]["input"] == "Analyze this."
        assert result.examples[0]["output"] == "Analysis complete."

    def test_tool_role_appended_to_instruction(self) -> None:
        parser = SDKParser()
        messages = [
            {"role": "system", "content": "You use tools."},
            {"role": "tool", "content": "Result: 42"},
        ]
        result = parser.parse(messages)
        assert "You use tools." in result.instruction_text
        assert "[tool]: Result: 42" in result.instruction_text

    def test_consecutive_user_messages_flush_pending_input(self) -> None:
        parser = SDKParser()
        messages = [
            {"role": "user", "content": "First question"},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Answer"},
        ]
        result = parser.parse(messages)
        assert result.examples == [
            {"input": "First question", "output": ""},
            {"input": "Second question", "output": "Answer"},
        ]

    def test_assistant_without_preceding_user(self) -> None:
        parser = SDKParser()
        result = parser.parse([{"role": "assistant", "content": "Orphan output"}])
        assert result.examples == [{"input": "", "output": "Orphan output"}]

    def test_trailing_user_message_flushed_with_empty_output(self) -> None:
        parser = SDKParser()
        result = parser.parse([{"role": "user", "content": "Unanswered question"}])
        assert result.examples == [{"input": "Unanswered question", "output": ""}]

    def test_get_content_returns_none_for_unrecognized_payloads(self) -> None:
        assert SDKParser._get_content({"role": "user", "content": 42}) is None
        assert (
            SDKParser._get_content({"role": "user", "content": [{"type": "image"}]})
            is None
        )
        assert SDKParser._get_content({"role": "user", "parts": ["not a dict"]}) is None
        assert SDKParser().can_parse([{"role": "user", "content": 42}]) is False


class TestAutoParser:
    """Tests for AutoParser."""

    def setup_method(self) -> None:
        self._original = list(AutoParser._parsers)

    def teardown_method(self) -> None:
        AutoParser._parsers = self._original

    def test_parse_string(self) -> None:
        result = AutoParser.parse("Hello world")
        assert result.instruction_text == "Hello world"

    def test_parse_sdk(self) -> None:
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
