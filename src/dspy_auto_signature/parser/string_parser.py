"""Parser for raw strings / system prompts."""

from __future__ import annotations

import re
from typing import Any

from dspy_auto_signature.parser.base import PromptParser
from dspy_auto_signature.types.signature_spec import ParsedPrompt


class StringParser(PromptParser):
    """Parse a plain string into a :class:`ParsedPrompt`.

    Also extracts any few-shot examples that look like
    ``Input: ... Output: ...`` blocks.
    """

    def can_parse(self, raw: Any) -> bool:
        return isinstance(raw, str)

    def parse(self, raw: Any) -> ParsedPrompt:
        text = str(raw).strip()
        instruction, examples = self._extract_examples(text)
        return ParsedPrompt(
            instruction_text=instruction,
            examples=examples,
            raw_input=raw,
        )

    @staticmethod
    def _extract_examples(text: str) -> tuple[str, list[dict[str, str]]]:
        """Split out few-shot examples from the instruction text."""
        # Simple heuristic: look for "Input:" / "Output:" pairs
        pattern = re.compile(
            r"Input:\s*(.+?)\s*Output:\s*(.+?)(?=\n\s*Input:|$)",
            re.IGNORECASE | re.DOTALL,
        )
        matches = pattern.findall(text)
        if not matches:
            return text, []

        # Strip examples from instruction
        instruction = pattern.sub("", text).strip()
        examples = [
            {"input": inp.strip(), "output": out.strip()} for inp, out in matches
        ]
        return instruction, examples
