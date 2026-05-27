"""Parser for Vercel AI SDK message arrays."""

from __future__ import annotations

from typing import Any

from dspy_auto_signature.parser.base import PromptParser
from dspy_auto_signature.types.signature_spec import ParsedPrompt


class VercelParser(PromptParser):
    """Parse Vercel AI SDK ``messages`` arrays.

    Expected format::

        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Summarize: {article}"},
        ]

    The system message becomes the instruction; user messages with
    template variables like ``{var}`` are scanned for input field hints.
    """

    def can_parse(self, raw: Any) -> bool:
        return (
            isinstance(raw, list)
            and len(raw) > 0
            and all(isinstance(m, dict) and "role" in m and "content" in m for m in raw)
        )

    def parse(self, raw: Any) -> ParsedPrompt:
        messages: list[dict[str, str]] = raw  # type: ignore[assignment]

        system_parts: list[str] = []
        user_parts: list[str] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                user_parts.append(content)
            else:
                # assistant / tool / function — treat as examples
                system_parts.append(f"[{role}]: {content}")

        instruction = "\n\n".join(filter(None, system_parts))
        if user_parts:
            instruction += f"\n\nUser context: {' '.join(user_parts)}"

        return ParsedPrompt(
            instruction_text=instruction.strip(),
            examples=[],
            raw_input=raw,
        )
