"""Parser for chat message arrays from OpenAI, Anthropic, Google, and similar SDKs."""

from __future__ import annotations

from typing import Any

from dspy_auto_signature.parser.base import PromptParser
from dspy_auto_signature.types.signature_spec import ParsedPrompt


class SDKParser(PromptParser):
    """Parse chat message arrays into instruction text and examples.

    Maps SDK roles to signature semantics:

    - ``system`` / ``developer`` → instruction text (signature docstring)
    - ``user`` → example inputs
    - ``assistant`` / ``model`` → example outputs
    - ``tool`` → appended to instruction as context

    Handles OpenAI, Anthropic, LiteLLM, Google Gemini, and Vercel AI SDK formats.
    """

    def can_parse(self, raw: Any) -> bool:
        return (
            isinstance(raw, list)
            and len(raw) > 0
            and all(isinstance(m, dict) and "role" in m for m in raw)
            and any(self._get_content(m) is not None for m in raw)
        )

    def parse(self, raw: Any) -> ParsedPrompt:
        messages: list[dict[str, Any]] = raw  # type: ignore[assignment]

        instruction_parts: list[str] = []
        examples: list[dict[str, str]] = []
        pending_input: str | None = None

        for msg in messages:
            role = msg.get("role", "")
            content = self._get_content(msg) or ""

            if role in ("system", "developer"):
                instruction_parts.append(content)
            elif role == "user":
                # If there's a pending input without an output, flush it
                if pending_input is not None:
                    examples.append({"input": pending_input, "output": ""})
                pending_input = content
            elif role in ("assistant", "model"):
                if pending_input is not None:
                    examples.append({"input": pending_input, "output": content})
                    pending_input = None
                else:
                    # Assistant message without a preceding user message
                    examples.append({"input": "", "output": content})
            else:
                # tool / function / other — append as context to instruction
                instruction_parts.append(f"[{role}]: {content}")

        # Flush any remaining pending input
        if pending_input is not None:
            examples.append({"input": pending_input, "output": ""})

        instruction = "\n\n".join(filter(None, instruction_parts))

        return ParsedPrompt(
            instruction_text=instruction.strip(),
            examples=examples,
            raw_input=raw,
        )

    @staticmethod
    def _get_content(msg: dict[str, Any]) -> str | None:
        """Extract text content from a message, handling various SDK formats."""
        # Standard: {"content": "text"}
        content = msg.get("content")
        if isinstance(content, str):
            return content

        # Google Gemini: {"parts": [{"text": "..."}]}
        parts = msg.get("parts")
        if isinstance(parts, list):
            texts = [
                p.get("text", "") for p in parts if isinstance(p, dict) and "text" in p
            ]
            if texts:
                return "\n".join(texts)

        # Anthropic content blocks: {"content": [{"type": "text", "text": "..."}]}
        if isinstance(content, list):
            texts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if texts:
                return "\n".join(texts)

        return None
