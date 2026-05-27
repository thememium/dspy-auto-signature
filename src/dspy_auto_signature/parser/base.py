"""Abstract base for prompt parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from dspy_auto_signature.types.signature_spec import ParsedPrompt


class PromptParser(ABC):
    """Base class for all prompt parsers.

    Each parser implementation normalises a specific input format into a
    :class:`~dspy_auto_signature.types.signature_spec.ParsedPrompt`.
    """

    @abstractmethod
    def parse(self, raw: Any) -> ParsedPrompt:
        """Convert *raw* into a :class:`ParsedPrompt`.

        Args:
            raw: The raw input in the parser's expected format.

        Returns:
            A normalised :class:`ParsedPrompt`.

        """
        ...

    @abstractmethod
    def can_parse(self, raw: Any) -> bool:
        """Return ``True`` if this parser can handle *raw*."""
        ...
