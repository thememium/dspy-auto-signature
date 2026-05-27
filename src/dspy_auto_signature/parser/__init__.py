"""Orchestrator that tries all registered parsers."""

from __future__ import annotations

from typing import Any

from dspy_auto_signature.parser.base import PromptParser
from dspy_auto_signature.parser.string_parser import StringParser
from dspy_auto_signature.parser.vercel_parser import VercelParser
from dspy_auto_signature.types.signature_spec import ParsedPrompt


class AutoParser:
    """Automatically selects and runs the right parser for the input."""

    _parsers: list[type[PromptParser]] = [
        VercelParser,
        StringParser,
    ]

    @classmethod
    def parse(cls, raw: Any) -> ParsedPrompt:
        """Find a compatible parser and normalise *raw*.

        Parsers are tested in order of registration. The first one whose
        :meth:`~PromptParser.can_parse` returns ``True`` wins.

        Args:
            raw: Heterogeneous prompt input.

        Returns:
            A :class:`ParsedPrompt`.

        Raises:
            ValueError: When no parser can handle the input.

        """
        for parser_cls in cls._parsers:
            parser = parser_cls()
            if parser.can_parse(raw):
                return parser.parse(raw)

        raise ValueError(f"No parser available for input type: {type(raw).__name__}")

    @classmethod
    def register(cls, parser_cls: type[PromptParser], *, prepend: bool = False) -> None:
        """Add a custom parser to the registry.

        Args:
            parser_cls: A concrete :class:`PromptParser` subclass.
            prepend: If ``True``, insert at the front (higher priority).

        """
        if prepend:
            cls._parsers.insert(0, parser_cls)
        else:
            cls._parsers.append(parser_cls)
