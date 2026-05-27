"""Package-level configuration."""

from __future__ import annotations

import logging

import dspy

logger = logging.getLogger(__name__)


class Config:
    """Global package configuration.

    Holds the LM used by the meta-program, default naming conventions,
    and other tunables.
    """

    _lm: dspy.LM | None = None

    @classmethod
    def configure(cls, lm: dspy.LM | None = None) -> None:
        """Set the language model used for signature generation.

        If not explicitly configured, the package will attempt to use
        whatever LM is globally configured via ``dspy.configure()``.

        Args:
            lm: A ``dspy.LM`` instance, or ``None`` to use the global default.

        """
        cls._lm = lm
        if lm is not None:
            logger.info("Configured dspy-auto-signature with LM: %s", lm)

    @classmethod
    def get_lm(cls) -> dspy.LM:
        """Return the configured LM, falling back to the global DSPy settings."""
        if cls._lm is not None:
            return cls._lm

        # DSPy 3.2+ stores config in dspy.settings.config (a dotdict)
        lm = dspy.settings.get("lm")
        if lm is None:
            raise RuntimeError(
                "No language model configured. "
                "Call dspy_auto_signature.configure(lm=...) "
                "or dspy.configure(lm=...) first.",
            )
        return lm  # type: ignore[return-value]

    @classmethod
    def reset(cls) -> None:
        """Clear any explicitly-set LM and revert to global defaults."""
        cls._lm = None
