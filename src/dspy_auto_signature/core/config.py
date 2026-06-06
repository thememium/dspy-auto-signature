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
    _dataset_lm: dspy.LM | None = None
    _sub_lm: dspy.LM | None = None

    @classmethod
    def configure(
        cls,
        lm: dspy.LM | None = None,
        dataset_lm: dspy.LM | None = None,
        sub_lm: dspy.LM | None = None,
    ) -> None:
        """Set the language models used for signature generation.

        If not explicitly configured, the package will attempt to use
        whatever LM is globally configured via ``dspy.configure()``.

        Args:
            lm: A ``dspy.LM`` instance, or ``None`` to use the global default.
                Used for prompt context and as the fallback for ``dataset_lm``
                and ``sub_lm``.
            dataset_lm: Optional outer LM used when the unified RLM receives
                dataset context. Falls back to ``lm`` if unset.
            sub_lm: The cheap inner LM used by RLM for sub-queries. Falls
                back to ``lm`` if unset.

        """
        if lm is not None:
            cls._lm = lm
            logger.info("Configured dspy-auto-signature with LM: %s", lm)
        if dataset_lm is not None:
            cls._dataset_lm = dataset_lm
            logger.info("Configured dspy-auto-signature dataset_lm: %s", dataset_lm)
        if sub_lm is not None:
            cls._sub_lm = sub_lm
            logger.info("Configured dspy-auto-signature sub_lm: %s", sub_lm)

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
    def get_dataset_lm(cls) -> dspy.LM:
        """Return the outer LM for dataset context, falling back to ``get_lm()``."""
        if cls._dataset_lm is not None:
            return cls._dataset_lm
        return cls.get_lm()

    @classmethod
    def get_sub_lm(cls) -> dspy.LM:
        """Return the cheap inner LM for RLM sub-queries, falling back to ``get_lm()``."""
        if cls._sub_lm is not None:
            return cls._sub_lm
        return cls.get_lm()

    @classmethod
    def reset(cls) -> None:
        """Clear any explicitly-set LMs and revert to global defaults."""
        cls._lm = None
        cls._dataset_lm = None
        cls._sub_lm = None
