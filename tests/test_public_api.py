"""Tests for the public API (__init__.py)."""

from __future__ import annotations

import dspy

import dspy_auto_signature as das


class TestPublicAPI:
    """Tests for the public API surface."""

    def test_imports(self) -> None:
        """Verify all public symbols are importable."""
        assert hasattr(das, "from_prompt")
        assert hasattr(das, "configure")
        assert hasattr(das, "SignatureSpec")

    def test_configure_sets_lm(self) -> None:
        lm = dspy.LM("openai/gpt-4o")
        das.configure(lm=lm)
        # Should not raise
        from dspy_auto_signature.core.config import Config

        assert Config.get_lm() is lm
