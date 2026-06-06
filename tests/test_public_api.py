"""Tests for the public API (__init__.py)."""

from __future__ import annotations

import dspy
import pytest

import dspy_auto_signature as das


class TestPublicAPI:
    """Tests for the public API surface."""

    def test_imports(self) -> None:
        """Verify all public symbols are importable."""
        assert hasattr(das, "from_prompt")
        assert hasattr(das, "from_dataset")
        assert hasattr(das, "generate")
        assert hasattr(das, "configure")
        assert hasattr(das, "SignatureSpec")

    def test_configure_sets_lm(self) -> None:
        lm = dspy.LM("openai/gpt-4o")
        das.configure(lm=lm)
        # Should not raise
        from dspy_auto_signature.core.config import Config

        assert Config.get_lm() is lm

    def test_from_dataset_rejects_prompt_input(self) -> None:
        with pytest.raises(TypeError, match=r"Use generate\(\)"):
            das.from_dataset("Summarize this")
