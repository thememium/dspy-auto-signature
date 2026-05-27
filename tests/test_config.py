"""Tests for config module."""

from __future__ import annotations

import dspy
import pytest

from dspy_auto_signature.core.config import Config


class TestConfig:
    """Tests for Config."""

    def setup_method(self) -> None:
        """Reset config before each test."""
        Config.reset()

    def teardown_method(self) -> None:
        """Reset config after each test."""
        Config.reset()

    def test_configure_sets_lm(self) -> None:
        lm = dspy.LM("openai/gpt-4o")
        Config.configure(lm=lm)
        assert Config.get_lm() is lm

    def test_get_lm_without_configure_raises(self) -> None:
        with pytest.raises(RuntimeError, match="No language model configured"):
            Config.get_lm()

    def test_reset_clears_lm(self) -> None:
        lm = dspy.LM("openai/gpt-4o")
        Config.configure(lm=lm)
        Config.reset()
        with pytest.raises(RuntimeError):
            Config.get_lm()

    def test_configure_with_none_uses_global(self) -> None:
        """Configuring with None defers to global DSPy settings."""
        Config.configure(lm=None)
        # Should raise because no global LM is configured
        with pytest.raises(RuntimeError):
            Config.get_lm()
