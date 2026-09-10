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

    def test_configure_sets_dataset_lm(self) -> None:
        dataset_lm = dspy.LM("openai/gpt-4o-mini")
        Config.configure(dataset_lm=dataset_lm)
        assert Config.get_dataset_lm() is dataset_lm

    def test_configure_sets_sub_lm(self) -> None:
        sub_lm = dspy.LM("openai/gpt-4o-mini")
        Config.configure(sub_lm=sub_lm)
        assert Config.get_sub_lm() is sub_lm

    def test_get_lm_falls_back_to_global_dspy_lm(self) -> None:
        """Without an explicit LM, get_lm uses the globally configured DSPy LM."""
        lm = dspy.LM("openai/gpt-4o")
        dspy.configure(lm=lm)
        try:
            assert Config.get_lm() is lm
        finally:
            dspy.configure(lm=None)

    def test_get_dataset_lm_falls_back_to_lm(self) -> None:
        """get_dataset_lm falls back to the explicitly configured lm."""
        lm = dspy.LM("openai/gpt-4o")
        Config.configure(lm=lm)
        assert Config.get_dataset_lm() is lm

    def test_get_dataset_lm_falls_back_to_global(self) -> None:
        """get_dataset_lm falls back through get_lm to the global DSPy LM."""
        lm = dspy.LM("openai/gpt-4o")
        dspy.configure(lm=lm)
        try:
            assert Config.get_dataset_lm() is lm
        finally:
            dspy.configure(lm=None)

    def test_get_sub_lm_falls_back_to_lm(self) -> None:
        """get_sub_lm falls back to the explicitly configured lm."""
        lm = dspy.LM("openai/gpt-4o")
        Config.configure(lm=lm)
        assert Config.get_sub_lm() is lm

    def test_get_sub_lm_falls_back_to_global(self) -> None:
        """get_sub_lm falls back through get_lm to the global DSPy LM."""
        lm = dspy.LM("openai/gpt-4o")
        dspy.configure(lm=lm)
        try:
            assert Config.get_sub_lm() is lm
        finally:
            dspy.configure(lm=None)
