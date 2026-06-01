"""Tests for the RLM-based signature generator (no LLM calls)."""

from __future__ import annotations

import json

from dspy_auto_signature.generator.rlm_signature_generator import RLMSignatureGenerator
from dspy_auto_signature.generator.rlm_signatures import ProposeSignatureFromData


class TestRLMSignatureGenerator:
    """Tests for RLMSignatureGenerator helper methods."""

    def test_parse_spec_json_valid(self) -> None:
        """A well-formed JSON string is parsed into a dict."""
        payload = json.dumps(
            {
                "name": "TicketClassifier",
                "instructions": "Classify support tickets.",
                "inputs": [
                    {
                        "name": "message",
                        "description": "The support ticket text",
                        "type": "string",
                    }
                ],
                "outputs": [
                    {
                        "name": "urgency",
                        "description": "Urgency level",
                        "type": "string",
                    }
                ],
            }
        )
        result = RLMSignatureGenerator._parse_spec_json(payload)
        assert result is not None
        assert result["name"] == "TicketClassifier"
        assert len(result["inputs"]) == 1
        assert len(result["outputs"]) == 1

    def test_parse_spec_json_invalid_returns_none(self) -> None:
        """Garbage input returns None."""
        assert RLMSignatureGenerator._parse_spec_json("not json at all") is None

    def test_parse_spec_json_embedded_in_text(self) -> None:
        """JSON embedded in surrounding text is extracted via regex."""
        payload = json.dumps(
            {
                "name": "Foo",
                "instructions": "Do foo.",
                "inputs": [{"name": "x", "description": "input x", "type": "string"}],
                "outputs": [{"name": "y", "description": "output y", "type": "string"}],
            }
        )
        wrapped = f"Here is the result:\n{payload}\nDone."
        result = RLMSignatureGenerator._parse_spec_json(wrapped)
        assert result is not None
        assert result["name"] == "Foo"

    def test_parse_spec_json_missing_keys(self) -> None:
        """JSON missing required keys returns None."""
        payload = json.dumps({"name": "Foo", "instructions": "bar"})
        assert RLMSignatureGenerator._parse_spec_json(payload) is None

    def test_parse_spec_json_inputs_not_list(self) -> None:
        """JSON where inputs is not a list returns None."""
        payload = json.dumps(
            {
                "name": "Foo",
                "instructions": "bar",
                "inputs": "not a list",
                "outputs": [],
            }
        )
        assert RLMSignatureGenerator._parse_spec_json(payload) is None

    def test_parse_spec_json_item_missing_fields(self) -> None:
        """JSON where an input item is missing 'type' returns None."""
        payload = json.dumps(
            {
                "name": "Foo",
                "instructions": "bar",
                "inputs": [{"name": "x", "description": "desc"}],  # missing type
                "outputs": [],
            }
        )
        assert RLMSignatureGenerator._parse_spec_json(payload) is None

    def test_init_accepts_kwargs(self) -> None:
        """Constructor does not raise with valid keyword arguments."""
        gen = RLMSignatureGenerator(
            max_iterations=5, max_llm_calls=10, sub_lm=None, verbose=False
        )
        assert gen is not None


class TestProposeSignatureFromData:
    """Tests for the ProposeSignatureFromData DSPy Signature."""

    def test_signature_has_required_fields(self) -> None:
        """The signature must declare all expected input and output fields."""
        sig = ProposeSignatureFromData
        # Input fields
        assert "data_profile_json" in sig.input_fields
        assert "sample_rows_json" in sig.input_fields
        assert "task_hint" in sig.input_fields
        # Output fields
        assert "spec_json" in sig.output_fields

    def test_signature_docstring_present(self) -> None:
        """The signature class must have a meaningful docstring."""
        assert ProposeSignatureFromData.__doc__ is not None
        assert "column profiles" in ProposeSignatureFromData.__doc__
