"""Tests for DataFrameParser."""

from __future__ import annotations

import sys
from typing import Any

import pandas as pd
import pytest

from dspy_auto_signature.parser.dataframe_parser import DataFrameParser


class _ToDictsStub:
    def to_dicts(self) -> list[dict[str, Any]]:
        return [{"a": 1}]


class _ToPandasStub:
    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame({"a": [1]})


class _ToDictStub:
    def to_dict(self) -> dict[str, Any]:
        return {"a": 1}


class Example:
    """Bears the same type name as dspy.Example for duck-typed detection."""

    def __init__(self, question: str = "What?") -> None:
        self.question = question


class TestDataFrameParserCanParse:
    def test_can_parse_plain_dict(self) -> None:
        assert DataFrameParser().can_parse({"a": [1, 2]}) is True

    def test_can_parse_pandas_dataframe(self) -> None:
        df = pd.DataFrame({"question": ["What is AI?"], "answer": ["Smart robots"]})
        assert DataFrameParser().can_parse(df) is True

    def test_can_parse_survives_missing_pandas(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "pandas", None)
        parser = DataFrameParser()
        assert parser.can_parse([{"a": 1}]) is True

    def test_can_parse_to_dicts_object(self) -> None:
        assert DataFrameParser().can_parse(_ToDictsStub()) is True

    def test_can_parse_to_pandas_object(self) -> None:
        assert DataFrameParser().can_parse(_ToPandasStub()) is True

    def test_can_parse_to_dict_object(self) -> None:
        assert DataFrameParser().can_parse(_ToDictStub()) is True

    def test_cannot_parse_empty_list(self) -> None:
        assert DataFrameParser().can_parse([]) is False

    def test_can_parse_list_of_examples(self) -> None:
        examples = [Example("What is AI?"), Example("What is ML?")]
        assert DataFrameParser().can_parse(examples) is True

    def test_can_parse_single_example(self) -> None:
        assert DataFrameParser().can_parse(Example("What is AI?")) is True

    def test_cannot_parse_plain_string(self) -> None:
        assert DataFrameParser().can_parse("just text") is False


class TestDataFrameParserParse:
    def test_parse_plain_dict(self) -> None:
        result = DataFrameParser().parse(
            {"question": ["What is AI?"], "answer": ["Robots"]}
        )
        assert result.source_kind == "dataset"
        assert result.examples[0] == {
            "question": "['What is AI?']",
            "answer": "['Robots']",
        }
        assert result.data_profile is not None
        assert result.data_profile["n_cols"] == 2

    def test_parse_truncates_rows_to_five_columns(self) -> None:
        row = {f"col{i}": i for i in range(7)}
        result = DataFrameParser().parse([row])
        assert list(result.examples[0].keys()) == [f"col{i}" for i in range(5)]

    def test_parse_renders_none_as_null(self) -> None:
        result = DataFrameParser().parse([{"a": None, "b": 1}])
        assert result.examples[0]["a"] == "null"

    def test_instruction_lists_column_profiles(self) -> None:
        result = DataFrameParser().parse(
            {"question": ["What is AI?"], "answer": ["Robots"]}
        )
        assert "Dataset with 1 rows and 2 columns." in result.instruction_text
        assert "Column profiles:" in result.instruction_text
        assert "question" in result.instruction_text
