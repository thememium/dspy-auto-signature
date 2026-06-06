"""Tests for to_records data conversion."""

from __future__ import annotations

import pytest

from dspy_auto_signature.data.to_records import to_records


class TestToRecords:
    def test_list_of_dicts_passthrough(self) -> None:
        result = to_records([{"a": 1}, {"a": 2}])
        assert result == [{"a": 1}, {"a": 2}]

    def test_empty_list(self) -> None:
        assert to_records([]) == []

    def test_single_dict(self) -> None:
        assert to_records({"a": 1}) == [{"a": 1}]

    def test_mixed_list_raises_clear_error(self) -> None:
        with pytest.raises(TypeError, match="only mapping records"):
            to_records([{"a": 1}, "not a record"])

    def test_unsupported_raises(self) -> None:
        with pytest.raises(TypeError, match="Cannot convert.*to list of dicts"):
            to_records(42)

    def test_unsupported_string_raises(self) -> None:
        with pytest.raises(TypeError, match="Cannot convert.*to list of dicts"):
            to_records("hello")

    def test_pandas_dataframe(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = to_records(df)
        assert result == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    def test_pandas_empty_dataframe(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [], "b": []})
        result = to_records(df)
        assert result == []

    def test_dspy_example(self) -> None:
        dspy = pytest.importorskip("dspy")
        ex = dspy.Example(question="What is AI?", answer="Artificial Intelligence")
        result = to_records(ex)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert result[0]["question"] == "What is AI?"
        assert result[0]["answer"] == "Artificial Intelligence"

    def test_list_of_dspy_examples(self) -> None:
        dspy = pytest.importorskip("dspy")
        ex1 = dspy.Example(a=1)
        ex2 = dspy.Example(a=2)
        result = to_records([ex1, ex2])
        assert len(result) == 2
        assert all(isinstance(r, dict) for r in result)
        assert result[0]["a"] == 1
        assert result[1]["a"] == 2

    def test_object_with_to_dicts(self) -> None:
        class FakeFrame:
            def to_dicts(self) -> list[dict[str, int]]:
                return [{"x": 1}, {"x": 2}]

        result = to_records(FakeFrame())
        assert result == [{"x": 1}, {"x": 2}]

    def test_object_with_plain_to_dict(self) -> None:
        class Record:
            def to_dict(self) -> dict[str, int]:
                return {"x": 1}

        assert to_records(Record()) == [{"x": 1}]
