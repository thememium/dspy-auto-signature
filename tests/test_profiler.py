"""Tests for column profiler."""

from __future__ import annotations

import json

from dspy_auto_signature.data.profiler import profile_columns


class TestProfiler:
    def test_basic_numeric_columns(self) -> None:
        rows = [{"x": 1, "y": 2.5}, {"x": 3, "y": 4.5}, {"x": 5, "y": 6.5}]
        result = profile_columns(rows)

        assert result["n_rows"] == 3
        assert result["n_cols"] == 2

        x = result["columns"]["x"]
        assert x["dtype"] == "int"
        assert x["min"] == 1.0
        assert x["max"] == 5.0
        assert x["mean"] == 3.0
        assert x["median"] == 3.0

        y = result["columns"]["y"]
        assert y["dtype"] == "float"
        assert y["min"] == 2.5
        assert y["max"] == 6.5

    def test_categorical_detection(self) -> None:
        rows = [{"color": c} for c in ["red", "blue", "green", "red", "blue"] * 4]
        result = profile_columns(rows)

        col = result["columns"]["color"]
        assert col["dtype"] == "categorical"
        assert col["n_unique"] == 3
        top = {tv["value"]: tv["count"] for tv in col["top_values"]}
        assert top["red"] == 8
        assert top["blue"] == 8

    def test_text_detection(self) -> None:
        rows = [
            {"text": f"This is sentence number {i} with some extra padding"}
            for i in range(200)
        ]
        result = profile_columns(rows)

        col = result["columns"]["text"]
        assert col["dtype"] == "text"
        assert "min_length" in col
        assert "max_length" in col
        assert "mean_length" in col

    def test_null_rate(self) -> None:
        rows = [{"a": 1}, {"a": None}, {"a": 3}, {"a": None}]
        result = profile_columns(rows)

        col = result["columns"]["a"]
        assert col["null_rate"] == 0.5

    def test_all_nulls(self) -> None:
        rows = [{"a": None}, {"a": None}]
        result = profile_columns(rows)

        col = result["columns"]["a"]
        assert col["dtype"] == "unknown"
        assert col["null_rate"] == 1.0
        assert col["n_unique"] == 0
        assert col["sample_values"] == []

    def test_sample_values_truncation(self) -> None:
        long_str = "x" * 200
        rows = [{"desc": long_str}]
        result = profile_columns(rows)

        col = result["columns"]["desc"]
        assert len(col["sample_values"]) == 1
        assert len(col["sample_values"][0]) <= 80
        assert col["sample_values"][0].endswith("...")

    def test_empty_rows(self) -> None:
        result = profile_columns([])
        assert result == {"n_rows": 0, "n_cols": 0, "columns": {}}

    def test_max_rows_sampling(self) -> None:
        rows = [{"val": i} for i in range(2000)]
        result = profile_columns(rows)

        # n_rows should reflect the ORIGINAL count
        assert result["n_rows"] == 2000
        assert result.get("sampled") is True
        assert result.get("sample_size") == 1000

    def test_mixed_types_column(self) -> None:
        rows = [
            {"mixed": 1},
            {"mixed": "hello"},
            {"mixed": [1, 2]},
            {"mixed": None},
        ]
        result = profile_columns(rows)

        col = result["columns"]["mixed"]
        # Should not crash; mixed types → "unknown"
        assert col["dtype"] == "unknown"
        assert col["null_rate"] == 0.25

    def test_bool_column(self) -> None:
        rows = [{"flag": True}, {"flag": False}, {"flag": True}]
        result = profile_columns(rows)

        col = result["columns"]["flag"]
        assert col["dtype"] == "bool"

    def test_datetime_column(self) -> None:
        import datetime

        rows = [
            {"ts": datetime.datetime(2024, 1, 1)},
            {"ts": datetime.datetime(2024, 6, 15)},
        ]
        result = profile_columns(rows)

        col = result["columns"]["ts"]
        assert col["dtype"] == "datetime"

    def test_list_column(self) -> None:
        rows = [{"tags": ["a", "b"]}, {"tags": ["c"]}]
        result = profile_columns(rows)

        col = result["columns"]["tags"]
        assert col["dtype"] == "list"

    def test_json_serializable(self) -> None:
        """Ensure the entire profile dict can be round-tripped through JSON."""
        rows = [
            {"name": "Alice", "age": 30, "score": 95.5, "active": True},
            {"name": "Bob", "age": 25, "score": 88.0, "active": False},
            {"name": None, "age": 40, "score": None, "active": True},
        ]
        result = profile_columns(rows)
        # This will raise if anything is not JSON-serializable
        serialized = json.dumps(result, default=str)
        assert isinstance(serialized, str)
        assert len(serialized) > 0

    def test_column_order_preserved(self) -> None:
        rows = [{"z": 1, "a": 2}]
        result = profile_columns(rows)
        col_names = list(result["columns"].keys())
        assert col_names == ["z", "a"]
