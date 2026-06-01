"""Column profiling for tabular data."""

from __future__ import annotations

import datetime
import logging
import random
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

MAX_PROFILE_CELLS = 1000


def profile_columns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Profile columns of a list-of-dicts dataset.

    Returns a JSON-serializable dict describing each column's dtype,
    null rate, unique count, sample values, and dtype-specific stats.

    Args:
        rows: A list of dicts, one per row (output of :func:`to_records`).

    Returns:
        A dict with keys ``n_rows``, ``n_cols``, ``columns``.

    """
    if not rows:
        return {"n_rows": 0, "n_cols": 0, "columns": {}}

    original_n = len(rows)
    sampled = False
    if original_n > MAX_PROFILE_CELLS:
        rows = random.Random(42).sample(rows, MAX_PROFILE_CELLS)
        sampled = True

    # Collect all column names preserving first-seen order.
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    columns: dict[str, Any] = {}
    for col in all_keys:
        columns[col] = _profile_one_column(col, rows)

    result: dict[str, Any] = {
        "n_rows": original_n,
        "n_cols": len(all_keys),
        "columns": columns,
    }
    if sampled:
        result["sampled"] = True
        result["sample_size"] = len(rows)
    return result


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _profile_one_column(col: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Profile a single column across all rows."""
    values: list[Any] = [row.get(col) for row in rows]
    non_null = [v for v in values if v is not None]
    n = len(values)
    null_count = n - len(non_null)
    null_rate = round(null_count / n, 4) if n else 0.0

    if not non_null:
        return {
            "dtype": "unknown",
            "null_rate": null_rate,
            "n_unique": 0,
            "sample_values": [],
        }

    dtype = _classify_dtype(non_null, n)
    n_unique = len({repr(v) for v in non_null})
    sample_values = [_safe_repr(v) for v in non_null[:5]]

    info: dict[str, Any] = {
        "dtype": dtype,
        "null_rate": null_rate,
        "n_unique": n_unique,
        "sample_values": sample_values,
    }

    if dtype in ("int", "float"):
        nums = [float(v) for v in non_null if isinstance(v, (int, float, bool))]
        if nums:
            info["min"] = min(nums)
            info["max"] = max(nums)
            info["mean"] = round(sum(nums) / len(nums), 4)
            nums_sorted = sorted(nums)
            mid = len(nums_sorted) // 2
            if len(nums_sorted) % 2:
                info["median"] = nums_sorted[mid]
            else:
                info["median"] = round((nums_sorted[mid - 1] + nums_sorted[mid]) / 2, 4)

    elif dtype == "categorical":
        counter: Counter[Any] = Counter(non_null)
        info["top_values"] = [
            {"value": _json_safe(v), "count": c} for v, c in counter.most_common(10)
        ]

    elif dtype == "text":
        lengths = [len(str(v)) for v in non_null]
        info["min_length"] = min(lengths)
        info["max_length"] = max(lengths)
        info["mean_length"] = round(sum(lengths) / len(lengths), 2)

    return info


def _classify_dtype(non_null: list[Any], total_n: int) -> str:
    """Determine the dtype label for a column."""
    # bool must be checked before int (bool is a subclass of int in Python)
    if all(isinstance(v, bool) for v in non_null):
        return "bool"

    if all(isinstance(v, int) and not isinstance(v, bool) for v in non_null):
        return "int"

    if all(isinstance(v, float) for v in non_null):
        return "float"

    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
        return "float"

    if all(isinstance(v, str) for v in non_null):
        n_unique = len(set(non_null))
        if n_unique <= 50 or (total_n > 0 and n_unique / total_n <= 0.2):
            return "categorical"
        return "text"

    if all(isinstance(v, (list, tuple)) for v in non_null):
        return "list"

    if all(isinstance(v, datetime.datetime) for v in non_null):
        return "datetime"

    # Mixed types — check if str-heavy
    str_count = sum(1 for v in non_null if isinstance(v, str))
    if str_count > len(non_null) * 0.5:
        n_unique = len({str(v) for v in non_null})
        if n_unique <= 50 or (total_n > 0 and n_unique / total_n <= 0.2):
            return "categorical"
        return "text"

    return "unknown"


def _safe_repr(v: Any, max_len: int = 80) -> str:
    """Return repr(v) truncated to *max_len* characters."""
    r = repr(v)
    return r[:max_len] if len(r) <= max_len else r[: max_len - 3] + "..."


def _json_safe(v: Any) -> int | float | str | bool | None:
    """Ensure a value is JSON-serializable."""
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return repr(v)
