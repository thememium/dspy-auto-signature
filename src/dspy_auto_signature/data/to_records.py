"""Convert heterogeneous tabular data into list[dict] records."""

from __future__ import annotations

from typing import Any


def _example_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dspy.Example to a plain dict."""
    try:
        return obj.toDict()  # type: ignore[union-attr]
    except AttributeError:
        pass
    try:
        return obj.to_dict()  # type: ignore[union-attr]
    except AttributeError:
        pass
    return dict(vars(obj))


def _to_dicts(obj: Any) -> list[dict[str, Any]]:
    """Attempt to coerce *obj* to a list of dicts using duck typing."""
    if isinstance(obj, list):
        return obj

    if hasattr(obj, "to_dicts"):
        result = obj.to_dicts()
        if isinstance(result, list):
            return result

    if hasattr(obj, "collect"):
        df = obj.collect()
        if hasattr(df, "to_dicts"):
            return df.to_dicts()

    if hasattr(obj, "to_dict"):
        result = obj.to_dict(orient="records")
        if isinstance(result, list):
            return result

    if hasattr(obj, "to_pandas"):
        pd_df = obj.to_pandas()
        result = pd_df.to_dict(orient="records")
        if isinstance(result, list):
            return result

    raise TypeError(
        f"Cannot convert {type(obj).__name__} to list of dicts. "
        "Expected: list[dict], pandas DataFrame, polars DataFrame/LazyFrame, "
        "or any object with .to_dicts() or .to_pandas()."
    )


def to_records(obj: Any) -> list[dict[str, Any]]:
    """Convert heterogeneous tabular data into a list of plain dicts.

    Supported inputs:

    - ``list[dict]`` — returned as-is.
    - ``pandas.DataFrame`` — converted via ``.to_dict(orient="records")``.
    - ``polars.DataFrame`` / ``polars.LazyFrame`` — converted via
      ``.to_dicts()`` (LazyFrame is collected first).
    - Any object with a ``.to_dicts()`` or ``.to_pandas()`` method.
    - A single ``dspy.Example`` — converted via ``.toDict()``.
    - ``list[dspy.Example]`` — each element converted individually.

    Args:
        obj: The source data.

    Returns:
        A list of dicts, one per row.

    Raises:
        TypeError: If *obj* cannot be converted.

    """
    # Single dspy.Example
    if _is_dspy_example(obj):
        return [_example_to_dict(obj)]

    # List — check if elements are dspy.Example instances
    if isinstance(obj, list) and obj and any(_is_dspy_example(el) for el in obj):
        return [_example_to_dict(el) for el in obj]

    return _to_dicts(obj)


def _is_dspy_example(obj: Any) -> bool:
    """Check whether *obj* is a dspy.Example without a hard import."""
    type_name = type(obj).__name__
    if type_name == "Example":
        return True
    # Fallback: check the module path
    module = getattr(type(obj), "__module__", "") or ""
    return module.startswith("dspy") and type_name == "Example"
