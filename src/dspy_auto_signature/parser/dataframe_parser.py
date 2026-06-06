"""Parser for tabular data (DataFrames, list[dict], dspy.Example)."""

from __future__ import annotations

import logging
from typing import Any

from dspy_auto_signature.data.profiler import profile_columns
from dspy_auto_signature.data.to_records import to_records
from dspy_auto_signature.parser.base import PromptParser
from dspy_auto_signature.types.signature_spec import ParsedPrompt

logger = logging.getLogger(__name__)


class DataFrameParser(PromptParser):
    """Parse tabular data into a :class:`ParsedPrompt` with column profiles.

    Supports pandas DataFrames, polars DataFrames/LazyFrames,
    ``list[dict]``, ``list[dspy.Example]``, single ``dspy.Example``,
    and any object with ``.to_dicts()`` / ``.to_pandas()`` methods.

    Example::

        import pandas as pd
        from dspy_auto_signature.parser.dataframe_parser import DataFrameParser

        df = pd.DataFrame({"question": ["What is AI?"], "answer": ["AI is ..."]})
        parser = DataFrameParser()
        assert parser.can_parse(df)
        prompt = parser.parse(df)
        print(prompt.instruction_text)
    """

    def can_parse(self, raw: Any) -> bool:  # noqa: C901 — intentionally exhaustive
        # Cheapest checks first.
        try:
            import pandas as pd  # type: ignore[import-untyped]

            if isinstance(raw, pd.DataFrame):
                return True
        except ImportError:
            pass

        if hasattr(raw, "to_dicts"):
            return True
        if hasattr(raw, "to_pandas"):
            return True

        if isinstance(raw, list):
            if not raw:
                return False
            first = raw[0]
            if isinstance(first, dict):
                return True
            # dspy.Example check without hard import
            if type(first).__name__ == "Example":
                return True

        if type(raw).__name__ == "Example":
            return True

        return False

    def parse(self, raw: Any) -> ParsedPrompt:
        rows = to_records(raw)
        profile = profile_columns(rows)

        instruction_text = self._build_instruction(profile)
        examples = self._build_examples(rows)

        return ParsedPrompt(
            instruction_text=instruction_text,
            examples=examples,
            raw_input=raw,
        )

    @staticmethod
    def _build_instruction(profile: dict[str, Any]) -> str:
        n_rows = profile["n_rows"]
        n_cols = profile["n_cols"]
        lines = [f"Dataset with {n_rows} rows and {n_cols} columns.", ""]
        lines.append("Column profiles:")

        for col_name, col_info in profile["columns"].items():
            dtype = col_info["dtype"]
            null_pct = col_info["null_rate"] * 100
            n_unique = col_info["n_unique"]
            samples = col_info.get("sample_values", [])
            sample_str = ", ".join(str(s) for s in samples[:3])
            line = f"  - {col_name} ({dtype}): null_rate {null_pct:.1f}%, n_unique {n_unique}"
            if sample_str:
                line += f", samples: [{sample_str}]"
            lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def _build_examples(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
        examples: list[dict[str, str]] = []
        for row in rows[:3]:
            truncated: dict[str, str] = {}
            for i, (k, v) in enumerate(row.items()):
                if i >= 5:
                    break
                s = repr(v) if v is not None else "null"
                truncated[k] = s[:200] if len(s) > 200 else s
            examples.append(truncated)
        return examples
