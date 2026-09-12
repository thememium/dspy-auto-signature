"""Dataset-driven example for dspy-auto-signature.

Demonstrates ``generate()`` with a pandas DataFrame and a list of dicts.
Under the default ``auto`` mode, dataset signatures are designed
deterministically from the column profile — no LLM call. Passing
``mode="cot"`` instead asks a single ``dspy.ChainOfThought`` call to design
the signature from the same profile; ``mode="rlm"`` opts into the recursive
RLM architect. Only ``mode="rlm"`` needs Deno installed:

    brew install deno   # macOS

Prerequisites:
    export OPENROUTER_API_KEY="sk-..."
    uv sync --extra dataset  # or: pip install -e ".[dataset]"

Run:
    python examples/dataset.py
"""

from __future__ import annotations
from typing import Any
import dspy

import dspy_auto_signature as das

lm = dspy.LM(
    model="openrouter/openai/gpt-oss-120b",
    cache=False,
    extra_body={"provider": {"order": ["groq"], "allow_fallbacks": False}},
)


def _support_tickets_df() -> Any:
    """Build the example support-ticket DataFrame (pandas optional)."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas not installed. Run: uv sync --extra dataset") from exc

    return pd.DataFrame(
        {
            "message": [
                "The server room AC is out and equipment is overheating.",
                "Can someone clean conference room B next week?",
                "Thanks for fixing the VPN, works perfectly now!",
                "All login credentials expired overnight.",
            ],
            "urgency": ["high", "low", "medium", "high"],
            "sentiment": ["negative", "neutral", "positive", "negative"],
        },
    )


def from_dataframe_example() -> None:
    """Generate a signature from a pandas DataFrame (deterministic path)."""
    das.configure(lm=lm)
    df = _support_tickets_df()

    sig = das.generate(
        df,
        task_hint="Classify support tickets by urgency and sentiment",
    )

    print("=== DataFrame → Signature ===")
    print(f"Generated: {sig}")
    print(f"Docstring: {sig.__doc__}")
    print(f"Inputs:    {list(sig.input_fields.keys())}")
    print(f"Outputs:   {list(sig.output_fields.keys())}")

    with open(
        "examples/output/example_dataset_signature.py", "w", encoding="utf-8"
    ) as f:
        f.write(sig.to_source())


def cot_dataframe_example() -> None:
    """Generate the same signature with one ChainOfThought call (mode="cot")."""
    das.configure(lm=lm)
    df = _support_tickets_df()

    sig = das.generate(
        df,
        task_hint="Classify support tickets by urgency and sentiment",
        mode="cot",
    )

    print("\n=== DataFrame → Signature (ChainOfThought mode) ===")
    print(f"Generated: {sig}")
    print(f"Docstring: {sig.__doc__}")
    print(f"Inputs:    {list(sig.input_fields.keys())}")
    print(f"Outputs:   {list(sig.output_fields.keys())}")

    with open(
        "examples/output/example_dataset_cot_signature.py", "w", encoding="utf-8"
    ) as f:
        f.write(sig.to_source())

    print("=== DataFrame → Signature ===")
    print(f"Generated: {sig}")
    print(f"Docstring: {sig.__doc__}")
    print(f"Inputs:    {list(sig.input_fields.keys())}")
    print(f"Outputs:   {list(sig.output_fields.keys())}")

    with open(
        "examples/output/example_dataset_signature.py", "w", encoding="utf-8"
    ) as f:
        f.write(sig.to_source())


def from_list_example() -> None:
    """Generate a signature from a list of dicts (deterministic path)."""
    das.configure(lm=lm)

    rows = [
        {
            "message": "The server room AC is out and equipment is overheating.",
            "urgency": "high",
            "sentiment": "negative",
        },
        {
            "message": "Can someone clean conference room B next week?",
            "urgency": "low",
            "sentiment": "neutral",
        },
        {
            "message": "Thanks for fixing the VPN, works perfectly now!",
            "urgency": "medium",
            "sentiment": "positive",
        },
    ]

    sig = das.generate(
        rows,
        task_hint="Classify support tickets by urgency and sentiment",
    )

    print("\n=== List of dicts → Signature ===")
    print(f"Generated: {sig}")
    print(f"Docstring: {sig.__doc__}")
    print(f"Inputs:    {list(sig.input_fields.keys())}")
    print(f"Outputs:   {list(sig.output_fields.keys())}")

    with open("examples/output/example_list_signature.py", "w", encoding="utf-8") as f:
        f.write(sig.to_source())


def main() -> None:
    from_dataframe_example()
    cot_dataframe_example()
    from_list_example()


if __name__ == "__main__":
    main()
