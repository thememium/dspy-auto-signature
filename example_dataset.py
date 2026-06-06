"""Dataset-driven (slow path) example for dspy-auto-signature.

Demonstrates ``from_dataset()`` with a pandas DataFrame and a list of dicts.
Requires Deno to be installed for the RLM (Recursive Language Model) sandbox:

    brew install deno   # macOS

Prerequisites:
    export OPENROUTER_API_KEY="sk-..."
    uv sync --extra dataset  # or: pip install -e ".[dataset]"

Run:
    python example_dataset.py
"""

from __future__ import annotations

import dspy

import dspy_auto_signature as das

lm = dspy.LM(
    model="openrouter/openai/gpt-oss-120b",
    cache=False,
    extra_body={"provider": {"order": ["groq"], "allow_fallbacks": False}},
)


def from_dataframe_example() -> None:
    """Generate a signature from a pandas DataFrame (slow path)."""
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed. Run: uv sync --extra dataset")
        return

    das.configure(lm=lm)

    df = pd.DataFrame(
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

    sig = das.from_dataset(
        df,
        task_hint="Classify support tickets by urgency and sentiment",
    )

    print("=== DataFrame → Signature ===")
    print(f"Generated: {sig}")
    print(f"Docstring: {sig.__doc__}")
    print(f"Inputs:    {list(sig.input_fields.keys())}")
    print(f"Outputs:   {list(sig.output_fields.keys())}")

    with open("output/example_dataset_signature.py", "w", encoding="utf-8") as f:
        f.write(sig.to_source())


def from_list_example() -> None:
    """Generate a signature from a list of dicts (slow path)."""
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

    sig = das.from_dataset(
        rows,
        task_hint="Classify support tickets by urgency and sentiment",
    )

    print("\n=== List of dicts → Signature ===")
    print(f"Generated: {sig}")
    print(f"Docstring: {sig.__doc__}")
    print(f"Inputs:    {list(sig.input_fields.keys())}")
    print(f"Outputs:   {list(sig.output_fields.keys())}")

    with open("output/example_list_signature.py", "w", encoding="utf-8") as f:
        f.write(sig.to_source())


def main() -> None:
    from_dataframe_example()
    from_list_example()


if __name__ == "__main__":
    main()
