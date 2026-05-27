"""Basic end-to-end example of dspy-auto-signature.

Prerequisites:
    export OPENROUTER_API_KEY="sk-..."
    uv sync  # or: pip install -e .

Run:
    python example.py
"""

from __future__ import annotations

import dspy

import dspy_auto_signature as das


def main() -> None:
    # 1. Configure the meta-model for signature generation.
    #    One-time setup — use a strong model for best results.
    das.configure(lm=dspy.LM("openrouter/openai/gpt-oss-120b"))

    # 2. Generate a signature from a raw prompt.
    sig = das.from_prompt("Summarize the following article into 3 short bullet points")

    # 3. Inspect what was generated.
    print(f"Generated signature: {sig}")
    print(f"Docstring: {sig.__doc__}")
    print(f"Inputs:  {list(sig.input_fields.keys())}")
    print(f"Outputs: {list(sig.output_fields.keys())}")

    # 4. Configure the runtime model separately.
    #    Use a cheaper/faster model for repeated inference.
    dspy.configure(lm=dspy.LM("openrouter/openai/gpt-oss-120b"))

    # 5. Use it immediately with any DSPy predictor.
    summarizer = dspy.ChainOfThought(sig)

    article = (
        "Artificial intelligence has transformed industries ranging from "
        "healthcare to finance. Machine learning models can now diagnose "
        "diseases, predict market trends, and automate customer service. "
        "However, ethical concerns around bias, privacy, and job displacement "
        "remain significant challenges that researchers and policymakers "
        "continue to address."
    )

    result = summarizer(article=article)

    print("\n--- Summary ---")
    print(result)

    print(dspy.inspect_history())


if __name__ == "__main__":
    main()
