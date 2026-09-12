"""End-to-end example of dspy-auto-signature with SDK message arrays.

Shows how to generate DSPy signatures from OpenAI, Anthropic, and Google
Gemini message formats — no conversion needed, pass your existing messages
directly.

Prerequisites:
    export OPENROUTER_API_KEY="sk-..."
    uv sync  # or: pip install -e .

Run:
    python examples/sdk.py
"""

from __future__ import annotations

import dspy

import dspy_auto_signature as das


def main() -> None:
    lm = dspy.LM(
        model="openrouter/openai/gpt-oss-120b",
        cache=False,
        extra_body={"provider": {"order": ["groq"], "allow_fallbacks": False}},
    )
    das.configure(lm=lm)

    # --- OpenAI SDK format ---
    openai_messages = [
        {
            "role": "system",
            "content": "You are a support ticket classifier that determines urgency and sentiment.",
        },
        {
            "role": "user",
            "content": "The server room AC is out and equipment is overheating.",
        },
        {
            "role": "assistant",
            "content": "urgency: high, sentiment: negative",
        },
    ]
    sig_openai = das.generate(openai_messages)

    with open(
        "examples/output/example_sdk_openai_signature.py", "w", encoding="utf-8"
    ) as f:
        f.write(sig_openai.to_source())

    print("Saved OpenAI signature to output/example_sdk_openai_signature.py")
    print(f"  Inputs:  {list(sig_openai.input_fields.keys())}")
    print(f"  Outputs: {list(sig_openai.output_fields.keys())}")

    # --- Anthropic SDK format ---
    anthropic_messages = [
        {"role": "user", "content": "Analyze the sentiment of customer reviews."},
        {
            "role": "assistant",
            "content": "I'll classify each review as positive, negative, or neutral.",
        },
    ]
    sig_anthropic = das.generate(anthropic_messages)

    with open(
        "examples/output/example_sdk_anthropic_signature.py", "w", encoding="utf-8"
    ) as f:
        f.write(sig_anthropic.to_source())

    print("\nSaved Anthropic signature to output/example_sdk_anthropic_signature.py")
    print(f"  Inputs:  {list(sig_anthropic.input_fields.keys())}")
    print(f"  Outputs: {list(sig_anthropic.output_fields.keys())}")

    # --- Google Gemini SDK format ---
    gemini_contents = [
        {
            "role": "user",
            "parts": [{"text": "Extract key entities from this legal contract."}],
        },
        {
            "role": "model",
            "parts": [{"text": "I'll identify parties, dates, and obligations."}],
        },
    ]
    sig_gemini = das.generate(gemini_contents)

    with open(
        "examples/output/example_sdk_gemini_signature.py", "w", encoding="utf-8"
    ) as f:
        f.write(sig_gemini.to_source())

    print("\nSaved Gemini signature to output/example_sdk_gemini_signature.py")
    print(f"  Inputs:  {list(sig_gemini.input_fields.keys())}")
    print(f"  Outputs: {list(sig_gemini.output_fields.keys())}")

    # --- ChainOfThought mode ---
    # Structured SDK arrays like these are designed deterministically under
    # the default "auto" mode. mode="cot" forces a single ChainOfThought
    # call to design the signature from the message semantics instead.
    sig_cot = das.generate(openai_messages, mode="cot")

    with open(
        "examples/output/example_sdk_openai_cot_signature.py", "w", encoding="utf-8"
    ) as f:
        f.write(sig_cot.to_source())

    print("\nSaved OpenAI CoT signature to output/example_sdk_openai_cot_signature.py")
    print(f"  Inputs:  {list(sig_cot.input_fields.keys())}")
    print(f"  Outputs: {list(sig_cot.output_fields.keys())}")

    sig_anthropic_cot = das.generate(anthropic_messages, mode="cot")

    with open(
        "examples/output/example_sdk_anthropic_cot_signature.py", "w", encoding="utf-8"
    ) as f:
        f.write(sig_anthropic_cot.to_source())

    print(
        "\nSaved Anthropic CoT signature to"
        " output/example_sdk_anthropic_cot_signature.py"
    )
    print(f"  Inputs:  {list(sig_anthropic_cot.input_fields.keys())}")
    print(f"  Outputs: {list(sig_anthropic_cot.output_fields.keys())}")

    sig_gemini_cot = das.generate(gemini_contents, mode="cot")

    with open(
        "examples/output/example_sdk_gemini_cot_signature.py", "w", encoding="utf-8"
    ) as f:
        f.write(sig_gemini_cot.to_source())

    print("\nSaved Gemini CoT signature to output/example_sdk_gemini_cot_signature.py")
    print(f"  Inputs:  {list(sig_gemini_cot.input_fields.keys())}")
    print(f"  Outputs: {list(sig_gemini_cot.output_fields.keys())}")


if __name__ == "__main__":
    main()
