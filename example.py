"""End-to-end example of dspy-auto-signature with RLM for best quality.

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
    # 1. Configure the meta-models for signature generation.
    #    - lm: Strong model for the RLM outer loop (analyzes the prompt)
    #    - sub_lm: Cheap model for RLM inner loop (sub-queries, exploration)
    #    Using a strong lm + cheap sub_lm gives the best quality/cost tradeoff.
    lm = dspy.LM(
        model="openrouter/openai/gpt-oss-120b",
        cache=False,
        extra_body={"provider": {"order": ["groq"], "allow_fallbacks": False}},
    )
    das.configure(lm=lm)

    # 2. Generate a signature from a raw prompt.
    #    This uses the RLM-based generator which iteratively analyzes the
    #    prompt using a sandboxed Python REPL before proposing a spec.
    sig = das.generate(
        "Given a customer support ticket with {message}, {category}, and "
        "{priority}, predict the urgency level and suggest a response tone."
    )

    # 3. Save the generated signature to a file.
    with open("output/example_prompt_signature.py", "w", encoding="utf-8") as f:
        f.write(sig.to_source())
    print("Saved signature to output/ticket_signature.py")

    # 4. Inspect what was generated.
    print(f"\nGenerated signature: {sig.__name__}")
    print(f"Docstring: {sig.__doc__}")
    print(f"Inputs:  {list(sig.input_fields.keys())}")
    print(f"Outputs: {list(sig.output_fields.keys())}")

    # 5. Configure the runtime model separately.
    #    Use a cheaper/faster model for repeated inference.
    dspy.configure(lm=lm)

    # 6. Use it immediately with any DSPy predictor.
    classifier = dspy.ChainOfThought(sig.to_signature())

    values = {
        "message": "My server is on fire and I can't access the dashboard!",
        "category": "technical",
        "priority": "P0",
    }
    runtime_inputs: dict[str, str] = {}
    for field_name in sig.input_fields:
        matching_keys = [
            key for key in values if field_name == key or field_name.endswith(f"_{key}")
        ]
        if not matching_keys:
            raise RuntimeError(
                f"No example value configured for generated input {field_name!r}"
            )
        runtime_inputs[field_name] = values[matching_keys[0]]
    result = classifier(**runtime_inputs)

    print("\n--- Result ---")
    print(result)

    print(dspy.inspect_history())


if __name__ == "__main__":
    main()
