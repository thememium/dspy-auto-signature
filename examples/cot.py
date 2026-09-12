"""End-to-end example of dspy-auto-signature in ChainOfThought mode.

The middle tier between ``fast`` (deterministic) and ``rlm`` (heavy recursive
architect): a single ``dspy.ChainOfThought`` call designs the signature.
No Deno sandbox and no iterative sub-queries, but still fully LLM-driven.

Prerequisites:
    export OPENROUTER_API_KEY="sk-..."
    uv sync  # or: pip install -e .

Run:
    python examples/cot.py
"""

from __future__ import annotations

import dspy

import dspy_auto_signature as das


def main() -> None:
    # 1. Configure the meta-model. ChainOfThought mode makes one direct call,
    #    so there is no sub_lm — the single model handles analysis and design.
    lm = dspy.LM(
        model="openrouter/openai/gpt-oss-120b",
        cache=False,
        extra_body={"provider": {"order": ["groq"], "allow_fallbacks": False}},
    )
    das.configure(lm=lm)

    # 2. Generate a signature from a raw prompt with mode="cot".
    #    A plain structureless prompt would fall back to the deterministic
    #    heuristic under "auto"/"fast"; "cot" always uses the LLM architect.
    sig = das.generate(
        "Given a customer support ticket with message,"
        "predict the urgency level and sentiment.",
        mode="cot",
    )

    # 3. Save the generated signature to a file.
    with open(
        "examples/output/example_basic_cot_signature.py", "w", encoding="utf-8"
    ) as f:
        f.write(sig.to_source())
    print("Saved signature to output/example_basic_cot_signature.py")

    # 4. Inspect what was generated.
    print(f"\nGenerated signature: {sig.__name__}")
    print(f"Docstring: {sig.__doc__}")
    print(f"Inputs:  {list(sig.input_fields.keys())}")
    print(f"Outputs: {list(sig.output_fields.keys())}")


if __name__ == "__main__":
    main()
