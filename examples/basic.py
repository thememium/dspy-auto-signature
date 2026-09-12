"""End-to-end example of dspy-auto-signature's three generation modes.

The same prompt is run through every tier so you can compare what each
one designs:

1. ``fast`` — deterministic structural heuristics, no LLM contact
2. ``cot``  — a single ``dspy.ChainOfThought`` call designs the signature
3. ``rlm``  — the recursive ``dspy.RLM`` architect with a sandboxed Python
   REPL and sub-queries (requires Deno, makes several LLM calls)

Prerequisites:
    export OPENROUTER_API_KEY="sk-..."
    uv sync  # or: pip install -e .

Run:
    python examples/basic.py
"""

from __future__ import annotations

from typing import Literal

import dspy

import dspy_auto_signature as das

Mode = Literal["fast", "cot", "rlm"]

PROMPT = (
    "Given a customer support ticket with message,"
    "predict the urgency level and sentiment."
)


def generate_and_save(mode: Mode, filename: str) -> None:
    """Generate a signature with *mode*, save its source, and summarize it."""
    sig = das.generate(PROMPT, mode=mode)

    with open(f"examples/output/{filename}", "w", encoding="utf-8") as f:
        f.write(sig.to_source())
    print(f"Saved signature to output/{filename}")

    print(f"\n[{mode}] Generated signature: {sig.__name__}")
    print(f"Docstring: {sig.__doc__}")
    print(f"Inputs:  {list(sig.input_fields.keys())}")
    print(f"Outputs: {list(sig.output_fields.keys())}")


def main() -> None:
    # 1. Configure the meta-model. One model handles the ChainOfThought call
    #    and the RLM outer loop. A cheap sub_lm is only used by the RLM's
    #    inner loop, if you enable it.
    lm = dspy.LM(
        model="openrouter/openai/gpt-oss-120b",
        cache=False,
        extra_body={"provider": {"order": ["groq"], "allow_fallbacks": False}},
    )
    das.configure(lm=lm)

    # 2. mode="fast": never contacts an LLM. Without placeholders or message
    #    structure to lean on, this produces the deterministic fallback
    #    signature (inferred input/output names). Quickest and cheapest.
    generate_and_save("fast", "example_basic_fast_signature.py")

    # 3. mode="cot": one ChainOfThought round trip. LLM-driven quality
    #    without a sandbox or iterative exploration. This is also what
    #    mode="auto" falls back to for structureless prompts.
    generate_and_save("cot", "example_basic_cot_signature.py")

    # 4. mode="rlm": the RLM architect iteratively analyzes the prompt in a
    #    Deno-sandboxed Python REPL before proposing a spec. Slowest, but
    #    the richest signatures on complex prompts. (Deno required.)
    generate_and_save("rlm", "example_basic_rlm_signature.py")


if __name__ == "__main__":
    main()
