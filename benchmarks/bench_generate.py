"""Deterministic, offline benchmark for dspy-auto-signature generation latency.

Every workload runs the full public ``das.generate()`` pipeline end to end:
parser -> spec -> signature class -> to_source(). The RLM path is driven by a
canned offline LM that answers in the exact format the RLM's action/extract
adapters expect, so the benchmark measures the real orchestration overhead
(Deno/Pyodide sandbox startup, Predict machinery, parsing, class building)
with zero network. Emitted metrics:

- METRIC generate_ms            median per-workload median wall time (primary)
- METRIC bench_<workload>_ms    per-workload median wall time
- METRIC llm_calls              mean canned-LLM calls per generation (0 = fast path)
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from typing import Any

import dspy
from dspy.dsp.utils.utils import dotdict

import dspy_auto_signature as das
from dspy_auto_signature.core.config import Config

REPS = 5

# One representative draft; both RLM signatures (unified and SDK) submit it.
DRAFT = {
    "name": "SummarizeArticle",
    "instructions": "Summarize the given article and extract key takeaways.",
    "inputs": [
        {"name": "article", "description": "The article text", "type": "string"},
    ],
    "outputs": [
        {"name": "summary", "description": "Concise summary", "type": "string"},
        {
            "name": "takeaways",
            "description": "Three key takeaways",
            "type": "list of strings",
        },
    ],
}
SUBMIT_CODE = f"SUBMIT(draft={json.dumps(DRAFT)})"

CHAT_RESPONSE = "[[ ## reasoning ## ]]\nReady.\n\n[[ ## code ## ]]\n" + SUBMIT_CODE
JSON_RESPONSE = json.dumps({"reasoning": "Ready.", "code": SUBMIT_CODE})


class CannedLM(dspy.LM):
    """Offline LM returning a canned SUBMIT action for every RLM iteration."""

    def __init__(self) -> None:
        super().__init__("canned", "chat", 0.0, 100_000, True)
        self.calls = 0

    def forward(  # type: ignore[override]
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        self.calls += 1
        content = messages[-1]["content"] if messages else (prompt or "")
        tail = str(content)[-200:]
        response = (
            JSON_RESPONSE if "Respond with a JSON object" in tail else CHAT_RESPONSE
        )
        return dotdict(
            choices=[
                dotdict(
                    message=dotdict(content=response, tool_calls=None),
                    finish_reason="stop",
                ),
            ],
            usage=dotdict(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            model="canned",
        )


def workloads() -> list[tuple[str, object, str | None]]:
    """Named (metric_key, source, task_hint) fixtures; no randomness, no network."""
    tickets = [
        {"message": "Server is down", "urgency": "high"},
        {"message": "Please update my profile", "urgency": "low"},
        {"message": "Payment failed", "urgency": "high"},
    ]
    openai_messages = [
        {
            "role": "system",
            "content": "You are a technical writer producing clear documentation.",
        },
        {
            "role": "user",
            "content": "Write a README for a Python CLI tool that converts CSV to JSON.",
        },
    ]
    anthropic_messages = [
        {"role": "user", "content": "Analyze the sentiment of customer reviews."},
        {
            "role": "assistant",
            "content": "I'll classify each review as positive, negative, or neutral.",
        },
    ]
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
    return [
        (
            "prompt",
            "Given an article, produce a concise summary and three key takeaways.",
            None,
        ),
        (
            "prompt_placeholders",
            "Summarize the {article} and translate the {paragraph} into French.",
            None,
        ),
        ("sdk_openai", openai_messages, None),
        ("sdk_anthropic", anthropic_messages, None),
        ("sdk_gemini", gemini_contents, None),
        ("dataset", tickets, "Classify support ticket urgency from the message"),
    ]


def main() -> int:
    lm = CannedLM()
    Config.reset()
    das.configure(lm=lm, sub_lm=lm)

    # Warmup: imports, Pyodide compile cache, adapters.
    warm = das.generate(workloads()[0][1])  # type: ignore[arg-type]
    assert warm is not None

    per_workload_ms: dict[str, float] = {}
    calls_per_gen: list[float] = []
    for key, source, task_hint in workloads():
        walls: list[float] = []
        calls: list[float] = []
        for _ in range(REPS):
            lm.calls = 0
            t0 = time.perf_counter()
            signature = das.generate(source, task_hint=task_hint)
            walls.append((time.perf_counter() - t0) * 1000.0)
            # The generated class must be a usable dspy.Signature.
            assert (
                hasattr(signature, "input_fields") and len(signature.input_fields) >= 1
            )
            assert len(signature.output_fields) >= 1
            calls.append(lm.calls)
        per_workload_ms[f"bench_{key}_ms"] = statistics.median(walls)
        calls_per_gen.append(statistics.mean(calls))

    primary = statistics.median(per_workload_ms.values())
    print(f"METRIC generate_ms={primary:.2f}")
    for key, value in sorted(per_workload_ms.items()):
        print(f"METRIC {key}={value:.2f}")
    print(f"METRIC llm_calls={statistics.mean(calls_per_gen):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
