"""RLM-based slow-path generator that produces SignatureSpec from a profiled dataset."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

import dspy

from dspy_auto_signature.core.config import Config
from dspy_auto_signature.generator.rlm_signatures import ProposeSignatureFromData
from dspy_auto_signature.types.signature_spec import FieldSpec, FieldType, SignatureSpec

if TYPE_CHECKING:
    from dspy_auto_signature.types.signature_spec import ParsedPrompt

logger = logging.getLogger(__name__)


class RLMSignatureGenerator(dspy.Module):
    """Meta-DSPy module that generates ``SignatureSpec`` from a profiled dataset.

    Uses :class:`dspy.RLM` (Recursive Language Model) to introspect column
    profiles and sample rows, producing a more thoughtful signature than the
    fast-path :class:`SignatureGenerator`.

    This is the **slow path** — it trades latency for quality by giving the
    LLM a sandboxed Python REPL to explore the data before proposing a spec.
    """

    def __init__(
        self,
        max_iterations: int = 20,
        max_llm_calls: int = 50,
        sub_lm: dspy.LM | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.sub_lm = sub_lm
        self.rlm = dspy.RLM(
            ProposeSignatureFromData,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            sub_lm=sub_lm,
            verbose=verbose,
        )

    def forward(self, prompt: ParsedPrompt) -> SignatureSpec:
        """Generate a :class:`SignatureSpec` from a parsed prompt whose ``raw_input`` is a dataset.

        Args:
            prompt: A normalised :class:`ParsedPrompt`. Its ``raw_input``
                should be a tabular dataset (list[dict], DataFrame, etc.).

        Returns:
            A fully-populated :class:`SignatureSpec`.

        """
        from dspy_auto_signature.data.profiler import profile_columns
        from dspy_auto_signature.data.to_records import to_records

        rows = to_records(prompt.raw_input)
        profile = profile_columns(rows)

        data_profile_json = json.dumps(profile, indent=2, default=str)
        sample_rows_json = json.dumps(rows[:5], indent=2, default=str)

        lm = Config.get_lm()

        with dspy.settings.context(lm=lm):
            result = self.rlm(
                data_profile_json=data_profile_json,
                sample_rows_json=sample_rows_json,
                task_hint="",
            )

        parsed = self._parse_spec_json(result.spec_json)
        if parsed is None:
            logger.warning(
                "RLM returned malformed spec JSON; falling back to minimal spec"
            )
            return SignatureSpec(
                name="AutoSignature",
                instructions="Process the input and produce an output.",
                inputs=[
                    FieldSpec(
                        name="input_text",
                        description="The input to process",
                        suggested_type="string",
                        field_type=FieldType.INPUT,
                    )
                ],
                outputs=[
                    FieldSpec(
                        name="output_text",
                        description="The generated output",
                        suggested_type="string",
                        field_type=FieldType.OUTPUT,
                    )
                ],
            )

        inputs: list[FieldSpec] = []
        outputs: list[FieldSpec] = []

        for f in parsed["inputs"]:
            inputs.append(
                FieldSpec(
                    name=f["name"],
                    description=f["description"],
                    suggested_type=f.get("type", "string"),
                    field_type=FieldType.INPUT,
                )
            )

        for f in parsed["outputs"]:
            outputs.append(
                FieldSpec(
                    name=f["name"],
                    description=f["description"],
                    suggested_type=f.get("type", "string"),
                    field_type=FieldType.OUTPUT,
                )
            )

        return SignatureSpec(
            name=parsed["name"],
            instructions=parsed["instructions"],
            inputs=inputs,
            outputs=outputs,
        )

    @staticmethod
    def _parse_spec_json(raw: str) -> dict | None:
        """Parse and validate the JSON spec from the RLM output.

        Tries direct ``json.loads`` first, then regex extraction of an
        embedded JSON object.  Returns ``None`` if the structure is invalid.
        """
        parsed: dict | None = None

        # Direct parse
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass

        # Regex fallback — extract first {...} block
        if parsed is None:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except (json.JSONDecodeError, TypeError):
                    pass

        if parsed is None:
            return None

        # Validate top-level keys
        required_keys = {"name", "instructions", "inputs", "outputs"}
        if not required_keys.issubset(parsed.keys()):
            return None

        # Validate inputs/outputs are lists of dicts with required keys
        for key in ("inputs", "outputs"):
            items = parsed[key]
            if not isinstance(items, list):
                return None
            for item in items:
                if not isinstance(item, dict):
                    return None
                if not {"name", "description", "type"}.issubset(item.keys()):
                    return None

        return parsed
