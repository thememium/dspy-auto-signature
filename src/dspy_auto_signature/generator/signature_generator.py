"""DSPy module that generates SignatureSpec from ParsedPrompt."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

import dspy

from dspy_auto_signature.types.signature_spec import FieldSpec, FieldType, SignatureSpec

if TYPE_CHECKING:
    from dspy_auto_signature.types.signature_spec import ParsedPrompt

logger = logging.getLogger(__name__)


class AnalyzePrompt(dspy.Signature):
    """Analyze a raw prompt and extract the core task."""

    raw_prompt: str = dspy.InputField(desc="The raw prompt text to analyze")
    task_name: str = dspy.OutputField(desc="A PascalCase name for this task")
    task_instruction: str = dspy.OutputField(
        desc="A clear, concise instruction describing what the task should do"
    )


class ExtractFields(dspy.Signature):
    """Extract input and output fields for an AI task."""

    task_instruction: str = dspy.InputField(desc="The core task instruction")
    raw_prompt_context: str = dspy.InputField(desc="The original prompt for context")
    fields_json: str = dspy.OutputField(
        desc="A JSON array of field objects. Each object has: name (snake_case), description (string), type (string like 'string' or 'list of strings'), field_type ('input' or 'output')",
    )


class RefineSignature(dspy.Signature):
    """Refine a signature specification for clarity and completeness."""

    draft_name: str = dspy.InputField(desc="Current task name")
    draft_instruction: str = dspy.InputField(desc="Current task instruction")
    draft_fields_json: str = dspy.InputField(desc="Current fields as JSON array")
    refined_name: str = dspy.OutputField(desc="Improved PascalCase name")
    refined_instruction: str = dspy.OutputField(desc="Improved, concise instruction")
    refined_fields_json: str = dspy.OutputField(
        desc="Improved fields as JSON array with same schema"
    )


class SignatureGenerator(dspy.Module):
    """Meta-DSPy module that generates ``SignatureSpec`` from ``ParsedPrompt``.

    Uses a 3-step chain internally:

    1. **Analyze** — extracts the core task name and instruction
    2. **Extract Fields** — identifies inputs and outputs
    3. **Refine** — polishes names, descriptions, and types
    """

    def __init__(self) -> None:
        super().__init__()
        self.analyze = dspy.ChainOfThought(AnalyzePrompt)
        self.extract_fields = dspy.ChainOfThought(ExtractFields)
        self.refine = dspy.ChainOfThought(RefineSignature)

    def forward(self, prompt: ParsedPrompt) -> SignatureSpec:
        """Generate a :class:`SignatureSpec` from a parsed prompt.

        Args:
            prompt: A normalised :class:`ParsedPrompt`.

        Returns:
            A fully-populated :class:`SignatureSpec`.

        """
        raw_text = prompt.instruction_text

        # Step 1: Analyze the task
        analysis = self.analyze(raw_prompt=raw_text)
        task_name = analysis.task_name.strip()
        task_instruction = analysis.task_instruction.strip()

        # Step 2: Extract fields
        extraction = self.extract_fields(
            task_instruction=task_instruction,
            raw_prompt_context=raw_text,
        )

        try:
            fields = json.loads(extraction.fields_json)
        except json.JSONDecodeError:
            logger.warning("Failed to parse fields JSON, attempting fallback")
            fields = self._fallback_field_extraction(extraction.fields_json)

        # Step 3: Refine
        draft_fields_json = json.dumps(fields)
        refined = self.refine(
            draft_name=task_name,
            draft_instruction=task_instruction,
            draft_fields_json=draft_fields_json,
        )

        try:
            refined_fields = json.loads(refined.refined_fields_json)
        except json.JSONDecodeError:
            logger.warning("Failed to parse refined fields JSON, using draft")
            refined_fields = fields

        # Build the final spec
        inputs: list[FieldSpec] = []
        outputs: list[FieldSpec] = []

        for f in refined_fields:
            field_type = FieldType(f.get("field_type", "input").lower())
            spec = FieldSpec(
                name=f.get("name", "unnamed").strip(),
                description=f.get("description", "").strip(),
                suggested_type=f.get("type", "string").strip(),
                field_type=field_type,
            )
            if field_type == FieldType.INPUT:
                inputs.append(spec)
            else:
                outputs.append(spec)

        return SignatureSpec(
            name=refined.refined_name.strip(),
            instructions=refined.refined_instruction.strip(),
            inputs=inputs,
            outputs=outputs,
        )

    @staticmethod
    def _fallback_field_extraction(raw: str) -> list[dict[str, str]]:
        """Attempt to salvage malformed JSON by looking for structured patterns."""
        # Very naive fallback: look for key-value pairs

        fields = []
        # Try to find field definitions in the text
        pattern = re.compile(
            r'["\']?name["\']?\s*[:=]\s*["\'](\w+)["\'].*?'
            r'["\']?field_type["\']?\s*[:=]\s*["\'](\w+)["\']',
            re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(raw):
            name = match.group(1)
            field_type = match.group(2).lower()
            fields.append(
                {
                    "name": name,
                    "description": f"The {name} field",
                    "type": "string",
                    "field_type": field_type,
                }
            )

        if not fields:
            # Ultimate fallback: assume single text input and output
            fields = [
                {
                    "name": "input_text",
                    "description": "The input to process",
                    "type": "string",
                    "field_type": "input",
                },
                {
                    "name": "output_text",
                    "description": "The generated output",
                    "type": "string",
                    "field_type": "output",
                },
            ]

        return fields
