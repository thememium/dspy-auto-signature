"""DSPy Signature for unified RLM signature generation."""

from __future__ import annotations

from typing import Any

import dspy
from pydantic import BaseModel, Field


class ProposedField(BaseModel):
    """Normalized field proposal produced by the RLM."""

    name: str = Field(description="Semantic snake_case field name")
    description: str = Field(description="Specific description of the field's value")
    type: str = Field(
        default="string",
        description="Natural-language type such as string, integer, or list of strings",
    )


class ProposedSignature(BaseModel):
    """Normalized complete signature proposal produced by the RLM."""

    name: str = Field(description="Specific PascalCase Signature class name")
    instructions: str = Field(description="Specific task doctrine and instructions")
    inputs: list[ProposedField] = Field(description="All required input fields")
    outputs: list[ProposedField] = Field(description="All required output fields")


class GenerateSignature(dspy.Signature):
    """Think deeply about the supplied task context and design one DSPy Signature.

    You are the sole signature architect. Use the recursive environment to inspect
    all available context before deciding the task doctrine, inputs, outputs, field
    names, descriptions, and types. Do not delegate these decisions to a later
    workflow and do not stop after a superficial reading.

    ## Required analysis

    1. Determine the actual transformation the runtime model must perform.
    2. Distinguish information available at runtime from values the model must create.
    3. For datasets, inspect profiles and sample rows together. Use the task hint to
       identify targets; do not classify columns using cardinality alone.
    4. For prompts, inspect instructions, examples, placeholders, requested formats,
       constraints, and implied outputs.
       Preserve explicitly named runtime inputs exactly. For example, if the prompt
       names ``message``, ``category``, and ``priority``, use those names rather than
       inventing ``ticket_message``, ``ticket_category``, or ``ticket_priority``.
       Placeholder names such as ``{article}`` are authoritative input names.
    5. Include every necessary input and output, but do not expose internal reasoning
       steps as fields unless the task explicitly requests them.
    6. Use semantic names. Never use generic placeholders such as ``input_text``,
       ``output_text``, ``data``, ``result``, or ``AutoSignature``.
    7. Write useful doctrine: specific instructions describing the task, constraints,
       and expected output behavior.
    8. Use the most specific practical types, including literal types for known
       categorical outputs.
       Express literal types as ``literal low, medium, high`` without JSON brackets.

    ## Final submission

    Call ``FINAL(draft=...)`` exactly once after completing the analysis. ``draft``
    must represent the complete signature with:

    - ``name``: specific PascalCase class name
    - ``instructions``: specific task doctrine
    - ``inputs``: field objects containing name, description, and type
    - ``outputs``: field objects containing name, description, and type

    The final draft may be a dictionary or equivalent structured object. Do not
    serialize it into a JSON string.
    """

    source_kind: str = dspy.InputField(desc="Source kind: prompt or dataset")
    task_context: str = dspy.InputField(
        desc="Normalized prompt instructions and optional explicit task hint"
    )
    examples_json: str = dspy.InputField(
        desc="JSON examples extracted from the prompt, or an empty array"
    )
    data_profile_json: str = dspy.InputField(
        desc="JSON dataset profile, or an empty object for prompt sources"
    )
    sample_rows_json: str = dspy.InputField(
        desc="JSON representative dataset rows, or an empty array for prompt sources"
    )
    draft: Any = dspy.OutputField(
        desc="Complete proposed Signature containing name, instructions, inputs, outputs"
    )
