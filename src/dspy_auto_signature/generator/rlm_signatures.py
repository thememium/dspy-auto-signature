"""DSPy Signature for the RLM-based slow-path signature generator."""

from __future__ import annotations

import dspy


class ProposeSignatureFromData(dspy.Signature):
    """Propose a DSPy Signature by analyzing a dataset's column profiles and sample rows.

    Given a structured profile of each column (dtype, null rate, cardinality,
    sample values, dtype-specific stats) and a few representative rows, decide:
    - The PascalCase class name for the signature
    - A clear, concise instruction string (the docstring)
    - Which columns are INPUTS vs OUTPUTS
    - The natural-language type for each field
    - A clear description for each field

    Rules:
    - Low-cardinality categorical columns are usually OUTPUTS
    - High-cardinality text/numeric columns are usually INPUTS
    - The instruction should be 1-2 sentences explaining the task
    - Field names should be snake_case
    - Descriptions should be 5-15 words and tell the LLM what to put there
    """

    data_profile_json: str = dspy.InputField(
        desc=(
            "JSON of column profiles: name, dtype, n_unique, null_rate, "
            "sample_values, and dtype-specific stats"
        )
    )
    sample_rows_json: str = dspy.InputField(
        desc="JSON of 3-5 representative rows from the dataset"
    )
    task_hint: str = dspy.InputField(
        desc="Optional natural-language hint about what the task is. Empty string if not provided."
    )
    spec_json: str = dspy.OutputField(
        desc=(
            "JSON object with: name (PascalCase string), instructions (1-2 sentence "
            "string), inputs (array of {name, description, type} where type is "
            'natural-language like "string", "integer", "list of strings"), '
            "outputs (same shape as inputs). "
            'Example: {"name": "TicketClassifier", "instructions": "Classify support '
            'tickets.", "inputs": [{"name": "message", "description": "The support '
            'ticket text", "type": "string"}], "outputs": [{"name": "urgency", '
            '"description": "Urgency level", "type": "string"}]}'
        )
    )
