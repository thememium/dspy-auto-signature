"""DSPy module that generates SignatureSpec from ParsedPrompt."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

import dspy

from dspy_auto_signature.core.config import Config
from dspy_auto_signature.types.signature_spec import FieldSpec, FieldType, SignatureSpec

if TYPE_CHECKING:
    from dspy_auto_signature.types.signature_spec import ParsedPrompt

logger = logging.getLogger(__name__)


class AnalyzePrompt(dspy.Signature):
    """Analyze a raw prompt and extract the core task identity.

    You are decomposing a user's prompt into two elements:

    1. **task_name** — a PascalCase class name that a DSPy Signature would bear.
    2. **task_instruction** — a 1-3 sentence docstring that becomes the Signature's
       instruction, guiding the LLM's behaviour at inference time.

    ## Naming rules

    - Use PascalCase, 2-4 words. Prefer domain verbs: *Extract*, *Classify*,
      *Summarize*, *Translate*, *Generate*, *Analyze*, *Rank*, *Evaluate*.
    - Name the transformation, not the format: `ClassifySentiment` not `GetJSON`.
    - Keep it specific: `ExtractMedicalEntities` not `ProcessText`.

    Good names:
      ExtractMemoryOperations, ClassifySentiment, SummarizeArticle,
      TranslateToSQL, GenerateUnitTests, EvaluateCodeQuality,
      ReconcileMemory, RankSearchResults, AnalyzePullRequest

    Bad names:
      ProcessInput, HandleData, DoTask, TextToOutput, MainFunction

    ## Instruction guidelines

    - The instruction becomes the Signature's docstring — the LLM reads it every call.
    - 1-3 sentences. Be action-oriented and specific about the expected output.
    - For simple tasks (classify, extract): 1 sentence is enough.
    - For complex tasks (multi-step reasoning, analysis): 2-3 sentences describing
      the expected reasoning path and output format.
    - Do NOT repeat the field names or describe the I/O schema — that is handled
      separately by the field descriptions.

    Good instructions:
      - "Extract named entities from the given text, categorizing each as a
        person, organization, or location."
      - "Analyze the sentiment of customer reviews as positive, negative, or
        neutral, providing a confidence score."
      - "Given source code and a bug description, identify the root cause and
        suggest a fix with a brief explanation of the reasoning."
    """

    raw_prompt: str = dspy.InputField(desc="The raw prompt text to analyze")
    task_name: str = dspy.OutputField(
        desc=(
            "A PascalCase class name for the Signature (2-4 words, e.g. "
            "ClassifySentiment, ExtractEntities, AnalyzePullRequest)"
        )
    )
    task_instruction: str = dspy.OutputField(
        desc=(
            "A 1-3 sentence instruction that will become the Signature's docstring. "
            "Be specific about what the LLM should do and how to format the output."
        )
    )


class ExtractFields(dspy.Signature):
    """Extract input and output fields for a DSPy Signature from the raw prompt.

    You are analyzing a prompt to identify what data flows IN to the task and
    what data flows OUT. Output a JSON array of field objects.

    ## Guidelines

    **Naming — be semantic, not generic:**
    - GOOD: `article`, `source_code`, `bug_description`, `customer_review`,
      `medical_record`, `search_query`, `product_catalog`
    - BAD: `input_text`, `text`, `data`, `content`, `input`, `raw`
    - GOOD: `summary`, `sentiment_label`, `entities`, `risk_score`,
      `sql_query`, `suggested_fix`, `confidence`
    - BAD: `output`, `result`, `response`, `output_text`, `prediction`

    **Descriptions — 5-15 words telling the LLM what to put there:**
    - GOOD: "The full article text to summarize"
    - GOOD: "SQL query generated from the natural-language question"
    - GOOD: "List of extracted named entities with their types"
    - BAD: "The input" / "The output" / "A field"

    **Multiple inputs and outputs are common. Detect ALL of them:**
    - Code review prompt → inputs: `pull_request_description`, `diff`;
      outputs: `summary`, `risk_score`
    - Translation prompt → inputs: `source_text`, `target_language`;
      outputs: `translated_text`, `confidence`
    - Data analysis prompt → inputs: `dataset_description`, `question`;
      outputs: `answer`, `explanation`

    **Type vocabulary — use the most specific type:**
    - `string` — single text value (most common)
    - `integer` — whole number (e.g., count, score, ID)
    - `float` — decimal number (e.g., probability, rating)
    - `boolean` — true/false flag
    - `list of strings` — multiple text items (e.g., tags, keywords)
    - `list of dicts` — structured records (e.g., entities, operations)
    - `dict` — single structured object
    - `optional string` — text that may be absent
    - `string or null` — nullable text
    - `literal X, Y, Z` — constrained set of known values (see below)

    **When to use `literal` types — constrained categorical outputs:**
    - Use `literal X, Y, Z` (or `one of X, Y, Z`) when the output must be
      one of a small, known set of values.
    - Signals to look for:
      * The prompt lists specific options (e.g., "positive, negative, or neutral")
      * The task is classification with known categories
      * The output is a label, status, priority, or enum-like value
      * The prompt says "one of", "either", "among", "categorize as"
    - Examples:
      * Sentiment analysis → `literal positive, negative, neutral`
      * Priority classification → `literal critical, high, medium, low`
      * Boolean-like with 3+ options → `literal yes, no, maybe`
      * Status field → `literal pending, in_progress, completed, failed`
    - Do NOT use literal for free-text outputs, numeric ranges, or open-ended values.

    ## Example — complex signature with multiple I/O

    Prompt: "Given a pull request description and diff, provide a summary of
    changes, list potential bugs, and assign a risk score from 1-10.
    Classify the overall risk as low, medium, or high."

    Output:
    [
      {"name": "pr_description", "description": "The pull request description text", "type": "string", "field_type": "input"},
      {"name": "diff", "description": "The full code diff of the pull request", "type": "string", "field_type": "input"},
      {"name": "summary", "description": "Concise summary of the changes made", "type": "string", "field_type": "output"},
      {"name": "potential_bugs", "description": "List of potential bugs or issues found", "type": "list of strings", "field_type": "output"},
      {"name": "risk_score", "description": "Risk score from 1 to 10 indicating severity", "type": "integer", "field_type": "output"},
      {"name": "risk_level", "description": "Overall risk classification", "type": "literal low, medium, high", "field_type": "output"}
    ]
    """

    raw_prompt: str = dspy.InputField(desc="The raw prompt text to analyze")
    fields_json: str = dspy.OutputField(
        desc=(
            "JSON array of field objects. Each object has: "
            "name (snake_case, semantic — e.g. 'article' not 'input_text'), "
            "description (5-15 words telling the LLM what to put there), "
            "type (one of: string, integer, float, boolean, list of strings, "
            "list of dicts, dict, optional string, string or null, "
            "or 'literal X, Y, Z' for constrained categorical outputs), "
            "field_type ('input' or 'output'). "
            "Detect ALL inputs and ALL outputs — most tasks have multiple. "
            "Use 'literal X, Y, Z' when the output must be one of a small "
            "known set of values (e.g., sentiment labels, priority levels, "
            "status categories)."
        ),
    )


class RefineSignature(dspy.Signature):
    """Refine a draft Signature specification for maximum clarity and quality.

    You are reviewing a draft Signature (name, instruction, fields) and
    improving it. Apply these refinement criteria:

    ## Name refinement
    - Ensure PascalCase, 2-4 words, starts with a domain verb.
    - BAD → GOOD: `TextProcessor` → `ExtractEntities`,
      `Analyze` → `ClassifySentiment`, `HandleInput` → `ParseContractClauses`

    ## Instruction refinement
    - Ensure 1-3 sentences, action-oriented, specific about output format.
    - Remove filler ("Given the following...", "Your task is to...").
    - Add specificity if the draft is vague.
    - BAD → GOOD:
      "Analyze the text and provide results."
        → "Extract named entities from the text, categorizing each as
           person, organization, or location with confidence scores."

    ## Field refinement
    - Names: ensure semantic, snake_case, specific (not `input_text`).
    - Descriptions: ensure 5-15 words, tell the LLM what to put there.
    - Types: ensure most specific type is used (not always `string`).
    - **Literal types**: if the output is a constrained set of known values
      (labels, categories, statuses, priority levels), use `literal X, Y, Z`
      instead of `string`. Look for signals like "one of", "either", "among",
      "categorize as", or a list of specific options in the prompt.
    - Check that ALL necessary inputs and outputs are present.
    - BAD → GOOD:
      name: `text` → `source_text`
      description: "The input" → "The original article to be summarized"
      type: `string` → `list of dicts` (when output is structured)
      type: `string` → `literal positive, negative, neutral` (for classification)

    ## Before / After example

    Before:
      name: "ProcessText", instruction: "Process the text.", fields:
      [{"name": "input_text", "description": "The text", "type": "string", "field_type": "input"},
       {"name": "output", "description": "The result", "type": "string", "field_type": "output"}]

    After:
      name: "ExtractKeyPhrases", instruction: "Extract the top key phrases
      from the text, ranked by relevance.", fields:
      [{"name": "source_text", "description": "The text to extract key phrases from",
        "type": "string", "field_type": "input"},
       {"name": "key_phrases", "description": "Top key phrases ranked by relevance",
        "type": "list of strings", "field_type": "output"},
       {"name": "relevance_scores", "description": "Relevance score for each phrase, 0.0 to 1.0",
        "type": "list of floats", "field_type": "output"},
       {"name": "difficulty", "description": "Difficulty level of the text",
        "type": "literal easy, medium, hard", "field_type": "output"}]
    """

    draft_name: str = dspy.InputField(desc="Current task name")
    draft_instruction: str = dspy.InputField(desc="Current task instruction")
    draft_fields_json: str = dspy.InputField(desc="Current fields as JSON array")
    refined_name: str = dspy.OutputField(
        desc=(
            "Improved PascalCase name — semantic, specific, 2-4 words "
            "starting with a domain verb (e.g. ClassifySentiment, ExtractEntities)"
        )
    )
    refined_instruction: str = dspy.OutputField(
        desc=(
            "Improved 1-3 sentence instruction — action-oriented, specific "
            "about the expected output format, no filler phrases"
        )
    )
    refined_fields_json: str = dspy.OutputField(
        desc=(
            "Improved fields JSON array — semantic snake_case names, "
            "5-15 word descriptions, most specific types, all necessary "
            "inputs and outputs present"
        )
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
        self.extract_fields = dspy.Predict(ExtractFields)
        self.refine = dspy.Predict(RefineSignature)
        self.parallel = dspy.Parallel(num_threads=2)

    def forward(self, prompt: ParsedPrompt) -> SignatureSpec:
        """Generate a :class:`SignatureSpec` from a parsed prompt.

        Uses the LM configured via :func:`~dspy_auto_signature.configure`
        (or the global DSPy default) in an isolated context so that the
        meta-program does not pollute the user's runtime DSPy settings.

        Args:
            prompt: A normalised :class:`ParsedPrompt`.

        Returns:
            A fully-populated :class:`SignatureSpec`.

        """
        raw_text = prompt.instruction_text
        lm = Config.get_lm()

        with dspy.settings.context(lm=lm):
            # Step 1 & 2: Analyze and Extract Fields in parallel
            # Both operate on raw_prompt, so they can run concurrently
            results = self.parallel(
                [
                    (self.analyze, {"raw_prompt": raw_text}),
                    (self.extract_fields, {"raw_prompt": raw_text}),
                ]
            )

            analysis = results[0]
            extraction = results[1]

            task_name = analysis.task_name.strip()
            task_instruction = analysis.task_instruction.strip()

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
            name = f.get("name", "unnamed").strip()
            description = f.get("description", "").strip()
            if not description:
                description = f"The {name} field"
            spec = FieldSpec(
                name=name,
                description=description,
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
