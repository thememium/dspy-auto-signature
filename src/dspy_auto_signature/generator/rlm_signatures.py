"""DSPy Signature for unified RLM signature generation."""

from __future__ import annotations

import dspy
from pydantic import BaseModel, Field

from dspy_auto_signature.types.signature_spec import PydanticModelSchema


class ProposedField(BaseModel):
    """Normalized field proposal produced by the RLM."""

    name: str = Field(description="Semantic snake_case field name")
    description: str = Field(description="Specific description of the field's value")
    type: str = Field(
        default="string",
        description=(
            "Natural-language type such as string, integer, or list of strings. "
            "Use 'pydantic' for structured outputs and provide pydantic_model."
        ),
    )
    literal_values: list[str | int] | None = Field(
        default=None,
        description="Allowed values when the type is a Literal (enumerated output)",
    )
    pydantic_model: PydanticModelSchema | None = Field(
        default=None,
        description=(
            "Complete nested Pydantic model schema (model_name, description, "
            "typed fields) when type is 'pydantic'"
        ),
    )


class ProposedSignature(BaseModel):
    """Normalized complete signature proposal produced by the RLM."""

    name: str = Field(
        description="Specific PascalCase Signature class name ending with 'Signature'"
    )
    instructions: str = Field(
        description=(
            "Rewritten, generalized task doctrine for the runtime model. "
            "Never a verbatim copy of the source prompt: no placeholder markers, "
            "JSON format templates, section headers, examples, or separators."
        )
    )
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
    6. Use semantic names. Never use generic placeholders such as ``input``,
       ``output``, ``input_text``, ``output_text``, ``data``, ``result``, or
       ``AutoSignature``. The ``input`` and ``output`` keys in example dicts are
       structural labels, not field name suggestions.
   7. Write the ``instructions`` as your own concise task doctrine: what the
      runtime model must do, the key constraints, and the expected output
      behavior. NEVER copy the source prompt verbatim into ``instructions``.
      The prompt's literal scaffolding — ``{}`` and ``{placeholder}`` markers,
      ``**Section**`` headers, JSON format templates, "answer in this exact
      format" blocks, example payloads, and separators such as ``---`` — must
      NOT appear in ``instructions``. DSPy renders inputs and outputs
      automatically as typed fields, so formatting directives and field
      listings in the prompt are redundant. Express each placeholder or
      template slot as an input or output field (rules 4 and 9), and keep
      ``instructions`` abstract enough to hold for any runtime values.
    8. Use the most specific practical types, including literal types for known
       categorical outputs.
       Express literal types as ``literal low, medium, high`` without JSON brackets,
       or set ``type`` to ``Literal`` with an explicit ``literal_values`` list.
    9. For structured outputs with multiple related fields or nested objects, do not
       use a plain ``str`` type with JSON instructions. Set ``type`` to ``pydantic``
       and provide a complete ``pydantic_model`` schema: ``model_name`` (PascalCase),
       optional ``description``, and typed ``fields`` where each field has a
       ``name``, a concrete ``type`` (``str``, ``int``, ``float``, ``bool``,
       ``list[str]``, ``dict[str, str]``, ``Literal``, ``pydantic`` for nesting),
       a ``description``, a ``required`` flag, ``literal_values`` for Literal
       fields, and ``nested_model`` for nested objects. Never describe a structured
       output as a JSON string.
    10. Simple scalar outputs (a single answer, score, or label) stay as plain types;
        reserve Pydantic models for well-defined multi-field structures.

    ## Final submission

    Call ``FINAL(draft=...)`` exactly once after completing the analysis. ``draft``
    must represent the complete signature with:

    - ``name``: specific PascalCase class name ending with ``Signature``
      (for example ``TicketClassificationSignature``)
    - ``instructions``: rewritten task doctrine, never a verbatim copy of the
      source prompt
    - ``inputs``: field objects containing name, description, and type
    - ``outputs``: field objects containing name, description, and type (plus
      ``literal_values`` for enumerated outputs and ``pydantic_model`` for
      structured outputs)

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
    draft: ProposedSignature = dspy.OutputField(
        desc="Complete proposed Signature containing name, instructions, inputs, outputs"
    )


class GenerateSDKSignature(dspy.Signature):
    """Analyze SDK message arrays and design a DSPy Signature from their semantic content.

    You are a signature architect specializing in LLM SDK message formats. You receive
    raw message arrays from OpenAI, Anthropic, Google Gemini, LiteLLM, or similar SDKs.

    ## CRITICAL: Understand the SDK structure

    These messages use ``role`` and ``content`` (or ``parts``) as STRUCTURAL metadata,
    NOT as actual input/output fields for the DSPy signature. You must NEVER create
    fields named ``role``, ``content``, ``parts``, or ``messages``.

    ### Role semantics

    - ``system`` / ``developer`` → The system prompt, which becomes the signature's
      docstring / instructions. Extract the task description, constraints, and behavior.
    - ``user`` → Contains the actual task request. Analyze the CONTENT to infer what
      information the user provides (inputs) and what they ask for (implied outputs).
    - ``assistant`` / ``model`` → Shows the expected response format. Analyze the
      CONTENT to understand what outputs the model should produce.
    - ``tool`` / ``function`` → Additional context; append to instructions.

    ### Content analysis rules

    1. **Disregard the key names** (role, content, parts). Focus entirely on the
       semantic meaning of the message content.
    2. **Infer inputs from user messages**: What information does the user provide?
       Look for placeholders like ``{article}``, ``{code}``, explicit parameters,
       or described inputs.
    3. **Infer outputs from assistant messages**: What does the model produce?
       Look for structured data, classifications, summaries, generated text, etc.
    4. **If the user message describes a task without clear inputs**, the input might
       be implicit (e.g., ``query``, ``text``, ``article`` based on context).
    5. **If assistant messages show structured output** (JSON, lists, classifications),
       infer the appropriate output fields and types.

    ## Forbidden fields

    NEVER generate these field names — they are SDK structural metadata, not semantic
    fields derived from message content:

    - Structural keys: ``role``, ``content``, ``parts``, ``messages``
    - Role values: ``system``, ``developer``, ``user``, ``assistant``, ``model``, ``tool``
    - Generic placeholders: ``input``, ``output``, ``text``, ``data``, ``result``

    Field names MUST describe the semantic content (e.g., ``article``, ``summary``,
    ``classification``), never the SDK envelope that carried the content.

    ## Required analysis

    1. Read all system messages to understand the task doctrine and constraints.
    2. Read all user messages to identify what inputs are provided and what is requested.
    3. Read all assistant messages to understand the expected output format and fields.
    4. Combine these insights into a coherent signature.
    5. Use semantic, specific field names. ``article`` is better than ``input_text``.
   6. Write the ``instructions`` as your own concise task doctrine. NEVER paste
      message content verbatim into ``instructions``: no JSON format templates,
      "answer in this exact format" blocks, placeholder markers, or separators.
      Those become input and output fields; ``instructions`` states what the
      runtime model must do and holds for any runtime values.
    7. Use specific types including literals for categorical outputs. For structured
       assistant outputs (multi-field JSON objects, nested records), set ``type`` to
       ``pydantic`` with a complete ``pydantic_model`` schema instead of a plain
       ``str`` described as JSON; use ``literal_values`` for enumerated outputs.

    ## Final submission

    - ``instructions``: rewritten task doctrine derived from system + user
      context, never a verbatim copy of message content

    - ``name``: specific PascalCase class name ending with ``Signature``
      (for example ``TicketClassificationSignature``)
    - ``instructions``: complete task doctrine derived from system + user context
    - ``inputs``: field objects with name, description, and type
    - ``outputs``: field objects with name, description, and type (plus
      ``literal_values`` for enumerated outputs and ``pydantic_model`` for
      structured outputs)
    """

    sdk_format: str = dspy.InputField(
        desc="Detected SDK format: openai, anthropic, gemini, litellm, or generic"
    )
    messages_json: str = dspy.InputField(
        desc="JSON array of raw SDK messages with role, content, parts, etc."
    )
    task_hint: str = dspy.InputField(
        desc="Optional additional task context from the user", default=""
    )
    draft: ProposedSignature = dspy.OutputField(
        desc="Complete proposed Signature containing name, instructions, inputs, outputs"
    )
