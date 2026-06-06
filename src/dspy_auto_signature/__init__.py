"""Public API for dspy-auto-signature."""

from __future__ import annotations

import logging
from typing import Any, cast

import dspy

from dspy_auto_signature.core.config import Config
from dspy_auto_signature.core.signature_builder import (
    GeneratedSignature,
    SignatureBuilder,
)
from dspy_auto_signature.generator.rlm_signature_generator import RLMSignatureGenerator
from dspy_auto_signature.parser import AutoParser
from dspy_auto_signature.types.signature_spec import SignatureSpec

__all__ = [
    "from_prompt",
    "from_dataset",
    "configure",
    "SignatureSpec",
    "GeneratedSignature",
]

logger = logging.getLogger(__name__)


def configure(
    lm: dspy.LM | None = None,
    dataset_lm: dspy.LM | None = None,
    sub_lm: dspy.LM | None = None,
) -> None:
    """Configure the language model used for signature generation.

    If not called, the package will use whatever LM is globally configured
    via ``dspy.configure(lm=...)``.

    Args:
        lm: A ``dspy.LM`` instance used by prompt-driven signature generation
            and as the fallback for ``dataset_lm`` and ``sub_lm``.
        dataset_lm: Optional outer LM used when the unified RLM receives
            dataset context. Falls back to ``lm`` if unset.
        sub_lm: The cheap inner LM used by RLM for sub-queries. Falls back
            to ``lm`` if unset.

    Example:
        >>> import dspy_auto_signature as das
        >>> import dspy
        >>> das.configure(
        ...     lm=dspy.LM("openai/gpt-4o"),
        ...     sub_lm=dspy.LM("openai/gpt-4o-mini"),
        ... )

    """
    Config.configure(lm=lm, dataset_lm=dataset_lm, sub_lm=sub_lm)


def from_prompt(
    prompt: str | list[dict[str, str]] | Any,
    *,
    input_hints: dict[str, str] | None = None,
    output_hints: dict[str, str] | None = None,
) -> GeneratedSignature:
    """Generate a DSPy Signature class from an arbitrary prompt.

    Accepts raw strings, Vercel AI SDK message arrays, or any combination.
    Uses the same unified RLM architect as :func:`from_dataset` to inspect the
    complete task context and infer fields, types, and instructions.

    For data-grounded signatures from tabular inputs, use :func:`from_dataset`.

    Args:
        prompt: The prompt material. Can be:
            - A raw string (system prompt or task description)
            - A Vercel AI SDK message array: ``[{"role": "system", "content": "..."}, ...]``
            - Any combination the parser layer can normalise
        input_hints: Optional mapping of field-name → description for known inputs.
        output_hints: Optional mapping of field-name → description for known outputs.

    Returns:
        A fresh ``dspy.Signature`` subclass ready for use in ``dspy.Predict``,
        ``dspy.ChainOfThought``, etc.

    Raises:
        RuntimeError: If no language model is configured.

    Example:
        >>> import dspy_auto_signature as das
        >>> import dspy
        >>>
        >>> sig = das.from_prompt("Summarize the following article into 3 bullet points")
        >>> summarizer = dspy.ChainOfThought(sig)
        >>> result = summarizer(article="Long text here...")

    """
    logger.debug("from_prompt called with input type: %s", type(prompt).__name__)

    # 1. Parse heterogeneous input into a normalised form
    parsed = AutoParser.parse(prompt)

    # 2. Run the meta-generator (DSPy module)
    generator = RLMSignatureGenerator()
    spec = cast(SignatureSpec, generator(parsed))

    # 3. Apply any user-supplied hints
    if input_hints:
        spec = _apply_hints(spec, input_hints, is_input=True)
    if output_hints:
        spec = _apply_hints(spec, output_hints, is_input=False)

    # 4. Build the actual DSPy Signature class
    sig_class = SignatureBuilder.build(spec)

    logger.info(
        "Generated signature '%s' (%d inputs, %d outputs)",
        spec.name,
        len(spec.inputs),
        len(spec.outputs),
    )

    return sig_class


def from_dataset(
    data: Any,
    task_hint: str | None = None,
    *,
    input_hints: dict[str, str] | None = None,
    output_hints: dict[str, str] | None = None,
) -> GeneratedSignature:
    """Generate a DSPy Signature class from a tabular dataset.

    Profiles the dataset's columns (dtypes, null rates, cardinality, sample
    values, dtype-specific stats) and passes that complete context to the same
    unified :class:`dspy.RLM` architect used by :func:`from_prompt`.

    Requires **Deno** to be installed (RLM uses a Deno-sandboxed Pyodide
    REPL). See the README for install instructions.

    Args:
        data: The dataset. Accepts ``list[dict]``, ``pandas.DataFrame``,
            ``polars.DataFrame`` / ``polars.LazyFrame``, ``list[dspy.Example]``,
            a single ``dspy.Example``, or any object with ``.to_dicts()`` /
            ``.to_pandas()`` / ``.to_dict()`` methods.
        task_hint: Optional natural-language description of the task to
            bias the RLM.
        input_hints: Optional mapping of field-name → description for known inputs.
        output_hints: Optional mapping of field-name → description for known outputs.

    Returns:
        A fresh ``dspy.Signature`` subclass ready for use in ``dspy.Predict``,
        ``dspy.ChainOfThought``, etc.

    Raises:
        RuntimeError: If no language model is configured.
        TypeError: If *data* cannot be converted to tabular records.

    Example:
        >>> import pandas as pd
        >>> import dspy
        >>> import dspy_auto_signature as das
        >>>
        >>> das.configure(
        ...     lm=dspy.LM("openai/gpt-4o"),
        ...     sub_lm=dspy.LM("openai/gpt-4o-mini"),
        ... )
        >>>
        >>> df = pd.DataFrame({
        ...     "message": ["Server is on fire", "Please clean conf room B"],
        ...     "urgency": ["high", "low"],
        ...     "sentiment": ["negative", "neutral"],
        ... })
        >>> sig = das.from_dataset(df, task_hint="Classify support tickets")

    """
    logger.debug("from_dataset called with input type: %s", type(data).__name__)

    # Lazy imports keep dataset dependencies optional for prompt-only usage.
    from dspy_auto_signature.generator.rlm_signature_generator import (
        RLMSignatureGenerator,
    )
    from dspy_auto_signature.parser import DataFrameParser
    from dspy_auto_signature.types.signature_spec import ParsedPrompt

    if not DataFrameParser().can_parse(data):
        raise TypeError(
            f"from_dataset() cannot handle input of type {type(data).__name__}. "
            "Expected: list[dict], pandas DataFrame, polars DataFrame/LazyFrame, "
            "list[dspy.Example], or any object with .to_dicts()/.to_pandas()/.to_dict()."
        )
    parsed = DataFrameParser().parse(data)

    if task_hint:
        parsed = ParsedPrompt(
            instruction_text=f"{parsed.instruction_text}\n\nTask: {task_hint}",
            examples=parsed.examples,
            raw_input=parsed.raw_input,
        )

    lm = Config.get_dataset_lm()
    sub_lm = Config.get_sub_lm()
    generator = RLMSignatureGenerator(sub_lm=sub_lm)
    with dspy.settings.context(lm=lm):
        spec = cast(SignatureSpec, generator(parsed))

    if input_hints:
        spec = _apply_hints(spec, input_hints, is_input=True)
    if output_hints:
        spec = _apply_hints(spec, output_hints, is_input=False)

    sig_class = SignatureBuilder.build(spec)

    logger.info(
        "Generated signature '%s' (%d inputs, %d outputs) from dataset",
        spec.name,
        len(spec.inputs),
        len(spec.outputs),
    )

    return sig_class


def _apply_hints(
    spec: SignatureSpec,
    hints: dict[str, str],
    *,
    is_input: bool,
) -> SignatureSpec:
    """Merge user hints into a SignatureSpec."""
    from dspy_auto_signature.types.signature_spec import FieldSpec, FieldType

    target_list = spec.inputs if is_input else spec.outputs
    field_type = FieldType.INPUT if is_input else FieldType.OUTPUT

    # Build a lookup by name
    existing = {f.name: f for f in target_list}

    for name, description in hints.items():
        if name in existing:
            existing[name].description = description
        else:
            target_list.append(
                FieldSpec(
                    name=name,
                    description=description,
                    field_type=field_type,
                ),
            )

    return spec
