"""Public API for dspy-auto-signature."""

from __future__ import annotations

import keyword
import logging
from typing import Any, cast

import dspy

from dspy_auto_signature.core.config import Config
from dspy_auto_signature.core.signature_builder import (
    GeneratedSignature,
    SignatureBuilder,
)
from dspy_auto_signature.generator.rlm_signature_generator import RLMSignatureGenerator
from dspy_auto_signature.parser import AutoParser, DataFrameParser
from dspy_auto_signature.types.signature_spec import SignatureSpec

__all__ = [
    "generate",
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
    return generate(
        prompt,
        input_hints=input_hints,
        output_hints=output_hints,
    )


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
    if not DataFrameParser().can_parse(data):
        raise TypeError(
            f"from_dataset() cannot handle input of type {type(data).__name__}. "
            "Use generate() for automatic source detection."
        )
    return generate(
        data,
        task_hint=task_hint,
        input_hints=input_hints,
        output_hints=output_hints,
    )


def generate(
    source: Any,
    task_hint: str | None = None,
    *,
    input_hints: dict[str, str] | None = None,
    output_hints: dict[str, str] | None = None,
) -> GeneratedSignature:
    """Generate a DSPy Signature from prompt material or tabular data.

    This is the simplest entry point. The parser automatically distinguishes
    prompts, message arrays, datasets, and registered custom source types.

    Args:
        source: Prompt material or tabular data accepted by the parser layer.
        task_hint: Optional task description. Most useful for datasets, where
            it identifies which columns should be predicted.
        input_hints: Field names mapped to improved input descriptions.
        output_hints: Field names mapped to improved output descriptions.

    Returns:
        A fresh ``dspy.Signature`` subclass.

    """
    logger.debug("generate called with input type: %s", type(source).__name__)
    parsed = AutoParser.parse(source)
    if task_hint:
        parsed = parsed.model_copy(
            update={
                "instruction_text": f"{parsed.instruction_text}\n\nTask: {task_hint}"
            }
        )

    generator = RLMSignatureGenerator(sub_lm=Config.get_sub_lm())
    spec = cast(SignatureSpec, generator(parsed))
    spec = _apply_hints(spec, input_hints, output_hints)
    signature = SignatureBuilder.build(spec)

    logger.info(
        "Generated signature '%s' (%d inputs, %d outputs)",
        spec.name,
        len(spec.inputs),
        len(spec.outputs),
    )
    return signature


def _apply_hints(
    spec: SignatureSpec,
    input_hints: dict[str, str] | None,
    output_hints: dict[str, str] | None,
) -> SignatureSpec:
    """Merge user hints into a copied ``SignatureSpec``."""
    from dspy_auto_signature.types.signature_spec import FieldSpec, FieldType

    spec = spec.model_copy(deep=True)
    used = {field.name for field in spec.all_fields}
    for hints, target_list, field_type in (
        (input_hints, spec.inputs, FieldType.INPUT),
        (output_hints, spec.outputs, FieldType.OUTPUT),
    ):
        existing = {field.name: field for field in target_list}
        for name, description in (hints or {}).items():
            if not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError(
                    f"Hint field name must be a valid identifier: {name!r}"
                )
            if not description.strip():
                raise ValueError(f"Hint description for {name!r} cannot be empty")
            if name in existing:
                existing[name].description = description.strip()
            elif name in used:
                raise ValueError(f"Hint field {name!r} conflicts with another field")
            else:
                target_list.append(
                    FieldSpec(
                        name=name,
                        description=description.strip(),
                        field_type=field_type,
                    ),
                )
                used.add(name)

    return spec
