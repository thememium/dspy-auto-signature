"""Public API for dspy-auto-signature."""

from __future__ import annotations

import logging
from typing import Any, cast

import dspy

from dspy_auto_signature.core.config import Config
from dspy_auto_signature.core.signature_builder import SignatureBuilder
from dspy_auto_signature.generator.signature_generator import SignatureGenerator
from dspy_auto_signature.parser import AutoParser
from dspy_auto_signature.types.signature_spec import SignatureSpec

__all__ = [
    "from_prompt",
    "configure",
    "SignatureSpec",
]

logger = logging.getLogger(__name__)


def configure(lm: dspy.LM | None = None) -> None:
    """Configure the language model used for signature generation.

    If not called, the package will use whatever LM is globally configured
    via ``dspy.configure(lm=...)``.

    Args:
        lm: A ``dspy.LM`` instance, or ``None`` to use the global default.

    Example:
        >>> import dspy_auto_signature as das
        >>> import dspy
        >>> das.configure(lm=dspy.LM("openai/gpt-4o"))

    """
    Config.configure(lm=lm)


def from_prompt(
    prompt: str | list[dict[str, str]] | Any,
    *,
    input_hints: dict[str, str] | None = None,
    output_hints: dict[str, str] | None = None,
) -> type[dspy.Signature]:
    """Generate a DSPy Signature class from an arbitrary prompt.

    Accepts raw strings, Vercel AI SDK message arrays, or any combination.
    Uses a DSPy meta-program internally to analyse the prompt and infer
    fields, types, and instructions.

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
    generator = SignatureGenerator()
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
