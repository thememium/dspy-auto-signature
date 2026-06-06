"""Unified RLM-based signature generator."""

from __future__ import annotations

import json
import keyword
import logging
import re
from typing import TYPE_CHECKING, Any

import dspy
from pydantic import BaseModel

from dspy_auto_signature.core.config import Config
from dspy_auto_signature.generator.rlm_signatures import (
    GenerateSignature,
    ProposedField,
)
from dspy_auto_signature.types.signature_spec import FieldSpec, FieldType, SignatureSpec

if TYPE_CHECKING:
    from dspy_auto_signature.types.signature_spec import ParsedPrompt

logger = logging.getLogger(__name__)


class RLMSignatureGenerator(dspy.Module):
    """Generate a ``SignatureSpec`` through one recursive analysis workflow."""

    def __init__(
        self,
        max_iterations: int = 20,
        max_llm_calls: int = 50,
        sub_lm: dspy.LM | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.rlm = dspy.RLM(
            GenerateSignature,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            sub_lm=sub_lm,
            verbose=verbose,
        )

    def forward(self, prompt: ParsedPrompt) -> SignatureSpec:
        """Build unified context, run one RLM, and normalize its complete draft."""
        context = self._build_context(prompt)
        lm = (
            Config.get_dataset_lm()
            if context["source_kind"] == "dataset"
            else Config.get_lm()
        )

        try:
            with dspy.settings.context(lm=lm):
                result = self.rlm(**context)
            return self._draft_to_spec(result.draft)
        except Exception as exc:
            logger.warning(
                "Unified RLM signature generation failed; using grounded fallback: %s",
                exc,
            )
            return self._fallback_from_context(context)

    @classmethod
    def _build_context(cls, prompt: ParsedPrompt) -> dict[str, str]:
        """Represent every supported source using one RLM input contract."""
        if not cls._is_dataset(prompt):
            return {
                "source_kind": "prompt",
                "task_context": prompt.instruction_text,
                "examples_json": json.dumps(prompt.examples, indent=2, default=str),
                "data_profile_json": "{}",
                "sample_rows_json": "[]",
            }

        from dspy_auto_signature.data.profiler import profile_columns
        from dspy_auto_signature.data.to_records import to_records

        rows = to_records(prompt.raw_input)
        profile = profile_columns(rows)
        return {
            "source_kind": "dataset",
            "task_context": prompt.instruction_text,
            "examples_json": json.dumps(prompt.examples, indent=2, default=str),
            "data_profile_json": json.dumps(profile, indent=2, default=str),
            "sample_rows_json": json.dumps(rows[:5], indent=2, default=str),
        }

    @staticmethod
    def _is_dataset(prompt: ParsedPrompt) -> bool:
        """Return whether the parsed source contains tabular data."""
        raw = prompt.raw_input
        if raw is None:
            return False
        if isinstance(raw, dict):
            return True
        if isinstance(raw, list):
            return bool(raw) and not all(
                isinstance(item, dict) and {"role", "content"}.issubset(item)
                for item in raw
            )
        return type(raw).__name__ in ("DataFrame", "LazyFrame", "Example")

    @classmethod
    def _draft_to_spec(cls, raw_draft: Any) -> SignatureSpec:
        """Normalize one complete RLM draft into a buildable SignatureSpec."""
        draft = cls._as_mapping(raw_draft)
        if not draft:
            raise ValueError("RLM returned no complete signature draft")

        name = cls._normalize_class_name(
            cls._coerce_text(draft.get("name")) or "GeneratedSignature"
        )
        instructions = cls._coerce_text(
            draft.get("instructions")
            or draft.get("task_instructions")
            or draft.get("doctrine")
        )
        inputs = cls._convert_fields(draft.get("inputs"), FieldType.INPUT)
        outputs = cls._convert_fields(draft.get("outputs"), FieldType.OUTPUT)
        if not instructions or not inputs or not outputs:
            raise ValueError("RLM returned an incomplete signature draft")

        used = {field.name for field in inputs}
        for output in outputs:
            if output.name in used:
                output.name = cls._unique_name(f"{output.name}_result", used)
            used.add(output.name)

        spec = SignatureSpec(
            name=name,
            instructions=instructions,
            inputs=inputs,
            outputs=outputs,
        )
        if cls._is_placeholder_spec(spec):
            raise ValueError("RLM returned a generic placeholder draft")
        return spec

    @classmethod
    def _convert_fields(cls, raw_fields: Any, field_type: FieldType) -> list[FieldSpec]:
        """Normalize every usable field in a complete draft."""
        raw_fields = cls._as_sequence(raw_fields)
        fields: list[FieldSpec] = []
        used: set[str] = set()
        for raw_field in raw_fields:
            proposed = cls._normalize_proposed_field(raw_field)
            if proposed is None:
                continue
            name = cls._unique_name(cls._normalize_field_name(proposed.name), used)
            used.add(name)
            fields.append(
                FieldSpec(
                    name=name,
                    description=proposed.description.strip(),
                    suggested_type=proposed.type.strip() or "string",
                    field_type=field_type,
                )
            )
        return fields

    @classmethod
    def _normalize_proposed_field(cls, raw_field: Any) -> ProposedField | None:
        """Normalize common field representations from one RLM draft."""
        field = cls._as_mapping(raw_field)
        if not field:
            if isinstance(raw_field, str) and raw_field.strip():
                field = {"name": raw_field}
            else:
                return None

        name = cls._coerce_text(field.get("name") or field.get("field_name"))
        if not name:
            return None
        description = (
            cls._coerce_text(
                field.get("description")
                or field.get("desc")
                or field.get("field_description")
            )
            or f"The {name.replace('_', ' ')} value"
        )
        suggested_type = (
            cls._coerce_text(
                field.get("type")
                or field.get("suggested_type")
                or field.get("data_type")
            )
            or "string"
        )
        return ProposedField(name=name, description=description, type=suggested_type)

    @staticmethod
    def _as_mapping(value: Any) -> dict[str, Any]:
        """Coerce a structured value into a mapping when possible."""
        if isinstance(value, BaseModel):
            value = value.model_dump()
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return {}
        return value if isinstance(value, dict) else {}

    @classmethod
    def _as_sequence(cls, value: Any) -> list[Any]:
        """Coerce field collections into a list."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return [value]
        if isinstance(value, (dict, BaseModel)):
            return [value]
        if isinstance(value, tuple):
            return list(value)
        return value if isinstance(value, list) else []

    @staticmethod
    def _coerce_text(value: Any) -> str:
        """Return clean text without creating a separate required-text workflow."""
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _normalize_field_name(name: str) -> str:
        """Return a valid, non-reserved snake_case Python identifier."""
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
        if not normalized:
            normalized = "field"
        if normalized[0].isdigit():
            normalized = f"field_{normalized}"
        if keyword.iskeyword(normalized):
            normalized = f"{normalized}_value"
        return normalized

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        """Return a valid PascalCase signature class name."""
        words = re.findall(r"[a-zA-Z0-9]+", name)
        normalized = "".join(word[:1].upper() + word[1:] for word in words)
        if not normalized:
            return "GeneratedSignature"
        return f"Task{normalized}" if normalized[0].isdigit() else normalized

    @staticmethod
    def _unique_name(name: str, used: set[str]) -> str:
        """Return a field name that does not collide with existing fields."""
        candidate = name
        suffix = 2
        while candidate in used:
            candidate = f"{name}_{suffix}"
            suffix += 1
        return candidate

    @classmethod
    def _fallback_from_context(cls, context: dict[str, str]) -> SignatureSpec:
        """Create a grounded fallback only when the single RLM cannot finish."""
        if context["source_kind"] == "dataset":
            profile = json.loads(context["data_profile_json"])
            return cls._fallback_from_dataset(profile, context["task_context"])
        return cls._fallback_from_prompt(context["task_context"])

    @classmethod
    def _fallback_from_dataset(
        cls,
        profile: dict[str, Any],
        task_context: str,
    ) -> SignatureSpec:
        """Build a useful signature deterministically from a dataset profile."""
        columns: dict[str, dict[str, Any]] = profile.get("columns", {})
        if not columns:
            return cls._fallback_from_prompt(task_context or "Analyze the dataset")

        hint = cls._extract_task_hint(task_context).lower()
        output_names = [
            name
            for name in columns
            if re.search(rf"\b{re.escape(name.lower().replace('_', ' '))}\b", hint)
        ]
        if not output_names:
            output_names = [
                name
                for name, info in columns.items()
                if cls._looks_like_target(name, info)
            ]
        if not output_names:
            output_names = [next(reversed(columns))]
        if len(output_names) == len(columns) and len(columns) > 1:
            output_names = output_names[1:]

        used: set[str] = set()
        inputs: list[FieldSpec] = []
        outputs: list[FieldSpec] = []
        for name, info in columns.items():
            role = FieldType.OUTPUT if name in output_names else FieldType.INPUT
            field = cls._field_from_profile(name, info, role)
            field.name = cls._unique_name(field.name, used)
            used.add(field.name)
            (outputs if role == FieldType.OUTPUT else inputs).append(field)

        if not inputs:
            first_output = outputs[0]
            inputs.append(
                FieldSpec(
                    name=cls._unique_name(f"{first_output.name}_context", used),
                    description=f"Context used to predict {first_output.name.replace('_', ' ')}",
                    suggested_type="string",
                    field_type=FieldType.INPUT,
                )
            )

        output_phrase = " and ".join(name.replace("_", " ") for name in output_names)
        task_hint = cls._extract_task_hint(task_context)
        return SignatureSpec(
            name=cls._normalize_class_name(task_hint or f"Predict {output_phrase}"),
            instructions=task_hint
            or f"Use the provided dataset fields to predict {output_phrase}.",
            inputs=inputs,
            outputs=outputs,
        )

    @classmethod
    def _fallback_from_prompt(cls, prompt: str) -> SignatureSpec:
        """Build a conservative semantic signature from raw prompt text."""
        text = prompt.strip() or "Produce the requested result."
        input_names = list(
            dict.fromkeys(re.findall(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", text))
        ) or [cls._infer_prompt_input_name(text)]
        output_name = cls._infer_prompt_output_name(text)
        used: set[str] = set()
        inputs: list[FieldSpec] = []
        for raw_name in input_names:
            name = cls._unique_name(cls._normalize_field_name(raw_name), used)
            used.add(name)
            inputs.append(
                FieldSpec(
                    name=name,
                    description=f"The {raw_name.replace('_', ' ')} provided for the task",
                    suggested_type="string",
                    field_type=FieldType.INPUT,
                )
            )
        output_name = cls._unique_name(output_name, used)
        return SignatureSpec(
            name=cls._normalize_class_name(f"{output_name} task"),
            instructions=text,
            inputs=inputs,
            outputs=[
                FieldSpec(
                    name=output_name,
                    description=f"The generated {output_name.replace('_', ' ')}",
                    suggested_type="string",
                    field_type=FieldType.OUTPUT,
                )
            ],
        )

    @staticmethod
    def _looks_like_target(name: str, info: dict[str, Any]) -> bool:
        target_terms = {
            "answer",
            "category",
            "class",
            "label",
            "output",
            "prediction",
            "result",
            "score",
            "sentiment",
            "status",
            "target",
            "urgency",
        }
        tokens = set(re.split(r"[^a-z0-9]+", name.lower()))
        return bool(tokens & target_terms) and info.get("n_unique", 100) < 50

    @classmethod
    def _field_from_profile(
        cls,
        name: str,
        info: dict[str, Any],
        field_type: FieldType,
    ) -> FieldSpec:
        dtype = info.get("dtype", "unknown")
        suggested_type = {
            "bool": "boolean",
            "int": "integer",
            "float": "float",
            "list": "list of strings",
        }.get(dtype, "string")
        top_values = info.get("top_values", [])
        if field_type == FieldType.OUTPUT and 1 < len(top_values) <= 10:
            values = [str(item["value"]) for item in top_values if "value" in item]
            if values:
                suggested_type = f"literal {', '.join(values)}"
        if float(info.get("null_rate", 0)) > 0.15 and not suggested_type.startswith(
            "literal "
        ):
            suggested_type = f"optional {suggested_type}"
        role = "Input" if field_type == FieldType.INPUT else "Predicted"
        return FieldSpec(
            name=cls._normalize_field_name(name),
            description=f"{role} {name.replace('_', ' ')} value from the dataset",
            suggested_type=suggested_type,
            field_type=field_type,
        )

    @staticmethod
    def _infer_prompt_input_name(prompt: str) -> str:
        lowered = prompt.lower()
        for term in (
            "article",
            "code",
            "document",
            "message",
            "question",
            "query",
            "review",
            "ticket",
            "text",
        ):
            if term in lowered:
                return term
        return "source_content"

    @staticmethod
    def _infer_prompt_output_name(prompt: str) -> str:
        lowered = prompt.lower()
        for signal, name in {
            "classif": "classification",
            "extract": "extracted_items",
            "generat": "generated_content",
            "rank": "ranked_results",
            "sentiment": "sentiment",
            "summar": "summary",
            "translat": "translation",
        }.items():
            if signal in lowered:
                return name
        return "task_result"

    @staticmethod
    def _is_placeholder_spec(spec: SignatureSpec) -> bool:
        instruction = spec.instructions.strip().lower().rstrip(".")
        return (
            spec.name.lower() == "autosignature"
            or instruction == "process the input and produce an output"
        )

    @staticmethod
    def _extract_task_hint(instruction_text: str) -> str:
        marker = "\n\nTask: "
        return (
            instruction_text.rsplit(marker, maxsplit=1)[1].strip()
            if marker in instruction_text
            else ""
        )
