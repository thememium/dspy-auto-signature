"""Unified RLM-based signature generator."""

from __future__ import annotations

import json
import keyword
import logging
import re
import threading
from typing import TYPE_CHECKING, Any

import dspy
from dspy.primitives.python_interpreter import PythonInterpreter
from pydantic import BaseModel

from dspy_auto_signature.core.config import Config
from dspy_auto_signature.generator.rlm_signatures import (
    GenerateSDKSignature,
    GenerateSignature,
    ProposedField,
)
from dspy_auto_signature.types.signature_spec import FieldSpec, FieldType, SignatureSpec

if TYPE_CHECKING:
    from dspy_auto_signature.types.signature_spec import ParsedPrompt

logger = logging.getLogger(__name__)

_PLACEHOLDER_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")

_INTERPRETERS = threading.local()


def _thread_interpreter() -> PythonInterpreter:
    """Return the calling thread's warm code interpreter, creating it on first use.

    ``PythonInterpreter`` lazily spawns its Deno/Pyodide sandbox on first
    ``execute()`` and is single-threaded, so one interpreter is cached per
    thread and reused across RLM runs instead of paying sandbox startup per
    generation.
    """
    interpreter = getattr(_INTERPRETERS, "interpreter", None)
    if interpreter is None:
        interpreter = PythonInterpreter()
        _INTERPRETERS.interpreter = interpreter
    return interpreter


def close_interpreter() -> None:
    """Shut down this thread's warm interpreter, if one exists."""
    interpreter = getattr(_INTERPRETERS, "interpreter", None)
    if interpreter is not None:
        interpreter.shutdown()
        _INTERPRETERS.interpreter = None


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
        interpreter = _thread_interpreter()
        self.rlm = dspy.RLM(
            GenerateSignature,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            sub_lm=sub_lm,
            verbose=verbose,
            interpreter=interpreter,
        )
        self.sdk_rlm = dspy.RLM(
            GenerateSDKSignature,
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            sub_lm=sub_lm,
            verbose=verbose,
            interpreter=interpreter,
        )

    def forward(self, prompt: ParsedPrompt) -> SignatureSpec:
        """Prefer deterministic structure; run the RLM only when structure is thin."""
        if self._is_sdk_format(prompt):
            spec = self._structural_sdk_spec(prompt)
            if spec is not None:
                return spec
            return self._forward_sdk(prompt)

        context = self._build_context(prompt)
        if context["source_kind"] == "dataset":
            profile = json.loads(context["data_profile_json"])
            try:
                return self._fallback_from_dataset(profile, context["task_context"])
            except Exception as exc:
                logger.warning("Dataset structural generation failed: %s", exc)

        if _PLACEHOLDER_PATTERN.search(context["task_context"]):
            return self._fallback_from_prompt(context["task_context"])

        lm = Config.get_lm()
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

    def _forward_sdk(self, prompt: ParsedPrompt) -> SignatureSpec:
        context = self._build_sdk_context(prompt)
        lm = Config.get_lm()

        try:
            with dspy.settings.context(lm=lm):
                result = self.sdk_rlm(**context)
            spec = self._draft_to_spec(result.draft)
            return self._sanitize_sdk_spec(spec)
        except Exception as exc:
            logger.warning(
                "SDK RLM signature generation failed; using standard fallback: %s",
                exc,
            )
            return self._fallback_from_context(self._build_context(prompt))

    _SDK_USER_ROLES = frozenset({"user"})
    _SDK_SYSTEM_ROLES = frozenset({"system", "developer"})
    _SDK_ASSISTANT_ROLES = frozenset({"assistant", "model"})
    _SDK_CONTEXT_ROLES = frozenset({"tool", "function"})
    _NAME_STOPWORDS = frozenset(
        {
            "the",
            "and",
            "for",
            "with",
            "you",
            "your",
            "that",
            "this",
            "from",
            "into",
            "then",
            "when",
            "what",
            "which",
            "are",
            "was",
            "were",
            "have",
            "has",
            "will",
            "would",
            "should",
            "them",
            "their",
            "about",
            "after",
            "before",
            "each",
            "user",
            "message",
            "please",
            "can",
            "could",
        }
    )

    @classmethod
    def _structural_sdk_spec(cls, prompt: ParsedPrompt) -> SignatureSpec | None:
        """Build a ``SignatureSpec`` deterministically from SDK message structure.

        Returns ``None`` when the structure is too thin (no user message) and the
        RLM should take over. System/developer messages become instructions, user
        message structure (placeholders, JSON objects) becomes inputs, and
        assistant/model message structure (JSON objects) becomes typed outputs.
        """
        raw = prompt.raw_input
        messages: list[dict[str, Any]] = raw if isinstance(raw, list) else []

        system_parts: list[str] = []
        user_parts: list[str] = []
        assistant_parts: list[str] = []
        context_parts: list[str] = []
        for msg in messages:
            content = cls._message_text(msg)
            if not content:
                continue
            role = str(msg.get("role", "")).lower()
            if role in cls._SDK_SYSTEM_ROLES:
                system_parts.append(content)
            elif role in cls._SDK_USER_ROLES:
                user_parts.append(content)
            elif role in cls._SDK_ASSISTANT_ROLES:
                assistant_parts.append(content)
            elif role in cls._SDK_CONTEXT_ROLES:
                context_parts.append(content)

        if not user_parts:
            return None

        instructions = "\n\n".join(
            part for part in (*system_parts, *context_parts, *user_parts) if part
        )
        task_hint = cls._extract_task_hint(prompt.instruction_text)
        if task_hint:
            instructions = (
                f"{instructions}\n\nTask: {task_hint}" if instructions else task_hint
            )

        inputs = cls._inputs_from_user(user_parts)
        outputs = cls._outputs_from_assistant(assistant_parts)
        name = cls._sdk_class_name(user_parts[0], outputs)

        used = {field.name for field in inputs}
        outputs = [
            FieldSpec(
                name=cls._unique_name(field.name, used),
                description=field.description,
                suggested_type=field.suggested_type,
                field_type=FieldType.OUTPUT,
            )
            for field in outputs
        ]
        return SignatureSpec(
            name=name,
            instructions=instructions,
            inputs=inputs,
            outputs=outputs,
        )

    @staticmethod
    def _message_text(msg: dict[str, Any]) -> str:
        """Extract text from one SDK message, mirroring ``SDKParser`` semantics."""
        from dspy_auto_signature.parser.sdk_parser import SDKParser

        return SDKParser._get_content(msg) or ""

    @classmethod
    def _inputs_from_user(cls, user_parts: list[str]) -> list[FieldSpec]:
        """Derive inputs from user-message structure: placeholders, JSON, or the message itself."""
        names: list[str] = []
        for part in user_parts:
            names.extend(_PLACEHOLDER_PATTERN.findall(part))
        if names:
            used: set[str] = set()
            inputs: list[FieldSpec] = []
            for raw_name in names:
                name = cls._unique_name(cls._normalize_field_name(raw_name), used)
                used.add(name)
                inputs.append(
                    FieldSpec(
                        name=name,
                        description=f"The {raw_name.replace('_', ' ')} provided for the task",
                        suggested_type="string",
                        field_type=FieldType.INPUT,
                    ),
                )
            return inputs

        for part in user_parts:
            obj = cls._json_object(part)
            if obj:
                return [
                    FieldSpec(
                        name=cls._normalize_field_name(key),
                        description=f"The {key.replace('_', ' ')} value from the request payload",
                        suggested_type=cls._infer_value_type(value),
                        field_type=FieldType.INPUT,
                    )
                    for key, value in obj.items()
                ]

        return [
            FieldSpec(
                name="message",
                description="The user's initial message containing the task request",
                suggested_type="string",
                field_type=FieldType.INPUT,
            ),
        ]

    @classmethod
    def _outputs_from_assistant(cls, assistant_parts: list[str]) -> list[FieldSpec]:
        """Derive typed outputs from assistant-message structure."""
        for part in assistant_parts:
            obj = cls._json_object(part)
            if obj:
                return [
                    FieldSpec(
                        name=cls._normalize_field_name(key),
                        description=f"The {key.replace('_', ' ')} value in the response",
                        suggested_type=cls._infer_value_type(value),
                        field_type=FieldType.OUTPUT,
                    )
                    for key, value in obj.items()
                ]
        return [
            FieldSpec(
                name="response",
                description="The model's response to the user's request",
                suggested_type="string",
                field_type=FieldType.OUTPUT,
            ),
        ]

    @staticmethod
    def _json_object(text: str) -> dict[str, Any] | None:
        """Return the decoded JSON object in *text*, ignoring code fences."""
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            stripped = "\n".join(lines[1:-1] if len(lines) > 2 else lines[1:])
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) and value else None

    @staticmethod
    def _infer_value_type(value: Any) -> str:
        """Map a parsed JSON value to a natural-language type hint."""
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "float"
        if isinstance(value, list):
            return "list of strings"
        return "string"

    @classmethod
    def _sdk_class_name(
        cls,
        user_text: str,
        outputs: list[FieldSpec],
    ) -> str:
        """Derive a specific PascalCase class name from the task's leading keywords."""
        words = [
            word
            for word in re.findall(r"[a-zA-Z]{3,}", user_text)
            if word.lower() not in cls._NAME_STOPWORDS
        ][:4]
        if not words and outputs:
            words = outputs[0].name.split("_")
        if not words:
            return "TaskSignature"
        return cls._normalize_class_name(" ".join(words))

    @staticmethod
    def _is_sdk_format(prompt: ParsedPrompt) -> bool:
        raw = prompt.raw_input
        if not isinstance(raw, list) or not raw:
            return False
        return all(isinstance(m, dict) and "role" in m for m in raw)

    @classmethod
    def _build_sdk_context(cls, prompt: ParsedPrompt) -> dict[str, str]:
        raw = prompt.raw_input
        messages: list[dict[str, Any]] = raw if isinstance(raw, list) else []

        sdk_format = cls._detect_sdk_format(messages)

        return {
            "sdk_format": sdk_format,
            "messages_json": json.dumps(messages, indent=2, default=str),
            "task_hint": prompt.instruction_text,
        }

    @staticmethod
    def _detect_sdk_format(messages: list[dict[str, Any]]) -> str:
        if not messages:
            return "generic"
        first_msg = messages[0]
        if "parts" in first_msg:
            return "gemini"
        if any(isinstance(m.get("content"), list) for m in messages):
            return "anthropic"
        if "role" in first_msg and "content" in first_msg:
            return "openai"
        return "generic"

    @classmethod
    def _sanitize_sdk_spec(cls, spec: SignatureSpec) -> SignatureSpec:
        """Strip any field names that leaked from SDK structural metadata."""
        forbidden = {
            "role",
            "content",
            "parts",
            "messages",
            "system",
            "developer",
            "user",
            "assistant",
            "model",
            "tool",
            "input",
            "output",
            "text",
            "data",
            "result",
        }

        inputs = [f for f in spec.inputs if f.name not in forbidden]
        outputs = [f for f in spec.outputs if f.name not in forbidden]

        if not inputs or not outputs:
            raise ValueError("SDK spec contained only forbidden field names")

        return spec.model_copy(update={"inputs": inputs, "outputs": outputs})

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

        profile = prompt.data_profile
        rows = prompt.sample_rows
        if profile is None:
            from dspy_auto_signature.data.profiler import profile_columns
            from dspy_auto_signature.data.to_records import to_records

            all_rows = to_records(prompt.raw_input)
            profile = profile_columns(all_rows)
            rows = all_rows[:5]
        return {
            "source_kind": "dataset",
            "task_context": prompt.instruction_text,
            "examples_json": json.dumps(prompt.examples, indent=2, default=str),
            "data_profile_json": json.dumps(profile, indent=2, default=str),
            "sample_rows_json": json.dumps(rows, indent=2, default=str),
        }

    @staticmethod
    def _is_dataset(prompt: ParsedPrompt) -> bool:
        """Return whether the parsed source contains tabular data."""
        if prompt.source_kind == "dataset":
            return True
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
