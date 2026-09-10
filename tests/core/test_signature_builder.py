"""Tests for the SignatureBuilder."""

from __future__ import annotations

import types
from typing import IO, Any, Callable, List, Literal, Optional, Union, cast

import dspy
import pytest
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from dspy_auto_signature.core.signature_builder import (
    SignatureBuilder,
    _collect_imports,
    _escape_for_docstring,
    _type_to_str,
)
from dspy_auto_signature.types.signature_spec import FieldSpec, FieldType, SignatureSpec
from dspy_auto_signature.utils.type_resolver import TypeResolver


def _get_json_schema_extra(field_info: FieldInfo, key: str) -> object:
    """Safely extract a value from a FieldInfo's json_schema_extra dict."""
    extra = field_info.json_schema_extra
    if extra is None or callable(extra):
        return None
    return extra.get(key)


class TestSignatureBuilder:
    def test_build_simple_signature(self) -> None:
        spec = SignatureSpec(
            name="TestSummarizer",
            instructions="Summarize text into bullet points.",
            inputs=[
                FieldSpec(
                    name="text",
                    description="Text to summarize",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="summary",
                    description="Bullet point summary",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        Sig = SignatureBuilder.build(spec)

        assert issubclass(cast(type, Sig), dspy.Signature)
        assert Sig.__name__ == "TestSummarizer"
        assert Sig.instructions == "Summarize text into bullet points."
        assert "text" in Sig.input_fields
        assert "summary" in Sig.output_fields

    def test_build_with_multiple_fields(self) -> None:
        spec = SignatureSpec(
            name="MultiFieldTask",
            instructions="Do something complex.",
            inputs=[
                FieldSpec(
                    name="query", description="Search query", field_type=FieldType.INPUT
                ),
                FieldSpec(
                    name="context",
                    description="Additional context",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="answer", description="The answer", field_type=FieldType.OUTPUT
                ),
                FieldSpec(
                    name="confidence",
                    description="Confidence score",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        Sig = SignatureBuilder.build(spec)

        assert len(Sig.input_fields) == 2
        assert len(Sig.output_fields) == 2
        assert list(Sig.input_fields.keys()) == ["query", "context"]
        assert list(Sig.output_fields.keys()) == ["answer", "confidence"]

    def test_field_descriptions_preserved(self) -> None:
        spec = SignatureSpec(
            name="DescTest",
            instructions="Test descriptions.",
            inputs=[
                FieldSpec(
                    name="input_text",
                    description="The main input text to process",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="result",
                    description="The processing result",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        Sig = SignatureBuilder.build(spec)

        assert (
            _get_json_schema_extra(Sig.input_fields["input_text"], "desc")
            == "The main input text to process"
        )
        assert (
            Sig.input_fields["input_text"].description
            == "The main input text to process"
        )
        assert (
            _get_json_schema_extra(Sig.output_fields["result"], "desc")
            == "The processing result"
        )
        assert Sig.output_fields["result"].description == "The processing result"

    def test_signature_has_docstring(self) -> None:
        """Generated signatures must have their instructions as a class docstring."""
        spec = SignatureSpec(
            name="DocstringTest",
            instructions="This is the task description that should become the docstring.",
            inputs=[
                FieldSpec(
                    name="text",
                    description="Input text",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="output",
                    description="Output text",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        Sig = SignatureBuilder.build(spec)

        assert Sig.__doc__ is not None
        assert "task description" in Sig.__doc__
        assert Sig.instructions == spec.instructions

    def test_all_fields_have_types(self) -> None:
        """Every field in the generated signature must have a type annotation."""
        spec = SignatureSpec(
            name="TypedTest",
            instructions="Test type annotations.",
            inputs=[
                FieldSpec(
                    name="text",
                    description="A string input",
                    suggested_type="string",
                    field_type=FieldType.INPUT,
                ),
                FieldSpec(
                    name="count",
                    description="An integer input",
                    suggested_type="integer",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="result",
                    description="A list output",
                    suggested_type="list of strings",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        Sig = SignatureBuilder.build(spec)

        assert Sig.__annotations__["text"] is str
        assert Sig.__annotations__["count"] is int
        assert Sig.__annotations__["result"] == list[str]

    def test_all_fields_are_dspy_fields(self) -> None:
        """Input fields must use InputField and output fields must use OutputField."""
        spec = SignatureSpec(
            name="FieldKindTest",
            instructions="Test field kinds.",
            inputs=[
                FieldSpec(
                    name="query",
                    description="Search query",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="answer",
                    description="The answer",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        Sig = SignatureBuilder.build(spec)

        assert (
            _get_json_schema_extra(Sig.input_fields["query"], "__dspy_field_type")
            == "input"
        )
        assert (
            _get_json_schema_extra(Sig.output_fields["answer"], "__dspy_field_type")
            == "output"
        )

    def test_missing_description_raises(self) -> None:
        """Building a signature with an empty description must raise ValueError."""
        spec = SignatureSpec(
            name="BadSpec",
            instructions="Test missing description.",
            inputs=[
                FieldSpec(
                    name="text",
                    description="",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[],
        )

        with pytest.raises(ValueError, match="missing a description"):
            SignatureBuilder.build(spec)

    def test_full_class_based_signature(self) -> None:
        """A complete signature must have docstring, typed fields, and descriptions."""
        spec = SignatureSpec(
            name="CompleteSignature",
            instructions="Given a product review, extract the sentiment and key points.",
            inputs=[
                FieldSpec(
                    name="review",
                    description="The customer product review to analyze",
                    suggested_type="string",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="sentiment",
                    description="The overall sentiment: positive, negative, or neutral",
                    suggested_type="string",
                    field_type=FieldType.OUTPUT,
                ),
                FieldSpec(
                    name="key_points",
                    description="A list of key points mentioned in the review",
                    suggested_type="list of strings",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        Sig = SignatureBuilder.build(spec)

        assert Sig.__doc__ is not None
        assert "product review" in Sig.__doc__

        assert Sig.__annotations__["review"] is str
        assert (
            Sig.input_fields["review"].description
            == "The customer product review to analyze"
        )
        assert (
            _get_json_schema_extra(Sig.input_fields["review"], "desc")
            == "The customer product review to analyze"
        )

        assert Sig.__annotations__["sentiment"] is str
        assert Sig.__annotations__["key_points"] == list[str]
        assert (
            Sig.output_fields["sentiment"].description
            == "The overall sentiment: positive, negative, or neutral"
        )
        assert (
            Sig.output_fields["key_points"].description
            == "A list of key points mentioned in the review"
        )

    def test_to_source_generates_valid_python(self) -> None:
        spec = SignatureSpec(
            name="SourceTest",
            instructions="Test the to_source method.",
            inputs=[
                FieldSpec(
                    name="text",
                    description="Input text",
                    suggested_type="string",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="result",
                    description="The result",
                    suggested_type="list of strings",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        Sig = SignatureBuilder.build(spec)
        source = Sig.to_source()

        assert "class SourceTest(dspy.Signature):" in source
        assert 'text: str = dspy.InputField(desc="Input text")' in source
        assert 'result: list[str] = dspy.OutputField(desc="The result")' in source
        assert "import dspy" in source

        compile(source, "<generated>", "exec")

    def test_signature_builder_to_source_classmethod(self) -> None:
        spec = SignatureSpec(
            name="ClassMethodTest",
            instructions="Test the classmethod.",
            inputs=[
                FieldSpec(
                    name="query",
                    description="Search query",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="answer",
                    description="The answer",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        source = SignatureBuilder.to_source(spec)

        assert "class ClassMethodTest(dspy.Signature):" in source
        assert 'query: str = dspy.InputField(desc="Search query")' in source
        assert 'answer: str = dspy.OutputField(desc="The answer")' in source


class _CustomModel(BaseModel):
    """A plain model whose module is not ``builtins`` (for import tests)."""


class _FakeField:
    """Minimal field spec stand-in with an arbitrary resolved type."""

    def __init__(self, resolved_type: object) -> None:
        self.resolved_type = resolved_type


class TestTypeToStr:
    """Tests for ``_type_to_str`` type rendering."""

    def test_renders_any(self) -> None:
        assert _type_to_str(Any) == "Any"

    def test_renders_none_type(self) -> None:
        assert _type_to_str(type(None)) == "None"

    def test_renders_ellipsis(self) -> None:
        assert _type_to_str(Ellipsis) == "..."

    def test_renders_literal(self) -> None:
        assert _type_to_str(Literal["a", "b"]) == "Literal['a', 'b']"

    def test_renders_parameterized_generic_alias(self) -> None:
        assert _type_to_str(list[str]) == "list[str]"
        assert _type_to_str(dict[str, int]) == "dict[str, int]"

    def test_renders_bare_generic_alias(self) -> None:
        """A GenericAlias without parameters renders as its origin type."""
        assert _type_to_str(types.GenericAlias(list, ())) == "list"

    def test_renders_union_type(self) -> None:
        assert _type_to_str(str | int | None) == "str | int | None"

    def test_renders_typing_generic_alias_with_name(self) -> None:
        assert _type_to_str(List[int]) == "List[int]"

    def test_renders_typing_optional(self) -> None:
        """typing.Optional is a typing generic alias: name plus nested args."""
        assert _type_to_str(Optional[str]) == "Optional[str, None]"

    def test_renders_callable(self) -> None:
        """typing.Callable has a type origin and no name, so the origin is named."""
        assert _type_to_str(Callable[[int], str]) == "Callable[int, str]"

    def test_renders_typing_union(self) -> None:
        """typing.Union has a special-form origin, so repr(origin) is used."""
        assert _type_to_str(Union[str, int]) == "typing.Union[str, int]"

    def test_renders_typing_alias_with_type_origin_and_no_name(self) -> None:
        """typing.IO has a type origin but no _name, so the origin's name is used."""
        assert _type_to_str(IO[str]) == "IO[str]"

    def test_renders_bare_typing_generic_alias(self) -> None:
        """A typing generic alias without parameters renders as its name."""
        assert _type_to_str(List) == "List"

    def test_renders_non_type_value_via_repr(self) -> None:
        """Values that are not types fall back to repr."""
        assert _type_to_str(42) == "42"


class TestCollectImports:
    """Tests for ``_collect_imports`` import discovery."""

    def test_any_requires_typing_import(self) -> None:
        fields = [
            FieldSpec(
                name="value",
                description="Any value",
                suggested_type="any",
                field_type=FieldType.INPUT,
            ),
        ]
        assert _collect_imports(fields) == {"from typing import Any"}

    def test_literal_requires_typing_import(self) -> None:
        fields = [
            FieldSpec(
                name="level",
                description="Severity level",
                suggested_type="one of low, medium, high",
                field_type=FieldType.INPUT,
            ),
        ]
        assert _collect_imports(fields) == {"from typing import Literal"}

    def test_builtin_type_requires_no_import(self) -> None:
        fields = [
            FieldSpec(
                name="text",
                description="Some text",
                suggested_type="string",
                field_type=FieldType.INPUT,
            ),
        ]
        assert _collect_imports(fields) == set()

    def test_generic_alias_walks_builtin_args(self) -> None:
        fields = [
            FieldSpec(
                name="items",
                description="Some items",
                suggested_type="list of strings",
                field_type=FieldType.INPUT,
            ),
        ]
        assert _collect_imports(fields) == set()

    def test_typing_generic_alias_walks_type_parameters(self) -> None:
        """typing generic aliases recurse into their type parameters."""
        imports = _collect_imports(
            cast("list[FieldSpec]", [_FakeField(Optional[_CustomModel])])
        )
        assert imports == {
            f"from {_CustomModel.__module__} import _CustomModel",
        }

    def test_typing_generic_alias_with_builtin_args_needs_no_import(self) -> None:
        imports = _collect_imports(cast("list[FieldSpec]", [_FakeField(Optional[str])]))
        assert imports == set()

    def test_custom_class_requires_import(self) -> None:
        """A plain class from a non-builtin module needs an import statement."""
        imports = _collect_imports(cast("list[FieldSpec]", [_FakeField(_CustomModel)]))
        assert imports == {f"from {_CustomModel.__module__} import _CustomModel"}


class TestEscapeForDocstring:
    def test_escapes_backslashes_and_triple_quotes(self) -> None:
        assert _escape_for_docstring('a\\b """ c') == 'a\\\\b \\"\\"\\" c'


class TestGenerateSource:
    def test_multiline_instructions_use_block_docstring(self) -> None:
        """Instructions with newlines or triple quotes produce a block docstring."""
        spec = SignatureSpec(
            name="MultilineSpec",
            instructions='Step one.\nStep two with """ embedded.\nDone.',
            inputs=[
                FieldSpec(
                    name="text",
                    description="Input text",
                    field_type=FieldType.INPUT,
                ),
            ],
            outputs=[
                FieldSpec(
                    name="result",
                    description="The result",
                    field_type=FieldType.OUTPUT,
                ),
            ],
        )

        source = SignatureBuilder.to_source(spec)

        assert source.count('"""') >= 2
        assert 'Step two with \\"\\"\\" embedded.' in source
        compile(source, "<generated>", "exec")

        Sig = SignatureBuilder.build(spec)
        assert Sig.instructions == spec.instructions

    def test_source_includes_import_for_custom_type(self) -> None:
        """A field whose type lives in a non-builtin module emits an import."""
        TypeResolver.register("_CustomModel", _CustomModel)
        try:
            spec = SignatureSpec(
                name="CustomTypeSpec",
                instructions="Use a custom type.",
                inputs=[
                    FieldSpec(
                        name="model",
                        description="A custom model",
                        suggested_type="_CustomModel",
                        field_type=FieldType.INPUT,
                    ),
                ],
                outputs=[],
            )

            source = SignatureBuilder.to_source(spec)
            assert f"from {_CustomModel.__module__} import _CustomModel" in source
            assert "model: _CustomModel = dspy.InputField" in source
            compile(source, "<generated>", "exec")

            Sig = SignatureBuilder.build(spec)
            assert Sig.input_fields["model"].annotation is _CustomModel
        finally:
            TypeResolver._ALIASES.pop("_custommodel", None)


class TestMakeFieldTuple:
    def test_constraints_forwarded_to_field_factory(self) -> None:
        """A field with constraints passes them via json_schema_extra to the factory."""
        field_spec = FieldSpec(
            name="score",
            description="Score between 0 and 1",
            field_type=FieldType.INPUT,
            constraints="between 0 and 1",
        )
        captured: dict[str, object] = {}

        def recording_factory(**kwargs: object) -> FieldInfo:
            captured.update(kwargs)
            return FieldInfo()

        resolved, _ = SignatureBuilder._make_field_tuple(field_spec, recording_factory)

        assert resolved is str
        assert captured["json_schema_extra"] == {"constraints": "between 0 and 1"}

    def test_without_constraints_omits_json_schema_extra(self) -> None:
        field_spec = FieldSpec(
            name="score",
            description="Score between 0 and 1",
            field_type=FieldType.INPUT,
        )
        captured: dict[str, object] = {}

        def recording_factory(**kwargs: object) -> FieldInfo:
            captured.update(kwargs)
            return FieldInfo()

        _, info = SignatureBuilder._make_field_tuple(field_spec, recording_factory)

        assert "json_schema_extra" not in captured
        assert captured["desc"] == "Score between 0 and 1"
        assert captured["description"] == "Score between 0 and 1"

    def test_with_real_dspy_field_factory(self) -> None:
        """The tuple works end-to-end with dspy's real field factories."""
        field_spec = FieldSpec(
            name="score",
            description="Score between 0 and 1",
            field_type=FieldType.OUTPUT,
        )

        resolved, info = SignatureBuilder._make_field_tuple(
            field_spec, dspy.OutputField
        )

        assert resolved is str
        assert isinstance(info, FieldInfo)
        assert info.description == "Score between 0 and 1"
