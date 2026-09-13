<a name="readme-top"></a>

<div align="center">
  <h3 align="center">AutoSignature</h3>

  <p align="center">
    Generate typed <a href="https://dspy.ai"><code>dspy.Signature</code></a> classes from prompts, SDK messages, or datasets.
    <br />
    <a href="#table-of-contents"><strong>Explore the Documentation »</strong></a>
    <br />
    <a href="https://github.com/thememium/dspy-auto-signature/issues">Report Bug</a>
    <a href="https://github.com/thememium/dspy-auto-signature/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->

<a name="table-of-contents"></a>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#sdk-message-formats">SDK Message Formats</a></li>
    <li><a href="#dataframe-example">DataFrame Example</a></li>
    <li><a href="#api">API</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT -->

<a name="about"></a>

## About

AutoSignature inspects a prompt, SDK message array, or dataset and generates a
complete `dspy.Signature` subclass. Structured inputs are designed
deterministically; anything else gets a single LLM call (`dspy.ChainOfThought`).
No sandbox, no Deno, no iterative loop — one call and you're done.

- **Automatic signature design** — Infers instructions, inputs, outputs, field descriptions, and types
- **Strongly typed outputs** — When the LLM designs the signature (`mode="cot"` or `mode="rlm"`), structured outputs become real Pydantic `BaseModel` types with nested models, `Literal` enums, and validation
- **Dataset-aware generation** — Profiles DataFrame columns, distributions, and representative rows
- **Ready to use** — Returns signatures compatible with `dspy.Predict`, `dspy.ChainOfThought`, and other DSPy modules
- **Exportable** — Renders generated signatures as Python source with `to_source()`

Requires **Python 3.12+** and **DSPy 3.2+**.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- QUICK START -->

<a name="quick-start"></a>

## Quick Start

### Install

Install AutoSignature with uv (recommended):

```bash
uv add dspy-auto-signature
```

Or with pip:

```bash
pip install dspy-auto-signature
```

### Basic Usage

```python
import dspy
import dspy_auto_signature as das

das.configure(lm=dspy.LM("openrouter/openai/gpt-oss-120b"))

signature = das.generate(
    "Given an article, produce a concise summary and three key takeaways."
)

print(signature.to_source())

dspy.configure(lm=dspy.LM("openrouter/openai/gpt-oss-120b"))
summarize = dspy.ChainOfThought(signature.to_signature())
result = summarize(article="...")
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- SDK MESSAGE FORMATS -->

<a name="sdk-message-formats"></a>

## SDK Message Formats

AutoSignature accepts message arrays from popular LLM SDKs. Pass your existing
conversation history directly — no conversion needed.

### OpenAI SDK

```python
import dspy
import dspy_auto_signature as das

das.configure(lm=dspy.LM("openrouter/openai/gpt-oss-120b"))

messages = [
    {"role": "system", "content": "You are a technical writer who produces clear documentation."},
    {"role": "user", "content": "Write a README for a Python CLI tool that converts CSV to JSON."},
]

signature = das.generate(messages)
print(signature.to_source())
```

### Anthropic SDK

```python
messages = [
    {"role": "user", "content": "Analyze the sentiment of customer reviews."},
    {"role": "assistant", "content": "I'll classify each review as positive, negative, or neutral."},
]

signature = das.generate(messages)
```

### Google Gemini SDK

```python
contents = [
    {"role": "user", "parts": [{"text": "Extract key entities from this legal contract."}]},
    {"role": "model", "parts": [{"text": "I'll identify parties, dates, and obligations."}]},
]

signature = das.generate(contents)
```

### LiteLLM / OpenAI-Compatible

Any SDK that produces OpenAI-style `[{"role": "...", "content": "..."}]` arrays
works out of the box — including LiteLLM, Azure OpenAI, Ollama, and vLLM.

See [`examples/sdk.py`](examples/sdk.py) for a complete runnable example.

<!-- DATAFRAME EXAMPLE -->

<a name="dataframe-example"></a>

## DataFrame Example

Install with the `pandas` extra to get pandas:

```bash
uv add "dspy-auto-signature[pandas]"
```

Or with pip:

```bash
pip install "dspy-auto-signature[pandas]"
```

Datasets are profiled before generation so the signature reflects your column
names, types, and representative rows. Add `mode="cot"` to let an LLM design
the signature from the profile instead.

```python
import dspy
import pandas as pd
import dspy_auto_signature as das

das.configure(lm=dspy.LM("openrouter/openai/gpt-oss-120b"))

tickets = pd.DataFrame(
    [
        {"message": "Server is down", "urgency": "high"},
        {"message": "Please update my profile", "urgency": "low"},
        {"message": "Payment failed", "urgency": "high"},
    ]
)

signature = das.generate(
    tickets,
    task_hint="Classify support ticket urgency from the message",
)

print(signature.to_source())
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- API -->

<a name="api"></a>

## API

### `generate(source, task_hint=None, *, input_hints=None, output_hints=None)`

Generates a `dspy.Signature` subclass from prompt material or tabular data.

| Parameter | Type | Description |
| --- | --- | --- |
| `source` | `Any` | Prompt string, SDK message array (OpenAI, Anthropic, Google, LiteLLM), DataFrame, list of dictionaries, or list of `dspy.Example` objects |
| `task_hint` | `str \| None` | Optional task description, especially useful for identifying dataset targets |
| `input_hints` | `dict[str, str] \| None` | Input field descriptions to supplement or override generated descriptions |
| `output_hints` | `dict[str, str] \| None` | Output field descriptions to supplement or override generated descriptions |
| `mode` | `"auto" \| "fast" \| "cot" \| "rlm"` | Generation strategy. Defaults to `auto` — see below |

**Supported input formats:**

- Raw strings (system prompts, task descriptions)
- OpenAI SDK message arrays: `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]`
- Anthropic SDK message arrays: `[{"role": "user", "content": "..."}]`
- Google Gemini SDK contents: `[{"role": "user", "parts": [{"text": "..."}]}]`
- LiteLLM / Azure OpenAI / Ollama / vLLM message arrays
- pandas DataFrames, polars DataFrames / LazyFrames
- `list[dict]`, `list[dspy.Example]`, single `dspy.Example`

**Generation modes** (`mode` parameter):

| Mode | Behavior |
| --- | --- |
| `auto` | **Default.** Deterministic design for structured inputs (placeholders, SDK arrays, datasets); a single `dspy.ChainOfThought` call for everything else |
| `fast` | Fully deterministic — never contacts an LLM |
| `cot` | Always uses a single `dspy.ChainOfThought` call, even for structured inputs |
| `rlm` | Heavy recursive `dspy.RLM` architect (sandboxed exploration; requires [Deno](https://deno.com/)) |

`auto` is right for almost everything. Pass `mode="cot"` or `mode="rlm"` when
you want the LLM to design even structured inputs.

### Typed outputs with Pydantic models

Whenever ChainOfThought or the RLM architect designs a signature, structured
outputs are strongly typed. Instead of a plain `str` field described as "JSON
containing ...", the architect proposes a full Pydantic model schema, and the
generated signature uses a real `BaseModel` class as the output field type:

```python
import dspy
import dspy_auto_signature as das

das.configure(lm=dspy.LM("openai/gpt-4o-mini"))

Signature = das.generate(
    "Extract the customer's contact information from the support ticket: {ticket}",
    mode="cot",
)
contact_field = Signature.output_fields["contact"].annotation
# A generated pydantic model, e.g. ContactRecord(full_name=..., age=..., ...)
print(contact_field.model_json_schema())
```

Behavior:

- Multi-field and nested outputs become `pydantic` model schemas with concrete
  field types (`str`, `int`, `float`, `bool`, `list[str]`, `dict[str, str]`,
  `Literal`, and nested `pydantic` models). DSPy validates model responses
  against the generated model at runtime.
- Enumerated outputs become `typing.Literal` types.
- Simple scalar outputs stay plain (`str`, `int`, ...).
- `to_source()` renders the Pydantic model classes above the signature class,
  so exported source is self-contained and importable.
- Deterministic paths (`mode="fast"` and structural `auto` inputs) keep their
  existing inferred types.

The generated `PydanticModelSchema` type is exported for programmatic use with
`SignatureSpec` and `FieldSpec.model_schema`.

### `configure(lm=None, dataset_lm=None, sub_lm=None)`

Configures the models used during signature generation.

| Parameter | Type | Description |
| --- | --- | --- |
| `lm` | `dspy.LM \| None` | Default generation model. Falls back to the model configured through `dspy.configure` |
| `dataset_lm` | `dspy.LM \| None` | Optional generation model override for dataset sources |
| `sub_lm` | `dspy.LM \| None` | Optional cheap inner model used by `mode="rlm"` sub-queries |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->

<a name="contributing"></a>

## Contributing

Quick workflow:

1. Fork and branch: `git checkout -b feature/name`
2. Make changes
3. Commit and push
4. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

<a name="license"></a>

## License

MIT (as declared in `pyproject.toml`).

---

<div align="center">
  <p>
    <sub>Built by <a href="https://github.com/thememium">thememium</a></sub>
  </p>
</div>
