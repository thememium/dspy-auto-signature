<a name="readme-top"></a>

<div align="center">
  <h3 align="center">AutoSignature</h3>

  <p align="center">
    Generate typed <a href="https://dspy.ai"><code>dspy.Signature</code></a> classes with <code>dspy.RLM</code>.
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

AutoSignature uses `dspy.RLM` to inspect a prompt or dataset and generate a
complete `dspy.Signature` subclass.

- **Automatic signature design** — Infers instructions, inputs, outputs, field descriptions, and types
- **Dataset-aware generation** — Profiles DataFrame columns, distributions, and representative rows
- **Ready to use** — Returns signatures compatible with `dspy.Predict`, `dspy.ChainOfThought`, and other DSPy modules
- **Exportable** — Renders generated signatures as Python source with `to_source()`

Requires **Python 3.12+**, **DSPy 3.2+**, and
[Deno](https://deno.com/) for the RLM sandbox.

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

das.configure(
    lm=dspy.LM("openai/gpt-4o"),
    sub_lm=dspy.LM("openai/gpt-4o-mini"),
)

signature = das.generate(
    "Given an article, produce a concise summary and three key takeaways."
)

print(signature.to_source())

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
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

das.configure(
    lm=dspy.LM("openai/gpt-4o"),
    sub_lm=dspy.LM("openai/gpt-4o-mini"),
)

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

See [`example_sdk.py`](example_sdk.py) for a complete runnable example.

<!-- DATAFRAME EXAMPLE -->

<a name="dataframe-example"></a>

## DataFrame Example

Install with the `pandas` extra to get pandas:

```bash
uv add dspy-auto-signature[pandas]
```

Or with pip:

```bash
pip install dspy-auto-signature[pandas]
```

Datasets are profiled before generation so the RLM can use column names,
types, distributions, and representative rows when designing the signature.

```python
import dspy
import pandas as pd
import dspy_auto_signature as das

das.configure(
    lm=dspy.LM("openai/gpt-4o"),
    sub_lm=dspy.LM("openai/gpt-4o-mini"),
)

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

**Supported input formats:**

- Raw strings (system prompts, task descriptions)
- OpenAI SDK message arrays: `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]`
- Anthropic SDK message arrays: `[{"role": "user", "content": "..."}]`
- Google Gemini SDK contents: `[{"role": "user", "parts": [{"text": "..."}]}]`
- LiteLLM / Azure OpenAI / Ollama / vLLM message arrays
- pandas DataFrames, polars DataFrames / LazyFrames
- `list[dict]`, `list[dspy.Example]`, single `dspy.Example`

### `configure(lm=None, dataset_lm=None, sub_lm=None)`

Configures the models used during signature generation.

| Parameter | Type | Description |
| --- | --- | --- |
| `lm` | `dspy.LM \| None` | Default RLM model. Falls back to the model configured through `dspy.configure` |
| `dataset_lm` | `dspy.LM \| None` | Optional RLM model override for dataset sources |
| `sub_lm` | `dspy.LM \| None` | Optional model used for recursive sub-queries |

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
