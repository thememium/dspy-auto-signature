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

<!-- DATAFRAME EXAMPLE -->

<a name="dataframe-example"></a>

## DataFrame Example

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
| `source` | `Any` | Prompt string, message array, DataFrame, list of dictionaries, or list of `dspy.Example` objects |
| `task_hint` | `str \| None` | Optional task description, especially useful for identifying dataset targets |
| `input_hints` | `dict[str, str] \| None` | Input field descriptions to supplement or override generated descriptions |
| `output_hints` | `dict[str, str] \| None` | Output field descriptions to supplement or override generated descriptions |

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
