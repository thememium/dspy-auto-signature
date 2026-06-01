<a name="readme-top"></a>

<div align="center">
  <h3 align="center">AutoSignature</h3>

  <p align="center">
    Generate production-ready <code>dspy.Signature</code> classes from any prompt material.
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
    <li><a href="#usage">Usage</a></li>
    <li><a href="#dataset-input">Dataset Input (Slow Path)</a></li>
    <li><a href="#api">API</a></li>
    <li><a href="#architecture">Architecture</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>

<!-- ABOUT -->

<a name="about"></a>

## About

AutoSignature is a **meta-DSPy** program: a DSPy module that generates DSPy signatures. Instead of hand-writing prompts to extract structure, it uses DSPy signatures to generate DSPy signatures.

- **Zero manual signature writing** — Pass any prompt material and get a real `dspy.Signature` subclass back
- **Heterogeneous input support** — Raw strings, Vercel AI SDK message arrays, or Anthropic XML prompts
- **Full class-based signatures** — Generated signatures have docstrings, typed `dspy.InputField` / `dspy.OutputField` fields, and descriptions
- **Composable** — The meta-generator benefits from DSPy optimizers (BootstrapFewShot, MIPRO, etc.)
- **Type safe** — Generated signatures are real Python classes with typed fields

Requires **Python 3.10+** and **DSPy 3.1+**.

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

# 1. Configure the meta-model for signature generation
#    (One-time setup — use a strong model for best results)
das.configure(lm=dspy.LM("openrouter/openai/gpt-oss-120b"))

# 2. Generate a signature from a raw prompt
sig = das.from_prompt("Summarize the following article into 3 short bullet points")

# 3. Inspect what was generated
print(f"Generated signature: {sig}")
print(f"Docstring: {sig.__doc__}")
print(f"Inputs:  {list(sig.input_fields.keys())}")
print(f"Outputs: {list(sig.output_fields.keys())}")

# 4. Save the generated signature to a file
with open("summary_signature.py", "w", encoding="utf-8") as f:
    f.write(sig.to_source())

# 5. Configure the runtime model separately
#    (Use a cheaper/faster model for repeated inference)
dspy.configure(lm=dspy.LM("openrouter/openai/gpt-oss-120b"))

# 6. Use it immediately with any DSPy predictor
summarizer = dspy.ChainOfThought(sig.to_signature())

article = (
    "Artificial intelligence has transformed industries ranging from "
    "healthcare to finance. Machine learning models can now diagnose "
    "diseases, predict market trends, and automate customer service. "
    "However, ethical concerns around bias, privacy, and job displacement "
    "remain significant challenges that researchers and policymakers "
    "continue to address."
)

result = summarizer(article=article)
print(result)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE -->

<a name="usage"></a>

## Usage

### Raw String

```python
sig = das.from_prompt("Extract named entities from the given text")
```

### Vercel AI SDK Format

```python
sig = das.from_prompt([
    {"role": "system", "content": "You are a summarizer"},
    {"role": "user", "content": "Summarize: {article}"}
])
```

### With Explicit Hints

Override or supplement inferred fields:

```python
sig = das.from_prompt(
    "Extract entities from text",
    input_hints={"text": "The article to analyze"},
    output_hints={"entities": "list of named entities"}
)
```

### Full Workflow Example

```python
import dspy
import dspy_auto_signature as das

# 1. Configure the meta-model for signature generation
#    (One-time — use a strong, slow model)
das.configure(lm=dspy.LM("openrouter/openai/gpt-oss-120b"))

# 2. Generate signature from a complex prompt
sig = das.from_prompt("""
You are an expert code reviewer. Given a pull request description and diff,
provide:
1. A summary of the changes
2. Potential bugs or issues
3. A risk score from 1-10
""")

# 3. Inspect what was generated
print(f"Generated signature: {sig}")
print(f"Docstring: {sig.__doc__}")
print(f"Inputs:  {list(sig.input_fields.keys())}")
print(f"Outputs: {list(sig.output_fields.keys())}")

# 4. Configure the runtime model separately
#    (Repeated inference — use a cheaper/faster model)
dspy.configure(lm=dspy.LM("openrouter/openai/gpt-oss-120b"))

# 5. Use it
reviewer = dspy.ChainOfThought(sig.to_signature())
result = reviewer(
    pr_description="Add user authentication",
    diff="...diff text..."
)

print(result.summary)
print(result.issues)
print(result.risk_score)
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- DATASET INPUT -->

<a name="dataset-input"></a>

## Dataset Input (Slow Path)

For more thoughtful, data-grounded signatures, pass a dataset directly. AutoSignature will profile your columns (dtypes, null rates, cardinality, sample values, distributions) and use **DSPy RLM** (Recursive Language Model) to iteratively analyze the data and produce a robust signature.

### Why use the slow path?

The fast path reads your prompt text and infers field shapes from the description alone. The slow path **looks at your actual data**, so:

- Field names come from your real column names
- Field descriptions reference your real value distributions (e.g. "One of: high, medium, low")
- Input vs. output direction is inferred from cardinality and data shape
- Type hints are grounded in observed dtypes

### Prerequisites

The slow path uses `dspy.RLM`, which requires **Deno** (sandboxed Python REPL runtime):

```bash
brew install deno   # macOS
# or see https://deno.land for Linux/Windows
```

### From a pandas DataFrame

```python
import dspy
import pandas as pd
import dspy_auto_signature as das

das.configure(
    lm=dspy.LM("openai/gpt-4o"),         # meta-LM for analysis
    dataset_lm=dspy.LM("openai/gpt-4o"),  # RLM outer LM (defaults to lm)
    sub_lm=dspy.LM("openai/gpt-4o-mini"), # RLM inner LM (cheap)
)

df = pd.read_csv("tickets.csv")
sig = das.from_dataset(df, task_hint="Classify support tickets by urgency and sentiment")
```

### From a list of dicts

```python
rows = [
    {"message": "Server is on fire", "urgency": "high", "sentiment": "negative"},
    {"message": "Please clean conf room B", "urgency": "low", "sentiment": "neutral"},
    {"message": "Thanks for the quick fix!", "urgency": "low", "sentiment": "positive"},
]
sig = das.from_dataset(rows, task_hint="Classify support tickets")
```

### From a list of `dspy.Example`

```python
import dspy

examples = [
    dspy.Example(message="Server is on fire", urgency="high", sentiment="negative").with_inputs("message"),
    dspy.Example(message="Please clean conf room B", urgency="low", sentiment="neutral").with_inputs("message"),
]
sig = das.from_dataset(examples)
```

### Supported input types

`from_dataset` accepts (duck-typed, no hard imports required):

- `list[dict]`
- `pandas.DataFrame`
- `polars.DataFrame` / `polars.LazyFrame`
- `list[dspy.Example]`
- Any object with `.to_dicts()`, `.to_pandas()`, or `.to_dict()` methods

### Performance & cost

The slow path uses `dspy.RLM` with `max_iterations=20` and `max_llm_calls=50` by default. A cheap `sub_lm` (e.g. `gpt-4o-mini`) handles inner refinement passes. Expect a few seconds to a minute per signature, depending on dataset size and LM latency.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- API -->

<a name="api"></a>

## API

### `from_prompt(prompt, input_hints=None, output_hints=None)`

Generate a `dspy.Signature` subclass from arbitrary prompt material.

| Parameter | Type | Description |
| --- | --- | --- |
| `prompt` | `str \| list[dict[str, str]] \| Any` | The prompt material. Can be a raw string, Vercel AI SDK message array, or any supported format |
| `input_hints` | `dict[str, str] \| None` | Optional mapping of field-name → description for known inputs |
| `output_hints` | `dict[str, str] \| None` | Optional mapping of field-name → description for known outputs |

**Returns:** A fresh `dspy.Signature` subclass compatible with `dspy.Predict`, `dspy.ChainOfThought`, etc.

### `from_dataset(data, task_hint=None, *, input_hints=None, output_hints=None)`

Generate a `dspy.Signature` subclass from a dataset by profiling its columns and using DSPy RLM to iteratively analyze the data.

| Parameter | Type | Description |
| --- | --- | --- |
| `data` | `DataFrame \| list[dict] \| list[dspy.Example] \| Any` | The dataset. Accepts pandas/polars DataFrames, lists of dicts, lists of `dspy.Example`, or any object with `.to_dicts()`/`.to_pandas()`/`.to_dict()` |
| `task_hint` | `str \| None` | Optional natural-language description of the task to bias the RLM |
| `input_hints` | `dict[str, str] \| None` | Optional mapping of field-name → description for known inputs |
| `output_hints` | `dict[str, str] \| None` | Optional mapping of field-name → description for known outputs |

**Returns:** A fresh `dspy.Signature` subclass compatible with `dspy.Predict`, `dspy.ChainOfThought`, etc.

See the [Dataset Input](#dataset-input) section for prerequisites and detailed usage.

### `configure(lm=None, dataset_lm=None, sub_lm=None)`

Set the language models used for signature generation.

| Parameter | Type | Description |
| --- | --- | --- |
| `lm` | `dspy.LM \| None` | The meta-LM used for the fast path (`from_prompt`). Falls back to `dspy.settings.lm` if unset |
| `dataset_lm` | `dspy.LM \| None` | The RLM outer LM used for the slow path (`from_dataset`). Falls back to `lm` if unset |
| `sub_lm` | `dspy.LM \| None` | The cheap inner LM used by RLM for sub-queries. Falls back to `lm` if unset |

The fast-path `lm` and slow-path `dataset_lm` are completely independent from the runtime LM you use with your generated signatures.

```python
# Use a strong model for one-time signature generation
das.configure(
    lm=dspy.LM("openai/gpt-4o"),
    dataset_lm=dspy.LM("openai/gpt-4o"),
    sub_lm=dspy.LM("openai/gpt-4o-mini"),
)

# Later, use a different model for runtime inference
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ARCHITECTURE -->

<a name="architecture"></a>

## Architecture

```
src/dspy_auto_signature/
├── __init__.py              # Public API: from_prompt(), configure()
├── core/
│   ├── signature_builder.py # Builds dspy.Signature classes from specs
│   └── config.py            # Package configuration (LM, defaults)
├── generator/
│   └── signature_generator.py  # The DSPy meta-module
├── parser/
│   ├── base.py              # Abstract prompt parser
│   ├── string_parser.py     # Raw string / system prompt
│   ├── vercel_parser.py     # Vercel AI SDK format
│   └── __init__.py          # AutoParser orchestrator
├── types/
│   └── signature_spec.py    # Pydantic models for intermediate representation
└── utils/
    └── type_resolver.py     # Map "list of strings" → list[str], etc.
```

### Key Components

1. **Parser Layer** (`parser/`): Accepts heterogeneous inputs and normalises them into `ParsedPrompt`
2. **Signature Generator** (`generator/signature_generator.py`): A 3-step DSPy `Module` that analyses prompts and produces `SignatureSpec`
3. **Signature Builder** (`core/signature_builder.py`): Constructs actual `dspy.Signature` subclasses using `dspy.signatures.make_signature`
4. **Type Resolver** (`utils/type_resolver.py`): Maps natural language type descriptions to Python types

### How It Works

```
Raw Prompt Input
    → Parser (normalises heterogeneous formats)
    → SignatureGenerator (DSPy module)
        1. Analyze: Extract task name and instruction
        2. Extract Fields: Identify inputs and outputs
        3. Refine: Polish names, types, and descriptions
    → SignatureBuilder (constructs dspy.Signature subclass)
    → type[dspy.Signature] (a real class, ready to use)
```

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
