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

### `configure(lm=None)`

Set the language model used **only for signature generation** (the meta-program). This is completely independent from the LM you use at runtime with your generated signatures.

| Parameter | Type | Description |
| --- | --- | --- |
| `lm` | `dspy.LM \| None` | A `dspy.LM` instance, or `None` to use the global default |

If not called, the package will attempt to use whatever LM is globally configured via `dspy.configure(lm=...)`.

```python
# Use a strong model for one-time signature generation
das.configure(lm=dspy.LM("openai/gpt-4o"))

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
