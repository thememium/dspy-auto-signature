# DSPy Auto-Signature

Generate production-ready `dspy.Signature` classes from any prompt material — raw strings, Vercel AI SDK message arrays, or Anthropic XML prompts. No manual signature writing required.

## Quick Start

```python
import dspy
import dspy_auto_signature as das

# 1. Configure the meta-model for signature generation
#    (One-time setup — use a strong model for best results)
das.configure(lm=dspy.LM("openai/gpt-4o"))

# 2. Generate a signature from a raw prompt
sig = das.from_prompt("Summarize the following article into 3 bullet points")

# 3. Configure the runtime model separately
#    (Use a cheaper/faster model for repeated inference)
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# 4. Use it immediately with any DSPy predictor
summarizer = dspy.ChainOfThought(sig)
result = summarizer(article="Long article text...")
```

## Installation

```bash
pip install dspy-auto-signature
```

Requires Python 3.10+ and DSPy 3.1+.

## How It Works

This is a **meta-DSPy** program: a DSPy module that generates DSPy signatures.

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

## Supported Input Formats

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

## Public API

### `from_prompt(prompt, input_hints=None, output_hints=None)`

Generate a `dspy.Signature` subclass from arbitrary prompt material.

**Args:**
- `prompt`: Raw string, Vercel AI SDK message array, or any supported format
- `input_hints`: Optional `dict[str, str]` mapping field names to descriptions
- `output_hints`: Optional `dict[str, str]` mapping field names to descriptions

**Returns:** A fresh `dspy.Signature` subclass compatible with `dspy.Predict`, `dspy.ChainOfThought`, etc.

### `configure(lm=None)`

Set the language model used **only for signature generation** (the meta-program). This is completely independent from the LM you use at runtime with your generated signatures.

If not called, the package will attempt to use whatever LM is globally configured via `dspy.configure(lm=...)`.

```python
# Use a strong model for one-time signature generation
das.configure(lm=dspy.LM("openai/gpt-4o"))

# Later, use a different model for runtime inference
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
```

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

## Example: Full Workflow

```python
import dspy
import dspy_auto_signature as das

# 1. Configure the meta-model for signature generation
#    (One-time — use a strong, slow model)
das.configure(lm=dspy.LM("openai/gpt-4o"))

# 2. Generate signature from a complex prompt
sig = das.from_prompt("""
You are an expert code reviewer. Given a pull request description and diff,
provide:
1. A summary of the changes
2. Potential bugs or issues
3. A risk score from 1-10
""")

# 3. Inspect what was generated
print(sig)  # CodeReviewer(pr_description, diff -> summary, issues, risk_score)

# 4. Configure the runtime model separately
#    (Repeated inference — use a cheaper/faster model)
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# 5. Use it
reviewer = dspy.ChainOfThought(sig)
result = reviewer(
    pr_description="Add user authentication",
    diff="...diff text..."
)

print(result.summary)
print(result.issues)
print(result.risk_score)
```

## Why Meta-DSPy?

Instead of hand-writing prompts to extract structure, we use DSPy signatures to generate DSPy signatures. This means:

- **Composability**: The meta-generator benefits from DSPy optimizers (BootstrapFewShot, MIPRO, etc.)
- **Type Safety**: Generated signatures are real Python classes with typed fields
- **No Prompt Engineering**: The system prompt is the signature docstring; the rest is inferred

## License

MIT