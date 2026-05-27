# DSPy Auto-Signature Design

## Vision

Take any AI prompt material — a system prompt string, a Vercel AI SDK message array, an Anthropic XML prompt, or a combination — and generate a production-ready `dspy.Signature` from it. The signature is returned as a concrete DSPy class that can be immediately passed into `dspy.Predict`, `dspy.ChainOfThought`, or any other DSPy predictor.

## Core Idea: Meta-DSPy

We build a DSPy program that generates DSPy signatures. This is a "meta" layer:

```
Raw Prompt Input 
    → Prompt Analyzer (DSPy module) 
    → Extracts: task description, input fields, output fields, types, constraints
    → Schema Builder 
    → Returns: type[dspy.Signature] (a class, not an instance)
```

The user can then use it like:

```python
from dspy_auto_signature import from_prompt

signature = from_prompt("Summarize the following article into 3 bullet points")

# Use it immediately
summarizer = dspy.ChainOfThought(signature)
result = summarizer(article="Long article text...")
```

## Architecture

### Module Layout

```
src/dspy_auto_signature/
├── __init__.py             # Public API: from_prompt(), configure()
├── core/
│   ├── __init__.py
│   ├── signature_builder.py    # Builds dspy.Signature classes from specs
│   ├── field_inferencer.py     # Infers types/descriptions for fields
│   └── config.py               # Package configuration (LM, defaults)
├── generator/
│   ├── __init__.py
│   └── signature_generator.py  # The DSPy module that analyzes prompts
├── parser/
│   ├── __init__.py
│   ├── base.py                 # Abstract prompt parser
│   ├── string_parser.py        # Raw string / system prompt
│   ├── vercel_parser.py        # Vercel AI SDK format
│   └── anthropic_parser.py     # Anthropic XML / Claude format
├── types/
│   ├── __init__.py
│   └── signature_spec.py       # Pydantic model for intermediate representation
└── utils/
    ├── __init__.py
    └── type_resolver.py        # Map "list of strings" → list[str], etc.
```

### Key Components

#### 1. Parser Layer (`parser/`)

Accepts heterogeneous inputs and normalizes them into a uniform internal representation (`ParsedPrompt`).

-   `StringParser`: A raw string like `"You are a helpful assistant. Summarize articles into bullet points."`
-   `VercelParser`: Vercel AI SDK format: `[{ role: "system", content: "..." }, { role: "user", content: "..." }]`
-   `AnthropicParser`: Anthropic XML format or Claude-style prompts.

#### 2. Signature Generator (`generator/signature_generator.py`)

A DSPy `Module` that takes a `ParsedPrompt` and produces a `SignatureSpec`.

```python
class SignatureGenerator(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought(AnalyzePrompt)
        self.extract_fields = dspy.ChainOfThought(ExtractFields)
        self.refine = dspy.Predict(RefineSignature)
```

It uses a 3-step chain:
1.  **Analyze**: Understand the task from the prompt (instruction text)
2.  **Extract Fields**: Identify what inputs the task needs and what outputs it produces
3.  **Refine**: Polish the signature (field names, types, descriptions)

#### 3. Schema Builder (`core/signature_builder.py`)

Takes a `SignatureSpec` and constructs the actual `dspy.Signature` subclass using `dspy.signatures.make_signature`.

```python
def build_signature(spec: SignatureSpec) -> type[dspy.Signature]:
    fields = {}
    for field in spec.inputs:
        fields[field.name] = (field.resolved_type, dspy.InputField(desc=field.description))
    for field in spec.outputs:
        fields[field.name] = (field.resolved_type, dspy.OutputField(desc=field.description))
    
    return dspy.signatures.make_signature(
        fields, 
        instructions=spec.instructions,
        signature_name=spec.name
    )
```

#### 4. Type Inference (`utils/type_resolver.py`)

Maps natural language type descriptions to actual Python types:

-   `"a list of strings"` → `list[str]`
-   `"a number"` → `float`
-   `"yes or no"` → `bool`
-   Supports custom Pydantic models if schema is provided

### Data Flow

```
User Input (str | dict | list)
    ↓
[Parser] → ParsedPrompt (normalized text + metadata)
    ↓
[SignatureGenerator] → SignatureSpec (Pydantic model)
    ↓
[SignatureBuilder] → type[dspy.Signature]
    ↓
User receives a class that's fully compatible with DSPy
```

## Public API Design

```python
import dspy_auto_signature as das

# Configure the LM used for signature generation
das.configure(lm=dspy.LM("openai/gpt-4o"))

# Generate from a raw string
sig = das.from_prompt("Summarize articles into 3 bullet points")

# Generate from Vercel AI SDK format
sig = das.from_prompt([
    {"role": "system", "content": "You are a summarizer"},
    {"role": "user", "content": "Summarize: {article}"}
])

# Generate with explicit hints
sig = das.from_prompt(
    "Extract entities from text",
    input_hints={"text": "The article to analyze"},
    output_hints={"entities": "list of named entities"}
)

# The signature is a real DSPy Signature class
summarizer = dspy.ChainOfThought(sig)
```

## Implementation Order

1.  `types/signature_spec.py` — Define the intermediate representation
2.  `utils/type_resolver.py` — Type mapping utilities
3.  `core/signature_builder.py` — Build actual DSPy signatures
4.  `parser/base.py`, `string_parser.py` — Parse raw strings
5.  `generator/signature_generator.py` — The DSPy module
6.  `__init__.py` — Public API and configuration
8.  Tests for each module
