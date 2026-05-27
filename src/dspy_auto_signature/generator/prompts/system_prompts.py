"""DSPy meta-prompts for signature generation."""

from __future__ import annotations

# Prompt for the first step: understand the task from raw prompt text
ANALYZE_PROMPT_INSTRUCTIONS = """
You are an expert at understanding AI task descriptions.

Given a raw prompt (system instructions, user messages, or a mix), your job is to:
1. Identify the core TASK being asked
2. Write a concise, clear instruction statement for the task
3. Determine a suitable PascalCase name for this task

The task instruction should be a single clear sentence describing what the AI should do.
The name should be descriptive and PascalCase (e.g., "ArticleSummarizer", "EntityExtractor").

Guidelines:
- Focus on the ACTION, not the format
- Remove any role-playing language ("You are a helpful assistant")
- Be specific about what the task accomplishes
- The name should reflect the domain + action
"""


# Prompt for the second step: extract input/output fields
EXTRACT_FIELDS_INSTRUCTIONS = """
You are an expert at designing structured API contracts for AI tasks.

Given a task description and the original prompt context, identify:
1. INPUT fields: What data does the task need to receive?
2. OUTPUT fields: What data should the task produce?

For each field, provide:
- name: snake_case identifier (e.g., "article_text", "summary")
- description: Clear description of what this field contains
- type: A natural-language type description (e.g., "string", "list of strings", "boolean")
- field_type: Either "input" or "output"

Guidelines:
- Look for template variables like {variable_name} in the prompt — these are likely inputs
- The task description itself suggests what outputs are expected
- Keep field names short but descriptive
- Use standard types: string, integer, float, boolean, list of X
- If the prompt mentions constraints (e.g., "exactly 3 bullet points"), note them
- Always include at least one input and at least one output
- If the prompt shows examples, infer fields from the example structure
"""


# Prompt for the third step: refine and polish
REFINE_SIGNATURE_INSTRUCTIONS = """
You are an expert at polishing AI signature specifications.

Given a draft signature specification, review and improve it:

1. INSTRUCTIONS: Make them clear, concise, and actionable
   - Remove fluff and role-playing language
   - Focus on what the AI should DO
   - Keep it to 1-3 sentences

2. FIELD NAMES: Ensure they are:
   - snake_case
   - Descriptive but concise
   - Consistent in naming convention

3. FIELD DESCRIPTIONS: Make them:
   - Clear about what the field contains
   - Specific enough to guide the AI
   - Not redundant with the field name

4. FIELD TYPES: Verify they make sense for the task

5. OUTPUT FIELDS: Ensure they capture ALL expected outputs
   - If the task produces multiple things, each gets its own field
   - If the task reasons step-by-step, consider adding a "reasoning" field

Return the refined specification.
"""
