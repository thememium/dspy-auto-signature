# Reporting a Vulnerability

To report a security vulnerability, please email boswell.labs@gmail.com.

We take security seriously and will respond to security reports within 48 hours. Please include as much detail as possible about the vulnerability, including:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if any)

While the discovery of new vulnerabilities is rare, we also recommend always using the latest version of dspy-auto-signature to ensure your application remains as secure as possible.

## Security Considerations for dspy-auto-signature

As dspy-auto-signature uses `dspy.RLM` to generate `dspy.Signature` classes from prompts and datasets via a Deno sandbox, please be aware of the following security practices:

- **Data Exposure**: Prompt text and dataset contents (including column names, types, distributions, and representative rows) are sent to LLMs during signature generation. Do not use AutoSignature with sensitive, personally identifiable, or proprietary data unless you fully trust and have contractual agreements with your LLM provider.
- **RLM Sandbox**: AutoSignature executes LLM-generated code inside DSPy's `dspy.RLM` Deno sandbox. While the sandbox provides isolation, review the generated `dspy.Signature` classes before using them in production pipelines, especially when derived from untrusted input.
- **DSPy Dependency**: AutoSignature relies on DSPy's `RLM`, `Predict`, and `Signature` infrastructure. Security properties are bounded by DSPy's own guarantees. Always pin to a known-compatible DSPy version (requires DSPy 3.2+).

## Security Hall of Fame

We would like to thank the following security researchers for responsibly disclosing security issues to us.

*No security researchers have been added to the hall of fame yet. Will you be the first?*
