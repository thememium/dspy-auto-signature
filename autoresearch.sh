#!/usr/bin/env bash
# Canonical benchmark entrypoint for dspy-auto-signature generation latency.
# Deterministic: fixed fixtures, offline canned LM, no network, no time deps.
set -euo pipefail
cd "$(dirname "$0")"
exec uv run python benchmarks/bench_generate.py
