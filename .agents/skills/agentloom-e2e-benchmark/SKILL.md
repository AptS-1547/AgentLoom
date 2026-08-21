---
name: agentloom-e2e-benchmark
description: Run, audit, compare, and document AgentLoom Persona Gateway E2E performance tests, including Fake LLM baselines, CUDA Emotion/L0 warm-up, real-provider capacity ladders, overload boundaries, and formal reports. Use for AgentLoom Gateway load tests and benchmark-data audits; do not use for isolated model microbenchmarks or unrelated unit-test performance.
---

# AgentLoom E2E Benchmark

Use the repository's existing Gateway E2E server, quota mock, Emotion server, configs, load generator, and report schema. Do not create a parallel harness unless the existing flow cannot measure a required boundary.

## Required Invariants

- Read the repository `AGENTS.md`; reuse `tools/gateway_concurrency_benchmark.py`, `tools/persona_gateway_e2e_server.cpp`, and existing config examples.
- On Windows, use the VS2026/v145 build and a GPU-enabled CMakeCache. Confirm logs say `provider=cuda` and Emotion says `active_provider=cuda`; DLL presence alone is not evidence.
- Record machine, OS, CPU/RAM/GPU, driver, CMake/MSVC, ONNX Runtime, build directory, code/worktree provenance, models, Provider, and relevant pool/limit settings.
- Treat CUDA health and CUDA readiness separately. After every model-process restart, send real inference warm-up requests until one succeeds before collecting stable data.
- Preserve same-Session ordering. External waits may hold a deferred Session fence but must not occupy a worker.
- Keep Fake and real-provider results separate. Never spend cloud quota for a Fake baseline.
- Do not expose API keys, tokens, full private replies, or credential-file contents.
- Stop all benchmark processes after collection unless the user asks to keep them running.
- Calculate improvements only for controlled A/B runs; label cross-version observations as trends.

## Modes

- **Execute:** Read [references/execution-flow.md](references/execution-flow.md) before starting services or load.
- **Audit/report:** Read [references/reporting.md](references/reporting.md) before classifying JSON or writing conclusions.
- **Summarize:** Run `scripts/summarize_gateway_reports.py` over dated report roots; inspect boundary errors before accepting classification.

## Completion

Return the exact config/build used, warm-up status, requested/completed Chat counts, error distribution, burst throughput, P50/P95/P99, backend/queue/LLM/Memory/callback metrics, memory/thread peaks, raw report paths, stopped-process status, and limitations. Update `docs/performance/README.md` for a new formal baseline.
