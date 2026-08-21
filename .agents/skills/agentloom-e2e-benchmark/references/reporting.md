# Benchmark Data And Reporting Rules

Read this reference when auditing JSON or writing a formal report.

## 1. Classes

**Complete success:** parsable UTF-8 Gateway report, planned Chat count, zero errors, meaningful latency fields.

**Capacity boundary:** successful Chat samples plus explicit overload evidence such as tenant/fairness or async gRPC inflight limits.

**Diagnostic failure:** zero-Chat setup error, missing Persona metadata, bad SQLite path, CUDA prewarm timeout, unavailable Provider, or load-generator failure. Preserve it; do not average it into stable results.

**Not Gateway E2E:** config JSON, Mock metrics, databases, model microbenchmarks, unit tests, and coordinator-only benchmarks.

## 2. Comparability

- Controlled A/B keeps hardware, warm state, config, data, Provider, load shape, and tool fixed; change one variable.
- Cross-version trend states all changed dimensions and does not claim causal speedup.
- Cold and warm are both valid but remain separate.
- Real and Fake distributions never combine.
- Chat availability and semantic correctness are separate; report Memory isolation/retrieval independently.

## 3. Required Environment

Record date/timezone, commit and uncommitted code, OS/machine/CPU/RAM, GPU/VRAM/driver, disk/filesystem, CMake/generator/MSVC/triplet, ONNX Runtime/CUDA, Redis, Provider/model, build/config paths, pool sizes, queues, and admission ceilings.

If a value cannot be collected, say so. Never infer physical RAM or Provider quota from observed usage. Third-party documentation uses repository-relative or placeholder paths, not machine-only absolute paths.

## 4. Required Metrics

- planned/completed Chat, success rate, setup/Chat/close errors;
- Chat burst and throughput;
- E2E P50/P95/P99/max;
- backend, compute queue/stage, IO queue/stage, Memory, LLM, callback P95;
- RSS/private bytes/thread peaks;
- Provider inflight/queued/rejected when available;
- Memory isolation checked/passed/failed separately;
- crash/dump and shutdown status.

Interpret metrics from code, not names alone. Current `backendTotal` and `callbackToResponse` do not cover every later Admission/commit/write stage.

## 5. Formal Report Shape

1. Executive summary;
2. data inventory and validity;
3. exact environment and provenance;
4. controlled A/B;
5. historical trends with comparability warnings;
6. current ladder and capacity boundaries;
7. semantic correctness/isolation;
8. measured bottlenecks;
9. prioritized optimization and acceptance baselines;
10. relative raw paths and reproduction commands.

Never include API keys, tokens, private dialogue, or credential-file contents. Redact replies unless they are synthetic markers.
