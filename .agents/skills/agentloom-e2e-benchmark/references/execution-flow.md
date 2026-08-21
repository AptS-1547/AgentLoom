# AgentLoom E2E Execution Flow

Read this reference before starting benchmark processes.

## 1. Preflight

1. Inspect `git status --short`; preserve user changes.
2. Read target configs and the prior dated report for the same scenario.
3. Confirm Redis with `redis-cli -p 5000 ping`.
4. Inspect the selected `CMakeCache.txt` for VS2026/x64, ABI-compatible dependencies, `x64-windows`, `BERT_USE_ONNXRUNTIME_GPU=ON`, and the intended ONNX Runtime root.
5. Confirm the Release directory contains the ONNX CUDA, CUDA 12, cuBLAS, and cuDNN runtime DLLs.
6. Check ports and existing processes. Reuse only a process whose config is known.
7. Create the dated report directory before invoking the load generator.

Resolve a CMake version that supports `Visual Studio 18 2026`. Do not use an older CMake or a VS2022/v143 build directory.

## 2. Build And Regression

Build the existing GPU-enabled Release targets:

```powershell
cmake --build build/x64-Release-Tests-v145-refactor --config Release `
  --target service_tests vector_tests persona_gateway_e2e_server `
           emotion_inference_server openai_compatible_quota_mock_server `
  --parallel 1
```

Run focused Persona/Emotion/Memory/Embedding tests before capacity work. Retry a transient MSBuild temp permission failure without changing code; do not describe an old executable as newly built.

## 3. Fake LLM Baseline

Start the C++ quota mock with explicit delay, quota, and IO threads. The standard saturation baseline is 2,000ms delay and quota 500.

Start Gateway with Emotion disabled and production-standard CUDA L0. Confirm `TextEmbedding ... provider=cuda`.

Run a cold round and an identical warm round without restarting Gateway or Mock. Cold is readiness evidence; warm is the stable baseline.

Collect Mock peak inflight/queued/completed/rejected, Chat success/errors and burst, E2E/backend/queue/LLM/Memory/callback P95, and Gateway RSS/private bytes/thread peak.

## 4. Real CUDA Full Flow

1. Start `emotion_inference_server` with the intended config.
2. Confirm gRPC health is `SERVING` and logs say `requested_provider=cuda, active_provider=cuda`.
3. Send real inference warm-up. If the first request times out during CUDA cold start, keep the process alive and repeat; stable collection starts only after success.
4. Start/restart Gateway. Confirm CUDA L0, gRPC Emotion/fusion, async Persona LLM, Persona metadata, and writable SQLite paths.
5. Send one complete Gateway warm-up after each Gateway restart.
6. Run a capacity ladder such as `10 -> 25 -> 50 -> 100 -> 200 -> 300 -> 500` without restarting services.
7. Run 1,000 only when requested or needed for capacity validation. Preserve resource ceilings first; structured rejection identifies the real boundary.

Separate real-provider 429, model timeout, local admission rejection, SQLite failure, and load-generator transport failure.

## 5. Limit Tuning

Change one boundary family at a time:

1. Gateway Session affinity/fairness/tenant limits and queues;
2. Gateway HTTP/compute/IO/LLM worker counts;
3. Emotion worker count and derived async gRPC inflight ceiling.

After a model-process restart, warm again. A successful tuned 1,000 run is capacity evidence, not a recommended default SLO.

## 6. Shutdown And Evidence

- Stop Gateway, then Emotion and Mock; verify no process remains.
- Check logs for crash, duplicate callback, timeout, ResourceExhausted, SQLite, Redis, and admission failures.
- Check dumps for new crashes.
- Preserve setup/prewarm failures as diagnostics, but exclude them from stable tables.
- Run focused regression after code changes discovered during E2E.
