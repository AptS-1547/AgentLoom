# Benchmark Architecture Decision (2026-08)

> 状态：阶段性决策
>
> 适用范围：AgentLoom Runtime、Persona Gateway、LLM、Memory、Media/IPC 的组件基准与端到端压测
>
> 本文记录当前性能验证工具的边界，不改变当前业务开发优先级。

## 1. 决策摘要

当前性能已经足以支撑下一阶段业务开发，暂不把 Google Benchmark 引入作为主线工作，也不替换现有 Gateway E2E 场景 runner。

后续保留三层结构：

```text
Google Benchmark（可选）
  -> 本地组件微基准

AgentLoom StageRecorder
  -> Runtime operation 阶段时间线与阶段分布

Scenario runner / E2E harness
  -> Gateway、LLM、数据库、HTTP transport、租户隔离和生命周期压测
```

Google Benchmark 只适合函数或组件级的重复测量，不能替代跨进程异步 E2E 测试。当前优先级应转向 LLM 云并行任务异步接口、数据库重构、HTTP SSE 和 PostgreSQL 接入。

## 2. 当前工具边界

### 2.1 现有 E2E runner

`tools/gateway_concurrency_benchmark.py` 负责：

- HTTP Gateway 生命周期和并发屏障；
- Session create/chat/close 完整业务路径；
- Chat barrier、ramp-up、tenant 隔离和 memory isolation probe；
- Gateway 返回的 `pipelineLatency` 汇总；
- JSON 原始报告、分位数和资源采样。

它适合验证真实服务行为，但只能统计 Runtime 已经暴露的字段。

### 2.2 现有 C++ bench

仓库已有多个手写 C++ benchmark，覆盖 embedding batch、Session affinity、IPC、spool、media 和网络组件。它们适合验证专用组件，但输出格式、热身策略和样本统计方式还没有完全统一。

### 2.3 当前主要缺口

`ChatLatencyBreakdown` 目前主要记录 duration 聚合值，尚不能重建一个异步 Turn 的完整阶段时间线。当前性能工作的真正缺口是 Runtime 埋点，而不是更换统计框架。

后续需要按 operation 记录至少以下阶段：

```text
admission
user_emotion_queue / user_emotion_model
embedding_lookup_queue / embedding_lookup_model
memory_vector_search / memory_redis / memory_sqlite
prompt_build
llm_dispatch / llm_wait
ai_emotion_queue / ai_emotion_model
finalize / session_commit
response_ready / response_write / first_byte / response_complete
memory_admission / store_embedding / vector_write
```

阶段记录必须使用单调时钟，并保留 queued、running、completed、failed、cancelled、timed_out、skipped 等状态。异步阶段可能重叠，报告不得简单把所有阶段 duration 相加，而应同时提供 critical path 和 overlap 信息。

## 3. Google Benchmark 的使用边界

### 3.1 适合引入的场景

- Prompt 构建和 message serialization；
- `EmbeddingBatchCoordinator` 的 batch wait、batch size 和吞吐；
- Session affinity scheduler 的 enqueue/dequeue；
- continuation dispatch 和线程池任务调度；
- Answer cache key/hash 与本地 lookup；
- IPC frame claim、spool append/replay；
- 本地 vector、tokenizer、编码和压缩路径。

### 3.2 不由 Google Benchmark 承担的场景

- 启动和管理 Gateway、Emotion Server、Redis、LLM provider；
- 100/500/1000 并发的 open-loop、barrier 和资源边界压测；
- 租户隔离、Session 顺序、失败重试、shutdown/drain/replay；
- HTTP connect、request write、first byte、response complete 的跨进程时间线；
- CUDA cold/warm、模型加载和真实 Provider capacity。

如果后续引入，Google Benchmark 应作为可选测试依赖，仅加入测试 feature，不进入生产 target 和 SDK package 的强制依赖。建议使用 `AGENTLOOM_BUILD_BENCHMARKS=OFF` 默认关闭，并通过 `benchmark::benchmark` 与现有 CTest/GoogleTest 并列接入。

## 4. 报告和埋点约定

阶段性实现可继续输出现有 JSON 字段，以保持旧报告兼容；新增字段应以 operation trace 为主：

```text
trace_id / request_id / tenant_id / user_id / session_id
turn_sequence / runtime_revision / provider / model
stage / outcome / queue_depth
started_at / completed_at / duration_ms
```

每个阶段至少输出样本数、缺失数、成功/失败/取消/超时数、P50/P95/P99/P99.9、最大值，并按 cold/warm、cache hit/miss、provider/model 做可比较的分组。

原始 trace 默认写入 ignored 的 dated report 目录；正式文档只记录 schema、实验方法和结论，不写入 API key、机器私有路径或业务 payload。

## 5. 当前阶段结论

已有性能基线足以支撑下一阶段业务工作。近期不再扩展压测框架，而是保持现有 E2E runner 可复现，必要时只补最小的 transport 和 Runtime stage 埋点。

当前优先级调整为：

1. LLM 云并行任务接口的真正异步化；
2. SQLite migration/repository 边界重构和 PostgreSQL runtime 最小闭环；
3. HTTP SSE 流式响应及取消、断连、背压语义；
4. 以上业务路径稳定后，再补 StageRecorder 和 Google Benchmark 组件迁移。

## 6. 相关设计

- `docs/performance/AGENT_RUNTIME_E2E_PERFORMANCE_REPORT_2026_08.md`
- `tools/gateway_concurrency_benchmark.py`
- `src/llm/cloud_task_coordinator.h`
- `src/service/persona/persona_runtime.h`
