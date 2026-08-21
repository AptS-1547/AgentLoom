# AgentLoom Agent Runtime E2E 压测与优化报告（2026-08）

> 文档状态：正式实验报告
>
> 数据截止：2026-08-21
>
> 适用范围：Persona Gateway、Session Runtime、Emotion gRPC、L0/L3 Memory、OpenAI-compatible LLM 及资源治理边界
>
> 原始数据：本机 `data/` 与 `build/reports/` 中仍可解析和交叉核验的 E2E JSON

## 1. 执行摘要

本报告重新整理磁盘中现存的 Gateway E2E 原始结果，并结合 2026-08-21 完成的全链路异步改造与 CUDA 极限测试，得到以下结论。

1. Cloud LLM 异步化已经消除“远端等待占住有限 worker”造成的结构性排队。固定 2 秒 Provider、100 并发时，同步路径 E2E P95 为 48.132 秒，异步路径为 3.049 秒；同步 `computeQueueWait` P95 为 45.279 秒，异步为 0。
2. 当前生产链路已经覆盖异步用户 Emotion、异步 L0 Lookup、Prompt、异步 LLM、异步 AI Emotion、Session commit 和异步 Memory Store。100 并发真实 DeepSeek + Emotion CUDA + L0 CUDA 的 E2E P95 为 2.581 秒，100/100 成功，业务池 queue wait P95 均为 0。
3. 调优后的 1000 并发完整链路实现 1000/1000 Chat 成功、0 错误、0 queue wait；Chat burst 为 35.001 秒，E2E P95 为 34.270 秒，Backend P95 为 9.457 秒，LLM P95 为 2.888 秒。
4. 1000 并发证明了容量，不代表推荐 SLO。500 并发后，主要放大已从远端 LLM 转移到 Gateway pipeline 之外的入站/负载端，以及 LLM 完成后的 AI Emotion/finalize 和尚未独立计时的 Admission/commit/response 尾段。
5. 当前最高优先级不是继续增加线程，而是拆分 response-ready 与 Memory Admission、细化阶段指标、为 Emotion 建立 micro-batch、分离 Lookup/Store batch lane，并补充客户端到 Gateway pipeline 之间的 transport 分段计时。
6. CUDA 首次推理存在显著冷启动。MiniLM 在 500 并发冷轮中 Memory P95 达 40.291 秒，warm 轮降至 11 毫秒；Emotion CUDA 新 Session 首轮也曾超过 30 秒 deadline。生产 readiness 必须执行真实推理预热，不能只做 gRPC health check。

## 2. 数据治理与有效性

### 2.1 整理范围

本次整理四个仍保留原始结果的目录，共发现 196 个 JSON：

| 数据根目录 | JSON | E2E schema | 完整成功轮 | 完整成功 Chat | 边界轮 | 边界成功 Chat | 失败诊断轮 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `data/persona_gateway_async_benchmark/` | 119 | 66 | 53 | 6,042 | 6 | 2,010 | 7 |
| `data/persona_gateway_e2e/reports/` | 39 | 39 | 20 | 7,255 | 7 | 3,328 | 12 |
| `build/reports/production_stress_20260819/` | 18 | 15 | 15 | 8,041 | 0 | 0 | 0 |
| `build/reports/production_stress_20260821/` | 20 | 20 | 13 | 3,188 | 2 | 1,485 | 5 |
| 合计 | 196 | 140 | 101 | 24,526 | 15 | 6,823 | 24 |

`build/reports` 与部分 `data` 目录被 `.gitignore` 忽略，但原始数据仍存在本机磁盘。本报告只引用相对路径，不复制凭据，不保存 API key 或完整回复。

### 2.2 有效性标准

完整成功轮必须同时满足：

- JSON 可按 UTF-8 解析；
- 存在 `concurrency`、`turnsPerUser`、`totalChatRequests` 和延迟分位数字段；
- `totalChatRequests == concurrency * turnsPerUser`；
- `errorCount == 0`；
- 请求经过注册/认证、Session create、Chat、Session commit 与 close，而非直接调用模型或单元测试接口。

边界轮允许部分请求被明确拒绝，但必须产生成功 Chat，并能识别稳定容量边界，例如：

- `session affinity tenant limit reached`；
- `async gRPC in-flight limit reached`；
- 单个 setup `ReadError`/`ConnectError`，其余 Chat 完成。

以下数据不参与主性能均值：

- Persona metadata 缺失、SQLite 路径错误等夹具失败；
- Emotion CUDA 首次预热 timeout；
- `totalChatRequests == 0` 的 calibration/probe；
- Mock metrics、Gateway 配置 JSON 和数据库文件；
- embedding coordinator、模型和网络微基准。

这些文件不删除，继续作为故障诊断和复现证据。

### 2.3 证据等级

| 等级 | 含义 | 可用于 |
|---|---|---|
| A | 同一夹具、固定 Provider、只切换单一变量 | 因果判断和收益百分比 |
| B | 同一工具链、配置可核验，但日期/Provider/功能开关有变化 | 容量趋势和回归判断 |
| C | 失败轮、冷启动、过载拒绝或旧 schema | 边界、故障模式和运维要求 |

## 3. 测试环境与当前链路

### 3.1 近期生产标准环境

- 仓库基线：`98f36e43f60908560c59aba964b776763d6c3e67`；2026-08-21 数据还包含本报告前尚未提交的异步 Emotion、Memory Store 和 Turn 状态机工作区改动，因此不能仅凭该 commit 重建最终二进制；
- 设备：Lenovo 83LT，x64；
- 操作系统：Microsoft Windows 11 家庭版中文版，25H2，build `26200.9168`；
- CPU：AMD Ryzen 9 8940HX，16 核 / 32 逻辑处理器；
- 内存：16,962,326,528 bytes，约 15.80 GiB；最终极限轮前可用物理内存随其他进程变化，不作为受控常量；
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8,151 MiB VRAM，compute capability 12.0，驱动 592.01；
- 存储：UMIS RPJYJ1T24MML1AWY，约 1.024 TB；仓库位于 D: NTFS 分区，采集时分区大小约 551.6 GiB、剩余约 89.2 GiB；
- 构建：CMake 4.3.3，generator `Visual Studio 18 2026`，x64，MSVC `19.50.35724` / v145，triplet `x64-windows`；
- build 目录：`build/x64-Release-Tests-v145-refactor`；
- ONNX Runtime GPU 1.20.1，`BERT_USE_ONNXRUNTIME_GPU=ON`；
- Python 3.14.0，Git 2.50.0.windows.2；
- Redis 3.0.504，测试端点 `127.0.0.1:5000`；
- CUDA 12、cuDNN 9、cuBLAS 和 `onnxruntime_providers_cuda.dll` 从 Release 目录加载；
- MiniLM 384 维 embedding，日志确认 `provider=cuda`；
- Emotion BERT `joint_model_ir9.onnx`，日志确认 `active_provider=cuda`；
- Redis `127.0.0.1:5000`；
- 真实 LLM 为 DeepSeek OpenAI-compatible API，凭据来自 ignored key 文件；
- 压测工具为 `tools/gateway_concurrency_benchmark.py`，Chat barrier 同时起跑。

硬件与 OS 数据由 Windows CIM、注册表、`nvidia-smi`、CMakeCache 和工具版本命令于 2026-08-21 采集。该环境是本报告数值的适用边界；换用桌面 GPU、不同驱动、不同内存容量、Linux、不同 Provider 账户或不同网络后必须重新建立基线。

### 3.2 当前 Turn 时序

```text
Session admission / deferred lane
  -> User Emotion async
  -> L0/L3 Memory Lookup async
  -> Prompt Building
  -> LLM async
  -> AI Emotion async
  -> Memory Admission async
  -> Session commit
  -> HTTP response callback
```

同一 Session 从 admission 到 commit 持有 deferred scheduler fence；Emotion、Memory 和 LLM 等待不占用 worker，不同 Session 可以并行推进。

当前仍有一个重要边界：Memory Admission 位于 Session commit 和用户 response callback 之前。因此“Store 已异步”不等于“响应与 Store 已解耦”。当前 `callbackToResponse` 在 AI Emotion 后构造 `ChatResponse` 时已经冻结，不能用来测量后续 Admission/commit/HTTP write；这些阶段只能从 E2E 与现有 pipeline 指标的差值中看到混合影响。

## 4. 历史演进证据

### 4.1 同步与异步 LLM 固定延迟 A/B

2026-08-13 的 30 个矩阵文件全部成功，共 1,116 个 Chat。异步组使用 `IAsyncLlmClient + deferred completion`，同步组通过 `--sync-llm` 强制阻塞等待。

| Provider delay | 并发 | Async burst | Sync burst | 加速 | Async E2E P95 | Sync E2E P95 |
|---:|---:|---:|---:|---:|---:|---:|
| 100ms | 1 | 0.105s | 0.105s | 1.00x | 105ms | 105ms |
| 100ms | 10 | 0.111s | 0.311s | 2.80x | 110ms | 310ms |
| 100ms | 25 | 0.145s | 0.731s | 5.05x | 139ms | 626ms |
| 100ms | 50 | 0.279s | 1.362s | 4.88x | 258ms | 1,257ms |
| 100ms | 100 | 2.216s | 2.713s | 1.22x | 2,002ms | 2,538ms |
| 500ms | 1 | 0.504s | 0.504s | 1.00x | 504ms | 504ms |
| 500ms | 10 | 0.518s | 1.511s | 2.92x | 518ms | 1,510ms |
| 500ms | 25 | 0.544s | 3.532s | 6.49x | 538ms | 3,028ms |
| 500ms | 50 | 0.664s | 6.562s | 9.89x | 644ms | 6,043ms |
| 500ms | 100 | 2.430s | 12.718s | 5.23x | 2,212ms | 12,126ms |
| 2,000ms | 1 | 2.004s | 2.004s | 1.00x | 2,004ms | 2,004ms |
| 2,000ms | 10 | 2.011s | 6.012s | 2.99x | 2,010ms | 6,011ms |
| 2,000ms | 25 | 2.043s | 14.032s | 6.87x | 2,038ms | 12,032ms |
| 2,000ms | 50 | 2.161s | 26.061s | 12.06x | 2,140ms | 24,047ms |
| 2,000ms | 100 | 3.151s | 50.205s | 15.93x | 3,049ms | 48,132ms |

2 秒、100 并发是最清晰的因果证据：异步 Provider peak in-flight 为 100，同步仅为 4；同步 `computeQueueWait` P95 为 45.279 秒，异步为 0。

原始目录：

- `data/persona_gateway_async_benchmark/fake_matrix_20260813/`
- `data/persona_gateway_async_benchmark/sync_matrix_20260813/`

### 4.2 真实 DeepSeek 100 并发趋势

| 版本/链路 | 成功 | E2E P50 | E2E P95 | E2E P99 | Backend P95 | LLM P95 | Queue P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-06 同步 Gateway | 100/100 | 3,196ms | 5,171ms | 5,762ms | 4,630ms | 1,296ms | 旧 schema |
| 2026-08-13 异步 LLM | 100/100 | 2,230ms | 2,581ms | 2,697ms | 1,658ms | 1,657ms | 0ms |
| 2026-08-20 异步 LLM + CUDA L0 | 100/100 | 未单列 | 2,511ms | 未单列 | 1,669ms | 1,661ms | 0ms |
| 2026-08-21 完整 Emotion/L0/Store 异步 | 100/100 | 2,269ms | 2,581ms | 2,684ms | 1,807ms | 1,678ms | 0ms |

与 2026-06 同步链路相比，当前完整链路 E2E P95 下降 50.1%。与 2026-08-13 纯异步 LLM 相比，当前又加入真实 Emotion gRPC、Fusion、CUDA L0 Lookup/Store 和 admission，但 E2E P95 没有回归；Backend P95 增加约 149ms。

主要原始文件：

- `data/persona_gateway_e2e/reports/gateway_benchmark_1780563536.json`
- `data/persona_gateway_async_benchmark/real_api_100_20260813/result.json`
- `data/persona_gateway_async_benchmark/embedding_batch_20260820/real_gpu_l0_deepseek_c100_t1_final.json`
- `build/reports/production_stress_20260821/real_cuda_c100.json`

这是跨版本趋势，不是严格 A/B。

### 4.3 历史 1000 并发 IO worker 阻塞

2026-06 保留六组 1000 并发报告。其核心特征是 LLM P95 约 1.5–2.0 秒，但 `ioQueueWait` P95 可达 18.063 秒。

| 旧组 | 起跑 | Chat | E2E P95 | Backend P95 | IO queue P95 | LLM P95 |
|---|---|---:|---:|---:|---:|---:|
| A | 瞬时 | 1000/1000 | 47,666ms | 19,503ms | 18,063ms | 1,551ms |
| A | 60s ramp | 1000/1000 | 17,740ms | 17,723ms | 16,426ms | 1,494ms |
| B | 瞬时 | 1000/1000 | 35,843ms | 8,173ms | 6,667ms | 1,676ms |
| B | 60s ramp | 1000/1000 | 4,336ms | 4,291ms | 2,743ms | 1,775ms |
| C | 瞬时 | 999/1000 | 45,422ms | 16,186ms | 13,960ms | 1,999ms |
| C | 60s ramp | 1000/1000 | 2,188ms | 2,168ms | 655ms | 1,817ms |

Ramp 能缓解但不能修复同步等待占住 worker。当前 1000 并发完整异步轮的 compute/io queue P95 均为 0。

重点原始文件：`gateway_benchmark_1780566147`、`1780566238`、`1780566801`、`1780566878`、`1780567065`、`1780567158`。

### 4.4 企业 500 与 Mock 实现

| 场景 | Chat | Provider peak | Burst | E2E P95 | Pipeline/LLM P95 | RSS |
|---|---:|---:|---:|---:|---:|---:|
| Python aiohttp，2s，C250 | 250/250 | 250 | 4.021s | 3,792ms | 2,013ms | 31.5MiB |
| Python aiohttp，2s，C500 | 500/500 | 500 | 10.960s | 9,978ms | 2,014ms | 40.9MiB |
| 调优 Gateway + Python，C500 | 499/500 | 499 | 10.769s | 10,005ms | 2,013ms | 47.4MiB |
| 调优 Gateway + C++ Mock，C500 | 499/500 | 499 | 10.228s | 9,725ms | 2,015ms | 49.8MiB |

C++ Mock 比 Python 单轮 E2E P95 低约 280ms，但两者都出现一个 setup ReadError，说明数秒额外延迟不由 Mock 实现方式单独决定。

### 4.5 Keep-alive 对照

固定 500ms Provider、100 Session × 2 turn、开关各三轮，共 1,200 个 Chat 全部成功：

| Keep-alive | Provider 连接数中位数 | 冷轮 E2E P95 | 热轮 E2E P95 | Backend/LLM P95 | Queue P95 |
|---|---:|---:|---:|---:|---:|
| 关闭 | 200 | 1,463ms | 1,956ms | 514–515ms | 0ms |
| 开启 | 65 | 1,470ms | 2,015ms | 514–515ms | 0ms |

Keep-alive 将连接建立减少约 67.5%，但没有降低 E2E 尾延迟。它解决连接 churn，不是当前入站长尾的充分条件。

## 5. L0、CUDA 与 Memory 隔离

### 5.1 2026-08-19 CPU/CUDA 三轮对话

每个虚拟用户执行三轮对话并检查私密 marker。性能结果与 Memory isolation 断言需要分别解释：Chat 全部成功不等于 marker 断言全部通过。

| Provider | 并发 | Chat | E2E P95 | Backend P95 | LLM P95 | Memory P95 | Queue P95 | RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU Fake 500ms | 10 | 30/30 | 578ms | 558ms | 524ms | 35ms | 36ms | 810MiB |
| CPU Fake 500ms | 100 | 300/300 | 3,141ms | 536ms | 515ms | 21ms | 21ms | 818MiB |
| CPU Fake 500ms | 250 | 750/750 | 7,995ms | 598ms | 563ms | 42ms | 50ms | 826MiB |
| CPU Fake 500ms | 500 | 1500/1500 | 22,106ms | 596ms | 554ms | 49ms | 64ms | 832MiB |
| CUDA Fake cold | 10 | 30/30 | 38,751ms | 38,735ms | 554ms | 38,181ms | 38,182ms | 2,201MiB |
| CUDA Fake warm | 100 | 300/300 | 2,605ms | 656ms | 534ms | 106ms | 114ms | 1,394MiB |
| CUDA Fake warm | 250 | 750/750 | 2,840ms | 2,690ms | 2,240ms | 112ms | 969ms | 1,401MiB |
| CUDA Fake warm | 500 | 1500/1500 | 9,107ms | 6,461ms | 4,702ms | 109ms | 2,148ms | 1,415MiB |
| Real CPU DeepSeek | 100 | 300/300 | 1,980ms | 1,085ms | 1,075ms | 46ms | 48ms | 849MiB |
| Real CPU DeepSeek | 250 | 750/750 | 8,400ms | 1,534ms | 1,517ms | 44ms | 45ms | 883MiB |
| Real CPU DeepSeek | 500 | 1500/1500 | 24,353ms | 1,479ms | 1,457ms | 47ms | 48ms | 880MiB |

Isolation 原始字段如下：

| 场景 | Checked | Report passed | Report failed | 解释 |
|---|---:|---:|---:|---|
| CPU Fake C500 | 1,500 | 1,500 | 0 | 零泄漏强证据 |
| CUDA Fake C500 | 1,500 | 1,500 | 0 | 零泄漏强证据 |
| Real CPU C100 | 300 | 244 | 56 | 保存样本为 turn 0 未回显，不是外租户 marker |
| Real CPU C250 | 750 | 638 | 112 | 保存样本为 turn 0 回显自己的 marker，属于旧断言语义 |
| Real CPU C500 | 1,500 | 1,500 | 0 | 零泄漏强证据 |

C100/C250 只保存前 20 个失败样本，完整 `records` 为空，无法用当前规则重算全部 56/112 个断言。因此这两轮不能写成“全部通过”，也不能据此声称发生跨租户泄漏；它们仅作为旧版召回/断言正确性问题保留。

该批数据说明：CUDA cold start 必须从稳定态性能中剥离；100–500 并发时 E2E 与 Backend 的差距远大于 Memory P95；CUDA 的收益依赖预热、batch 和并发形态。

### 5.2 Embedding batch A/B

同一 CUDA Gateway、100 Session × 3 turn，只切换 `embedding_batch.enabled`：

| 指标 | Batch Off | Batch On | 变化 |
|---|---:|---:|---:|
| Chat | 300/300 | 300/300 | 一致 |
| Burst | 7.108s | 5.851s | -17.7% |
| E2E P95 | 2,406ms | 1,842ms | -23.5% |
| Backend P95 | 1,090ms | 776ms | -28.8% |
| Memory P95 | 574ms | 271ms | -52.8% |
| RSS | 1,334MiB | 1,351MiB | +17MiB |

Embedding batch 对 Memory 和 Backend 有明确收益，应保持默认开启。

原始数据：

- `data/persona_gateway_async_benchmark/embedding_batch_20260820/ab_gpu_l0_batch_off_c100_t3.json`
- `data/persona_gateway_async_benchmark/embedding_batch_20260820/ab_gpu_l0_batch_on_c100_t3.json`

真实 CUDA isolation 有效轮 `real_gpu_l0_deepseek_c20_t3_isolation_valid.json` 完成 60/60 Chat，60/60 reply 无泄漏。

## 6. 2026-08-21 完整异步链路

### 6.1 Fake CUDA 500 冷/热

配置为 C++ Mock、2,000ms 固定延迟、quota 500、真实 CUDA L0 Lookup/Store；Emotion 关闭以隔离 LLM + Memory 主链。

| 轮次 | Chat | Burst | E2E P95 | Backend P95 | Memory P95 | LLM P95 | Callback P95 | Queue P95 | RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cold | 500/500 | 50.905s | 50,200ms | 42,311ms | 40,291ms | 2,019ms | 1ms | 0ms | 2,181MiB |
| Warm | 500/500 | 10.398s | 9,678ms | 2,024ms | 11ms | 2,015ms | 1ms | 0ms | 1,401MiB |

Warm 轮证明 LLM 等待没有占用业务 worker；Cold 轮证明仅 health check 不足以代表 CUDA readiness。

### 6.2 真实 Emotion + L0 + DeepSeek 阶梯

所有运行均包含真实 Emotion gRPC、Fusion、CUDA embedding、Redis/SQLite L0、真实 DeepSeek、AI Emotion、异步 Store 和 Session commit。

| 并发 | Chat | Burst | Chat RPS | E2E P95 | Backend P95 | Compute P95 | Memory P95 | LLM P95 | Callback P95 | Queue P95 | RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 1.446s | 0.69 | 1,446ms | 1,415ms | 86ms | 60ms | 1,328ms | 18ms | 0ms | 2,017MiB |
| 10 | 10/10 | 1.729s | 5.78 | 1,728ms | 1,689ms | 406ms | 197ms | 1,326ms | 27ms | 0ms | 2,028MiB |
| 25 | 25/25 | 2.157s | 11.59 | 2,040ms | 1,997ms | 486ms | 63ms | 1,519ms | 86ms | 0ms | 2,035MiB |
| 50 | 50/50 | 1.905s | 26.25 | 1,834ms | 1,712ms | 392ms | 105ms | 1,447ms | 126ms | 0ms | 2,040MiB |
| 100 | 100/100 | 2.813s | 35.55 | 2,581ms | 1,807ms | 284ms | 82ms | 1,678ms | 104ms | 0ms | 2,050MiB |
| 200 | 200/200 | 3.922s | 51.00 | 3,778ms | 2,954ms | 1,754ms | 215ms | 1,696ms | 627ms | 0ms | 2,073MiB |
| 300 | 300/300 | 6.168s | 48.64 | 5,870ms | 3,246ms | 2,180ms | 206ms | 1,366ms | 1,128ms | 0ms | 2,079MiB |
| 500 | 500/500 | 13.508s | 37.02 | 13,012ms | 4,797ms | 3,731ms | 300ms | 1,360ms | 2,811ms | 0ms | 2,123MiB |
| 1000 full tuned | 1000/1000 | 35.001s | 28.57 | 34,270ms | 9,457ms | 8,405ms | 134ms | 2,888ms | 8,145ms | 0ms | 2,193MiB |

### 6.3 1000 并发资源边界

| Gateway/Emotion 配置 | 成功 | 失败 | 明确边界 |
|---|---:|---:|---|
| Gateway tenant ceiling 512 | 815 | 185 Chat + close 错误 | `session affinity tenant limit reached` |
| Gateway ceiling 4096，Emotion 默认 inflight | 670 | 330 Chat | `async gRPC in-flight limit reached` |
| Gateway ceiling 4096，Emotion 32 worker / 1024 inflight | 1000 | 0 | 无拒绝 |

资源治理按预期逐层工作。提升上限后 1000 请求全部完成，但代价是更高线程数、内存和尾延迟：

- Gateway HTTP/compute/IO/LLM worker：32 / 16 / 32 / 64；
- Gateway 线程采样约 214；
- Gateway peak RSS 2,192.6MiB；
- Gateway peak private bytes 5,262.6MiB；
- Emotion 32 worker，runtime inflight ceiling 1,024；
- E2E P95 34.270 秒。

1000 是极限承载证明，不是默认租户配额建议。

## 7. 当前瓶颈

### 7.1 Worker 等待问题已经解决

完整链路从 C1 到 C1000 的 `computeQueueWait` 和 `ioQueueWait` P95 始终为 0。阶段 wall time 增加不能解释为 worker 被外部等待占住；deferred 设计有效。

### 7.2 E2E 到 Backend 之间仍有巨大缺口

| 并发 | E2E P95 | Backend P95 | 差值 |
|---:|---:|---:|---:|
| 100 | 2,581ms | 1,807ms | 774ms |
| 200 | 3,778ms | 2,954ms | 824ms |
| 300 | 5,870ms | 3,246ms | 2,624ms |
| 500 | 13,012ms | 4,797ms | 8,215ms |
| 1000 | 34,270ms | 9,457ms | 24,813ms |

`backendTotal` 当前在 AI Emotion、Memory Admission、Session commit 和 HTTP write 完成前就确定。因此差值可能同时来自两端：一端是负载端开始计时到 Gateway pipeline 开始，另一端是 pipeline 当前截止点之后的 AI Emotion、Admission、commit、响应回写和客户端调度。现有数据不能将其单独归因给 Python/httpx、Windows socket、Gateway accept/read、认证 Redis、Emotion、Admission 或响应背压。

### 7.3 LLM 不是 500/1000 的主长尾

500 并发 LLM P95 为 1.360 秒，1000 为 2.888 秒；对应 E2E P95 为 13.012 和 34.270 秒。继续增加 LLM worker 不能解决主要尾延迟。

### 7.4 AI Emotion 与未计时的响应尾段成为瓶颈

`callbackToResponse` P95 从 C100 的 104ms 增至 C500 的 2.811 秒和 C1000 的 8.145 秒。按当前实现，它从 LLM finalize 前的 `total` 截止点计到 AI Emotion 完成并构造 `ChatResponse`，主要混合：

- LLM callback dispatch；
- AI Emotion gRPC；
- Emotion Fusion/Vector evidence；
- response 对象构造前的 finalize。

它不包含之后的 Memory Admission、Store embedding、Session commit 和 HTTP write。后四者当前没有独立指标，但 response callback 确实要等 Admission 和 commit 收口，尚未满足“响应不等待 Store”的设计目标。因此 C1000 的 8.145 秒首先指向 AI Emotion/fusion 高压成本；其余未解释 E2E 长尾还需要 `response_ready`、Admission、commit 和 transport 指标才能拆开。

### 7.5 Emotion 尚未使用 batch RPC

Emotion Server 支持 `PredictEmotionBatch`，但本轮 stats 的 `max_batch_size=1`。提高 worker/inflight 能提高上限，却让同一 CUDA Session 承受大量单条 `Run()`。micro-batch 是下一步合理优化。

### 7.6 Lookup 与 Store 共享协调器

当前 L0 `LookupAsync` 和 `StoreAsync` 共享 embedding coordinator。高压后台 Store 可能与下一轮 Lookup 竞争 batch/inflight；设计要求的 Lookup、Store、offline 分 lane 尚未完成。

### 7.7 CUDA readiness 不完整

两个独立数据集都复现首次 CUDA 推理超过 30–40 秒。Health check 只证明 listener 可用，不证明 kernel/allocator 已就绪。

## 8. 优化计划

### P0：响应与持久化解耦

目标时序：

```text
LLM success
  -> AI Emotion / state update
  -> Session commit
  -> response-ready
  -> background Memory Admission
  -> Store embedding / vector / Redis / SQLite
```

要求：

- `response_success` 与 `memory_admission_success` 分开；
- Store 失败记录结构化日志并进入 maintenance/retry，不修改已提交 Turn；
- 同 Session Store 按 turn sequence 排序，但不阻塞响应；
- shutdown 使用 admission drain fence，保证 exactly once。

### P0：重构阶段指标

至少新增：

- `user_emotion_queue_wait_ms` / `user_emotion_model_ms`；
- `embedding_lookup_batch_wait_ms` / `embedding_lookup_model_ms`；
- `memory_vector_search_ms` / `memory_sqlite_ms` / `memory_redis_ms`；
- `prompt_build_ms`；
- `llm_dispatch_ms` / `llm_wait_ms`；
- `ai_emotion_queue_wait_ms` / `ai_emotion_model_ms`；
- `session_commit_ms`；
- `response_ready_ms` / `response_write_ms`；
- `memory_admission_queue_wait_ms`；
- `store_embedding_model_ms` / `vector_write_ms`。

同时修正 `computeStage` 与 `callbackToResponse` 语义，禁止一个字段混合多个异步 wall-time。

### P0：客户端与入站 transport 分段

负载发生器需要记录 DNS/connect、connection acquired、request write、Gateway accepted、auth complete、pipeline start、first byte 和 response complete。考虑增加 C++ load generator，与 Python/httpx 交叉验证。

### P1：Emotion micro-batch

- Gateway 增加 Emotion coordinator；
- 调用 `PredictEmotionBatch`；
- user/AI Emotion 分优先级 lane；
- 设置 1–2ms max wait、合理 batch size 和 bounded queue；
- 保持同 Session continuation 顺序。

### P1：Lookup/Store/offline 分 lane

- Lookup：最高优先级、短 max wait；
- Store：有界、可降级、响应后执行；
- Offline：文档/历史 replay 最低优先级；
- 分别统计 queue、batch、model 和 write。

### P1：启动预热与 readiness

- Emotion Server 启动后执行真实 CUDA inference；
- Gateway 执行 representative lookup embedding；
- 完成 SQLite schema、Redis ping 和连接池预建；
- 预热成功后才置为 serving；
- 预热 deadline 与线上请求 deadline 分开。

### P1：容量配置按 SLO 分档

| Profile | 用途 | HTTP / compute / IO / LLM | Tenant ceiling | Emotion inflight |
|---|---|---|---:|---:|
| Normal | 100–200 活跃 Chat | 16 / 8 / 16 / 32 | 256–512 | 256 |
| High | 300–500 burst | 32 / 16 / 32 / 64 | 1,024–2,048 | 512 |
| Extreme | 1000 容量验证 | 32 / 16 / 32 / 64 | 4,096 | 1,024 |

最终值必须按目标硬件和 SLO 校准。明确拒绝比无限排队健康；不能因为极限轮成功就永久放开上限。

### P2：横向扩展与恢复

- Emotion 多实例 + gRPC load balancing；
- Memory Admission durable spool/replay；
- Store retry budget 与 dead-letter maintenance；
- Runtime restart 后按 fencing token 恢复 admission；
- 多 tenant 配额与隔离压测扩展至 500/1000 并发。

## 9. 下一轮验收基线

- C100 真实完整链路：100/100，E2E P95 不高于 2.7 秒；
- C500：500/500，无 queue wait；
- C1000：无 crash、无重复 callback，拒绝必须是结构化 ResourceExhausted；
- Fake C500 × 2,000ms warm：LLM P95 约 2.0 秒、queue P95 0；
- same-session order 100%；
- cross-tenant memory leakage 0；
- shutdown 无悬挂 callback、无 crash dump；
- response-ready 不等待 Memory Admission；
- Cold start 在 readiness 阶段完成，不进入用户请求。

## 10. 原始证据索引

### 同步/异步与 transport

- `data/persona_gateway_async_benchmark/fake_matrix_20260813/`
- `data/persona_gateway_async_benchmark/sync_matrix_20260813/`
- `data/persona_gateway_async_benchmark/enterprise_quota_500_20260813/`
- `data/persona_gateway_async_benchmark/enterprise_quota_500_tuned_20260813/`
- `data/persona_gateway_async_benchmark/cpp_quota_500_20260813/`
- `data/persona_gateway_async_benchmark/keepalive_e2e_20260813/`
- `data/persona_gateway_async_benchmark/real_api_100_20260813/`

### 历史 Gateway

- `data/persona_gateway_e2e/reports/`
- 同步真实 C100：`gateway_benchmark_1780563536.json`
- 1000 queue：`1780566147`、`1780566238`、`1780566801`、`1780566878`、`1780567065`、`1780567158`

### L0、隔离与 embedding batch

- `build/reports/production_stress_20260819/`
- `data/persona_gateway_async_benchmark/embedding_batch_20260820/`

### 当前完整链路

- `build/reports/production_stress_20260821/`
- 稳定阶梯：`real_cuda_c10`、`c25`、`c50`、`c100`、`c200`、`c300`、`c500`
- 最终极限：`real_cuda_c1000_full_tuned.json`
- 容量边界：`real_cuda_c1000_extreme.json`、`real_cuda_c1000_tuned.json`
- CUDA 冷/热：`fake_cuda_c500_d2000.json`、`fake_cuda_c500_d2000_warm.json`

## 11. 最终结论

Agent Runtime 已从“有限 worker 同步等待外部系统”迁移到“同 Session 严格保序、跨 Session 异步推进”的结构。固定延迟 A/B、真实 DeepSeek、CUDA L0、真实 Emotion gRPC 和 1000 并发极限轮共同证明：外部等待不再形成业务池 queue wait，资源治理能明确拒绝过载，调优后可完成 1000/1000 全链路请求。

下一阶段不应继续以增加线程为主。正式优化方向是缩短 response critical path、拆分可观测阶段、批处理 Emotion、隔离 Lookup/Store lane，并定位客户端到 Gateway pipeline 之间的 transport 长尾。只有完成这些工作，1000 并发的“可以完成”才能转化为可接受的交互 SLO。
