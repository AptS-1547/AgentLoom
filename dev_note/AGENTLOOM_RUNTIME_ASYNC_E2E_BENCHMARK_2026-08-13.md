# AgentLoom Runtime 异步 Persona E2E 压测报告

日期：2026-08-13
范围：Persona Session 长锁第一阶段、异步 Cloud LLM 接入、不同 Session 并发和 Provider 高并发容量
不在范围：入站 HTTP Server 重构、SQLite 轻量后端治理、Application 下游、异步 fallback gRPC 组合

## 1. 结论先行

本轮第一阶段目标已通过：纯 Cloud OpenAI-compatible Gateway 的 Chat 主链路已经真正异步化。100 个不同 Session 同时发起 Chat 时，云 LLM 等待不再占用 `gateway-llm-pool` worker；同 Session 保序、不同 Session 并发、主动关闭取消和 callback 恰好一次由专项测试覆盖，完整 Gateway E2E 又验证了认证、Session 创建、Chat、LLM HTTP、commit 和关闭链路。

最有区分度的 Fake A/B 结果如下。两组使用相同 Gateway 夹具、Redis 鉴权、相同 2 秒 Provider 延迟和相同并发客户端；历史 A/B 夹具实际使用 4 个 LLM worker，异步组的 outbound HTTP runtime 为 1 个 IO thread，同步组强制使用同步 `ILlmClient`：

| Chat 并发 | 异步完成窗口 | 同步完成窗口 | burst 加速 | 异步 P95 | 同步 P95 |
|---:|---:|---:|---:|---:|---:|
| 10 | 2.011 s | 6.012 s | 2.99x | 2.010 s | 6.011 s |
| 25 | 2.043 s | 14.032 s | 6.87x | 2.038 s | 12.032 s |
| 50 | 2.161 s | 26.061 s | 12.06x | 2.140 s | 24.047 s |
| 100 | 3.151 s | 50.205 s | 15.93x | 3.049 s | 48.132 s |

这不是把同步等待简单搬到另一个线程池：异步组的 Mock Provider 峰值在途请求达到 100，而同步组稳定只有 4 个在途请求，正好对应同步 LLM worker 数。异步组在 100×2 秒时完成窗口约 3.15 秒；同步组约 50.21 秒，已经接近 `ceil(100 / 4) * 2s` 的串行批次行为。

真实 API 100 并发也通过：

| 指标 | 结果 |
|---|---:|
| 不同 Session / Chat 请求 | 100 |
| 成功 | 100 |
| 失败 | 0 |
| Chat burst 完成窗口 | 2.843 s |
| Chat P50 / P95 / P99 | 2230 / 2581 / 2697 ms |
| Provider pipeline `llmTotal` P50 / P95 | 1440 / 1657 ms |
| `ioQueueWait` P95 | 0 ms |
| Gateway 峰值 RSS | 39.7 MiB |
| Gateway 线程数采样范围 | 28–31 |

真实 API 结果只代表当前个人 API 配额和当次 Provider 状态，不外推企业 500 并发配额。

更早的 1000 并发压测还保留了旧同步 IO pool 的直接埋点：在 2 个 IO worker、8 个 compute
worker 的旧路径中，三轮瞬时起跑的 `ioQueueWait` P95 分别达到 18.063/6.667/13.960 秒；
对应 60 秒 ramp 轮次仍有 16.426/2.743/0.655 秒。当前异步真实 API、固定延迟异步组和
500 配额组的 `ioQueueWait` P95 均为 0。该证据与 Fake A/B 一致地说明长延迟网络调用占用
worker 会形成显著本地排队，但历史 1000 并发配置不同，不用于计算当前版本的直接加速比。

更早的同步真实 DeepSeek 100 并发报告仍保存在
`data/persona_gateway_e2e/reports/gateway_benchmark_1780563536.json`。旧结果的 Provider
`llmTotal` 更低，但端到端尾延迟显著更高：当前异步路径的 E2E P95 从 5171 ms 降至
2581 ms，降低 50.1%；P99 从 5762 ms 降至 2697 ms，降低 53.2%。这说明收益来自消除
本地同步排队放大，而不是来自当次 Provider 响应更快。完整口径和限制见 5.2。

HTTP/1.1 outbound keep-alive 空闲连接池也已完成并通过三轮开关对照。100 个 Session 各执行
2 轮 500 ms Chat 时，关闭连接池每轮为 200 个 Chat 建立 200 条 Provider 连接；开启后仅建立
65/67/55 条，中位数由 200 降至 65，减少 67.5%。两组 `backendTotal`/`llmTotal` P95 均稳定在
514 至 515 ms，`computeQueueWait`/`ioQueueWait` P95 均为 0；本轮 E2E 尾延迟没有随连接数下降，
因此该结果证明的是连接复用和资源治理有效，不宣称当前 100 并发场景的延迟收益。完整结果见 6.4。

## 2. 测试环境与夹具

- Windows 11 10.0.26200，AMD Ryzen 9 8940HX，16 核 / 32 逻辑处理器，15.8 GiB 可见内存；
- VS2026/v145，Release，独立目录 `build/x64-Release-Tests-v145-refactor`；
- CMake 4.3.3；
- Gateway E2E Server 使用 AgentLoom `HttpServer`，Redis 鉴权 / Session backend 为 `127.0.0.1:5000`；
- Emotion analyzer、L0 memory、Skill Session、静态文件均关闭，避免 BERT、Redis memory 和前端静态资源污染 LLM 调度数据；
- 默认 Persona 由 E2E 夹具配置预置，只读用于 Session create；压测默认不调用低频的 `/api/persona` upsert；
- 每个虚拟用户拥有独立 `userUuid`、token、Session 和 trace；所有 Chat 在初始化完成后通过 barrier 同时起跑；
- setup、Chat、close 均经过 HTTP Gateway。报告中的 Chat burst 窗口从首个 Chat 发出开始，到最后一个 Chat 完成结束，不包含 setup 和 close；
- 原始 JSON、Mock metrics 和过程日志位于被 `.gitignore` 忽略的 `data/persona_gateway_async_benchmark/` 下。

关键夹具：

- [异步 / 高并发夹具](../tools/persona_gateway_async_benchmark.example.json)
- [真实 API 夹具](../tools/persona_gateway_async_real_benchmark.example.json)
- [旧 A/B 结果的可复现夹具](../tools/persona_gateway_async_benchmark_legacy.example.json)
- [并发压测脚本](../tools/gateway_concurrency_benchmark.py)
- [Python 固定延迟 Mock](../tools/openai_compatible_delay_server.py)
- [Python asyncio 高配额 Mock](../tools/openai_compatible_quota_server.py)
- [C++ AgentLoom HTTP Server 高配额 Mock](../tools/openai_compatible_quota_mock_server.cpp)

## 3. Fake 延迟分级 A/B

### 3.1 执行模式

延迟档位为 100 / 500 / 2000 ms，并发档位为 1 / 10 / 25 / 50 / 100，每格一轮 Chat，共 15 格；异步和同步各跑一遍，共 30 格、558 个 Chat/模式、1116 个 Chat 总计。

- 异步：`IAsyncLlmClient -> AsyncBeastHttpClient -> deferred completion -> Session commit`；
- 同步：`ILlmClient -> BeastHttpClient -> 同步 Turn task`，通过 E2E Server 的 `--sync-llm` 开关强制；
- 两组都使用 Redis 鉴权；
- A/B 结果产生时的 Gateway 参数：入站 HTTP 8 threads、IO/LLM 实际 4 workers、异步 outbound HTTP 1 IO thread；
- 所有 30 格均成功，Chat error count 为 0；
- `peakInflight` 来自 Provider Mock metrics，不是估算值。

### 3.2 完整矩阵

窗口单位为秒，P95 单位为毫秒；`speedup` 为同步窗口 / 异步窗口。

| Delay | Concurrency | Async burst | Sync burst | Speedup | Async P95 | Sync P95 | Async peak in-flight | Sync peak in-flight |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 ms | 1 | 0.105 | 0.105 | 1.00x | 105.0 | 105.1 | 1 | 1 |
| 100 ms | 10 | 0.111 | 0.311 | 2.80x | 110.3 | 310.0 | 10 | 4 |
| 100 ms | 25 | 0.145 | 0.731 | 5.05x | 138.7 | 626.1 | 25 | 4 |
| 100 ms | 50 | 0.279 | 1.362 | 4.88x | 257.6 | 1257.5 | 38 | 4 |
| 100 ms | 100 | 2.216 | 2.713 | 1.22x | 2001.8 | 2538.3 | 12 | 4 |
| 500 ms | 1 | 0.504 | 0.504 | 1.00x | 503.6 | 504.1 | 1 | 1 |
| 500 ms | 10 | 0.518 | 1.511 | 2.92x | 517.8 | 1510.3 | 10 | 4 |
| 500 ms | 25 | 0.544 | 3.532 | 6.49x | 538.0 | 3027.5 | 25 | 4 |
| 500 ms | 50 | 0.664 | 6.562 | 9.89x | 644.1 | 6043.0 | 50 | 4 |
| 500 ms | 100 | 2.430 | 12.718 | 5.23x | 2211.8 | 12125.9 | 42 | 4 |
| 2000 ms | 1 | 2.004 | 2.004 | 1.00x | 2004.1 | 2004.1 | 1 | 1 |
| 2000 ms | 10 | 2.011 | 6.012 | 2.99x | 2010.1 | 6010.9 | 10 | 4 |
| 2000 ms | 25 | 2.043 | 14.032 | 6.87x | 2038.4 | 12032.0 | 25 | 4 |
| 2000 ms | 50 | 2.161 | 26.061 | 12.06x | 2139.6 | 24047.3 | 50 | 4 |
| 2000 ms | 100 | 3.151 | 50.205 | 15.93x | 3049.2 | 48132.0 | 100 | 4 |

100 / 500 ms 的 100 并发格出现更明显的连接建立尾延迟，不能简单用固定 Provider delay 解释；这促成了后面的 Python-vs-C++ Provider 对照和连接复用建议。2000 ms 格最适合观察 Session/LLM 等待是否占用业务 worker。

原始结果目录：

- `data/persona_gateway_async_benchmark/fake_matrix_20260813/`
- `data/persona_gateway_async_benchmark/sync_matrix_20260813/`

## 4. 企业 500 并发配额 Fake

### 4.1 Python asyncio 高配额 Mock

`openai_compatible_quota_server.py` 使用 `aiohttp` 事件循环和 `asyncio.sleep()`，Provider semaphore 为 500，固定延迟 2000 ms；不是 ThreadingHTTPServer，也不是每请求 `sleep_for` 的同步模型。

| Provider quota | Chat 数 | 成功 | Provider peak in-flight | Provider peak queued | Burst | Chat P50 | Chat P95 / P99 | pipeline `llmTotal` P95 | 峰值 RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 250 | 250 | 250 | 1 | 4.021 s | 2596 ms | 3792 / 3920 ms | 2013 ms | 31.5 MiB |
| 500 | 500 | 500 | 500 | 1 | 10.960 s | 5166 ms | 9978 / 10178 ms | 2014 ms | 40.9 MiB |

500 格 0 rejected、0 Chat error；Gateway 线程峰值 31。`llmTotal` P95 约 2.014 秒且 `ioQueueWait` P95 为 0，说明请求已经同时进入 Provider，LLM pool 没有按 4/8 个 worker 分批阻塞。端到端 P95 约 10 秒，主要发生在客户端请求计时到 Gateway pipeline 计时开始之间，不能把这段直接归因给 Provider 或 Session。

原始结果目录：`data/persona_gateway_async_benchmark/enterprise_quota_500_20260813/`。

### 4.2 调高 Gateway threads 的对照

将夹具调为入站 HTTP 32 threads、LLM pool 8 workers、async outbound IO 8 threads后重跑 500 格：499 个 Chat 成功，1 个 setup `ReadError`，Provider peak in-flight 499；Chat burst 10.769 s，E2E P95 10005 ms，pipeline / `llmTotal` P95 2013 ms，峰值 RSS 47.4 MiB，线程峰值 66。

这个结果没有改善 8 秒级额外尾延迟，反而增加资源并出现一次负载端 setup 读错误。因此当前结论不是“继续盲目加线程”，而是需要做连接复用和负载发生器分段测量。原始结果目录：`data/persona_gateway_async_benchmark/enterprise_quota_500_tuned_20260813/`。

## 5. 真实 API 100 并发

### 5.1 当前异步结果

使用现有 ignored `tools/llm_smoke_test.key`，endpoint 为 `https://api.deepseek.com/v1`，model 为 `deepseek-chat`；报告和原始结果不保存 API key，Chat records 中的回复预览已替换为 `[redacted]`。

| 指标 | 结果 |
|---|---:|
| 并发用户 / 独立 Session | 100 |
| Chat 成功 / 失败 | 100 / 0 |
| Chat burst | 2.843 s |
| Chat P50 / P95 / P99 | 2230 / 2581 / 2697 ms |
| pipeline `llmTotal` P50 / P95 / P99 | 1440 / 1657 / 1689 ms |
| `ioQueueWait` P95 | 0 ms |
| 峰值 RSS | 39.7 MiB |
| 线程数采样 | 28–31 |

这是一次真实 Provider smoke-capacity 试验，不等价于企业 500 并发配额。个人 API 账户的 Provider 限流、模型排队和网络状态仍会影响结果；本轮没有观察到 429 或 transport error。

原始结果目录：`data/persona_gateway_async_benchmark/real_api_100_20260813/`。

### 5.2 历史同步真实 API 对照

历史报告 `gateway_benchmark_1780563536.json` 生成于 2026-06-04，使用同步
`OpenAiLlmClient::Complete()`。配置和结果记录共同确认 endpoint 为
`https://api.deepseek.com/v1`、model 为 `deepseek-chat`，`local_llm=false`、
`allow_placeholder=false`；100 个不同 Session 各执行一次 Chat，100/100 成功。当前结果使用相同
Provider 和 model，也是 100 个不同 Session、每 Session 一次 Chat、100/100 成功。

| 指标 | 历史同步真实 API | 当前异步真实 API | 变化 |
|---|---:|---:|---:|
| Chat 成功 / 失败 | 100 / 0 | 100 / 0 | 一致 |
| E2E P50 | 3196 ms | 2230 ms | -30.2% |
| E2E P95 | 5171 ms | 2581 ms | -50.1% |
| E2E P99 | 5762 ms | 2697 ms | -53.2% |
| E2E max | 6037 ms | 2698 ms | -55.3% |
| `backendTotal` P50 | 2359 ms | 1440 ms | -39.0% |
| `backendTotal` P95 | 4630 ms | 1658 ms | -64.2% |
| `backendTotal` P99 | 5063 ms | 1689 ms | -66.6% |
| Provider `llmTotal` P50 | 1108 ms | 1440 ms | +30.0% |
| Provider `llmTotal` P95 | 1296 ms | 1657 ms | +27.9% |
| Provider `llmTotal` P99 | 1384 ms | 1689 ms | +22.0% |
| `ioQueueWait` P50 / P95 / P99 | 未单独埋点 | 0 / 0 / 0 ms | 当前无可见排队 |
| `backendTotal - llmTotal` P50 | 1292 ms | 0 ms | 本地放大基本消除 |
| `backendTotal - llmTotal` P95 | 3655 ms | 1 ms | 本地放大基本消除 |
| `backendTotal - llmTotal` P99 | 3934 ms | 1 ms | 本地放大基本消除 |

这个对照最重要的事实是：当前测试中的真实 Provider 本身并没有更快，`llmTotal` P95 反而比历史
结果高 361 ms；但 E2E P95 仍降低 2591 ms。旧同步路径的 `llmTotal` 从同步调用真正开始时计时，
不包含任务进入 IO pool 后等待 worker 的时间；因此旧报告里逐请求的
`backendTotal - llmTotal` 包含 compute/IO 排队和短本地处理，可作为本地同步放大的保守代理指标，
不能伪装成精确的 `ioQueueWait`。当前实现已经直接拆分 queue wait，100 并发下
`computeQueueWait` 和 `ioQueueWait` 的 P50/P95/P99 均为 0。

两轮属于跨版本历史观察，不是严格受控 A/B：Provider 当次状态、prompt、L0/Emotion 开关、认证后端
和负载起跑方式均有差异。历史脚本未设置 Chat barrier，也没有独立 Chat burst 窗口，其完整
setup + Chat + close 运行时间为 9.847 秒；当前完整运行时间为 5.669 秒，另有严格 barrier 后的纯
Chat burst 2.843 秒。完整运行时间可作为背景证据，但不与当前 burst 直接计算加速比。Fake 固定延迟
A/B 仍是证明因果关系的主要实验，真实 API 历史对照用于证明改造在真实 Provider 上同样产生了显著收益。

### 5.3 历史 1000 并发的 IO pool 直接排队证据

2026-06-04 的旧 Gateway 压测在任务提交和开始执行之间已经分别记录
`computeQueueWait` 与 `ioQueueWait`。日志进一步确认下表三组运行时均为 8 个 compute worker、
2 个 IO worker；每一行的瞬时起跑和 60 秒 ramp 结果之间没有重启 Gateway，因此同一行共享运行时
配置。三行之间发生过 Gateway 重启，代表不同调参或实现阶段，适合观察旧同步 IO 路径的排队事实，
不构成三行之间的严格 A/B。

| 旧运行时组 | 起跑方式 | Chat 成功 | E2E P50 / P95 / P99 | `backendTotal` P95 | `ioQueueWait` P50 / P95 / P99 | `llmTotal` P95 |
|---|---|---:|---:|---:|---:|---:|
| A | 瞬时 1000 并发 | 1000 / 1000 | 19130 / 47666 / 59679 ms | 19503 ms | 8152 / 18063 / 20792 ms | 1551 ms |
| A | 60 秒 ramp | 1000 / 1000 | 9433 / 17740 / 18706 ms | 17723 ms | 8306 / 16426 / 17315 ms | 1494 ms |
| B | 瞬时 1000 并发 | 1000 / 1000 | 10753 / 35843 / 49600 ms | 8173 ms | 1884 / 6667 / 7763 ms | 1676 ms |
| B | 60 秒 ramp | 1000 / 1000 | 3667 / 4336 / 4502 ms | 4291 ms | 2229 / 2743 / 2938 ms | 1775 ms |
| C | 瞬时 1000 并发 | 999 / 1000 | 16121 / 45422 / 59486 ms | 16186 ms | 2539 / 13960 / 15024 ms | 1999 ms |
| C | 60 秒 ramp | 1000 / 1000 | 1504 / 2188 / 23464 ms | 2168 ms | 0 / 655 / 788 ms | 1817 ms |

组 C 的瞬时轮有 1 个 setup `ConnectError`，其余进入 Chat 的 999 个请求完成；P99 还受到单个
23.574 秒 Provider/transport 长尾影响。即使排除这一个异常，A/B/C 的 `ioQueueWait` P95 仍直接
证明旧同步 LLM 调用长期占用两个 IO worker，并把约 1.5 至 2 秒的 Provider 延迟放大成秒级至十秒级
本地排队。60 秒 ramp 能降低瞬时突发，但 A、B 两组仍保留 16.426 秒和 2.743 秒的 P95，说明平滑
流量只能缓解队列，不能修复“网络等待占住 worker”的结构问题。

当前指标需要按任务路由解释，而不是只看字段名称：

- 当前纯异步 Cloud 路径发起 HTTP 后立即归还业务 worker，所以 Fake 100x2 秒、真实 API 100 并发和
  企业 Fake 500 并发的 `computeQueueWait`/`ioQueueWait` P95 均为 0；
- 当前同步兼容 A/B 把整个同步 Turn 放在 compute task 中，因此 100x2 秒同步组的直接埋点是
  `computeQueueWait` P95 45.279 秒、`ioQueueWait` P95 0，而不是旧版本的 `ioQueueWait`；
- 两种旧路径的队列字段不同，但共同根因相同：有限 worker 在同步等待 LLM HTTP，后续 Session 只能
  分批执行。异步组 Provider peak in-flight=100 与同步兼容组 peak in-flight=4 是相同结论的外部证据。

历史原始报告：

- `data/persona_gateway_e2e/reports/gateway_benchmark_1780566147.json` 与 `1780566238.json`；
- `data/persona_gateway_e2e/reports/gateway_benchmark_1780566801.json` 与 `1780566878.json`；
- `data/persona_gateway_e2e/reports/gateway_benchmark_1780567065.json` 与 `1780567158.json`。

## 6. Python Mock 与 C++ AgentLoom HTTP Server 对照

### 6.1 为什么要做对照

早期 100/500 ms A/B 使用的 `openai_compatible_delay_server.py` 是 `ThreadingHTTPServer`，每连接一个 Python 线程，handler 内阻塞等待；它不适合作为 500 并发的最终负载端。企业配额测试后来使用了 `aiohttp` + `asyncio.sleep()`，已经是异步 I/O，但仍需要与库内 C++ Server 对照。

### 6.2 C++ Mock 实现

`openai_compatible_quota_mock_server.cpp` 复用 AgentLoom `net::HttpServer`：

- HTTP accept/read/write、keep-alive 和连接 RAII 由库实现；
- 固定 Provider latency 由独立 `asio::steady_timer` 驱动，不阻塞 HttpServer IO thread；
- quota admission、peak in-flight、completed 采用原子计数；
- `metrics` 返回实现标识、in-flight、peak、queued、completed；
- C++ Mock 自身不为每请求创建线程。

### 6.3 500×2000 ms 结果

两组使用相同调优 Gateway（入站 HTTP 32、LLM pool 8、outbound async IO 8）和相同 500 quota：

| Mock | Chat 成功 | Provider peak | Burst | E2E P50 | E2E P95 / P99 | pipeline P95 | errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python aiohttp | 499 | 499 | 10.769 s | 5124 ms | 10005 / 10123 ms | 2013 ms | 1 setup ReadError |
| C++ AgentLoom HttpServer | 499 | 499 | 10.228 s | 4925 ms | 9725 / 9905 ms | 2015 ms | 1 setup ReadError |

单轮结果中，C++ Mock 的 burst 缩短 541 ms（5.0%），E2E P50/P95/P99 分别降低
199/280/218 ms，最大值降低 506 ms。对于本轮固定 2 秒 Provider 延迟场景，相对比例被硬性延迟
稀释，但数百毫秒的绝对差距并非没有工程意义，也说明 AgentLoom C++ 异步 Server 的调度开销具有
竞争力。由于当前每种实现仅有一轮 500 并发结果，不能把差值直接宣称为稳定性能优势；需要多轮重复、
预热和客户端分段计时后再给置信区间。

两组都复现同类负载端 setup `ReadError`。因此 Python asyncio 不是本轮 8 秒级额外尾延迟的主解释，
但也不能据此认为两者完全等价：企业配额 Fake 已使用 Python 最成熟的 asyncio/aiohttp 路径，C++
AgentLoom Server 在相同条件下仍测得可见的绝对延迟优势。

更严谨的定位是：额外延迟位于压测客户端开始计时和 Gateway pipeline 计时开始之间。共同候选包括
客户端连接池/事件循环、Gateway 入站 accept/read 调度、Windows socket 资源和首次连接建立；当前数据
不足以单独归因某一个环节。6.4 已通过冷/热两轮试验证明出站连接复用本身有效，但它没有消除这段
额外 E2E 尾延迟，因此首次 Provider 连接建立不是当前唯一主因。

### 6.4 HTTP/1.1 outbound keep-alive 三轮对照

`AsyncBeastHttpClient` 已加入按 `scheme + lowercase(host) + port` 隔离的 HTTP/1.1 空闲连接池。
每条连接同一时刻只服务一个请求；只有响应完整读取、双方均允许 keep-alive 且 Beast buffer 无残留时
才回池，timeout、cancel、transport error 和协议不可复用响应都会关闭连接。全局/单 origin 空闲上限、
idle timeout 和 outbound IO thread 数均可配置；Shutdown 会先关闭空闲连接、取消 active operation、
等待 callback 收口，再停止 IO runtime。该实现不透明重放 POST，请求失败仍由上层显式 retry policy 决定。

同时修复了即时取消竞态：`Cancel()` 可能先于异步 `Start()` handler 被 IO 线程消费，Start handler 现在
在 DNS/connect 前检查取消标记并以 `Cancelled` 恰好完成一次。即时取消专项重复 100 次通过；异步 HTTP
专项 10/10、HTTP client 全量 35/35、LLM async 3/3、Session/生命周期 12/12、配置专项 10/10 均通过。

压测使用 C++ AgentLoom Provider Mock，固定延迟 500 ms、quota 500；Gateway 入站 16 threads、
outbound async IO 8 threads；100 个不同 Session 各执行两轮 Chat，每一轮独立 barrier 同时起跑。
Session metadata 使用内存 store 和默认 Persona overlay，不调用 `/api/persona`，避免 SQLite 写争用污染
transport 结果。Provider `/metrics` 的控制连接已从 accepted connection 统计中扣除。

| keep-alive | 三轮 Provider 出站连接 | 连接数中位数 | 冷轮 burst 中位数 | 热轮 burst 中位数 | 冷轮 E2E P95 中位数 | 热轮 E2E P95 中位数 | `backendTotal` P95 | `llmTotal` P95 | compute / IO queue P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 关闭 | 200 / 200 / 200 | 200 | 1.555 s | 2.140 s | 1463 ms | 1956 ms | 514–515 ms | 514–515 ms | 0 / 0 ms |
| 开启 | 65 / 67 / 55 | 65 | 1.561 s | 2.284 s | 1470 ms | 2015 ms | 514–515 ms | 514–515 ms | 0 / 0 ms |

六轮共 1200 个 Chat，全部成功。关闭时 200 个 Chat 严格对应 200 条新连接；开启后中位数仅 65 条，
连接建立减少 67.5%。开启组连接数可以低于首轮 100 并发，是因为首轮较早完成的连接会被同一 burst
中稍后进入 outbound pipeline 的请求再次借用，并非漏记请求。8 Session × 2 turn smoke 还观察到 16 个
Chat 仅建立 8 条 Gateway outbound 连接，第二轮直接复用同一批连接。

这组数据没有显示 keep-alive 带来 E2E 延迟改善：冷轮接近持平，热轮中位数还受客户端/Gateway 入站
调度波动影响而略高；但 Provider pipeline 始终约 515 ms，两个内部 queue wait 始终为 0。因此结论是：

- HTTP/1.1 keep-alive 已显著降低连接 churn、握手频率和 Provider accept/close 压力；
- 当前 100 并发尾延迟主要仍位于负载端开始计时到 Gateway pipeline 开始之间，不在 LLM、compute pool
  或 IO pool 内；
- 后续 transport 定位应使用 C++ load generator 拆分 connect、request-write、first-byte 和
  response-complete，并将 HTTP/2 multiplexing 作为独立能力验证，而不是继续通过增加 worker 猜测瓶颈。

原始结果目录：`data/persona_gateway_async_benchmark/keepalive_e2e_20260813/`。

## 7. 本轮验收与边界

已通过：

- 纯 Cloud Gateway 注入 async LLM；
- 同 Session 保序、不同 Session 并发专项测试；
- Cancel / Shutdown / callback exactly once；
- Fake 30 组 A/B，1116 Chat，0 error；
- 真实 API 100 Chat，100% 成功；
- Python asyncio 与 C++ AgentLoom HTTP Server quota mock 对照；
- 企业 quota Fake 500 peak in-flight，500 Chat 全成功（基线组）；
- C++ Mock 编译为独立 manual benchmark target；
- HTTP/1.1 outbound keep-alive 空闲连接池、配置接入、取消竞态回归和 100x2 三轮开关 E2E；

仍未完成：

- async cloud + local fallback 的统一异步组合；
- async callback 后的 emotion/memory/cache 短后处理从 outbound runtime thread 转移到 continuation pool；
- HTTP/2 multiplexing 与 C++ load generator transport 分段计时；
- 动态 quota policy/provider 和 ingress 前置 admission；
- SQLite metadata/auth 写入的全局串行化、busy timeout 与独立资源治理。

SQLite 单写者约束是其作为本地开发、测试夹具或轻量部署后端时的客观限制；上述治理不代表生产
元数据后端建议。实际生产项目使用 PostgreSQL，生产侧应按 PostgreSQL 的连接池、短事务、索引、
隔离级别和可恢复错误重试进行容量设计，而不是沿用 SQLite 的全局写串行化方案。

## 8. 原始证据与复现

```powershell
# Fake / C++ / 真实工具均使用 VS2026/v145 Release 产物
cmake --build build/x64-Release-Tests-v145-refactor --config Release --parallel -- /nodeReuse:false

# Python asyncio quota mock
python tools/openai_compatible_quota_server.py --delay-ms 2000 --max-inflight 500

# C++ AgentLoom HTTP Server quota mock
build/x64-Release-Tests-v145-refactor/Release/openai_compatible_quota_mock_server.exe 18081 2000 500 8

# 100/500 Chat benchmark
python tools/gateway_concurrency_benchmark.py \
  --concurrency 500 --turns 1 --timeout 60 \
  --scenario cpp_quota500_d2000_c500
```

真实 API 命令使用 `tools/persona_gateway_async_real_benchmark.example.json`，凭据从 ignored key 文件或环境变量读取；不要把 key 放入报告、配置样例或 Git 跟踪文件。
