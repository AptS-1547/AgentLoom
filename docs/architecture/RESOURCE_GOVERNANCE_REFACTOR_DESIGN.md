# AgentLoom 资源治理重构设计

状态：当前设计，尚未作为运行时默认策略发布。

本文定义 AgentLoom 后续“压力前馈 + 有界 worker 弹性 + 资源归档”重构的共同边界。它与 WebSocket 内存池、账户级并发配额、Session/Projection TTL、Crashdump 和运行时压测报告配套使用。

## 1. 背景与已确认事实

近期跨端压力验证已经证明：

- HTTP/WS 连接配额、Session 配额和 Turn 配额可以稳定返回 `RESOURCE_EXHAUSTED`；
- 连接关闭后 `sessionCount` 和 `projectionCount` 能回落，修复后的 Session 生命周期没有出现按轮次线性累积；
- AgentLoom WebSocket 接收路径使用固定 `MutableBufferSequence`，不是 Beast `DynamicBuffer`；
- 当前 WebSocket fragment 由 AgentLoom 的 `SharedBuffer` 和 Session 私有 `BucketMemoryPool` 管理；
- 读取粒度已经支持按普通路由/流式路由分级，并根据 `OnReadSome()` 的真实字节数在有界档位内增长；
- 线程池调度器已经提供 `queued_tasks`、`running_tasks`、`ready_keys`、`active_keys`、拒绝计数和 `max_lane_depth` 等快照；
- 异步 LLM Turn 使用 deferred completion，外部等待不会继续占用业务 worker。

因此后续治理的目标不是“看到连接多就创建更多线程”，也不是用动态扩容掩盖生命周期泄漏，而是根据可运行压力和资源余量做有界、可回落的调整。

## 2. 目标与非目标

### 2.1 目标

1. 在连接风暴、Session/Turn 突发和大流量 WebSocket 场景下，尽量利用空闲 CPU 与可用 worker。
2. 保持账户、租户、Session、全局队列和进程资源的硬上限；任何治理动作都不能绕过 admission 或并发配额。
3. 压力结束后允许线程、连接、Projection、内存池 slab 和临时文件逐步回落。
4. 每次扩缩容、拒绝、降级和资源回收都能与 `request_id`、`trace_id`、`tenant_id`、`session_id` 或 execution id 关联。
5. 将压测输入、版本、配置、资源曲线、Crashdump 和结论归档为可复核证据。

### 2.2 非目标

- 不按活跃连接数线性创建线程。
- 不把 WebSocket 空闲连接误判为 CPU 压力。
- 不让远端配置中心替代进程内硬限制。
- 不在第一阶段重写 Beast DynamicBuffer 或替换现有 Session fragment 所有权模型。
- 不把 worker 扩容当作句柄泄漏修复；句柄回落需要安全缩容和对象生命周期正确。

## 3. 资源平面

### 3.1 Net 平面

`HttpServer` 使用共享 `io_context` 和固定数量的 IO 线程，连接不是线程的一对一映射。Net 平面主要消耗：

- TCP/socket、定时器和 HTTP/WS Session 句柄；
- WebSocket fragment、内存池 slab 和 outbound queue；
- Asio handler 执行时间与 event-loop 延迟。

`active_connections` 只能作为容量和风险信号，不能单独触发 Net worker 扩容。后续若要在线调整 Net IO 线程，必须先补充 event-loop lag、pending handler 和 handler service time 指标。

### 3.2 业务线程池

当前 Gateway 主要包含 compute、io、llm 三个池：

- compute：CPU 计算、Session 状态处理和部分同步业务；
- io：存储、网络适配和可等待的外部操作；
- llm：按 Session affinity 排队的 Turn 执行，异步客户端等待期间通过 deferred completion 释放 worker。

每个池既有 worker 状态，也有 scheduler 并发配额。扩容判断优先使用“可运行任务压力”，而不是连接数或逻辑请求总数。

### 3.3 Session、Projection 与内存

Session、Runtime Session、Projection 和底层 Application 资源必须分别建模。TTL、有界容器、关闭回调和析构顺序是内存回落的前提；worker 扩容不能替代这些生命周期约束。

### 3.4 进程级资源

治理器必须同时观察：

- Working Set、Private Commit、Peak Working Set；
- 线程数、句柄数、TCP 连接数；
- CPU 时间和采样期间 CPU 利用率；
- Crashdump 数量及新增 dump 的时间窗口。

任务管理器显示值与 allocator 统计不等价。`BucketMemoryPool` 的 allocated/free/slab 统计应与进程级 Private Commit 分开记录。

## 4. 压力前馈模型

### 4.1 输入信号

每个治理采样周期（建议 1 秒）读取：

| 信号 | 来源 | 用途 |
| --- | --- | --- |
| `active_connections` | Net `ConnectionPool::Stats()` | 容量与连接风险，不直接扩线程 |
| `active_workers` / `worker_count` | `ThreadPool::Stats()` | 判断 worker 是否饱和 |
| `queued_tasks` | ThreadPool/Scheduler snapshot | 判断是否有排队压力 |
| `ready_keys` | Session affinity scheduler | 判断是否存在可运行 lane |
| `running_tasks` | Scheduler snapshot | 区分执行中与排队中任务 |
| `max_lane_depth` | Scheduler snapshot | 识别单 Session 或热点 key |
| 拒绝计数 | 配额/队列/scheduler | 识别 admission 已触顶 |
| CPU 利用率与核心数 | 进程/系统采样 | 共享 CPU 预算 |
| Working Set/Private Commit | 进程采样 | 防止扩容导致资源越界 |

### 4.2 可运行压力

对普通池，基础压力定义为：

```text
idle_workers = worker_count - active_workers
runnable_pressure = max(queued_tasks, ready_keys)
```

只有同时满足以下条件，才允许考虑扩容：

```text
runnable_pressure > 0
active_workers == worker_count
idle_workers == 0
未达到 CPU、内存、句柄和池 max_workers 上限
```

如果 `queued_tasks` 很高但 `ready_keys == 0`，通常说明任务被同一 Session lane 或并发键串行化；增加 worker 不会让同一 key 并行执行，应优先报告热点 lane 或返回稳定的资源耗尽。

### 4.3 压力等级

治理器将压力归一化为四级：

- `Normal`：有空闲 worker，队列稳定为空或短暂波动；
- `Warm`：队列出现但未持续饱和，保持配置不变；
- `Saturated`：连续多个采样周期 worker 全忙且存在可运行排队任务；
- `Critical`：队列/配额/内存/句柄任一硬上限接近或已经触发，优先 admission 拒绝和降级，不再扩容。

状态切换必须有迟滞，避免单个采样点造成扩缩容抖动。

## 5. Worker 弹性策略

### 5.1 配置模型

现有 `worker_count` 保留为固定模式下的数量或弹性模式下的初始数量。后续建议增加：

```json
{
  "worker_count": 4,
  "scaling": {
    "enabled": false,
    "min_workers": 2,
    "max_workers": 8,
    "sample_interval_ms": 1000,
    "scale_up_samples": 3,
    "scale_down_samples": 60,
    "scale_cooldown_ms": 5000
  }
}
```

语义约束：

- `min_workers <= initial_workers <= max_workers`；
- `max_workers` 必须经过进程级 CPU 预算裁剪；
- `scale_up_samples`、`scale_down_samples` 和冷却时间必须为正；
- 弹性关闭时不改变现有固定 worker 行为；
- 配置解析失败在启动阶段返回稳定错误，不静默使用无限制值。

### 5.2 第一阶段：有界扩容

第一阶段只实现安全扩容，不实现在线缩容：

1. 采样器连续 `scale_up_samples` 次观察到 `Saturated`；
2. 校验全局 CPU、Private Commit、句柄和目标池 `max_workers`；
3. 每次只增加一个 worker，或按受控 `growth_step` 增加；
4. 扩容后进入 `scale_cooldown`，期间不重复扩容；
5. 输出结构化扩容事件，包括 pool、旧值、新值、pressure、CPU 和内存快照。

现有 `ThreadPool` 只在 `Start()` 创建 worker，在线扩容需要新增 `GrowTo(target)`。新增线程应复用现有 scheduler，不改变 Session affinity、任务完成和 deferred completion 语义。

### 5.3 第二阶段：安全缩容

缩容必须晚于扩容，建议连续 60 个采样周期满足：

```text
queued_tasks == 0
ready_keys == 0
active_workers < worker_count / 2
距离最近一次扩容已超过冷却时间
```

每次只减少一个 worker，且不低于 `min_workers`。缩容不能中断正在执行或 deferred completion 任务。

当前 scheduler 的 `WaitDequeue()` 需要先正确响应单 worker 的 `stop_token`，否则无法在不关闭整个 scheduler 的情况下退休一个线程。缩容实现还必须保证：

- retired worker 不再接收新任务；
- 正在运行的任务自然完成；
- `jthread` 在生命周期锁外 join；
- worker index、状态快照和统计不发生数据竞争；
- 缩容过程中仍有 worker 消费 ready queue。

### 5.4 CPU 核心共享预算

worker 上限不能由每个池独立读取 `hardware_concurrency()` 决定。建议由进程级预算分配：

```text
cpu_threads = max(1, hardware_concurrency())
```

compute、io、llm 和 Net IO 共用总预算，具体权重由部署 profile 配置。默认策略应优先保证 compute 和事件循环，再允许 IO/LLM 有限超配；不得因为三个池都配置 `worker_count = 0` 而各自创建完整 CPU 核心数。

CPU 利用率高、队列持续积压时，应优先 admission/降级，而不是无限扩线程。CPU 利用率低但队列饱和时，才是扩容的有效窗口。

## 6. Net 连接与句柄治理

### 6.1 连接数的正确用法

连接数用于：

- 校验连接配额和拒绝策略；
- 估计 Session/fragment/slab 的潜在内存压力；
- 为 Net event-loop 指标提供背景分母；
- 触发更积极的采样频率。

连接数不用于：

- `worker_count = active_connections`；
- 每个 WebSocket 创建一个线程；
- 在空闲连接风暴下自动扩大业务池。

### 6.2 WebSocket 内存与 worker 的关系

当前 fragment 读取采用 AgentLoom 自有 `SharedBuffer` 和 `BucketMemoryPool`。路由 profile 和 `OnReadSome()` 动态读取档位可以降低空闲连接首次 slab 成本，同时保留大流量通道的吞吐。该策略不改变 `max_message_bytes` 硬上限，也不替代 Session 析构和 Projection TTL。

### 6.3 句柄回落验收

任何声称“资源已回落”的压测必须同时记录：

- `active_connections`、`closed_connections`；
- 线程数和句柄数；
- Application/Go 进程 Private Commit 与 Working Set；
- Session/Projection 存活数；
- 关闭后至少一个 TTL/冷却窗口的二次采样。

只看到业务计数归零，不能证明线程、句柄、slab 或工作集已经回落。

## 7. 压测与归档闭环

### 7.1 每轮压测必须归档

每个压力场景生成不可变的 run directory，至少包含：

```text
manifest.json              # 版本、commit、平台、配置摘要、场景和时间窗口
scenario.json              # 并发、连接、payload、慢链路和恶意请求参数
resource-samples.jsonl     # 1 秒或更高频率的进程/池/连接采样
gateway.stdout.log
gateway.stderr.log
application.stdout.log
application.stderr.log
round-*.json                # 每轮业务结果和错误分布
crashdumps/                 # 本轮新增 dump 及索引
summary.json                # 结论、阈值判断和 artifact 哈希
```

归档目录名称必须包含 UTC 时间或单调 run id；禁止覆盖既有结果。真实账号、token、密钥和完整敏感对话不得进入归档。

### 7.2 压力矩阵

至少覆盖：

1. 空闲 HTTP/WS 连接风暴；
2. 慢 header、慢 body、慢 WS frame 和损坏断连；
3. 大 Binary/Text message 与多 fragment 流；
4. Session/Turn 配额触顶和恶意 Session 构造；
5. 同一 Session 热点 lane 与多租户公平性；
6. Projection/Runtime Session TTL 回收；
7. worker 饱和、扩容冷却和压力结束后的回落；
8. Go/C++ 跨进程管道异常、Crashdump 和重启恢复。

### 7.3 验收指标

每个场景同时给出：

- 成功率、稳定拒绝率和错误码分布；
- P50/P95/P99 延迟与吞吐；
- peak/after-cooldown 的线程、句柄、Private Commit、Working Set；
- pool 的 worker、queued、running、rejected、ready lane；
- Session/Projection 最终数量；
- 新增 Crashdump 数量和分类；
- 扩缩容次数、触发原因和是否触发硬上限。

结论必须标注“峰值期间测量”或“轮次结束测量”。轮次结束采样不能替代连接保持期间的瞬时峰值采样。

## 8. 实施阶段与回滚

### 阶段 A：观测基线

- 统一池、Net、连接、进程和 Crashdump 采样格式；
- 将现有压力报告迁移到 manifest + summary 归档格式；
- 不改变 worker 默认值和 admission 语义。

### 阶段 B：启动时自适应

- 按 CPU 核心数、Net IO 线程数和各池初始配置计算安全默认值；
- 仅在启动时调整，失败则回退到显式配置；
- 增加配置解析和启动日志。

### 阶段 C：有界在线扩容

- 实现 `ThreadPool::GrowTo()`；
- 只依据连续饱和和可运行压力扩容；
- 加入冷却、CPU/内存/句柄预算；
- 保持默认 `enabled = false`，先用压测 profile 开启。

### 阶段 D：安全缩容与自动归档

- 修复 scheduler 对单 worker stop token 的响应；
- 实现带迟滞的 `Resize()`；
- 将资源回落和 Crashdump 结果自动写入归档 summary；
- 通过长时间连接风暴和恢复压测后，再考虑打开生产默认值。

任何阶段出现错误率、P99、Private Commit、句柄或 Crashdump 回归，都可以关闭 scaling profile，恢复显式固定 worker 配置；不得通过提高队列或消息上限掩盖回归。

## 9. 当前状态与下一步

当前已经具备：

- 线程池和 Session affinity scheduler 的压力快照；
- Net ConnectionPool 的连接/内存快照；
- WebSocket 路由分级和基于真实读取字节数的有界 fragment 调优；
- 连接、Session、Turn、Projection 和 Crashdump 压力验证材料。

尚未实现：

- 在线 `GrowTo()` / `Resize()`；
- 共享 CPU worker 预算协调器；
- Net event-loop lag 指标；
- 统一资源采样和不可变归档 manifest；
- scheduler 对单 worker stop token 的安全缩容支持。

下一步应先完成阶段 A 的观测和归档格式，再以固定 profile 做 A/B 压测，最后实现阶段 C 的有界扩容。没有峰值与冷却窗口的证据，不应宣称动态治理已经改善资源利用率或句柄回落。
