# Gateway 会话亲和并发调度设计

> 状态：当前实现。core 默认 FIFO scheduler、可选 Gateway session-affinity scheduler 和
> SessionManager metadata、JSON 配置、tenant/user 配额和管理快照已落地；IO 长锁拆分仍待完成。  
> 日期：2026-08-11  
> 范围：Gateway Compute/IO Pool 的并发键、亲和调度、公平准入、生命周期和验收约束。

## 1. 决策摘要

Gateway 会话状态采用串行执行模型。同一个 session 在同一个线程池内任意时刻最多由一个
worker 执行，其余 worker 应继续处理其他 session，不允许多个 worker 先取走同一 session
的任务后阻塞在 session mutex 上。

项目不把 session 内部业务流程强行拆成并行 task。当前三池架构已经把网络协议、CPU 计算和
慢 IO 分离；需要并行的外部调用应在 Compute/IO 边界显式拆分，而不是并发修改同一份
`SessionState`。

`core::ThreadPool` 将接受一个可选的并发调度组件，但内部始终持有一个 scheduler：

- 未配置组件时自动构造 core 内置的 `DefaultFifoThreadPoolTaskScheduler`，保持现有单一全局
  FIFO、有界队列和 `TryPush` 背压语义；
- 配置组件时，由组件负责并发键串行化、任务准入、ready lane 选择和完成通知；
- core 层只认识通用 `concurrency_key`、`fairness_key` 和 `execution_id`，不认识 session、user
  或 tenant；
- 参考 Gateway 提供一个可选的 session-affinity 实现，将 `session_id` 作为
  `concurrency_key`，将认证后的 user/tenant 标识作为 `fairness_key`；
- HTTP/WS Server 默认行为是否启用该实现由 Gateway 配置决定，不改变其他 ThreadPool
  使用者的行为。

## 2. 当前问题

当前 `SessionManager::Submit` 先把每个任务提交到共享 ThreadPool，worker 取到任务后才查找
`SessionSlot` 并获取 session mutex。一次会话洪泛可能形成：

```text
session-a task 1 -> worker 0 -> 持有 session mutex 并执行
session-a task 2 -> worker 1 -> 等待同一 mutex
session-a task 3 -> worker 2 -> 等待同一 mutex
session-a task 4 -> worker 3 -> 等待同一 mutex
session-b task 1 -> 仍在全局队列中等待
```

全局队列容量只能限制总任务数，不能保证 session 公平性。多个 worker 等待同一个 session
mutex 也会削弱 Compute/IO 分池带来的隔离效果。

项目已有 `KeyedSerialExecutor`，但它位于 ThreadPool 之上，默认每 key 队列容量较大，且当前
`SessionManager` 没有使用它。新设计应复用其“同 key 串行”的原则，但把队列容量、调度统计、
shutdown 和 Gateway 准入契约统一到可替换的 ThreadPool 调度接口中。

## 3. 目标与非目标

### 3.1 目标

- 同一 `concurrency_key` 在一个 pool 内最多有一个 running task；
- 不占用 worker 等待同一 key 的业务 mutex；
- 不同 key 在 worker 数允许时并行执行；
- 在任务进入底层 runnable queue 前限制 per-key 和 per-fairness-key outstanding；
- 保留全局 queue capacity 作为最终硬上限；
- submit 失败、任务异常、取消、discard shutdown 和正常完成均不泄漏计数；
- 提供不暴露高基数业务 key 的聚合排队、运行、拒绝和 lane 深度快照；
- 默认 FIFO 调度路径保持源码和行为兼容，并与可选实现经过同一
  `Start/Enqueue/Dequeue/Complete/Close` 生命周期；
- 调度失败统一返回 `core::Status`，Gateway 记录日志并映射为协议错误。

### 3.2 非目标

- 不在 core 层实现用户、租户或认证规则；
- 不保证一个 session 在整个生命周期永久绑定同一 OS 线程；
- 不依赖 thread-local session 状态；
- 不替代 Net Pool 的连接数、请求速率、消息字节数和 outbound backpressure；
- 不通过等待 permit 阻塞提交线程或 worker；
- 不把所有任务强制声明为 session task，无并发键的维护任务仍可使用普通调度。

## 4. 并发键语义

`concurrency_key` 表示需要串行访问的状态所有者。Gateway 对话主链路必须使用稳定的
`session_id`，不能使用每次请求都变化的 trace ID 或 execution ID。

`execution_id` 用于一次 Skill、媒体或长流程执行的追踪、取消和统计。它可以作为更细粒度
调度组件的辅助字段，但不能替代 session 隔离键，除非该 execution 的状态与其他 execution
完全独立且不访问同一 SessionState。

`fairness_key` 表示配额聚合主体。参考 Gateway 使用认证后的 `tenant_id/user_uuid`，防止攻击者
创建大量 session 绕过 per-session 限制。外部请求不得直接提供未经验证的 fairness key。

```text
concurrency_key = session_id
fairness_key    = tenant_id + ":" + user_uuid
tenant_key      = tenant_id
execution_id   = 当前业务执行 ID，可为空
```

## 5. 亲和范围

“同一 session 同一 worker”在本设计中解释为：一个 lane 处于 active 状态期间只能有一个
worker owner。lane 排空后可以解除绑定，后续任务允许根据实时负载重新选择 worker。

不采用 session 生命周期永久绑定，原因如下：

- session 的任务量和耗时不均匀，固定哈希容易造成热点 worker；
- 某些 session 可能长时间空闲，永久映射会保留无效表项；
- 当前 SessionState 不依赖 thread-local 数据，不需要线程身份稳定性；
- 排空后迁移不破坏顺序，因为迁移发生时不存在 running 或 queued task。

如未来某个独立模块确实依赖线程本地模型上下文，可以提供严格 sticky 的另一种调度实现，
但不能把该约束变成 ThreadPool 默认语义。

## 6. 三池边界

### 6.1 Net Pool

Net Pool 继续负责连接、协议解析和轻量 handler。它不使用 session worker-share 策略，安全边界
由连接池上限、每 IP/用户连接数、请求/帧速率、消息大小、解析预算和 outbound backpressure
构成。Net handler 不得等待 session lane。

### 6.2 Compute Pool

Compute Pool 对同一 session 的 `max_running_per_key` 固定为 1。对话状态、情绪状态、Prompt
组装和需要 SessionState 一致性的计算都通过同一 lane 串行执行。

Compute task 需要慢 IO 时，应提交到 IO Pool 并结束当前计算阶段，不得持有 SessionState mutex
等待 IO 结果。

### 6.3 IO Pool

IO Pool 使用独立调度组件和独立 lane 表。默认仍对同一 session 串行，先保证状态安全和公平性。
只有明确不持有 SessionState、具有独立结果合并协议的 IO task，才允许使用更细的
`concurrency_key`，例如 `session_id + execution_id + operation`。

IO task 应采用“短锁快照 -> 无锁慢 IO -> 短锁提交结果”的结构。当前持有 SessionSlot mutex
执行整个 IO task 的路径需要在接入调度器时逐项审计。

## 7. 建议的 core 接口

以下代码用于冻结职责，不代表最终命名已经提交：

```cpp
namespace core {

struct ThreadPoolTaskMetadata {
    std::string concurrency_key;
    std::string fairness_key;
    std::string tenant_key;
    std::string execution_id;
    std::string task_name;
    std::size_t cost = 1;
};

struct ThreadPoolSchedulerOptions {
    std::size_t worker_count = 0;
    std::size_t queue_capacity = 0;
    std::string pool_name;
};

// 不透明的任务信封。scheduler 只能读取 metadata 并共享持有任务，
// 只有 ThreadPool 可以调用内部 TaskFunction 和完成 TaskGroup token。
class ThreadPoolWorkItem {
public:
    ThreadPoolWorkItem(ThreadPoolWorkItem&&) = delete;
    ThreadPoolWorkItem& operator=(ThreadPoolWorkItem&&) = delete;
    ~ThreadPoolWorkItem();

    const ThreadPoolTaskMetadata& metadata() const noexcept;

    ThreadPoolWorkItem(const ThreadPoolWorkItem&) = delete;
    ThreadPoolWorkItem& operator=(const ThreadPoolWorkItem&) = delete;

private:
    friend class ThreadPool;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class IThreadPoolTaskScheduler {
public:
    virtual ~IThreadPoolTaskScheduler() = default;

    virtual Status Start(const ThreadPoolSchedulerOptions& options) = 0;
    virtual Status TryEnqueue(const std::shared_ptr<ThreadPoolWorkItem>& item) = 0;
    virtual Result<std::shared_ptr<ThreadPoolWorkItem>> WaitDequeue(
        std::size_t worker_index,
        std::stop_token stop_token) = 0;
    virtual void Complete(const ThreadPoolWorkItem& item,
                          std::size_t worker_index,
                          const Status& status) noexcept = 0;
    virtual void Close(bool discard) noexcept = 0;
    virtual std::size_t QueuedTaskCount() const noexcept = 0;
    virtual ThreadPoolConcurrencySnapshot Snapshot() const = 0;
};

} // namespace core
```

最终接口必须保证：

- 不使用 `void*` 传递调度状态；
- work item 禁止复制和移动，通过受控 shared ownership 在 scheduler 与 ThreadPool 间传递，
  TaskGroup token 和 payload 不发生裸指针转发；
- `TryEnqueue` 只做有界、非阻塞操作，失败时 ThreadPool 仍持有 item 并完成拒绝终态；
- `WaitDequeue` 是唯一允许等待 ready task 的接口，不能返回同 key 的第二个 running task；
- 生命周期回调为 `noexcept`，业务异常统一由 ThreadPool 转换为 `core::Status`；
- scheduler 生命周期由 `shared_ptr` 或明确的 Server 所有权管理，长于 ThreadPool worker；
- TaskGroup root/child task 与普通 task 使用同一准入和释放路径；
- 未配置 scheduler 时不额外分配 per-key 状态。

`ThreadPoolWorkItem` 隔离 scheduler 与 ThreadPool 内部 callable、trace、payload 和 TaskGroup
细节。默认 `DefaultFifoThreadPoolTaskScheduler` 与 Gateway session-affinity scheduler 使用
同一接口，避免在 `ThreadPool::WorkerLoop` 中维护两套完成语义。仅提供计数 callback、不能
控制 runnable queue 的组件不满足本设计。

## 8. Gateway 示例调度器

建议示例类型名为 `GatewaySessionAffinityScheduler`，实现放在 service/gateway 或
service/persona，而不是 core。

每个 lane 至少维护：

```text
concurrency_key
fairness_key
queued task count
running task count（只能为 0 或 1）
owner worker（lane active 时可选）
outstanding cost
last active time
rejected count
```

推荐调度流程：

```text
TrySubmit(metadata, task)
  -> 校验 key 长度和已认证 owner
  -> CAS 获取 global outstanding
  -> CAS 获取 fairness-key outstanding
  -> CAS 获取 concurrency-key outstanding
  -> 放入该 key 的有界 lane
  -> lane 原为空时加入 ready-lane queue
  -> 立即返回，不等待 worker

worker dequeue
  -> 从 ready-lane queue 取得一个 key
  -> 将 lane 标为 running，并记录 worker owner
  -> 只取该 lane 的一个 task
  -> 执行 task

task complete
  -> RAII lease 释放 running/outstanding
  -> lane 仍有任务时重新进入 ready-lane queue
  -> lane 排空时删除映射并解除 owner
```

ready-lane queue 而不是“所有任务的 FIFO queue”是避免多个 worker 同时取得同一 session 的
关键。一个 key 在 ready queue 中最多出现一次，running 时不能再次进入 ready 状态。

## 9. 配额与拒绝

当前 JSON 配置字段如下：

```json
{
  "persona_gateway": {
    "compute_pool": {
      "scheduler": "session_affinity",
      "max_outstanding_per_key": 8,
      "max_outstanding_per_fairness_key": 32,
      "max_outstanding_per_tenant": 256
    },
    "io_pool": {
      "scheduler": "session_affinity",
      "max_outstanding_per_key": 16,
      "max_outstanding_per_fairness_key": 64,
      "max_outstanding_per_tenant": 512
    }
  }
}
```

第一版 lane 已强制 `max_running_per_key = 1`。`max_active_keys` 限制活动 lane 表规模，
其余三个 `max_outstanding_*` 字段分别限制 session key、认证用户 fairness key 和 tenant 的
queued + running 总量。

任何一级配额失败都立即返回 `core::ErrorCode::ResourceExhausted`。HTTP 映射为 429；WebSocket
返回结构化错误。持续违反限制的连接可由 Gateway 安全策略关闭，但 scheduler 本身不处理协议。

日志必须包含 pool、module、operation、受控 session 标识、fairness key 的安全摘要、当前值、
阈值和 trace ID，不记录 token、Prompt 或用户正文。

## 10. 生命周期与异常安全

- `Start` 在 worker 启动前初始化 scheduler；失败时 ThreadPool 不进入 running 状态；
- 入队失败时 lease 自动释放全部已取得配额；
- worker 开始任务时只执行一次 `MarkRunning`；
- 正常返回、`AppException`、标准异常和未知异常都在转换为 Status 后执行 `Complete`；
- `Shutdown(true)` 完成所有 lane 后关闭 scheduler；
- `Shutdown(false)` 丢弃 queued task，并通过 lease 析构释放计数；
- running task 通过 stop token 协作取消，scheduler 不强杀线程；
- session close 后尚未运行的任务必须形成明确的 Cancelled/NotFound 终态并释放 lease；
- lane 表项只能在 queued=0 且 running=0 时删除，防止 ABA 和计数下溢；
- scheduler 回调异常不得逃逸到 worker loop。

## 11. 可观测性

ThreadPool 通用统计增加 scheduler 摘要，但不把业务 session 字段写入 core 类型：

```text
active_keys
ready_keys
queued_tasks
running_tasks
rejected_global
rejected_per_key
rejected_per_fairness_key
rejected_per_tenant
max_lane_depth
```

当前管理端通过 `/api/health` 和 `/api/system/stats` 暴露上述 pool 级聚合快照，不返回 session
ID、fairness key 或 tenant key。未来如需 worker owner 或逐 session 明细，应使用有界、受权限
保护的管理快照，而不是直接把高基数字段加入公开指标。

高基数 key 不直接作为 Prometheus label。逐 session 明细只进入有界管理快照或采样日志，聚合
指标使用 pool/reason 等低基数字段。

## 12. 实施顺序

1. 已完成：引入 scheduler 接口和 `DefaultFifoThreadPoolTaskScheduler`，未传入可选 scheduler 时
   ThreadPool 回归测试保持通过；
2. 已完成：普通 task、TaskGroup root/child task 统一进入 scheduler 路径；
3. 已完成：实现 ready-lane 驱动的 `GatewaySessionAffinityScheduler`；
4. 已完成：SessionManager dispatch 传入稳定 session key 和 user fairness key；
5. 已完成：Gateway Server 通过 JSON 配置或依赖注入选择默认 FIFO/示例 scheduler；
6. 审计 IO task 的 SessionSlot 长锁，拆成快照、IO、提交三个阶段；
7. 部分完成：管理端 stats 已包含 scheduler 快照；聚合日志和持续违规连接策略仍待补充；
8. 已完成聚焦单测和本地 A/B 压测；真实 Gateway 洪泛、竞态和长稳压测仍待完成。

## 13. 验收矩阵

### 13.1 单元测试

- 同 key 提交多个阻塞任务，任意时刻只有一个 running；
- 不同 key 在多个 worker 上并行；
- per-key、per-fairness-key 和 global limit 分别拒绝；
- queue full、空 task、scheduler 未启动均返回正确 Status；
- task 成功、失败、抛异常、TaskGroup 完成后计数归零；
- `Shutdown(false)` 丢弃任务后 lane 和 permit 无泄漏；
- lane 排空后映射删除，后续任务可以重新分配 worker；
- worker 数为 1、2、4 时无除零、零阈值和永久饥饿。

### 13.2 E2E 与安全测试

- session-a 洪泛时，session-b 在限定延迟内完成；
- 同一用户创建多个 session 时受到 fairness-key 总配额限制；
- 不同用户在全局容量内保持前进；
- HTTP 超额返回 429，WS 超额返回 RESOURCE_EXHAUSTED；
- session close、断连和 Server shutdown 与洪泛并发时无死锁或计数漂移；
- Compute 满载不阻塞 Net Pool；IO 慢请求不占用 Compute worker 等待。

### 13.3 压力测试

- 比较 FIFO 基线与 session-affinity 的吞吐、p50/p95/p99、公平性和拒绝率；
- 混合短任务、长任务和热点 session，验证无单 worker 永久热点；
- 连续运行期间检查 `outstanding = queued + running`；
- 使用 ThreadSanitizer/Linux 和 Windows Release 测试并发状态机；
- scheduler 开启后的开销应按每次 submit/dequeue 记录，并设置可接受阈值。

## 14. 安全结论

会话亲和调度解决的是“同一状态所有者占用多个 worker”和“共享队列缺少公平性”问题。它不是
完整的抗资源耗尽方案，仍需与认证主体配额、连接限制、请求速率、消息大小、内存池预算、任务
deadline、取消和下游熔断共同使用。

参考实现必须先在提交点限制 queued + running，再通过 ready lane 保证同 key 单 worker。
仅依赖 worker 汇报表、仅统计 running，或等任务进入 worker 后再拒绝，都不能满足本设计。
