# AgentLoom Runtime 通用能力补充需求

> 状态：分项实施中；通用 unary gRPC Server Runtime、Gateway 通用化首轮和 Persona Session 长锁第一阶段已完成，后续 transport/动态 quota 仍在推进
> 日期：2026-08-12
> 进度更新：2026-08-13
> 来源：EnterpriseTrainingSystem Application Runtime Gateway 设计

## 1. 目的

本文记录适合在 AgentLoom 通用运行时中补充的能力。异步流式、CloudTaskCoordinator 和观测增强可以与
EnterpriseTrainingSystem Application 同步开发，不构成核心智能体交互闭环的统一前置条件；第 11.4
至 11.7 节的 Gateway 解耦则是正式 Application 组合根落地的默认包边界门禁。若工期必须先形成纵向
闭环，只能按 11.11 节的受控应急方案临时推进。

本文只定义通用 Runtime、并发、流式传输和云任务能力，不把 EnterpriseTrainingSystem 的产品业务下沉到 AgentLoom。

不属于 AgentLoom 的内容包括：

- 企业 OA/SSO、用户、角色和权限事实；
- 浏览器登录 Session、Cookie 和公网 API；
- 训练业务 Session、PostgreSQL 事务、revision、业务幂等和审计；
- Go Front-Gateway 的 SSE bridge、浏览器事件重放和业务路由；
- 企业租户预算、数据合规策略和 usage 账单事实。

## 2. 当前实现基线

AgentLoom 已具备可继续扩展的基础：

- `core::BlockingQueue`：有界生产者/消费者队列；
- `core::KeyedSerialExecutor`：同 key 保序、不同 key 并行；
- `core::TaskGroup`：任务树、seal、drain 和失败传播；
- `core::OrderedBitmapWindow`：乱序完成、连续序号消费；
- `core::ThreadPool`：`std::stop_token`、TaskGroup 和 scheduler；
- `net::BackpressureQueue`：条目数和字节数双重背压；
- `net::ConnectionPool`、`SharedBuffer` 和 WebSocket 串行写；
- `llm::CloudTaskCoordinator`：单批 chunk fan-out/fan-in 和有序结果；
- `OpenAiLlmClient`、`FallbackLlmClient` 和基础 retry；
- `RuntimeMaintenanceService` 和 Runtime Session 生命周期设施。

本次补充不应复制上述原语，而应在其上增加一致的取消、流式输出、调度、观测和 RAII 契约。

AgentLoom 已有的 BERT 情绪链路可直接投入使用：

```text
AgentLoom::emotion_server
  -> EmotionGrpcService
  -> EmotionInferenceService
  -> OnnxBERTModel

AgentLoom::runtime
  -> GrpcEmotionAnalyzer
  -> IEmotionAnalyzer
  -> PersonaRuntime
```

该链路采用同步 unary gRPC，但同步 Server 已配置多 CQ/poller，并具备认证、请求校验、trace、统计、异常边界、deadline 检查和 `core::Status` 映射。它已经满足当前功能复用要求，不需要等待本文件中的异步化工作。

## 3. 推进关系与优先级

### 3.1 当前进度快照

以下状态以 AgentLoom 当前工作区和已完成的本地验证为准。构建与安装包的详细变更、验证目录和后续
收尾项单独记录在 `AGENTLOOM_BUILD_AND_PACKAGE_PROGRESS_2026_08.md`，避免构建问题淹没 Runtime
架构主线。

| 能力 | 状态 | 当前结论 |
|---|---|---|
| 通用异步 gRPC unary Server Runtime | 已完成基础版 | `AgentLoom::grpc_runtime` 已导出；具备 typed handler、原子 admission、取消/deadline、`core::Status` 映射、恰好一次 Finish 和 snapshot；Emotion 仅作为示例派生接入 |
| 异步 gRPC Client / server-streaming | 未完成 | Client runtime 尚未实现；streaming 的单 writer、event/byte 背压和关闭状态机仍待实现 |
| SQLite / Config ABI / ConfigSection 选择性加载 | 已完成，待提交 | 安装前缀内 SQLite、统一 nlohmann/json ABI、whole-archive helper 和 `ConfigSectionSelection` 已通过独立 consumer 验证 |
| Reference Gateway 开关与 IPC/media 目标边界 | 部分完成 | `REFERENCE_GATEWAY=OFF` 时 `AgentLoom::gateway` 可构建和消费，普通 IPC 已从 media 开关独立；但 target 仍承载参考产品依赖，不等于通用 Gateway 抽取完成 |
| 11.4 至 11.7 Gateway 通用化 | 已完成首轮抽取，已验证 | 已导出 `gateway_foundation`、`persona_interaction`、`gateway_routing` 和通用聚合 `gateway`；`reference_gateway` 仅在参考 Gateway 启用时导出，并单向依赖通用层。生命周期、typed routing 和 Persona interaction 已有单元测试及独立 package consumer E2E 覆盖 |
| 11.8 Session 生命周期 observer | 未完成，P0 | 当前仍是可覆盖的单 callback，不能满足多个下游订阅和恰好一次关闭投影 |
| 11.9 动态 quota/admission | 未完成，P0 | scheduler 有静态 process/key/tenant 限额，但缺少可信动态 policy/provider 和 ingress 前置 admission |
| 11.10 Session 长锁与同步下游 IO | 第一阶段与 HTTP/1.1 连接复用已完成，后续优化进行中，P0 | Cloud LLM async HTTP、deferred completion、Session 短锁 snapshot/commit、保序/跨 Session 并发、主动关闭取消和 outbound keep-alive pool 已接入纯 cloud Gateway，并通过单元、生命周期、Fake/真实 API E2E 压测；async fallback、短后处理线程迁移、HTTP/2 和动态 admission 仍待完成 |
| CloudTaskCoordinator / LLM streaming / 统一 snapshot | 未开始本轮实现 | 保持原规划，优先级低于 11.4 至 11.10 的 Application 组合根门禁 |

### 3.2 下一阶段顺序

`gateway` 已成为通用兼容聚合，不能再以“存在一个 target”替代其边界与 consumer 验证。下一阶段建议按
以下顺序推进：

1. 补 11.8 的多 observer 生命周期通知，避免形成第二份 Session registry；
2. 在 ingress 保留请求对象前实现 11.9 的动态 admission，再进入 scheduler；
3. 继续完成 11.10 的剩余项：异步 fallback、后处理 continuation、HTTP/2 和真实 transport 分段测量；
4. 最后扩展 async gRPC Client/server-streaming、CloudTaskCoordinator 和增量输出。

| 优先级 | 能力 | 与 Application 主链路的关系 |
|---|---|---|
| P0 | Reference Gateway 解耦与通用组件安装导出 | 正式 Application 组合根的包边界；Route/Proto 可并行开发 |
| P0 | gRPC Client/Server 异步运行时 | 所有层间和微服务调用的公共底座；新 Adapter 优先采用，现有情绪服务不强制重写 |
| P1 | TaskGroup、deadline、取消和统一 snapshot 集成 | 提升运行治理；可在 unary 主链路建立后逐步接入 |
| P1 | CloudTaskCoordinator 增强 | 主要服务算法层管道，不阻塞普通 Persona Turn |
| P2 | gRPC server-streaming、LLM 增量输出和浏览器 SSE 支撑 | 前端体验和过程可观测性优化，不阻塞完整结果交互 |
| P2 | ConnectionPool 流连接治理 | 仅在启用大量长连接和流式输出时成为必要能力 |

各项应保持独立接口和独立测试，允许按模块分别合入 AgentLoom。不得使用一个总开关迫使下游等到全部能力完成后才能升级安装包。

## 4. CloudTaskCoordinator

### 4.1 取消与 deadline

Coordinator 的同步、异步和流式执行接口应接受 `std::stop_token` 和绝对 deadline，并把它们传播到：

- admission/排队等待；
- orchestrator；
- 每个 chunk worker；
- Provider HTTP 请求；
- retry backoff；
- ordered result sink。

取消或超时统一返回 `core::Status`。不得只停止等待方而让 Provider 请求在后台无界继续运行。

### 4.2 多 batch 并发

移除单个 Coordinator 实例由全局 `execute_mutex_` 导致的 batch 级串行瓶颈。可选实现方式：

1. Coordinator 原生管理多个 active batch；
2. 提供有明确容量和生命周期的 shard abstraction；
3. 将 batch state 独立为 RAII execution object，由共享 worker scheduler 调度。

无论采用何种方式，都必须设置 active batch、pending batch、总 chunk 和内存上限，不能用无限制创建线程或 Coordinator 实例换取并行。

### 4.3 增量有序结果

除现有完整结果返回外，提供 ordered incremental sink/callback。建议语义：

```cpp
class ICloudTaskResultSink {
public:
    virtual ~ICloudTaskResultSink() = default;
    virtual core::Status OnChunk(CloudTaskChunkResult result) = 0;
    virtual core::Status OnCompleted(CloudTaskSummary summary) = 0;
};
```

具体签名可按 AgentLoom 现有接口调整，但必须满足：

- `OnChunk` 按连续 `chunk_sequence` 调用；
- 单个 chunk 失败能够形成结构化失败结果或明确 skip，不阻塞后续序号；
- sink 背压可反馈给 producer；
- sink 返回失败后停止继续投递并触发 TaskGroup 取消；
- `OnCompleted` 恰好一次，携带成功、部分失败、取消或 deadline 状态；
- sink 生命周期由 RAII execution handle 保证，不使用无类型 `void*` context。

### 4.4 路由和执行元数据

`route_alias`、`attempt` 不应只作为不可观察字段。每个结果和 snapshot 至少暴露：

```text
task_id / batch_id / chunk_sequence
route_alias
provider/model（由选定 route 返回）
attempt
queue_wait / provider_latency / total_latency
provider_request_id
input/output/total token usage
terminal core::Status
```

AgentLoom 只提供通用 route policy 和执行元数据；具体租户预算和产品用途策略由调用方注入。

## 5. Provider Client 与 Retry

### 5.1 完整响应和增量响应

`ILlmClient` 应同时提供完整响应与增量响应抽象。增量接口需支持：

- headers/首 token/完成事件；
- delta sequence；
- finish reason；
- 最终 usage；
- Provider error；
- `std::stop_token` 和 deadline；
- consumer backpressure 或明确的有界缓冲策略。

`ChatCompletionRequest.stream=true` 必须对应真实的增量 transport，不能只保留请求字段而仍等待完整响应。

### 5.2 RetryPolicy

通用 retry 应增加：

- full jitter 或等价抖动算法；
- transport、HTTP status、Provider error 的结构化错误分类；
- `Retry-After` 支持；
- 最大 attempt 和最大累计 backoff；
- 剩余 deadline budget 检查；
- 是否已经产生可见增量输出的判断；
- 可重试、不可重试和结果不确定三种语义。

已经向 consumer 输出 token 后，不得透明重试并重复输出；需要由上层明确选择终止、续接或重新生成策略。

### 5.3 Fallback 和路由可观察性

`FallbackLlmClient`/route policy 应暴露结构化选择结果：

```text
selected route/provider/model
selection reason
fallback source/target
failure category
attempt
usage
health/circuit state
```

日志和指标不得包含 API key、完整 Prompt 或未经策略处理的敏感响应。

## 6. gRPC Client/Server 异步运行时

这是当前优先级最高、同时与业务逻辑耦合较低的通用演进项。其优先级不依赖 SSE 或增量输出，而是因为 AgentLoom 及其下游应用的层间交互、推理调用和微服务协作高度依赖 gRPC。目标不是修复现有同步 gRPC 的功能缺陷，而是为所有服务调用建立一致的并发、生命周期和资源模型。

现有 `EmotionGrpcService` 和相关同步 Client 保持兼容并继续使用。新 Runtime 可以先提供异步基础设施，由新的 Application Adapter 采用；是否迁移情绪服务由并发压测和线程占用数据决定。

### 6.1 架构价值

在多级调用链中，阻塞式 gRPC Client 会让网络等待、下游排队、retry backoff 和 deadline 等待持续占用调用线程。即使每个服务单独能够通过增加线程维持吞吐，多个服务串联后仍会产生线程放大、关闭困难和取消无法传递等问题。

通用异步 gRPC runtime 应成为以下调用的共同底座：

```text
gateway -> application
application -> emotion/inference
application -> algorithms/knowledge
service -> service
control/health/usage internal RPC
```

其价值包括：

- 等待 RPC 时不占用 Application IO/Compute worker；
- 一个上游 stop/deadline 可传播到所有下游 `ClientContext`；
- channel、stub、in-flight、admission 和 completion 线程统一治理；
- metadata、trace、认证和 `core::Status` 映射集中复用；
- 服务能够按“停止接入、取消/排空、完成 CQ、释放资源”的顺序关闭；
- 新增微服务只实现类型化 Adapter，不重新实现一套线程和取消机制。

因此该模块应尽早形成稳定公共接口，但允许业务服务一边使用现有同步实现推进、一边逐步接入异步 runtime。

### 6.2 异步 Client

> 实施状态（2026-08-13）：未完成。当前 `AgentLoom::grpc_runtime` 是通用异步 unary Server
> runtime，不提供异步 Client；不得把已完成的 Server callback runtime 误记为 Client/Server 全覆盖。

提供 unary 和 server-streaming 的异步调用封装，至少支持：

- CompletionQueue、callback 或等价 future/receiver 接口；
- RAII `ClientContext` 和 in-flight call handle；
- absolute deadline；
- `std::stop_token` 到 `ClientContext::TryCancel()` 的安全桥接；
- metadata/trace 注入；
- gRPC status 到 `core::Status` 的统一映射；
- completion 恰好一次；
- channel/stub 复用和有界 in-flight 数；
- 调用方无需阻塞 IO/Compute worker 等待 RPC 完成。

同步 Client API 可作为兼容外观保留，但不得通过在任意共享线程池中简单阻塞等待异步结果来伪装异步。

### 6.3 异步 Server

> 实施状态（2026-08-13）：部分完成。typed unary callback runtime 已完成并由
> `AsyncEmotionGrpcService` 验证；server-streaming、单 writer、有界 outbound queue 和统一 shutdown
> coordinator 尚未实现。

提供基于 AgentLoom Core/Net 生命周期约定的异步 unary 和 server-streaming 封装。可以采用 CompletionQueue 状态机或 callback/reactor API，但应由压测、关闭复杂度和当前 gRPC 版本共同决定，不要求重写已稳定的同步 Service。

该封装应：

- 以 RAII 管理 RPC context、writer/reactor、连接租约和 pending write；
- 保证单条 stream 同时最多一个 write；
- 使用有界 outbound queue，并同时限制 event count 和 bytes；
- 传播 client cancellation 和 deadline 到 `std::stop_source`/TaskGroup；
- 支持正常完成、取消、deadline、慢消费者和内部失败；
- 在 Finish 前完成明确的 seal/drain/cancel 顺序；
- 将 gRPC status 与 `core::Status` 做集中映射；
- 提供 stream snapshot 和健康指标。

该模块只负责通用 gRPC stream，不需要实现浏览器 SSE 或企业认证。

## 7. ConnectionPool 流连接治理

在现有 HTTP/WS 类别之外增加 stream 类别，或提供等价的可扩展连接分类机制。每类连接应支持：

- 最大连接数；
- 每连接和总内存预算；
- outbound queue 条目/字节预算；
- idle、heartbeat 和最大生命周期策略；
- tenant/purpose metadata；
- RAII `ConnectionLease`；
- active/rejected/slow-consumer/bytes snapshot。

实现应保持跨平台；平台句柄必须继续通过 `UniqueHandle` 或等价 RAII 类型封装。

## 8. TaskGroup 与 Runtime Session 生命周期

将 TaskGroup 接入 Runtime Session registry：

```text
Runtime Session TaskGroup
  -> Turn root task
     -> Persona/Skill child task
     -> Memory/Embedding child task
     -> Cloud batch/chunk child task
     -> Event/usage child task
```

要求：

- Session 进入 Finalize 后拒绝新的 root task；
- `Seal` 后等待已接收任务 drain，或在 deadline 到达后取消；
- `Close` 不得在仍有 task 使用 Session 资源时提前释放；
- 保留首个失败，同时统计后续失败；
- token/lease 使用 RAII，异常和提前返回时也能正确减少 active count；
- Session snapshot 暴露 active roots/children、sealed、cancelled 和 drain timeout。

## 9. 统一 Snapshot 与可观察性

为以下组件提供可聚合 snapshot：

- Blocking/Backpressure queue；
- OrderedBitmapWindow；
- KeyedSerialExecutor；
- TaskGroup；
- ThreadPool/scheduler；
- ConnectionPool/stream；
- CloudTaskCoordinator；
- Provider client、retry、fallback 和 circuit state。

统一指标至少覆盖：

```text
capacity / active / queued / bytes
queue_wait / execution_latency
accepted / rejected / cancelled / deadline_exceeded
ordered_gap / skipped / stale_sequence
retry / fallback / provider_error
drain_time / drain_timeout
```

Snapshot API 应读取成本可控、线程安全，且不会为采集指标阻塞关键执行路径。

## 10. 公共接口契约

所有新增或调整的公共接口必须明确：

- 输入、日志和协议文本采用 UTF-8；
- ownership 和对象生命周期；
- 线程安全级别及允许调用的线程；
- deadline 使用绝对时间还是相对时长；
- 取消是 best-effort 还是完成后保证无回调；
- queue 满、关闭、重复关闭和 drain timeout 的行为；
- callback 是否允许重入、阻塞或触发取消；
- 完成回调是否恰好一次；
- 错误统一包装为 `core::Status`/`core::Result<T>`；
- C API、gRPC handle 和连接资源使用 RAII；
- 不使用无类型 `void*` 传递业务上下文。

## 11. 构建与安装要求

新增通用能力需要：

- 纳入 AgentLoom 对应静态库目标，而不是示例 Gateway 私有实现；
- 通过安装后的 CMake Config Package 导出；
- 下游使用 `find_package` 和 `AgentLoom::*` target 链接；
- 公共头文件和传递依赖正确安装；
- gRPC 生成头文件在安装包中可定位；
- media、llama.cpp/mtmd 等可选大型依赖关闭时仍能使用这些能力；
- ONNX embedding、vector 和 emotion 的现有依赖边界不被误归入 media；
- Windows/Linux 的 feature option 和 target 名称保持一致。

### 11.1 Windows 运行库部署与 SQLite 包边界

> 实施状态（2026-08-13）：已完成，尚未提交。SQLite 头文件、Windows import library 与 runtime
> DLL 安装到 SDK 前缀，导出 target 不再引用源码树或构建树。详细验证见
> `AGENTLOOM_BUILD_AND_PACKAGE_PROGRESS_2026_08.md`。

Application 当前仿照 AgentLoom 的部署方式，通过 `ETS_RUNTIME_DEPENDENCY_DIRS`
收集已配置依赖目录中的 DLL，并在构建后用 `copy_if_different` 自动复制到各可执行
目标目录。该临时方案覆盖同一 vcpkg triplet 的运行库以及 ONNX Runtime、SQLite、
Faiss/MKL 等预编译依赖，避免依靠开发机全局 `PATH` 隐式满足运行条件。

当前 AgentLoom Config Package 仍将 `AgentLoom_SQLITE_LIBRARY` 导出为 SDK 构建树中
生成的 `sqlite3.lib` 绝对路径，并将 SQLite include/runtime 指向维护仓库源码布局。
因此独立 consumer 仍需在本机 preset 显式配置：

```text
AgentLoom_SQLITE_INCLUDE_DIR
AgentLoom_SQLITE_LIBRARY
AgentLoom_SQLITE_RUNTIME
```

AgentLoom 安装包应最终自包含或规范安装 SQLite 的 import library、runtime DLL 与头文件，
并让导出的 `sqlite3` imported target 只引用安装前缀内的可迁移路径。完成后 Application
可移除这三项本机覆盖；其他开发者在此之前仍应通过自己的 `CMakeUserPresets.json`
配置实际依赖位置，不应把个人绝对路径写入公共 CMake。

### 11.2 ConfigSection 公共 SDK ABI 与静态扩展边界

> 实施状态（2026-08-13）：已完成，尚未提交。公开 helper 已由 `AgentLoom::config` 提供；
> `nlohmann_json::nlohmann_json` 成为 public 依赖以固定 JSON inline ABI；安装包提供跨平台
> `agentloom_link_whole_archive()`，独立 consumer 的静态 registrar 已通过。

`AgentLoom::config` 当前已安装并公开 `config_section.h`，其中声明了供下游扩展
`server_config::IConfigSection` 使用的 JSON 辅助函数：

```text
BuildConfigSections / ValidateOptions
FindSection / FindField
SetString / SetPath / SetInt / SetSize / SetBool / SetMegabytes / ...
```

Application 以 `find_package(AgentLoom CONFIG)` 链接 `AgentLoom::config` 后，
`IConfigSection`、registry 和 `BuildConfigSections()` 可以链接，但 `FindSection`、`FindField`
及 `Set*` 辅助函数没有可供 consumer 链接的实现。下游因此不能只依赖安装 SDK 完整实现
自定义 ConfigSection；链接阶段会产生 `LNK2019` 未解析外部符号。

这属于 AgentLoom Config Package 的公共 ABI/静态库打包缺口，不应要求 consumer 复制
`config_section.cpp`，也不应通过引用 AgentLoom 维护仓库构建树规避。Application 当前以本地
强类型 JSON 读取辅助函数临时绕过该缺口，但仍复用 `IConfigSection`、注册器、内置 `llm`
section 和 `LlmOptions`，因此不阻塞 Gateway 推进。

AgentLoom 应选择并固定以下一种对外契约：

1. 将所有公开声明的 `server_config` 辅助函数编入并导出 `AgentLoom::config`，并由安装后
   独立 consumer 的 `find_package` 编译、链接测试验收；或
2. 将这些辅助函数移出公共头文件，提供稳定的、头文件自包含的 public parser API；
   consumer 不再依赖未导出的实现细节。

无论采用哪种形式，都应补充静态 section 扩展的安装包验收：consumer 自建一个静态库，其中
通过 `ConfigSectionRegistrar` 注册 section，使用公开 JSON parser API 完成加载和校验；可执行
目标对 `AgentLoom::config` 与 consumer 静态库使用 `/WHOLEARCHIVE`（或 Linux/macOS 等价链接器
选项）后，必须能发现 AgentLoom 内置 `llm` section 与 consumer section，并完成 LLM API key
文件的相对路径解析。该测试能够同时防止符号遗漏和静态 registrar 被链接器裁剪。

### 11.3 ConfigSection consumer profile 与选择性校验

> 实施状态（2026-08-13）：已完成，尚未提交。已新增 `ConfigSectionSelection::All/Only`，并为
> section 构造、JSON 加载、CLI fallback 和校验提供选择性重载；旧入口保持全量语义。

`BuildConfigSections()` 当前会构造并校验 SDK 中注册的全部服务器 section。Application Gateway
复用 `AgentLoom::config` 时会因此加载多模态服务器专用的 `models` section；该 section 要求
`--llm` 或 `--bert`，但这并不是 Application Gateway 的运行时装配契约。下游若跳过整个
AgentLoom 配置系统，又会失去 `llm`、Persona 默认配置和统一路径解析。

AgentLoom 应提供命名/profile 过滤的 section 构造与校验，例如由 consumer 明确选择
`core + llm + persona + gateway-foundation`，或者允许每个 section 声明适用的 server profile。
不能要求下游复制 loader、伪造无关模型参数或以 section 名称硬编码跳过校验。安装包测试需覆盖
一个只启用云 LLM、BERT emotion 和 Persona Runtime、未启用 VLM/media server 的 consumer。

### 11.4 Reference Gateway 解耦与目标图重构

> 实施状态（2026-08-13）：首轮抽取已完成并完成本地验证。`gateway_foundation`、
> `persona_interaction`、`gateway_routing` 与通用聚合 `gateway` 已独立导出；参考产品层以
> `reference_gateway` 保留认证、Document、Classroom 和参考 Route。`AGENTLOOM_BUILD_REFERENCE_GATEWAY=OFF`
> 的独立 package consumer 已编译、链接并运行，不会获得参考 Gateway target。

当前 `PersonaGatewayServer` 不是可直接导出的通用 Gateway foundation，而是 AgentLoom 参考产品的
组合根。它在同一个类中构造并持有：

- HTTP/WS server、Compute/IO pool、`SessionManager`、`PersonaRuntime` 和维护服务；
- JWT/Cookie 认证、注册接口、SQLite/Redis auth store；
- Persona metadata overlay；
- Document service、repository 和 retention task；
- Classroom scheduler、训练报告和参考 Route adapter；
- 静态文件处理和参考错误响应格式。

这些对象中既有可复用 Runtime 生命周期，也有参考产品策略。把当前类整体安装为
`AgentLoom::gateway`，会迫使下游接受并不属于自身产品的认证、文档、课堂和 Route context；在
Application 内重写 Session、线程池、维护和 HTTP 生命周期，又会产生第二套权威所有者。因此应在
AgentLoom 内重构耦合点，而不是要求 consumer 在边界外复制或绕过生命周期。

建议形成以下可独立安装的目标；具体名称可以调整，但职责和依赖方向不能倒置：

```text
AgentLoom::gateway_foundation
  HttpServer/ConnectionPool 生命周期辅助
  RuntimeMaintenanceService / SessionMaintenanceTask
  通用启动、失败回滚和关闭编排设施
  不包含 JWT/Cookie、Document、Classroom 和参考 Route

AgentLoom::persona_interaction
  transport-neutral Ensure/Get/Close/SubmitTurn/SystemStats
  复用 SessionManager + PersonaRuntime
  不包含 HTTP/WS DTO、参考认证、课堂、报告和文档业务

AgentLoom::gateway_routing
  通用 HTTP/WS Route registry 与 dispatcher
  consumer 定义的 route context/dependency provider
  可注入 prefix、认证适配器和错误映射
  不固定 PersonaGatewayService 或文档上传状态

AgentLoom::reference_gateway
  PersonaGatewayServer
  JWT/Cookie、metadata、Document、Classroom、报告和参考 Routes
  依赖上述通用组件
```

目标依赖方向应保持：

```text
core / net / runtime
          ^
gateway_foundation / persona_interaction / gateway_routing
          ^
reference_gateway
          ^
reference executable
```

Application 只链接通用组件，在自己的 `main` 中组装进程，不链接
`AgentLoom::reference_gateway`。普通 LLM、Memory、Cache、Storage、IPC 等静态库进入必要依赖闭包
只会增加有限静态体积，不应为了维持过窄边界而复制生命周期；但关闭 local LLM、media/VLM 时，
通用组件不得引入 llama.cpp、mtmd 或其他大型本地 LLM 推理依赖。

当前实现的实际边界如下：`agent_gateway_foundation` 提供 `GatewayLifecycleCoordinator`、
`RuntimeMaintenanceService` 和 `SessionMaintenanceTask`；`agent_persona_interaction` 提供
`IPersonaInteraction`；`agent_gateway_routing` 提供 header-only 的 typed route/middleware 扩展点。
`agent_gateway` 仅聚合这三项。`agent_gateway_server_lib`（导出为 `reference_gateway`）依赖公共层，
公共层不反向依赖 JWT/Cookie、Document、Classroom 或参考 Route。

### 11.5 Transport-neutral Persona 交互服务抽取

> 实施状态（2026-08-13）：已完成首轮抽取。`IPersonaInteraction` / `PersonaInteraction` 已提供
> `EnsureSession`、`CreateSession`、`GetSession`、`CloseSession`、`SubmitTurn` 和 `SystemSnapshot`；
> `PersonaGatewayService` 保留参考 DTO、metadata、Classroom 与报告职责，并将共用的 Session/Turn
> 用例委托给 interaction facade。

当前 `PersonaGatewayService` 同时包含可复用的 Session/Turn 用例和参考产品功能。可复用部分至少包括：

```text
Ensure/Create Runtime Session
Get Runtime Session snapshot
Close Runtime Session
Submit Persona Turn
Runtime/System snapshot
```

这些用例应抽取为不依赖 HTTP、WebSocket 和参考 Gateway DTO 的强类型接口与实现。实现继续直接使用
`SessionManager` 和 `PersonaRuntime`，由它们负责 affinity、Compute/IO 调度、Persona、Memory、
Emotion 和 LLM 交互闭环；Application 不再包装第二层队列或 Session registry。

接口应允许 consumer 注入可信身份、Persona/Session 配置解析、`RuntimeBootstrap`、动态 entitlement
和幂等/fencing 信息，但 AgentLoom 不保存 Go 业务 Session、PostgreSQL revision 或 OA 权限事实。
错误统一返回 `core::Status`/`core::Result<T>`，异步完成必须恰好一次并明确取消、deadline 和回调线程。

当前 `PersonaTurnRequest` 将 `ChatRequest` 与 `trusted_user_uuid` 分离；可信身份仅用于 Session
所有权校验，不会被混入模型请求。该 facade 不建立第二份 Session registry 或队列。

### 11.6 通用 Route registry、context 与认证适配

> 实施状态（2026-08-13）：已完成首轮抽取。`gateway_routing.h` 提供 consumer 自定义 Context 的
> `ITypedHttpRoute`、`ITypedWsRoute`、`TypedRouteRegistry`、dispatcher、prefix 处理和 ingress
> middleware；参考 route core 已在这些泛型抽象上保留其既有丰富 context。

现有 Route registry 的注册思想可以复用，但 `HttpRouteContext`、`WsRouteContext` 和
`PersonaGatewayHttpAdapter` 固定依赖 `PersonaGatewayService`、Document service、LLM、embedding、
cache、Skill、注册服务和参考 `AuthIdentity`，并硬编码 `/api`、错误 envelope 与上传状态。这些类型
不能作为 Application 的通用 Route 基座。

AgentLoom 应将 registry/dispatcher 重构为 consumer 可定义上下文的类型化扩展点，至少支持：

- Application 自定义 HTTP/WS Route，不必构造参考文档或课堂依赖；
- 可配置路径前缀、协议版本和错误映射；
- 注入可信身份适配器，不要求 JWT/Cookie authenticator；
- 每请求构造最小依赖视图，禁止用无类型 `void*` context；
- Route 静态注册可按组件显式 whole-archive，不能因链接一个通用 registry 自动拉入所有参考 Routes；
- HTTP 与 WS ingress 在进入业务服务前执行统一 admission、deadline、trace 和日志策略。

参考 Gateway 的 routes 应迁移为 `reference_gateway` 对通用 registry 的消费者，用这一回归验证抽取后的
扩展接口足以承载原有功能。

独立单元测试已验证 consumer 自定义路由、路径前缀剥离、path parameter 与 ingress middleware，
不要求构造 JWT、Document、Classroom 或参考 Route context。

### 11.7 稳定的组合生命周期契约

> 实施状态（2026-08-13）：已完成首轮实现。`GatewayLifecycleCoordinator` 支持注册期约束、顺序启动、
> 失败组件清理、逆序回滚、逆序停止、deadline 传递、重复 Start/Stop 和析构停止。参考
> `PersonaGatewayServer` 已通过该 coordinator 注册并编排其组件。

AgentLoom 应提供可由下游组合根调用的稳定生命周期契约，而不是要求下游复制
`PersonaGatewayServer::Start/Stop`。该契约可以是 foundation host、lifecycle coordinator 或一组
明确的 RAII service 接口，但必须支持：

- 构造阶段只建立依赖关系，不隐式启动线程或 listener；
- 分阶段 `Start`，每阶段失败按相反顺序回滚；
- 先关闭 admission 和 ingress，再 seal/drain/cancel in-flight work；
- 维护任务在 Session 和 pool 存活期间停止；
- IO pool 与 Compute pool 的排空顺序有明确契约；
- internal async gRPC server 与 HTTP/WS listener 可纳入同一关闭 deadline；
- 重复 Start/Stop、部分启动失败和析构均有确定行为；
- 所有失败使用 `core::Status` 并由宿主 logger 记录。

Application 拥有最终 `main` 和进程拓扑，意味着它选择并连接身份、Bootstrap、Route、gRPC adapter
与 AgentLoom 组件；这不意味着 Application 重写 ThreadPool、SessionManager、PersonaRuntime、
maintenance 或 HTTP transport。

参考 Server 当前注册顺序为 reference storage、compute pool、IO pool、maintenance、HTTP ingress，
因此关闭顺序确定为 HTTP ingress、maintenance、IO pool、compute pool、reference storage。生命周期
测试覆盖正常启动/逆序停止，以及组件启动失败后的失败组件清理和已启动组件回滚。

### 11.8 Session 关闭通知与单一生命周期所有权

> 实施状态（2026-08-13）：未完成，P0。当前 `SetSessionClosedCallback` 是单一可覆盖 callback；
> 需要升级为可组合 subscription/observer，并覆盖显式关闭、idle cleanup、maintenance 和 shutdown。

Application 需要把 Go 侧 `runtime_instance_id`、revision 和 fencing 状态投影到 AgentLoom
Session，但 Persona Session 的创建、自动卸载、Finalize 和 Close 必须只有 `SessionManager`
一个权威所有者。`CleanupExpired()` 删除 Session 后，下游投影必须收到同一关闭通知，否则会
留下仍显示 Active、实际 Session 已不存在的 ghost record。

AgentLoom 应提供可组合的 Session lifecycle observer/subscription，而不是只允许一个可覆盖的
`session_closed_callback_`。通知需覆盖显式 Close、idle expiry、维护任务清理和 shutdown，明确
回调线程、顺序、恰好一次语义及 unsubscribe/宿主析构关系。Application 不应为了接收通知而
维护第二份 Session registry 或清理线程。

### 11.9 动态配额策略与准入位置

> 实施状态（2026-08-13）：未完成，P0。现有 scheduler 的 fairness key、per-key 和 tenant 限额属于
> 静态运行时能力，尚未提供由可信 Bootstrap/entitlement 驱动的动态 quota provider，也未保证在
> ingress 保留大请求对象前完成 admission。

`GatewaySessionAffinityScheduler` 已支持 process-wide 的 active key、per-key、fairness-key 和
tenant 上限，且 metadata 来自可信 Session 状态。Application 的静态配置负责部署硬上限，Go
`RuntimeBootstrap` 则提供不同租户、用户和 Session 的动态 entitlement/预算；有效上限通常取
两者更严格值。

当前 scheduler options 主要是静态值。AgentLoom 应提供可注入的 quota policy/provider，或在
组合根暴露严格位于 scheduler 之前的有界 admission hook。准入必须发生在 consumer 保留 HTTP/WS
请求对象之前。若 Application 先把请求放入本地 `pending_turns`，scheduler 只能看到每个 Session
一个任务，既无法执行真实的 per-user/per-tenant 配额，也无法限制被请求对象占用的内存和连接。

### 11.10 Session 长锁与同步下游 IO

> 实施状态（2026-08-13）：第一阶段已完成，后续优化进行中，P0。纯 cloud Persona 主链路已经将
> LLM HTTP 等待移出 Session affinity worker，并使用 deferred completion 保留 lane/quota 到最终 commit；
> HTTP/1.1 outbound keep-alive 空闲连接池也已完成；async fallback、后处理 continuation、HTTP/2
> multiplexing 和动态 quota 仍未完成。

本轮额外修复了一个独立的确定性死锁：compute task 在持有 `SessionSlot` mutex 时向 IO pool 提交同一
Session，旧 admission 路径会再次锁定同一非递归 mutex。`SubmitIoFromSessionTask` 使用已锁定 Session
的可信身份完成转投，避免重复 admission 加锁；IO 执行阶段仍会重新定位并校验 Session。

`GatewaySessionAffinityScheduler` 已解决同 Session 多 worker 同时取任务的竞争；异步 Turn 路径改为短锁
读取不可变 snapshot、释放锁后发起异步 LLM、callback 中短锁 commit，最后才释放 deferred lane/quota。
同步兼容路径仍保留，因此未注入 async client 的部署仍可能占用 worker。

`PersonaGatewayServiceTest.RunsCreateChatReportAndCloseLifecycle` 曾卡死在该嵌套锁路径；
`SessionManagerTest.TransfersFromLockedComputeTaskToIoWithoutRelockingAdmission` 现以 2 秒 deadline
覆盖 compute-to-IO 转投。该修复不是长锁重构，不能据此降低本节优先级。

该结构已经在纯 cloud Persona 路径落地。新的 emotion/LLM Adapter 仍应传播 deadline/cancel；兼容同步
实现可保留，但不应由 Application 绕过 `SessionManager` 建立另一套队列。已补充同 Session 保序、不同
Session 并行、取消/关闭竞争及慢下游压测；详细数据见
`dev_note/AGENTLOOM_RUNTIME_ASYNC_E2E_BENCHMARK_2026-08-13.md`。

本轮 E2E 结论：真实 DeepSeek 100 个不同 Session Chat 全部成功；Fake 2 秒 Provider 延迟下，100
并发异步 burst 约 3.15 秒，同步 4-worker 基线约 50.21 秒；企业 quota Fake 在 500 并发时实际
达到 500 peak in-flight 且 500/500 成功。Python asyncio Mock 与复用 AgentLoom `HttpServer` 的 C++
Mock 结果相近，8 秒级额外客户端尾延迟不能简单归因于 Python 同步调度。HTTP/1.1 keep-alive 已在
后续冷/热轮试验中验证连接复用有效；剩余定位应通过 C++ load generator 分段测量客户端 connect、
request-write、first-byte 与 response-complete，并独立验证 HTTP/2 multiplexing。

历史同步真实 DeepSeek 100 并发报告也已完成交叉对照：当前异步 E2E P50/P95/P99 相比历史同步路径
分别降低 30.2%/50.1%/53.2%，`backendTotal` P95 降低 64.2%。当前测试的 Provider
`llmTotal` P95 反而从历史 1296 ms 上升到 1657 ms，但当前直接埋点的 `ioQueueWait` P50/P95/P99
均为 0；历史路径未单独埋点 queue wait，其 `backendTotal - llmTotal` 代理值 P50/P95/P99 为
1292/3655/3934 ms，当前仅 0/1/1 ms。该结果说明真实 API 收益来自消除本地同步排队放大，而非
Provider 当次变快。两轮配置和起跑方式不同，因此作为跨版本真实场景证据，因果关系仍以固定延迟
Fake A/B 为准。

前期 1000 并发压测提供了更直接的旧 IO pool 排队证据。Gateway 日志确认当时为 8 个 compute
worker、2 个 IO worker；三组瞬时起跑的 `ioQueueWait` P95 分别为 18.063/6.667/13.960 秒，
相同运行时配置下对应的 60 秒 ramp 轮次分别为 16.426/2.743/0.655 秒。三组的 Provider
`llmTotal` P95 仅约 1.5 至 2.0 秒，说明同步 LLM HTTP 长期占用有限 IO worker 后，Provider 延迟被
放大为秒级至十秒级本地排队；ramp-up 可以缓解瞬时突发，但不能修复结构问题。完整 E2E、backend、
IO queue 分位数和原始报告索引见专项压测报告 5.3 节。

固定延迟同步兼容 A/B 的任务路由与上述旧版本不同：整个同步 Turn 位于 compute task，因此
100x2 秒同步组记录为 `computeQueueWait` P95 45.279 秒、`ioQueueWait` P95 0；异步组两项均为 0。
这不是指标矛盾，而是同步等待所在 worker pool 已变化。验收必须同时检查任务路由、compute/IO queue
wait、Provider in-flight 与 E2E，不能只凭某一个 queue 字段为 0 宣称没有本地排队。

Python asyncio 与 C++ Mock 的单轮 500×2 秒对照中，C++ burst 缩短约 541 ms，E2E P95 降低约
280 ms。Python asyncio 不是 8 秒级额外尾延迟的主因，但数百毫秒绝对差距仍值得通过多轮预热和
客户端分段计时复核，不能简单写成两种实现完全等价。

HTTP/1.1 outbound keep-alive 已按 origin 接入 `AsyncBeastHttpClient`，并具备全局/单 origin 空闲连接
上限、idle timeout、错误/取消连接淘汰和 Shutdown 收口。100 个 Session 各执行 2 轮固定 500 ms Chat
的三轮开关对照中，关闭连接池每轮建立 200 条 Provider 出站连接；开启后三轮分别建立 65/67/55 条，连接数
中位数由 200 降至 65，减少 67.5%。六轮共 1200 个 Chat 全部成功，`backendTotal`/`llmTotal` P95
始终约 514 至 515 ms，`computeQueueWait`/`ioQueueWait` P95 均为 0。E2E P95 没有同步改善，说明
连接 churn 已显著降低，但当前尾延迟主因仍位于负载端计时开始到 Gateway pipeline 开始之间；该结果
不能包装成延迟优化成功。原始数据和冷/热轮分位数见专项压测报告 6.4 节。

SQLite metadata/auth 的单写者争用仍作为轻量后端的客观限制保留记录，但它不是生产推荐方案，也
不是当前异步主链路验收的阻塞项。实际生产元数据后端为 PostgreSQL，后续生产治理应围绕 PG 连接池、
短事务、索引、隔离级别和可恢复错误重试展开；SQLite 的全局写串行化仅适用于保留该轻量后端时的
局部兼容，不应推广为生产关系型后端架构。

### 11.11 工期应急方案及退出条件

若交付窗口明确不允许先完成上述 AgentLoom 拆分，可从一个固定、已测试的 AgentLoom commit 将参考
Gateway 的必要组合代码复制到 Application 后进行产品化修改。该路径只作为应急方案，不是并行维护
两套 Runtime 的目标架构，并必须满足：

- 复制范围以 composition、Route 和 adapter 为主，不复制 `SessionManager`、`PersonaRuntime`、
  scheduler、ThreadPool、maintenance primitive 或 HTTP transport；
- 在文档和源码中记录上游 commit、差异清单和回归公共组件的删除计划；
- 修复仍优先提交到 AgentLoom，Application 副本不形成独立基础设施演进线；
- AgentLoom 通用目标可用后删除副本，以 package consumer E2E 证明行为等价。

只要当前工期仍允许修改 AgentLoom，默认选择 11.4 至 11.7 的上游重构。已经确认的耦合属于库本身的
技术债，修复它能够同时改善参考 Gateway 和后续 consumer，长期成本低于复制后持续同步。

## 12. 测试与验收

### 12.1 单元测试

- 多 batch 并行且不会被全局 mutex 串行；
- active/pending/chunk 容量限制和 `ResourceExhausted`；
- stop、deadline、retry sleep 和 Provider 请求取消；
- chunk 乱序完成后增量有序投递；
- chunk 失败、skip、sink 失败和 completion 恰好一次；
- Retry-After、jitter、deadline budget 和错误分类；
- gRPC 单 writer、队列背压和 Status 映射；
- TaskGroup Finalize/Close 竞争；
- snapshot 在高并发下的一致性和低开销。

### 12.2 集成与 E2E

- OpenAI-compatible mock server 的完整/增量响应；
- 429、5xx、连接中断和半流失败；
- gRPC client 断开向 CloudTaskCoordinator 传播取消；
- 慢 gRPC consumer 触发背压且内存有上界；
- Runtime Finalize 等待已接收 Turn，拒绝新 Turn；
- 多 Session 并行、单 Session 事件保序；
- `AGENTLOOM_BUILD_REFERENCE_GATEWAY=OFF` 时安装 AgentLoom，由独立 consumer 通过 `find_package`
  编译、链接并运行 foundation、interaction 和 routing；
- consumer 注册自定义 HTTP/WS Route，无需构造 JWT、Document、Classroom 或参考 Route context；
- lifecycle coordinator 的正常启动/逆序停止及失败组件清理、已启动组件回滚；
- compute task 向同 Session IO task 转投时不得发生 admission mutex 重入死锁；
- reference Gateway 在抽取后的公共组件上保持原有认证、Persona、Document、Classroom 与 Report E2E；
- 纯 cloud Persona Session 异步 LLM E2E：不同 Session 并发、同 Session 保序、主动取消和关闭收口；
- 启动各阶段注入失败时完整回滚，HTTP/WS/RPC in-flight 关闭时按 deadline drain/cancel；
- Session idle cleanup 同时通知 L0 release 和下游 runtime projection，且每个 observer 恰好一次。

### 12.3 压测

- 多 batch、多 chunk 和多 Provider 并行吞吐；
- Persona 多 Session 慢 LLM 并行吞吐；真实 API 100 并发；Fake Provider 250/500 quota 并发；
- Python asyncio Provider Mock 与 AgentLoom C++ HttpServer Provider Mock 对照；
- 长连接数、每连接队列和总内存上界；
- P50/P95/P99 queue wait、首 chunk 和完成延迟；
- 取消风暴、Provider 429 风暴和慢消费者；
- snapshot/metrics 开启与关闭时的性能差异。

## 13. 各演进项完成定义

以下条件是各模块独立完成后的总体目标。Application 的配置、Route、Proto 和 Go adapter 可以并行开发；
正式 Runtime 组合默认以第 8 项及 11.4 至 11.7 节的目标拆分可用为门禁，或显式采用 11.11 节的应急路径：

1. CloudTaskCoordinator 支持有界多 batch、取消、deadline 和 ordered incremental sink；
2. LLM client 具备真实增量 transport，retry/fallback 行为结构化且可观察；
3. gRPC async Client/Server 支持 unary 和 server-streaming，并具备 RAII、单 writer、背压和取消传播；
4. ConnectionPool 能治理流连接及其字节预算；
5. Runtime Session 的 Finalize/Close 与 TaskGroup 正确集成；
6. 核心并发、stream、cloud 和 provider 组件提供统一 snapshot；
7. 公共接口满足 UTF-8、线程安全、关闭顺序、`core::Status` 和 RAII 契约；
8. foundation、interaction、routing 等静态库、公共头文件和依赖通过安装包正确导出，reference
   Gateway 只依赖它们而不被它们反向依赖；
9. 每个独立演进项均有对应的单元、集成、E2E 和性能测试；
10. 企业 OA、PostgreSQL 业务事务、浏览器 SSE 和产品路由没有被耦合进 AgentLoom。
