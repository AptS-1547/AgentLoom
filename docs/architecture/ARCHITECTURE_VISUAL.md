# 教育智能体架构图解

> 文档状态：历史架构图解。图中的旧项目名和职责划分用于说明演进背景，不作为当前 AgentLoom 部署拓扑。

> 简化版架构说明文档，提取自 COMMERCIAL_ARCHITECTURE.md
> 版本：v1.3 | 创建日期：2026-05-24 | 更新日期：2026-05-27
>
> **更新说明（v1.3）**：
> - 记忆体系收敛为 **L0 + L3 + L4** 三层，原 L1（working）/ L2（state）由 L0 的 Context Window 模式 + 批次轮换在事实上覆盖，不再单独实现。
> - L4 保留但重新界定职责：**统计意义上的用户画像** + **低频写、高频读的工具库记忆**，与 L3 的"日级事实压缩"明确区分。
> - 新增 §5.5「L0 事实替代 L1 / L2」、§5.6「L3 长期记忆压缩器」、§5.7「L4 用户画像与工具库记忆」。
> - 修订 §4 缓存层和 §7 降级策略中过时的层级描述。

---

## 1. 总体架构

```
                                    ┌─────────────────┐
                                    │   客户端/前端    │
                                    │  (Vue 3 + Vite) │
                                    └────────┬────────┘
                                             │ HTTPS/WSS
                                             ▼
                                    ┌─────────────────┐
                                    │      Nginx      │
                                    │  (反向代理/SSL)  │
                                    └────────┬────────┘
                                             │ HTTP/WS
                                             ▼
┌────────────────────────────────────────────────────────────────────────┐
│                          C++ 网关服务 (Gateway)                          │
│  - HTTP/WebSocket 协议处理                                              │
│  - Session 路由和管理                                                   │
│  - Exact / Semantic / RAG / Memory 缓存路由                               │
│  - 推理引擎路由、降级策略和故障隔离                                      │
└───┬────────────────┬────────────────┬────────────────┬─────────────────┘
    │ gRPC           │ gRPC           │ gRPC           │ gRPC (async)
    ▼                ▼                ▼                ▼
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌──────────────┐
│ C++ 人格 │    │ C++ 推理 │    │ Python   │    │ Python RAG   │
│  服务   │    │  服务   │    │ 记忆服务  │    │    服务      │
└─────────┘    └─────────┘    └─────────┘    └──────────────┘
```

---

## 2. 对话热路径缓存策略

```text
请求规范化 / 意图识别 / 上下文风险判断
  -> Exact Cache
  -> Curated Semantic Cache
  -> Knowledge RAG Cache
  -> Memory Context Layer
  -> vLLM 在线推理
  -> llama.cpp 备用推理 / 降级响应
```

**查询顺序说明**：
- 知识型问题优先 RAG
- 通用对话优先语义缓存
- 强上下文依赖问题必须先检查当前会话和记忆 scope

---

## 3. 修订后的服务边界（2026-05-20）

```
┌────────────────────────────────────────────────────────────────┐
│  C++ Gateway（HTTP/WS, 8080）                                  │
│  - 协议接入、Session 路由、限流、降级、超时治理                   │
└──────────┬─────────────────────────────────────┬───────────────┘
           │                                     │
   缓存命中（高频热路径）                         缓存未命中（低频）
           │                                     │
           ▼                                     ▼
┌──────────────────────────────────┐   ┌──────────────────────────────┐
│  C++ Cache Service (50055)       │   │  Python Persona Service      │
│                                  │   │  (50052, gRPC + asyncio)     │
│  - HF Tokenizer Pool             │   │                              │
│  - ONNX Embedding (MiniLM)       │   │  保留主项目原型逻辑：         │
│  - Faiss / 自研向量索引          │   │  - OU 情绪状态机             │
│  - SQLite metadata + 指纹        │   │  - BERT+LLM 情绪融合          │
│  - Redis payload (Phase 8+)      │   │  - Prompt 模板 + 组装        │
│                                  │   │  - 三路 LLM Router           │
│  对话热路径 Read 主力：           │   │  - 云 RAG 调用               │
│  返回 cached response 或 nil     │   │  - 多模型异步并发管理         │
└──────────────────────────────────┘   └──────┬───────────────────────┘
           ▲                                  │
           │ Cache Write                      │
           │ (Persona 端结果回写)             │
           └──────────────────────────────────┤
                                              ▼
                                   ┌─────────────────────────┐
                                   │  C++ Inference Service  │
                                   │  (50053)                │
                                   │  - BERT 情绪小模型      │
                                   │  - 本地 llama.cpp / mtmd│
                                   │  Python 通过 gRPC 调    │
                                   └─────────────────────────┘
                                              │
                                              ▼
                                   ┌─────────────────────────┐
                                   │  Cloud LLMs             │
                                   │  Claude / DeepSeek /     │
                                   │  GLM-4V / 智谱 RAG       │
                                   └─────────────────────────┘
```

---

## 4. 缓存层设计

| 缓存层 | 主要内容 | 命中范围 | 存储建议 |
|--------|----------|----------|----------|
| **Exact Cache** | 完全相同 prompt/context/model/persona 的响应 | 单用户、单 session 或严格 scope | 本地 LRU + Redis |
| **Curated Semantic Cache** | 离线生成和审核的高频通用教育对话 | 全局、租户、persona 或主题 scope | 本地热向量索引 + Redis payload |
| **Knowledge RAG Cache** | 教材、课标、经典例题、知识卡和标准讲解 | 学科、年级、教材版本、知识点 scope | SQLite metadata + Faiss/向量索引 + Redis 热缓存 |
| **Memory Context Cache** | L0 工作记忆 + L3 日级事实 + L4 用户画像 / 工具库 | user / session / tenant scope | L0：Redis 批次 + SQLite 索引；L3 / L4：SQLite 向量仓储 |

---

## 5. L0 语义缓存实现（Phase 5，2026-05-24）

### 5.1 双层架构设计

L0 语义缓存采用双层架构，通过枚举类型路由到不同的检索策略：

```cpp
enum class CacheMode {
    CONTEXT_WINDOW,      // 上下文窗口模式：Top-K 窗口筛选
    SEMANTIC_SIMILARITY  // 语义相似度模式：Top-1 最大相似度
};
```

**两种模式对比**：

| 特性 | Context Window 模式 | Semantic Similarity 模式 |
|------|---------------------|-------------------------|
| **用途** | 上下文重建，提取对话窗口 | 缓存命中，找最相似回答 |
| **检索目标** | Top-K 条记录 + 周围邻居 | Top-1 最大相似度记录 |
| **相似度阈值** | 较低（0.7-0.8） | 较高（0.85-0.95） |
| **邻居提取** | 需要（[idx-N/2, idx+N/2]） | 不需要 |
| **返回结构** | `vector<SearchWithContextResult>` | `CacheLookupResult` |
| **典型 top_k** | 8-16 | 1 |

### 5.2 核心数据流

```text
用户请求
  │
  ├─> HF Tokenizer (Rust FFI)
  │     └─> token_ids
  │
  ├─> ONNX Embedding Model (MiniLM-L12-v2, 384 dim)
  │     └─> normalized embedding vector
  │
  ├─> VectorIndexManager::Search / SearchWithContext
  │     │
  │     ├─> LoadActiveBatch (SQLite active_batch 表)
  │     ├─> LoadTimestampIndex (SQLite timestamp_index 表)
  │     │
  │     ├─> 计算总记录数 = active_count + (timestamp_index.size() × 10K)
  │     │
  │     ├─> 策略选择：
  │     │     ├─ 总记录 < 2×MAX_CACHE_RECORDS (20K)
  │     │     │    └─> 全量扫描所有批次
  │     │     │
  │     │     └─ 总记录 ≥ 2×MAX_CACHE_RECORDS
  │     │          └─> 随机采样批次 (uniform_int_distribution)
  │     │
  │     ├─> LRANGE 从 Redis 加载批次数据
  │     │     key = "cache:batch:{user_uuid}:{timestamp}"
  │     │     format = [dim][embedding×384][input_len][input][response_len][response]
  │     │
  │     ├─> SIMD 向量相似度计算 (AVX2 FMA, dot_product_unrolled<384>)
  │     │
  │     └─> partial_sort Top-K / Top-1
  │           └─> 返回结果
  │
  └─> 根据 CacheMode 返回：
        ├─ CONTEXT_WINDOW: 构建 message 数组上下文
        └─ SEMANTIC_SIMILARITY: 返回 cached response 或 cache miss
```

### 5.3 批次轮换机制

**写入流程**：
```text
AddRecord(record)
  │
  ├─> RPUSH "cache:batch:{user_uuid}:{active_timestamp}" {serialized_record}
  ├─> active_count++
  ├─> SaveActiveBatch() → SQLite active_batch 表
  │
  └─> if active_count >= MAX_CACHE_RECORDS (10K):
        ├─> PromoteBatchToIndex()
        │     ├─> timestamp_index.push_back(active_timestamp)
        │     ├─> StoreTimestampIndex() → SQLite
        │     └─> DELETE FROM active_batch
        │
        └─> 重置 active_timestamp = now(), active_count = 0
```

**读取流程（≥2×MAX 时）**：
```text
ReloadNextBatch()
  │
  ├─> 随机选择一个历史批次 (uniform_int_distribution)
  ├─> LRANGE "cache:batch:{user_uuid}:{selected_timestamp}" 0 -1
  ├─> 反序列化到 search_records_
  │
  └─> 从 timestamp_index_ 中移除该批次（swap to back + pop）
```

**设计理由**：
- 随机采样（非 LRU）保证无偏近似，符合统计假设
- 10K 记录窗口在 384 维下误差极低（Top-K 近似准确率 >99%）
- 单用户 1000 QPS 竞态窗口 <1s，顺序冲突概率可忽略

### 5.4 技术细节

**向量相似度计算**：
- 预归一化 embedding（L2 norm = 1）
- AVX2 SIMD 点积：`dot_product_unrolled<384>`
- 4 路展开 FMA 指令（`_mm256_fmadd_ps`）
- 性能：~0.5 µs/pair @ 384 dim

**存储格式**：
- Redis：二进制序列化 CacheRecord（~1556 bytes/record @ 384 dim）
- SQLite：active_batch 表（user_uuid, timestamp, count）
- SQLite：timestamp_index 表（user_uuid, timestamps BLOB）

**连接池**：
- 4 个 Redis 连接，每个独立 io_context + jthread
- Round-robin 调度（`atomic<size_t>` fetch_add）
- 每个连接独立 mutex 保护 async_exec

**生命周期**：
- Per-request：HTTP handler 创建 VectorIndexManager → 使用 → 析构
- Per-record 持久化：每条记录立即 RPUSH 到 Redis
- 内存峰值：单请求 ~3 MB（10K records × 1556 bytes + search_records_）

### 5.5 L0 事实替代 L1 / L2（2026-05-27）

原 COMMERCIAL_ARCHITECTURE 设计的 L1（working memory，进程内最近对话窗口）和 L2（session state，会话级压缩缓存）在 Phase 5 落地后**不再单独实现**。原因是 L0 的双层架构已经天然覆盖了这两层职责：

| 原设计层 | 原职责 | L0 中的对应实现 |
|----------|--------|-----------------|
| **L1 working memory** | 当前会话最近 N 轮对话，毫秒级读取 | `CacheMode::CONTEXT_WINDOW` + Top-K 邻居窗口（[idx-N/2, idx+N/2]） |
| **L2 session state** | 会话级语义压缩 / 主题聚合 | active_batch（10K 记录滚动）+ timestamp_index 历史批次 |
| **L0 semantic similarity** | 跨会话语义命中 | `CacheMode::SEMANTIC_SIMILARITY`，Top-1 高阈值 |

**关键差异**：
- L0 的批次轮换机制（§5.3）使得"最近对话窗口"和"会话状态"都是同一份 Redis + SQLite 数据的不同读取模式，没有独立的内存数据结构。
- 邻居提取（CONTEXT_WINDOW）直接拿到时间相邻的 message 数组，等价于 L1 的滑动窗口；不需要在网关里再维护一层进程内 LRU。
- 单用户单批次 10K 记录 ≈ 数千轮对话，远超传统 session window 容量，L2 的"会话压缩"在统计上也已经被 partial_sort Top-K 取代。

**收益**：
- 砍掉两层抽象，少两套并发同步 / 失效 / TTL 治理。
- 写入路径只写一份（RPUSH），读取路径按 CacheMode 分发，没有跨层一致性问题。
- 故障域收敛到 Redis + SQLite + VectorIndexManager 这一条链。

**代价**：
- L0 的访问延迟略高于纯进程内 L1（多一次 Redis LRANGE），目前实测在 10K 记录下仍在毫秒级。
- 没有显式的"会话边界"，需要靠 user_uuid + 时间戳过滤来约束 scope（已通过 timestamp_index 实现）。

### 5.6 L3 长期记忆压缩器

L3 是当前唯一的"持久化记忆"层，由 `src/memory/long_term_memory_compressor` 实现，2026-05-27 接入。

```text
每日触发（cron 或会话结束）
  │
  ├─> FetchDailyRecords(user_uuid, date)
  │     └─> 从 Redis 拉取当日所有 L0 CacheRecord
  │
  ├─> CompressToFacts(records)
  │     └─> 调用 OpenAI 兼容 LLM（DeepSeek / GLM 等）
  │         └─> 抽取 bullet-point 结构化事实
  │
  ├─> EmbeddingPipeline.Embed(facts)
  │     └─> 每条 fact → 384 维向量
  │
  └─> IVectorRepository.UpsertEntries
        └─> SQLite vector_entries 表，memory_level = "L3"
            partition: (collection_id, tenant_id, user_uuid, "L3")
```

**对外接口**（`LongTermMemoryCompressor`）：

| 方法 | 用途 |
|------|------|
| `CompressDailyMemory(user, date)` | 同步压缩当日记忆，返回处理记录数 |
| `CompressDailyMemoryAsync` | 后台线程异步压缩，返回 future |
| `GetDailySummary` / `GetUserSummaries` | 按日期或用户读取已压缩摘要 |
| `SearchFacts(user, query, top_k)` | 对 L3 事实做语义检索（注入 Prompt） |
| `StoreSummary` | 测试 / 迁移用，直接写入预构建摘要 |

**模式**：
- `Production`：要求 llm_client + embedding_pipeline 全部就绪
- `Testing`：允许两者为 nullptr，跳过 LLM/embedding 调用，只验证存储路径

**与 L0 / L4 的协作**：
- L0 负责"短期 + 完整"：原始对话内容、毫秒级命中
- L3 负责"中期 + 摘要"：当日抽取的稳定事实，按日期可枚举，跨日跨会话语义检索
- L4 负责"长期 + 统计 / 工具"：用户画像聚合 + 低频写、高频读的工具库（详见 §5.7）
- 三层之间靠 `memory_level` 字段在同一份 `IVectorRepository` 中分区，无双写，无 cross-layer 一致性协议

### 5.7 L4 用户画像与工具库记忆（2026-05-27 修订）

L4 在 v1.3 中保留，但职责从早期"L3 的更冷归档层"重新界定为**两类正交数据**：

| 子类 | 内容 | 写入特征 | 读取特征 |
|------|------|----------|----------|
| **L4-Profile** 用户画像 | 学科偏好、错题分布、学习风格、活跃时段、典型表达 | 由 L3 / L0 异步聚合，秒级延迟可接受 | 每次会话开场注入 Prompt，高频只读 |
| **L4-Toolbox** 工具库记忆 | 公式表、知识卡、教师批改模板、官方答案、固定话术 | 极低频，多由人工或离线流程写入 | 几乎每轮都可能命中，读多写极少 |

**关键设计原则**：

1. **统计意义**而非事件流：L4-Profile 不存储原始对话，只存聚合后的统计量（计数、平均、分布、Top-K 标签），单条记录覆盖一段时间或一个维度。这一点和 L3 "每日事实列表" 形成天然分层。
2. **永不遗忘**：L4 没有 TTL，没有 LRU 淘汰。Toolbox 类内容尤其需要稳定，遗忘等于事故。
3. **低频写、高频读**：写路径走异步 batch，读路径走与 L3 共用的向量索引 + SQLite 主键直查。读多写少 → 不需要锁分级，直接走 IVectorRepository 的现成接口即可。
4. **明确的功能边界**：
   - 不是"L3 的归档"——L3 的事实是按日期组织的对话摘要，过期可裁剪；L4 是跨时间的稳定知识。
   - 不是"知识库（KnowledgeRagCache）"——KnowledgeRag 是教材 / 课标级别的全局共享内容；L4 是用户绑定（Profile）或租户绑定（Toolbox）的私有内容。
   - 不是"Persona 配置"——Persona 是模型行为参数；L4 是被注入 Prompt 的事实数据。

**存储与分区**：

```text
vector_entries  (memory_level = "L4")
  ├─ scope = User    → L4-Profile（partition: tenant + user）
  └─ scope = Tenant  → L4-Toolbox（partition: tenant + "_toolbox_")
```

复用现有的 `IVectorRepository` + `PartitionRegistry`，不新增表结构。和 L3 唯一的区分仅在 `memory_level` 与 `scope_extras`。

**实现节奏建议**：

| 阶段 | 内容 | 依赖 |
|------|------|------|
| Phase 6.1 | L4-Toolbox 只读路径：手工写入工具，运行时读取注入 Prompt | 已就绪：IVectorRepository + EmbeddingPipeline |
| Phase 6.2 | L4-Profile 统计聚合器：定时 job，从 L3 / L0 抽取标签和分布 | 依赖 L3 稳定运行一段时间，积累足够样本 |
| Phase 6.3 | L4 → Prompt 注入策略：Persona 服务读取 L4 并合成 system prompt | 依赖 LlmEngineRouter |

**与 L3 的边界规则**（避免重叠）：

- 一条信息出现 1 天以内 → 仅 L0
- 出现 1 天以上、按天可枚举 → L3
- 出现至少 7 天 / 在多日 L3 中重复 → 由聚合器升格写入 L4-Profile
- 人工 / 离线写入的稳定知识 → 直接进 L4-Toolbox，不走 L0 / L3

---

## 6. 服务端口规划

| 服务 | 端口 | 协议 | 说明 |
|------|------|------|------|
| Nginx | 443 / 80 | HTTPS / HTTP | 外部入口，HTTP → HTTPS 重定向 |
| Gateway | 8080 | HTTP / WebSocket | 内部服务，不对外暴露 |
| Persona | 50052 | gRPC | 人格服务 |
| Inference | 50051 | gRPC | 推理服务（已改为 50053） |
| Memory | 50053 | gRPC | 记忆服务 |
| RAG | 50054 | gRPC | RAG 服务 |
| Cache | 50055 | gRPC | 缓存服务 |
| Prometheus | 9090 | HTTP | 监控数据采集 |
| Grafana | 3000 | HTTP | 监控看板 |

---

## 7. 故障隔离和降级策略

| 服务崩溃 | 影响范围 | 降级策略 |
|---------|---------|---------|
| **vLLM 主推理服务** | 在线 LLM 主路径不可用或显存压力过高 | 优先返回缓存/RAG 结果；必要时降级到 llama.cpp 备用模型 |
| **llama.cpp 备用推理** | 备用推理不可用 | 返回缓存结果或明确的降级响应 |
| **人格服务** | 当前请求使用默认情绪 | 使用 baseline 情绪和通用 Prompt |
| **记忆服务（L3 压缩器）** | 当日长期记忆压缩失败，但对话不中断 | 仅依赖 L0 的最近对话窗口，跳过 L3 事实召回 |
| **L4 用户画像 / 工具库** | Profile 注入或 Toolbox 命中失败 | Profile：使用通用 system prompt 占位；Toolbox：跳过工具记忆注入，由 LLM 自行作答 |
| **RAG 服务** | 不注入知识库上下文 | Prompt 中不包含 RAG 召回内容 |
| **Redis 缓存** | 跨实例共享缓存不可用 | 使用本地热索引和进程内缓存 |
| **网关服务** | 整个系统不可用 | 多实例部署 + Nginx 负载均衡 |

---

## 8. 职责边界

| 职责 | 归属 | 原因 |
|------|------|------|
| 协议接入、限流、Session、降级 | C++ Gateway | 高 QPS、多连接、需要资源治理 |
| 语义缓存查询（热路径 Read） | C++ Cache Service | 90%+ 命中率目标，必须毫秒级 |
| 缓存写入（Cache Write） | C++ Cache Service | Persona 完成调用后回写 |
| Tokenizer / Embedding 计算 | C++ | 高频，CPU/GPU 并行 |
| Faiss 向量索引、SQLite 元数据 | C++ | 数据层，复用基础设施 |
| 本地 LLM 推理（llama.cpp） | C++ Inference Service | 已 C++ 实现，无理由跨 IPC |
| BERT 情绪小模型推理 | C++ Inference Service | 已 ONNX 化 |
| OU 状态机、情绪融合、Prompt 组装 | Python Persona | 业务编排，迭代频繁 |
| 三路 LLM 路由、Cloud LLM 客户端 | Python Persona | 瓶颈在外部 API，C++ 无收益 |
| 云 RAG 调用（教师库 / 学生档案） | Python Persona | SDK 生态在 Python，冷路径 |

---

## 9. 设计原则

1. **故障隔离**：通过 gRPC 隔离各服务，单个服务崩溃不影响整体
2. **性能优先**：热路径（高频操作）用 C++ 实现，冷路径（低频操作）保留 Python
3. **快速迭代**：业务逻辑密集的模块（记忆压缩、RAG）保留 Python
4. **水平扩展**：各服务独立部署，可按需扩展
5. **缓存优先**：高频通用对话、标准知识和可复用回答优先通过缓存/RAG 返回
6. **降级策略**：每个服务都有明确的降级方案

---

## 10. 流媒体与视觉感知模块

**详细架构请参阅**：[STREAMING_ARCHITECTURE.md](STREAMING_ARCHITECTURE.md)

### 10.1 核心组件

```
客户端摄像头流
  │ WebSocket (WSS)
  ▼
流媒体专用连接池 (C++)
  │ 帧数据
  ▼
全局计算线程池
  ├─> OpenCV 筛帧 + FFT 分析
  ├─> Faiss 向量匹配
  └─> SIMD Top-K 搜索
  │
  ▼
VLM 推理服务 (gRPC 异步)
  │ 视觉事件 + 语义描述
  ▼
结果回写 L0 语义缓存
  └─> 注入对话上下文
```

### 10.2 关键技术

- **MOG2 背景减除**：自适应背景模型，区分有意义变化和噪声
- **三信号融合**：面积占比 + 直方图距离 + 边缘密度，权重 0.5/0.2/0.3
- **峰值检测**：时域平滑 + 局部极大值 + 清晰度优选
- **批量 VLM 推理**：batch=4-8，降低单帧延迟从 2s 至 0.5s
- **异步回写**：视觉事件包装为 CacheRecord，写入 L0 + 注入 AgentLoop

### 10.3 性能指标

| 指标 | 值 |
|------|-----|
| 帧处理吞吐量 | 1000 fps |
| 本地处理延迟 | ~10 ms |
| VLM 推理延迟 | 500-2000 ms (批量优化后) |
| WebSocket 连接数 | 100-500 (单实例) |
| L0 缓存写入 | 1000 QPS |

---

**文档版本**：v1.3
**创建日期**：2026-05-24
**更新日期**：2026-05-27
**来源**：COMMERCIAL_ARCHITECTURE.md v2.2 + Phase 5 L0 实现 + L3 长期记忆压缩器 + L4 画像/工具库重新界定 + 流媒体架构
**状态**：记忆体系收敛为 L0 + L3 + L4，原 L1 / L2 已废弃；L4 重新界定为统计画像 + 工具库记忆
