# 教育智能体商业化架构方案

> 基于多模态推理、人格管线、记忆系统的微服务架构设计
> 版本：v2.1 | 创建日期：2026-05-10 | 最后更新：2026-05-16

## 目录

- [1. 架构概览](#1-架构概览)
- [2. 商业化需求分析](#2-商业化需求分析)
- [3. 微服务架构设计](#3-微服务架构设计)
- [4. 代码复用与迁移分析](#4-代码复用与迁移分析)
- [5. 工作量估算](#5-工作量估算)
- [6. 部署方案](#6-部署方案)
- [7. 实施路线图](#7-实施路线图)
- [8. 附录](#8-附录)

---

## 1. 架构概览

### 1.1 总体架构图

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

### 1.2 设计原则

1. **故障隔离**：通过 gRPC 隔离各服务，单个服务崩溃不影响整体
2. **性能优先**：热路径（高频操作）用 C++ 实现，冷路径（低频操作）保留 Python
3. **快速迭代**：业务逻辑密集的模块（记忆压缩、RAG）保留 Python
4. **水平扩展**：各服务独立部署，可按需扩展
5. **缓存优先**：高频通用对话、标准知识和可复用回答优先通过缓存/RAG 返回，降低在线 LLM 推理压力
6. **降级策略**：每个服务都有明确的降级方案，LLM 主路径采用 vLLM，低显存/边缘/故障场景保留 llama.cpp 备用路径

---

## 2. 商业化需求分析

### 2.1 规模需求

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **并发用户数** | 1,000 - 10,000 | 千级乃至万级才能回本 |
| **QPS** | 约 30 - 330 对话请求/秒 | 1,000 - 10,000 并发用户 × 每用户每分钟 2 条消息 ÷ 60 |
| **Session 数** | 1,000 - 10,000 | 每个用户独立 Session |
| **记忆 IO** | 约 3 - 30 写入/秒 | 按每 10 轮触发一次 L2/L3/L4 压缩和持久化估算 |
| **响应延迟** | p95 < 3s | 包含推理时间 |

### 2.2 性能瓶颈分析

| 操作 | 频率 | 单次延迟 | Python 瓶颈 | 解决方案 |
|------|------|---------|------------|---------|
| **WebSocket 连接管理** | 持续 | ~1ms | GIL 限制多核 | C++ 协议层 |
| **Session 路由** | 每次请求 | ~1ms | 字典查找，可接受 | C++ 优化 |
| **情绪状态机更新** | 每次请求 | ~2ms | GIL 限制并发 | C++ 实现 |
| **L1 缓存读取** | 每次请求 | ~1ms | 内存操作，可接受 | C++ 优化 |
| **Prompt 构建** | 每次请求 | ~5ms | 字符串拼接，可接受 | C++ 优化 |
| **Exact Cache 查询** | 每次请求 | <1ms | 无 | C++ 本地缓存 + Redis |
| **语义缓存检索** | 高频 | 1-10ms | Python/Numpy 不适合网关热路径 | C++ SIMD 向量核 + 本地热索引 + Redis |
| **RAG 标准知识缓存** | 按需 | 10-100ms | Python 生态成熟但需隔离热路径 | 结构化知识卡 + 向量检索 |
| **BERT 推理** | 每次请求 | ~10ms | 已在 C++ | 无需改动 |
| **LLM 在线推理** | 缓存/RAG 未命中时 | 秒级 | 32B 级模型高 QPS 成本高 | vLLM 主路径 + llama.cpp 备用 |
| **VLM 推理** | 低频 | ~3000ms | 已在 C++ | 保留 llama.cpp / mtmd 路径 |
| **L1→L2 记忆压缩** | 每 N 轮 | ~50ms | 文件 IO 瓶颈 | 异步 + SQLite |
| **L3/L4 记忆召回** | 会话开始 | ~100ms | 数据库查询 | Python + 索引优化 |
| **深度 RAG 检索** | 按需 | ~200ms | 向量数据库 | Python 生态成熟 |

**结论**：热路径（每次请求）需要 C++，冷路径（低频触发）可以保留 Python。商业化部署不能假设 32B 级本地 LLM 能直接承载数百 QPS；对话链路应优先经过 Exact Cache、离线构建的语义缓存基底、RAG 标准知识缓存和多级记忆上下文筛选，仅在缓存/RAG/记忆无法满足时进入在线 LLM 推理。

### 2.3 对话热路径缓存策略

对话热路径缓存是商业化高并发场景的核心降载机制，不应仅作为推理失败后的兜底。建议采用混合方案：

1. **离线语义缓存基底**：从中文教育对话数据库中筛选高频场景，使用 DeepSeek、GLM、Qwen 等中文能力强的模型生成回答，经清洗、去重、质量评审后写入语义缓存。该层覆盖通用教学话术、学习困惑、常见课堂互动和稳定的低上下文依赖回答。
2. **知识库 RAG 缓存**：将课本、课程标准、经典例题、常见误区和题型讲解沉淀为结构化知识卡与向量索引。知识型问题优先从该层返回或组合回答，保证标准性和可追溯性。
3. **多级记忆兜底**：保留 L1/L2/L3/L4 记忆作为长上下文基础设施。记忆层用于判断缓存命中是否安全、提供个体化渲染上下文，并在缓存/RAG 不足时作为在线 LLM 推理输入。

建议查询顺序如下：

```text
请求规范化 / 意图识别 / 上下文风险判断
  -> Exact Cache
  -> Curated Semantic Cache
  -> Knowledge RAG Cache
  -> Memory Context Layer
  -> vLLM 在线推理
  -> llama.cpp 备用推理 / 降级响应
```

实际执行顺序应由意图路由决定：知识型问题优先 RAG；通用对话优先语义缓存；强上下文依赖问题必须先检查当前会话和记忆 scope。

缓存命中不能只依赖向量相似度，必须同时满足：

- intent / answer_type 兼容
- subject / grade / topic scope 兼容
- persona scope 兼容
- memory scope 兼容
- prompt / policy / embedding model / corpus version 一致
- response quality score 达标
- entry 未过期且未被人工或自动评估淘汰

包含“这个、那个、刚才、上面、这一步、图中、为什么错”等强指代词的请求默认视为上下文依赖请求，不允许直接命中全局语义缓存。

---

## 3. 微服务架构设计

### 3.1 服务划分

#### 3.1.1 C++ 网关服务 (Gateway Service)

**职责**：
- HTTP/WebSocket 协议处理（面向客户端）
- Session 路由和管理
- L1 记忆缓存（最近 12 轮对话，内存中）
- Exact Cache、语义缓存、RAG 缓存和记忆上下文的路由协调
- 本地热向量索引和 Redis 共享缓存的访问协调
- vLLM / llama.cpp 推理后端选择和降级
- 降级策略协调
- 健康检查和监控

**技术栈**：
- C++20
- Boost.Beast (HTTP/WebSocket)
- gRPC Client (调用下游服务)
- nlohmann/json (配置解析)

**关键特性**：
- 轻量级：只做协议、路由、缓存协调和降级，不承载复杂业务推理
- 无状态：Session 状态可以从 L1 缓存恢复
- 缓存优先：优先命中 Exact / Semantic / RAG / Memory 缓存，降低 LLM 在线推理压力
- 降级优先：下游服务不可用时，使用缓存、默认人格策略或备用推理路径

**代码规模**：~10,000 行（包含流媒体解析、WebSocket 管理、Session 池）

---

#### 3.1.2 C++ 人格服务 (Persona Service)

**职责**：
- 情绪状态机更新（Ornstein-Uhlenbeck 过程）
- 动态 Prompt Hint 生成
- Prompt 构建（模板系统）
- 人格参数管理

**技术栈**：
- C++20
- gRPC Server
- Eigen (数学计算，可选)

**关键特性**：
- 无状态：所有状态从请求中传入
- 纯计算：无 IO，延迟低（~5ms）
- 可水平扩展：部署多个实例

**核心算法**：
```cpp
// 情绪状态机：二维连续情绪 (valence, arousal)
class EmotionStateMachine {
    float valence, arousal;
    float baseline_valence, baseline_arousal;

    void Update(float user_v, float user_a, float ai_v, float ai_a) {
        // Ornstein-Uhlenbeck 过程
        valence = std::tanh(phi_v * valence + theta_v * baseline_valence
                           + kappa * (arousal - baseline_arousal)
                           + beta * ai_v + gamma * user_v + noise());
        arousal = std::tanh(phi_a * arousal + theta_a * baseline_arousal
                           + kappa * (valence - baseline_valence)
                           + beta * ai_a + gamma * user_a + noise());
    }

    std::string GenerateHint() const;
};
```

**代码规模**：~8,000 行（C++ 代码膨胀系数 3.2x）

---

#### 3.1.3 C++ 推理服务 (Inference Service)

**职责**：
- BERT 情绪分类
- VLM 视觉语言推理
- 人脸检测 + 情绪识别
- 本地向量相似度核心和轻量向量缓存
- 语义缓存 embedding 生成 / 归一化 / Top-K 检索接口
- VRAM 监控和模型生命周期

**技术栈**：
- C++20
- ONNX Runtime (BERT + 人脸情绪模型)
- llama.cpp (VLM / 低显存备用 LLM)
- vLLM (主力高 QPS LLM serving，作为独立 OpenAI-compatible 服务接入)
- Redis (可选共享语义缓存)
- Faiss + SQLite (向量索引和元数据持久化，逐步接入)
- gRPC Server

**关键特性**：
- 已有代码基础：AgentBackendPredict v0.0.1beta1
- 故障隔离：CUDA OOM 不影响其他服务
- LLM 主路径采用 vLLM 承担高并发在线推理；llama.cpp 保留为低显存、边缘设备和 OOM 降级备用路径
- 轻量向量缓存：基于预归一化 embedding、SIMD 点积、固定维度 384/768 模板特化和 Top-K 检索减少重复推理
- 本地热索引优先，Redis 作为共享缓存和失效协调层，不在每次相似度扫描中承担全量向量 IO

**代码规模**：~5,500 行（已有服务端推理核心约 4,600 行，需扩展人脸情绪模型和 gRPC 服务）

---

#### 3.1.4 Python 视觉服务 (Vision Service)

**职责**：
- 视觉感知管线（复用小橘 VisualPerceptionPipeline）
- 自适应帧采样
- 视觉语义分析
- 视觉监控和事件检测

**技术栈**：
- Python 3.10+
- gRPC Server
- OpenCV (图像处理)
- 复用 MyNeuroLikeSystem 视觉模块

**关键特性**：
- 复用小橘代码：~2,000 行可直接复用
- 改造为 gRPC 服务
- 已验证有效：端到端测试通过

**代码规模**：~3,500 行（复用 ~2,000 行 + 新增/改造 ~1,500 行 gRPC 封装和服务化适配）

---

#### 3.1.5 Python 记忆服务 (Memory Service)

**职责**：
- L1→L2 记忆压缩（结构化状态提取）
- L3/L4 跨会话记忆沉淀
- 记忆召回和检索
- 持久化（SQLite/RocksDB）
- 遗忘曲线机制
- 为语义缓存提供 memory scope 校验和个体化渲染上下文

**技术栈**：
- Python 3.10+
- gRPC Server
- SQLite/RocksDB (持久化)
- asyncio (异步 IO)

**关键特性**：
- 异步调用：网关不等待压缩完成
- 降级策略：失败时返回空记忆，不阻塞主流程
- 复用小橘代码：MyNeuroLikeSystem 的 memory_manager.py（更完善版本）
- 包含遗忘曲线机制
- 不作为全局语义缓存的来源；包含用户长期记忆或会话上下文的回答只能进入 user/session scope 缓存

**代码规模**：~2,500 行（复用 ~1,700 行 + 新增 ~800 行 gRPC 封装和持久化）

---

#### 3.1.6 Python RAG 服务 (RAG Service)

**职责**：
- 课本、课程标准、经典例题和常见误区的知识库检索
- 结构化知识卡构建和查询
- RAG 标准答案缓存管理
- 证据片段和引用来源返回
- 必要时为在线 LLM 提供上下文注入

**技术栈**：
- Python 3.10+
- gRPC Server
- LangChain/LlamaIndex
- Milvus/Qdrant

**关键特性**：
- 知识型问题优先触发，通用对话不强制调用
- 标准知识回答优先来自知识卡和教材/课标证据，避免离线语义缓存长期替代权威知识源
- 降级策略：失败时不注入知识库上下文，但不得错误返回无来源的权威性结论
- Python 生态成熟；后续可将热知识卡索引迁移到 C++ 本地向量层

**代码规模**：~1,500 行（新实现）

---

#### 3.1.7 商业化必需模块

**用户认证授权**（~3,000 行 C++）：
- JWT/OAuth2 实现
- 多租户隔离
- RBAC 权限管理
- Session 持久化

**计费系统**（~2,000 行 C++）：
- 使用量统计（对话轮次/Token/时长）
- 配额管理和限流
- 计费规则引擎
- 账单生成

**管理后台 API**（~2,000 行 C++）：
- 用户管理
- 数据统计和报表
- 系统配置
- 日志查询

**公共库和工具**（~3,000 行 C++）：
- 日志系统
- 配置管理
- 监控 Metrics
- 工具函数

**测试代码**（~18,000 行）：
- 单元测试（C++ + Python）
- 集成测试
- 压力测试
- E2E 测试

**配置和部署**（~2,500 行）：
- Docker/K8s 配置
- CI/CD 脚本
- Nginx 配置
- Prometheus/Grafana 配置

**前端**（~18,000 行）：
- 管理后台（~8,000 行 Vue 3）
- 学生端（~10,000 行 Vue 3）

---

### 3.2 服务总览

| 服务 | 代码量 | 语言 | 复杂度 | 说明 |
|------|--------|------|--------|------|
| **C++ 网关服务** | ~10,000 | C++ | ⭐⭐⭐⭐⭐ | 流媒体 + WebSocket + Session 池 |
| **C++ 人格服务** | ~8,000 | C++ | ⭐⭐⭐⭐ | 情绪状态机 + Prompt 构建 |
| **C++ 推理服务** | ~5,500 | C++ | ⭐⭐⭐ | BERT + VLM + 人脸情绪 |
| **Python 视觉服务** | ~3,500 | Python | ⭐⭐⭐ | 复用小橘视觉链路 |
| **Python 记忆服务** | ~2,500 | Python | ⭐⭐⭐ | 复用小橘记忆系统 |
| **Python RAG 服务** | ~1,500 | Python | ⭐⭐ | LangChain 封装 |
| **认证授权** | ~3,000 | C++ | ⭐⭐⭐⭐ | JWT + 多租户 + RBAC |
| **计费系统** | ~2,000 | C++ | ⭐⭐⭐ | 使用量统计 + 配额管理 |
| **管理后台 API** | ~2,000 | C++ | ⭐⭐ | 用户管理 + 报表 |
| **公共库** | ~3,000 | C++ | ⭐⭐⭐ | 日志 + 配置 + 监控 |
| **测试代码** | ~18,000 | C++/Python | ⭐⭐ | 单元 + 集成 + 压力 + E2E |
| **配置部署** | ~2,500 | YAML/Shell | ⭐⭐ | Docker + K8s + CI/CD |
| **前端** | ~18,000 | Vue 3 | ⭐⭐⭐ | 管理后台 + 学生端 |
| **总计** | **~80,000** | - | - | |

---

### 3.3 服务间通信

#### 3.3.1 gRPC 协议定义

```protobuf
// gateway.proto
service GatewayService {
    rpc HandleRequest(UserRequest) returns (UserResponse);
    rpc Health(HealthCheckRequest) returns (HealthCheckResponse);
}

// persona.proto
service PersonaService {
    rpc Process(PersonaRequest) returns (PersonaResponse);
    rpc Health(HealthCheckRequest) returns (HealthCheckResponse);
}

message PersonaRequest {
    string user_id = 1;
    repeated Message l1_messages = 2;  // 最近 12 轮对话
    EmotionInfo user_emotion = 3;      // 用户情绪（来自 BERT）
    EmotionInfo ai_emotion = 4;        // AI 上一轮情绪
    PersonaConfig persona = 5;         // 人格配置
}

message PersonaResponse {
    string prompt = 1;                 // 构建好的 Prompt
    float valence = 2;                 // 更新后的情绪状态
    float arousal = 3;
    string hint = 4;                   // 动态 Prompt Hint
}

// memory.proto
service MemoryService {
    rpc Compress(CompressRequest) returns (CompressResponse);
    rpc Recall(RecallRequest) returns (RecallResponse);
    rpc Health(HealthCheckRequest) returns (HealthCheckResponse);
}

// inference.proto (已有，需扩展)
service InferenceService {
    rpc PredictEmotion(EmotionRequest) returns (EmotionResponse);
    rpc GenerateVLM(VLMRequest) returns (stream VLMResponse);
    rpc DetectFaceEmotion(FaceEmotionRequest) returns (FaceEmotionResponse);  // 新增
    rpc Health(HealthCheckRequest) returns (HealthCheckResponse);
}
```

#### 3.3.2 故障隔离和降级策略

| 服务崩溃 | 影响范围 | 降级策略 |
|---------|---------|---------|
| **vLLM 主推理服务** | 在线 LLM 主路径不可用或显存压力过高 | 优先返回缓存/RAG 结果；必要时降级到 llama.cpp 备用模型 |
| **llama.cpp 备用推理** | 备用推理不可用 | 返回缓存结果或明确的降级响应 |
| **人格服务** | 当前请求使用默认情绪 | 使用 baseline 情绪和通用 Prompt |
| **记忆服务** | 记忆压缩失败，但对话继续 | 只用 L1 缓存，不压缩到 L2/L3/L4 |
| **RAG 服务** | 不注入知识库上下文 | Prompt 中不包含 RAG 召回内容 |
| **Redis 缓存** | 跨实例共享缓存不可用 | 使用本地热索引和进程内缓存，禁止将 Redis 作为唯一真源 |
| **网关服务** | 整个系统不可用 | 多实例部署 + Nginx 负载均衡 |

**网关服务的降级逻辑示例**：
```cpp
Response GatewayService::HandleRequest(const Request& req) {
    auto route = intent_router.Classify(req);

    // 1. Exact Cache / 语义缓存 / RAG 缓存优先
    if (auto exact = cache.ExactLookup(req); exact.ok()) {
        return BuildCachedResponse(exact.value());
    }
    if (!route.context_dependent) {
        if (auto semantic = cache.SemanticLookup(req, route); semantic.ok()) {
            return BuildCachedResponse(semantic.value());
        }
    }
    if (route.knowledge_question) {
        if (auto rag = rag_cache.Lookup(req, route); rag.ok()) {
            return BuildRagResponse(rag.value());
        }
    }

    // 2. 调用人格服务（带超时和降级）
    auto persona_result = persona_client.Process(req, /*timeout=*/100ms);
    if (!persona_result.ok()) {
        persona_result = GetDefaultPersona(req.user_id);
        LOG_WARN("Persona service unavailable, using default");
    }

    // 3. 在线 LLM 推理：vLLM 主路径，llama.cpp 备用路径
    auto inference_result = vllm_client.Generate(req, persona_result, /*timeout=*/5s);
    if (!inference_result.ok()) {
        inference_result = llama_fallback.Generate(req, persona_result, /*timeout=*/10s);
        LOG_WARN("vLLM unavailable or over capacity, trying llama.cpp fallback");
    }

    // 4. 异步触发记忆压缩和缓存写回（不阻塞主流程）
    if (l1_cache[req.user_id].NeedsCompression()) {
        memory_client.CompressAsync(req.user_id, l1_cache[req.user_id].messages);
    }
    cache.AdmitAsync(req, inference_result);

    return BuildResponse(persona_result, inference_result);
}
```

### 3.4 缓存与检索层设计

商业化架构中的缓存层分为四类，分别解决不同问题，不能混用：

| 缓存层 | 主要内容 | 命中范围 | 存储建议 | 风险控制 |
|--------|----------|----------|----------|----------|
| **Exact Cache** | 完全相同 prompt/context/model/persona 的响应 | 单用户、单 session 或严格 scope | 本地 LRU + Redis | 依赖版本 hash |
| **Curated Semantic Cache** | 离线生成和审核的高频通用教育对话 | 全局、租户、persona 或主题 scope | 本地热向量索引 + Redis payload | intent、scope、quality score、policy version |
| **Knowledge RAG Cache** | 教材、课标、经典例题、知识卡和标准讲解 | 学科、年级、教材版本、知识点 scope | SQLite metadata + Faiss/向量索引 + Redis 热缓存 | 必须保留来源和 corpus version |
| **Memory Context Cache** | L1/L2/L3/L4 记忆、用户画像、当前会话状态 | user/session scope | 记忆服务持久化 + 网关 L1 | 禁止进入全局缓存 |

推荐组件划分：

```text
ConversationAccelerationLayer
  -> ExactPromptCache
  -> CuratedSemanticCache
  -> KnowledgeRagCache
  -> MemoryContextLayer
  -> LocalHotVectorIndex
  -> RedisSharedCache
  -> LlmFallbackRouter
```

其中 `LocalHotVectorIndex` 负责本地相似度热路径，使用预归一化 float32 embedding、固定维度 384/768 模板特化、SIMD dot product 和 Top-K/threshold scan。Redis 负责跨实例共享、payload 存储、bucket 列表和失效协调，不应作为每次相似度扫描的全量向量来源。

离线语义缓存基底的构建流程：

```text
中文教育对话数据筛选
  -> 去重 / 聚类 / 意图分类
  -> DeepSeek / GLM / Qwen 等强中文模型生成
  -> 多模型交叉评审和规则清洗
  -> answer_core / response_text / metadata 结构化
  -> embedding 编码和归一化
  -> 写入 Redis / SQLite / 本地热索引快照
```

建议缓存 entry 至少包含：

- `entry_id`
- `canonical_question`
- `question_variants`
- `intent`
- `subject`
- `grade`
- `answer_type`
- `answer_core`
- `response_text`
- `persona_scope`
- `context_requirements`
- `embedding_model_fingerprint`
- `model_fingerprint`
- `prompt_version`
- `policy_version`
- `quality_score`
- `corpus_version`
- `created_at_ms`
- `ttl_seconds`

RAG 缓存应优先沉淀为结构化知识卡，而不是只存原始 chunk。知识卡建议包含定义、例子、常见误区、题型模板、引用来源和教材/课标版本。知识型回答优先由知识卡和证据片段生成，在线 LLM 只在多知识点综合、上下文复杂或检索冲突时介入。


---

## 4. 代码复用与迁移分析

### 4.1 现有代码资产

#### 4.1.1 AgentBackendPredict（C++ 推理端）

**已有代码**：~4,565 行（不含测试客户端和工具脚本）

| 模块 | 行数 | 状态 | 复用度 |
|------|------|------|--------|
| `multimodal_inference_server.cpp` | 1,132 | 需重构为独立推理服务 | 70% |
| `vlm_cache.cpp/h` | 660 + 92 | 直接复用 | 100% |
| `llama_runner.cpp/h` | 459 + 99 | 直接复用 | 100% |
| `request_validation.cpp/h` | 518 + 53 | 移到网关服务 | 80% |
| `onnx_model.cpp/h` | 422 + 117 | 直接复用 | 100% |
| `server_config.cpp` | 404 | 各服务独立配置 | 60% |
| `server_common.cpp/h` | 290 + 128 | 提取为公共库 | 100% |
| `vector_cache.cpp/h` | 136 + 53 | 作为轻量向量缓存原型继续演进 | 80% |

**汇总**：
- **可直接复用**：~2,300 行（推理核心、缓存系统）
- **需重构/迁移**：~1,700 行（服务端框架、配置系统、请求校验）
- **需新增**：~1,500 行（人脸情绪模型、gRPC 服务化封装等）
- **需新增缓存基础设施**：轻量 SIMD 向量相似度核、本地热向量索引、Redis 共享缓存适配和对话缓存 admission policy。

---

#### 4.1.2 EducationalAgentProject（Python 业务端）

**已有代码**：约 40,674 行（`agent/src`，仅列与商业化迁移直接相关的核心文件）

| 模块 | 行数 | 状态 | 复用度 |
|------|------|------|--------|
| `core/persona.py` | 760 | 核心逻辑移到 C++，配置保留 | 20% |
| `core/emotion_state.py` | 311 | 移到 C++ 人格服务 | 10% |
| `core/prompt_builder.py` | 155 | 移到 C++ 人格服务 | 30% |
| `core/scheduler.py` | 407 | 移到 C++ 网关服务 | 40% |
| `core/small_model.py` | 606 | 已在 C++ 推理服务 | 0% |
| `memory/memory_manager.py` | 545 | 改造为 gRPC 服务 | 80% |
| `adapters/openai_adapter.py` | 703 | 移到 C++ 网关服务 | 30% |
| `adapters/run_api_server.py` | 202 | 废弃（C++ 网关替代） | 0% |
| `llm/client.py` | 332 | 保留（调用云端 LLM） | 100% |
| 训练相关脚本 | ~3,000 | 完全保留 | 100% |
| 工具脚本 | ~1,500 | 完全保留 | 100% |

**汇总**：
- **可直接复用**：~5,000 行（记忆系统、LLM 客户端、训练工具）
- **需移植到 C++**：~2,500 行（人格管线核心逻辑）
- **需废弃**：~1,500 行（API 服务器、适配器）
- **完全保留**：~18,000 行（训练、工具、测试）

---

#### 4.1.3 MyNeuroLikeSystem（Python 小橘核心系统）

商业化评估只统计与迁移直接相关的核心模块。

| 模块 | 行数 | 状态 | 复用度 |
|------|------|------|--------|
| **vision/** | | | |
| `visual_pipeline.py` | 757 | **核心资产，移植到商业化** | 90% |
| `visual_monitor.py` | 443 | 视觉监控，移植 | 80% |
| `visual_semantics.py` | 366 | 语义分析，移植 | 85% |
| `visual_types.py` | 259 | 类型定义，移植 | 100% |
| `adaptive_sampler.py` | 151 | 自适应采样，移植 | 90% |
| `visual_analysis.py` | 91 | 分析模块，移植 | 90% |
| **memory/** | | | |
| `memory_manager.py` | 1,134 | 比 EducationalAgentProject 更完善 | 90% |
| `forgetting.py` | 415 | 遗忘曲线，移植 | 80% |
| **core_engine/** | | | |
| `persona.py` | 908 | 人格系统，部分移植到 C++ | 30% |
| `emotion_state.py` | 351 | 情绪状态机，移植到 C++ | 20% |
| `scheduler.py` | 407 | 调度器，移植到 C++ | 40% |
| `small_model.py` | 606 | 已在 C++ 推理服务 | 0% |
| **media/** | | | |
| `audio_pipeline.py` | 531 | 音频处理，保留 | 100% |
| `image_utils.py` | 233 | 图像工具，保留 | 100% |
| `speech_recognition.py` | 275 | 语音识别，保留 | 100% |
| **attention/** | | | |
| `attention_tracker.py` | 289 | 注意力追踪，移植 | 80% |
| **agent/** | | | |
| `agent_loop.py` | 412 | Agent 主循环，保留 | 100% |
| `llm/client.py` | 332 | LLM 客户端，保留 | 100% |
| 训练相关脚本 | ~3,000 | 完全保留 | 100% |
| 工具脚本 | ~2,000 | 完全保留 | 100% |

**汇总**：
- **视觉感知链路（核心资产）**：~2,067 行，约 85-90% 可复用
  - `VisualPerceptionPipeline`：完整的视觉感知管线，已通过端到端测试
  - `AdaptiveFrameSampler`：自适应帧采样
  - `VisualSemantics`：视觉语义分析
  - `VisualMonitor`：视觉监控和事件检测
- **记忆系统（比教育版更完善）**：~1,549 行，约 85-90% 可复用
  - 包含遗忘曲线（forgetting.py）
- **多模态媒体处理**：~1,039 行，100% 保留
- **完全保留**：~5,000 行（训练、工具、Agent 主循环）

> **关键决策**：记忆服务优先采用 MyNeuroLikeSystem 版本，而非 EducationalAgentProject 版本。

---

### 4.2 各服务迁移明细

#### 4.2.1 C++ 网关服务（新实现）

| 模块 | 功能 | 参考代码 | 行数估算 |
|------|------|---------|---------|
| `gateway_server.cpp` | HTTP/WebSocket 服务器 | 无（新实现，用 Boost.Beast） | 800 |
| `session_manager.cpp/h` | Session 路由和管理 | `scheduler.py` 部分逻辑 | 400 |
| `l1_cache.cpp/h` | L1 记忆缓存 | 无（新实现） | 300 |
| `conversation_cache_router.cpp/h` | Exact / Semantic / RAG / Memory 缓存路由 | 无（新实现） | 500 |
| `local_hot_vector_index.cpp/h` | 本地热向量索引、Top-K、threshold scan | `vector_cache.cpp/h` 原型 | 600 |
| `service_clients.cpp/h` | gRPC 客户端封装 | 无（新实现） | 300 |
| `degradation.cpp/h` | 降级策略 | 无（新实现） | 200 |

**核心逻辑约 2,000 行**，完整商业化服务约 10,000 行（另含流媒体解析、WebSocket 长连接治理、HTTP/2、多租户 Session 池、限流、审计日志、可观测性和压测支撑）。

---

#### 4.2.2 C++ 人格服务（移植 + 新实现）

| 模块 | 功能 | 参考代码 | 行数估算 |
|------|------|---------|---------|
| `persona_server.cpp` | gRPC 服务器 | 无（新实现） | 400 |
| `emotion_state.cpp/h` | 情绪状态机 | `emotion_state.py`（小橘版，409 行） | 500 |
| `prompt_builder.cpp/h` | Prompt 构建 | `prompt_builder.py`（188 行） | 400 |
| `persona_config.cpp/h` | 人格配置加载 | `persona.py` 部分逻辑 | 300 |
| `emotion_fusion.cpp/h` | BERT+LLM 情绪融合 | `emotion_fusion.py`（211 行） | 300 |
| `hint_generator.cpp/h` | 动态 Hint 生成 | `emotion_state.py` 部分逻辑 | 200 |
| `generation_params.cpp/h` | 生成参数自适应 | `emotion_state.py` 部分逻辑 | 200 |

**关键移植点**：Ornstein-Uhlenbeck 情绪状态机的 C++ 实现、Prompt 模板系统、单神经元 softmax 情绪融合。

核心算法约 2,300 行，完整服务约 8,000 行（另含配置热更新、人格版本管理、灰度策略、服务治理、监控指标）。

---

#### 4.2.3 C++ 推理服务（扩展现有代码）

**需新增的模块**：

| 模块 | 功能 | 参考代码 | 行数估算 |
|------|------|---------|---------|
| `face_emotion_model.cpp/h` | 人脸检测+情绪识别 | 无（新实现，用 ONNX Runtime） | 600 |
| `multimodal_inference_server.cpp` | 重构为独立服务 | 已有 1,132 行，需重构 | +200 |
| `vector_similarity.cpp/h` | 预归一化向量点积、SIMD、Top-K | 无（新实现） | 500 |
| `semantic_cache_client.cpp/h` | 语义缓存查询和写回接口 | 无（新实现） | 400 |
| `llm_engine_router.cpp/h` | vLLM 主路径和 llama.cpp 备用路径路由 | 无（新实现） | 400 |

**已有可复用代码（直接沿用）**：

| 文件 | 行数 | 说明 |
|------|------|------|
| `llama_runner.cpp/h` | 570 + 117 | VLM 推理 |
| `onnx_model.cpp/h` | 507 + 134 | BERT 推理 |
| `vlm_cache.cpp/h` | 744 + 108 | 向量缓存 |
| `vector_cache.cpp/h` | 165 + 66 | 轻量向量索引原型，需升级为可复用 similarity kernel |

---

#### 4.2.4 Python 视觉感知服务（移植小橘核心）

| 模块 | 功能 | 原始代码 | 改造工作量 |
|------|------|---------|-----------|
| `visual_service.py` | gRPC 服务器 | 无（新实现） | 200 行 |
| `visual_pipeline.py` | 视觉感知管线 | 已有 857 行 | 改造 10% |
| `adaptive_sampler.py` | 自适应帧采样 | 已有 186 行 | 改造 10% |
| `visual_semantics.py` | 语义分析 | 已有 399 行 | 改造 15% |
| `visual_monitor.py` | 视觉监控 | 已有 509 行 | 改造 20% |
| `visual_analysis.py` | 分析模块 | 已有 111 行 | 改造 10% |

**改造要点**：将 `VisualPerceptionPipeline` 封装为 gRPC 服务；集成 C++ 推理服务的人脸情绪识别；保留自适应帧采样逻辑（已验证有效）。

---

#### 4.2.5 Python 记忆服务（采用小橘完善版）

| 模块 | 功能 | 原始代码 | 改造工作量 |
|------|------|---------|-----------|
| `memory_service.py` | gRPC 服务器 | 无（新实现） | 200 行 |
| `memory_manager.py` | 记忆压缩和召回 | 已有 1,264 行（小橘版本） | 改造 10% |
| `forgetting.py` | 遗忘曲线 | 已有 464 行 | 改造 20% |
| `storage.py` | SQLite/RocksDB 持久化 | 无（新实现） | 300 行 |

**改造要点**：采用 MyNeuroLikeSystem 的 memory_manager.py；集成遗忘曲线机制；替换文件系统持久化为 SQLite/RocksDB。

---

#### 4.2.6 Python RAG 服务（新实现）

| 模块 | 功能 | 行数估算 |
|------|------|---------|
| `rag_service.py` | gRPC 服务器 | 200 |
| `vector_store.py` | 向量数据库封装（LangChain） | 300 |
| `retrieval.py` | 检索和排序 | 300 |
| `knowledge_card.py` | 知识卡结构化、证据引用和版本管理 | 300 |
| `rag_cache.py` | RAG 标准答案缓存和 corpus version 管理 | 300 |

---

### 4.3 代码复用总结

| 类别 | 行数 | 占比 | 来源 |
|------|------|------|------|
| **可直接复用** | ~12,000 | 15% | 推理核心、缓存、视觉/记忆部分模块、LLM 客户端 |
| **需移植改造** | ~26,000 | 33% | Python → C++、gRPC 服务化、微服务适配 |
| **需新实现** | ~42,000 | 52% | 协议层、认证、计费、前端、测试、部署 |
| **不计入商业化工作量** | ~23,000 | - | 训练脚本、工具脚本、历史 API、被替换模块 |
| **总计（交付口径）** | **~80,000** | 100% | |

> **注意**："可直接复用"仅指低改造成本代码，不包含需服务化、C++ 移植或接口重构的代码。


---

## 5. 工作量估算

### 5.1 估算修正说明

原始估算存在几处系统性低估，汇总如下：

| 低估原因 | 原估算 | 修正值 | 说明 |
|---------|--------|--------|------|
| **协议层复杂度** | ~2,000 行 | ~10,000 行 | 流媒体解析 + WebSocket 长连接管理 + Session 池 |
| **商业化必需功能（遗漏）** | 0 行 | ~7,000 行 | 认证授权 3,000 + 计费 2,000 + 管理后台 API 2,000 |
| **C++ 代码膨胀系数** | 1:1.2 | 1:3–5 | 类型声明、内存管理、RAII、错误处理 |
| **测试代码** | 未单独计入 | ~18,000 行 | 单元 + 集成 + 压力 + E2E |
| **配置和部署** | ~200 行 | ~2,500 行 | Docker / K8s / CI/CD / Nginx / Prometheus |

**修正后的总代码量：~80,000 行**（而非原估算的 ~7,000 行）。

---

### 5.2 模块工作量明细

| 模块 | 代码量 | 工作量（人月） | 复杂度 | 说明 |
|------|--------|--------------|--------|------|
| **C++ 协议层** | ~10,000 | 6.0 | ⭐⭐⭐⭐⭐ | 流媒体解析 + WebSocket + HTTP/2 + Session 池 |
| **用户认证授权** | ~3,000 | 2.0 | ⭐⭐⭐⭐ | JWT + OAuth2 + 多租户隔离 + RBAC |
| **计费系统** | ~2,000 | 1.5 | ⭐⭐⭐ | 使用量统计 + 配额管理 + 计费规则引擎 |
| **管理后台 API** | ~2,000 | 1.5 | ⭐⭐ | 用户管理 + 数据统计 + 系统配置 |
| **C++ 人格服务** | ~8,000 | 5.0 | ⭐⭐⭐⭐ | 情绪状态机 + Prompt 构建（C++ 膨胀 3.2x） |
| **C++ 推理服务** | ~5,500 | 2.5 | ⭐⭐⭐ | 扩展人脸情绪模型 + 多卡推理 |
| **Python 视觉服务** | ~3,500 | 2.0 | ⭐⭐⭐ | 改造小橘视觉链路为 gRPC 服务 |
| **Python 记忆服务** | ~2,500 | 1.5 | ⭐⭐⭐ | 改造小橘记忆系统 + SQLite 持久化 |
| **Python RAG 服务** | ~1,500 | 1.0 | ⭐⭐ | LangChain 封装 + 向量数据库集成 |
| **公共库和工具** | ~3,000 | 2.0 | ⭐⭐⭐ | 日志、配置、监控、工具函数 |
| **测试代码** | ~18,000 | 8.0 | ⭐⭐ | 单元 + 集成 + 压力 + E2E |
| **配置和部署** | ~2,500 | 2.0 | ⭐⭐ | Docker + K8s + CI/CD + Nginx + Prometheus |
| **前端管理后台** | ~8,000 | 3.0 | ⭐⭐⭐ | Vue 3 + Vite，用户管理、报表、配置 |
| **前端学生端** | ~10,000 | 3.5 | ⭐⭐⭐ | Vue 3 + Vite，学生交互界面 |
| **文档和培训** | — | 2.0 | ⭐⭐ | 技术文档 + API 文档 + 运维手册 |
| **总计** | **~80,000** | **约 36 人月** | | |

---

### 5.3 AI 辅助编程的效率影响

#### 5.3.1 任务类型效率分析

| 任务类型 | 传统效率 | AI 辅助效率 | 提升倍数 | 说明 |
|---------|---------|------------|---------|------|
| **样板代码** | 50 行/天 | 200 行/天 | 4x | gRPC 服务定义、CRUD 接口、配置解析 |
| **算法移植** | 100 行/天 | 250 行/天 | 2.5x | Python → C++（情绪状态机、Prompt 构建） |
| **新功能开发** | 80 行/天 | 150 行/天 | 1.9x | 认证授权、计费系统、协议层 |
| **测试代码** | 100 行/天 | 300 行/天 | 3x | 单元测试、Mock 数据生成 |
| **Bug 修复** | 5 个/天 | 10 个/天 | 2x | AI 辅助定位和修复 |
| **文档编写** | 500 字/天 | 1,500 字/天 | 3x | API 文档、技术文档 |

**综合效率提升（加权平均）：~2.2x**

AI 辅助效率因任务性质差异显著：

- **高效场景（3–4x）**：gRPC 框架生成、数据库 CRUD、配置解析代码、测试用例生成
- **中效场景（2–2.5x）**：Python → C++ 算法移植、情绪状态机、Prompt 构建逻辑
- **低效场景（< 1.2x）**：架构设计与技术选型、复杂业务建模、性能调优、跨服务集成调试、生产问题排查

**AI 辅助的风险**：生成代码质量参差不齐，需要人工审查；可能引入安全漏洞（SQL 注入、XSS 等）；过度依赖可能降低代码理解深度。

---

#### 5.3.2 基于 Filerestore_CLI 的效率校准

**参照项目数据**：

| 指标 | 数据 |
|------|------|
| 代码量 | ~49,326 行 C++（不含 build / third_party） |
| 开发时间 | 5 个月 |
| 实测效率 | **~9,865 行/月** |
| 技术特点 | NTFS MFT/USN 解析、minifilter 驱动、SIMD 优化、多线程 NVMe、TUI |

**商业化项目难度系数推导**：

| 维度 | Filerestore_CLI | 商业化项目 | 系数 |
|------|----------------|-----------|------|
| **代码复杂度** | 极高（底层系统编程） | 中等（业务逻辑为主） | 0.7x |
| **调试难度** | 极高（文件系统、内核驱动） | 中等（gRPC、微服务） | 0.6x |
| **可复用代码** | 0%（从零开发） | 约 25%（小橘等已有资产） | 1.5x |
| **测试覆盖** | 高（45 个单元测试） | 高（商业化要求） | 1.0x |
| **综合系数** | 1.0x | 0.7 × 0.6 × 1.5 × 1.0 = **0.63x** | |

**调整后效率**：9,865 × 0.63 ≈ **6,215 行/月**

---

#### 5.3.3 团队配置与排期对比

> 以下估算以 ~60,000 行有效开发工作量（总量 80,000 行，减去约 20,000 行可直接复用或低成本改造代码）为基准，含 30% 风险缓冲。

| 方案 | 团队配置 | 协作效率系数 | 有效开发速度 | 开发时间（含缓冲） | 可行性 | 推荐度 |
|------|---------|------------|------------|-----------------|--------|--------|
| **方案 1** | 1 人（单主力） | 1.0x | 6,215 行/月 | **~13 个月** | ❌ 超出交付周期 | ⭐ |
| **方案 2** | 2 主力 | 1.6x | 9,944 行/月 | **~8 个月** | ✅ 可行但紧张 | ⭐⭐⭐ |
| **方案 3** | 2 主力 + 1 辅助 | 2.2x | 13,673 行/月 | **~6 个月** | ✅ 推荐 | ⭐⭐⭐⭐⭐ |

**推荐方案 3**：
- 主力 1（协议层 / 认证授权 / RAG 服务 / 性能优化）
- 主力 2（人格服务 / 推理服务 / 视觉服务 / 记忆服务）
- 辅助（单元测试 / 集成测试 / 配置脚本 / 技术文档）

---

#### 5.3.4 阶段工作量分配（方案 3）

| 阶段 | 时间 | 主力 1 | 主力 2 | 辅助 |
|------|------|--------|--------|------|
| **Phase 1** | Month 1–2 | 协议层 + 认证授权 | 人格服务 + 推理服务 | Docker 配置 + 文档 |
| **Phase 2** | Month 3–4 | 计费系统 + 管理后台 | 视觉服务 + 记忆服务 | 单元测试 + 集成测试 |
| **Phase 3** | Month 5–6 | RAG 服务 + 性能优化 | 降级策略 + 健康检查 | 压力测试 + 部署脚本 |

---

## 6. 部署方案

### 6.1 Docker Compose 配置

```yaml
# docker-compose.yml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - gateway
    restart: always

  gateway:
    build:
      context: ./gateway
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - PERSONA_SERVICE_ADDR=persona:50052
      - INFERENCE_SERVICE_ADDR=inference:50051
      - MEMORY_SERVICE_ADDR=memory:50053
      - RAG_SERVICE_ADDR=rag:50054
    depends_on:
      inference:
        condition: service_healthy
      persona:
        condition: service_healthy
      memory:
        condition: service_healthy
    restart: always
    healthcheck:
      test: ["CMD", "grpc_health_probe", "-addr=:8080"]
      interval: 10s
      timeout: 5s
      retries: 3

  persona:
    build:
      context: ./persona
      dockerfile: Dockerfile
    ports:
      - "50052:50052"
    volumes:
      - ./config/persona:/config
    restart: always
    healthcheck:
      test: ["CMD", "grpc_health_probe", "-addr=:50052"]
      interval: 10s
      timeout: 5s
      retries: 3

  inference:
    build:
      context: ./inference
      dockerfile: Dockerfile
    ports:
      - "50051:50051"
    volumes:
      - ./models:/models
      - ./cache:/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - CUDA_VISIBLE_DEVICES=0
    restart: always
    healthcheck:
      test: ["CMD", "grpc_health_probe", "-addr=:50051"]
      interval: 10s
      timeout: 5s
      retries: 3

  memory:
    build:
      context: ./memory
      dockerfile: Dockerfile
    ports:
      - "50053:50053"
    volumes:
      - memory_data:/data/memory
    restart: always
    healthcheck:
      test: ["CMD", "grpc_health_probe", "-addr=:50053"]
      interval: 10s
      timeout: 5s
      retries: 3

  rag:
    build:
      context: ./rag
      dockerfile: Dockerfile
    ports:
      - "50054:50054"
    volumes:
      - rag_data:/data/rag
    restart: always
    healthcheck:
      test: ["CMD", "grpc_health_probe", "-addr=:50054"]
      interval: 10s
      timeout: 5s
      retries: 3

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: always

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: always

volumes:
  memory_data:
  rag_data:
  prometheus_data:
  grafana_data:
```

---

### 6.2 Nginx 配置

```nginx
# nginx/nginx.conf
upstream gateway {
    server gateway:8080;
    # 多实例扩容时追加:
    # server gateway-2:8080;
}

server {
    listen 443 ssl http2;
    server_name edu-agent.example.com;

    ssl_certificate     /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    # WebSocket 长连接
    location /ws {
        proxy_pass http://gateway;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    # HTTP API
    location /v1 {
        proxy_pass http://gateway;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # 健康检查（不计入访问日志）
    location /health {
        proxy_pass http://gateway;
        access_log off;
    }

    # 对话接口限流：10 req/s，burst 20
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    location /v1/chat {
        limit_req zone=api_limit burst=20 nodelay;
        proxy_pass http://gateway;
    }
}

server {
    listen 80;
    server_name edu-agent.example.com;
    return 301 https://$server_name$request_uri;
}
```

---

### 6.3 Systemd 服务配置（非容器部署备选）

```ini
# /etc/systemd/system/edu-gateway.service
[Unit]
Description=Educational Agent Gateway
After=network.target edu-inference.service edu-persona.service
Wants=edu-inference.service edu-persona.service edu-memory.service

[Service]
Type=simple
User=edu-agent
WorkingDirectory=/opt/edu-agent
ExecStart=/opt/edu-agent/bin/gateway_server --config /etc/edu-agent/gateway.json
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

---

### 6.4 监控与日志

#### 6.4.1 Prometheus Metrics

各服务均暴露 `/metrics` 端点：

```cpp
// C++ 服务 Metrics 导出示例
class MetricsExporter {
public:
    void IncrementRequestCount(const std::string& service) {
        request_count_[service]++;
    }

    void RecordLatency(const std::string& service, double latency_ms) {
        latency_histogram_[service].Observe(latency_ms);
    }

    std::string Export() const {
        return "# HELP request_count Total requests\n"
               "# TYPE request_count counter\n"
               "request_count{service=\"gateway\"} "
               + std::to_string(request_count_.at("gateway")) + "\n";
    }

private:
    std::unordered_map<std::string, std::atomic<uint64_t>> request_count_;
    std::unordered_map<std::string, Histogram> latency_histogram_;
};
```

关键指标：

| 指标名 | 类型 | 说明 |
|--------|------|------|
| `request_count{service}` | Counter | 各服务请求总数 |
| `request_latency_ms{service,p50/p95/p99}` | Histogram | 请求延迟分位数 |
| `grpc_error_count{service,code}` | Counter | gRPC 错误次数 |
| `l1_cache_hit_rate` | Gauge | 网关 L1 缓存命中率 |
| `exact_cache_hit_rate` | Gauge | Exact Cache 命中率 |
| `semantic_cache_hit_rate` | Gauge | 离线语义缓存命中率 |
| `rag_cache_hit_rate` | Gauge | RAG 标准知识缓存命中率 |
| `semantic_cache_wrong_hit_count` | Counter | 语义缓存误命中或人工回退次数 |
| `llm_saved_request_count` | Counter | 缓存/RAG 避免的在线 LLM 请求数 |
| `vllm_request_count` | Counter | vLLM 主路径请求数 |
| `llama_fallback_request_count` | Counter | llama.cpp 备用路径请求数 |
| `active_sessions` | Gauge | 当前活跃会话数 |
| `inference_queue_depth` | Gauge | 推理队列积压深度 |

#### 6.4.2 日志规范

所有服务统一采用 JSON 格式，包含 `trace_id`、`service_name`、`timestamp` 字段：

```cpp
// C++ 服务（spdlog）
spdlog::info(R"({{"trace_id":"{}","service":"gateway","event":"request_received","user_id":"{}"}})",
             trace_id, user_id);
```

```python
# Python 服务（logging）
logger.info(json.dumps({
    "trace_id": trace_id,
    "service": "memory",
    "event": "compression_started",
    "user_id": user_id,
}))
```

日志流向：`stdout → Docker logs → journald`（或通过 Fluentd 聚合到 ELK）

---

## 7. 实施路线图

### 7.1 六个月时间线

```
Month 1–2: Phase 1 基础架构
  ├── C++ 协议层（HTTP/WebSocket/Session 管理）
  ├── 用户认证授权（JWT + OAuth2 + RBAC）
  ├── C++ 人格服务基础版（情绪状态机 + Prompt 构建）
  └── C++ 推理服务基础版（ONNX BERT + 人脸情绪）

Month 3–4: Phase 2 功能完善
  ├── 计费系统（使用量统计 + 配额管理）
  ├── 管理后台 API（用户管理 + 数据统计）
  ├── Python 视觉服务（小橘视觉链路 → gRPC 服务）
  └── Python 记忆服务（小橘记忆系统 + SQLite 持久化）

Month 5–6: Phase 3 交付准备
  ├── Python RAG 服务（LangChain + Milvus/Qdrant）
  ├── 前端（Vue 3 管理后台 + 学生端）
  ├── 压力测试（1,000 并发，p95 < 3s）
  └── 部署脚本、文档、运维手册
```

---

### 7.2 里程碑验收标准

| 里程碑 | 时间节点 | 验收标准 |
|--------|---------|---------|
| **M1：基础架构可用** | Month 2 末 | 单用户完整对话链路可用；gRPC 健康检查全部通过；人格 + 推理服务基础功能正常 |
| **M2：多用户可用** | Month 4 末 | 多租户并发对话；认证授权通过安全审查；故障隔离和降级策略验证完毕 |
| **M3：性能达标** | Month 5 末 | 1,000 并发用户，p95 延迟 < 3s；压力测试无内存泄漏；视觉 + 记忆服务集成完毕 |
| **M4：交付就绪** | Month 6 末 | 文档齐全（API 文档 + 运维手册）；部署脚本一键可用；E2E 测试全部通过 |

---

### 7.3 风险与缓解措施

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| **C++ 协议层开发超期** | 中 | 高 | Phase 1 提前做 Boost.Beast 技术预研；复杂度超预期时缩减流媒体协议范围 |
| **gRPC 跨语言联调困难** | 中 | 中 | 使用 `grpc_health_probe` 做服务隔离测试；完善 JSON trace 日志 |
| **AI 辅助效率不达预期** | 中 | 中 | 保守估算已按 1.8x 折扣计入；方案 3 排期含 30% 缓冲 |
| **多租户隔离 / 计费逻辑复杂** | 中 | 中 | 参考开源实现（Casbin RBAC）；计费规则引擎使用配置驱动，不硬编码 |
| **性能不达标** | 低 | 高 | M3 前置压力测试；预留 1 个月优化窗口 |
| **团队成员离开** | 低 | 高 | 所有模块写设计文档；AI 辅助可降低新人上手成本 |
| **甲方需求变更** | 高 | 中 | Python 服务层保持灵活性；接口变更优先改 proto 定义，隔离影响范围 |

---

## 8. 附录

### 8.1 关键技术选型

| 类别 | 技术选型 | 选型理由 |
|------|---------|---------|
| **C++ HTTP/WebSocket** | Boost.Beast | 成熟稳定，与 Boost.Asio 深度集成，性能优秀 |
| **C++ gRPC** | gRPC C++ | 官方支持，类型安全，生态完善 |
| **C++ JSON** | nlohmann/json | Header-only，API 直观，无额外依赖 |
| **C++ 推理（BERT/情绪）** | ONNX Runtime | 跨平台，支持 CUDA/DirectML，推理稳定 |
| **LLM 主推理** | vLLM | 高吞吐在线 serving，支持连续批处理、PagedAttention、OpenAI-compatible server |
| **C++ 备用推理 / VLM** | llama.cpp | 原生 C/C++，低显存和边缘设备友好，支持量化，可作为 vLLM OOM/过载时的备用路径 |
| **Python gRPC** | grpcio | 官方支持，与 proto 文件自动生成代码 |
| **Python 持久化** | SQLite / RocksDB | SQLite 轻量易部署；RocksDB 高吞吐写入 |
| **Python RAG** | LangChain / LlamaIndex | 生态成熟，适合冷路径知识库构建和快速迭代 |
| **向量检索** | C++ SIMD kernel + Faiss / Qdrant / Milvus | 热路径优先本地轻量向量核；大规模 ANN 可接 Faiss/Qdrant/Milvus |
| **共享缓存** | Redis / Boost.Redis | 存储 payload、metadata、bucket 列表和失效事件；不作为唯一真源 |
| **容器编排** | Docker Compose | 单机部署首选，配置简洁，便于本地开发 |
| **反向代理** | Nginx | 成熟稳定，WebSocket 支持完善，限流配置灵活 |
| **监控** | Prometheus + Grafana | 开源标准，社区生态完善 |
| **前端框架** | Vue 3 + Vite | 响应式 UI，热更新快，TypeScript 支持好 |
| **C++ 日志** | spdlog | 异步日志，性能高，格式化灵活 |
| **认证授权** | JWT + Casbin | JWT 无状态；Casbin 提供灵活的 RBAC 策略 |

---

### 8.2 服务端口规划

| 服务 | 端口 | 协议 | 说明 |
|------|------|------|------|
| Nginx | 443 / 80 | HTTPS / HTTP | 外部入口，HTTP → HTTPS 重定向 |
| Gateway | 8080 | HTTP / WebSocket | 内部服务，不对外暴露 |
| Persona | 50052 | gRPC | 人格服务 |
| Inference | 50051 | gRPC | 推理服务 |
| Memory | 50053 | gRPC | 记忆服务 |
| RAG | 50054 | gRPC | RAG 服务 |
| Visual | 50055 | gRPC | 视觉服务 |
| Prometheus | 9090 | HTTP | 监控数据采集 |
| Grafana | 3000 | HTTP | 监控看板 |

---

### 8.3 参考资料

- [AgentBackendPredict README](../README.md)
- [旧架构问题分析与重构方向](./LEGACY_ARCHITECTURE_ANALYSIS.md)
- [EducationalAgentProject README](../../EducationalAgentProject/README.md)
- [E2E 测试与情绪管线文档](./E2E_TEST_AND_EMOTION_PIPELINE.md)
- [团队实施方案（内部参考）](./TEAM_IMPLEMENTATION_PLAN.md)
- [gRPC C++ Quick Start](https://grpc.io/docs/languages/cpp/quickstart/)
- [Boost.Beast Documentation](https://www.boost.org/doc/libs/release/libs/beast/)
- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/)
- [vLLM](https://github.com/vllm-project/vllm)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Boost.Redis](https://www.boost.org/doc/libs/release/libs/redis/)

---

**文档版本**：v2.1
**创建日期**：2026-05-10
**最后更新**：2026-05-16
**作者**：Orange & Claude
**状态**：待评审
