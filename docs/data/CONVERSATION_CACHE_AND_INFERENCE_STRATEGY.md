# 对话热路径缓存与推理引擎路由策略

> 面向团队内部开发人员的设计记录  
> 日期：2026-05-16  
> 状态：方案讨论稿

## 1. 背景

教育智能体的商业化场景不能依赖在线大模型推理直接承载全部请求。即便使用多张高端 GPU，32B 级本地 LLM 在长上下文、多 persona、流式输出和高并发条件下也很难稳定支撑数百 QPS。推理成本、首 token 延迟、显存压力、KV cache 调度和 batch 碎片化都会成为瓶颈。

因此，对话热路径需要建立“缓存优先”的加速层。该层不应只是推理失败后的 fallback，而应成为正式的请求路由阶段：优先复用离线生成的高质量回答、标准知识库和可安全复用的上下文，再将剩余复杂请求交给在线 LLM。

## 2. 结论

采用混合方案：

1. 使用中文教育对话数据构建离线语义缓存基底。
2. 建立课程知识库 RAG，保证教材、课标、经典问题和标准讲解的可靠性。
3. 保留原有多级记忆系统，作为长上下文基础设施和个体化兜底。
4. 使用轻量向量相似度核心和本地热索引完成高频相似度检索。
5. 使用 Redis 作为跨实例共享缓存、payload 存储、bucket 列表和失效协调层。
6. 在线 LLM 推理以 vLLM 为主路径，llama.cpp 作为低显存、边缘设备和主路径过载/OOM 时的备用路径。

## 3. 在线请求链路

推荐主链路：

```text
客户端请求
  -> 请求规范化
  -> 意图识别
  -> 上下文依赖判断
  -> Exact Cache
  -> Curated Semantic Cache
  -> Knowledge RAG Cache
  -> Memory Context Layer
  -> vLLM 在线推理
  -> llama.cpp 备用推理
  -> 降级响应
```

实际执行顺序需要由意图路由决定：

| 请求类型 | 优先路径 | 说明 |
|----------|----------|------|
| 完全重复请求 | Exact Cache | 依赖 prompt/context/model/persona hash |
| 通用课堂互动 | Curated Semantic Cache | 适合离线生成和审核 |
| 标准知识问题 | Knowledge RAG Cache | 课本、课标、经典题优先 |
| 强上下文依赖问题 | Memory Context Layer -> RAG / LLM | 禁止直接命中全局语义缓存 |
| 个体化问题 | Memory Context Layer -> LLM | 仅可使用 user/session scope 缓存 |
| 复杂综合问题 | RAG + Memory -> vLLM | 缓存不足时进入在线生成 |

强上下文依赖词包括但不限于：

- 这个
- 那个
- 刚才
- 上面
- 下面
- 这一步
- 这里
- 图中
- 题目里
- 为什么错

出现这些表达时，默认禁止直接命中全局语义缓存，除非当前 session scope 和缓存 entry 的 context hash 一致。

## 4. 缓存分层

### 4.1 Exact Cache

Exact Cache 用于完全相同请求的快速复用。

命中条件：

- prompt hash 一致
- model fingerprint 一致
- persona version 一致
- system prompt version 一致
- memory scope hash 一致
- policy version 一致

适用场景：

- 重试请求
- 前端重复提交
- 固定 prompt 的结构化任务
- session 内完全重复问题

### 4.2 Curated Semantic Cache

Curated Semantic Cache 是离线构建的语义缓存基底。它覆盖高频、稳定、低上下文依赖的教育对话。

数据来源：

- 中文教育对话数据
- 课堂互动语料
- 常见学习困惑
- 通用教学话术
- 高质量模型离线生成结果

构建流程：

```text
中文教育对话数据筛选
  -> 清洗
  -> 去重
  -> 聚类
  -> 意图分类
  -> DeepSeek / GLM / Qwen 等强中文模型生成
  -> 多模型交叉评审
  -> 规则过滤
  -> 质量打分
  -> embedding 编码和归一化
  -> 写入缓存基底
```

缓存 entry 需要同时保存最终回答和结构化语义信息：

```json
{
  "entry_id": "...",
  "canonical_question": "老师，我没听懂这个知识点怎么办？",
  "question_variants": [
    "我没听懂怎么办",
    "这个地方我还是不明白",
    "能不能再讲一遍"
  ],
  "intent": "learning_confusion",
  "subject": "general",
  "grade": "unknown",
  "answer_type": "encouragement_and_guidance",
  "answer_core": "先安抚学生，再建议从定义、例子和步骤三个角度重新解释。",
  "response_text": "没关系，我们可以换个角度再看一遍...",
  "persona_scope": "neutral",
  "context_requirements": {
    "requires_current_problem": false,
    "requires_memory": false,
    "requires_rag": false
  },
  "embedding_model_fingerprint": "...",
  "model_fingerprint": "...",
  "prompt_version": "...",
  "policy_version": "...",
  "quality_score": 0.93,
  "created_at_ms": 0,
  "ttl_seconds": 3600
}
```

缓存不应只存 persona 化后的最终回答。建议保留 `answer_core`，并在线进行轻量 persona 渲染。这样可以避免为每个 persona 重复构建一套完整缓存。

### 4.3 Knowledge RAG Cache

Knowledge RAG Cache 负责标准知识问题。

来源：

- 教材
- 课程标准
- 教学大纲
- 经典例题
- 常见误区
- 题型模板
- 教师讲义

该层不应只是 raw chunk 检索。建议将知识沉淀为结构化知识卡：

```json
{
  "knowledge_id": "physics_middle_inertia",
  "subject": "physics",
  "grade": "middle_school",
  "title": "惯性",
  "definition": "...",
  "examples": [
    "公交车刹车时身体前倾"
  ],
  "misconceptions": [
    "惯性不是力",
    "质量越大惯性越大"
  ],
  "teaching_script": "...",
  "classic_questions": [],
  "citations": [
    {
      "source": "教材或课标名称",
      "chapter": "...",
      "chunk_id": "..."
    }
  ],
  "corpus_version": "..."
}
```

知识型回答优先从知识卡和证据片段生成。在线 LLM 只在以下情况介入：

- 多知识点综合
- 用户问题表述复杂
- 检索结果冲突
- 当前记忆上下文需要个体化解释
- 需要自然语言重写但缓存中没有合适模板

### 4.4 Memory Context Layer

多级记忆继续保留，职责不是替代语义缓存，而是提供上下文安全和个体化能力。

用途：

1. 判断缓存是否安全命中。
2. 提供个体化渲染上下文。
3. 作为 LLM fallback 的长上下文输入。
4. 为 session-bound cache 提供 memory scope。

任何包含用户长期记忆、私人信息或当前会话状态的回答，不允许进入全局缓存。最多写入 user scope 或 session scope。

## 5. 向量相似度核心

轻量向量相似度核心用于语义缓存和本地热索引。它与 Faiss 不是互斥关系：

- 小规模/中规模热 bucket 使用本地 SIMD kernel。
- 大规模 ANN 检索可以接 Faiss、Qdrant 或 Milvus。
- Redis 不承担全量向量相似度计算。

第一阶段建议实现：

- `NormalizeInPlace`
- `DotDynamic`
- `DotFixed<384>`
- `DotFixed<768>`
- `TopKNormalizedDot`
- `ThresholdScan`

策略：

- embedding 插入时归一化一次。
- query 归一化一次。
- 相似度比较只做 dot product。
- 常见维度 384 / 768 走编译期固定维度路径。
- 其他维度走 dynamic fallback。

后续优化：

- SSE2 baseline
- AVX2/FMA 可选路径
- runtime CPU feature dispatch
- 32/64 字节对齐的连续矩阵存储
- 批量 query

数据布局优先级高于单个指令集优化。若 embedding 仍以大量独立 `std::vector<float>` 存储，cache miss 和间接寻址会抵消部分 SIMD 收益。

## 6. Redis 的角色

Redis 用于共享缓存，不作为唯一真源。

职责：

- response payload 存储
- metadata 存储
- embedding blob 可选存储
- bucket entry id 列表
- TTL
- 热点统计
- 多实例缓存失效协调

推荐访问模式：

```text
LocalHotVectorIndex
  -> 本地相似度 Top-K
  -> Redis 读取 top hit payload
  -> 二次校验
  -> 返回
```

不推荐每次请求都从 Redis 拉取大量 embedding 后再计算相似度。该模式会将网络 IO 放入热路径，降低延迟稳定性。

## 7. 推理引擎路由

推理层采用双后端：

| 后端 | 角色 | 使用场景 |
|------|------|----------|
| vLLM | 主力在线 serving | 高 QPS、连续批处理、流式输出、OpenAI-compatible API |
| llama.cpp | 备用和边缘路径 | vLLM 过载/OOM、低显存部署、量化小模型、原生 C++ 集成 |

降级顺序：

```text
缓存 / RAG 命中
  -> vLLM 主路径
  -> 降低 max_tokens / 上下文 / 并发预算后重试
  -> llama.cpp 备用模型
  -> 明确降级响应
```

llama.cpp 备用路径应优先使用更小模型、更低精度量化和更短上下文。若使用与 vLLM 相同尺寸、相同上下文和相同输出预算的模型，备用路径不一定能解决显存问题。

## 8. 缓存写入策略

在线 LLM 输出不能无条件写回缓存。建议设置 admission policy。

写入条件：

- 请求成功
- 无 tool error
- 无安全策略异常
- 不包含用户私人记忆或仅写入受限 scope
- temperature 和随机性参数可控
- response token 数在合理范围
- intent / subject / answer_type 可归类
- 质量评分达到阈值
- policy version 和 prompt version 已记录

建议记录：

- `cache_admission_score`
- `source_model`
- `review_status`
- `hit_count`
- `last_hit_at`
- `wrong_hit_count`

## 9. 监控指标

至少需要监控：

- exact cache hit rate
- semantic cache hit rate
- RAG cache hit rate
- cache miss reason
- wrong hit count
- LLM saved request count
- LLM saved token estimate
- vLLM request count
- llama.cpp fallback request count
- vLLM OOM / overload count
- Redis latency
- local hot index size
- local hot index refresh count
- embedding model version distribution
- cache entry quality score distribution

## 10. 实施顺序

建议路线：

1. 实现 `vector_similarity` 基础核。
2. 将当前 `VectorIndex` 改为使用统一 similarity kernel。
3. 实现 `LocalHotVectorIndex` 和内存版 `SemanticCacheStore`。
4. 定义 `SemanticCacheEntry`、`SemanticCacheKey` 和 `CacheAdmissionPolicy`。
5. 构建第一批离线 Curated Semantic Cache。
6. 设计 Knowledge Card schema 和最小 RAG cache。
7. 接入 Redis payload / metadata / bucket 管理。
8. 实现 vLLM 主路径和 llama.cpp fallback 的统一路由接口。
9. 增加 cache hit、wrong hit、LLM saved request 等观测指标。

## 11. 当前约束

- 现有 C++ 推理端已基于 llama.cpp 构建，适合保留为 VLM 和 fallback runtime。
- 高 QPS 对话主路径不应长期依赖 llama.cpp 单独承担。
- vLLM 建议作为外部 OpenAI-compatible 服务接入，C++ 网关只做路由和状态治理。
- 缓存/RAG/记忆层是降低在线推理成本的关键，优先级高于单纯更换推理引擎。
- 初期缓存阈值应保守，宁可 miss，也不能产生高相似误命中。
