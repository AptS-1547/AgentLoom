# 当前 Runtime 后续路线图

> 日期：2026-06-20  
> 目的：在正式 Server、E2E 测试、文档解析链路、L3 长期记忆、语义缓存基础设施、情绪感知融合层基本完成后，重新整理下一阶段最值得投入的工程方向。

## 1. 当前阶段判断

当前系统已经基本具备旧 MVP 功能的完整 C++ 后端能力，并且已经形成较完整的 E2E Server 与正式 Server 入口。P0 级别的“能跑通主链路并交付”的能力已经基本完成，后续 P0 不再作为主要开发任务，而是作为每轮重要改动后的回归门槛。

已经具备的关键能力：

- 正式 Server 入口已经集成主链路、文档解析链路、长期记忆 L3、守护线程等核心接口。
- E2E Server 已有较完整测试基础，可作为正式 Server 的回归参照。
- Redis、SQLite、语义缓存、文档解析缓存、user registry 等基础设施已经接入。
- BERT gRPC 情绪推理链路可用，支持 ONNX / CUDA / CPU 与 batch 推理。
- 情绪融合层已经从简单 softmax evidence 融合演进为 BERT/keyword/vector 快速融合 + LLM gate 仲裁。
- MAP 标定器已经能评估 mixed5000 与 SMP2020，并输出 LLM gate 阈值扫描。
- Docker 多容器部署方向已经明确，BERT/VLM 推理服务更适合作为独立容器。

当前最重要的阶段性结论：

```text
P0 已完成，后续作为回归门槛；
P1 进入语义缓存主链路接入；
P2 补完情绪层 domain/OOD pre-gate；
模型升级和 VLM 进入后续迭代。
```

## 2. 更新后的优先级

| 优先级 | 方向 | 当前定位 |
|--------|------|----------|
| P0 | 交付态回归验证 | 已基本完成，作为 release gate |
| P1 | 语义缓存主链路接入 | 当前最值得优先实现的功能 |
| P2 | 情绪层 domain/OOD pre-gate | 情绪感知工程闭环的最后关键结构 |
| P3 | BERT/情绪模型升级与数据闭环 | 赛后或下一阶段模型工程 |
| P4 | VLM 多模态实时链路 | 高价值但高风险，需分阶段推进 |

## 3. P0：交付态回归验证

P0 已经不再是主要开发任务，而是保留为回归检查项。每次改动主链路、缓存、情绪感知或部署脚本后，应按需执行。

回归门槛：

- 正式 Server 能启动。
- 主链路 E2E 能过。
- 文档解析链路 E2E 能过。
- Redis 相关测试能过。
- BERT gRPC / emotion server 能跑。
- L3 长期记忆路径正常。
- SQLite user registry 路径正常。
- Docker / 多容器脚本在 Linux target 下可构建或至少具备明确构建路径。
- 配置文件不包含硬编码敏感 key。
- Windows/Linux 条件编译不互相污染。

P0 的核心目标是防止后续功能开发破坏已经具备的交付能力。

## 4. P1：语义缓存主链路接入

语义缓存是当前最值得优先推进的功能。原因是底层能力已经相对完备，接口清晰，文档解析缓存已经证明了缓存价值，且收益对主链路延迟、成本和一致性都比较确定。

已经具备：

- L0 语义搜索算法。
- Redis 语义缓存基础设施。
- `CacheRecord` 扩展。
- embedding/vector 基础设施。
- batch 检索能力。
- 文档解析缓存接入经验。

### 4.1 第一版策略：cached_reference 注入

主链路第一版不建议“命中即直接返回旧答案”。直接返回会带来明显重复感，也可能忽略当前 session、当前问题细节和用户表达差异。

建议第一版采用保守策略：

```text
semantic cache hit
  -> 生成 cached_reference
  -> 注入 LLM prompt / context
  -> LLM 基于当前问题、会话上下文和缓存参考生成最终回复
```

这将语义缓存定位为：

```text
cache-augmented generation
```

而不是：

```text
answer replacement
```

### 4.2 命中分层

建议按相似度和作用方式分层：

| 命中层级 | 策略 |
|----------|------|
| High similarity | 注入缓存答案摘要、结构、关键事实，允许 LLM 复用 |
| Mid similarity | 只注入相关历史答案或关键点，不要求复用表达 |
| Low similarity | 不使用缓存 |

### 4.3 抗重复策略

需要避免用户明显感知到“复制粘贴”：

- 不直接返回原缓存文本。
- 缓存记录维护 `hit_count`。
- 同一 session 连续命中同一 cache record 时降权。
- `hit_count` 高时提示 LLM 换表达或重组结构。
- 当前问题与缓存问题存在关键差异时，只作为参考，不作为答案。
- 置信不足、上下文冲突或用户显式要求新答案时回退完整 LLM。

### 4.4 后续进阶：selective bypass

当 reference 模式稳定后，可以再引入更激进的 bypass：

```text
极高相似度 + 低风险问题 + 非连续重复命中
  -> 允许模板化轻改写或直接复用部分答案
```

但 bypass 应该是第二阶段，不应作为第一版主链路策略。

## 5. P2：情绪层 domain/OOD pre-gate

情绪融合层经过本轮实验后，已经不适合继续投入大量时间调 softmax 权重或堆关键词。唯一还值得补的工程闭环是基于 embedding 域距离的 domain/OOD pre-gate。

实验结论详见：

- `docs/EMOTION_FUSION_GATE_EXPERIMENT.md`

关键观察：

| 数据集 | 数据域 | 最优 LLM 接管率 | 结论 |
|--------|--------|----------------:|------|
| mixed5000 | 训练分布/近训练分布 | 0.02% | LLM 基本不应接管 |
| SMP2020 | 外部分布 | 约 42% | LLM 有明显补偿作用 |

这说明 LLM 的主要价值不是常规融合，而是域外样本补偿。因此需要在调用 LLM 前先判断样本是否偏离训练/标定分布。

### 5.1 pre-gate 目标

当前已经有 post-gate：

```text
LLM 调用后，判断是否覆盖快速融合结果。
```

还缺 pre-gate：

```text
调用 LLM 前，判断是否值得调用。
```

pre-gate 的目标：

- 降低训练分布内不必要的 LLM 调用。
- 保留 SMP2020 这类域外样本上的 LLM 收益。
- 降低成本、延迟和 LLM 噪声。
- 在域外或冲突样本上降低状态机更新强度。

### 5.2 第一版 domain score

建议先实现轻量版本：

```text
domain_score = max cosine_similarity(text_embedding, in_domain_centroids)
```

或者按标签维护：

```text
domain_score[label] = cosine_similarity(text_embedding, label_domain_centroid[label])
```

第一版可以使用少量原型或 centroid：

- 训练域正常样本 centroid。
- mixed5000 / 弱项标签 centroid。
- 教育问答风格 centroid。
- SMP/社交媒体风格 centroid。
- BERT 弱项标签 centroid。

### 5.3 策略

当 `domain_score` 较高：

```text
信任 BERT/MAP 快速融合层；
收紧 LLM post-gate；
甚至跳过 LLM 调用。
```

当 `domain_score` 较低：

```text
降低 BERT margin 信任；
放宽 LLM fallback；
必要时降低情绪状态机更新强度。
```

### 5.4 分工边界

当前情绪感知层的职责应保持清晰：

```text
上下文靠记忆和状态；
域漂移靠 gate；
主观性靠概率和平滑；
显著性能突破靠模型和数据。
```

上下文缺失不应主要交给 gate 解决，因为 message array、LLM context、短期会话、长期记忆 L3 和情绪状态机已经能承载上下文。gate 的核心职责是跨域泛化风险识别。

## 6. P3：BERT/情绪模型升级与数据闭环

这不是当前最紧急的工程交付任务，但它是情绪感知层后续显著提升的主要来源。

当前实验表明：

- 域内数据上 BERT/感知层已经接近先前统计诊断提示的经验上限。
- SMP2020 上存在明显域漂移。
- 正则/vector/MAP 只能小幅修正域外表现。
- LLM gate 能补偿域外样本，但成本和调用率较高。

后续模型方向：

- 使用更专业的中文情绪识别模型。
- 继续微调当前 `hfl/chinese-roberta-wwm-ext` 派生模型。
- 使用教育对话目标域数据。
- 构建真实 Agent 高冲突样本回流集。
- 尝试多标签情绪建模。
- 做更强模型到小 ONNX 模型的蒸馏。
- 重新标定 per-label reliability。

这一阶段需要数据、训练和重新评估，不建议在当前主链路交付阶段打断工程收敛。

## 7. P4：VLM 多模态实时链路

VLM 仍然是高价值方向，但当前完成度最低、工程风险最高。短期不建议把它作为拉升情绪准确率的主要手段。

已知问题：

- GStreamer/WebRTC 服务尚未完成。
- Qwen VL 3B 在人脸、身份、性别、客体持续性上存在明显幻觉。
- 场景识别相对可用，但表情/身份类描述不稳定。
- 多模态结果如何影响情绪状态机仍需要谨慎定义。

建议推进顺序：

```text
1. 完成 GStreamer/WebRTC 实时视频输入服务。
2. 做帧采样与 saliency 检测。
3. 接入 VLM 缓存与动作变化检测。
4. 做 identity persistence / 跨帧一致性。
5. 最后再考虑接入情绪状态机。
```

VLM 第一阶段定位：

```text
场景/动作辅助感知，而不是直接决定用户情绪。
```

## 8. 当前建议执行顺序

短期建议：

```text
1. 语义缓存主链路保守接入。
2. 情绪 domain/OOD pre-gate。
3. 赛前或交付前做一次全量 E2E 回归。
4. 源码交付清单整理。
```

赛后或下一阶段：

```text
1. BERT/情绪模型升级。
2. 真实 Agent 数据闭环。
3. 多模态实时链路。
4. 语义缓存从 reference 模式升级到 selective bypass。
```

不建议继续作为主方向投入：

```text
1. 继续让 LLM 参与普通 softmax logits 平均。
2. 无限制增加关键词试图全局超过 BERT。
3. 在没有 domain gate 的情况下扩大 LLM fallback 调用。
4. 让 VLM 直接接管情绪判断。
```

## 9. 与旧 Roadmap 的关系

`NEXT_RUNTIME_ROADMAP.md` 记录了从 tokenizer、embedding、vector storage、semantic cache、document pipeline 到 runtime foundation 的历史路线图。该文档仍保留历史价值。

本文档是 2026-06-20 之后的当前执行路线图，用于指导后续最值得投入的工程任务。
