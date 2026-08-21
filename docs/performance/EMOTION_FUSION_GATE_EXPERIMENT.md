# 情绪融合层 MAP 标定与 LLM Gate 实验总结

> 记录 2026-06-20 对情绪感知融合层的实验、结论与后续工程策略。本文档重点解释为什么将 LLM 从 softmax logits 融合中拆出，并改为置信度门控式 fallback。

## 背景

当前情绪感知层由四类证据组成：

| 证据源 | 作用 | 特点 |
|--------|------|------|
| BERT 情绪分类器 | 主分类器 | 低延迟、稳定、可批处理，但存在域漂移 |
| 正则关键词 | 显性情绪证据 | 极快、可解释，但覆盖有限 |
| 向量临近原型 | 语义近邻弱证据 | 可复用 embedding provider，低边际成本，但依赖原型质量 |
| LLM fallback | 复杂/低置信样本仲裁 | 能补偿域外样本，但成本、延迟和噪声较高 |

之前的设计曾将 LLM 输出作为 `llm_score` 加入 fusion logits，然后统一做 softmax。这种做法在工程上简单，但实验后发现它不符合 LLM 的最佳角色：LLM 更适合做低置信或域外样本的仲裁器，而不是对每个样本都作为普通证据参与概率平均。

因此当前实现调整为：

```text
BERT + keyword/vector -> 快速融合层
低置信或低 margin -> 触发 LLM fallback
LLM gate -> 判断是否允许 LLM 覆盖最终标签
MAP calibrator -> 标定快速融合层参数，并扫描最优 LLM gate 阈值
```

## 当前实现

核心代码位置：

| 文件 | 作用 |
|------|------|
| `src/service/persona/emotion_fusion_analyzer.h` | fusion 参数与 gate 参数定义 |
| `src/service/persona/emotion_fusion_analyzer.cpp` | 运行时 fusion 与 LLM gate 应用 |
| `tools/emotion_fusion_map_calibrator.cpp` | MAP 标定、BERT warmup、LLM gate scan |
| `tools/persona_gateway_e2e_server.cpp` | E2E server 读取 fusion/gate 配置 |
| `src/server/main/agent_gateway_server.cpp` | 正式 server 读取 fusion/gate 配置 |

### 运行时策略

当前 `BuildLogitsForCalibration()` 不再将 `llm_score` 加入普通 logits：

```text
logit[label] =
    head_bias
  + bert_signal_weight    * bert_weight * bert_prob[label] * reliability[label]
  + keyword_signal_weight * keyword_score[label]
  + vector_signal_weight  * vector_score[label]
  + margin_signal_weight  * margin_bonus[label]
```

LLM fallback 触发后，进入独立 gate：

```text
accept_llm =
    llm_confidence >= llm_gate_confidence
 && llm_confidence - fused_prob[llm_label] >= llm_gate_min_delta
```

如果 gate 通过，则 LLM 标签覆盖快速融合层输出；否则保留快速融合层结果。

### 校准器策略

`emotion_fusion_map_calibrator` 现在支持：

- BERT gRPC batch 推理；
- vector evidence batch 收集；
- LLM fallback 并发采集；
- MAP/Adam 标定快速融合层参数；
- LLM gate 网格扫描；
- BERT CUDA cold start warmup。

BERT warmup 使用单请求，最多 3 次重试。实测 CUDA cold start 下第一轮 BERT 请求可能超过 30 秒 gRPC deadline，但服务端最终会成功完成初始化，因此 warmup 重试是必要的。

## 数据集

### mixed5000

来源：`data/emotion_fusion_calibration_mixed5000.json`

构成：

- 2500 条 BERT 弱项标签困难样本；
- 2500 条原训练分布附近的正常分布样本；
- 共 5000 条。

标签分布：

| 标签 | 数量 |
|------|----:|
| neutral | 798 |
| surprise | 797 |
| disgust | 637 |
| sadness | 637 |
| anger | 621 |
| fear | 597 |
| joy | 387 |
| excitement | 217 |
| curiosity | 175 |
| tenderness | 134 |

该数据集接近当前 BERT 训练分布或其困难样本分布。

### SMP2020 6-label

来源：

- `usual_test_labeled.txt`
- `virus_test_labeled.txt`

转换后文件：

- `data/emotion_fusion_smp2020_test_labeled.json`

标签映射：

| SMP2020 | 系统标签 |
|---------|----------|
| angry | anger |
| sad | sadness |
| happy | joy |
| fear | fear |
| surprise | surprise |
| neutral | neutral |

共 8000 条，仅保留系统中也存在的 6 个标签：

| 标签 | 数量 |
|------|----:|
| joy | 2558 |
| anger | 1971 |
| neutral | 1510 |
| sadness | 1119 |
| surprise | 442 |
| fear | 400 |

SMP2020 是人工标注数据，但其标注形式更接近问卷/众包式人群共识，不是专家型情绪标注。因此它适合作为外部分布泛化测试集，不应被视作唯一真值来源。

## 实验命令

### mixed5000 只评估

```powershell
build\x64-Release-Tests-v145\Release\emotion_fusion_map_calibrator.exe `
  --config tools/persona_gateway_e2e_server.json `
  --dataset data/emotion_fusion_calibration_mixed5000.json `
  --output data/emotion_fusion_eval_mixed5000_full_llm_gate.json `
  --max-samples 5000 `
  --io-workers 24 `
  --batch-size 64 `
  --epochs 0
```

### mixed5000 MAP 训练

```powershell
build\x64-Release-Tests-v145\Release\emotion_fusion_map_calibrator.exe `
  --config tools/persona_gateway_e2e_server.json `
  --dataset data/emotion_fusion_calibration_mixed5000.json `
  --output data/emotion_fusion_calibrated_mixed5000_full_llm_gate_trained.json `
  --max-samples 5000 `
  --io-workers 24 `
  --batch-size 64 `
  --epochs 200
```

### SMP2020 6-label 只评估

```powershell
build\x64-Release-Tests-v145\Release\emotion_fusion_map_calibrator.exe `
  --config tools/persona_gateway_e2e_server.json `
  --dataset data/emotion_fusion_smp2020_test_labeled.json `
  --output data/emotion_fusion_eval_smp2020_6labels_full_llm_gate.json `
  --labels anger,sadness,joy,fear,surprise,neutral `
  --max-samples 8000 `
  --io-workers 24 `
  --batch-size 64 `
  --epochs 0
```

### SMP2020 6-label MAP 训练

```powershell
build\x64-Release-Tests-v145\Release\emotion_fusion_map_calibrator.exe `
  --config tools/persona_gateway_e2e_server.json `
  --dataset data/emotion_fusion_smp2020_test_labeled.json `
  --output data/emotion_fusion_calibrated_smp2020_6labels_full_llm_gate_trained.json `
  --labels anger,sadness,joy,fear,surprise,neutral `
  --max-samples 8000 `
  --io-workers 24 `
  --batch-size 64 `
  --epochs 200
```

## 实验结果总览

### mixed5000

| 模式 | Accuracy | Macro-F1 | Loss | LLM 触发率 | LLM 接管率 |
|------|---------:|---------:|-----:|-----------:|-----------:|
| 未训练快速融合 | 67.64% | 66.72% | 1.3602 | 82.18% | - |
| 未训练 + 最优 LLM gate | 67.62% | 66.70% | 1.3603 | 82.18% | 0.02% |
| MAP 训练快速融合 | 70.22% | 69.34% | 1.0996 | 82.18% | - |
| MAP 训练 + 最优 LLM gate | 70.20% | 69.33% | 1.0997 | 82.18% | 0.02% |

MAP 训练后的参数：

| 参数 | 值 |
|------|---:|
| `bert_signal_weight` | 3.0168 |
| `keyword_signal_weight` | 1.8787 |
| `vector_signal_weight` | 2.1942 |
| `margin_signal_weight` | 1.2965 |
| `head_bias` | 约 0 |

最优 LLM gate：

| 参数 | 值 |
|------|---:|
| `llm_gate_confidence` | 0.90 |
| `llm_gate_min_delta` | 0.10 |
| `llm_accept_rate` | 0.02% |

结论：

```text
在训练分布或近训练分布上，MAP 快速融合有效；
LLM fallback 虽然被大量触发，但几乎不应该接管；
最优 gate 会自然收紧到接近完全关闭 LLM 覆盖。
```

### SMP2020 6-label

| 模式 | Accuracy | Macro-F1 | Loss | LLM 触发率 | LLM 接管率 |
|------|---------:|---------:|-----:|-----------:|-----------:|
| 未训练快速融合 | 49.48% | 47.05% | 1.4558 | 72.01% | - |
| 未训练 + 最优 LLM gate | 56.28% | 54.60% | 1.3139 | 72.01% | 42.75% |
| MAP 训练快速融合 | 50.38% | 47.40% | 1.4347 | 72.01% | - |
| MAP 训练 + 最优 LLM gate | 56.06% | 54.42% | 1.3091 | 72.01% | 42.41% |

MAP 训练后的参数：

| 参数 | 值 |
|------|---:|
| `bert_signal_weight` | 2.6244 |
| `keyword_signal_weight` | 1.9792 |
| `vector_signal_weight` | 2.1953 |
| `margin_signal_weight` | 0.3124 |
| `head_bias` | 约 0 |

最优 LLM gate：

| 参数 | 值 |
|------|---:|
| `llm_gate_confidence` | 0.65 |
| `llm_gate_min_delta` | 0.20 |
| `llm_accept_rate` | 42.41% |

结论：

```text
在 SMP2020 外部分布上，MAP 快速融合只能小幅改善；
LLM gate 带来主要收益；
最优 gate 允许 LLM 接管约 42% 样本，说明域外场景下 LLM 确实能补偿 BERT 缺陷。
```

## 关键对照

### 1. BERT 存在明显域漂移

mixed5000 上，训练后的快速融合可达到：

```text
Accuracy 70.22%
Macro-F1 69.34%
```

SMP2020 6-label 上，训练后的快速融合只有：

```text
Accuracy 50.38%
Macro-F1 47.40%
```

这说明 BERT 在训练分布附近和 SMP2020 外部分布上的表现差异很大。更关键的是，MAP 训练后的 margin 权重变化也支持这个判断：

| 数据集 | `bert_signal_weight` | `margin_signal_weight` |
|--------|---------------------:|-----------------------:|
| mixed5000 | 3.0168 | 1.2965 |
| SMP2020 | 2.6244 | 0.3124 |

在 mixed5000 上，BERT margin 是强信号；在 SMP2020 上，训练器显著降低 margin 权重，说明 BERT 的置信差距不再可靠。

### 2. LLM 的价值集中在域外补偿

mixed5000：

```text
LLM 触发率 82.18%
最优接管率 0.02%
```

SMP2020：

```text
LLM 触发率 72.01%
最优接管率 42.41%
```

这说明：

```text
LLM fallback 被触发，不等于 LLM 应该接管；
训练分布内，LLM 更像噪声源；
外部分布上，LLM 是有效补偿源。
```

因此 LLM 不适合进入所有样本的 softmax logits。它更适合被定位为域外、低置信或冲突样本的仲裁器。

### 3. 正则/vector 是低成本弱证据，不是全局主分类器

MAP 训练后，keyword/vector 权重在两个数据集上都保留中等强度：

| 数据集 | `keyword_signal_weight` | `vector_signal_weight` |
|--------|------------------------:|-----------------------:|
| mixed5000 | 1.8787 | 2.1942 |
| SMP2020 | 1.9792 | 2.1953 |

它们的作用不是直接取代 BERT，而是：

- 提供低成本可解释证据；
- 对 BERT 弱项标签做局部修正；
- 辅助 fallback gate 判断；
- 降低状态机更新时的单点误判风险。

尤其 vector evidence 可以复用服务器已加载的 embedding provider，因此边际成本较低，适合在主链路中常开。但其权重应受门控和标签范围约束。

## 对算法设计的解释

### 为什么不继续使用 LLM logits softmax 平均

softmax logits 融合隐含一个假设：

```text
所有证据源都是同构的、可比较的、应该同时参与每个样本的类别竞争。
```

这个假设对 BERT、keyword、vector 尚可接受，因为它们都是快速、低成本、局部证据。但 LLM 不同：

- 调用成本高；
- 延迟不稳定；
- 输出受 prompt 和标签解释影响；
- 自身存在噪声；
- 在训练分布内可能破坏 BERT 已经正确的判断；
- 在域外样本上才体现明显价值。

因此更合理的设计是：

```text
LLM 不参与常规概率平均；
LLM 只在 gate 允许时覆盖结果；
gate 阈值由数据标定，而不是手写经验决定。
```

### 为什么 MAP 训练仍然有价值

SMP2020 上，MAP 训练只能小幅提升 base fusion：

```text
49.48% -> 50.38%
```

但这并不说明 MAP 无用。它的价值是：

- 标定 BERT、keyword、vector、margin 的相对可信度；
- 暴露 BERT margin 在不同数据域中的可靠性差异；
- 给状态机和线上策略提供更平滑的概率分布；
- 为后续轻量 gate 学习器提供可解释的输入特征。

MAP 更像是融合层的校准器，而不是解决域漂移的万能模型。

## 工程结论

当前最合理的主链路策略是：

```text
1. BERT 作为主分类器。
2. keyword/vector 作为低成本可解释弱证据。
3. MAP 标定快速融合层权重。
4. LLM 从 logits 融合中拆出，作为 gate 控制的仲裁器。
5. 情绪状态机负责吸收时序惯性和降低单句误判影响。
```

线上默认策略建议：

| 场景 | 策略 |
|------|------|
| BERT 高置信且 margin 大 | 不调用或不接管 LLM |
| BERT 低置信 / margin 小 | 允许触发 LLM fallback |
| keyword/vector 与 BERT 强冲突 | 优先触发 LLM 或降低状态机更新强度 |
| 疑似域外样本 | 放宽 LLM gate |
| 训练分布内样本 | 收紧 LLM gate |

## 阶段性上限判断

在当前模型、标签体系和数据条件下，本轮实验已经可以支持一个较强的阶段性判断：

```text
补充基于 embedding 域距离的动态 LLM gate 后，情绪感知层基本达到了现阶段工程侧可解释、低成本优化的上限；后续显著增益主要应来自模型升级、目标域数据采集、标注体系优化和训练策略本身。
```

这里的“上限”不是情绪识别问题本身的理论极限，而是指：

- 当前 BERT 模型不更换；
- 当前标签体系不大改；
- 当前数据质量和标注噪声不发生根本变化；
- 不引入更复杂的深度训练框架；
- 仍以可解释、低延迟、低成本工程优化为主。

在这个约束下，系统已经具备了较完整的工程闭环：

| 模块 | 已完成的工程能力 |
|------|------------------|
| BERT | 低延迟主分类器，支持 ONNX / CUDA / CPU / gRPC batch |
| keyword | 低成本、可解释的显性情绪证据 |
| vector evidence | 复用 embedding provider 的语义近邻弱证据 |
| MAP fusion | 标定 BERT、keyword、vector、margin 的相对可信度 |
| LLM fallback | 作为域外、低置信或冲突样本的语义仲裁器 |
| LLM post-gate | 判断 LLM 是否值得覆盖快速融合结果 |
| emotion state machine | 平滑单句噪声，吸收情绪主观性和时序惯性 |
| memory/context | 通过 message array、短期会话、长期记忆补充上下文 |

还缺的关键拼图是 `embedding domain/OOD pre-gate`：

```text
domain_score = max cosine_similarity(text_embedding, in_domain_centroids)
```

或按标签维护域内中心：

```text
domain_score[label] = cosine_similarity(text_embedding, label_domain_centroid[label])
```

当 `domain_score` 较高时：

```text
信任 BERT/MAP 快速融合层；
收紧 LLM post-gate；
甚至跳过 LLM 调用。
```

当 `domain_score` 较低时：

```text
降低 BERT 置信解释权重；
放宽 LLM fallback；
必要时降低状态机更新强度。
```

这正好对应本轮实验观察：

| 数据集 | 数据域 | 最优 LLM 接管率 | 解释 |
|--------|--------|----------------:|------|
| mixed5000 | 训练分布/近训练分布 | 0.02% | BERT/MAP 已足够可靠，LLM 基本不应接管 |
| SMP2020 | 外部分布 | 约 42% | BERT 存在域漂移，LLM 有明显补偿作用 |

因此，embedding domain gate 是把实验结论转化为在线策略的关键结构件。它完成后，情绪感知层的工程优化重点就不再是继续堆规则、继续调平均权重或让 LLM 更频繁参与，而应转向：

- 更强或更专业的情绪识别模型；
- 目标域教育对话数据；
- 更一致的人工标注规范；
- 多标签或连续情绪建模；
- 线上困难样本回流与再训练。

更简洁地说：

```text
上下文靠记忆和状态；
域漂移靠 gate；
主观性靠概率和平滑；
显著性能突破靠模型和数据。
```

## 后续优化方向

### 1. 学习域外/低可信 gate

当前 gate scan 只扫描：

```text
llm_confidence
llm_confidence - fused_prob[llm_label]
```

后续可以把 gate 扩展为轻量分类器：

```text
accept_llm = sigmoid(
    bias
  + w_top_prob       * fused_top_prob
  + w_margin         * fused_margin
  + w_bert_margin    * bert_margin
  + w_kw_conflict    * keyword_conflict
  + w_vec_conflict   * vector_conflict
  + w_llm_confidence * llm_confidence
  + w_domain_shift   * domain_shift_score
)
```

训练目标不一定是单纯 accuracy，可以加入调用成本惩罚：

```text
loss = classification_loss + lambda * llm_accept_rate
```

### 2. 将 LLM 调用率从触发率和接管率拆开优化

当前 mixed5000 和 SMP2020 都存在较高 LLM 触发率：

| 数据集 | LLM 触发率 | 最优接管率 |
|--------|-----------:|-----------:|
| mixed5000 | 82.18% | 0.02% |
| SMP2020 | 72.01% | 42.41% |

这说明“是否调用 LLM”和“是否接管输出”还没有完全解耦。更好的工程策略是两级 gate：

```text
pre_gate：决定是否调用 LLM
post_gate：拿到 LLM 结果后决定是否接管
```

目标是：

- mixed5000 这类分布内样本：pre_gate 就应大量拦截；
- SMP2020 这类域外样本：pre_gate 放宽，但 post_gate 仍控制噪声。

### 3. 优化 BERT 模型本身

当前 BERT 是从 `hfl/chinese-roberta-wwm-ext` 微调而来，原始设计更贴近小橘/弹幕/陪伴式语料。实验表明它在 SMP2020 上存在明显域漂移。

可选优化方向：

- 使用更专业的中文情绪识别预训练模型；
- 使用 SMP2020 或教育对话数据继续微调；
- 采用多标签情绪建模而非单标签分类；
- 对 BERT 弱项标签做困难样本增强；
- 后续考虑蒸馏更强模型到小型 ONNX 模型。

### 4. 将评估指标从单句 top-1 扩展到状态机轨迹

情绪感知层最终服务于：

- 情绪状态机；
- Persona 响应策略；
- 长期记忆；
- 用户体验连续性。

因此后续评估不应只看 accuracy/Macro-F1，还应关注：

- 状态机 V-A 轨迹是否稳定；
- 单句误判是否会造成状态突变；
- 连续负面表达是否被正确累积；
- 正向恢复是否被合理吸收；
- 低置信样本是否降低状态更新强度。

## 最终结论

本轮实验提供了足够明确的证据：

```text
BERT 是主干，但存在明显域漂移；
keyword/vector 是低成本弱证据，适合常开但不应全局主导；
MAP 标定能改善训练分布内 fusion 表现，并揭示不同证据的可信度；
LLM 在训练分布内几乎不应接管，在域外分布中有明显补偿作用；
LLM 应从 softmax logits 融合中拆出，作为 gate 控制的 fallback 仲裁器。
```

当前最有价值的下一步不是继续堆关键词或让 LLM 参与全量平均，而是构建可学习的两级 gate：

```text
pre_gate：是否值得调用 LLM；
post_gate：LLM 是否值得覆盖快速融合结果。
```

这样才能同时控制准确率、延迟、成本和系统稳定性。
