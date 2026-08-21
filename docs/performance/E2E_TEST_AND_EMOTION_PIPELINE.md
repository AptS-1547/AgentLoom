# VLM 向量缓存端到端测试与情绪感知改造讨论

> 记录 2026-05-09 的端到端测试结果，以及基于测试观察得出的视觉情绪感知模块设计讨论。

## 第一部分：端到端测试结果

### 测试配置

- 服务端：`multimodal_inference_server` 启用 `vlm_cache.vector`（双阈值 high=0.97 / mid=0.93，按 prompt 分桶 flat cosine）
- 模型：Qwen2.5-VL-3B-Instruct，分别测试 Q4_K_M 与 Q8_0
- mmproj：`mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf`
- 客户端：`tools/e2e_visual/run_e2e_visual_test.py` —— 用 MyNeuroLikeSystem 的 `VisualPerceptionPipeline` + 自写 `GrpcVLMAnalyzer` shim 替换原 GLM 云端
- 素材：`test_adaptive.avi`（149MB，5 分钟桌面/人物混合录像），duration-seconds 300
- GPU：RTX 5060 Laptop 8GB

### 过程中发现并修复的三处问题

| 位置 | 症状 | 原因 | 修复 |
|------|------|------|------|
| `llama_runner.cpp::Generate` | 第二次请求起只生成 1 token 即 EOS | 连续 `Generate` 之间 KV cache 累积未清理 | 入口调用 `llama_memory_clear` + `llama_sampler_reset` |
| `llama_runner.cpp::EncodeImageOnly` | 启用向量缓存后，`Generate` 路径被连锁打坏 | mtmd 编码后内部状态残留，污染后续 decode | 退出前同样清理上下文 |
| `multimodal_inference_server.cpp::StoreVectorCacheEntry` | 短输出（1-token 空壳）进入向量缓存后，相似帧全部命中坏结果 | 缓存不区分输出质量 | 加 `kMinTokensForVectorCache = 8` 过滤，短输出不入缓存 |
| `llama_runner.cpp::Generate` | Q8 模型 100% 立即 EOS，Q4 勉强工作 | 裸 prompt 未套 chat template，Q8 严格 | 调用 `llama_chat_apply_template` 包装 user 消息 |

### Q4 vs Q8 对比

同一视频、同一向量缓存配置、关闭精确缓存以测向量路径：

| 指标 | Q4 + 原严格 prompt | Q8 + 放宽 prompt |
|------|:---:|:---:|
| 总事件 | 42 | 42 |
| 命中率 | 64.3% | **83.3%** |
| fresh 次数 | 15 | 7 |
| short_fresh（<8 tokens） | 11 / 15 = 73% | **0 / 7 = 0%** |
| Fresh p50 / p99 | 375 / 2896 ms | 594 / 3996 ms |
| Cache p50 / p99 | 129 / 134 ms | 125 / 137 ms |
| 平均延迟（加权） | 218 ms | **205 ms** |
| 输出风格 | `{"action": "unchanged"}` 模板化 | 自然中文描述，场景/物件具体 |

结论：Q8 在本缓存架构下既更准又更快。fresh 单次变慢 50%，但 short_fresh 归零 → 每次真实生成都能被缓存复用 → 命中率上限被解锁。

### Q8 全部 fresh 生成样本（5 分钟内 7 条独立输出）

| # | tokens | 生成内容 |
|---|--------|------|
| 1 | 35 | 一个穿着黄色衣服的**女孩**坐在房间里，背景是一个昏暗的房间，她戴着耳机看着镜头，表情平静中带着微笑，似乎在说话。 |
| 2 | 17 | 一个穿着灰色衣服的人从下边冒出头来，眼睛直视镜头。 |
| 3 | 14 | 一个戴着耳机的**男人**低着头，眼睛看向画面下方。 |
| 4 | 8 | 一个人戴着耳机，正在喝饮料。 |
| 5 | 21 | **男人**戴着耳环，穿着白色上衣，右手中指上有手链，嘴巴闭着。 |
| 6 | 28 | 一个黑发**男孩**，穿着白色衣服，侧对着镜头，露出洁白的牙齿。背景是室内，有书架和衣物。 |
| 7 | 22 | 一个穿着浅绿色长袖的人站在镜子前，他背对着镜头，只露出脖子和肩膀。 |
| 8 | 25 | 一个房间，右侧有书架，左侧有挂着一件大衣，门后有一个挂钩，里面是蓝色的。 |

> 注：视频中人物为同一人（项目作者本人）。模型在不同帧把同一人识别为"女孩/男人/男孩"，显示出**身份持续性（identity persistence）完全缺失**。

### 质量评估

**可用的部分**
- 环境识别：书架、衣架、房间布局、镜子、桌面等场景元素定位准确
- 服饰与颜色：黄/灰/白/浅绿 基本对得上
- 物件交互：耳机、耳环、镜头、喝饮料等动作被识别
- 空间关系：低头 / 侧对 / 背对 / 眼神方向 基本正确

**明显的弱项**
1. **性别在帧间反复横跳**：#1 "女孩" → #3/#5/#6 "男人/男孩"。Qwen2.5-VL-3B 的视觉塔在小尺寸 + 量化下，对亚洲年轻男性长发或柔和五官误判系统性偏向女性。
2. **幻觉模板**：#6 "露出洁白的牙齿" 是典型训练数据污染——描述人脸时的 common phrase 被填充。
3. **表情空话**：#1 "表情平静中带着微笑似乎在说话"这类描述换任何人脸特写都成立，并非对当前帧的真实判断。
4. **位置 / 颜色细节幻觉**：#5 "右手中指上有手链"、#8 "门后有一个挂钩，里面是蓝色的" 是模型为了凑 token 数生成的伪细节。
5. **跨帧身份无感知**：三个不同的身份描述指向同一人，模型完全不知道。

### 对系统设计的启示

意外验证了向量缓存的合理性：

- 模型对人脸的描述本身就不稳定——重新推理未必更准。"视觉相似就复用旧描述"的语义损失，其实小于"信任每次新推理"带来的抖动
- 真正不能缓存的"即时信息"是**动作变化**（站起 / 喝水 / 转身），而不是身份 / 表情类描述——而 `saliency_hint` 设计正好抓的就是动作变化
- 架构和模型能力是匹配的：VLM 做叙事，缓存兜底复用，saliency 管即时信息拦截

---

## 第二部分：改造方向讨论 —— 专用情绪感知模块

### 为什么 3B → 7B 不是正解

Qwen2.5-VL 系列 3B / 7B / 72B **共享同一个视觉塔**（SigLIP-400M 量级 ViT）。换更大模型堆的是文本解码器的能力，视觉判别层不变。结果是：

- 描述更流畅 → **幻觉也更流畅**
- identity / 性别 / 微表情这些依赖视觉判别力的任务，该错还是错
- 只是输出看起来更专业，不代表看得更准

对 MyNeuroLikeSystem 这种需要情绪 / 身份稳定信号的项目，加模型参数并不划算。

### 为什么 mmproj LoRA 不是首选

mmproj 是视觉塔到 LLM embedding 空间的投影层（2-layer MLP 量级，几 M 参数）。LoRA 微调可行性：

**优点**：参数少，单卡一晚训完；不破坏 ViT 原能力；按需加载切换。

**代价与风险**：
- 数据收集是大头——公开数据集（CelebA / FFHQ）训出来泛化不到真实用户；自建样本少就过拟合
- 降低场景敏感度是真实风险，需要混合训练 + KL 约束，工程复杂度上升
- **最关键**：LoRA 改善的是单帧描述，但跨帧 identity persistence 和情绪稳定性依然无解——那是模型架构层面的局限，不是权重调整能改的

ROI 一般。是不错的 R&D 实验，但不是让比赛端跑得好的最短路径。

### 推荐的架构：专用情绪识别旁路

把情绪 / 身份从 VLM 这个通用工具中剥离到专用模块。

```
帧输入
  │
  ├─→ YOLOv8-face / RetinaFace（定位+对齐）
  │       │
  │       └─→ 112×112 对齐人脸
  │               │
  │               └─→ ArcFace backbone (frozen)
  │                       │
  │                       └─→ LoRA 最后 2-3 个 block
  │                               │
  │                               └─→ 情绪分类头
  │                                       │
  │                                       └─→ {"emotion": "focused",
  │                                            "valence": 0.3,
  │                                            "arousal": 0.5,
  │                                            "confidence": 0.87}
  │
  └─→ VLM（只描述动作/场景/物件）
          ↑ prompt 里注入已识别的情绪作为上下文
```

### 为什么要拆开"检测器"和"embedding 器"

工程上是两个不同模型：

- **检测器**（YOLOv8-face / RetinaFace）：输出 bbox + 5 关键点，backbone 为检测优化，对整脸语义 embedding 不是最强
- **embedding 器**（ArcFace / MobileFaceNet）：输出固定维度向量（通常 512d），训练目标是"同人近、异人远"，是真正的人脸语义空间

标准做法：检测器定位 + 对齐 → embedding 器提特征 → LoRA + 情绪头。不要把两步合一。

### 各模块的合理分工

| 模块 | 负责 | 为什么它适合 |
|------|------|------|
| YOLOv8-face / RetinaFace | 人脸定位与对齐 | 专门为小目标 + 多尺度人脸训练 |
| ArcFace + LoRA | 情绪辨识 / 身份 embedding | 专用特征空间，数据密集任务 |
| VLM | 场景 + 动作 + 物件 | 擅长叙事，不擅长细粒度判别 |

### 与现有向量缓存的天然联动

情绪模型的输出正好可以喂 `saliency_hint`：

```python
# Python pipeline 内
prev_emo = last_event.emotion_vec  # (valence, arousal)
cur_emo = emotion_model(face_crop)

emotion_delta = l2_distance(prev_emo, cur_emo)
saliency = max(visual_change_score, emotion_delta)

grpc_request.saliency_hint = saliency
```

这样"情绪突变"被正确归类为即时信息，即使向量缓存视觉相似也不会命中旧结果。向量缓存 + 情绪突变检测形成互补——之前担心的"阈值误判即时信息"从这里得到正解。

### 对 VLM prompt 的相应改造

不要把情绪字段硬塞 prompt 让 VLM 复述。作为上下文让它**避开**这个维度：

```
"画面中的人物情绪已由专用模型识别为：专注、轻度积极。
 请只描述画面中的动作、场景、物件互动，
 不必再分析人物表情、不必猜测性别。
 一两句话，不超过 50 字。"
```

效果：
- VLM 不再重复计算情绪，节省 token 和幻觉空间
- VLM 不再冒险猜性别，identity 维度绕开
- 最终 `VisualAnalysis`：emotion 字段来自确定性模型，raw_text 来自 VLM —— 两路独立信号进入 Agent 记忆

### 需要提前想清楚的风险

1. **遮挡和侧脸**：口罩 / 背对 / 强侧光时检测置信度低。策略：`face_conf < 0.5 → emotion = None`，退化回 VLM 自由描述，不要硬编。
2. **多人场景**：bbox 多于一张时的选择策略必须明确——"最大脸" / "画面中心最近" / "per-face 分别输出"，影响 Agent 行为设计。
3. **数据集偏差**：公开 emotion 数据集（AffectNet / RAF-DB / FER+）偏西方、偏夸张表情。东亚轻微表情（抿嘴 / 略皱眉 / 专注）标注稀疏——**LoRA 微调正好是修这个的手段**，用少量自建样本做第二阶段 fine-tune。
4. **延迟预算**：检测 ~30ms + 对齐 ~5ms + emotion 推理 ~15ms ≈ **50-80ms / 帧**。串在 VLM 前面总 pipeline 仍在目标区间。多人场景需考虑 batch。
5. **评估指标**：不要只看 7 类准确率。**V-A 连续回归 MAE** 或**情绪稳定性**（相邻帧不乱跳）对 Agent 体验更关键——抖动比略偏更难忍受。

### 最小验证路径（建议先做这一步再决定是否训练）

不必一上来就训 LoRA。用一天先验证方向值不值得：

1. **用预训练 EmoNet / py-feat 直接接入**——几行代码拿到 V-A 连续值，把它注入 `VisualPerceptionPipeline` 的 emotion 字段
2. **观察情绪信号 vs VLM 自由描述的分歧**——找出"EmoNet 说专注低唤醒、VLM 说微笑开心"的帧。分歧帧就是 VLM 幻觉多发区
3. **评估对 Agent 输出稳定性的影响**——是否减少了性别跳变和情绪描述抖动
4. 如果方向确实有效且 EmoNet 对东亚面孔精度不够，**再投入 LoRA 微调**补齐短板

这样用一天验证，而不是先训练两周后发现整体 gain 小。

---

## 第三部分：路线图与优先级

测试和讨论后确认的工作排序。**排序逻辑：越靠上的事越影响当前 Agent 输出质量，越靠下的事越影响"某一天能否顺利上线/维护"**。顺序反了就是过度工程。

### 优先级 1：识别能力（当前瓶颈）

模型本身的识别能力是当前端到端体验的真正短板——人脸性别误判、identity 不稳定、表情幻觉。向量缓存和服务端架构已经到位，但缓存的内容本身不够好。

- [ ] **EmoNet / py-feat 预训练模型接入验证**：几行代码拿到 V-A 连续值，注入 `VisualPerceptionPipeline` 的 emotion 字段，观察 1-2 天内是否改善 Agent 输出稳定性
- [ ] **情绪信号 vs VLM 描述分歧分析**：找出"EmoNet 说低唤醒、VLM 说微笑"的帧，验证 VLM 幻觉规律
- [ ] **多人场景策略确定与实现**："最大脸" / "中心最近" / "per-face 分别输出" 三选一
- [ ] **（条件）ArcFace + LoRA 情绪分类头训练**：仅当预训练方案对东亚面孔精度不足时启动

### 优先级 2：主项目集成债务

v0.0.1beta1 的所有新增（统一 proto / VLM / 向量缓存 / saliency_hint）在主项目侧**处于"写好但未接入"状态**。修这个不需要改推理端代码，需要把新接口真正连到 Agent 主回路。

- [ ] 本项目构建时产出 `multimodal_inference.desc`，主项目 `grpc_contract.py` 切到新协议
- [ ] 迁移 `core_engine/small_model.py` 到 `multimodal_inference.proto` 的 `PredictEmotion` / `PredictEmotionBatch`
- [ ] 把 `tools/e2e_visual/run_e2e_visual_test.py` 里的 `GrpcVLMAnalyzer` 提升为主项目 `src/vision/grpc_vlm_analyzer.py` 一等公民
- [ ] 主项目 `AdaptiveFrameSampler.change_score` 接入 `VLMRequest.saliency_hint`

### 优先级 3：向量缓存持久化（下一轮开发计划）

`VectorOptions::persist` 开关已预留但未实现。**优先级低于上面两项**——持久化能提升长期运行命中率，但解决不了"缓存内容本身不够好"的上游问题。合适的启动时机是需要长期部署稳定性验证的阶段。

- [ ] 参考 `vlm_cache` 的 binary record 格式，为 `VectorIndex` 实现落盘 + 启动恢复
- [ ] 持久化 schema 版本号（bump-safe）
- [ ] 磁盘加载时跳过 `vlm_cache` 已淘汰的 cache_key（co-eviction 一致性）
- [ ] `kMinTokensForVectorCache` 挪到 `VectorOptions`，不同场景可配

### 优先级 4：工程性补强（长期）

生产化必须但不紧急。这些问题在"主项目想用推理端但不想编译"时会立刻成为阻塞。

- [ ] Dockerfile + 容器化部署
- [ ] Prometheus metrics exporter（当前 `ServerStats` 只走日志）
- [ ] `vlm_cache` / `vector_cache` 单元测试（当前完全依赖端到端验证，迭代慢）
- [ ] `llama.cpp` 依赖的 `find_package` 化（当前路径硬编码在 CMakeLists）
- [ ] proto 版本号与兼容性策略文档
- [ ] 户外 / 人脸特写等更多场景的向量缓存行为测试

### 已完成（2026-05-09 至 05-10）

- [x] 默认配置切换 Q8 模型（`config/e2e_test.json`）
- [x] Chat template 套装（`llama_chat_apply_template`）
- [x] KV cache 跨请求清理（`Generate` + `EncodeImageOnly` 入出口）
- [x] 短输出过滤（`kMinTokensForVectorCache = 8`）
- [x] Python 端到端测试驱动和 gRPC stub 生成脚手架

