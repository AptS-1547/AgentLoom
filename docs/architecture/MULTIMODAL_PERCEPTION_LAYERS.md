# 多模态感知分层架构设计

> **文档版本**: v1.1  
> **创建日期**: 2026-07-06  
> **实现核对**: 2026-07-10  
> **作者**: Orange & Claude & Codex  
> **状态**: 部分实现，L1 与生产 VLM 接线待完成

## 0. 文档目的

本文档定义 AgentLoom 项目中多模态视觉感知的分层架构,明确 ViT 专业模型与 VLM 通用模型的职责边界、协同策略和触发条件,以及各层的性能优化路径。

核心原则:**分层感知、各司其职、按需调用、成本可控**。

### 0.1 当前实现快照

| 能力 | 当前状态 | 代码证据/缺口 |
|------|----------|---------------|
| WebRTC signaling/session/media | 组件与 smoke 可用 | `src/media/webrtc_*` 已实现；仅 `tools/webrtc_signaling_smoke_server.cpp` 挂载正式路径，`agent_gateway_server` 尚未接线 |
| OpenCV 动态抽帧 | 已实现 | `opencv_frame_sampler.*` 与 `tests/media/*` 已覆盖主要算法路径 |
| VisionEvent 聚合、去重、Skill 注入 | 已实现基础版 | `vision_runtime_interfaces.*`、`skill_vision_event_sink.*`、`PersonaRuntime` 支持 `vision.observe` 受控注入 |
| L1 ViT 人脸/情绪/身份分析 | 未实现 | `vit_model` 仅有配置；`DetectSaliency` 返回固定零值；无 `IFaceAnalyzer`、`FaceAnalysisResult` 或 `face_identities` repository |
| L2 VLM 推理服务 | 核心推理与 gRPC 传帧已实现 | VLMRequest.image_data 通过 gRPC bytes 接收 JPEG/PNG，质量测试、内存压测和视觉 E2E 已调用 GenerateVLMSync；结果缓存、runner pool、strict Prompt KV Cache，以及统一 RPC 异常/日志/trace/统计边界已接入 |
| Media -> VLM adapter | 未实现 | `IVlmVisionClient` 只有接口；pipeline 未编码关键帧、未调用 IO pool/VLM client |
| 多模态情绪融合 | 未实现 | 当前 `FusedEmotionAnalyzer` 仍是 BERT/keyword/vector/LLM 文本融合；视觉 observation 只进入 prompt，不直接更新情绪状态机 |
| L3 多帧叙事 | 设计阶段 | 无多帧请求协议、调度器或测试 |

本文后续伪代码均表示目标接口或策略，除非明确标记“已实现”，不应视为当前仓库 API。

---

## 1. 架构概览

多模态感知系统分为三层,从快速结构化感知到开放场景理解,逐层递进:

```
视频帧输入
  ↓
人脸检测/裁剪(目标能力，当前未实现)
  ↓
┌─────────────────────────────────────────────────┐
│ L1: 快速结构化感知(ViT 人脸模型，待实现)       │
│   - 延迟: ~30-50ms/帧                            │
│   - 频率: 每帧(或抽帧后的关键帧)                │
│   - 输出: 情绪/性别/身份/强度(结构化)          │
│   - 用途: 情绪状态机、身份追踪、属性一致性      │
└─────────────────────────────────────────────────┘
  ↓ (置信度 < 阈值 / 冲突 / 复杂场景)
┌─────────────────────────────────────────────────┐
│ L2: 场景描述与细节推理(VLM 单帧，部分实现)     │
│   - 延迟: ~500-800ms/次                         │
│   - 频率: 按需触发(5-30% 帧率)                  │
│   - 输出: 自然语言场景描述                      │
│   - 用途: LLM context 注入、冲突解释、细节推理  │
└─────────────────────────────────────────────────┘
  ↓ (检测到动作变化 / 跨帧理解需求)
┌─────────────────────────────────────────────────┐
│ L3: 跨帧叙事(VLM 多帧，设计阶段)               │
│   - 延迟: ~1.5-3s/次                            │
│   - 频率: 罕见触发(<5% 场景)                    │
│   - 输出: 动作序列描述                          │
│   - 用途: 行为理解、动作模式识别                │
└─────────────────────────────────────────────────┘
  ↓
Persona Runtime / 后续融合层
  当前: 文本融合与 vision.observe 上下文注入; 目标: BERT + ViT + VLM 多源融合
```


---

## 2. 各层详细设计

### 2.1 L1: 快速结构化感知(ViT 人脸模型)

**当前实现状态**: 未实现。仓库只有 `vit_model` 配置项和 `DetectSaliency` RPC 形状；`MultimodalService::DetectSaliency` 当前返回固定零值。以下接口、数据表、阈值和性能数字均是目标设计，实施前需要先确定模型许可证、数据合规和评测集。

#### 职责边界

专注于**可枚举、可量化、需要精准**的结构化属性识别:
- 情绪分类(7-8 类:happy/sad/angry/surprise/fear/disgust/neutral/contempt)
- 性别识别(male/female + 置信度)
- 身份识别(embedding 相似度匹配)
- 表情强度(0-1 连续值)
- 年龄估计(可选)
- 面部属性(眼镜/胡须等,可选)

**不负责**:场景理解、空间关系、开放问答、动作叙事。

#### 目标技术实现

**候选模型**:
- **人脸 embedding**: ArcFace 系列或其他可商用、可审计的人脸 embedding 模型
- **情绪识别头**: 在目标域数据上评估或微调的 FER/Affect 模型
- **统一 backbone**: 在许可证、精度和部署成本允许时共享 embedding，再接多个任务头
- **性别/年龄等敏感属性**: 默认不进入第一版；只有产品必要性、用户授权和独立准确率评估成立后再启用

**推理流程**:
```cpp
// 伪码
FaceAnalysisResult AnalyzeFace(const cv::Mat& face_crop) {
    // 1. ViT backbone 提取 embedding
    auto embedding = vit_backbone.Forward(face_crop);  // ~30ms
    
    // 2. 多任务头并行推理(共享 embedding)
    auto emotion = emotion_head.Forward(embedding);     // +5ms
    auto gender = gender_head.Forward(embedding);       // +3ms
    auto intensity = intensity_head.Forward(embedding); // +2ms
    
    // 3. 身份检索(与历史 embedding 对比)
    auto identity = face_db.FindSimilar(embedding, threshold=0.6); // +5ms
    
    return {embedding, emotion, gender, intensity, identity};
}
```

**输出数据结构**:
```cpp
struct FaceAnalysisResult {
    std::vector<float> embedding;       // 512 维,用于身份检索
    std::string emotion;                // "happy"
    float emotion_confidence;           // 0.85
    std::string gender;                 // "male"
    float gender_confidence;            // 0.92
    float intensity;                    // 0.7(表情强度)
    std::optional<std::string> face_id; // 匹配到的历史身份 UUID
    bool is_new_identity;               // 是否新人物
    int64_t timestamp_ms;
};
```

#### 身份追踪与记忆（可选，未实现）

人脸 embedding 属于敏感生物特征。第一版建议优先做短会话内匿名 track id，不默认持久化身份库。只有明确用户授权、删除机制、租户隔离、加密和审计策略完成后，才考虑以下持久化设计。

**`face_identities` 表**(SQLite):
```sql
CREATE TABLE face_identities (
    face_id TEXT PRIMARY KEY,
    user_uuid TEXT NOT NULL,
    tenant_id TEXT NOT NULL,
    embedding BLOB NOT NULL,           -- 512 × 4 bytes
    first_seen_at INTEGER,
    last_seen_at INTEGER,
    occurrence_count INTEGER DEFAULT 1,
    avg_emotion TEXT,
    gender TEXT,
    notes TEXT,                         -- 可选标注(如"朋友小李")
    INDEX idx_user_uuid (user_uuid, tenant_id)
);
```

**检索逻辑**:
```cpp
std::optional<std::string> FindIdentity(const std::vector<float>& embedding,
                                       std::string_view user_uuid) {
    // 1. 查该用户的所有历史 embedding
    auto rows = db.Query("SELECT face_id, embedding FROM face_identities 
                         WHERE user_uuid = ? AND tenant_id = ?", 
                         user_uuid, tenant_id);
    
    // 2. 暴力 cosine 相似度(TODO: Faiss 加速)
    float best_sim = 0.0f;
    std::string best_id;
    for (auto& row : rows) {
        auto hist_emb = DeserializeEmbedding(row.embedding);
        float sim = CosineSimilarity(embedding, hist_emb);
        if (sim > best_sim) {
            best_sim = sim;
            best_id = row.face_id;
        }
    }
    
    // 3. 阈值判断
    if (best_sim >= 0.6) {
        // 更新 last_seen_at, occurrence_count
        db.Execute("UPDATE face_identities SET last_seen_at = ?, 
                    occurrence_count = occurrence_count + 1 
                    WHERE face_id = ?", now_ms, best_id);
        return best_id;
    }
    
    // 4. 新人物
    auto new_id = GenerateUUID();
    db.Execute("INSERT INTO face_identities (...) VALUES (...)", 
               new_id, user_uuid, embedding, ...);
    return std::nullopt;  // 标记为新身份
}
```


#### 目标性能指标（待基准验证）

| 指标 | 目标值 | 备注 |
|------|--------|------|
| 单帧推理延迟 | <50ms | CPU(ResNet50)或 GPU <20ms |
| Embedding 检索 | <5ms | 100 条历史记录暴力搜索;>1000 条需 Faiss |
| 情绪分类准确率 | >85% | FER2013 验证集基线 |
| 性别分类准确率 | >95% | CelebA 验证集基线 |
| 身份识别 FAR | <0.1% | False Accept Rate,LFW 基线 |

---

### 2.2 L2: 场景描述与细节推理(VLM 单帧)

**当前实现状态**: 推理服务核心与 gRPC 传帧协议已实现，实时 media 接线未实现。VLMRequest.image_data 当前通过 gRPC bytes 传输 JPEG/PNG，质量测试、内存压测和视觉 E2E 已实际调用 GenerateVLMSync；推理端现已统一业务 `core::Status`、gRPC status、异常兜底、trace metadata、失败统计与脱敏日志，并有进程内真实 gRPC 测试覆盖主要错误路径。IVlmVisionClient 尚无具体实现，WebRtcMediaPipeline 也尚未把 sampler 命中的帧编码并提交给该协议。

#### 职责边界

专注于**需要自然语言描述、空间理解、细节推理**的开放场景:
- 场景描述("用户在厨房做饭,背景有冰箱和灶台")
- 细节观察("桌上有三本书,最上面是红色的")
- 冲突解释(ViT 判断微笑但文本说悲伤 → VLM:"眼眶泛红,强颜欢笑")
- 开放问答("用户在干什么?""为什么皱眉?")
- 多对象关系("两个人在交谈,左边的人在比划手势")

**不负责**:高精度结构化属性(性别/情绪分类应优先 ViT)。

#### 目标触发策略（待实现）

**触发条件**(满足任一即调用):
```cpp
bool ShouldCallVLM(const FaceAnalysisResult& vit_result,
                   const BertEmotionResult& text_emotion,
                   const FrameContext& ctx) {
    // 1. ViT 低置信度
    if (vit_result.emotion_confidence < 0.7) {
        logger.info("VLM trigger: low ViT confidence {}", 
                   vit_result.emotion_confidence);
        return true;
    }
    
    // 2. 文本与视觉情绪冲突(距离 > 阈值)
    float emotion_dist = EmotionDistance(text_emotion.category, 
                                        vit_result.emotion);
    if (emotion_dist > 0.5) {  // 0=完全一致,1=完全相反
        logger.info("VLM trigger: emotion conflict text={} vit={} dist={}", 
                   text_emotion.category, vit_result.emotion, emotion_dist);
        return true;
    }
    
    // 3. 用户明确提问场景/环境
    if (ctx.query_keywords.contains("在哪") || 
        ctx.query_keywords.contains("周围") ||
        ctx.query_keywords.contains("看到什么")) {
        logger.info("VLM trigger: user asked about scene");
        return true;
    }
    
    // 4. 多人或无清晰人脸
    if (ctx.face_count == 0) {
        logger.info("VLM trigger: no clear face detected");
        return true;
    }
    if (ctx.face_count > 1) {
        logger.info("VLM trigger: multi-person scene");
        return true;
    }
    
    // 5. Rate limit:距上次调用太近 且 场景无显著变化
    auto elapsed = ctx.now_ms - ctx.last_vlm_call_ms;
    if (elapsed < 5000 && ctx.scene_similarity > 0.95) {
        logger.debug("VLM suppressed: too soon + similar scene");
        return false;
    }
    
    return false;  // 默认不调,信任 ViT
}
```

**目标 Prompt 设计**(分场景):
```cpp
std::string BuildVLMPrompt(const VLMTriggerReason& reason,
                          const FaceAnalysisResult& vit_result,
                          const std::string& user_query) {
    switch (reason) {
    case LOW_CONFIDENCE:
        return "请仔细描述画面中人物的表情细节和周围环境";
        
    case EMOTION_CONFLICT:
        return fmt::format(
            "画面中人物表面看起来{},但可能隐藏着其他情绪。"
            "请观察眼神、姿态、周围环境等细节,判断真实情绪状态",
            vit_result.emotion);
        
    case USER_QUERY_SCENE:
        return fmt::format("用户问:{}\n请根据画面回答", user_query);
        
    case MULTI_PERSON:
        return "画面中有多个人,请描述他们各自在做什么,以及相互关系";
        
    default:
        return "请描述画面中的场景和人物状态";
    }
}
```


#### VLM 输出处理

当前通用 VLM RPC 输出仍以自然语言文本和性能/cache metadata 为主。Media 层已经定义 `VisionInferenceResult` 与 `VisionAnalysis`，目标是由 adapter 将 VLM 文本解析为 scene/facts/weak_interpretations/agent_hint，再通过 `SkillVisionEventSink` 记录为 `vision.observe` observation。该 observation 带“不确定观察”语义，只在置信度达标且非 duplicate/rate-limited 时注入 prompt。

**后处理**:
```cpp
struct VLMSceneDescription {
    std::string raw_text;              // VLM 原始输出
    float confidence;                  // 置信度(如有)
    std::vector<std::string> entities; // 提取的实体(可选)
    std::string summary;               // 摘要(可选,用于缓存键)
    int64_t inference_ms;
};

// 注入 LLM context
std::string BuildLLMContext(const VLMSceneDescription& vlm,
                           const FaceAnalysisResult& vit) {
    return fmt::format(
        "视觉观察:\n"
        "- 结构化属性(ViT):情绪={},性别={},强度={:.2f}\n"
        "- 场景描述(VLM):{}\n",
        vit.emotion, vit.gender, vit.intensity, vlm.raw_text);
}
```

#### 性能指标与成本控制

| 指标 | 目标值 | 备注 |
|------|--------|------|
| 单次推理延迟 | 500-800ms | Qwen-VL 7B 基线;14B 更慢 |
| 调用频率 | <30% 帧率 | 常规对话 5-10%,复杂场景 20-30% |
| Rate limit | 最多 1 次/5 秒 | 防止连续相似帧重复调用 |
| 缓存命中率 | >50% | 结合 Exist Cache + Vector Cache |

**成本估算**(假设 10fps 抽帧后视频流):
- 无优化:10 fps × 500ms = 每秒 5 次 VLM 调用,不可持续
- L1 ViT 替代:10 fps × 50ms(ViT)+ 0.5 fps × 500ms(VLM,5% 触发率)= 每秒 0.5 次 VLM 调用,可接受
- **L1 + L2 分层后,VLM 调用减少 90-95%**

---

### 2.3 L3: 跨帧叙事(VLM 多帧,探索性)

#### 职责边界

理解**时序动作和行为模式**:
- 动作序列("用户先站起来,走到窗边,然后拿起手机")
- 行为模式("用户反复看手表,显得焦急")
- 情绪变化轨迹("从平静到逐渐激动")

**当前状态**:探索性,非主线。路线图 P4 多模态深化阶段考虑。

#### 触发条件(严格)

```cpp
bool ShouldCallMultiFrameVLM(const FrameSequence& frames) {
    // 1. 检测到显著运动(MOG2 前景比例 > 阈值)
    if (frames.motion_intensity < 0.3) return false;
    
    // 2. 单帧 VLM 连续返回"不确定"或矛盾结果
    if (frames.consecutive_uncertain_count < 3) return false;
    
    // 3. 用户明确询问动作("我刚才做了什么?")
    if (!frames.user_asked_about_action) return false;
    
    // 4. 距上次多帧调用 > 30 秒(成本极高)
    if (frames.last_multiframe_call_ms + 30000 > now_ms) return false;
    
    return true;
}
```

#### 实现方式

**输入**:选择 3-5 个关键帧(间隔 0.5-1 秒,避免冗余)

**Prompt**:
```
以下是时间序列的 3 张图片,请描述用户的动作变化:
图1(0 秒):...
图2(1 秒):...
图3(2 秒):...

请用一句话概括用户做了什么。
```

**挑战**:
- 多帧输入 → 推理时间线性增加(3 帧 × 500ms ≈ 1.5s)
- Qwen-VL 对多图的理解能力需验证
- KV Cache 复用策略更复杂(帧间 image embeddings 不同)

**优先级**:低(P4),先做 L1+L2,实测瓶颈再考虑 L3。


---

## 3. 典型场景流程示例

本节描述 L1/L2/L3 全部接通后的目标行为，不代表当前端到端链路已经具备 ViT 分析或多模态情绪融合。当前可用路径仅能把已有 VisionEvent/VisionAnalysis 作为受控 vision.observe 上下文注入。

### 场景 1:用户在视频通话中哭泣(L1 高置信,无需 L2)

```
输入:
  - 视频帧:单人清晰人脸,泪痕明显
  - 文本:"我真的好难过..."

处理流程:
  1. L1 ViT 分析 (~40ms)
     - 情绪:sad,置信度 0.95(高)
     - 性别:female,置信度 0.93
     - 表情强度:0.9(强烈)
     - 身份:匹配到 face_id="uuid-123"(第 5 次见)
  
  2. BERT 文本情绪 (~30ms)
     - 情绪:悲伤,置信度 0.92
  
  3. 触发判断:
     - ViT 高置信 ✓
     - 文本与视觉一致 ✓
     - 不触发 VLM
  
  4. 融合层:
     - BERT(0.92,悲伤)+ ViT(0.95,悲伤)→ 最终:悲伤(高置信)
     - 更新情绪状态机:valence ↓↓, arousal ↑
  
  5. Agent 回复:
     "我看到你很难过,眼泪都流下来了。想跟我聊聊吗?"
     (利用 ViT 的"强度 0.9"生成更贴合的回复)

总延迟:~70ms(ViT + BERT)
```

---

### 场景 2:用户在复杂场景,表情不明显(触发 L2)

```
输入:
  - 视频帧:人脸侧面/模糊,背景复杂(书桌、文件堆)
  - 文本:"唉,又要加班了"

处理流程:
  1. L1 ViT 分析 (~40ms)
     - 人脸质量差,部分遮挡
     - 情绪:neutral,置信度 0.52(低)← 触发 L2
  
  2. BERT 文本情绪 (~30ms)
     - 情绪:疲惫/沮丧,置信度 0.78
  
  3. 触发判断:
     - ViT 置信度 0.52 < 0.7 → 触发 VLM
     - Reason: LOW_CONFIDENCE
  
  4. L2 VLM 推理 (~600ms)
     - Prompt:"请仔细描述画面中人物的表情细节和周围环境"
     - 输出:"用户坐在书桌前,面前堆满文件和笔记本,
              低头看手机,肩膀有些耷拉,表情看不太清但整体显疲惫"
  
  5. 融合层:
     - BERT(疲惫)+ ViT(中性,低置信)+ VLM("肩膀耷拉"、"疲惫")
     - 最终:疲惫/沮丧(中高置信)
  
  6. Agent 回复:
     "看到你面前堆了不少工作,肩膀都塌下来了。要不要先休息一下?"
     (利用 VLM 的"肩膀耷拉"细节,让回复更具情境感)

总延迟:~670ms(ViT + BERT + VLM)
```

---

### 场景 3:情绪冲突,需要 L2 细节推理

```
输入:
  - 视频帧:人脸清晰,嘴角上扬(微笑)
  - 文本:"我笑不代表我开心"

处理流程:
  1. L1 ViT 分析 (~40ms)
     - 情绪:happy,置信度 0.88(高)
     - 表情强度:0.6(中等微笑)
  
  2. BERT 文本情绪 (~30ms)
     - 情绪:矛盾/防御,置信度 0.72
  
  3. 触发判断:
     - EmotionDistance(happy, 矛盾) = 0.65 > 0.5 → 触发 VLM
     - Reason: EMOTION_CONFLICT
  
  4. L2 VLM 推理 (~650ms)
     - Prompt:"画面中人物表面看起来 happy,但可能隐藏着其他情绪。
               请观察眼神、姿态、周围环境等细节,判断真实情绪状态"
     - 输出:"虽然嘴角上扬,但眼睛没有笑意(杜乡式微笑),
              眉头微皱,双手交叉抱胸,可能是防御姿态或勉强的笑容"
  
  5. 融合层:
     - BERT(矛盾)+ ViT(表面微笑)+ VLM("眼睛无笑意"、"防御姿态")
     - 最终:强颜欢笑 / 隐藏负面情绪(高置信)
  
  6. Agent 回复:
     "我注意到你虽然在笑,但眼神里好像有些不一样,双手也抱在胸前。
      有什么不开心的事吗?想跟我说说吗?"
     (利用 VLM 的细节观察,识别"微笑抑郁"这种复杂情绪)

总延迟:~720ms(ViT + BERT + VLM)
```


---

## 4. 多层融合策略

### 4.1 当前已实现边界

当前仓库没有 BERT + ViT + VLM 的三源 EmotionFusionAnalyzer。现有路径分为两条：

~~~text
文本情绪:
  BERT primary
  + keyword/vector evidence
  + LLM fallback gate
  -> FusedEmotionAnalyzer
  -> emotion state update

视觉观察:
  VisionEvent
  -> monitor / dedup
  -> SkillVisionEventSink
  -> SkillObservation(vision.observe)
  -> Persona Runtime system context
~~~

SkillVisionEventSink 会根据 analysis confidence 或 saliency score 计算 observation confidence，并将 duplicate/rate-limited 事件标记为 stale。只有置信度达到阈值且事件非 stale 时，观察内容才进入 prompt。该路径不会直接覆盖文本情绪，也不会把单帧 VLM 输出写入长期事实。

### 4.2 目标扩展接口

L1 落地后，建议通过独立视觉证据接口扩展，而不是把人脸模型调用直接塞进现有文本融合器：

~~~cpp
struct VisualEmotionEvidence {
    std::string label;
    double confidence = 0.0;
    double quality = 0.0;
    std::string track_id;
    std::uint64_t frame_id = 0;
};

class IVisualEmotionEvidenceProvider {
public:
    virtual ~IVisualEmotionEvidenceProvider() = default;
    virtual core::Result<std::optional<VisualEmotionEvidence>> Latest(
        std::string_view session_id,
        std::chrono::milliseconds max_age) const = 0;
};
~~~

建议由新的多模态编排层消费文本分析、视觉证据和场景观察，再输出对 Persona Runtime 的建议。现有 FusedEmotionAnalyzer 继续保持文本域职责，避免同时承担模型调度、视频时序和状态机策略。

### 4.3 置信度与状态更新原则

| 条件 | 处理策略 |
|------|----------|
| 文本与 L1 视觉一致，且两者高置信 | 可提高最终置信度，但仍受时间新鲜度和人脸质量约束 |
| L1 低置信、遮挡或多人 | 降低视觉权重，必要时触发 L2 场景观察 |
| 文本与视觉冲突 | 不做简单权重平均；记录冲突并将 VLM 用于解释性上下文 |
| 只有单帧 VLM 推断 | 只进入 prompt，不直接更新情绪状态机 |
| 多帧一致且高置信 | 后续可通过独立 gate 小幅影响状态更新强度 |
| duplicate/rate-limited/stale | 记录但不注入，避免连续帧重复强化 |

视觉情绪和身份属性具有高误判成本。第一版应优先保证“不会错误强化”，而不是追求每帧都产生结论。

## 5. 性能优化路径

### 5.1 L1 层优化（待实现）

L1 还没有实际模型与基准，因此当前只能确定优化顺序：

1. 先建立 IFaceAnalyzer、固定输入尺寸、质量评估和可复现 benchmark。
2. 使用 track id 与短时间窗口复用检测结果，避免对连续近似帧重复做人脸检测。
3. 在模型支持时做 batch 推理和共享 backbone，多任务头只消费同一次 embedding。
4. 身份检索先采用受控规模的精确 cosine；达到数据规模阈值后再评估 HNSW/Faiss。
5. 不把 VLM 的 pooled image embedding 当作可直接复用的 ViT token 输出，两者语义和形状并不等价。

### 5.2 L2 层优化（当前实现）

当前 VLM 已有三类缓存/并行能力，职责需要严格区分：

| 能力 | 当前实现 | 可复用内容 |
|------|----------|------------|
| Exact result cache | VLMCache | 完全匹配请求的最终文本结果 |
| Vector result cache | VectorIndex | 相似图像的历史结果或 dedup 候选，不复用 KV |
| Prompt KV Cache | IPromptKvCache + llama.cpp sequence state | 同模型、同 session、完全相同图像的 chat-template 前缀和 image tokens |
| Runner pool | LlamaRunnerPool | 共享只读 model/mmproj，每个 slot 保持独立 context/sampler |

Prompt KV key 当前由以下内容构成：

~~~text
prompt-kv-v1
+ model/mmproj/gpu fingerprint
+ session_id
+ exact SHA-256(image bytes)
~~~

命中后恢复 sequence state，跳过已缓存的前导文本和图像 chunk；图像之后的 prompt suffix 与生成 token 仍重新计算。请求失败时会清理对应 context，避免污染后续请求。

以下方案不应实现：

- 按 prompt template 长期绑定 seq_id 并跨 runner/context 共享可变 KV。
- 根据 cosine/SSIM 命中近似图像后加载历史 KV state。
- 把 pooled image embedding 当作完整 vision token 序列注入其他请求。
- 在 GStreamer callback 中同步执行图片编码、Redis 或 VLM RPC。

### 5.3 当前测试与指标缺口

Prompt KV Cache 和 runner pool 已有核心代码与配置解析，但还缺少独立自动化回归。需要补齐：

- sequence state store/restore、损坏 state、失败清理和同 runner 热命中测试；
- memory backend 容量淘汰测试；
- Redis TTL、跨进程恢复和连接失败降级测试；
- runner acquire/release、并发请求、Unload 等待和异常隔离测试；
- 固定图像、prompt、生成长度和硬件条件的 benchmark；
- 将 prompt_kv_cache_hit、pool queue wait、restore failure 暴露到 gRPC response 或 metrics。

已有“约 70%-80%”提升属于本地阶段性观测，在上述 benchmark 建立前不应作为稳定 SLA。

### 5.4 实时链路背压

Media pipeline 的下一步性能重点不是继续增加帧率，而是建立有界队列：

~~~text
appsink callback
  -> compute pool: sampler
  -> GStreamer JPEG/PNG image encoder
  -> IO pool: shared-memory producer
  -> inference receiver/private backlog/VLM coordinator
  -> event monitor/dedup
~~~

原始 RTC 帧在 sampler 前仍可通过 appsink/采样频率控制避免阻塞 GStreamer；sampler 已选中的帧则进入有限 Skill 的异步 publish 与推理端 mapped spool，不以普通队列满作为正常丢弃条件。多路会话需要记录每个 execution 的 selected、publish pending、spool admission、VLM terminal、drain progress 和完成数量，避免高活跃会话长期占满共享 worker 或磁盘预算。

推理端现已接入第一版真实共享内存 IPC 核心：`agent_ipc` 使用 Boost.Interprocess 管理跨平台 region，固定 slot header 内含 session/trace/frame/timestamp/format 等描述符，payload 位于对应定长槽位；每槽 `sequence` 与 region enqueue/dequeue 原子位置完成 MPMC publish/claim/ack，magic/version/layout/epoch 用于打开校验和重建隔离。`ClaimedSharedFrame` 以 move-only RAII 保证普通失败路径释放槽位。

`InferenceFrameIpcReceiver` 是共享内存生命周期的推理侧终点：它校验 descriptor，把 shared payload 一次复制到 core 内存池，立即 ack，然后才把私有 `OwnedInferenceFrame` 投入 `SegmentedInferenceFrameBacklog`。backlog 按 session 分区，segment 使用短生命周期原子 lease，bitmap/token 负责非线性定位可用任务；`InferenceFrameCoordinator` 使用已有 `core::ThreadPool` 并发取帧并同步调用 `IVlmVisionClient`，成功、业务失败和清洗后的意外异常都会形成 terminal record；多个 worker 乱序完成后，`SessionInferenceFrameResultTable` 在各 session 内按 `timestamp_us + frame_id` 重组。worker、VLM/API 和结果表都不得持有 shared span。

RTC 关键帧编码和 Gateway producer adapter 已完成。`GStreamerVideoFrameEncoder` 支持 RGB/BGR stride、分辨率重新协商、JPEG/PNG，以及 Auto/Software/NVIDIA/D3D11/VAAPI/QSV image-encoder preference。Auto 只探测适合独立图片的 encoder；本机虽然存在 NVENC H.264，但缺少 `nvjpegenc`，因此正确回退 `jpegenc`，没有把有状态视频编码协议强塞给 VLM 单帧输入。编码发生在 compute pool 的 sampler 命中之后，IPC publish 通过 IO pool 执行。

`InferenceFrameGatewayProducer`、`RecoverableInferenceFrameIpcSink` 与 `ReconnectableInferenceFrameIpcSource` 已完成 producer publish 和协调式 lifecycle。真实双进程 E2E 覆盖 JPEG encode -> IPC -> receiver -> private backlog；故障注入子进程在持有 claim 时 `_Exit`，父进程验证 stale slot 背压，再 remove/recreate region、检查新 epoch 并恢复消费。`media_rtc_ipc_fake_vlm_stress` 还覆盖 synthetic decoded RGB -> OpenCV sampler -> JPEG -> Gateway producer -> 独立 shared-memory create/open 端点 -> receiver -> backlog -> coordinator -> FakeVLM -> result table，并提供逐阶段丢失、saliency、事件覆盖、session 公平性和 p50/p95/p99。当前尚未完成的是正式 Gateway signaling/config 接线、真实 VLM adapter、health supervisor 自动调用 recreate/reconnect、轻量 wake/control、per-execution mapped spool、Skill Seal/Drain completion fence，以及不重建旧 region 的 stale-slot owner/lease 回收。

首轮准 E2E 数据显示，4 session x 15 FPS、4 worker x 20 ms 时 selected 帧零丢失且事件覆盖 100%；8 session x 30 FPS、2 worker x 100 ms 时 selected 帧已有 77.4% 在 receiver -> private backlog admission 被拒绝，但 16/16 事件仍至少保留一帧；VLM 延迟增加到 250-300 ms 后 selected 帧丢失达到 89.9%-91.8%，高 saliency 丢失达到 97.4%-98.7%，事件覆盖下降到 37.5%-50%。compute queue、编码、IO queue 和 IPC publish 在四组场景中均未丢帧，说明当前首要结构性瓶颈是 VLM 服务率与 `RejectNewest` backlog admission，而不是共享内存数据面吞吐。

因此 sampler 后的帧不能继续沿用普通原始视频帧的无差别丢弃语义。项目媒体管线绑定有限 `SkillSession`，目标生产路径调整为推理端 private file-backed mapped spool：selected frame 正常路径无静默丢失，流结束后通过 Seal expected-count fence 完成 VLM drain 和事件 finalize，再结束 Skill。任何策略都不得覆盖 shared-memory claimed/in-flight slot；普通丢弃策略只作为异常资源耗尽兜底。完整生命周期与当前 Skill 实现校准见 `docs/architecture/SKILL_MEDIA_DRAIN_SPOOL_DESIGN.md`。

## 6. 实施路线图

### Phase 0: 生产链路接线与回归（当前优先，P1）

目标：把已经存在的 media、VLM 与 persona 组件接成正式链路。

- 在 agent_gateway_server 挂载 /ws/vision/signaling。
- 接入 WebRtcSessionRegistry、checkpoint store 和维护任务。
- 将已完成的 JPEG/PNG encoder、IO-pool sink 和共享内存 producer 组装进正式 Gateway signaling route。
- 实现 gRPC 或 OpenAI-compatible IVlmVisionClient adapter，并组装已完成的 inference coordinator。
- 将 VisionInferenceResult 转换为 VisionAnalysis 和 VisionEvent。
- 接入 monitor、dedup、rate limit 与 SkillVisionEventSink。
- 增加多 session、断线恢复、重复 close、pipeline fatal error 和长稳 E2E。
- 为 Prompt KV Cache 与 runner pool 补齐专门测试和 benchmark。

验收标准：

- 正式 Gateway 可以完成浏览器推流到受控 vision.observe 注入。
- VLM 请求不运行在 GStreamer callback 或 WebSocket 线程。
- 队列容量、丢帧策略和 session cleanup 可观测。
- 单帧 VLM 结果不会直接更新情绪状态机或 L3 长期事实。

### Phase 1: L1 ViT 最小闭环（P1/P2）

目标：先交付结构化情绪证据，不一次性承诺身份、性别和年龄能力。

- 定义 IFaceAnalyzer 和 FaceAnalysisResult。
- 实现人脸检测、质量评估和匿名 track id。
- 选择许可证明确的 ONNX 模型并建立 CPU/GPU benchmark。
- 输出 emotion、confidence、quality、track_id 和 frame_id。
- 增加单元测试、模型 smoke、人工标注质量集与 E2E。
- 第一版不持久化 face embedding；身份库作为独立隐私评审后的可选模块。

验收标准：

- 目标硬件上 p95 延迟与吞吐满足抽帧频率。
- 遮挡、侧脸、多人和无人脸场景能明确降级。
- 模型输出经 core::Status 和 logger 统一处理。
- 精度指标来自项目自己的目标域验证集，而不是直接引用公开数据集成绩。

### Phase 2: L1/L2 触发与多模态 gate（P2）

- 实现 IVisualEmotionEvidenceProvider 或等价接口。
- 按质量、置信度、文本冲突、用户显式视觉请求触发 L2。
- 加入 session 级 rate limit、连续命中降权和 freshness。
- 让 VLM 解释冲突，但不把自然语言描述硬映射为高置信情绪标签。
- 建立 text-only、L1-only、L1+L2 的消融评测。

### Phase 3: 性能与存储深化（P3）

- 多路推流压力测试和公平调度。
- 评估硬件解码、batch ViT 和模型量化。
- 完成共享内存 Gateway producer、双进程故障注入和 stale slot 恢复后，再通过真实视频压测决定是否正式替换 gRPC 帧 payload。
- 只有用户授权、删除和加密策略完成后，才评估持久化身份 embedding。

### Phase 4: L3 多帧叙事（探索性，P4）

触发条件：L1/L2 已稳定，产品评测证明动作理解是主要缺口。第一版限制为 3-5 个关键帧和显式用户请求，不默认持续运行。

## 7. 监控指标与可观测性

### 7.1 当前可记录指标

现有结构已经能够承载：

- frame id、session id、saliency score、宽高；
- VisionEvent peak score、duplicate、rate-limited、analysis confidence；
- VLM image encode、prompt eval、generation latency 和 token 数；
- exact/vector cache 的 hit、stale、result source；
- Persona Runtime 的 L0/L3/L4 hit 与阶段延迟。

### 7.2 需要新增指标

| 模块 | 指标 |
|------|------|
| WebRTC | active sessions、reconnect、pipeline create/fatal/close、cleanup latency |
| Sampler | evaluated/submitted/dropped frames、reason、per-session rate |
| Encode/IO queue | queue depth、queue wait、encode latency、drop count |
| IPC/backlog/coordinator | publish/admission reject、queue high-water、selected-to-start、event coverage、per-session completion |
| VLM adapter | request count、deadline、transport error、parse error |
| Prompt KV | hit rate、state bytes、restore/store failure、backend latency |
| Runner pool | active slots、queue wait、acquire timeout、unload wait |
| L1 ViT | inference p50/p95/p99、face quality、no-face/multi-face rate |
| 多模态 gate | text/vision agreement、conflict、L2 trigger reason、state update suppression |

### 7.3 质量指标

- L1 emotion macro-F1、per-label recall 与校准误差；
- 遮挡、侧脸、低光、多人等分桶指标；
- VLM scene/fact precision 与 hallucination rate；
- vision.observe 注入后的回复相关性和错误事实率；
- text-only、L1-only、L1+L2 消融对比；
- 身份模块如启用，单独评估 FAR/FRR，不与情绪指标混合。

### 7.4 建议日志结构

~~~json
{
  "trace_id": "xxx",
  "session_id": "session-1",
  "frame_id": 123,
  "sampler": {
    "submitted": true,
    "saliency": 0.81,
    "reason": "scene-change"
  },
  "vlm": {
    "called": true,
    "latency_ms": 620,
    "result_cache_source": "model",
    "prompt_kv_cache_hit": true
  },
  "vision_event": {
    "confidence": 0.72,
    "duplicate": false,
    "rate_limited": false,
    "prompt_injected": true
  }
}
~~~

日志不应输出原始图像、完整人脸 embedding、Redis 密码或未经脱敏的用户隐私字段。

## 8. 风险与挑战

### 8.1 技术风险

**模型域漂移**：公开 FER/Affect 数据与真实摄像头环境不同。必须使用目标域样本做分桶评测和校准，低质量帧要显式降级。

**VLM 幻觉**：VLM 更适合场景和弱解释，不适合把身份、性别或单帧情绪作为强事实。输出默认进入不确定 observation。

**KV 对齐**：只有完全相同图像和可证明一致的前缀才能恢复 sequence state。近似图像、跨模型、跨 mmproj 或跨 context 的 KV 复用均禁止。

**并发资源竞争**：共享 model/mmproj 可以降低显存占用，但 mtmd 编码、runner slot、Redis 和 GPU 仍可能形成串行瓶颈，需要用指标和压测决定 pool size。

### 8.2 工程挑战

**正式接线缺口**：现有 smoke server 证明组件可组装，但正式 Gateway 尚未挂载 signaling、VLM adapter 和维护生命周期。

**背压与关闭竞态**：多路视频必须有界排队，close/reconnect/unload 时需要确保 frame buffer、pipeline、runner lease 和异步 callback 生命周期安全。

**测试成本**：媒体、模型和 Redis 组合测试依赖较重。应保留纯单元 fake、组件 smoke 和真实模型 E2E 三层测试，不把全部验证压在单一 E2E 上。

### 8.3 隐私与产品风险

- 人脸 embedding 属于敏感生物特征，默认不持久化；
- 身份、性别和年龄属性并非多模态第一版的必要条件；
- 必须提供用户授权、删除、保留期限、租户隔离和审计策略；
- 错误视觉判断比缺失视觉判断更容易损害信任，因此产品策略应允许“不确定”和“不注入”；
- 原始帧、embedding 和视觉描述进入日志、缓存或 L3 前必须有明确数据治理规则。

## 9. 总结

分层方向仍然成立，但当前实现状态需要准确区分：

1. WebRTC/GStreamer/OpenCV、VisionEvent 和 vision.observe 已形成组件底座。
2. VLM 推理服务、结果缓存、strict Prompt KV Cache 和 runner pool 已实现核心代码。
3. 关键帧编码、IPC producer/receiver、coordinator、准 E2E 分阶段压测和双进程恢复 E2E 已完成；正式 Gateway 接线、真实 VLM adapter 和 analysis 数据源尚未完成。
4. L1 ViT 人脸情绪分析、身份库和真正的多模态情绪 gate 尚未实现。
5. 近似图像 KV 复用和跨 runner 长期 seq_id 不进入路线。
6. L3 多帧叙事继续保持探索性。

当前最合理的下一步不是立即扩展三源融合，而是先完成 Phase 0：把正式 Gateway 到受控 vision.observe 的生产链路接通，并为 Prompt KV/runner pool 建立自动化回归。随后再以最小、可评测、默认不持久化身份的方式交付 L1。

---

**文档状态**: 部分实现，持续核对  
**最后更新**: 2026-07-11
