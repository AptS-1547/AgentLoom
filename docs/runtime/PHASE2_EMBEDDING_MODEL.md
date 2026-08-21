# Phase 2: ONNX Text Embedding 模型集成

## 概述

Phase 2 实现了基于 ONNX Runtime 的文本 embedding 推理能力，支持 sentence-transformers 等主流 embedding 模型。

**核心能力**：
- Tokenizer → ONNX 推理 → Pooling → L2 Normalize 完整链路
- Mean / Cls 两种 pooling 策略
- 自动检测模型 IO（兼容 HF 标准导出 + optimum 导出）
- CPU / CUDA 自动 fallback
- 线程安全并发推理

## 快速开始

### 1. 导出 ONNX 模型

使用带有 Python 导出依赖的虚拟环境：

```powershell
# 激活虚拟环境
cd <python-environment-root>
.\venv\Scripts\Activate.ps1

# 导出 MiniLM embedding 模型
cd <agentloom-repository>
python scripts\export_embedding_onnx.py `
    --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `
    --output .\onnx_models\minilm `
    --verify `
    --benchmark
```

**输出**：
- `onnx_models/minilm/model.onnx` — ONNX 模型
- `onnx_models/minilm/model_metadata.json` — 元数据（维度、pooling 策略）
- tokenizer.json 已在 HF 缓存中，路径：
  ```
  <huggingface-cache>/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/<revision>/tokenizer.json
  ```

### 2. 运行 E2E 测试

```powershell
$env:HF_TOKENIZER_FIXTURE_JSON = "<huggingface-cache>/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/<revision>/tokenizer.json"
$env:HF_TEXT_EMBEDDING_ONNX = "<agentloom-repository>/onnx_models/minilm/model.onnx"

ctest --test-dir build -C Release --output-on-failure -R "Embedding"
```

**预期结果**：
- 7 个 pooling 单元测试通过
- 3 个 E2E 测试通过（单句 / batch / pipeline）

### 3. C++ 集成示例

```cpp
#include "embedding_pipeline.h"
#include "hf_tokenizer.h"
#include "onnx_text_embedding_model.h"

using namespace vector;

// 初始化（启动时一次）
auto tokenizer = HfTokenizer::LoadFromFile("tokenizer.json").value();

EmbeddingModelOptions opts;
opts.model_path = "model.onnx";
opts.pooling = PoolingStrategy::Mean;
opts.normalize = true;
opts.expected_dimension = 384;  // MiniLM-L12 维度

auto model = OnnxTextEmbeddingModel::Load(std::move(opts)).value();

auto pipeline = std::make_shared<EmbeddingPipeline>(
    std::make_shared<HfTokenizer>(std::move(tokenizer)),
    std::move(model)
);

// 对话热路径
std::string user_query = "今天天气怎么样";
auto embedding = pipeline->Encode(user_query).value();  // vector<float> [384]

// Batch 推理
std::array<std::string_view, 3> queries{
    "第一个问题",
    "第二个问题",
    "第三个问题"
};
auto batch = pipeline->EncodeBatch(std::span(queries)).value();
// batch.embeddings: [3 * 384] 连续存储
// batch.row(0): 第一个 embedding 的 span
```

## 架构设计

### 组件清单

| 组件 | 文件 | 职责 |
|------|------|------|
| **ONNX 公共基础** | `src/models/onnx_session_utils.{h,cpp}` | Env 单例、session 创建、CUDA fallback |
| **Pooling 算子** | `src/vector/text_embedding_pooling.cpp` | Mean / Cls / L2Normalize |
| **Embedding 接口** | `src/vector/text_embedding_model.h` | IEmbeddingModel 抽象 + PoolingStrategy |
| **ONNX 实现** | `src/vector/onnx_text_embedding_model.{h,cpp}` | 自动 IO 检测 + 推理 + pooling |
| **Pipeline** | `src/vector/embedding_pipeline.{h,cpp}` | Tokenizer + Model 一站式 |

### 关键设计决策

#### 1. IO 自动检测

**问题**：不同导出工具产生的 ONNX 模型 IO 命名不一致
- HF `transformers.onnx`：`input_ids` / `attention_mask`
- `optimum`：`input.1` / `attention_mask:0`
- 手工导出：任意命名

**解决**：`ClassifyInputName` 模糊匹配 + `IsLikelyPooledOutputName` 启发式检测

```cpp
InputKind ClassifyInputName(const std::string& name) {
    const std::string lower = LowerCopy(name);
    if (lower == "input_ids" || lower == "input.1" || lower == "input_ids:0") {
        return InputKind::InputIds;
    }
    // ...
}
```

#### 2. 双输出模式

**问题**：部分模型导出时已内置 pooling（输出 [B, H]），部分仅输出 hidden state（[B, S, H]）

**解决**：运行时检测 output tensor rank
- rank-2 → 直接拷贝（已 pool）
- rank-3 → 外部 pool（Mean / Cls）

```cpp
if (out_shape.size() == 2) {
    // 已 pool，直接拷贝
    result.embeddings.assign(output.GetTensorMutableData<float>(), ...);
} else if (out_shape.size() == 3) {
    // 需要 pool
    pooling::MeanPool(hidden_data, attention_mask, ...);
}
```

#### 3. Pooling 策略

| 策略 | 适用模型 | 实现 |
|------|----------|------|
| **Mean** | sentence-transformers（MiniLM / MPNet / E5） | 对 attention_mask=1 的 token 求平均 |
| **Cls** | BERT-CLS 分类头 | 提取 `[CLS]` token（首位） |

**Mean pooling 伪代码**：
```cpp
for each batch:
    sum = 0, count = 0
    for each token:
        if attention_mask[token] == 1:
            sum += hidden[token]
            count += 1
    embedding = sum / max(count, 1)
```

#### 4. L2 Normalize

**为什么需要**：语义缓存用余弦相似度，normalize 后余弦 = 点积，Faiss 可直接用 `METRIC_INNER_PRODUCT`。

```cpp
void L2NormalizeRows(float* data, size_t batch_size, size_t dimension) {
    for each row:
        norm = sqrt(sum(x^2))
        if norm > 1e-12:
            row /= norm
}
```

## 性能特征

### MiniLM-L12-v2 (384 维)

**硬件**：Intel i7-12700H (12 核) / CPU only

| Batch | 延迟 (ms) | 吞吐量 (samples/s) |
|-------|-----------|-------------------|
| 1     | ~15       | ~67               |
| 4     | ~45       | ~89               |
| 8     | ~85       | ~94               |
| 16    | ~165      | ~97               |

**瓶颈分析**：
- Tokenizer：~1.6ms / sample（已并发优化，见 Phase 1 TokenizerPool）
- ONNX 推理：~12ms / sample（batch=1）
- Pooling + Normalize：<0.5ms

**优化方向**（Phase 3+）：
- CUDA 推理（GPU 可降至 ~2ms / sample）
- Batch 聚合（对话热路径多请求合并）
- 模型量化（INT8 可减半延迟）

## 测试覆盖

### Pooling 单元测试（7 个）

| 测试 | 覆盖点 |
|------|--------|
| `MeanPoolSingleTokenAllAttended` | 单 token，mask=1 |
| `MeanPoolTwoTokensPartialMask` | 部分 mask=0 |
| `MeanPoolBatchOfTwo` | batch 维度正确性 |
| `ClsPoolExtractsFirstToken` | 提取首 token |
| `L2NormalizeZeroVectorStaysZero` | 零向量边界 |
| `L2NormalizeUnitVectorStaysUnit` | 单位向量幂等 |
| `L2NormalizeMakesNormOne` | 任意向量归一化 |

### E2E 测试（3 个，需 ONNX fixture）

| 测试 | 覆盖点 |
|------|--------|
| `LoadAndEmbedSingleText` | 单句 encode + normalize 校验 |
| `EmbedBatchProducesSameShapeForAll` | batch shape 一致性 |
| `EncodeReturnsNormalizedVector` | Pipeline 端到端 |

## 下一步

### Phase 3 — Faiss 向量索引

**目标**：内存 HNSW / IVF 索引，支持 top-k 相似度检索

**接口草案**：
```cpp
class VectorIndex {
public:
    virtual Status Add(std::span<const float> embeddings, std::span<const int64_t> ids) = 0;
    virtual Result<std::vector<SearchResult>> Search(
        std::span<const float> query, int k) const = 0;
};
```

### Phase 4 — SQLite 持久化

**目标**：metadata + fingerprint 存储，支持模型升级时清空旧缓存

**Schema 草案**：
```sql
CREATE TABLE vector_cache_entries (
    id INTEGER PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_hash BLOB NOT NULL,
    embedding BLOB NOT NULL,
    tokenizer_fingerprint TEXT NOT NULL,
    model_fingerprint TEXT NOT NULL,
    created_at INTEGER NOT NULL
);
```

### Phase 5 — 语义缓存完整链路

**目标**：query → embed → search → hit/miss → store

**流程**：
1. 用户 query → Tokenizer → Embedding
2. Faiss 检索 top-1，余弦相似度 > 0.95 → cache hit
3. Cache miss → 调用 LLM → 存储 (query, embedding, response)

## 故障排查

### 问题：E2E 测试 SKIP

**原因**：未设置 `HF_TEXT_EMBEDDING_ONNX` 环境变量

**解决**：
```powershell
$env:HF_TEXT_EMBEDDING_ONNX = "path\to\model.onnx"
ctest --test-dir build -C Release -R Embedding
```

### 问题：ONNX 推理报错 "Invalid input shape"

**原因**：tokenizer 输出的 `sequence_length` 超过模型 `max_position_embeddings`

**解决**：
```cpp
EncodeOptions opts;
opts.max_length = 128;  // 与模型导出时一致
opts.truncation = true;
auto tokenized = tokenizer->Encode(text, opts);
```

### 问题：embedding 维度不符合预期

**原因**：模型替换后维度变化，但未更新 `expected_dimension`

**解决**：
```cpp
EmbeddingModelOptions opts;
opts.expected_dimension = 384;  // MiniLM-L12
// opts.expected_dimension = 768;  // BERT-base
```

首次 `Embed` 会记录实际维度，后续不匹配会返回 `FailedPrecondition`。

## 参考资料

- [ONNX Runtime C++ API](https://onnxruntime.ai/docs/api/c/)
- [sentence-transformers 文档](https://www.sbert.net/)
- [HuggingFace Tokenizers Rust](https://github.com/huggingface/tokenizers)
- [Faiss Wiki](https://github.com/facebookresearch/faiss/wiki)
