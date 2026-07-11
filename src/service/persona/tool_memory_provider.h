#pragma once

#include "result.h"
#include "vector_index_manager.h"
#include "../../storage/vector/vector_metadata.h"
#include "../../storage/vector/vector_partition_registry.h"
#include "embedding_pipeline.h"

#include <memory>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

namespace agent::service::persona {

struct ToolMemoryQuery {
    std::string session_id;
    std::string user_uuid;
    std::string persona_id;
    std::string trace_id;
    std::string query;
};

struct ToolMemoryHit {
    std::string tool_id;
    std::string memory_hash;
    std::string instruction;
    std::string schema_json;
    double score = 0.0;
    int priority = 0;
    bool regex_hit = false;
    bool vector_hit = false;
};

struct ToolMemoryContext {
    bool hit = false;
    std::vector<ToolMemoryHit> hits;
    std::string prompt_block;
};

class IToolMemoryProvider {
public:
    virtual ~IToolMemoryProvider() = default;
    /// 查询与本轮文本相关的工具记忆。
    /// @param request session/user/persona scope、trace 和 UTF-8 查询文本。
    /// @return 命中列表及可注入 prompt block；provider 必须执行租户和用户隔离。
    virtual core::Result<ToolMemoryContext> Query(const ToolMemoryQuery& request) = 0;
};

struct VectorToolMemoryProviderOptions {
    std::int64_t collection_id = 0;
    std::string tenant_id;
    int top_k = 3;
    double min_score = 0.78;
    bool enable_regex = true;
    bool enable_vector = true;
};

class VectorToolMemoryProvider final : public IToolMemoryProvider {
public:
    VectorToolMemoryProvider(std::shared_ptr<::vector::EmbeddingPipeline> embedding_pipeline,
                             std::shared_ptr<vector::VectorIndexManager> index_manager,
                             std::shared_ptr<vector_storage::PartitionRegistry> partition_registry,
                             VectorToolMemoryProviderOptions options = {});

    core::Result<ToolMemoryContext> Query(const ToolMemoryQuery& request) override;

private:
    static ToolMemoryHit MakeVisionRegexHit();
    static std::string BuildPromptBlock(const std::vector<ToolMemoryHit>& hits);
    std::vector<ToolMemoryHit> RegexHits(std::string_view query) const;
    core::Result<std::vector<ToolMemoryHit>> VectorHits(std::string_view query) const;

    std::shared_ptr<::vector::EmbeddingPipeline> embedding_pipeline_;
    std::shared_ptr<vector::VectorIndexManager> index_manager_;
    std::shared_ptr<vector_storage::PartitionRegistry> partition_registry_;
    VectorToolMemoryProviderOptions options_;
    std::regex vision_regex_;
};

} // namespace agent::service::persona
