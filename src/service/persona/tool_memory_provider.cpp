#include "tool_memory_provider.h"

#include "third_party/nlohmann/json.hpp"

#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace agent::service::persona {
namespace {

using Json = nlohmann::json;

constexpr std::string_view kVisionToolId = "vision.observe";

int JsonInt(const Json& body, std::string_view key, int fallback) {
    auto it = body.find(std::string(key));
    if (it == body.end() || !it->is_number_integer()) {
        return fallback;
    }
    return it->get<int>();
}

std::string JsonString(const Json& body, std::string_view key, std::string fallback = {}) {
    auto it = body.find(std::string(key));
    if (it == body.end() || !it->is_string()) {
        return fallback;
    }
    return it->get<std::string>();
}

std::string DefaultVisionInstruction() {
    return "用户可能正在请求视觉观察。若需要调用视觉工具，只输出 "
           "<agent_tool_call>{\"tool\":\"vision.observe\",\"arguments\":{\"top_k\":2,\"reason\":\"简短原因\"}}</agent_tool_call>。"
           "没有工具结果前，不要编造画面内容。";
}

std::string DefaultVisionSchema() {
    return R"({"tool":"vision.observe","arguments":{"top_k":2,"reason":"用户请求观察画面"}})";
}

} // namespace

VectorToolMemoryProvider::VectorToolMemoryProvider(
    std::shared_ptr<::vector::EmbeddingPipeline> embedding_pipeline,
    std::shared_ptr<vector::VectorIndexManager> index_manager,
    std::shared_ptr<vector_storage::PartitionRegistry> partition_registry,
    VectorToolMemoryProviderOptions options)
    : embedding_pipeline_(std::move(embedding_pipeline)),
      index_manager_(std::move(index_manager)),
      partition_registry_(std::move(partition_registry)),
      options_(std::move(options)),
      vision_regex_(R"(看看|看一下|看到|你看[看到见]?|你能看|画面|摄像头|你的眼|周围|在做什么|在干什么|你见|你注意到|屏幕|你面前|我的样子)") {}

core::Result<ToolMemoryContext> VectorToolMemoryProvider::Query(const ToolMemoryQuery& request) {
    ToolMemoryContext context;
    std::vector<ToolMemoryHit> hits;

    if (options_.enable_regex) {
        auto regex_hits = RegexHits(request.query);
        hits.insert(hits.end(), std::make_move_iterator(regex_hits.begin()), std::make_move_iterator(regex_hits.end()));
    }
    if (options_.enable_vector) {
        auto vector_hits = VectorHits(request.query);
        if (!vector_hits.ok()) {
            return vector_hits.status();
        }
        hits.insert(hits.end(), std::make_move_iterator(vector_hits.value().begin()), std::make_move_iterator(vector_hits.value().end()));
    }

    std::unordered_set<std::string> seen;
    std::vector<ToolMemoryHit> deduped;
    for (auto& hit : hits) {
        const std::string key = hit.tool_id.empty() ? hit.memory_hash : hit.tool_id;
        if (key.empty() || seen.insert(key).second) {
            deduped.push_back(std::move(hit));
        }
    }
    std::sort(deduped.begin(), deduped.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.priority != rhs.priority) {
            return lhs.priority > rhs.priority;
        }
        return lhs.score > rhs.score;
    });
    if (deduped.size() > static_cast<std::size_t>(std::max(1, options_.top_k))) {
        deduped.resize(static_cast<std::size_t>(options_.top_k));
    }

    context.hit = !deduped.empty();
    context.prompt_block = BuildPromptBlock(deduped);
    context.hits = std::move(deduped);
    return context;
}

ToolMemoryHit VectorToolMemoryProvider::MakeVisionRegexHit() {
    ToolMemoryHit hit;
    hit.tool_id = std::string(kVisionToolId);
    hit.memory_hash = "global:l4:vision.observe:regex";
    hit.instruction = DefaultVisionInstruction();
    hit.schema_json = DefaultVisionSchema();
    hit.score = 1.0;
    hit.priority = 100;
    hit.regex_hit = true;
    return hit;
}

std::vector<ToolMemoryHit> VectorToolMemoryProvider::RegexHits(std::string_view query) const {
    if (query.empty() || !std::regex_search(query.begin(), query.end(), vision_regex_)) {
        return {};
    }
    return {MakeVisionRegexHit()};
}

core::Result<std::vector<ToolMemoryHit>> VectorToolMemoryProvider::VectorHits(std::string_view query) const {
    if (query.empty() || !embedding_pipeline_ || !index_manager_ || !partition_registry_) {
        return std::vector<ToolMemoryHit>{};
    }

    vector_storage::PartitionKey key;
    key.collection_id = options_.collection_id;
    key.tenant_id = options_.tenant_id;
    key.user_id = "";
    key.memory_level = "L4";
    auto partition = partition_registry_->Lookup(key);
    if (!partition.ok()) {
        return partition.status();
    }
    if (!partition.value().has_value()) {
        return std::vector<ToolMemoryHit>{};
    }

    auto embedding = embedding_pipeline_->Encode(query);
    if (!embedding.ok()) {
        return embedding.status();
    }
    vector::IndexSearchOptions search_options;
    search_options.top_k = static_cast<std::size_t>(std::max(1, options_.top_k));
    search_options.include_forgotten = false;
    search_options.memory_type = "capability";
    search_options.min_score = static_cast<float>(options_.min_score);
    auto entries = index_manager_->Search(partition.value().value(), embedding.value(), search_options);
    if (!entries.ok()) {
        return entries.status();
    }

    std::vector<ToolMemoryHit> hits;
    for (const auto& entry : entries.value()) {
        ToolMemoryHit hit;
        hit.memory_hash = entry.memory_hash;
        hit.score = entry.emotion_intensity;
        hit.vector_hit = true;
        hit.instruction = entry.payload;
        try {
            auto meta = Json::parse(entry.extra_metadata_json.empty() ? "{}" : entry.extra_metadata_json);
            hit.tool_id = JsonString(meta, "tool_id");
            hit.schema_json = JsonString(meta, "schema", JsonString(meta, "schema_json"));
            hit.priority = JsonInt(meta, "priority", 0);
            const auto instruction = JsonString(meta, "instruction");
            if (!instruction.empty()) {
                hit.instruction = instruction;
            }
        } catch (const Json::exception&) {
        }
        if (hit.tool_id.empty()) {
            hit.tool_id = entry.cache_key;
        }
        if (!hit.tool_id.empty() && !hit.instruction.empty()) {
            hits.push_back(std::move(hit));
        }
    }
    return hits;
}

std::string VectorToolMemoryProvider::BuildPromptBlock(const std::vector<ToolMemoryHit>& hits) {
    if (hits.empty()) {
        return {};
    }
    std::ostringstream out;
    out << "<tool_memory_l4>\n";
    for (const auto& hit : hits) {
        out << "- tool: " << hit.tool_id << "\n";
        out << "  instruction: " << hit.instruction << "\n";
        if (!hit.schema_json.empty()) {
            out << "  schema: " << hit.schema_json << "\n";
        }
    }
    out << "工具指令只在需要调用工具时使用；没有工具结果前不要编造外部观察。\n";
    out << "</tool_memory_l4>";
    return out.str();
}

} // namespace agent::service::persona
