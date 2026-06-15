#include "document_types.h"
#include "semantic_cache_pipeline.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>

namespace agent::document {
namespace {

using EmbeddingVector = std::vector<float>;
using BlockGroup = std::vector<std::size_t>;
using Clock = std::chrono::steady_clock;

std::uint64_t SinceMs(Clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count());
}

std::size_t Utf8SequenceLength(unsigned char ch) noexcept;
bool IsUtf8Continuation(unsigned char ch) noexcept;
constexpr std::string_view kChunkPromptVersion = "document_chunk_title_v2";

bool IsStructured(const DocumentBlock& block) {
    return block.kind == "heading" || block.heading_level.has_value();
}

bool IsSeparatorText(std::string_view text) {
    const auto clean = CleanText(text);
    if (clean.empty()) {
        return true;
    }
    return clean.find_first_not_of("-_=*#—") == std::string::npos;
}

bool ShouldChunk(const DocumentBlock& block) {
    if (block.text.empty() || IsSeparatorText(block.text) || IsStructured(block)) {
        return false;
    }
    return block.confidence < 0.82 || block.text.size() >= 90;
}

std::string JoinBlockText(const std::vector<DocumentBlock>& blocks, const BlockGroup& group) {
    std::string text;
    for (const auto index : group) {
        if (!text.empty()) {
            text.push_back('\n');
        }
        text += blocks[index].text;
    }
    return text;
}

std::string Hex64(std::uint64_t value) {
    std::ostringstream out;
    out << std::hex << value;
    return out.str();
}

std::string StableTextHash(std::string_view text) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const auto ch : text) {
        hash ^= static_cast<unsigned char>(ch);
        hash *= 1099511628211ull;
    }
    return Hex64(hash);
}

DocumentLlmChunkCacheKey MakeLlmCacheKey(std::string_view model, std::string_view text) {
    return DocumentLlmChunkCacheKey{
        .prompt_version = std::string(kChunkPromptVersion),
        .model = std::string(model),
        .text_hash = StableTextHash(text),
    };
}

nlohmann::json SliceToCacheJson(const ChunkSlice& slice) {
    return {
        {"title", slice.title},
        {"summary", slice.summary},
        {"text", slice.text},
        {"kind", slice.kind},
        {"confidence", slice.confidence},
    };
}

ChunkSlice SliceFromCacheJson(const nlohmann::json& json) {
    ChunkSlice slice;
    if (!json.is_object()) {
        return slice;
    }
    slice.title = json.value("title", std::string{});
    slice.summary = json.value("summary", std::string{});
    slice.text = json.value("text", std::string{});
    slice.kind = json.value("kind", std::string{"paragraph"});
    slice.confidence = json.value("confidence", 0.5);
    return slice;
}

std::string ChunkToSemanticPayload(const ChunkTrunk& chunk) {
    nlohmann::json slices = nlohmann::json::array();
    for (const auto& slice : chunk.slices) {
        slices.push_back(SliceToCacheJson(slice));
    }
    nlohmann::json payload = {
        {"title", chunk.title},
        {"summary", chunk.summary},
        {"slices", slices},
        {"confidence", chunk.confidence},
        {"metadata", chunk.metadata},
    };
    return payload.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

std::optional<ChunkTrunk> ChunkFromSemanticPayload(std::string_view payload) {
    try {
        const auto json = nlohmann::json::parse(payload);
        if (!json.is_object()) {
            return std::nullopt;
        }
        ChunkTrunk chunk;
        chunk.title = json.value("title", std::string{});
        chunk.summary = json.value("summary", std::string{});
        chunk.confidence = json.value("confidence", 0.72);
        chunk.source = "llm";
        if (const auto it = json.find("metadata"); it != json.end() && it->is_object()) {
            chunk.metadata = it->get<std::map<std::string, std::string>>();
        }
        if (const auto it = json.find("slices"); it != json.end() && it->is_array()) {
            for (const auto& item : *it) {
                chunk.slices.push_back(SliceFromCacheJson(item));
            }
        }
        return chunk;
    } catch (...) {
        return std::nullopt;
    }
}

void StampReusedChunk(ChunkTrunk& chunk,
                      const std::vector<DocumentBlock>& blocks,
                      const BlockGroup& group,
                      std::string_view text) {
    const auto& first = blocks[group.front()];
    chunk.chunk_id = "chunk-" + std::to_string(first.order);
    chunk.block_ids.clear();
    chunk.block_ids.reserve(group.size());
    for (const auto index : group) {
        chunk.block_ids.push_back(blocks[index].id);
    }
    chunk.text = std::string(text);
    chunk.order = first.order;
    chunk.page = first.page;
    chunk.slide = first.slide;
    chunk.source = "llm";
    chunk.reused = true;
}

bool IsSupportedEmbeddingDim(std::size_t dim) noexcept {
    return dim == agent::semantic_cache::kExpectedEmbeddingDim;
}

bool StartsWithAny(std::string_view text, std::initializer_list<std::string_view> prefixes) {
    for (auto prefix : prefixes) {
        if (text.starts_with(prefix)) {
            return true;
        }
    }
    return false;
}

bool LooksLikeShortNumberedTitle(std::string_view text) {
    const auto clean = CleanText(text);
    if (clean.empty() || clean.size() > 128) {
        return false;
    }
    if (clean.starts_with("创新点")) {
        return true;
    }
    if (StartsWithAny(clean, {"(", "（", "一、", "二、", "三、", "四、", "五、", "六、"})) {
        return true;
    }
    if (std::isdigit(static_cast<unsigned char>(clean.front()))) {
        return clean.find('.') != std::string::npos ||
               clean.find(' ') != std::string::npos ||
               clean.find("、") != std::string::npos ||
               clean.find("。") != std::string::npos;
    }
    return false;
}

float SimilarityScore(const EmbeddingVector& lhs, const EmbeddingVector& rhs) {
    if (lhs.size() != rhs.size() || !IsSupportedEmbeddingDim(lhs.size())) {
        return 0.0f;
    }
    if (lhs.size() == agent::semantic_cache::kExpectedEmbeddingDim) {
        return dot_product_unrolled<agent::semantic_cache::kExpectedEmbeddingDim>(lhs.data(), rhs.data());
    }
    return 0.0f;
}

std::optional<EmbeddingVector> EmbedBlockText(std::string_view text,
                                              const std::shared_ptr<IDocumentEmbeddingProvider>& provider,
                                              DocumentChunkBuildMetrics* metrics) {
    if (!provider) {
        return std::nullopt;
    }
    if (metrics) {
        ++metrics->embedding_request_count;
    }
    const auto started = Clock::now();
    auto embedded = provider->EmbedText(text);
    if (metrics) {
        const auto elapsed = SinceMs(started);
        metrics->embedding_ms += elapsed;
        metrics->embedding_sample_ms.push_back(elapsed);
        metrics->embedding_sample_bytes.push_back(text.size());
    }
    if (embedded.ok() && IsSupportedEmbeddingDim(embedded.value().size())) {
        return std::move(embedded).value();
    }
    return std::nullopt;
}

bool IsAttachableTitle(const DocumentBlock& block) {
    if (block.text.empty() || IsSeparatorText(block.text)) {
        return false;
    }
    if (IsStructured(block)) {
        return true;
    }
    if (block.text.size() > 96) {
        return false;
    }
    return LooksLikeShortNumberedTitle(block.text) ||
           DetectNumberingLevel(block.text).has_value() ||
           InferHeadingLevel(block).has_value();
}

bool GroupHasExplicitTitle(const std::vector<DocumentBlock>& blocks, const BlockGroup& group) {
    if (group.empty()) {
        return false;
    }
    const auto& first = blocks[group.front()];
    return IsAttachableTitle(first);
}

void FlushCurrent(std::vector<BlockGroup>& groups,
                  BlockGroup& current,
                  std::optional<std::size_t>& pending_title,
                  EmbeddingVector& current_embedding,
                  bool& has_embedding) {
    if (!current.empty()) {
        groups.push_back(std::move(current));
        current = {};
    }
    pending_title.reset();
    current_embedding = {};
    has_embedding = false;
}

std::vector<std::string> SplitSentences(std::string_view text) {
    std::vector<std::string> out;
    std::string current;
    for (std::size_t i = 0; i < text.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        const bool ascii_break = ch == '\n' || ch == '\r' || ch == '.' || ch == '!' || ch == '?' || ch == ';';
        const bool cjk_break =
            i + 2 < text.size() &&
            ch == 0xE3 &&
            static_cast<unsigned char>(text[i + 1]) == 0x80 &&
            static_cast<unsigned char>(text[i + 2]) == 0x82;
        if (cjk_break) {
            current.push_back(static_cast<char>(ch));
            current.push_back(static_cast<char>(text[i + 1]));
            current.push_back(static_cast<char>(text[i + 2]));
            i += 2;
        } else {
            current.push_back(static_cast<char>(ch));
        }
        if (ascii_break || cjk_break) {
            auto cleaned = CleanText(current);
            if (!cleaned.empty()) {
                out.push_back(std::move(cleaned));
            }
            current.clear();
        }
    }
    auto cleaned = CleanText(current);
    if (!cleaned.empty()) {
        out.push_back(std::move(cleaned));
    }
    return out;
}

bool ContainsAny(std::string_view text, std::initializer_list<std::string_view> needles) {
    for (auto needle : needles) {
        if (text.find(needle) != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

std::string InferSliceKind(std::string_view text) {
    if (ContainsAny(text, {"例题", "示例", "案例", "example", "case"})) {
        return "example";
    }
    if (ContainsAny(text, {"练习", "习题", "作业", "检测", "practice", "exercise", "quiz"})) {
        return "practice";
    }
    if (ContainsAny(text, {"目标", "重点", "难点", "objective", "goal", "aim"})) {
        return "objective";
    }
    return text.size() <= 80 ? "concept" : "paragraph";
}

std::size_t Utf8SequenceLength(unsigned char ch) noexcept {
    if (ch < 0x80) {
        return 1;
    }
    if ((ch & 0xE0) == 0xC0) {
        return 2;
    }
    if ((ch & 0xF0) == 0xE0) {
        return 3;
    }
    if ((ch & 0xF8) == 0xF0) {
        return 4;
    }
    return 0;
}

bool IsUtf8Continuation(unsigned char ch) noexcept {
    return (ch & 0xC0) == 0x80;
}

std::string Truncate(std::string_view text, std::size_t max_bytes) {
    if (text.size() <= max_bytes) {
        return std::string(text);
    }

    std::size_t end = 0;
    for (std::size_t i = 0; i < text.size() && i < max_bytes;) {
        const auto len = Utf8SequenceLength(static_cast<unsigned char>(text[i]));
        if (len == 0 || i + len > text.size() || i + len > max_bytes) {
            break;
        }

        bool valid = true;
        for (std::size_t j = 1; j < len; ++j) {
            if (!IsUtf8Continuation(static_cast<unsigned char>(text[i + j]))) {
                valid = false;
                break;
            }
        }
        if (!valid) {
            break;
        }

        end = i + len;
        i += len;
    }
    return std::string(text.substr(0, end));
}

ChunkTrunk BuildChunk(const std::vector<DocumentBlock>& blocks,
                      const BlockGroup& group,
                      std::size_t max_slices,
                      std::string source = "local",
                      bool reused = false) {
    const auto& first = blocks[group.front()];
    const auto text = JoinBlockText(blocks, group);
    auto sentences = SplitSentences(text);
    if (sentences.empty() && !text.empty()) {
        sentences.push_back(CleanText(text));
    }

    ChunkTrunk chunk;
    chunk.chunk_id = "chunk-" + std::to_string(first.order);
    chunk.text = text;
    chunk.title = sentences.empty() ? "Unstructured Content" : CleanText(sentences.front());
    chunk.summary = sentences.empty() ? Truncate(text, 220) : Truncate(sentences.front(), 220);
    if (sentences.size() > 1) {
        chunk.summary = Truncate(sentences[0] + " " + sentences[1], 220);
    }
    chunk.order = first.order;
    chunk.page = first.page;
    chunk.slide = first.slide;
    chunk.confidence = 0.5;
    chunk.source = std::move(source);
    chunk.reused = reused;
    chunk.metadata["block_count"] = std::to_string(group.size());
    chunk.block_ids.reserve(group.size());
    for (const auto index : group) {
        chunk.block_ids.push_back(blocks[index].id);
    }

    const auto limit = std::min<std::size_t>(std::max<std::size_t>(1, max_slices), sentences.size());
    chunk.slices.reserve(limit);
    for (std::size_t i = 0; i < limit; ++i) {
        ChunkSlice slice;
        slice.title = CleanText(sentences[i]);
        slice.summary = Truncate(sentences[i], 220);
        slice.text = Truncate(sentences[i], 1000);
        slice.kind = InferSliceKind(sentences[i]);
        slice.confidence = 0.55;
        chunk.slices.push_back(std::move(slice));
    }
    if (chunk.slices.empty() && !text.empty()) {
        ChunkSlice slice;
        slice.title = chunk.title;
        slice.summary = chunk.summary;
        slice.text = Truncate(text, 1000);
        slice.kind = "paragraph";
        slice.confidence = 0.45;
        chunk.slices.push_back(std::move(slice));
    }
    return chunk;
}

std::vector<BlockGroup> CollectChunkGroups(
    const std::vector<DocumentBlock>& blocks,
    const DocumentAnalysisOptions& options,
    const std::shared_ptr<IDocumentEmbeddingProvider>& embedding_provider,
    DocumentChunkBuildMetrics* metrics) {
    std::vector<BlockGroup> groups;
    BlockGroup current;
    std::optional<std::size_t> pending_title;
    EmbeddingVector current_embedding;
    bool has_embedding = false;

    for (std::size_t index = 0; index < blocks.size(); ++index) {
        const auto& block = blocks[index];
        if (!ShouldChunk(block)) {
            if (IsAttachableTitle(block)) {
                if (!current.empty()) {
                    FlushCurrent(groups, current, pending_title, current_embedding, has_embedding);
                }
                pending_title = index;
            } else {
                FlushCurrent(groups, current, pending_title, current_embedding, has_embedding);
            }
            continue;
        }

        auto block_embedding = EmbedBlockText(block.text, embedding_provider, metrics);
        bool attach = current.empty();
        if (!attach && options.enable_embedding_clustering && has_embedding && block_embedding) {
            attach = SimilarityScore(current_embedding, *block_embedding) >=
                     static_cast<float>(options.chunk_similarity_threshold);
        } else if (!attach && !has_embedding) {
            attach = true;
        }
        if (!attach) {
            groups.push_back(std::move(current));
            current = {};
            has_embedding = false;
            current_embedding = {};
        }

        if (current.empty() && pending_title) {
            current.push_back(*pending_title);
            pending_title.reset();
        }
        current.push_back(index);
        if (!block_embedding) {
            continue;
        }
        if (!has_embedding) {
            current_embedding = std::move(*block_embedding);
            has_embedding = true;
        } else {
            for (std::size_t i = 0; i < current_embedding.size(); ++i) {
                current_embedding[i] += (*block_embedding)[i];
            }
            float norm_sq = 0.0f;
            for (const auto value : current_embedding) {
                norm_sq += value * value;
            }
            if (norm_sq > 1e-6f) {
                const float inv_norm = 1.0f / std::sqrt(norm_sq);
                for (auto& value : current_embedding) {
                    value *= inv_norm;
                }
            }
        }
    }
    if (!current.empty()) {
        groups.push_back(std::move(current));
    }
    if (metrics) {
        metrics->group_count = groups.size();
    }
    return groups;
}

std::optional<ChunkSlice> SliceFromJson(const nlohmann::json& item) {
    if (!item.is_object()) {
        return std::nullopt;
    }
    ChunkSlice slice;
    slice.title = item.value("title", "");
    slice.summary = item.value("summary", "");
    slice.text = item.value("text", slice.summary);
    slice.kind = item.value("kind", InferSliceKind(slice.text));
    slice.confidence = item.value("confidence", 0.65);
    if (slice.text.empty()) {
        return std::nullopt;
    }
    if (slice.title.empty()) {
        slice.title = CleanText(slice.text);
    }
    if (slice.summary.empty()) {
        slice.summary = Truncate(slice.text, 220);
    }
    slice.title = CleanText(slice.title);
    slice.summary = Truncate(slice.summary, 220);
    slice.text = Truncate(slice.text, 1000);
    slice.confidence = std::clamp(slice.confidence, 0.0, 1.0);
    return slice;
}

std::optional<nlohmann::json> ParseJsonObject(std::string_view content) {
    try {
        auto parsed = nlohmann::json::parse(content);
        if (parsed.is_object()) {
            return parsed;
        }
    } catch (...) {
    }

    const auto start = content.find('{');
    const auto end = content.rfind('}');
    if (start == std::string_view::npos || end == std::string_view::npos || end <= start) {
        return std::nullopt;
    }
    try {
        auto parsed = nlohmann::json::parse(content.substr(start, end - start + 1));
        if (parsed.is_object()) {
            return parsed;
        }
    } catch (...) {
    }
    return std::nullopt;
}

std::optional<ChunkTrunk> BuildLlmChunk(const std::vector<DocumentBlock>& blocks,
                                        const BlockGroup& group,
                                        const DocumentAnalysisOptions& options,
                                        llm::ILlmClient& llm_client,
                                        std::string* failure_reason = nullptr) {
    auto fallback = BuildChunk(blocks, group, options.max_chunk_slices);
    llm::ChatCompletionRequest request;
    request.model = options.chunk_llm_model;
    request.temperature = 0.1f;
    request.top_p = 1.0f;
    request.max_tokens = options.chunk_llm_max_tokens;
    request.messages = {
        {llm::ChatRole::System,
         "You extract a concise title and summary for a document chunk. Preserve important technical terms, "
         "model names, abbreviations, and English phrases exactly when they appear in the source. "
         "Do not invent content. Return one strict JSON object only: "
         "{\"title\":\"short title\",\"summary\":\"one sentence\",\"slices\":[{\"title\":\"slice title\","
         "\"summary\":\"slice summary\",\"text\":\"source evidence\",\"kind\":\"concept|example|practice|objective|paragraph\","
         "\"confidence\":0.8}]}"},
        {llm::ChatRole::User, Truncate(fallback.text, 6000)}
    };

    auto completed = llm_client.Complete(request);
    if (!completed.ok()) {
        if (failure_reason) {
            *failure_reason = completed.status().message();
        }
        return std::nullopt;
    }

    auto parsed = ParseJsonObject(completed.value().content);
    if (!parsed) {
        if (failure_reason) {
            *failure_reason = "LLM response did not contain a parseable JSON object";
        }
        return std::nullopt;
    }

    fallback.title = CleanText(parsed->value("title", fallback.title));
    fallback.summary = Truncate(parsed->value("summary", fallback.summary), 500);
    std::vector<ChunkSlice> llm_slices;
    if (const auto it = parsed->find("slices"); it != parsed->end() && it->is_array()) {
        for (const auto& item : *it) {
            if (llm_slices.size() >= std::max<std::size_t>(1, options.max_chunk_slices)) {
                break;
            }
            auto slice = SliceFromJson(item);
            if (slice) {
                llm_slices.push_back(std::move(*slice));
            }
        }
    }
    if (!llm_slices.empty()) {
        fallback.slices = std::move(llm_slices);
    } else {
        fallback.metadata["llm_partial"] = "true";
    }
    fallback.source = "llm";
    fallback.confidence = 0.72;
    fallback.metadata["llm_model"] = completed.value().model;
    return fallback;
}

} // namespace

std::vector<ChunkTrunk> BuildLocalChunks(const std::vector<DocumentBlock>& blocks,
                                         const DocumentAnalysisOptions& options,
                                         std::shared_ptr<llm::ILlmClient> llm_client,
                                         std::shared_ptr<IDocumentEmbeddingProvider> embedding_provider,
                                         std::shared_ptr<IDocumentLlmChunkCache> llm_chunk_cache,
                                         std::shared_ptr<semantic_cache::ISemanticCache> document_semantic_cache,
                                         DocumentChunkBuildMetrics* metrics) {
    std::vector<ChunkTrunk> chunks;
    auto groups = CollectChunkGroups(blocks, options, embedding_provider, metrics);
    chunks.reserve(groups.size());
    for (const auto& group : groups) {
        if (options.enable_llm_chunk_fallback && llm_client && !GroupHasExplicitTitle(blocks, group)) {
            const auto text = JoinBlockText(blocks, group);
            const auto cache_key = MakeLlmCacheKey(options.chunk_llm_model, text);
            if (llm_chunk_cache) {
                if (metrics) {
                    ++metrics->llm_cache_lookup_count;
                }
                const auto cache_started = Clock::now();
                auto cached = llm_chunk_cache->Lookup(cache_key);
                if (metrics) {
                    metrics->llm_cache_lookup_ms += SinceMs(cache_started);
                }
                if (cached.ok()) {
                    if (metrics) {
                        ++metrics->llm_cache_hit_count;
                    }
                    auto chunk = std::move(cached).value();
                    StampReusedChunk(chunk, blocks, group, text);
                    chunk.metadata["llm_cache_hit"] = "true";
                    chunk.metadata["llm_cache_key"] = cache_key.text_hash;
                    chunks.push_back(std::move(chunk));
                    continue;
                }
            }
            if (document_semantic_cache) {
                if (metrics) {
                    ++metrics->semantic_cache_lookup_count;
                }
                semantic_cache::CacheLookupRequest lookup;
                lookup.text = text;
                lookup.scope = semantic_cache::CacheScope::Global;
                lookup.answer_type = semantic_cache::AnswerType::Generic;
                lookup.topic = "document_chunk_title";
                lookup.extra["prompt_version"] = std::string(kChunkPromptVersion);
                lookup.extra["model_version"] = options.chunk_llm_model;
                lookup.extra["payload_type"] = "document_chunk";
                const auto semantic_started = Clock::now();
                auto semantic_hit = document_semantic_cache->Lookup(lookup);
                if (metrics) {
                    metrics->semantic_cache_lookup_ms += SinceMs(semantic_started);
                }
                if (semantic_hit.ok() && semantic_hit.value().hit) {
                    auto chunk = ChunkFromSemanticPayload(semantic_hit.value().payload);
                    if (chunk) {
                        if (metrics) {
                            ++metrics->semantic_cache_hit_count;
                        }
                        StampReusedChunk(*chunk, blocks, group, text);
                        chunk->metadata["llm_cache_hit"] = "false";
                        chunk->metadata["semantic_cache_hit"] = "true";
                        chunk->metadata["semantic_cache_score"] = std::to_string(semantic_hit.value().similarity_score);
                        chunk->metadata["llm_cache_key"] = cache_key.text_hash;
                        chunks.push_back(std::move(*chunk));
                        continue;
                    }
                }
            }
            std::string failure_reason;
            if (metrics) {
                ++metrics->llm_direct_count;
            }
            const auto llm_started = Clock::now();
            auto llm_chunk = BuildLlmChunk(blocks, group, options, *llm_client, &failure_reason);
            if (metrics) {
                metrics->llm_direct_ms += SinceMs(llm_started);
            }
            if (llm_chunk) {
                llm_chunk->metadata["llm_cache_hit"] = "false";
                llm_chunk->metadata["llm_cache_key"] = cache_key.text_hash;
                if (llm_chunk_cache) {
                    if (metrics) {
                        ++metrics->llm_cache_store_count;
                    }
                    const auto store_started = Clock::now();
                    auto store_status = llm_chunk_cache->Store(cache_key, *llm_chunk);
                    if (metrics) {
                        metrics->llm_cache_store_ms += SinceMs(store_started);
                    }
                    if (!store_status.ok()) {
                        llm_chunk->metadata["llm_cache_store_error"] = store_status.message();
                    }
                }
                if (document_semantic_cache) {
                    semantic_cache::CacheStoreRequest store;
                    store.origin.text = text;
                    store.origin.scope = semantic_cache::CacheScope::Global;
                    store.origin.answer_type = semantic_cache::AnswerType::Generic;
                    store.origin.topic = "document_chunk_title";
                    store.origin.extra["prompt_version"] = std::string(kChunkPromptVersion);
                    store.origin.extra["model_version"] = options.chunk_llm_model;
                    store.origin.extra["payload_type"] = "document_chunk";
                    store.response_payload = ChunkToSemanticPayload(*llm_chunk);
                    store.answer_type = semantic_cache::AnswerType::Generic;
                    store.quality_score = static_cast<float>(llm_chunk->confidence);
                    if (metrics) {
                        ++metrics->semantic_cache_store_count;
                    }
                    const auto semantic_store_started = Clock::now();
                    auto store_status = document_semantic_cache->Store(store);
                    if (metrics) {
                        metrics->semantic_cache_store_ms += SinceMs(semantic_store_started);
                    }
                    if (!store_status.ok()) {
                        llm_chunk->metadata["semantic_cache_store_error"] = store_status.message();
                    }
                }
                chunks.push_back(std::move(*llm_chunk));
                continue;
            }
            auto local = BuildChunk(blocks, group, options.max_chunk_slices);
            local.metadata["llm_fallback_attempted"] = "true";
            if (!failure_reason.empty()) {
                local.metadata["llm_fallback_error"] = std::move(failure_reason);
            }
            chunks.push_back(std::move(local));
            continue;
        }
        chunks.push_back(BuildChunk(blocks, group, options.max_chunk_slices));
    }
    return chunks;
}

} // namespace agent::document
