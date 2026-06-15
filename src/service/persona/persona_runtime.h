#pragma once

#include "isemantic_cache.h"
#include "long_term_memory_compressor.h"
#include "openai_llm_client.h"
#include "session_manager.h"

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <span>

namespace agent::service::persona {

struct RecalledContext {
    std::string system_context;
    std::vector<ConversationTurn> recent_turns;
    bool l0_hit = false;
    bool l3_hit = false;
};

struct MemoryContextRequest {
    std::string session_id;
    std::string user_uuid;
    std::string persona_id;
    std::string query;
    std::string trace_id;
    std::span<const ConversationTurn> current_session_recent;
    std::size_t max_recent_turns = 10;
};

class IMemoryContextProvider {
public:
    virtual ~IMemoryContextProvider() = default;

    virtual core::Result<RecalledContext> BuildContext(const MemoryContextRequest& request) = 0;
    virtual core::Status AdmitTurn(std::string_view session_id,
                                   std::string_view user_uuid,
                                   const ConversationTurn& turn,
                                   std::string_view trace_id) = 0;
};

struct SemanticMemoryContextProviderOptions {
    std::size_t max_recent_turns = 10;
    int l3_top_k = 5;
};

class SemanticMemoryContextProvider final : public IMemoryContextProvider {
public:
    SemanticMemoryContextProvider(
        std::shared_ptr<semantic_cache::ISemanticCache> l0_cache,
        std::shared_ptr<memory::LongTermMemoryCompressor> l3_memory = nullptr,
        SemanticMemoryContextProviderOptions options = {});

    core::Result<RecalledContext> BuildContext(const MemoryContextRequest& request) override;
    core::Status AdmitTurn(std::string_view session_id,
                           std::string_view user_uuid,
                           const ConversationTurn& turn,
                           std::string_view trace_id) override;

private:
    std::shared_ptr<semantic_cache::ISemanticCache> l0_cache_;
    std::shared_ptr<memory::LongTermMemoryCompressor> l3_memory_;
    SemanticMemoryContextProviderOptions options_;
};

class IEmotionAnalyzer {
public:
    virtual ~IEmotionAnalyzer() = default;
    virtual core::Result<EmotionAnalysis> Analyze(std::string_view text,
                                                  std::string_view trace_id,
                                                  std::shared_ptr<const PersonalityConfig> personality = nullptr) = 0;
};

class NeutralEmotionAnalyzer final : public IEmotionAnalyzer {
public:
    core::Result<EmotionAnalysis> Analyze(std::string_view text,
                                          std::string_view trace_id,
                                          std::shared_ptr<const PersonalityConfig> personality = nullptr) override;
};

struct EmotionCalibrationOptions {
    bool enabled = false;
    double low_confidence_threshold = 0.45;
    double top_margin_threshold = 0.15;
    std::size_t min_text_length = 8;
};

struct EmotionCalibrationSample {
    std::string trace_id;
    std::string session_id;
    std::string user_uuid;
    std::string persona_id;
    std::string text;
    EmotionAnalysis bert_result;
    EmotionState state_snapshot;
    std::vector<ConversationTurn> recent_turns;
    std::string reason;
};

class IEmotionCalibrationSampleSink {
public:
    virtual ~IEmotionCalibrationSampleSink() = default;
    virtual core::Status Record(const EmotionCalibrationSample& sample) = 0;
};

struct ChatRequest {
    std::string session_id;
    std::string user_input;
    std::string trace_id;
    std::string context_id;
    GenerationParams base_generation{0.7, 1024, 0.9};
    std::string model;
};

struct ChatLatencyBreakdown {
    std::chrono::milliseconds compute_queue_wait{0};
    std::chrono::milliseconds compute_stage{0};
    std::chrono::milliseconds io_queue_wait{0};
    std::chrono::milliseconds io_stage{0};
    std::chrono::milliseconds memory_context{0};
    std::chrono::milliseconds answer_cache{0};
    std::chrono::milliseconds prompt_build{0};
    std::chrono::milliseconds llm_total{0};
    std::chrono::milliseconds callback_to_response{0};
    std::chrono::milliseconds total{0};
};

struct AnswerCacheInfo {
    bool enabled = false;
    bool hit = false;
    bool bypassed = false;
    std::string source = "llm";
    std::string cache_key;
    float similarity_score = 0.0f;
};

struct AnswerCacheLookupRequest {
    std::string session_id;
    std::string user_uuid;
    std::string persona_id;
    std::string trace_id;
    std::string query;
    std::string model;
    GenerationParams generation;
    std::vector<llm::ChatMessage> messages;
};

struct AnswerCacheLookupResult {
    bool hit = false;
    std::string response;
    std::string cache_key;
    std::string source = "semantic_cache";
    float similarity_score = 0.0f;
};

struct AnswerCacheStoreRequest {
    AnswerCacheLookupRequest lookup;
    std::string response;
};

class IAnswerCacheProvider {
public:
    virtual ~IAnswerCacheProvider() = default;

    virtual core::Result<AnswerCacheLookupResult> Lookup(const AnswerCacheLookupRequest& request) = 0;
    virtual core::Status Store(const AnswerCacheStoreRequest& request) = 0;
};

struct ChatResponse {
    std::string session_id;
    std::string trace_id;
    std::string response;
    EmotionAnalysis user_emotion;
    EmotionAnalysis ai_emotion;
    bool l0_hit = false;
    bool l3_hit = false;
    std::uint64_t turn_index = 0;
    AnswerCacheInfo answer_cache;
    ChatLatencyBreakdown latency;
    std::vector<llm::ChatMessage> messages;
};

using ChatCallback = std::function<void(core::Result<ChatResponse>)>;

struct PersonaRuntimeOptions {
    std::size_t recent_raw_turns = 10;
    std::string default_model;
    EmotionCalibrationOptions emotion_calibration;
};

class PersonaRuntime {
public:
    PersonaRuntime(SessionManager& sessions,
                   std::shared_ptr<IMemoryContextProvider> memory_provider,
                   std::shared_ptr<IEmotionAnalyzer> emotion_analyzer,
                   std::shared_ptr<llm::ILlmClient> llm_client,
                   PersonaRuntimeOptions options = {},
                   std::shared_ptr<IAnswerCacheProvider> answer_cache_provider = nullptr,
                   core::LoggerAdapter logger = core::LoggerAdapter::ForModule("service"),
                   std::shared_ptr<IEmotionCalibrationSampleSink> emotion_calibration_sink = nullptr);

    core::Status SubmitChat(ChatRequest request, ChatCallback callback);

private:
    struct PreparedChat {
        ChatRequest request;
        RecalledContext memory;
        EmotionAnalysis user_emotion;
        GenerationParams generation;
        std::vector<llm::ChatMessage> messages;
        AnswerCacheInfo answer_cache;
        std::chrono::steady_clock::time_point started_at;
        std::chrono::steady_clock::time_point io_submitted_at;
        ChatLatencyBreakdown latency;
    };

    core::Result<PreparedChat> PrepareChat(SessionState& session, ChatRequest request);
    core::Result<std::vector<llm::ChatMessage>> BuildMessages(
        SessionState& session,
        const RecalledContext& memory,
        const EmotionAnalysis& emotion,
        std::string_view user_input,
        const std::optional<std::string>& emotion_hint) const;
    core::Status MaybeRecordEmotionCalibrationSample(const SessionState& session,
                                                     const ChatRequest& request,
                                                     const EmotionAnalysis& emotion,
                                                     const std::vector<ConversationTurn>& recent_turns) const;
    void CompleteWithLlm(SessionState& session, PreparedChat prepared, ChatCallback callback);

    SessionManager& sessions_;
    std::shared_ptr<IMemoryContextProvider> memory_provider_;
    std::shared_ptr<IEmotionAnalyzer> emotion_analyzer_;
    std::shared_ptr<llm::ILlmClient> llm_client_;
    std::shared_ptr<IAnswerCacheProvider> answer_cache_provider_;
    std::shared_ptr<IEmotionCalibrationSampleSink> emotion_calibration_sink_;
    PersonaRuntimeOptions options_;
    core::LoggerAdapter logger_;
};

} // namespace agent::service::persona
