#include "persona_runtime.h"

#include <algorithm>
#include <sstream>
#include <utility>

namespace agent::service::persona {
namespace {

std::string FormatRecentTurn(const ConversationTurn& turn) {
    return "用户: " + turn.user_input + "\n助手: " + turn.response;
}

std::string FormatL3Facts(const std::vector<vector_storage::EntryRecord>& facts) {
    if (facts.empty()) {
        return {};
    }
    std::string out = "<memory_l3>\n";
    for (const auto& fact : facts) {
        if (!fact.payload.empty()) {
            out += "- " + fact.payload + "\n";
        }
    }
    out += "</memory_l3>";
    return out;
}

std::chrono::milliseconds Since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
}

std::vector<ConversationTurn> TakeRecent(std::span<const ConversationTurn> turns, std::size_t limit) {
    std::vector<ConversationTurn> out;
    const auto count = std::min<std::size_t>(turns.size(), limit);
    out.reserve(count);
    const auto begin = turns.size() - count;
    for (std::size_t i = begin; i < turns.size(); ++i) {
        out.push_back(turns[i]);
    }
    return out;
}

} // namespace

SemanticMemoryContextProvider::SemanticMemoryContextProvider(
    std::shared_ptr<semantic_cache::ISemanticCache> l0_cache,
    std::shared_ptr<memory::LongTermMemoryCompressor> l3_memory,
    SemanticMemoryContextProviderOptions options)
    : l0_cache_(std::move(l0_cache)),
      l3_memory_(std::move(l3_memory)),
      options_(options) {}

core::Result<RecalledContext> SemanticMemoryContextProvider::BuildContext(
    const MemoryContextRequest& request) {
    RecalledContext context;
    const auto recent_limit = request.max_recent_turns == 0
        ? options_.max_recent_turns
        : request.max_recent_turns;
    context.recent_turns = TakeRecent(request.current_session_recent, recent_limit);

    std::vector<std::string> sections;
    if (l0_cache_) {
        semantic_cache::CacheLookupRequest lookup;
        lookup.text = request.query;
        lookup.scope = semantic_cache::CacheScope::User;
        lookup.answer_type = semantic_cache::AnswerType::Personalized;
        lookup.user_id = request.user_uuid;
        lookup.session_id = request.session_id;
        lookup.persona_id = request.persona_id;
        lookup.recent_turns.reserve(context.recent_turns.size());
        for (const auto& turn : context.recent_turns) {
            lookup.recent_turns.push_back(FormatRecentTurn(turn));
        }

        auto l0 = l0_cache_->Lookup(lookup);
        if (!l0.ok()) {
            return l0.status();
        }
        if (l0.value().hit && !l0.value().payload.empty()) {
            context.l0_hit = true;
            sections.push_back("<memory_l0>\n" + l0.value().payload + "\n</memory_l0>");
        }
    }

    if (l3_memory_) {
        auto facts = l3_memory_->SearchFacts(request.user_uuid, request.query, options_.l3_top_k);
        if (!facts.ok()) {
            return facts.status();
        }
        auto formatted = FormatL3Facts(facts.value());
        if (!formatted.empty()) {
            context.l3_hit = true;
            sections.push_back(std::move(formatted));
        }
    }

    for (std::size_t i = 0; i < sections.size(); ++i) {
        if (i != 0) {
            context.system_context += "\n";
        }
        context.system_context += sections[i];
    }
    return context;
}

core::Status SemanticMemoryContextProvider::AdmitTurn(std::string_view session_id,
                                                      std::string_view user_uuid,
                                                      const ConversationTurn& turn,
                                                      std::string_view) {
    if (!l0_cache_) {
        return core::Status::Ok();
    }

    semantic_cache::CacheStoreRequest store;
    store.origin.text = turn.user_input;
    store.origin.scope = semantic_cache::CacheScope::User;
    store.origin.answer_type = semantic_cache::AnswerType::Personalized;
    store.origin.user_id = std::string(user_uuid);
    store.origin.session_id = std::string(session_id);
    store.response_payload = turn.response;
    store.answer_type = semantic_cache::AnswerType::Personalized;
    return l0_cache_->Store(store);
}

core::Result<EmotionAnalysis> NeutralEmotionAnalyzer::Analyze(std::string_view, std::string_view) {
    EmotionAnalysis analysis;
    analysis.emotion.primary = "neutral";
    analysis.emotion.intensity = 0.0;
    analysis.emotion.primary_prob = 1.0;
    analysis.emotion.probabilities = {{"neutral", 1.0}};
    analysis.behavior = "unknown";
    analysis.tone = "neutral";
    return analysis;
}

PersonaRuntime::PersonaRuntime(SessionManager& sessions,
                               std::shared_ptr<IMemoryContextProvider> memory_provider,
                               std::shared_ptr<IEmotionAnalyzer> emotion_analyzer,
                               std::shared_ptr<llm::ILlmClient> llm_client,
                               PersonaRuntimeOptions options,
                               core::LoggerAdapter logger)
    : sessions_(sessions),
      memory_provider_(std::move(memory_provider)),
      emotion_analyzer_(std::move(emotion_analyzer)),
      llm_client_(std::move(llm_client)),
      options_(std::move(options)),
      logger_(std::move(logger)) {}

core::Status PersonaRuntime::SubmitChat(ChatRequest request, ChatCallback callback) {
    if (!callback) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "chat callback is required");
    }
    if (!memory_provider_ || !emotion_analyzer_ || !llm_client_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "persona runtime dependencies are incomplete");
    }
    if (request.session_id.empty() || request.user_input.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "session_id and user_input are required");
    }
    if (request.trace_id.empty()) {
        request.trace_id = core::GenerateTraceId();
    }

    const auto trace_id = request.trace_id;
    DispatchOptions compute_dispatch;
    compute_dispatch.session_id = request.session_id;
    compute_dispatch.trace_id = request.trace_id;
    compute_dispatch.module = "persona_runtime";
    compute_dispatch.operation = "prepare_chat";
    return sessions_.SubmitCompute(
        std::move(compute_dispatch),
        [this, request = std::move(request), callback = std::move(callback), trace_id](
            SessionState& session,
            core::ThreadPoolContext&) mutable -> core::Status {
            auto prepared = PrepareChat(session, std::move(request));
            if (!prepared.ok()) {
                callback(prepared.status());
                return prepared.status();
            }

            DispatchOptions io_dispatch;
            io_dispatch.session_id = prepared.value().request.session_id;
            io_dispatch.trace_id = trace_id;
            io_dispatch.user_uuid = session.user_uuid;
            io_dispatch.module = "persona_runtime";
            io_dispatch.operation = "llm_complete";
            auto status = sessions_.SubmitIo(
                std::move(io_dispatch),
                [this, prepared = std::move(prepared).value(), callback = std::move(callback)](
                    SessionState& session,
                    core::ThreadPoolContext&) mutable -> core::Status {
                    CompleteWithLlm(session, std::move(prepared), std::move(callback));
                    return core::Status::Ok();
                });
            if (!status.ok()) {
                callback(status);
            }
            return status;
        });
}

core::Result<PersonaRuntime::PreparedChat> PersonaRuntime::PrepareChat(SessionState& session,
                                                                       ChatRequest request) {
    const auto started = std::chrono::steady_clock::now();
    PreparedChat prepared;
    prepared.started_at = started;
    prepared.request = std::move(request);

    const auto memory_start = std::chrono::steady_clock::now();
    std::vector<ConversationTurn> recent_copy(session.recent_history.begin(), session.recent_history.end());
    MemoryContextRequest memory_req;
    memory_req.session_id = session.session_id;
    memory_req.user_uuid = session.user_uuid;
    memory_req.persona_id = session.persona_id;
    memory_req.query = prepared.request.user_input;
    memory_req.trace_id = prepared.request.trace_id;
    memory_req.current_session_recent = std::span<const ConversationTurn>(recent_copy.data(), recent_copy.size());
    memory_req.max_recent_turns = options_.recent_raw_turns;
    auto memory = memory_provider_->BuildContext(memory_req);
    if (!memory.ok()) {
        return memory.status();
    }
    prepared.memory = std::move(memory).value();
    prepared.latency.memory_context = Since(memory_start);

    auto emotion = emotion_analyzer_->Analyze(prepared.request.user_input, prepared.request.trace_id);
    if (!emotion.ok()) {
        return emotion.status();
    }
    prepared.user_emotion = std::move(emotion).value();

    auto adjusted = session.emotion_state.GetParamAdjustments(prepared.request.base_generation);
    prepared.generation = adjusted;
    auto hint = session.emotion_state.GetPromptHint();

    const auto prompt_start = std::chrono::steady_clock::now();
    auto messages = BuildMessages(session,
                                  prepared.memory,
                                  prepared.user_emotion,
                                  prepared.request.user_input,
                                  hint);
    if (!messages.ok()) {
        return messages.status();
    }
    prepared.messages = std::move(messages).value();
    prepared.latency.prompt_build = Since(prompt_start);

    logger_.info("[trace={}] [persona_runtime] prepared chat session={} user={} recent={} l0_hit={} l3_hit={}",
                 prepared.request.trace_id,
                 session.session_id,
                 session.user_uuid,
                 prepared.memory.recent_turns.size(),
                 prepared.memory.l0_hit,
                 prepared.memory.l3_hit);
    return prepared;
}

core::Result<std::vector<llm::ChatMessage>> PersonaRuntime::BuildMessages(
    SessionState& session,
    const RecalledContext& memory,
    const EmotionAnalysis& emotion,
    std::string_view user_input,
    const std::optional<std::string>& emotion_hint) const {
    if (!session.prompt_builder) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "session prompt builder is missing");
    }

    auto system = session.prompt_builder->BuildSystemPrompt(
        memory.system_context,
        emotion,
        emotion_hint);
    if (!system.ok()) {
        return system.status();
    }

    std::vector<llm::ChatMessage> messages;
    messages.reserve(2 + memory.recent_turns.size() * 2);
    messages.push_back(llm::ChatMessage{llm::ChatRole::System, std::move(system).value()});
    for (const auto& turn : memory.recent_turns) {
        if (!turn.user_input.empty()) {
            messages.push_back(llm::ChatMessage{llm::ChatRole::User, turn.user_input});
        }
        if (!turn.response.empty()) {
            messages.push_back(llm::ChatMessage{llm::ChatRole::Assistant, turn.response});
        }
    }
    messages.push_back(llm::ChatMessage{llm::ChatRole::User, std::string(user_input)});
    return messages;
}

void PersonaRuntime::CompleteWithLlm(SessionState& session, PreparedChat prepared, ChatCallback callback) {
    const auto llm_start = std::chrono::steady_clock::now();
    llm::ChatCompletionRequest llm_req;
    llm_req.model = prepared.request.model.empty() ? options_.default_model : prepared.request.model;
    llm_req.messages = prepared.messages;
    llm_req.temperature = static_cast<float>(prepared.generation.temperature);
    llm_req.max_tokens = prepared.generation.max_tokens;
    llm_req.top_p = static_cast<float>(prepared.generation.top_p);

    auto llm_result = llm_client_->Complete(llm_req);
    prepared.latency.llm_total = Since(llm_start);
    prepared.latency.total = Since(prepared.started_at);
    if (!llm_result.ok()) {
        callback(llm_result.status());
        return;
    }

    auto ai_emotion = emotion_analyzer_->Analyze(llm_result.value().content, prepared.request.trace_id);
    if (!ai_emotion.ok()) {
        callback(ai_emotion.status());
        return;
    }

    ConversationTurn turn;
    turn.user_input = prepared.request.user_input;
    turn.emotion = prepared.user_emotion.emotion.primary;
    turn.intensity = prepared.user_emotion.emotion.intensity;
    turn.behavior = prepared.user_emotion.behavior;
    turn.tone = prepared.user_emotion.tone;
    turn.response = llm_result.value().content;
    turn.context_id = prepared.request.context_id;

    auto admit_status = memory_provider_->AdmitTurn(
        prepared.request.session_id,
        session.user_uuid,
        turn,
        prepared.request.trace_id);
    if (!admit_status.ok()) {
        callback(admit_status);
        return;
    }

    auto state_update = session.emotion_state.Update(
        prepared.user_emotion.emotion.primary,
        prepared.user_emotion.emotion.intensity,
        ai_emotion.value().emotion.primary,
        ai_emotion.value().emotion.intensity);
    if (!state_update.ok()) {
        callback(state_update.status());
        return;
    }

    session.last_active = std::chrono::steady_clock::now();
    session.last_trace_id = prepared.request.trace_id;
    session.recent_history.push_back(turn);
    while (session.recent_history.size() > 20) {
        session.recent_history.pop_front();
    }

    ChatResponse response;
    response.session_id = prepared.request.session_id;
    response.trace_id = prepared.request.trace_id;
    response.response = llm_result.value().content;
    response.user_emotion = prepared.user_emotion;
    response.ai_emotion = std::move(ai_emotion).value();
    response.l0_hit = prepared.memory.l0_hit;
    response.l3_hit = prepared.memory.l3_hit;
    response.latency = prepared.latency;
    response.messages = std::move(prepared.messages);
    callback(std::move(response));
}

} // namespace agent::service::persona
