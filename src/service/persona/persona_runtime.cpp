#include "persona_runtime.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <string_view>
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

GenerationParams ApplyEmotionAdaptiveGeneration(const GenerationParams& base,
                                                const EmotionAnalysis& emotion) {
    static const std::map<std::string, double> token_weights{
        {"neutral", 0.0125},
        {"joy", 0.02},
        {"excitement", 0.025},
        {"sadness", 0.03},
        {"fear", 0.03},
        {"anger", 0.025},
        {"disgust", 0.02},
        {"surprise", 0.025},
        {"tenderness", 0.025},
        {"curiosity", 0.0375},
    };

    auto adjusted = base;
    const auto weight_it = token_weights.find(emotion.emotion.primary);
    double weight = weight_it == token_weights.end() ? 0.02 : weight_it->second;
    if (emotion.emotion.intensity >= 0.7) {
        weight *= 1.5;
    }

    const int adaptive_tokens = std::max(100, static_cast<int>(base.max_tokens * weight));
    adjusted.max_tokens = std::min(base.max_tokens, std::max(1, adaptive_tokens));
    return adjusted;
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

bool IsProactiveInput(std::string_view input) {
    return input.rfind("[proactive]", 0) == 0 ||
           input.rfind("[proactive_decision]", 0) == 0 ||
           input.rfind("conversation idle for ", 0) == 0 ||
           input.rfind("[system_event]", 0) == 0;
}

double TopProbabilityMargin(const EmotionInfo& emotion) {
    if (emotion.probabilities.size() < 2) {
        return 1.0;
    }
    double first = -1.0;
    double second = -1.0;
    for (const auto& [_, probability] : emotion.probabilities) {
        if (probability > first) {
            second = first;
            first = probability;
        } else if (probability > second) {
            second = probability;
        }
    }
    if (first < 0.0 || second < 0.0) {
        return 1.0;
    }
    return first - second;
}

std::optional<std::string> EmotionCalibrationReason(const EmotionAnalysis& emotion,
                                                    const EmotionCalibrationOptions& options) {
    if (emotion.emotion.primary_prob < options.low_confidence_threshold) {
        return "low_confidence";
    }
    const double margin = TopProbabilityMargin(emotion.emotion);
    if (margin < options.top_margin_threshold) {
        return "low_top_margin";
    }
    return std::nullopt;
}

bool ContainsVisionObserve(const ToolMemoryContext& context) {
    for (const auto& hit : context.hits) {
        if (hit.tool_id == "vision.observe") {
            return true;
        }
    }
    return context.prompt_block.find("vision.observe") != std::string::npos;
}

std::string SkillStateName(SkillSessionState state) {
    switch (state) {
    case SkillSessionState::Idle:
        return "idle";
    case SkillSessionState::Starting:
        return "starting";
    case SkillSessionState::Ready:
        return "ready";
    case SkillSessionState::Running:
        return "running";
    case SkillSessionState::WaitingInput:
        return "waiting_input";
    case SkillSessionState::Closing:
        return "closing";
    case SkillSessionState::Closed:
        return "closed";
    case SkillSessionState::Failed:
        return "failed";
    case SkillSessionState::Expired:
        return "expired";
    }
    return "unknown";
}

std::string FormatSkillPromptBlock(const SkillSessionSnapshot& snapshot) {
    if (!snapshot.last_observation.empty() && snapshot.state == SkillSessionState::Running) {
        return "<skill_observation skill=\"" + snapshot.skill_id + "\">\n"
            "summary: " + snapshot.last_observation + "\n"
            "source: " + snapshot.skill_id + "\n"
            "注意：这是工具的不确定观察，不是绝对事实。\n"
            "</skill_observation>";
    }
    std::string out = "<skill_status skill=\"" + snapshot.skill_id + "\" state=\"" +
        SkillStateName(snapshot.state) + "\">\n";
    out += snapshot.status_text.empty() ? "Skill session status updated." : snapshot.status_text;
    if (!snapshot.last_error.empty()) {
        out += "\nerror: " + snapshot.last_error;
    }
    out += "\n</skill_status>";
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
    store.origin.persona_id = turn.persona_id;
    store.origin.extra["emotion"] = turn.emotion;
    store.origin.extra["intensity"] = std::to_string(turn.intensity);
    store.origin.extra["behavior"] = turn.behavior;
    store.origin.extra["tone"] = turn.tone;
    store.origin.extra["context_id"] = turn.context_id;
    if (turn.valence) {
        store.origin.extra["valence"] = std::to_string(*turn.valence);
    }
    if (turn.arousal) {
        store.origin.extra["arousal"] = std::to_string(*turn.arousal);
    }
    store.origin.extra["payload_type"] = "conversation_turn";
    store.response_payload = turn.response;
    store.answer_type = semantic_cache::AnswerType::Personalized;
    return l0_cache_->Store(store);
}

core::Result<EmotionAnalysis> NeutralEmotionAnalyzer::Analyze(std::string_view,
                                                              std::string_view,
                                                              std::shared_ptr<const PersonalityConfig>) {
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
                               std::shared_ptr<IAnswerCacheProvider> answer_cache_provider,
                               std::shared_ptr<IToolMemoryProvider> tool_memory_provider,
                               std::shared_ptr<ISkillSessionManager> skill_session_manager,
                               core::LoggerAdapter logger,
                               std::shared_ptr<IEmotionCalibrationSampleSink> emotion_calibration_sink,
                               std::shared_ptr<llm::IAsyncLlmClient> async_llm_client)
    : sessions_(sessions),
      memory_provider_(std::move(memory_provider)),
      emotion_analyzer_(std::move(emotion_analyzer)),
      llm_client_(std::move(llm_client)),
      async_llm_client_(std::move(async_llm_client)),
      answer_cache_provider_(std::move(answer_cache_provider)),
      tool_memory_provider_(std::move(tool_memory_provider)),
      skill_session_manager_(std::move(skill_session_manager)),
      emotion_calibration_sink_(std::move(emotion_calibration_sink)),
      options_(std::move(options)),
      logger_(std::move(logger)) {}

PersonaRuntime::~PersonaRuntime() {
    Shutdown();
}

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
    if (async_llm_client_) {
        std::lock_guard lock(async_operations_mutex_);
        if (async_stopping_) {
            return core::Status::Error(core::ErrorCode::Cancelled,
                                       "persona runtime is shutting down");
        }
    }

    const auto trace_id = request.trace_id;
    const auto submitted_at = std::chrono::steady_clock::now();
    auto response_holder = std::make_shared<std::optional<ChatResponse>>();
    DispatchOptions dispatch;
    dispatch.session_id = request.session_id;
    dispatch.trace_id = request.trace_id;
    dispatch.module = "persona_runtime";
    dispatch.operation = "chat_turn";
    if (async_llm_client_) {
        return sessions_.SubmitTurnAsync(
            std::move(dispatch),
            [this, request = std::move(request), submitted_at, response_holder](
                const SessionTurnSnapshot& snapshot,
                core::ThreadPoolContext&,
                SessionManager::SessionTurnAsyncFinish finish) mutable -> core::Status {
                auto session = snapshot.state;
                const auto user_input = request.user_input;
                const auto context_id = request.context_id;
                const auto persona_id = session.persona_id;
                const auto compute_started_at = std::chrono::steady_clock::now();
                auto prepared = PrepareChat(session, std::move(request));
                if (!prepared.ok()) {
                    return prepared.status();
                }
                prepared.value().started_at = submitted_at;
                prepared.value().latency.compute_queue_wait = Since(submitted_at);
                prepared.value().latency.compute_stage = Since(compute_started_at);
                prepared.value().io_submitted_at = std::chrono::steady_clock::now();

                return CompleteWithLlmAsync(
                    std::move(session),
                    std::move(prepared).value(),
                    [response_holder, user_input, context_id, persona_id, finish = std::move(finish)](
                        core::Result<CompletedChat> completed) mutable {
                        if (!completed.ok()) {
                            finish(completed.status());
                            return;
                        }
                        auto value = std::move(completed).value();
                        SessionTurnCommit commit;
                        commit.trace_id = value.response.trace_id;
                        commit.turn.user_input = user_input;
                        commit.turn.emotion = value.response.user_emotion.emotion.primary;
                        commit.turn.intensity = value.response.user_emotion.emotion.intensity;
                        commit.turn.behavior = value.response.user_emotion.behavior;
                        commit.turn.tone = value.response.user_emotion.tone;
                        commit.turn.response = value.response.response;
                        commit.turn.context_id = context_id;
                        commit.turn.persona_id = persona_id;
                        commit.turn.valence = value.emotion_state.state().valence;
                        commit.turn.arousal = value.emotion_state.state().arousal;
                        commit.emotion_state = std::move(value.emotion_state);
                        commit.latency = value.response.latency.total;
                        *response_holder = std::move(value.response);
                        finish(std::move(commit));
                    });
            },
            [response_holder, callback = std::move(callback)](
                core::Result<SessionTurnReceipt> receipt) mutable {
                if (!receipt.ok()) {
                    callback(receipt.status());
                    return;
                }
                if (!response_holder->has_value()) {
                    callback(core::Status::Error(core::ErrorCode::InternalError,
                                                 "async session turn response is missing"));
                    return;
                }
                response_holder->value().turn_index = receipt.value().turn_index;
                callback(std::move(response_holder->value()));
            });
    }
    return sessions_.SubmitTurn(
        std::move(dispatch),
        [this, request = std::move(request), submitted_at, response_holder](
            const SessionTurnSnapshot& snapshot,
            SessionTurnCommit& commit,
            core::ThreadPoolContext&) mutable -> core::Status {
            auto session = snapshot.state;
            const auto user_input = request.user_input;
            const auto context_id = request.context_id;
            const auto compute_started_at = std::chrono::steady_clock::now();
            auto prepared = PrepareChat(session, std::move(request));
            if (!prepared.ok()) {
                return prepared.status();
            }
            prepared.value().started_at = submitted_at;
            prepared.value().latency.compute_queue_wait = Since(submitted_at);
            prepared.value().latency.compute_stage = Since(compute_started_at);
            prepared.value().io_submitted_at = std::chrono::steady_clock::now();

            auto completed = CompleteWithLlm(session, std::move(prepared).value());
            if (!completed.ok()) {
                return completed.status();
            }
            commit.trace_id = completed.value().trace_id;
            commit.turn.user_input = user_input;
            commit.turn.emotion = completed.value().user_emotion.emotion.primary;
            commit.turn.intensity = completed.value().user_emotion.emotion.intensity;
            commit.turn.behavior = completed.value().user_emotion.behavior;
            commit.turn.tone = completed.value().user_emotion.tone;
            commit.turn.response = completed.value().response;
            commit.turn.context_id = context_id;
            commit.turn.persona_id = session.persona_id;
            commit.turn.valence = session.emotion_state.state().valence;
            commit.turn.arousal = session.emotion_state.state().arousal;
            commit.emotion_state = session.emotion_state;
            commit.latency = completed.value().latency.total;
            *response_holder = std::move(completed).value();
            return core::Status::Ok();
        },
        [response_holder, callback = std::move(callback)](
            core::Result<SessionTurnReceipt> receipt) mutable {
            if (!receipt.ok()) {
                callback(receipt.status());
                return;
            }
            if (!response_holder->has_value()) {
                callback(core::Status::Error(core::ErrorCode::InternalError,
                                             "session turn response is missing"));
                return;
            }
            response_holder->value().turn_index = receipt.value().turn_index;
            callback(std::move(response_holder->value()));
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
    logger_.info("[trace={}] [persona_runtime] memory context start session={} user={} recent={}",
                 prepared.request.trace_id,
                 session.session_id,
                 session.user_uuid,
                 recent_copy.size());
    auto memory = memory_provider_->BuildContext(memory_req);
    if (!memory.ok()) {
        logger_.warn("[trace={}] [persona_runtime] memory context failed session={} code={} reason={}",
                     prepared.request.trace_id,
                     session.session_id,
                     static_cast<int>(memory.status().code()),
                     memory.status().message());
        return memory.status();
    }
    prepared.memory = std::move(memory).value();
    bool vision_tool_triggered = false;
    if (tool_memory_provider_) {
        ToolMemoryQuery tool_query;
        tool_query.session_id = session.session_id;
        tool_query.user_uuid = session.user_uuid;
        tool_query.persona_id = session.persona_id;
        tool_query.trace_id = prepared.request.trace_id;
        tool_query.query = prepared.request.user_input;
        auto tool_context = tool_memory_provider_->Query(tool_query);
        if (!tool_context.ok()) {
            logger_.warn("[trace={}] [persona_runtime] L4 tool memory skipped session={} reason={}",
                         prepared.request.trace_id,
                         session.session_id,
                         tool_context.status().message());
        } else if (tool_context.value().hit && !tool_context.value().prompt_block.empty()) {
            vision_tool_triggered = ContainsVisionObserve(tool_context.value());
            if (!prepared.memory.system_context.empty()) {
                prepared.memory.system_context += "\n";
            }
            prepared.memory.system_context += tool_context.value().prompt_block;
            prepared.memory.l4_hit = true;
        }
    }
    if (skill_session_manager_) {
        if (vision_tool_triggered) {
            SkillSessionStartRequest skill_start;
            skill_start.skill_id = "vision.observe";
            skill_start.session_id = session.session_id;
            skill_start.user_uuid = session.user_uuid;
            skill_start.persona_id = session.persona_id;
            skill_start.trace_id = prepared.request.trace_id;
            skill_start.source = "l4";
            skill_start.reason = "用户输入触发视觉 Skill";
            auto started_skill = skill_session_manager_->Start(skill_start);
            if (!started_skill.ok()) {
                logger_.warn("[trace={}] [persona_runtime] skill session start skipped session={} reason={}",
                             prepared.request.trace_id,
                             session.session_id,
                             started_skill.status().message());
            }
        }
        auto vision_session = skill_session_manager_->Get(session.session_id, "vision.observe");
        if (!vision_session.ok()) {
            logger_.warn("[trace={}] [persona_runtime] skill session query skipped session={} reason={}",
                         prepared.request.trace_id,
                         session.session_id,
                         vision_session.status().message());
        } else if (vision_session.value().has_value()) {
            if (!prepared.memory.system_context.empty()) {
                prepared.memory.system_context += "\n";
            }
            prepared.memory.system_context += FormatSkillPromptBlock(*vision_session.value());
        }
    }
    prepared.latency.memory_context = Since(memory_start);
    logger_.info("[trace={}] [persona_runtime] memory context done session={} latency_ms={} l0_hit={} l3_hit={} l4_hit={}",
                 prepared.request.trace_id,
                 session.session_id,
                 prepared.latency.memory_context.count(),
                 prepared.memory.l0_hit,
                 prepared.memory.l3_hit,
                 prepared.memory.l4_hit);

    auto emotion = emotion_analyzer_->Analyze(prepared.request.user_input,
                                              prepared.request.trace_id,
                                              session.personality);
    if (!emotion.ok()) {
        logger_.warn("[trace={}] [persona_runtime] user emotion failed session={} code={} reason={}",
                     prepared.request.trace_id,
                     session.session_id,
                     static_cast<int>(emotion.status().code()),
                     emotion.status().message());
        return emotion.status();
    }
    prepared.user_emotion = std::move(emotion).value();
    logger_.info("[trace={}] [persona_runtime] user emotion done session={}",
                 prepared.request.trace_id,
                 session.session_id);

    auto calibration_status = MaybeRecordEmotionCalibrationSample(
        session,
        prepared.request,
        prepared.user_emotion,
        recent_copy);
    if (!calibration_status.ok()) {
        logger_.warn("[trace={}] [persona_runtime] emotion calibration sample skipped session={} reason={}",
                     prepared.request.trace_id,
                     session.session_id,
                     calibration_status.message());
    }

    auto adjusted = ApplyEmotionAdaptiveGeneration(prepared.request.base_generation, prepared.user_emotion);
    adjusted = session.emotion_state.GetParamAdjustments(adjusted);
    prepared.generation = adjusted;
    auto hint = session.emotion_state.GetPromptHint();

    const auto prompt_start = std::chrono::steady_clock::now();
    auto messages = BuildMessages(session,
                                  prepared.memory,
                                  prepared.user_emotion,
                                  prepared.request.user_input,
                                  hint);
    if (!messages.ok()) {
        logger_.warn("[trace={}] [persona_runtime] prompt build failed session={} code={} reason={}",
                     prepared.request.trace_id,
                     session.session_id,
                     static_cast<int>(messages.status().code()),
                     messages.status().message());
        return messages.status();
    }
    prepared.messages = std::move(messages).value();
    prepared.latency.prompt_build = Since(prompt_start);
    logger_.info("[trace={}] [persona_runtime] prompt build done session={} latency_ms={} messages={}",
                 prepared.request.trace_id,
                 session.session_id,
                 prepared.latency.prompt_build.count(),
                 prepared.messages.size());

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

    auto system_prompt = std::move(system).value();
    const bool proactive = IsProactiveInput(user_input);
    if (proactive) {
        system_prompt += "\n<proactive_trigger>";
        system_prompt.append(user_input);
        system_prompt += "</proactive_trigger>";
        system_prompt += "\n你可以主动找话题聊，或者接上之前的对话继续说。如果实在没什么好说的，回复空字符串即可。";
    }

    std::vector<llm::ChatMessage> messages;
    messages.reserve(2 + memory.recent_turns.size() * 2);
    messages.push_back(llm::ChatMessage{llm::ChatRole::System, std::move(system_prompt)});
    for (const auto& turn : memory.recent_turns) {
        if (!turn.user_input.empty()) {
            messages.push_back(llm::ChatMessage{llm::ChatRole::User, turn.user_input});
        }
        if (!turn.response.empty()) {
            messages.push_back(llm::ChatMessage{llm::ChatRole::Assistant, turn.response});
        }
    }
    messages.push_back(llm::ChatMessage{llm::ChatRole::User, proactive ? std::string("...") : std::string(user_input)});
    return messages;
}

core::Status PersonaRuntime::MaybeRecordEmotionCalibrationSample(
    const SessionState& session,
    const ChatRequest& request,
    const EmotionAnalysis& emotion,
    const std::vector<ConversationTurn>& recent_turns) const {
    if (!options_.emotion_calibration.enabled || !emotion_calibration_sink_) {
        return core::Status::Ok();
    }
    if (request.user_input.size() < options_.emotion_calibration.min_text_length) {
        return core::Status::Ok();
    }
    if (!std::isfinite(emotion.emotion.primary_prob)) {
        return core::Status::Ok();
    }
    auto reason = EmotionCalibrationReason(emotion, options_.emotion_calibration);
    if (!reason) {
        return core::Status::Ok();
    }

    EmotionCalibrationSample sample;
    sample.trace_id = request.trace_id;
    sample.session_id = session.session_id;
    sample.user_uuid = session.user_uuid;
    sample.persona_id = session.persona_id;
    sample.text = request.user_input;
    sample.bert_result = emotion;
    sample.state_snapshot = session.emotion_state.state();
    sample.recent_turns = recent_turns;
    sample.reason = std::move(*reason);
    return emotion_calibration_sink_->Record(sample);
}

core::Result<ChatResponse> PersonaRuntime::CompleteWithLlm(SessionState& session,
                                                            PreparedChat prepared) {
    const auto io_stage_start = std::chrono::steady_clock::now();
    std::optional<AnswerCacheLookupRequest> answer_cache_lookup;
    std::optional<AnswerCacheLookupResult> answer_cache_hit;
    if (answer_cache_provider_) {
        const auto cache_start = std::chrono::steady_clock::now();
        prepared.answer_cache.enabled = true;
        AnswerCacheLookupRequest lookup;
        lookup.session_id = prepared.request.session_id;
        lookup.user_uuid = session.user_uuid;
        lookup.persona_id = session.persona_id;
        lookup.trace_id = prepared.request.trace_id;
        lookup.query = prepared.request.user_input;
        lookup.model = prepared.request.model.empty() ? options_.default_model : prepared.request.model;
        lookup.generation = prepared.generation;
        lookup.messages = prepared.messages;

        auto cache_result = answer_cache_provider_->Lookup(lookup);
        prepared.latency.answer_cache = Since(cache_start);
        if (!cache_result.ok()) {
            return cache_result.status();
        }
        if (cache_result.value().hit) {
            prepared.answer_cache.hit = true;
            prepared.answer_cache.source = cache_result.value().source;
            prepared.answer_cache.cache_key = cache_result.value().cache_key;
            prepared.answer_cache.similarity_score = cache_result.value().similarity_score;
            answer_cache_hit = std::move(cache_result).value();
        } else {
            answer_cache_lookup = std::move(lookup);
        }
    }

    const auto llm_start = std::chrono::steady_clock::now();
    llm::ChatCompletionResponse completion;
    if (answer_cache_hit) {
        completion.content = answer_cache_hit->response;
        completion.model = prepared.request.model.empty() ? options_.default_model : prepared.request.model;
        prepared.latency.llm_total = std::chrono::milliseconds{0};
    } else {
        llm::ChatCompletionRequest llm_req;
        llm_req.model = prepared.request.model.empty() ? options_.default_model : prepared.request.model;
        llm_req.messages = prepared.messages;
        llm_req.temperature = static_cast<float>(prepared.generation.temperature);
        llm_req.max_tokens = prepared.generation.max_tokens;
        llm_req.top_p = static_cast<float>(prepared.generation.top_p);

        auto llm_result = llm_client_->Complete(llm_req);
        prepared.latency.llm_total = Since(llm_start);
        if (!llm_result.ok()) {
            return llm_result.status();
        }
        completion = std::move(llm_result).value();
    }
    auto finalized = FinalizeLlmCompletion(
        session,
        std::move(prepared),
        std::move(completion),
        std::move(answer_cache_lookup),
        io_stage_start);
    if (!finalized.ok()) {
        return finalized.status();
    }
    session.emotion_state = finalized.value().emotion_state;
    return std::move(finalized).value().response;
}

core::Status PersonaRuntime::CompleteWithLlmAsync(
    SessionState session,
    PreparedChat prepared,
    std::function<void(core::Result<CompletedChat>)> completion) {
    if (!completion) {
        return core::Status::Error(core::ErrorCode::InvalidArgument,
                                   "async chat completion callback is required");
    }
    const auto io_stage_start = std::chrono::steady_clock::now();
    std::optional<AnswerCacheLookupRequest> answer_cache_lookup;
    if (answer_cache_provider_) {
        const auto cache_start = std::chrono::steady_clock::now();
        prepared.answer_cache.enabled = true;
        AnswerCacheLookupRequest lookup;
        lookup.session_id = prepared.request.session_id;
        lookup.user_uuid = session.user_uuid;
        lookup.persona_id = session.persona_id;
        lookup.trace_id = prepared.request.trace_id;
        lookup.query = prepared.request.user_input;
        lookup.model = prepared.request.model.empty() ? options_.default_model : prepared.request.model;
        lookup.generation = prepared.generation;
        lookup.messages = prepared.messages;

        auto cache_result = answer_cache_provider_->Lookup(lookup);
        prepared.latency.answer_cache = Since(cache_start);
        if (!cache_result.ok()) {
            return cache_result.status();
        }
        if (cache_result.value().hit) {
            prepared.answer_cache.hit = true;
            prepared.answer_cache.source = cache_result.value().source;
            prepared.answer_cache.cache_key = cache_result.value().cache_key;
            prepared.answer_cache.similarity_score = cache_result.value().similarity_score;
            llm::ChatCompletionResponse cached;
            cached.content = cache_result.value().response;
            cached.model = lookup.model;
            prepared.latency.llm_total = std::chrono::milliseconds{0};
            completion(FinalizeLlmCompletion(
                session, std::move(prepared), std::move(cached), std::nullopt, io_stage_start));
            return core::Status::Ok();
        }
        answer_cache_lookup = std::move(lookup);
    }

    llm::ChatCompletionRequest request;
    request.model = prepared.request.model.empty() ? options_.default_model : prepared.request.model;
    request.messages = prepared.messages;
    request.temperature = static_cast<float>(prepared.generation.temperature);
    request.max_tokens = prepared.generation.max_tokens;
    request.top_p = static_cast<float>(prepared.generation.top_p);
    const auto llm_started_at = std::chrono::steady_clock::now();
    std::uint64_t operation_id = 0;
    {
        std::lock_guard lock(async_operations_mutex_);
        if (async_stopping_) {
            return core::Status::Error(core::ErrorCode::Cancelled,
                                       "persona runtime is shutting down");
        }
        operation_id = next_async_operation_id_++;
        async_operations_.emplace(operation_id, nullptr);
    }
    auto submitted = async_llm_client_->CompleteAsync(
        std::move(request),
        [this, session = std::move(session), prepared = std::move(prepared),
         answer_cache_lookup = std::move(answer_cache_lookup), io_stage_start, llm_started_at,
         operation_id,
         completion = std::move(completion)](
            core::Result<llm::ChatCompletionResponse> result) mutable {
            struct OperationCleanup {
                PersonaRuntime* runtime;
                std::uint64_t id;
                ~OperationCleanup() {
                    {
                        std::lock_guard lock(runtime->async_operations_mutex_);
                        runtime->async_operations_.erase(id);
                    }
                    runtime->async_operations_drained_.notify_all();
                }
            } cleanup{this, operation_id};
            prepared.latency.llm_total = Since(llm_started_at);
            if (!result.ok()) {
                completion(result.status());
                return;
            }
            completion(FinalizeLlmCompletion(
                session,
                std::move(prepared),
                std::move(result).value(),
                std::move(answer_cache_lookup),
                io_stage_start));
        });
    if (!submitted.ok()) {
        {
            std::lock_guard lock(async_operations_mutex_);
            async_operations_.erase(operation_id);
        }
        async_operations_drained_.notify_all();
        return submitted.status();
    }

    bool cancel = false;
    {
        std::lock_guard lock(async_operations_mutex_);
        auto operation = async_operations_.find(operation_id);
        if (operation == async_operations_.end()) {
            // callback 允许在 CompleteAsync 返回前同步完成，此时无需再保存句柄。
            return core::Status::Ok();
        }
        operation->second = submitted.value();
        cancel = async_stopping_;
    }
    if (cancel) {
        submitted.value()->Cancel();
    }
    return core::Status::Ok();
}

void PersonaRuntime::Shutdown() noexcept {
    std::vector<std::shared_ptr<llm::IAsyncLlmOperation>> operations;
    {
        std::lock_guard lock(async_operations_mutex_);
        if (async_stopping_ && async_operations_.empty()) {
            return;
        }
        async_stopping_ = true;
        operations.reserve(async_operations_.size());
        for (const auto& [_, operation] : async_operations_) {
            if (operation) {
                operations.push_back(operation);
            }
        }
    }
    for (const auto& operation : operations) {
        operation->Cancel();
    }
    std::unique_lock lock(async_operations_mutex_);
    async_operations_drained_.wait(lock, [this] { return async_operations_.empty(); });
}

core::Result<PersonaRuntime::CompletedChat> PersonaRuntime::FinalizeLlmCompletion(
    SessionState& session,
    PreparedChat prepared,
    llm::ChatCompletionResponse completion,
    std::optional<AnswerCacheLookupRequest> answer_cache_lookup,
    std::chrono::steady_clock::time_point io_stage_start) {
    if (answer_cache_provider_ && answer_cache_lookup) {
        AnswerCacheStoreRequest store;
        store.lookup = std::move(*answer_cache_lookup);
        store.response = completion.content;
        auto store_status = answer_cache_provider_->Store(store);
        if (!store_status.ok()) {
            return store_status;
        }
    }
    prepared.latency.total = Since(prepared.started_at);
    prepared.latency.io_stage = Since(io_stage_start);

    auto ai_emotion = emotion_analyzer_->Analyze(completion.content,
                                                 prepared.request.trace_id,
                                                 session.personality);
    if (!ai_emotion.ok()) {
        return ai_emotion.status();
    }

    ConversationTurn turn;
    turn.user_input = prepared.request.user_input;
    turn.emotion = prepared.user_emotion.emotion.primary;
    turn.intensity = prepared.user_emotion.emotion.intensity;
    turn.behavior = prepared.user_emotion.behavior;
    turn.tone = prepared.user_emotion.tone;
    turn.response = completion.content;
    turn.context_id = prepared.request.context_id;
    turn.persona_id = session.persona_id;

    auto state_update = session.emotion_state.Update(
        prepared.user_emotion.emotion.primary,
        prepared.user_emotion.emotion.intensity,
        ai_emotion.value().emotion.primary,
        ai_emotion.value().emotion.intensity);
    if (!state_update.ok()) {
        return state_update.status();
    }
    turn.valence = state_update.value().valence;
    turn.arousal = state_update.value().arousal;

    auto admit_status = memory_provider_->AdmitTurn(
        prepared.request.session_id,
        session.user_uuid,
        turn,
        prepared.request.trace_id);
    if (!admit_status.ok()) {
        return admit_status;
    }

    ChatResponse response;
    response.session_id = prepared.request.session_id;
    response.trace_id = prepared.request.trace_id;
    response.response = completion.content;
    response.user_emotion = prepared.user_emotion;
    response.ai_emotion = std::move(ai_emotion).value();
    response.l0_hit = prepared.memory.l0_hit;
    response.l3_hit = prepared.memory.l3_hit;
    response.l4_hit = prepared.memory.l4_hit;
    response.turn_index = 0;
    response.answer_cache = std::move(prepared.answer_cache);
    response.latency = prepared.latency;
    response.messages = std::move(prepared.messages);
    response.latency.callback_to_response = Since(prepared.started_at) - response.latency.total;
    CompletedChat completed;
    completed.response = std::move(response);
    completed.emotion_state = session.emotion_state;
    return completed;
}

} // namespace agent::service::persona
