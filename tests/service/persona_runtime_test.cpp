#include "persona_runtime.h"
#include "runtime_maintenance_service.h"
#include "skill_session_manager.h"
#include "skill_vision_event_sink.h"

#include <gtest/gtest.h>

#include <future>
#include <mutex>
#include <thread>

namespace {

using agent::llm::ChatCompletionRequest;
using agent::llm::ChatCompletionResponse;
using agent::llm::ChatMessage;
using agent::semantic_cache::CacheLookupRequest;
using agent::semantic_cache::CacheLookupResult;
using agent::semantic_cache::CacheStoreRequest;
using agent::service::persona::ChatRequest;
using agent::service::persona::ChatResponse;
using agent::service::persona::CreateSessionRequest;
using agent::service::persona::EmotionAnalysis;
using agent::service::persona::EmotionCalibrationSample;
using agent::service::persona::NeutralEmotionAnalyzer;
using agent::service::persona::PersonaRuntime;
using agent::service::persona::PersonaRuntimeOptions;
using agent::service::persona::PersonalityConfig;
using agent::service::persona::SemanticMemoryContextProvider;
using agent::service::persona::SessionManager;
using agent::service::persona::SkillObservation;
using agent::service::persona::SkillSessionManager;
using agent::service::persona::SkillSessionOptions;
using agent::service::persona::SkillSessionStartRequest;
using agent::service::persona::SkillSessionState;
using agent::service::persona::SkillSessionStopRequest;
using agent::service::persona::SkillVisionEventSink;
using agent::service::persona::SkillVisionEventSinkOptions;
using agent::service::persona::ToolMemoryContext;
using agent::service::persona::ToolMemoryQuery;

class FakeSemanticCache final : public agent::semantic_cache::ISemanticCache {
public:
    core::Result<CacheLookupResult> Lookup(const CacheLookupRequest& req) override {
        std::lock_guard lock(mutex_);
        last_lookup = req;
        CacheLookupResult result;
        result.hit = lookup_hit;
        result.payload = lookup_payload;
        result.similarity_score = 0.97f;
        return result;
    }

    core::Status Store(const CacheStoreRequest& req) override {
        std::lock_guard lock(mutex_);
        stores.push_back(req);
        return core::Status::Ok();
    }

    bool lookup_hit = true;
    std::string lookup_payload = "历史筛选上下文";
    CacheLookupRequest last_lookup;
    std::vector<CacheStoreRequest> stores;
    std::mutex mutex_;
};

class FakeLlmClient final : public agent::llm::ILlmClient {
public:
    core::Result<ChatCompletionResponse> Complete(const ChatCompletionRequest& req) override {
        std::lock_guard lock(mutex_);
        last_request = req;
        ChatCompletionResponse response;
        response.content = "这是回复";
        response.total_tokens = 42;
        return response;
    }

    ChatCompletionRequest last_request;
    std::mutex mutex_;
};

class FixedEmotionAnalyzer final : public agent::service::persona::IEmotionAnalyzer {
public:
    explicit FixedEmotionAnalyzer(EmotionAnalysis analysis)
        : analysis_(std::move(analysis)) {}

    core::Result<EmotionAnalysis> Analyze(std::string_view,
                                          std::string_view,
                                          std::shared_ptr<const PersonalityConfig> = nullptr) override {
        return analysis_;
    }

private:
    EmotionAnalysis analysis_;
};

class RecordingEmotionCalibrationSink final : public agent::service::persona::IEmotionCalibrationSampleSink {
public:
    core::Status Record(const EmotionCalibrationSample& sample) override {
        std::lock_guard lock(mutex_);
        samples.push_back(sample);
        return core::Status::Ok();
    }

    std::vector<EmotionCalibrationSample> samples;
    std::mutex mutex_;
};

class FakeToolMemoryProvider final : public agent::service::persona::IToolMemoryProvider {
public:
    core::Result<ToolMemoryContext> Query(const ToolMemoryQuery& request) override {
        std::lock_guard lock(mutex_);
        last_query = request;
        ToolMemoryContext context;
        context.hit = hit;
        context.prompt_block = prompt_block;
        return context;
    }

    bool hit = true;
    std::string prompt_block =
        "<tool_memory_l4>\n"
        "- tool: vision.observe\n"
        "  instruction: use structured visual tool call only when needed\n"
        "</tool_memory_l4>";
    ToolMemoryQuery last_query;
    std::mutex mutex_;
};

CreateSessionRequest MakeSessionRequest() {
    PersonalityConfig personality;
    personality.name = "小橘";
    personality.description = "教育陪伴人格";

    CreateSessionRequest req;
    req.user_uuid = "user-runtime";
    req.persona_id = "persona-main";
    req.session_id = "session-runtime";
    req.trace_id = "trace-create";
    req.personality = std::move(personality);
    req.time_awareness = false;
    req.emotion_state_config.noise_sigma = 0.0;
    return req;
}

EmotionAnalysis MakeEmotion(std::string primary, double intensity) {
    EmotionAnalysis analysis;
    analysis.emotion.primary = std::move(primary);
    analysis.emotion.intensity = intensity;
    analysis.emotion.primary_prob = 1.0;
    analysis.emotion.probabilities = {{analysis.emotion.primary, 1.0}};
    analysis.behavior = "unknown";
    analysis.tone = "neutral";
    return analysis;
}

EmotionAnalysis MakeUncertainEmotion() {
    EmotionAnalysis analysis;
    analysis.emotion.primary = "neutral";
    analysis.emotion.intensity = 0.2;
    analysis.emotion.primary_prob = 0.42;
    analysis.emotion.probabilities = {{"neutral", 0.42}, {"sadness", 0.37}, {"fear", 0.21}};
    analysis.behavior = "unknown";
    analysis.tone = "neutral";
    return analysis;
}

TEST(PersonaRuntimeTest, BuildsMessagesFromL0AndLastTenRawTurns) {
    core::ThreadPool compute({1, 32, "runtime-compute"});
    core::ThreadPool io({1, 32, "runtime-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    SessionManager sessions(compute, io);
    ASSERT_TRUE(sessions.CreateSession(MakeSessionRequest()).ok());

    for (int i = 0; i < 12; ++i) {
        agent::service::persona::ConversationTurn turn;
        turn.user_input = "u" + std::to_string(i);
        turn.response = "a" + std::to_string(i);
        ASSERT_TRUE(sessions.AddTurn("session-runtime", std::move(turn), "trace-seed").ok());
    }

    auto cache = std::make_shared<FakeSemanticCache>();
    auto memory = std::make_shared<SemanticMemoryContextProvider>(cache);
    auto emotion = std::make_shared<NeutralEmotionAnalyzer>();
    auto llm = std::make_shared<FakeLlmClient>();
    PersonaRuntime runtime(
        sessions,
        memory,
        emotion,
        llm,
        PersonaRuntimeOptions{.recent_raw_turns = 10, .default_model = "test-model"});

    std::promise<core::Result<ChatResponse>> promise;
    auto future = promise.get_future();
    ChatRequest chat;
    chat.session_id = "session-runtime";
    chat.user_input = "当前问题";
    chat.trace_id = "trace-chat";
    chat.model = "test-model";
    auto submit = runtime.SubmitChat(
        std::move(chat),
        [&promise](core::Result<ChatResponse> result) mutable {
            promise.set_value(std::move(result));
        });
    ASSERT_TRUE(submit.ok()) << submit.message();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto result = future.get();
    ASSERT_TRUE(result.ok()) << result.status().message();

    ASSERT_EQ(result.value().messages.size(), 22u);
    EXPECT_NE(result.value().messages[0].content.find("<memory_l0>"), std::string::npos);
    EXPECT_NE(result.value().messages[0].content.find("历史筛选上下文"), std::string::npos);
    EXPECT_EQ(result.value().messages[1].content, "u2");
    EXPECT_EQ(result.value().messages[2].content, "a2");
    EXPECT_EQ(result.value().messages[19].content, "u11");
    EXPECT_EQ(result.value().messages[20].content, "a11");
    EXPECT_EQ(result.value().messages[21].content, "当前问题");
    EXPECT_TRUE(result.value().l0_hit);

    {
        std::lock_guard lock(llm->mutex_);
        EXPECT_EQ(llm->last_request.messages.size(), result.value().messages.size());
        EXPECT_EQ(llm->last_request.model, "test-model");
    }
    {
        std::lock_guard lock(cache->mutex_);
        EXPECT_EQ(cache->last_lookup.user_id, "user-runtime");
        EXPECT_EQ(cache->last_lookup.session_id, "session-runtime");
        EXPECT_EQ(cache->stores.size(), 1u);
        EXPECT_EQ(cache->stores[0].origin.user_id, "user-runtime");
        EXPECT_EQ(cache->stores[0].response_payload, "这是回复");
    }

    compute.Shutdown(true);
    io.Shutdown(true);
}

TEST(PersonaRuntimeTest, AppliesEmotionAdaptiveGenerationWithoutBehaviorTonePrompting) {
    core::ThreadPool compute({1, 32, "runtime-compute"});
    core::ThreadPool io({1, 32, "runtime-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    SessionManager sessions(compute, io);
    ASSERT_TRUE(sessions.CreateSession(MakeSessionRequest()).ok());

    auto cache = std::make_shared<FakeSemanticCache>();
    cache->lookup_hit = false;
    auto memory = std::make_shared<SemanticMemoryContextProvider>(cache);
    auto emotion = std::make_shared<FixedEmotionAnalyzer>(MakeEmotion("curiosity", 0.8));
    auto llm = std::make_shared<FakeLlmClient>();
    PersonaRuntime runtime(
        sessions,
        memory,
        emotion,
        llm,
        PersonaRuntimeOptions{.recent_raw_turns = 10, .default_model = "test-model"});

    std::promise<core::Result<ChatResponse>> promise;
    auto future = promise.get_future();
    ChatRequest chat;
    chat.session_id = "session-runtime";
    chat.user_input = "为什么天空是蓝色的？";
    chat.trace_id = "trace-adaptive";
    chat.base_generation = {0.7, 2000, 0.9};
    auto submit = runtime.SubmitChat(
        std::move(chat),
        [&promise](core::Result<ChatResponse> result) mutable {
            promise.set_value(std::move(result));
        });
    ASSERT_TRUE(submit.ok()) << submit.message();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto result = future.get();
    ASSERT_TRUE(result.ok()) << result.status().message();

    std::lock_guard lock(llm->mutex_);
    EXPECT_GE(llm->last_request.max_tokens, 100);
    EXPECT_LT(llm->last_request.max_tokens, 2000);
    ASSERT_FALSE(llm->last_request.messages.empty());
    EXPECT_EQ(llm->last_request.messages.front().content.find("unknown"), std::string::npos);
    EXPECT_EQ(llm->last_request.messages.front().content.find("neutral"), std::string::npos);

    compute.Shutdown(true);
    io.Shutdown(true);
}

TEST(PersonaRuntimeTest, BuildsProactivePromptWithTriggerInsteadOfUserUtterance) {
    core::ThreadPool compute({1, 32, "runtime-compute"});
    core::ThreadPool io({1, 32, "runtime-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    SessionManager sessions(compute, io);
    ASSERT_TRUE(sessions.CreateSession(MakeSessionRequest()).ok());

    auto cache = std::make_shared<FakeSemanticCache>();
    cache->lookup_hit = false;
    auto memory = std::make_shared<SemanticMemoryContextProvider>(cache);
    auto emotion = std::make_shared<NeutralEmotionAnalyzer>();
    auto llm = std::make_shared<FakeLlmClient>();
    PersonaRuntime runtime(
        sessions,
        memory,
        emotion,
        llm,
        PersonaRuntimeOptions{.recent_raw_turns = 10, .default_model = "test-model"});

    std::promise<core::Result<ChatResponse>> promise;
    auto future = promise.get_future();
    ChatRequest chat;
    chat.session_id = "session-runtime";
    chat.user_input = "[proactive] frontend requested proactive generation";
    chat.trace_id = "trace-proactive";
    auto submit = runtime.SubmitChat(
        std::move(chat),
        [&promise](core::Result<ChatResponse> result) mutable {
            promise.set_value(std::move(result));
        });
    ASSERT_TRUE(submit.ok()) << submit.message();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto result = future.get();
    ASSERT_TRUE(result.ok()) << result.status().message();

    std::lock_guard lock(llm->mutex_);
    ASSERT_GE(llm->last_request.messages.size(), 2u);
    EXPECT_NE(llm->last_request.messages.front().content.find("<proactive_trigger>"), std::string::npos);
    EXPECT_NE(llm->last_request.messages.front().content.find("你可以主动找话题聊"), std::string::npos);
    EXPECT_EQ(llm->last_request.messages.back().content, "...");

    compute.Shutdown(true);
    io.Shutdown(true);
}

TEST(PersonaRuntimeTest, DoesNotRecordEmotionCalibrationSamplesByDefault) {
    core::ThreadPool compute({1, 32, "runtime-compute"});
    core::ThreadPool io({1, 32, "runtime-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    SessionManager sessions(compute, io);
    ASSERT_TRUE(sessions.CreateSession(MakeSessionRequest()).ok());

    auto cache = std::make_shared<FakeSemanticCache>();
    cache->lookup_hit = false;
    auto memory = std::make_shared<SemanticMemoryContextProvider>(cache);
    auto emotion = std::make_shared<FixedEmotionAnalyzer>(MakeUncertainEmotion());
    auto llm = std::make_shared<FakeLlmClient>();
    auto sink = std::make_shared<RecordingEmotionCalibrationSink>();
    PersonaRuntime runtime(
        sessions,
        memory,
        emotion,
        llm,
        PersonaRuntimeOptions{.recent_raw_turns = 10, .default_model = "test-model"},
        nullptr,
        nullptr,
        nullptr,
        core::LoggerAdapter::ForModule("service"),
        sink);

    std::promise<core::Result<ChatResponse>> promise;
    auto future = promise.get_future();
    ChatRequest chat;
    chat.session_id = "session-runtime";
    chat.user_input = "我其实也不知道自己还能不能学会这个知识点";
    chat.trace_id = "trace-calibration-disabled";
    auto submit = runtime.SubmitChat(
        std::move(chat),
        [&promise](core::Result<ChatResponse> result) mutable {
            promise.set_value(std::move(result));
        });
    ASSERT_TRUE(submit.ok()) << submit.message();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto result = future.get();
    ASSERT_TRUE(result.ok()) << result.status().message();

    std::lock_guard lock(sink->mutex_);
    EXPECT_TRUE(sink->samples.empty());

    compute.Shutdown(true);
    io.Shutdown(true);
}

TEST(PersonaRuntimeTest, RecordsUncertainEmotionCalibrationSampleWhenEnabled) {
    core::ThreadPool compute({1, 32, "runtime-compute"});
    core::ThreadPool io({1, 32, "runtime-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    SessionManager sessions(compute, io);
    ASSERT_TRUE(sessions.CreateSession(MakeSessionRequest()).ok());

    agent::service::persona::ConversationTurn seed;
    seed.user_input = "前面的问题我一直没太懂";
    seed.response = "我们可以慢慢拆开看。";
    ASSERT_TRUE(sessions.AddTurn("session-runtime", std::move(seed), "trace-seed").ok());

    auto cache = std::make_shared<FakeSemanticCache>();
    cache->lookup_hit = false;
    auto memory = std::make_shared<SemanticMemoryContextProvider>(cache);
    auto emotion = std::make_shared<FixedEmotionAnalyzer>(MakeUncertainEmotion());
    auto llm = std::make_shared<FakeLlmClient>();
    auto sink = std::make_shared<RecordingEmotionCalibrationSink>();
    PersonaRuntimeOptions options{.recent_raw_turns = 10, .default_model = "test-model"};
    options.emotion_calibration.enabled = true;
    options.emotion_calibration.low_confidence_threshold = 0.45;
    options.emotion_calibration.top_margin_threshold = 0.15;
    PersonaRuntime runtime(
        sessions,
        memory,
        emotion,
        llm,
        options,
        nullptr,
        nullptr,
        nullptr,
        core::LoggerAdapter::ForModule("service"),
        sink);

    std::promise<core::Result<ChatResponse>> promise;
    auto future = promise.get_future();
    ChatRequest chat;
    chat.session_id = "session-runtime";
    chat.user_input = "我其实也不知道自己还能不能学会这个知识点";
    chat.trace_id = "trace-calibration-enabled";
    auto submit = runtime.SubmitChat(
        std::move(chat),
        [&promise](core::Result<ChatResponse> result) mutable {
            promise.set_value(std::move(result));
        });
    ASSERT_TRUE(submit.ok()) << submit.message();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto result = future.get();
    ASSERT_TRUE(result.ok()) << result.status().message();

    std::lock_guard lock(sink->mutex_);
    ASSERT_EQ(sink->samples.size(), 1u);
    EXPECT_EQ(sink->samples[0].trace_id, "trace-calibration-enabled");
    EXPECT_EQ(sink->samples[0].session_id, "session-runtime");
    EXPECT_EQ(sink->samples[0].user_uuid, "user-runtime");
    EXPECT_EQ(sink->samples[0].persona_id, "persona-main");
    EXPECT_EQ(sink->samples[0].bert_result.emotion.primary, "neutral");
    EXPECT_EQ(sink->samples[0].reason, "low_confidence");
    EXPECT_EQ(sink->samples[0].recent_turns.size(), 1u);

    compute.Shutdown(true);
    io.Shutdown(true);
}

TEST(PersonaRuntimeTest, InjectsTriggeredL4ToolMemoryIntoSystemPrompt) {
    core::ThreadPool compute({1, 32, "runtime-compute"});
    core::ThreadPool io({1, 32, "runtime-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    SessionManager sessions(compute, io);
    ASSERT_TRUE(sessions.CreateSession(MakeSessionRequest()).ok());

    auto cache = std::make_shared<FakeSemanticCache>();
    cache->lookup_hit = false;
    auto memory = std::make_shared<SemanticMemoryContextProvider>(cache);
    auto emotion = std::make_shared<NeutralEmotionAnalyzer>();
    auto llm = std::make_shared<FakeLlmClient>();
    auto tool_memory = std::make_shared<FakeToolMemoryProvider>();
    PersonaRuntime runtime(
        sessions,
        memory,
        emotion,
        llm,
        PersonaRuntimeOptions{.recent_raw_turns = 10, .default_model = "test-model"},
        nullptr,
        tool_memory);

    std::promise<core::Result<ChatResponse>> promise;
    auto future = promise.get_future();
    ChatRequest chat;
    chat.session_id = "session-runtime";
    chat.user_input = "你看一下现在画面里有什么";
    chat.trace_id = "trace-l4-tool-memory";
    auto submit = runtime.SubmitChat(
        std::move(chat),
        [&promise](core::Result<ChatResponse> result) mutable {
            promise.set_value(std::move(result));
        });
    ASSERT_TRUE(submit.ok()) << submit.message();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto result = future.get();
    ASSERT_TRUE(result.ok()) << result.status().message();

    EXPECT_TRUE(result.value().l4_hit);
    std::lock_guard lock(llm->mutex_);
    ASSERT_FALSE(llm->last_request.messages.empty());
    EXPECT_NE(llm->last_request.messages.front().content.find("<tool_memory_l4>"), std::string::npos);
    EXPECT_NE(llm->last_request.messages.front().content.find("vision.observe"), std::string::npos);

    {
        std::lock_guard tool_lock(tool_memory->mutex_);
        EXPECT_EQ(tool_memory->last_query.user_uuid, "user-runtime");
        EXPECT_EQ(tool_memory->last_query.query, "你看一下现在画面里有什么");
    }

    compute.Shutdown(true);
    io.Shutdown(true);
}

TEST(PersonaRuntimeTest, StartsVisionSkillSessionWhenL4VisionToolIsTriggered) {
    core::ThreadPool compute({1, 32, "runtime-compute"});
    core::ThreadPool io({1, 32, "runtime-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    SessionManager sessions(compute, io);
    ASSERT_TRUE(sessions.CreateSession(MakeSessionRequest()).ok());

    auto cache = std::make_shared<FakeSemanticCache>();
    cache->lookup_hit = false;
    auto memory = std::make_shared<SemanticMemoryContextProvider>(cache);
    auto emotion = std::make_shared<NeutralEmotionAnalyzer>();
    auto llm = std::make_shared<FakeLlmClient>();
    auto tool_memory = std::make_shared<FakeToolMemoryProvider>();
    auto skill_sessions = std::make_shared<SkillSessionManager>();
    PersonaRuntime runtime(
        sessions,
        memory,
        emotion,
        llm,
        PersonaRuntimeOptions{.recent_raw_turns = 10, .default_model = "test-model"},
        nullptr,
        tool_memory,
        skill_sessions);

    std::promise<core::Result<ChatResponse>> promise;
    auto future = promise.get_future();
    ChatRequest chat;
    chat.session_id = "session-runtime";
    chat.user_input = "你看一下现在画面里有什么";
    chat.trace_id = "trace-vision-skill-start";
    auto submit = runtime.SubmitChat(
        std::move(chat),
        [&promise](core::Result<ChatResponse> result) mutable {
            promise.set_value(std::move(result));
        });
    ASSERT_TRUE(submit.ok()) << submit.message();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto result = future.get();
    ASSERT_TRUE(result.ok()) << result.status().message();

    auto session = skill_sessions->Get("session-runtime", "vision.observe");
    ASSERT_TRUE(session.ok()) << session.status().message();
    ASSERT_TRUE(session.value().has_value());
    EXPECT_EQ(session.value()->state, SkillSessionState::Starting);

    std::lock_guard lock(llm->mutex_);
    ASSERT_FALSE(llm->last_request.messages.empty());
    EXPECT_NE(llm->last_request.messages.front().content.find("<skill_status"), std::string::npos);
    EXPECT_NE(llm->last_request.messages.front().content.find("vision.observe"), std::string::npos);

    compute.Shutdown(true);
    io.Shutdown(true);
}

TEST(PersonaRuntimeTest, InjectsVisionObservationFromRunningSkillSession) {
    core::ThreadPool compute({1, 32, "runtime-compute"});
    core::ThreadPool io({1, 32, "runtime-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    SessionManager sessions(compute, io);
    ASSERT_TRUE(sessions.CreateSession(MakeSessionRequest()).ok());

    auto cache = std::make_shared<FakeSemanticCache>();
    cache->lookup_hit = false;
    auto memory = std::make_shared<SemanticMemoryContextProvider>(cache);
    auto emotion = std::make_shared<NeutralEmotionAnalyzer>();
    auto llm = std::make_shared<FakeLlmClient>();
    auto skill_sessions = std::make_shared<SkillSessionManager>();

    SkillSessionStartRequest start;
    start.skill_id = "vision.observe";
    start.session_id = "session-runtime";
    start.user_uuid = "user-runtime";
    start.persona_id = "persona-main";
    start.trace_id = "trace-seed";
    start.source = "test";
    ASSERT_TRUE(skill_sessions->Start(start).ok());
    ASSERT_TRUE(skill_sessions->MarkReady("session-runtime", "vision.observe", "vision ready", "trace-ready").ok());
    SkillObservation observation;
    observation.skill_id = "vision.observe";
    observation.session_id = "session-runtime";
    observation.trace_id = "trace-observation";
    observation.summary = "画面中检测到明显移动";
    observation.confidence = 0.72;
    observation.source = "vlm";
    observation.should_inject_prompt = true;
    ASSERT_TRUE(skill_sessions->RecordObservation(observation).ok());

    PersonaRuntime runtime(
        sessions,
        memory,
        emotion,
        llm,
        PersonaRuntimeOptions{.recent_raw_turns = 10, .default_model = "test-model"},
        nullptr,
        nullptr,
        skill_sessions);

    std::promise<core::Result<ChatResponse>> promise;
    auto future = promise.get_future();
    ChatRequest chat;
    chat.session_id = "session-runtime";
    chat.user_input = "现在情况怎么样？";
    chat.trace_id = "trace-vision-observation";
    auto submit = runtime.SubmitChat(
        std::move(chat),
        [&promise](core::Result<ChatResponse> result) mutable {
            promise.set_value(std::move(result));
        });
    ASSERT_TRUE(submit.ok()) << submit.message();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    auto result = future.get();
    ASSERT_TRUE(result.ok()) << result.status().message();

    std::lock_guard lock(llm->mutex_);
    ASSERT_FALSE(llm->last_request.messages.empty());
    EXPECT_NE(llm->last_request.messages.front().content.find("<skill_observation"), std::string::npos);
    EXPECT_NE(llm->last_request.messages.front().content.find("画面中检测到明显移动"), std::string::npos);

    compute.Shutdown(true);
    io.Shutdown(true);
}

TEST(SkillSessionManagerTest, RunsStartReadyObservationAndStopLifecycle) {
    SkillSessionManager manager;
    SkillSessionStartRequest start;
    start.skill_id = "vision.observe";
    start.session_id = "session-runtime";
    start.user_uuid = "user-runtime";
    start.persona_id = "persona-main";
    start.trace_id = "trace-skill-start";
    start.source = "regex";
    start.reason = "用户请求观察画面";

    auto started = manager.Start(start);

    ASSERT_TRUE(started.ok()) << started.status().message();
    EXPECT_EQ(started.value().state, SkillSessionState::Starting);
    EXPECT_EQ(started.value().skill_id, "vision.observe");
    EXPECT_EQ(started.value().session_id, "session-runtime");

    auto ready = manager.MarkReady("session-runtime", "vision.observe", "vision ready", "trace-ready");
    ASSERT_TRUE(ready.ok()) << ready.message();

    SkillObservation observation;
    observation.skill_id = "vision.observe";
    observation.session_id = "session-runtime";
    observation.trace_id = "trace-observation";
    observation.summary = "画面中检测到明显移动";
    observation.confidence = 0.72;
    observation.source = "vlm";
    observation.should_inject_prompt = true;
    auto recorded = manager.RecordObservation(observation);
    ASSERT_TRUE(recorded.ok()) << recorded.message();

    auto current = manager.Get("session-runtime", "vision.observe");
    ASSERT_TRUE(current.ok()) << current.status().message();
    ASSERT_TRUE(current.value().has_value());
    EXPECT_EQ(current.value()->state, SkillSessionState::Running);
    EXPECT_EQ(current.value()->last_observation, "画面中检测到明显移动");
    ASSERT_EQ(current.value()->recent_observations.size(), 1u);

    SkillSessionStopRequest stop;
    stop.skill_id = "vision.observe";
    stop.session_id = "session-runtime";
    stop.trace_id = "trace-stop";
    stop.source = "user";
    stop.reason = "用户关闭视觉观察";
    auto stopped = manager.Stop(stop);
    ASSERT_TRUE(stopped.ok()) << stopped.status().message();
    EXPECT_EQ(stopped.value().state, SkillSessionState::Closing);
    ASSERT_TRUE(manager.CompleteClosing(
        start.session_id,
        start.skill_id,
        started.value().execution_id,
        "closed after async drain",
        "trace-stop").ok());
    current = manager.Get(start.session_id, start.skill_id);
    ASSERT_TRUE(current.ok());
    ASSERT_TRUE(current.value().has_value());
    EXPECT_EQ(current.value()->state, SkillSessionState::Closed);
    EXPECT_EQ(stopped.value().close_reason, "用户关闭视觉观察");
}

TEST(SkillSessionManagerTest, ExpiresStartingSessionThroughMaintenanceTask) {
    SkillSessionOptions options;
    options.startup_timeout = std::chrono::milliseconds(1);
    options.max_duration = std::chrono::seconds(10);
    options.idle_timeout = std::chrono::seconds(10);
    auto manager = std::make_shared<SkillSessionManager>(options);

    SkillSessionStartRequest start;
    start.skill_id = "vision.observe";
    start.session_id = "session-timeout";
    start.trace_id = "trace-timeout";
    start.source = "vector";
    auto started = manager->Start(start);
    ASSERT_TRUE(started.ok()) << started.status().message();

    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    agent::service::gateway::SkillSessionMaintenanceTask task(
        manager,
        std::chrono::milliseconds(10));
    auto tick = task.Tick(std::stop_token{});

    ASSERT_TRUE(tick.ok()) << tick.message();
    auto current = manager->Get("session-timeout", "vision.observe");
    ASSERT_TRUE(current.ok()) << current.status().message();
    ASSERT_TRUE(current.value().has_value());
    EXPECT_EQ(current.value()->state, SkillSessionState::Expired);
    EXPECT_EQ(current.value()->last_error, "startup_timeout");
}

TEST(SkillSessionManagerTest, ExpiresClosingSessionThatMissesAtomicDrainDeadline) {
    SkillSessionOptions options;
    options.closing_timeout = std::chrono::milliseconds(5);
    SkillSessionManager manager(options);
    SkillSessionStartRequest start;
    start.execution_id = "closing-timeout-execution";
    start.skill_id = "vision.observe";
    start.session_id = "closing-timeout-session";
    auto started = manager.Start(start);
    ASSERT_TRUE(started.ok());
    ASSERT_TRUE(manager.MarkReady(start.session_id, start.skill_id, "ready", "trace").ok());
    ASSERT_TRUE(manager.BeginClosing(
        start.session_id, start.skill_id, start.execution_id, "draining", "trace").ok());
    std::this_thread::sleep_for(std::chrono::milliseconds(8));
    EXPECT_EQ(manager.CleanupExpired({}), 1u);
    auto current = manager.Get(start.session_id, start.skill_id);
    ASSERT_TRUE(current.ok());
    ASSERT_TRUE(current.value().has_value());
    EXPECT_EQ(current.value()->state, SkillSessionState::Expired);
}

TEST(SkillVisionEventSinkTest, RecordsVisionEventAsSkillObservation) {
    auto manager = std::make_shared<SkillSessionManager>();
    SkillVisionEventSink sink(manager);

    media::VisionEvent event;
    event.event_id = "vision-event-1";
    event.session_id = "session-runtime";
    event.trace_id = "trace-vision-event";
    event.peak_frame_id = 7;
    event.representative_frame_id = 7;
    event.peak_score = 0.81;
    media::VisionAnalysis analysis;
    analysis.agent_hint = "画面中检测到明显移动";
    analysis.confidence = 0.76;
    analysis.facts = {"画面中有移动"};
    event.analysis = analysis;

    auto status = sink.Publish(event);

    ASSERT_TRUE(status.ok()) << status.message();
    auto session = manager->Get("session-runtime", "vision.observe");
    ASSERT_TRUE(session.ok()) << session.status().message();
    ASSERT_TRUE(session.value().has_value());
    EXPECT_EQ(session.value()->state, SkillSessionState::Running);
    EXPECT_EQ(session.value()->last_observation, "画面中检测到明显移动");
    ASSERT_EQ(session.value()->recent_observations.size(), 1u);
    EXPECT_EQ(session.value()->recent_observations[0].confidence, 0.76);
    EXPECT_NE(session.value()->recent_observations[0].metadata_json.find("vision-event-1"), std::string::npos);
}

TEST(SkillVisionEventSinkTest, DuplicateVisionEventIsRecordedButNotInjected) {
    auto manager = std::make_shared<SkillSessionManager>();
    SkillVisionEventSinkOptions options;
    options.min_prompt_confidence = 0.1;
    SkillVisionEventSink sink(manager, options);

    media::VisionEvent event;
    event.event_id = "vision-event-duplicate";
    event.session_id = "session-runtime";
    event.trace_id = "trace-vision-event";
    event.peak_score = 0.91;
    event.duplicate = true;
    event.analysis = media::VisionAnalysis{.agent_hint = "重复视觉事件", .confidence = 0.8};

    auto status = sink.Publish(event);

    ASSERT_TRUE(status.ok()) << status.message();
    auto session = manager->Get("session-runtime", "vision.observe");
    ASSERT_TRUE(session.ok()) << session.status().message();
    ASSERT_TRUE(session.value().has_value());
    EXPECT_EQ(session.value()->state, SkillSessionState::Running);
    EXPECT_TRUE(session.value()->last_observation.empty());
    ASSERT_EQ(session.value()->recent_observations.size(), 1u);
    EXPECT_TRUE(session.value()->recent_observations[0].stale);
    EXPECT_FALSE(session.value()->recent_observations[0].should_inject_prompt);
}

} // namespace
