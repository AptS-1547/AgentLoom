#include "persona_runtime.h"

#include <gtest/gtest.h>

#include <future>
#include <mutex>

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

} // namespace
