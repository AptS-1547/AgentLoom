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

} // namespace
