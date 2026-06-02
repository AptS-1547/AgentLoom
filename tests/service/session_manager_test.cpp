#include "session_manager.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>

namespace {

using namespace std::chrono_literals;
using agent::service::persona::ConversationTurn;
using agent::service::persona::CreateSessionRequest;
using agent::service::persona::DispatchOptions;
using agent::service::persona::PersonalityConfig;
using agent::service::persona::SessionManager;
using agent::service::persona::SessionOptions;

CreateSessionRequest MakeCreateRequest(std::string session_id = "session-a") {
    PersonalityConfig personality;
    personality.name = "小橘";
    personality.description = "教育陪伴人格";

    CreateSessionRequest req;
    req.user_uuid = "user-1";
    req.persona_id = "persona-main";
    req.session_id = std::move(session_id);
    req.trace_id = "trace-create";
    req.personality = std::move(personality);
    req.time_awareness = false;
    req.emotion_state_config.noise_sigma = 0.0;
    return req;
}

TEST(SessionManagerTest, CreatesTouchesAddsTurnsAndClosesSession) {
    core::ThreadPool compute({1, 8, "test-compute"});
    core::ThreadPool io({1, 8, "test-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    SessionOptions options;
    options.max_recent_turns = 2;
    SessionManager manager(compute, io, options);

    auto created = manager.CreateSession(MakeCreateRequest());
    ASSERT_TRUE(created.ok()) << created.status().message();
    EXPECT_EQ(created.value().session_id, "session-a");
    EXPECT_EQ(manager.SessionCount(), 1u);

    EXPECT_TRUE(manager.AddTurn("session-a", ConversationTurn{.user_input = "a", .response = "1"}, "trace-turn-1").ok());
    EXPECT_TRUE(manager.AddTurn("session-a", ConversationTurn{.user_input = "b", .response = "2"}, "trace-turn-2").ok());
    EXPECT_TRUE(manager.AddTurn("session-a", ConversationTurn{.user_input = "c", .response = "3"}, "trace-turn-3").ok());

    auto snapshot = manager.GetSessionSnapshot("session-a");
    ASSERT_TRUE(snapshot.ok()) << snapshot.status().message();
    EXPECT_EQ(snapshot.value().recent_turn_count, 2u);
    EXPECT_EQ(snapshot.value().last_trace_id, "trace-turn-3");

    EXPECT_TRUE(manager.TouchSession("session-a", "trace-touch").ok());
    snapshot = manager.GetSessionSnapshot("session-a");
    ASSERT_TRUE(snapshot.ok()) << snapshot.status().message();
    EXPECT_EQ(snapshot.value().last_trace_id, "trace-touch");

    EXPECT_TRUE(manager.CloseSession("session-a", "trace-close").ok());
    EXPECT_EQ(manager.SessionCount(), 0u);

    compute.Shutdown(true);
    io.Shutdown(true);
}

TEST(SessionManagerTest, DispatchPropagatesTraceIdIntoComputeAndIoPools) {
    core::ThreadPool compute({1, 8, "test-compute"});
    core::ThreadPool io({1, 8, "test-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    SessionManager manager(compute, io);
    ASSERT_TRUE(manager.CreateSession(MakeCreateRequest("session-dispatch")).ok());

    std::promise<std::string> compute_trace;
    auto compute_future = compute_trace.get_future();
    auto compute_status = manager.SubmitCompute(
        DispatchOptions{
            .session_id = "session-dispatch",
            .trace_id = "trace-compute",
            .user_uuid = "user-1",
            .module = "persona",
            .operation = "emotion_update",
        },
        [&compute_trace](auto& session, core::ThreadPoolContext&) {
            auto updated = session.emotion_state.Update("joy", 0.9, "neutral", 0.2);
            if (!updated.ok()) {
                return updated.status();
            }
            compute_trace.set_value(std::string(core::CurrentTraceId()));
            return core::Status::Ok();
        });
    ASSERT_TRUE(compute_status.ok()) << compute_status.message();
    ASSERT_EQ(compute_future.wait_for(2s), std::future_status::ready);
    EXPECT_EQ(compute_future.get(), "trace-compute");

    std::promise<std::string> io_trace;
    auto io_future = io_trace.get_future();
    auto io_status = manager.SubmitIo(
        DispatchOptions{
            .session_id = "session-dispatch",
            .trace_id = "trace-io",
            .user_uuid = "user-1",
            .module = "storage",
            .operation = "persist",
        },
        [&io_trace](auto&, core::ThreadPoolContext&) {
            io_trace.set_value(std::string(core::CurrentTraceId()));
            return core::Status::Ok();
        });
    ASSERT_TRUE(io_status.ok()) << io_status.message();
    ASSERT_EQ(io_future.wait_for(2s), std::future_status::ready);
    EXPECT_EQ(io_future.get(), "trace-io");

    compute.Shutdown(true);
    io.Shutdown(true);
}

TEST(SessionManagerTest, DispatchReportsNotFoundWhenSessionWasClosedBeforeExecution) {
    core::ThreadPool compute({1, 8, "test-compute"});
    core::ThreadPool io({1, 8, "test-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    SessionManager manager(compute, io);
    ASSERT_TRUE(manager.CreateSession(MakeCreateRequest("session-close-race")).ok());
    ASSERT_TRUE(manager.CloseSession("session-close-race", "trace-close").ok());

    std::atomic<bool> ran{false};
    auto status = manager.SubmitCompute(
        DispatchOptions{.session_id = "session-close-race", .trace_id = "trace-late"},
        [&ran](auto&, core::ThreadPoolContext&) {
            ran.store(true, std::memory_order_relaxed);
            return core::Status::Ok();
        });
    ASSERT_TRUE(status.ok()) << status.message();

    compute.Shutdown(true);
    io.Shutdown(true);
    EXPECT_FALSE(ran.load(std::memory_order_relaxed));
    EXPECT_EQ(compute.Stats().failed_tasks, 1u);
}

} // namespace
