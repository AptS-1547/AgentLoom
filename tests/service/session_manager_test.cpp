#include "session_manager.h"
#include "runtime_maintenance_service.h"
#include "inference_frame_ipc_control.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <filesystem>
#include <string_view>
#include <thread>

namespace {

using namespace std::chrono_literals;
using agent::service::persona::ConversationTurn;
using agent::service::persona::CreateSessionRequest;
using agent::service::persona::DispatchOptions;
using agent::service::persona::PersonalityConfig;
using agent::service::persona::SessionManager;
using agent::service::persona::SessionOptions;
using agent::service::gateway::IRuntimeMaintenanceTask;
using agent::service::gateway::InferenceFrameIpcPeerMaintenanceTask;
using agent::service::gateway::RuntimeMaintenanceService;
using agent::service::gateway::SessionMaintenanceTask;
using agent::service::gateway::AuthSessionMaintenanceTask;
using agent::service::gateway::AuthSessionRecord;
using agent::service::gateway::SqliteAuthSessionStore;

class CountingMaintenanceTask final : public IRuntimeMaintenanceTask {
public:
    explicit CountingMaintenanceTask(std::atomic<int>& ticks) : ticks_(ticks) {}

    std::string_view Name() const noexcept override {
        return "counting_task";
    }

    std::chrono::milliseconds Interval() const noexcept override {
        return 10ms;
    }

    core::Status Tick(std::stop_token stop_token) override {
        if (!stop_token.stop_requested()) {
            ticks_.fetch_add(1, std::memory_order_relaxed);
        }
        return core::Status::Ok();
    }

private:
    std::atomic<int>& ticks_;
};

class FakeIpcLeaseCoordinator final : public ipc::media::IInferenceFrameIpcLeaseCoordinator {
public:
    core::Status Start() override {
        ++starts;
        snapshot.state = ipc::media::InferenceFrameIpcControlState::Granted;
        return core::Status::Ok();
    }

    core::Status Revoke(std::string) override {
        snapshot.state = ipc::media::InferenceFrameIpcControlState::Fenced;
        return core::Status::Ok();
    }

    core::Status CheckPeer() override {
        ++probes;
        if (!fail_probe) {
            return core::Status::Ok();
        }
        snapshot.state = ipc::media::InferenceFrameIpcControlState::Fenced;
        return core::Status::Error(core::ErrorCode::Unavailable, "fake inference peer lost");
    }

    core::Status Recover() override {
        ++recoveries;
        snapshot.state = ipc::media::InferenceFrameIpcControlState::Granted;
        fail_probe = false;
        return core::Status::Ok();
    }

    void Shutdown() override {
        snapshot.state = ipc::media::InferenceFrameIpcControlState::Shutdown;
    }

    ipc::media::InferenceFrameIpcControlSnapshot Snapshot() const override {
        return snapshot;
    }

    ipc::media::InferenceFrameIpcControlSnapshot snapshot;
    bool fail_probe = false;
    int starts = 0;
    int probes = 0;
    int recoveries = 0;
};

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

TEST(SessionManagerTest, RejectsNewSessionsWhenActiveLimitIsReached) {
    core::ThreadPool compute({1, 8, "test-compute"});
    core::ThreadPool io({1, 8, "test-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    SessionOptions options;
    options.max_active_sessions = 2;
    SessionManager manager(compute, io, options);

    ASSERT_TRUE(manager.CreateSession(MakeCreateRequest("session-limit-a")).ok());
    ASSERT_TRUE(manager.CreateSession(MakeCreateRequest("session-limit-b")).ok());
    EXPECT_EQ(manager.SessionCount(), 2u);

    auto rejected = manager.CreateSession(MakeCreateRequest("session-limit-c"));
    ASSERT_FALSE(rejected.ok());
    EXPECT_EQ(rejected.status().code(), core::ErrorCode::ResourceExhausted);
    EXPECT_EQ(rejected.status().message(), "active session limit reached");

    ASSERT_TRUE(manager.CloseSession("session-limit-a").ok());
    auto admitted = manager.CreateSession(MakeCreateRequest("session-limit-c"));
    ASSERT_TRUE(admitted.ok()) << admitted.status().message();
    EXPECT_EQ(manager.SessionCount(), 2u);

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

TEST(RuntimeMaintenanceServiceTest, RunsRegisteredTaskOnDedicatedWorker) {
    std::atomic<int> ticks{0};
    RuntimeMaintenanceService maintenance;
    ASSERT_TRUE(maintenance.RegisterTask(std::make_shared<CountingMaintenanceTask>(ticks)).ok());
    ASSERT_TRUE(maintenance.Start().ok());

    const auto deadline = std::chrono::steady_clock::now() + 1s;
    while (ticks.load(std::memory_order_relaxed) < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(10ms);
    }
    maintenance.Stop();

    EXPECT_GE(ticks.load(std::memory_order_relaxed), 2);
    auto health = maintenance.SnapshotHealth();
    ASSERT_EQ(health.size(), 1u);
    EXPECT_EQ(health.front().name, "counting_task");
    EXPECT_GE(health.front().success_count, 1u);
    EXPECT_EQ(health.front().failure_count, 0u);
}

TEST(RuntimeMaintenanceServiceTest, SessionCleanupTaskExpiresIdleSessions) {
    core::ThreadPool compute({1, 8, "test-compute"});
    core::ThreadPool io({1, 8, "test-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    SessionOptions options;
    options.idle_timeout = std::chrono::minutes(0);
    SessionManager manager(compute, io, options);
    ASSERT_TRUE(manager.CreateSession(MakeCreateRequest("session-maintenance-expire")).ok());

    RuntimeMaintenanceService maintenance;
    ASSERT_TRUE(maintenance.RegisterTask(std::make_shared<SessionMaintenanceTask>(
                    manager,
                    10ms))
                    .ok());
    ASSERT_TRUE(maintenance.Start().ok());

    const auto deadline = std::chrono::steady_clock::now() + 1s;
    while (manager.SessionCount() != 0 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(10ms);
    }
    maintenance.Stop();

    EXPECT_EQ(manager.SessionCount(), 0u);
    compute.Shutdown(true);
    io.Shutdown(true);
}

TEST(RuntimeMaintenanceServiceTest, AuthSessionCleanupTaskRemovesExpiredSessions) {
    const auto path = std::filesystem::temp_directory_path() / "agent_gateway_auth_maintenance_test.db";
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(path.string() + "-wal", ec);
    std::filesystem::remove(path.string() + "-shm", ec);

    auto store = std::make_shared<SqliteAuthSessionStore>(path.string());
    ASSERT_TRUE(store->EnsureSchema().ok());
    AuthSessionRecord record;
    record.token_id = "maintenance-expired";
    record.user_uuid = "maintenance-user";
    record.issued_at = std::chrono::system_clock::now() - 2h;
    record.expires_at = std::chrono::system_clock::now() - 1h;
    ASSERT_TRUE(store->UpsertSession(record).ok());

    AuthSessionMaintenanceTask task(store, 10ms, 8);
    std::stop_source stop;
    ASSERT_TRUE(task.Tick(stop.get_token()).ok());
    auto resolved = store->ResolveSession(record.token_id);
    EXPECT_FALSE(resolved.ok());
    EXPECT_EQ(resolved.status().code(), core::ErrorCode::NotFound);

    std::filesystem::remove(path, ec);
    std::filesystem::remove(path.string() + "-wal", ec);
    std::filesystem::remove(path.string() + "-shm", ec);
}

TEST(RuntimeMaintenanceServiceTest, IpcPeerTaskFencesThenRecoversLostInferencePeer) {
    auto coordinator = std::make_shared<FakeIpcLeaseCoordinator>();
    InferenceFrameIpcPeerMaintenanceTask task(coordinator, 10ms, true);
    std::stop_source stop;

    ASSERT_TRUE(task.Tick(stop.get_token()).ok());
    EXPECT_EQ(coordinator->starts, 1);
    EXPECT_EQ(coordinator->snapshot.state, ipc::media::InferenceFrameIpcControlState::Granted);

    coordinator->fail_probe = true;
    const auto lost = task.Tick(stop.get_token());
    EXPECT_EQ(lost.code(), core::ErrorCode::Unavailable);
    EXPECT_EQ(coordinator->snapshot.state, ipc::media::InferenceFrameIpcControlState::Fenced);

    ASSERT_TRUE(task.Tick(stop.get_token()).ok());
    EXPECT_EQ(coordinator->recoveries, 1);
    EXPECT_EQ(coordinator->snapshot.state, ipc::media::InferenceFrameIpcControlState::Granted);
}

} // namespace
