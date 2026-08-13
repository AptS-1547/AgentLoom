#include "gateway_session_affinity_scheduler.h"
#include "session_manager.h"
#include "thread_pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using agent::service::persona::CreateSessionRequest;
using agent::service::persona::DispatchOptions;
using agent::service::persona::GatewaySessionAffinityScheduler;
using agent::service::persona::GatewaySessionAffinitySchedulerOptions;
using agent::service::persona::SessionManager;

struct ScenarioResult {
    std::string layout;
    double io_probe_latency_ms = 0.0;
    double turn_elapsed_ms = 0.0;
    double turns_per_second = 0.0;
    std::size_t llm_queue_high_water = 0;
    std::size_t io_queue_high_water = 0;
    std::size_t order_errors = 0;
    std::size_t rejected = 0;
};

std::shared_ptr<core::IThreadPoolTaskScheduler> MakeAffinityScheduler(
    std::size_t session_count,
    std::size_t turn_count) {
    return std::make_shared<GatewaySessionAffinityScheduler>(
        GatewaySessionAffinitySchedulerOptions{
            .max_active_keys = session_count + 4,
            .max_outstanding_per_key = turn_count + 2,
            .max_outstanding_per_fairness_key = session_count * turn_count + 4,
            .max_outstanding_per_tenant = session_count * turn_count + 4,
        });
}

CreateSessionRequest MakeSession(std::string id, std::size_t index) {
    CreateSessionRequest request;
    request.session_id = std::move(id);
    request.user_uuid = "bench-user-" + std::to_string(index);
    request.persona_id = "bench-persona";
    request.trace_id = "bench-create";
    request.personality.name = "benchmark";
    request.time_awareness = false;
    request.emotion_state_config.noise_sigma = 0.0;
    return request;
}

ScenarioResult RunScenario(bool dedicated,
                           std::size_t session_count,
                           std::size_t turns_per_session,
                           std::chrono::milliseconds llm_delay) {
    constexpr std::size_t kIoWorkers = 4;
    constexpr std::size_t kLlmWorkers = 4;
    const auto total_turns = session_count * turns_per_session;
    const auto queue_capacity = total_turns + session_count + 16;

    core::ThreadPool compute({1, 32, "persona-llm-bench-compute"});
    core::ThreadPool io({
        kIoWorkers,
        queue_capacity,
        dedicated ? "persona-llm-bench-io" : "persona-llm-bench-shared-io",
        MakeAffinityScheduler(session_count, turns_per_session),
    });
    std::unique_ptr<core::ThreadPool> llm;
    if (dedicated) {
        llm = std::make_unique<core::ThreadPool>(core::ThreadPoolOptions{
            kLlmWorkers,
            queue_capacity,
            "persona-llm-bench-dedicated",
            MakeAffinityScheduler(session_count, turns_per_session),
        });
    }

    if (!compute.Start().ok() || !io.Start().ok() || (llm && !llm->Start().ok())) {
        throw std::runtime_error("failed to start Persona LLM benchmark pools");
    }
    SessionManager sessions(
        compute,
        io,
        {},
        core::LoggerAdapter::ForModule("persona-llm-bench"),
        llm.get());

    for (std::size_t index = 0; index < session_count; ++index) {
        auto status = sessions.CreateSession(MakeSession("bench-session-" + std::to_string(index), index));
        if (!status.ok()) {
            throw std::runtime_error(status.status().message());
        }
    }
    auto probe_session = sessions.CreateSession(MakeSession("bench-io-probe", session_count));
    if (!probe_session.ok()) {
        throw std::runtime_error(probe_session.status().message());
    }

    std::mutex completed_mutex;
    std::condition_variable completed_cv;
    std::size_t completed = 0;
    std::size_t rejected = 0;
    std::size_t order_errors = 0;
    std::vector<std::uint64_t> expected_sequence(session_count, 1);
    const auto turns_started = Clock::now();

    for (std::size_t turn = 0; turn < turns_per_session; ++turn) {
        for (std::size_t session = 0; session < session_count; ++session) {
            const auto session_id = "bench-session-" + std::to_string(session);
            const auto status = sessions.SubmitTurn(
                DispatchOptions{
                    .session_id = session_id,
                    .trace_id = "bench-turn-" + std::to_string(turn),
                    .module = "persona-bench",
                    .operation = "slow-llm",
                },
                [llm_delay](const auto& snapshot, auto& commit, auto&) {
                    // 以 sleep 模拟数秒级云 LLM；完整等待发生在 Turn pool，不持有 Session 状态锁。
                    std::this_thread::sleep_for(llm_delay);
                    commit.trace_id = "bench-turn";
                    commit.turn.user_input = "benchmark";
                    commit.turn.response = "ok";
                    commit.emotion_state = snapshot.state.emotion_state;
                    commit.latency = llm_delay;
                    return core::Status::Ok();
                },
                [&, session](auto result) {
                    std::lock_guard lock(completed_mutex);
                    if (!result.ok() || result.value().sequence != expected_sequence[session]) {
                        ++order_errors;
                    }
                    ++expected_sequence[session];
                    ++completed;
                    completed_cv.notify_all();
                });
            if (!status.ok()) {
                std::lock_guard lock(completed_mutex);
                ++rejected;
            }
        }
    }

    std::atomic<bool> sampling{true};
    std::atomic<std::size_t> llm_queue_high_water{0};
    std::atomic<std::size_t> io_queue_high_water{0};
    std::jthread sampler([&](std::stop_token stop_token) {
        while (!stop_token.stop_requested() && sampling.load(std::memory_order_acquire)) {
            const auto stats = sessions.PoolStats();
            llm_queue_high_water.store(
                std::max(llm_queue_high_water.load(), stats.llm.scheduler.queued_tasks));
            io_queue_high_water.store(
                std::max(io_queue_high_water.load(), stats.io.scheduler.queued_tasks));
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    std::mutex probe_mutex;
    std::condition_variable probe_cv;
    bool probe_done = false;
    const auto probe_submitted = Clock::now();
    auto probe_status = sessions.SubmitIo(
        DispatchOptions{.session_id = "bench-io-probe", .trace_id = "bench-io-probe"},
        [&](auto&, auto&) {
            {
                std::lock_guard lock(probe_mutex);
                probe_done = true;
            }
            probe_cv.notify_one();
            return core::Status::Ok();
        });
    if (!probe_status.ok()) {
        throw std::runtime_error(probe_status.message());
    }
    {
        std::unique_lock lock(probe_mutex);
        probe_cv.wait(lock, [&] { return probe_done; });
    }
    const auto probe_latency = Clock::now() - probe_submitted;

    {
        std::unique_lock lock(completed_mutex);
        // SubmitTurn 即使 admission 失败也保证 completion 恰好调用一次。
        completed_cv.wait(lock, [&] { return completed == total_turns; });
    }
    const auto turn_elapsed = Clock::now() - turns_started;
    sampling.store(false, std::memory_order_release);
    sampler.request_stop();
    sampler.join();

    sessions.Shutdown();
    if (llm) {
        llm->Shutdown(true);
    }
    io.Shutdown(true);
    compute.Shutdown(true);

    const auto elapsed_ms = std::chrono::duration<double, std::milli>(turn_elapsed).count();
    return {
        .layout = dedicated ? "dedicated_llm_pool" : "shared_io_pool",
        .io_probe_latency_ms = std::chrono::duration<double, std::milli>(probe_latency).count(),
        .turn_elapsed_ms = elapsed_ms,
        .turns_per_second = static_cast<double>(total_turns - rejected) * 1000.0 / elapsed_ms,
        .llm_queue_high_water = llm_queue_high_water.load(),
        .io_queue_high_water = io_queue_high_water.load(),
        .order_errors = order_errors,
        .rejected = rejected,
    };
}

void Print(const ScenarioResult& result) {
    std::cout << std::fixed << std::setprecision(3)
              << result.layout << ','
              << result.io_probe_latency_ms << ','
              << result.turn_elapsed_ms << ','
              << result.turns_per_second << ','
              << result.llm_queue_high_water << ','
              << result.io_queue_high_water << ','
              << result.order_errors << ','
              << result.rejected << '\n';
}

} // namespace

int main(int argc, char** argv) {
    const auto delay_ms = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1000;
    const auto sessions = argc > 2
        ? static_cast<std::size_t>(std::max(1, std::atoi(argv[2])))
        : 12;
    const auto turns = argc > 3
        ? static_cast<std::size_t>(std::max(1, std::atoi(argv[3])))
        : 2;

    std::cout << "layout,io_probe_latency_ms,turn_elapsed_ms,turns_per_sec,"
                 "llm_queue_high_water,io_queue_high_water,order_errors,rejected\n";
    Print(RunScenario(false, sessions, turns, std::chrono::milliseconds(delay_ms)));
    Print(RunScenario(true, sessions, turns, std::chrono::milliseconds(delay_ms)));
    return 0;
}
