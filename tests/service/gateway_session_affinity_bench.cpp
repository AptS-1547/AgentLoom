#include "gateway_session_affinity_scheduler.h"
#include "thread_pool.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

struct Summary {
    double average_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double maximum_ms = 0.0;
};

Summary Summarize(std::vector<double> values) {
    if (values.empty()) {
        return {};
    }
    std::sort(values.begin(), values.end());
    const auto percentile = [&](double value) {
        const auto index = std::min(values.size() - 1,
                                    static_cast<std::size_t>(value * static_cast<double>(values.size() - 1)));
        return values[index];
    };
    return {
        .average_ms = std::accumulate(values.begin(), values.end(), 0.0) /
                      static_cast<double>(values.size()),
        .p50_ms = percentile(0.50),
        .p95_ms = percentile(0.95),
        .p99_ms = percentile(0.99),
        .maximum_ms = values.back(),
    };
}

struct ScenarioResult {
    std::string scheduler;
    std::string pool_role;
    Summary all_queue_wait;
    Summary normal_queue_wait;
    double elapsed_ms = 0.0;
    double tasks_per_second = 0.0;
    std::size_t rejected = 0;
};

ScenarioResult RunScenario(bool affinity,
                           std::string pool_role,
                           std::chrono::milliseconds task_duration) {
    constexpr std::size_t kWorkers = 4;
    constexpr std::size_t kHotTasks = 48;
    constexpr std::size_t kNormalSessions = 12;
    constexpr std::size_t kTasksPerNormalSession = 4;
    constexpr std::size_t kTaskCount = kHotTasks + kNormalSessions * kTasksPerNormalSession;

    std::shared_ptr<core::IThreadPoolTaskScheduler> scheduler;
    if (affinity) {
        scheduler = std::make_shared<agent::service::persona::GatewaySessionAffinityScheduler>(
            agent::service::persona::GatewaySessionAffinitySchedulerOptions{
                .max_active_keys = kNormalSessions + 2,
                .max_outstanding_per_key = kHotTasks,
                .max_outstanding_per_fairness_key = kTaskCount,
            });
    }
    core::ThreadPool pool({
        .worker_count = kWorkers,
        .queue_capacity = kTaskCount,
        .name = pool_role + (affinity ? "-affinity" : "-fifo"),
        .scheduler = scheduler,
    });
    if (!pool.Start().ok()) {
        throw std::runtime_error("failed to start benchmark pool");
    }

    std::unordered_map<std::string, std::shared_ptr<std::mutex>> session_locks;
    session_locks["hot-session"] = std::make_shared<std::mutex>();
    for (std::size_t index = 0; index < kNormalSessions; ++index) {
        session_locks["normal-" + std::to_string(index)] = std::make_shared<std::mutex>();
    }

    std::mutex result_mutex;
    std::condition_variable completed_cv;
    std::vector<double> all_wait;
    std::vector<double> normal_wait;
    std::size_t completed = 0;
    std::size_t rejected = 0;
    const auto benchmark_started = Clock::now();

    auto submit = [&](const std::string& session_id, bool hot) {
        const auto submitted_at = Clock::now();
        core::ThreadPoolTaskMetadata metadata;
        metadata.concurrency_key = session_id;
        metadata.fairness_key = hot ? "hot-user" : session_id + "-user";
        const auto status = pool.Submit(
            [&, session_lock = session_locks.at(session_id), submitted_at, hot] {
                const auto started_at = Clock::now();
                const auto wait_ms = std::chrono::duration<double, std::milli>(started_at - submitted_at).count();
                {
                    std::lock_guard result_lock(result_mutex);
                    all_wait.push_back(wait_ms);
                    if (!hot) {
                        normal_wait.push_back(wait_ms);
                    }
                }
                std::lock_guard session_lock_guard(*session_lock);
                std::this_thread::sleep_for(task_duration);
                {
                    std::lock_guard result_lock(result_mutex);
                    ++completed;
                }
                completed_cv.notify_one();
            },
            {},
            pool_role + ":" + session_id,
            std::move(metadata));
        if (!status.ok()) {
            std::lock_guard result_lock(result_mutex);
            ++rejected;
        }
    };

    for (std::size_t index = 0; index < kHotTasks; ++index) {
        submit("hot-session", true);
    }
    for (std::size_t turn = 0; turn < kTasksPerNormalSession; ++turn) {
        for (std::size_t session = 0; session < kNormalSessions; ++session) {
            submit("normal-" + std::to_string(session), false);
        }
    }

    {
        std::unique_lock lock(result_mutex);
        completed_cv.wait(lock, [&] { return completed + rejected == kTaskCount; });
    }
    pool.Shutdown(true);
    const auto elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - benchmark_started).count();
    return {
        .scheduler = affinity ? "session_affinity" : "default_fifo",
        .pool_role = std::move(pool_role),
        .all_queue_wait = Summarize(std::move(all_wait)),
        .normal_queue_wait = Summarize(std::move(normal_wait)),
        .elapsed_ms = elapsed_ms,
        .tasks_per_second = static_cast<double>(kTaskCount - rejected) * 1000.0 / elapsed_ms,
        .rejected = rejected,
    };
}

void Print(const ScenarioResult& result) {
    const auto& all = result.all_queue_wait;
    const auto& normal = result.normal_queue_wait;
    std::cout << std::fixed << std::setprecision(3)
              << result.scheduler << ',' << result.pool_role << ','
              << all.average_ms << ',' << all.p50_ms << ',' << all.p95_ms << ',' << all.p99_ms << ','
              << normal.average_ms << ',' << normal.p50_ms << ',' << normal.p95_ms << ',' << normal.p99_ms << ','
              << result.elapsed_ms << ',' << result.tasks_per_second << ',' << result.rejected << '\n';
}

} // namespace

int main() {
    std::cout << "scheduler,pool,all_avg_ms,all_p50_ms,all_p95_ms,all_p99_ms,"
                 "normal_avg_ms,normal_p50_ms,normal_p95_ms,normal_p99_ms,"
                 "elapsed_ms,tasks_per_sec,rejected\n";
    Print(RunScenario(false, "compute", 2ms));
    Print(RunScenario(true, "compute", 2ms));
    Print(RunScenario(false, "io", 8ms));
    Print(RunScenario(true, "io", 8ms));
    return 0;
}
