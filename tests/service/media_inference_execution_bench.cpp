#include "media_inference_execution.h"

#include "memory_pool.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Execution = agent::service::persona::MediaInferenceExecution;

class BenchVlm final : public media::IVlmVisionClient {
public:
    core::Result<media::VisionInferenceResult> Analyze(
        const media::VisionInferenceRequest& request) override {
        media::VisionInferenceResult result;
        result.scene_hint = "bench-" + std::to_string(request.frame_id);
        result.confidence = 0.9;
        return result;
    }
};

struct ExecutionContext {
    std::string id;
    std::string session_id;
    std::shared_ptr<media::inference::IInferenceFrameSpool> spool;
    std::shared_ptr<media::inference::IInferenceFrameSpoolReplayer> replayer;
    std::shared_ptr<Execution> execution;
};

int Fail(std::string_view stage, const core::Status& status) {
    std::cerr << stage << " failed: " << status.message() << '\n';
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    using namespace std::chrono_literals;
    const std::size_t execution_count = argc > 1 ? std::stoull(argv[1]) : 8;
    const std::size_t frames_per_execution = argc > 2 ? std::stoull(argv[2]) : 256;
    const std::size_t payload_bytes = argc > 3 ? std::stoull(argv[3]) : 16 * 1024;
    if (execution_count == 0 || frames_per_execution == 0 || payload_bytes == 0) {
        std::cerr << "execution_count, frames_per_execution, and payload_bytes must be positive\n";
        return 2;
    }

    const auto run_id = "execution-bench-" + std::to_string(Clock::now().time_since_epoch().count());
    const auto root = std::filesystem::current_path() / "build" / run_id;
    std::filesystem::create_directories(root);
    struct Cleanup {
        std::filesystem::path root;
        ~Cleanup() { std::error_code ec; std::filesystem::remove_all(root, ec); }
    } cleanup{root};

    auto runtime_result = agent::service::persona::MediaInferenceExecutionRuntime::Create({
        .io_pool = {4, execution_count * frames_per_execution + 16, "media-bench-io"},
        .control_pool = {2, execution_count * 2, "media-bench-control"},
        .aggregation_pool = {2, execution_count * 2, "media-bench-aggregation"},
    });
    if (!runtime_result.ok()) return Fail("runtime", runtime_result.status());
    auto runtime = std::move(runtime_result).value();
    core::BucketMemoryPool memory_pool;
    auto backlog = std::make_shared<media::inference::SegmentedInferenceFrameBacklog>(
        media::inference::SegmentedFrameBacklogOptions{
            execution_count,
            1,
            8,
            1ms,
        });
    auto results = std::make_shared<media::inference::SessionInferenceFrameResultTable>(
        media::inference::InferenceFrameResultTableOptions{
            execution_count,
            frames_per_execution,
        });
    auto skills = std::make_shared<agent::service::persona::SkillSessionManager>();
    std::mutex registry_mutex;
    std::unordered_map<std::string, std::weak_ptr<Execution>> registry;
    std::atomic<std::size_t> callback_count{0};
    std::atomic<std::size_t> callback_results{0};
    std::vector<ExecutionContext> executions;
    executions.reserve(execution_count);

    for (std::size_t index = 0; index < execution_count; ++index) {
        ExecutionContext context;
        context.id = run_id + "-" + std::to_string(index);
        context.session_id = "session-" + context.id;
        auto spool_result = media::inference::MappedInferenceFrameSpool::Create({
            .root_directory = root,
            .execution_id = context.id,
            .segment_bytes = 16 * 1024 * 1024,
            .max_spool_bytes = frames_per_execution * payload_bytes + 16 * 1024 * 1024,
            .flush_on_append = false,
            .remove_on_destroy = true,
        });
        if (!spool_result.ok()) return Fail("spool", spool_result.status());
        context.spool = std::shared_ptr<media::inference::IInferenceFrameSpool>(
            std::move(spool_result).value());
        context.replayer = std::make_shared<media::inference::InferenceFrameSpoolReplayer>(
            *context.spool,
            memory_pool,
            *backlog,
            media::inference::InferenceFrameSpoolReplayOptions{16});

        agent::service::persona::SkillSessionStartRequest start;
        start.execution_id = context.id;
        start.session_id = context.session_id;
        start.skill_id = "vision.observe";
        start.trace_id = "bench-trace";
        auto started = skills->Start(start);
        if (!started.ok()) return Fail("skill start", started.status());
        auto ready = skills->MarkReady(start.session_id, start.skill_id, "ready", start.trace_id);
        if (!ready.ok()) return Fail("skill ready", ready);

        auto execution_result = Execution::Create(
            {context.id, context.session_id, start.skill_id, start.trace_id, 0ms},
            {runtime, backlog, context.spool, context.replayer, results, skills},
            [&](const auto& completion) {
                callback_count.fetch_add(1, std::memory_order_relaxed);
                callback_results.fetch_add(completion.results.size(), std::memory_order_relaxed);
                return core::Status::Ok();
            });
        if (!execution_result.ok()) return Fail("execution", execution_result.status());
        context.execution = std::move(execution_result).value();
        {
            std::lock_guard lock(registry_mutex);
            registry.emplace(context.id, context.execution);
        }
        executions.push_back(std::move(context));
    }

    BenchVlm vlm;
    media::inference::InferenceFrameCoordinator coordinator(
        *backlog,
        vlm,
        *results,
        {
            .worker_count = 4,
            .wait_timeout = 1ms,
            .shutdown_backlog = false,
            .terminal_observer = [&](auto event) {
                std::shared_ptr<Execution> execution;
                {
                    std::lock_guard lock(registry_mutex);
                    if (auto it = registry.find(event.frame.execution_id); it != registry.end()) {
                        execution = it->second.lock();
                    }
                }
                if (execution) execution->ObserveTerminal(std::move(event));
            },
        });
    std::vector<std::byte> payload(payload_bytes, std::byte{0x5A});
    const auto started_at = Clock::now();
    for (std::size_t frame_index = 0; frame_index < frames_per_execution; ++frame_index) {
        for (auto& context : executions) {
            const auto sequence = static_cast<std::uint64_t>(frame_index + 1);
            media::inference::InferenceFrameMetadata metadata{
                .execution_id = context.id,
                .session_id = context.session_id,
                .trace_id = "bench-trace",
                .selected_sequence = sequence,
                .frame_id = sequence,
                .timestamp_us = static_cast<std::int64_t>(sequence * 33'333),
                .width = 640,
                .height = 360,
                .format = media::inference::InferenceFrameFormat::Jpeg,
                .saliency = 0.8,
            };
            auto frame = media::inference::CopyInferenceFrame(memory_pool, std::move(metadata), payload);
            if (!frame.ok()) return Fail("frame copy", frame.status());
            auto admitted = context.execution->AdmitFrame(std::move(frame).value());
            if (!admitted.ok()) return Fail("frame admission", admitted);
        }
    }
    const auto admission_deadline = Clock::now() + 60s;
    for (;;) {
        std::size_t committed = 0;
        for (const auto& context : executions) committed += context.execution->Snapshot().committed_frames;
        if (committed == execution_count * frames_per_execution) break;
        if (Clock::now() >= admission_deadline) {
            std::cerr << "admission fence timed out\n";
            return 1;
        }
        std::this_thread::sleep_for(1ms);
    }
    for (auto& context : executions) {
        auto closing = context.execution->BeginClosing("benchmark input complete");
        if (!closing.ok()) return Fail("begin closing", closing);
    }
    auto coordinator_status = coordinator.Start();
    if (!coordinator_status.ok()) return Fail("coordinator", coordinator_status);

    std::size_t hot_frames = 0;
    std::size_t spooled_frames = 0;
    for (auto& context : executions) {
        auto completed = context.execution->WaitForCompletion(60s);
        if (!completed.ok()) return Fail("wait", completed.status());
        if (completed.value().state != agent::service::persona::MediaInferenceExecutionState::Closed ||
            completed.value().committed_frames != frames_per_execution ||
            completed.value().terminal_frames != frames_per_execution) {
            std::cerr << "execution fence mismatch for " << context.id << '\n';
            return 1;
        }
        hot_frames += completed.value().hot_frames;
        spooled_frames += completed.value().spooled_frames;
    }
    const auto finished_at = Clock::now();
    coordinator.Shutdown();

    const auto total_frames = execution_count * frames_per_execution;
    const auto seconds = std::chrono::duration<double>(finished_at - started_at).count();
    const auto total_mib = static_cast<double>(total_frames * payload_bytes) / (1024.0 * 1024.0);
    std::cout << "executions=" << execution_count
              << " frames=" << total_frames
              << " payload_bytes=" << payload_bytes
              << " hot=" << hot_frames
              << " spooled=" << spooled_frames << '\n';
    std::cout << "elapsed_s=" << seconds
              << " frames_s=" << static_cast<double>(total_frames) / seconds
              << " payload_mib_s=" << total_mib / seconds
              << " callbacks=" << callback_count.load(std::memory_order_relaxed)
              << " callback_results=" << callback_results.load(std::memory_order_relaxed) << '\n';
    return callback_count == execution_count && callback_results == total_frames ? 0 : 1;
}
