#include "media_inference_execution.h"

#include "inference_frame_ipc_receiver.h"
#include "inference_frame_shared_memory.h"
#include "memory_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace std::chrono_literals;
using agent::service::persona::MediaInferenceExecution;
using agent::service::persona::MediaInferenceExecutionDependencies;
using agent::service::persona::MediaInferenceExecutionOptions;
using agent::service::persona::MediaInferenceExecutionRuntime;
using agent::service::persona::MediaInferenceExecutionState;
using agent::service::persona::SkillSessionManager;
using agent::service::persona::SkillSessionStartRequest;
using agent::service::persona::SkillSessionState;

std::string UniqueExecutionId() {
    static std::atomic<std::uint64_t> next{0};
    return "media-execution-test-" +
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
           std::to_string(next.fetch_add(1, std::memory_order_relaxed));
}

class SpoolDirectory final {
public:
    explicit SpoolDirectory(std::string_view execution_id)
        : path_(std::filesystem::current_path() / "build" / "media-execution-tests" / execution_id) {
        std::filesystem::create_directories(path_);
    }

    ~SpoolDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

class SharedChannelCleanup final {
public:
    explicit SharedChannelCleanup(std::string name) : name_(std::move(name)) {
        ipc::media::SharedMemoryInferenceFrameChannel::Remove(name_);
    }
    ~SharedChannelCleanup() { ipc::media::SharedMemoryInferenceFrameChannel::Remove(name_); }
    const std::string& name() const noexcept { return name_; }

private:
    std::string name_;
};

class FakeVlm final : public media::IVlmVisionClient {
public:
    core::Result<media::VisionInferenceResult> Analyze(
        const media::VisionInferenceRequest& request) override {
        media::VisionInferenceResult result;
        result.scene_hint = "frame-" + std::to_string(request.frame_id);
        result.confidence = 0.9;
        return result;
    }
};

class RejectingResultTable final : public media::inference::IInferenceFrameResultTable {
public:
    core::Status Publish(media::inference::InferenceFrameResultRecord) override {
        return core::Status::Error(core::ErrorCode::ResourceExhausted, "injected result publish failure");
    }

    core::Result<std::vector<media::inference::InferenceFrameResultRecord>> FinalizeSession(
        std::string_view) override {
        return std::vector<media::inference::InferenceFrameResultRecord>{};
    }

    void Shutdown() override {}
    media::inference::InferenceFrameResultTableSnapshot Snapshot() const override { return {}; }
};

media::inference::OwnedInferenceFrame MakeFrame(
    core::RawMemoryPool& memory_pool,
    std::string execution_id,
    std::string session_id,
    std::uint64_t sequence) {
    std::vector<std::byte> payload(256, std::byte{0x4A});
    media::inference::InferenceFrameMetadata metadata{
        .execution_id = std::move(execution_id),
        .session_id = std::move(session_id),
        .trace_id = "trace-media-execution",
        .selected_sequence = sequence,
        .frame_id = sequence,
        .timestamp_us = static_cast<std::int64_t>(sequence * 1000),
        .width = 320,
        .height = 180,
        .format = media::inference::InferenceFrameFormat::Jpeg,
        .saliency = 0.8,
    };
    auto copied = media::inference::CopyInferenceFrame(memory_pool, std::move(metadata), payload);
    EXPECT_TRUE(copied.ok()) << copied.status().message();
    return std::move(copied).value();
}

struct ExecutionHarness {
    std::string execution_id = UniqueExecutionId();
    std::string session_id = "session-" + execution_id;
    SpoolDirectory directory{execution_id};
    core::BucketMemoryPool memory_pool;
    std::shared_ptr<media::inference::SegmentedInferenceFrameBacklog> backlog =
        std::make_shared<media::inference::SegmentedInferenceFrameBacklog>(
            media::inference::SegmentedFrameBacklogOptions{1, 1, 2, 5ms});
    std::shared_ptr<media::inference::IInferenceFrameSpool> spool;
    std::shared_ptr<media::inference::IInferenceFrameSpoolReplayer> replayer;
    std::shared_ptr<media::inference::SessionInferenceFrameResultTable> results =
        std::make_shared<media::inference::SessionInferenceFrameResultTable>();
    std::shared_ptr<SkillSessionManager> skills = std::make_shared<SkillSessionManager>();
    std::shared_ptr<MediaInferenceExecutionRuntime> runtime;

    ExecutionHarness() {
        auto created_spool = media::inference::MappedInferenceFrameSpool::Create({
            .root_directory = directory.path(),
            .execution_id = execution_id,
            .segment_bytes = 4096,
            .max_spool_bytes = 1024 * 1024,
            .flush_on_append = false,
            .remove_on_destroy = true,
        });
        EXPECT_TRUE(created_spool.ok()) << created_spool.status().message();
        spool = std::shared_ptr<media::inference::IInferenceFrameSpool>(std::move(created_spool).value());
        replayer = std::make_shared<media::inference::InferenceFrameSpoolReplayer>(
            *spool, memory_pool, *backlog, media::inference::InferenceFrameSpoolReplayOptions{2});
        auto created_runtime = MediaInferenceExecutionRuntime::Create();
        EXPECT_TRUE(created_runtime.ok()) << created_runtime.status().message();
        runtime = std::move(created_runtime).value();

        SkillSessionStartRequest start;
        start.execution_id = execution_id;
        start.session_id = session_id;
        start.skill_id = "vision.observe";
        start.trace_id = "trace-media-execution";
        EXPECT_TRUE(skills->Start(start).ok());
        EXPECT_TRUE(skills->MarkReady(session_id, start.skill_id, "ready", start.trace_id).ok());
    }

    MediaInferenceExecutionDependencies Dependencies() {
        return {runtime, backlog, spool, replayer, results, skills};
    }
};

TEST(MediaInferenceExecutionTest, AtomicallyDrainsHotAndSpilledFramesBeforeClosingSkill) {
    ExecutionHarness harness;
    std::atomic<std::size_t> callback_count{0};
    std::mutex completion_mutex;
    std::vector<media::inference::InferenceFrameResultRecord> completed_results;
    auto execution_result = MediaInferenceExecution::Create(
        {harness.execution_id, harness.session_id, "vision.observe", "trace-media-execution", 1ms},
        harness.Dependencies(),
        [&](const auto& completion) {
            callback_count.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard lock(completion_mutex);
            completed_results = completion.results;
            return core::Status::Ok();
        });
    ASSERT_TRUE(execution_result.ok()) << execution_result.status().message();
    auto execution = std::move(execution_result).value();

    FakeVlm vlm;
    media::inference::InferenceFrameCoordinator coordinator(
        *harness.backlog,
        vlm,
        *harness.results,
        {
            .worker_count = 2,
            .wait_timeout = 2ms,
            .shutdown_backlog = false,
            .terminal_observer = [weak = std::weak_ptr<MediaInferenceExecution>(execution)](auto event) {
                if (auto locked = weak.lock()) locked->ObserveTerminal(std::move(event));
            },
        });
    ASSERT_TRUE(coordinator.Start().ok());

    SharedChannelCleanup channel("channel-" + harness.execution_id);
    auto producer = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = channel.name(),
        .slot_count = 4,
        .payload_capacity = 256,
    });
    ASSERT_TRUE(producer.ok()) << producer.status().message();
    auto source = ipc::media::SharedMemoryInferenceFrameChannel::Open({.name = channel.name()});
    ASSERT_TRUE(source.ok()) << source.status().message();
    media::inference::InferenceFrameIpcReceiver receiver(
        *source.value(),
        harness.memory_pool,
        *harness.backlog,
        {},
        {},
        {.admission_sink = execution});
    std::vector<std::byte> payload(256, std::byte{0x4A});
    for (std::uint64_t sequence = 1; sequence <= 10; ++sequence) {
        ipc::media::SharedFramePublishRequest request{
            .execution_id = harness.execution_id,
            .session_id = harness.session_id,
            .trace_id = "trace-media-execution",
            .selected_sequence = sequence,
            .frame_id = sequence,
            .timestamp_us = static_cast<std::int64_t>(sequence * 1000),
            .width = 320,
            .height = 180,
            .format = static_cast<std::uint32_t>(ipc::media::SharedFrameFormat::Jpeg),
            .saliency = 0.8,
            .payload = payload,
        };
        ASSERT_TRUE(producer.value()->Publish(request).ok());
        ASSERT_TRUE(receiver.PollOnce().ok());
    }
    EXPECT_EQ(receiver.Snapshot().submitted_frames, 10u);
    ASSERT_TRUE(execution->RecordBadFrame(
        harness.execution_id,
        11,
        core::Status::Error(core::ErrorCode::InvalidArgument, "decoder rejected frame")).ok());
    ASSERT_TRUE(execution->BeginClosing("input stream ended").ok());

    auto completed = execution->WaitForCompletion(5s);
    ASSERT_TRUE(completed.ok()) << completed.status().message();
    EXPECT_EQ(completed.value().state, MediaInferenceExecutionState::Closed);
    EXPECT_EQ(completed.value().selected_frames, 11u);
    EXPECT_EQ(completed.value().committed_frames, 10u);
    EXPECT_EQ(completed.value().bad_frames, 1u);
    EXPECT_EQ(completed.value().terminal_frames, 10u);
    EXPECT_EQ(completed.value().hot_frames + completed.value().spooled_frames, 10u);
    EXPECT_TRUE(completed.value().replay_complete);
    EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 1u);
    {
        std::lock_guard lock(completion_mutex);
        ASSERT_EQ(completed_results.size(), 10u);
        for (std::size_t index = 0; index < completed_results.size(); ++index) {
            EXPECT_EQ(completed_results[index].frame.selected_sequence, index + 1);
        }
    }
    auto skill = harness.skills->Get(harness.session_id, "vision.observe");
    ASSERT_TRUE(skill.ok());
    ASSERT_TRUE(skill.value().has_value());
    EXPECT_EQ(skill.value()->state, SkillSessionState::Closed);
    coordinator.Shutdown();
}

TEST(MediaInferenceExecutionTest, EmptyExecutionClosesAndInvokesCallbackExactlyOnce) {
    ExecutionHarness harness;
    std::atomic<std::size_t> callback_count{0};
    auto created = MediaInferenceExecution::Create(
        {harness.execution_id, harness.session_id, "vision.observe", "trace-media-execution", 0ms},
        harness.Dependencies(),
        [&](const auto& completion) {
            EXPECT_TRUE(completion.status.ok());
            EXPECT_TRUE(completion.results.empty());
            callback_count.fetch_add(1, std::memory_order_relaxed);
            return core::Status::Ok();
        });
    ASSERT_TRUE(created.ok());
    auto execution = std::move(created).value();
    ASSERT_TRUE(execution->BeginClosing().ok());
    ASSERT_TRUE(execution->BeginClosing().ok());
    auto completed = execution->WaitForCompletion(2s);
    ASSERT_TRUE(completed.ok()) << completed.status().message();
    EXPECT_EQ(completed.value().state, MediaInferenceExecutionState::Closed);
    EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 1u);
    EXPECT_EQ(
        execution->AdmitFrame(MakeFrame(harness.memory_pool, harness.execution_id, harness.session_id, 1)).code(),
        core::ErrorCode::FailedPrecondition);
}

TEST(MediaInferenceExecutionTest, RejectsStaleTerminalAndPreservesClosingStateOnObservation) {
    ExecutionHarness harness;
    auto created = MediaInferenceExecution::Create(
        {harness.execution_id, harness.session_id, "vision.observe", "trace-media-execution", 0ms},
        harness.Dependencies());
    ASSERT_TRUE(created.ok());
    auto execution = std::move(created).value();
    ASSERT_TRUE(execution->BeginClosing("draining").ok());

    agent::service::persona::SkillObservation observation;
    observation.execution_id = harness.execution_id;
    observation.session_id = harness.session_id;
    observation.skill_id = "vision.observe";
    observation.summary = "late but valid observation";
    ASSERT_TRUE(harness.skills->RecordObservation(observation).ok());
    auto skill = harness.skills->Get(harness.session_id, observation.skill_id);
    ASSERT_TRUE(skill.ok());
    ASSERT_TRUE(skill.value().has_value());
    EXPECT_EQ(skill.value()->state, SkillSessionState::Closing);
    EXPECT_EQ(
        harness.skills->MarkReady(harness.session_id, observation.skill_id, "wrong", "trace").code(),
        core::ErrorCode::FailedPrecondition);

    media::inference::InferenceFrameTerminalEvent stale;
    stale.frame.execution_id = "older-execution";
    stale.frame.selected_sequence = 1;
    EXPECT_EQ(execution->ObserveTerminal(std::move(stale)).code(), core::ErrorCode::FailedPrecondition);
    ASSERT_TRUE(execution->WaitForCompletion(2s).ok());
}

TEST(MediaInferenceExecutionTest, ResultPublishFailureFailsExecutionAndCallbackRunsOnce) {
    ExecutionHarness harness;
    auto rejecting_results = std::make_shared<RejectingResultTable>();
    auto dependencies = harness.Dependencies();
    dependencies.result_table = rejecting_results;
    std::atomic<std::size_t> callback_count{0};
    auto created = MediaInferenceExecution::Create(
        {harness.execution_id, harness.session_id, "vision.observe", "trace-media-execution", 0ms},
        std::move(dependencies),
        [&](const auto& completion) {
            EXPECT_FALSE(completion.status.ok());
            callback_count.fetch_add(1, std::memory_order_relaxed);
            return core::Status::Ok();
        });
    ASSERT_TRUE(created.ok());
    auto execution = std::move(created).value();

    FakeVlm vlm;
    media::inference::InferenceFrameCoordinator coordinator(
        *harness.backlog,
        vlm,
        *rejecting_results,
        {
            .worker_count = 1,
            .wait_timeout = 2ms,
            .shutdown_backlog = false,
            .terminal_observer = [weak = std::weak_ptr<MediaInferenceExecution>(execution)](auto event) {
                if (auto locked = weak.lock()) locked->ObserveTerminal(std::move(event));
            },
        });
    ASSERT_TRUE(coordinator.Start().ok());
    ASSERT_TRUE(execution->AdmitFrame(
        MakeFrame(harness.memory_pool, harness.execution_id, harness.session_id, 1)).ok());
    ASSERT_TRUE(execution->BeginClosing().ok());
    auto completed = execution->WaitForCompletion(2s);
    ASSERT_TRUE(completed.ok()) << completed.status().message();
    EXPECT_EQ(completed.value().state, MediaInferenceExecutionState::Failed);
    EXPECT_EQ(completed.value().terminal_status.code(), core::ErrorCode::ResourceExhausted);
    EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 1u);
    coordinator.Shutdown();
}

TEST(MediaInferenceExecutionTest, ConcurrentAdmissionAndCloseHasNoSilentCommittedLoss) {
    ExecutionHarness harness;
    std::atomic<std::size_t> callback_count{0};
    auto created = MediaInferenceExecution::Create(
        {harness.execution_id, harness.session_id, "vision.observe", "trace-media-execution", 0ms},
        harness.Dependencies(),
        [&](const auto&) {
            callback_count.fetch_add(1, std::memory_order_relaxed);
            return core::Status::Ok();
        });
    ASSERT_TRUE(created.ok());
    auto execution = std::move(created).value();
    FakeVlm vlm;
    media::inference::InferenceFrameCoordinator coordinator(
        *harness.backlog,
        vlm,
        *harness.results,
        {
            .worker_count = 4,
            .wait_timeout = 1ms,
            .shutdown_backlog = false,
            .terminal_observer = [weak = std::weak_ptr<MediaInferenceExecution>(execution)](auto event) {
                if (auto locked = weak.lock()) locked->ObserveTerminal(std::move(event));
            },
        });
    ASSERT_TRUE(coordinator.Start().ok());

    std::atomic<std::size_t> accepted{0};
    std::vector<std::jthread> producers;
    for (std::uint64_t worker = 0; worker < 4; ++worker) {
        producers.emplace_back([&, worker] {
            for (std::uint64_t index = 1; index <= 50; ++index) {
                const auto sequence = worker * 50 + index;
                auto status = execution->AdmitFrame(
                    MakeFrame(harness.memory_pool, harness.execution_id, harness.session_id, sequence));
                if (status.ok()) {
                    accepted.fetch_add(1, std::memory_order_relaxed);
                } else {
                    EXPECT_EQ(status.code(), core::ErrorCode::FailedPrecondition);
                }
            }
        });
    }
    std::jthread closer([&] {
        std::this_thread::yield();
        EXPECT_TRUE(execution->BeginClosing("concurrent close").ok());
    });
    producers.clear();
    closer.join();

    auto completed = execution->WaitForCompletion(10s);
    ASSERT_TRUE(completed.ok()) << completed.status().message();
    ASSERT_EQ(completed.value().state, MediaInferenceExecutionState::Closed)
        << completed.value().terminal_status.message();
    EXPECT_EQ(completed.value().committed_frames, accepted.load(std::memory_order_relaxed));
    EXPECT_EQ(completed.value().terminal_frames, completed.value().committed_frames);
    EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 1u);
    coordinator.Shutdown();
}

TEST(MediaInferenceExecutionTest, CompletionCallbackFailurePreventsFalseClosedState) {
    ExecutionHarness harness;
    std::atomic<std::size_t> callback_count{0};
    auto created = MediaInferenceExecution::Create(
        {harness.execution_id, harness.session_id, "vision.observe", "trace-media-execution", 0ms},
        harness.Dependencies(),
        [&](const auto&) {
            callback_count.fetch_add(1, std::memory_order_relaxed);
            return core::Status::Error(core::ErrorCode::Unavailable, "injected final publish failure");
        });
    ASSERT_TRUE(created.ok());
    auto execution = std::move(created).value();
    ASSERT_TRUE(execution->BeginClosing().ok());
    auto completed = execution->WaitForCompletion(2s);
    ASSERT_TRUE(completed.ok());
    EXPECT_EQ(completed.value().state, MediaInferenceExecutionState::Failed);
    EXPECT_EQ(completed.value().terminal_status.code(), core::ErrorCode::Unavailable);
    EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 1u);
}

TEST(MediaInferenceExecutionTest, MediaIoPoolRejectionFailsWithoutDroppingSilently) {
    ExecutionHarness harness;
    std::atomic<std::size_t> callback_count{0};
    auto created = MediaInferenceExecution::Create(
        {harness.execution_id, harness.session_id, "vision.observe", "trace-media-execution", 0ms},
        harness.Dependencies(),
        [&](const auto& completion) {
            EXPECT_FALSE(completion.status.ok());
            callback_count.fetch_add(1, std::memory_order_relaxed);
            return core::Status::Ok();
        });
    ASSERT_TRUE(created.ok());
    auto execution = std::move(created).value();
    harness.runtime->Shutdown(false);
    auto status = execution->AdmitFrame(
        MakeFrame(harness.memory_pool, harness.execution_id, harness.session_id, 1));
    EXPECT_EQ(status.code(), core::ErrorCode::Unavailable);
    auto completed = execution->WaitForCompletion(1s);
    ASSERT_TRUE(completed.ok());
    EXPECT_EQ(completed.value().state, MediaInferenceExecutionState::Failed);
    EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 1u);
}

} // namespace
