#pragma once

#include "inference_frame_backlog.h"
#include "inference_frame_coordinator.h"
#include "inference_frame_spool.h"
#include "inference_frame_spool_replayer.h"
#include "skill_session_manager.h"
#include "task_group.h"
#include "thread_pool.h"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace agent::service::persona {

enum class MediaInferenceExecutionState {
    Running,
    SealingInput,
    ReplayingSpool,
    Aggregating,
    Closed,
    Failed
};

struct MediaInferenceExecutionRuntimeOptions {
    core::ThreadPoolOptions io_pool{2, 256, "media-spool-io"};
    core::ThreadPoolOptions control_pool{1, 64, "media-inference-control"};
    core::ThreadPoolOptions aggregation_pool{1, 64, "media-inference-aggregation"};
};

class MediaInferenceExecutionRuntime final {
public:
    static core::Result<std::shared_ptr<MediaInferenceExecutionRuntime>> Create(
        MediaInferenceExecutionRuntimeOptions options = {});
    ~MediaInferenceExecutionRuntime();

    MediaInferenceExecutionRuntime(const MediaInferenceExecutionRuntime&) = delete;
    MediaInferenceExecutionRuntime& operator=(const MediaInferenceExecutionRuntime&) = delete;

    core::ThreadPool& io_pool() noexcept;
    core::ThreadPool& control_pool() noexcept;
    core::ThreadPool& aggregation_pool() noexcept;
    void Shutdown(bool drain = true);

private:
    explicit MediaInferenceExecutionRuntime(MediaInferenceExecutionRuntimeOptions options);
    core::Status Start();

    core::ThreadPool io_pool_;
    core::ThreadPool control_pool_;
    core::ThreadPool aggregation_pool_;
};

struct MediaInferenceExecutionCompletion {
    std::string execution_id;
    std::string session_id;
    core::Status status = core::Status::Ok();
    std::vector<media::inference::InferenceFrameResultRecord> results;
};

using MediaInferenceCompletionCallback =
    std::function<core::Status(const MediaInferenceExecutionCompletion&)>;

struct MediaInferenceExecutionOptions {
    std::string execution_id;
    std::string session_id;
    std::string skill_id = "vision.observe";
    std::string trace_id;
    std::chrono::milliseconds replay_retry_delay{1};
};

struct MediaInferenceExecutionDependencies {
    std::shared_ptr<MediaInferenceExecutionRuntime> runtime;
    std::shared_ptr<media::inference::IInferenceFrameBacklog> backlog;
    std::shared_ptr<media::inference::IInferenceFrameSpool> spool;
    std::shared_ptr<media::inference::IInferenceFrameSpoolReplayer> replayer;
    std::shared_ptr<media::inference::IInferenceFrameResultTable> result_table;
    std::shared_ptr<ISkillSessionManager> skill_sessions;
};

struct MediaInferenceExecutionSnapshot {
    MediaInferenceExecutionState state = MediaInferenceExecutionState::Running;
    std::size_t selected_frames = 0;
    std::size_t committed_frames = 0;
    std::size_t hot_frames = 0;
    std::size_t spooled_frames = 0;
    std::size_t bad_frames = 0;
    std::size_t terminal_frames = 0;
    std::size_t failed_inference_frames = 0;
    bool replay_complete = false;
    bool completion_callback_invoked = false;
    core::Status terminal_status = core::Status::Ok();
};

class IMediaInferenceExecution : public media::inference::IInferenceFrameAdmissionSink {
public:
    virtual ~IMediaInferenceExecution() = default;
    virtual core::Status AdmitFrame(media::inference::OwnedInferenceFrame frame) = 0;
    virtual core::Status RecordBadFrame(std::string_view execution_id,
                                        std::uint64_t selected_sequence,
                                        core::Status decode_status) = 0;
    virtual core::Status BeginClosing(std::string reason = {}) = 0;
    virtual core::Status ObserveTerminal(media::inference::InferenceFrameTerminalEvent event) = 0;
    virtual core::Result<MediaInferenceExecutionSnapshot> WaitForCompletion(
        std::chrono::milliseconds timeout) const = 0;
    virtual MediaInferenceExecutionSnapshot Snapshot() const = 0;
};

class MediaInferenceExecution final
    : public IMediaInferenceExecution,
      public std::enable_shared_from_this<MediaInferenceExecution> {
public:
    static core::Result<std::shared_ptr<MediaInferenceExecution>> Create(
        MediaInferenceExecutionOptions options,
        MediaInferenceExecutionDependencies dependencies,
        MediaInferenceCompletionCallback completion_callback = {},
        core::LoggerAdapter logger = {});
    ~MediaInferenceExecution() override = default;

    core::Status AdmitFrame(media::inference::OwnedInferenceFrame frame) override;
    core::Status RecordBadFrame(std::string_view execution_id,
                                std::uint64_t selected_sequence,
                                core::Status decode_status) override;
    core::Status BeginClosing(std::string reason = {}) override;
    core::Status ObserveTerminal(media::inference::InferenceFrameTerminalEvent event) override;
    core::Result<MediaInferenceExecutionSnapshot> WaitForCompletion(
        std::chrono::milliseconds timeout) const override;
    MediaInferenceExecutionSnapshot Snapshot() const override;

private:
    MediaInferenceExecution(MediaInferenceExecutionOptions options,
                            MediaInferenceExecutionDependencies dependencies,
                            MediaInferenceCompletionCallback completion_callback,
                            core::LoggerAdapter logger);

    core::Status ProcessAdmission(media::inference::OwnedInferenceFrame frame);
    void OnIngestDrained();
    core::Status ReplayUntilComplete();
    void TryScheduleAggregation();
    core::Status Aggregate();
    void Fail(core::Status status);
    bool TerminalLocked() const noexcept;
    MediaInferenceExecutionSnapshot SnapshotLocked() const;

    MediaInferenceExecutionOptions options_;
    MediaInferenceExecutionDependencies dependencies_;
    MediaInferenceCompletionCallback completion_callback_;
    core::LoggerAdapter logger_;
    mutable std::mutex mutex_;
    mutable std::condition_variable completion_cv_;
    MediaInferenceExecutionState state_ = MediaInferenceExecutionState::Running;
    core::Status terminal_status_ = core::Status::Ok();
    core::TaskGroup ingest_group_;
    std::set<std::uint64_t> selected_sequences_;
    std::set<std::uint64_t> committed_sequences_;
    std::set<std::uint64_t> bad_sequences_;
    std::set<std::uint64_t> terminal_sequences_;
    std::size_t hot_frames_ = 0;
    std::size_t spooled_frames_ = 0;
    std::size_t failed_inference_frames_ = 0;
    bool replay_complete_ = false;
    bool completion_callback_invoked_ = false;
    std::shared_ptr<MediaInferenceExecution> self_keepalive_;
};

} // namespace agent::service::persona
