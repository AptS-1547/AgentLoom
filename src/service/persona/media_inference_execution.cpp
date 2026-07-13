#include "media_inference_execution.h"

#include <algorithm>
#include <exception>
#include <thread>
#include <utility>

namespace agent::service::persona {

MediaInferenceExecutionRuntime::MediaInferenceExecutionRuntime(
    MediaInferenceExecutionRuntimeOptions options)
    : io_pool_(std::move(options.io_pool)),
      control_pool_(std::move(options.control_pool)),
      aggregation_pool_(std::move(options.aggregation_pool)) {}

core::Result<std::shared_ptr<MediaInferenceExecutionRuntime>>
MediaInferenceExecutionRuntime::Create(MediaInferenceExecutionRuntimeOptions options) {
    auto runtime = std::shared_ptr<MediaInferenceExecutionRuntime>(
        new MediaInferenceExecutionRuntime(std::move(options)));
    auto status = runtime->Start();
    if (!status.ok()) {
        runtime->Shutdown(false);
        return status;
    }
    return runtime;
}

MediaInferenceExecutionRuntime::~MediaInferenceExecutionRuntime() {
    Shutdown(true);
}

core::Status MediaInferenceExecutionRuntime::Start() {
    auto status = io_pool_.Start();
    if (!status.ok()) return status;
    status = control_pool_.Start();
    if (!status.ok()) return status;
    return aggregation_pool_.Start();
}

void MediaInferenceExecutionRuntime::Shutdown(bool drain) {
    io_pool_.Shutdown(drain);
    control_pool_.Shutdown(drain);
    aggregation_pool_.Shutdown(drain);
}

core::ThreadPool& MediaInferenceExecutionRuntime::io_pool() noexcept { return io_pool_; }
core::ThreadPool& MediaInferenceExecutionRuntime::control_pool() noexcept { return control_pool_; }
core::ThreadPool& MediaInferenceExecutionRuntime::aggregation_pool() noexcept { return aggregation_pool_; }

MediaInferenceExecution::MediaInferenceExecution(
    MediaInferenceExecutionOptions options,
    MediaInferenceExecutionDependencies dependencies,
    MediaInferenceCompletionCallback completion_callback,
    core::LoggerAdapter logger)
    : options_(std::move(options)),
      dependencies_(std::move(dependencies)),
      completion_callback_(std::move(completion_callback)),
      logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("media-execution")) {}

core::Result<std::shared_ptr<MediaInferenceExecution>> MediaInferenceExecution::Create(
    MediaInferenceExecutionOptions options,
    MediaInferenceExecutionDependencies dependencies,
    MediaInferenceCompletionCallback completion_callback,
    core::LoggerAdapter logger) {
    if (options.execution_id.empty() || options.session_id.empty() || options.skill_id.empty()) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "media execution, session, and skill identifiers are required");
    }
    if (options.replay_retry_delay.count() < 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "replay retry delay must not be negative");
    }
    if (!dependencies.runtime || !dependencies.backlog || !dependencies.spool ||
        !dependencies.replayer || !dependencies.result_table || !dependencies.skill_sessions) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "media execution dependencies are incomplete");
    }
    return std::shared_ptr<MediaInferenceExecution>(new MediaInferenceExecution(
        std::move(options),
        std::move(dependencies),
        std::move(completion_callback),
        std::move(logger)));
}

core::Status MediaInferenceExecution::AdmitFrame(media::inference::OwnedInferenceFrame frame) {
    if (!frame.valid()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "encoded inference frame is invalid");
    }
    const auto metadata = frame.metadata();
    if (metadata.execution_id != options_.execution_id || metadata.session_id != options_.session_id ||
        metadata.selected_sequence == 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "inference frame identity is invalid");
    }
    {
        std::lock_guard lock(mutex_);
        if (state_ != MediaInferenceExecutionState::Running) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "media execution is not accepting frames");
        }
        if (!selected_sequences_.insert(metadata.selected_sequence).second) {
            return core::Status::Error(core::ErrorCode::AlreadyExists, "selected frame sequence already exists");
        }
    }

    std::weak_ptr<MediaInferenceExecution> weak = shared_from_this();
    auto pending_frame = std::make_shared<std::optional<media::inference::OwnedInferenceFrame>>(
        std::move(frame));
    auto status = dependencies_.runtime->io_pool().Submit(
        ingest_group_,
        [weak, pending_frame]() mutable {
            auto self = weak.lock();
            if (!self) {
                return core::Status::Error(core::ErrorCode::Cancelled, "media execution was released");
            }
            if (!pending_frame->has_value()) {
                return core::Status::Error(core::ErrorCode::FailedPrecondition, "inference frame was already consumed");
            }
            auto owned_frame = std::move(pending_frame->value());
            pending_frame->reset();
            return self->ProcessAdmission(std::move(owned_frame));
        },
        {},
        "media-frame-admission");
    if (!status.ok()) {
        Fail(status);
    }
    return status;
}

core::Status MediaInferenceExecution::ProcessAdmission(media::inference::OwnedInferenceFrame frame) {
    const auto sequence = frame.metadata().selected_sequence;
    auto admission = dependencies_.backlog->TrySubmit(std::move(frame));
    bool spooled = false;
    if (!admission.accepted() && admission.status.code() == core::ErrorCode::ResourceExhausted &&
        admission.rejected_frame.has_value()) {
        auto rejected = std::move(admission.rejected_frame).value();
        media::inference::SpoolFrameMetadata metadata{
            .execution_id = options_.execution_id,
            .selected_sequence = sequence,
            .frame = rejected.metadata(),
        };
        auto appended = dependencies_.spool->Append(metadata, rejected.bytes());
        if (!appended.ok()) {
            Fail(appended.status());
            return appended.status();
        }
        spooled = true;
    } else if (!admission.accepted()) {
        Fail(admission.status);
        return admission.status;
    }

    {
        std::lock_guard lock(mutex_);
        committed_sequences_.insert(sequence);
        if (spooled) ++spooled_frames_;
        else ++hot_frames_;
    }
    TryScheduleAggregation();
    return core::Status::Ok();
}

core::Status MediaInferenceExecution::RecordBadFrame(
    std::string_view execution_id,
    std::uint64_t selected_sequence,
    core::Status decode_status) {
    if (execution_id != options_.execution_id || selected_sequence == 0 || decode_status.ok()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "bad frame record is invalid");
    }
    std::lock_guard lock(mutex_);
    if (state_ != MediaInferenceExecutionState::Running) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "media execution is not accepting frames");
    }
    if (!selected_sequences_.insert(selected_sequence).second) {
        return core::Status::Error(core::ErrorCode::AlreadyExists, "selected frame sequence already exists");
    }
    bad_sequences_.insert(selected_sequence);
    logger_.debug("[media-execution] bad frame dropped execution={} sequence={} code={} message={}",
                  options_.execution_id,
                  selected_sequence,
                  static_cast<int>(decode_status.code()),
                  decode_status.message());
    return core::Status::Ok();
}

core::Status MediaInferenceExecution::BeginClosing(std::string reason) {
    auto keep_alive = shared_from_this();
    {
        std::lock_guard lock(mutex_);
        if (state_ == MediaInferenceExecutionState::Closed) return core::Status::Ok();
        if (state_ == MediaInferenceExecutionState::Failed) return terminal_status_;
        if (state_ != MediaInferenceExecutionState::Running) return core::Status::Ok();
        state_ = MediaInferenceExecutionState::SealingInput;
        self_keepalive_ = keep_alive;
    }
    auto status = dependencies_.skill_sessions->BeginClosing(
        options_.session_id, options_.skill_id, options_.execution_id, std::move(reason), options_.trace_id);
    if (!status.ok()) {
        Fail(status);
        return status;
    }

    std::weak_ptr<MediaInferenceExecution> weak = shared_from_this();
    status = ingest_group_.OnDrained([weak] {
        if (auto self = weak.lock()) self->OnIngestDrained();
    });
    if (status.ok()) status = ingest_group_.Seal();
    if (!status.ok()) Fail(status);
    return status;
}

void MediaInferenceExecution::OnIngestDrained() {
    const auto ingest = ingest_group_.Snapshot();
    if (!ingest.first_failure.ok()) {
        Fail(ingest.first_failure);
        return;
    }
    auto status = dependencies_.spool->Seal();
    if (!status.ok()) {
        Fail(status);
        return;
    }
    bool invalid_terminal = false;
    {
        std::lock_guard lock(mutex_);
        if (state_ != MediaInferenceExecutionState::SealingInput) return;
        for (const auto sequence : terminal_sequences_) {
            if (!committed_sequences_.contains(sequence)) {
                invalid_terminal = true;
                break;
            }
        }
        if (!invalid_terminal) {
            state_ = MediaInferenceExecutionState::ReplayingSpool;
        }
    }
    if (invalid_terminal) {
        Fail(core::Status::Error(core::ErrorCode::FailedPrecondition, "terminal frame was not committed"));
        return;
    }
    std::weak_ptr<MediaInferenceExecution> weak = shared_from_this();
    status = dependencies_.runtime->control_pool().Submit(
        [weak] {
            auto self = weak.lock();
            if (!self) return core::Status::Error(core::ErrorCode::Cancelled, "media execution was released");
            return self->ReplayUntilComplete();
        },
        {},
        "media-spool-replay");
    if (!status.ok()) Fail(status);
}

core::Status MediaInferenceExecution::ReplayUntilComplete() {
    while (true) {
        {
            std::lock_guard lock(mutex_);
            if (state_ == MediaInferenceExecutionState::Failed) return terminal_status_;
        }
        auto replay = dependencies_.replayer->Pump();
        if (!replay.ok()) {
            Fail(replay.status());
            return replay.status();
        }
        if (replay.value().complete) break;
        if (options_.replay_retry_delay.count() > 0) std::this_thread::sleep_for(options_.replay_retry_delay);
    }
    {
        std::lock_guard lock(mutex_);
        replay_complete_ = true;
    }
    TryScheduleAggregation();
    return core::Status::Ok();
}

core::Status MediaInferenceExecution::ObserveTerminal(
    media::inference::InferenceFrameTerminalEvent event) {
    if (event.frame.execution_id != options_.execution_id) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "terminal callback belongs to a stale execution");
    }
    if (!event.publish_status.ok()) {
        Fail(event.publish_status);
        return event.publish_status;
    }
    bool invalid_terminal = false;
    {
        std::lock_guard lock(mutex_);
        if (TerminalLocked()) {
            return core::Status::Error(core::ErrorCode::FailedPrecondition, "media execution is terminal");
        }
        if (!terminal_sequences_.insert(event.frame.selected_sequence).second) {
            return core::Status::Error(core::ErrorCode::AlreadyExists, "terminal frame sequence already exists");
        }
        if (!event.inference_status.ok()) ++failed_inference_frames_;
        if (state_ != MediaInferenceExecutionState::Running &&
            state_ != MediaInferenceExecutionState::SealingInput &&
            !committed_sequences_.contains(event.frame.selected_sequence)) {
            invalid_terminal = true;
        }
    }
    if (invalid_terminal) {
        auto status = core::Status::Error(core::ErrorCode::FailedPrecondition, "terminal frame was not committed");
        Fail(status);
        return status;
    }
    TryScheduleAggregation();
    return core::Status::Ok();
}

void MediaInferenceExecution::TryScheduleAggregation() {
    {
        std::lock_guard lock(mutex_);
        if (state_ != MediaInferenceExecutionState::ReplayingSpool || !replay_complete_ ||
            terminal_sequences_ != committed_sequences_) return;
        state_ = MediaInferenceExecutionState::Aggregating;
    }
    std::weak_ptr<MediaInferenceExecution> weak = shared_from_this();
    auto status = dependencies_.runtime->aggregation_pool().Submit(
        [weak] {
            auto self = weak.lock();
            if (!self) return core::Status::Error(core::ErrorCode::Cancelled, "media execution was released");
            return self->Aggregate();
        },
        {},
        "media-result-aggregation");
    if (!status.ok()) Fail(status);
}

core::Status MediaInferenceExecution::Aggregate() {
    auto finalized = dependencies_.result_table->FinalizeExecution(options_.execution_id);
    std::vector<media::inference::InferenceFrameResultRecord> results;
    if (!finalized.ok()) {
        bool empty_execution = false;
        {
            std::lock_guard lock(mutex_);
            empty_execution = committed_sequences_.empty();
        }
        if (finalized.status().code() != core::ErrorCode::NotFound || !empty_execution) {
            const auto status = finalized.status();
            Fail(status);
            return status;
        }
    } else {
        results = std::move(finalized).value();
    }

    std::set<std::uint64_t> result_sequences;
    bool result_fence_valid = true;
    for (const auto& record : results) {
        if (record.frame.execution_id != options_.execution_id ||
            !result_sequences.insert(record.frame.selected_sequence).second) {
            result_fence_valid = false;
            break;
        }
    }
    {
        std::lock_guard lock(mutex_);
        result_fence_valid = result_fence_valid && result_sequences == committed_sequences_;
    }
    if (!result_fence_valid) {
        auto status = core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "finalized inference results do not match the committed frame fence");
        Fail(status);
        return status;
    }
    std::sort(results.begin(), results.end(), [](const auto& lhs, const auto& rhs) {
        return media::inference::InferenceFrameOrderKey{
                   lhs.frame.selected_sequence,
                   lhs.frame.timestamp_us,
                   lhs.frame.frame_id} <
               media::inference::InferenceFrameOrderKey{
                   rhs.frame.selected_sequence,
                   rhs.frame.timestamp_us,
                   rhs.frame.frame_id};
    });

    MediaInferenceExecutionCompletion completion{
        .execution_id = options_.execution_id,
        .session_id = options_.session_id,
        .status = core::Status::Ok(),
        .results = std::move(results),
    };
    {
        std::lock_guard lock(mutex_);
        completion_callback_invoked_ = true;
    }
    if (completion_callback_) {
        core::Status status;
        try {
            status = completion_callback_(completion);
        } catch (const std::exception&) {
            status = core::Status::Error(core::ErrorCode::InternalError, "media completion callback failed unexpectedly");
        } catch (...) {
            status = core::Status::Error(core::ErrorCode::Unknown, "media completion callback failed unexpectedly");
        }
        if (!status.ok()) {
            Fail(status);
            return status;
        }
    }
    auto status = dependencies_.skill_sessions->CompleteClosing(
        options_.session_id, options_.skill_id, options_.execution_id, "media inference complete", options_.trace_id);
    if (!status.ok()) {
        Fail(status);
        return status;
    }
    {
        std::lock_guard lock(mutex_);
        state_ = MediaInferenceExecutionState::Closed;
        terminal_status_ = core::Status::Ok();
        self_keepalive_.reset();
    }
    completion_cv_.notify_all();
    return core::Status::Ok();
}

void MediaInferenceExecution::Fail(core::Status status) {
    if (status.ok()) status = core::Status::Error(core::ErrorCode::Unknown, "media execution failed");
    bool invoke_callback = false;
    {
        std::lock_guard lock(mutex_);
        if (TerminalLocked()) return;
        state_ = MediaInferenceExecutionState::Failed;
        terminal_status_ = status;
        invoke_callback = !completion_callback_invoked_;
        completion_callback_invoked_ = true;
        self_keepalive_.reset();
    }
    logger_.warn("[media-execution] failed execution={} code={} message={}",
                 options_.execution_id, static_cast<int>(status.code()), status.message());
    const auto skill_status = dependencies_.skill_sessions->MarkFailed(
        options_.session_id, options_.skill_id, status.message(), options_.trace_id, options_.execution_id);
    if (!skill_status.ok()) {
        logger_.warn("[media-execution] failed to mark skill failed execution={} code={} message={}",
                     options_.execution_id,
                     static_cast<int>(skill_status.code()),
                     skill_status.message());
    }
    if (invoke_callback && completion_callback_) {
        try {
            completion_callback_({options_.execution_id, options_.session_id, status, {}});
        } catch (const std::exception&) {
            logger_.warn("[media-execution] failure completion callback threw execution={}", options_.execution_id);
        } catch (...) {
            logger_.warn("[media-execution] failure completion callback threw execution={}", options_.execution_id);
        }
    }
    completion_cv_.notify_all();
}

bool MediaInferenceExecution::TerminalLocked() const noexcept {
    return state_ == MediaInferenceExecutionState::Closed || state_ == MediaInferenceExecutionState::Failed;
}

MediaInferenceExecutionSnapshot MediaInferenceExecution::SnapshotLocked() const {
    return {
        .state = state_,
        .selected_frames = selected_sequences_.size(),
        .committed_frames = committed_sequences_.size(),
        .hot_frames = hot_frames_,
        .spooled_frames = spooled_frames_,
        .bad_frames = bad_sequences_.size(),
        .terminal_frames = terminal_sequences_.size(),
        .failed_inference_frames = failed_inference_frames_,
        .replay_complete = replay_complete_,
        .completion_callback_invoked = completion_callback_invoked_,
        .terminal_status = terminal_status_,
    };
}

MediaInferenceExecutionSnapshot MediaInferenceExecution::Snapshot() const {
    std::lock_guard lock(mutex_);
    return SnapshotLocked();
}

core::Result<MediaInferenceExecutionSnapshot> MediaInferenceExecution::WaitForCompletion(
    std::chrono::milliseconds timeout) const {
    if (timeout.count() < 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "completion timeout must not be negative");
    }
    std::unique_lock lock(mutex_);
    if (!completion_cv_.wait_for(lock, timeout, [this] { return TerminalLocked(); })) {
        return core::Status::Error(core::ErrorCode::Timeout, "media execution completion timed out");
    }
    return SnapshotLocked();
}

} // namespace agent::service::persona
