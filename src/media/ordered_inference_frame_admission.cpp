#include "ordered_inference_frame_admission.h"

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace media::inference {

class OrderedInferenceFrameAdmission::Impl {
public:
    struct ExecutionState {
        ExecutionState(std::string session, std::size_t capacity)
            : session_id(std::move(session)),
              window(core::OrderedBitmapWindowOptions{.capacity = capacity}) {}

        std::string session_id;
        core::OrderedBitmapWindow<OwnedInferenceFrame> window;
        std::mutex drain_mutex;
    };

    Impl(std::shared_ptr<IInferenceFrameAdmissionSink> downstream,
         OrderedInferenceFrameAdmissionOptions options,
         core::LoggerAdapter logger)
        : downstream_(std::move(downstream)),
          options_(options),
          logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("frame-ordering")) {}

    core::Status AdmitFrame(OwnedInferenceFrame frame) {
        const auto metadata = frame.metadata();
        if (metadata.transport_sequence == 0) {
            return Reject(core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "ordered IPC admission requires transport_sequence"));
        }
        auto state = GetOrCreate(metadata.session_id, metadata.execution_id);
        if (!state.ok()) return Reject(state.status());
        auto status = state.value()->window.Admit(metadata.transport_sequence, std::move(frame));
        if (!status.ok()) return Reject(status);
        admitted_frames_.fetch_add(1, std::memory_order_relaxed);
        return Drain(*state.value());
    }

    core::Status SealExecution(std::string_view session_id,
                               std::string_view execution_id,
                               std::uint64_t final_transport_sequence) {
        auto state = Find(session_id, execution_id);
        if (!state.ok()) {
            if (final_transport_sequence == 0 && state.status().code() == core::ErrorCode::NotFound) {
                return core::Status::Ok();
            }
            return Reject(state.status());
        }
        auto status = state.value()->window.Seal(final_transport_sequence);
        if (!status.ok()) return Reject(status);
        status = Drain(*state.value());
        if (!status.ok()) return status;
        if (!state.value()->window.Snapshot().drained) {
            return Reject(core::Status::Error(
                core::ErrorCode::FailedPrecondition,
                "ordered IPC admission has a transport sequence gap"));
        }
        std::lock_guard lock(executions_mutex_);
        executions_.erase(std::string(execution_id));
        return core::Status::Ok();
    }

    OrderedInferenceFrameAdmissionSnapshot Snapshot() const {
        OrderedInferenceFrameAdmissionSnapshot snapshot;
        {
            std::lock_guard lock(executions_mutex_);
            snapshot.execution_count = executions_.size();
        }
        snapshot.admitted_frames = admitted_frames_.load(std::memory_order_relaxed);
        snapshot.forwarded_frames = forwarded_frames_.load(std::memory_order_relaxed);
        snapshot.rejected_frames = rejected_frames_.load(std::memory_order_relaxed);
        return snapshot;
    }

private:
    core::Result<std::shared_ptr<ExecutionState>> GetOrCreate(
        std::string_view session_id,
        std::string_view execution_id) {
        if (!downstream_ || session_id.empty() || execution_id.empty()) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "ordered IPC admission identity or downstream is missing");
        }
        std::lock_guard lock(executions_mutex_);
        if (auto it = executions_.find(std::string(execution_id)); it != executions_.end()) {
            if (it->second->session_id != session_id) {
                return core::Status::Error(
                    core::ErrorCode::FailedPrecondition,
                    "ordered IPC execution belongs to another session");
            }
            return it->second;
        }
        if (executions_.size() >= options_.max_executions) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "ordered IPC execution limit reached");
        }
        auto state = std::make_shared<ExecutionState>(std::string(session_id), options_.window_capacity);
        executions_.emplace(std::string(execution_id), state);
        return state;
    }

    core::Result<std::shared_ptr<ExecutionState>> Find(
        std::string_view session_id,
        std::string_view execution_id) const {
        std::lock_guard lock(executions_mutex_);
        const auto it = executions_.find(std::string(execution_id));
        if (it == executions_.end()) {
            return core::Status::Error(core::ErrorCode::NotFound, "ordered IPC execution not found");
        }
        if (it->second->session_id != session_id) {
            return core::Status::Error(
                core::ErrorCode::FailedPrecondition,
                "ordered IPC execution belongs to another session");
        }
        return it->second;
    }

    core::Status Drain(ExecutionState& state) {
        std::lock_guard drain_lock(state.drain_mutex);
        for (;;) {
            auto next = state.window.TryTake();
            if (!next.ok()) {
                if (next.status().code() == core::ErrorCode::NotFound ||
                    next.status().code() == core::ErrorCode::Cancelled) {
                    return core::Status::Ok();
                }
                return Reject(next.status());
            }
            if (next.value().skipped()) {
                return Reject(core::Status::Error(
                    core::ErrorCode::FailedPrecondition,
                    "transport sequence must not contain skipped frames"));
            }
            auto status = downstream_->AdmitFrame(std::move(next).value().value.value());
            if (!status.ok()) return Reject(status);
            forwarded_frames_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    core::Status Reject(core::Status status) {
        rejected_frames_.fetch_add(1, std::memory_order_relaxed);
        logger_.warn("[ordered-frame-admission] code={} message={}",
                     static_cast<int>(status.code()),
                     status.message());
        return status;
    }

    std::shared_ptr<IInferenceFrameAdmissionSink> downstream_;
    OrderedInferenceFrameAdmissionOptions options_;
    core::LoggerAdapter logger_;
    mutable std::mutex executions_mutex_;
    std::unordered_map<std::string, std::shared_ptr<ExecutionState>> executions_;
    std::atomic<std::size_t> admitted_frames_{0};
    std::atomic<std::size_t> forwarded_frames_{0};
    std::atomic<std::size_t> rejected_frames_{0};
};

OrderedInferenceFrameAdmission::OrderedInferenceFrameAdmission(
    std::shared_ptr<IInferenceFrameAdmissionSink> downstream,
    OrderedInferenceFrameAdmissionOptions options,
    core::LoggerAdapter logger)
    : impl_(std::make_unique<Impl>(std::move(downstream), options, std::move(logger))) {}

OrderedInferenceFrameAdmission::~OrderedInferenceFrameAdmission() = default;

core::Status OrderedInferenceFrameAdmission::AdmitFrame(OwnedInferenceFrame frame) {
    return impl_->AdmitFrame(std::move(frame));
}

core::Status OrderedInferenceFrameAdmission::SealExecution(
    std::string_view session_id,
    std::string_view execution_id,
    std::uint64_t final_transport_sequence) {
    return impl_->SealExecution(session_id, execution_id, final_transport_sequence);
}

OrderedInferenceFrameAdmissionSnapshot OrderedInferenceFrameAdmission::Snapshot() const {
    return impl_->Snapshot();
}

} // namespace media::inference
