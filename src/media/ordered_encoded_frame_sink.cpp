#include "ordered_encoded_frame_sink.h"

#include <atomic>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <utility>

namespace media {

class OrderedEncodedFrameSink::Impl {
public:
    struct ExecutionState {
        ExecutionState(std::string session, std::size_t capacity)
            : session_id(std::move(session)),
              window(core::OrderedBitmapWindowOptions{.capacity = capacity}) {}

        std::string session_id;
        core::OrderedBitmapWindow<EncodedVideoFrame> window;
        std::mutex drain_mutex;
        std::optional<EncodedVideoFrame> pending_publish;
        std::optional<std::int64_t> last_published_timestamp_us;
        std::uint64_t next_transport_sequence = 1;
    };

    Impl(std::shared_ptr<IEncodedVideoFrameSink> downstream,
         OrderedEncodedFrameSinkOptions options,
         core::LoggerAdapter logger)
        : downstream_(std::move(downstream)),
          options_(options),
          logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("frame-ordering")) {}

    core::Status Publish(EncodedVideoFrame frame) {
        const auto& metadata = frame.metadata();
        auto state = GetOrCreate(metadata.session_id, metadata.execution_id);
        if (!state.ok()) return Reject(state.status());
        auto status = state.value()->window.Admit(metadata.selected_sequence, std::move(frame));
        if (!status.ok()) return Reject(status);
        admitted_frames_.fetch_add(1, std::memory_order_relaxed);
        return Drain(*state.value());
    }

    core::Status MarkSkipped(std::string_view session_id,
                             std::string_view execution_id,
                             std::uint64_t selected_sequence,
                             core::Status reason) {
        auto state = GetOrCreate(session_id, execution_id);
        if (!state.ok()) return Reject(state.status());
        auto status = state.value()->window.MarkSkipped(selected_sequence, std::move(reason));
        if (!status.ok()) return Reject(status);
        skipped_frames_.fetch_add(1, std::memory_order_relaxed);
        return Drain(*state.value());
    }

    core::Result<std::uint64_t> SealExecution(
        std::string_view session_id,
        std::string_view execution_id,
        std::uint64_t final_selected_sequence) {
        auto state = Find(session_id, execution_id);
        if (!state.ok()) {
            if (final_selected_sequence == 0 && state.status().code() == core::ErrorCode::NotFound) {
                return std::uint64_t{0};
            }
            return Reject(state.status());
        }
        auto status = state.value()->window.Seal(final_selected_sequence);
        if (!status.ok()) return Reject(status);
        status = Drain(*state.value());
        if (!status.ok()) return status;
        if (!state.value()->window.Snapshot().drained || state.value()->pending_publish.has_value()) {
            return Reject(core::Status::Error(
                core::ErrorCode::FailedPrecondition,
                "ordered encoded frame execution has a sequence gap"));
        }
        const auto final_transport_sequence = state.value()->next_transport_sequence - 1;
        {
            std::lock_guard lock(executions_mutex_);
            executions_.erase(std::string(execution_id));
        }
        return final_transport_sequence;
    }

    OrderedEncodedFrameSinkSnapshot Snapshot() const {
        OrderedEncodedFrameSinkSnapshot snapshot;
        {
            std::lock_guard lock(executions_mutex_);
            snapshot.execution_count = executions_.size();
        }
        snapshot.admitted_frames = admitted_frames_.load(std::memory_order_relaxed);
        snapshot.skipped_frames = skipped_frames_.load(std::memory_order_relaxed);
        snapshot.published_frames = published_frames_.load(std::memory_order_relaxed);
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
                "ordered encoded frame identity or downstream is missing");
        }
        std::lock_guard lock(executions_mutex_);
        if (auto it = executions_.find(std::string(execution_id)); it != executions_.end()) {
            if (it->second->session_id != session_id) {
                return core::Status::Error(
                    core::ErrorCode::FailedPrecondition,
                    "ordered encoded execution belongs to another session");
            }
            return it->second;
        }
        if (executions_.size() >= options_.max_executions) {
            return core::Status::Error(core::ErrorCode::ResourceExhausted, "ordered execution limit reached");
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
            return core::Status::Error(core::ErrorCode::NotFound, "ordered encoded execution not found");
        }
        if (it->second->session_id != session_id) {
            return core::Status::Error(
                core::ErrorCode::FailedPrecondition,
                "ordered encoded execution belongs to another session");
        }
        return it->second;
    }

    core::Status Drain(ExecutionState& state) {
        std::lock_guard drain_lock(state.drain_mutex);
        for (;;) {
            if (!state.pending_publish.has_value()) {
                auto next = state.window.TryTake();
                if (!next.ok()) {
                    if (next.status().code() == core::ErrorCode::NotFound ||
                        next.status().code() == core::ErrorCode::Cancelled) {
                        return core::Status::Ok();
                    }
                    return Reject(next.status());
                }
                if (next.value().skipped()) {
                    continue;
                }
                state.pending_publish = std::move(next).value().value;
                state.pending_publish->metadata().transport_sequence = state.next_transport_sequence;
            }

            const auto timestamp_us = state.pending_publish->metadata().timestamp_us;
            if (state.last_published_timestamp_us.has_value() &&
                timestamp_us < *state.last_published_timestamp_us) {
                const auto status = core::Status::Error(
                    core::ErrorCode::FailedPrecondition,
                    "ordered encoded frame timestamp regressed");
                Notify(state.pending_publish->metadata(), status);
                return Reject(status);
            }
            const auto status = downstream_->PublishBorrowed(*state.pending_publish);
            if (status.ok()) {
                Notify(state.pending_publish->metadata(), status);
                state.last_published_timestamp_us = timestamp_us;
                state.pending_publish.reset();
                ++state.next_transport_sequence;
                published_frames_.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (status.code() == core::ErrorCode::ResourceExhausted) {
                std::this_thread::yield();
                continue;
            }
            Notify(state.pending_publish->metadata(), status);
            return Reject(status);
        }
    }

    core::Status Reject(core::Status status) {
        rejected_frames_.fetch_add(1, std::memory_order_relaxed);
        logger_.warn("[ordered-frame-sink] code={} message={}",
                     static_cast<int>(status.code()),
                     status.message());
        return status;
    }

    void Notify(const EncodedVideoFrameMetadata& metadata, const core::Status& status) const noexcept {
        if (!options_.publish_observer) return;
        try {
            options_.publish_observer(metadata, status);
        } catch (...) {
            logger_.warn("[ordered-frame-sink] publish observer failed unexpectedly");
        }
    }

    std::shared_ptr<IEncodedVideoFrameSink> downstream_;
    OrderedEncodedFrameSinkOptions options_;
    core::LoggerAdapter logger_;
    mutable std::mutex executions_mutex_;
    std::unordered_map<std::string, std::shared_ptr<ExecutionState>> executions_;
    std::atomic<std::size_t> admitted_frames_{0};
    std::atomic<std::size_t> skipped_frames_{0};
    std::atomic<std::size_t> published_frames_{0};
    std::atomic<std::size_t> rejected_frames_{0};
};

OrderedEncodedFrameSink::OrderedEncodedFrameSink(
    std::shared_ptr<IEncodedVideoFrameSink> downstream,
    OrderedEncodedFrameSinkOptions options,
    core::LoggerAdapter logger)
    : impl_(std::make_unique<Impl>(std::move(downstream), options, std::move(logger))) {}

OrderedEncodedFrameSink::~OrderedEncodedFrameSink() = default;

core::Status OrderedEncodedFrameSink::Publish(EncodedVideoFrame frame) {
    return impl_->Publish(std::move(frame));
}

core::Status OrderedEncodedFrameSink::PublishBorrowed(const EncodedVideoFrame& frame) {
    return impl_->Publish(frame);
}

core::Status OrderedEncodedFrameSink::MarkSkipped(
    std::string_view session_id,
    std::string_view execution_id,
    std::uint64_t selected_sequence,
    core::Status reason) {
    return impl_->MarkSkipped(session_id, execution_id, selected_sequence, std::move(reason));
}

core::Result<std::uint64_t> OrderedEncodedFrameSink::SealExecution(
    std::string_view session_id,
    std::string_view execution_id,
    std::uint64_t final_selected_sequence) {
    return impl_->SealExecution(session_id, execution_id, final_selected_sequence);
}

OrderedEncodedFrameSinkSnapshot OrderedEncodedFrameSink::Snapshot() const {
    return impl_->Snapshot();
}

} // namespace media
