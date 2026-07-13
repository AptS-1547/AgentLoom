#include "inference_frame_backlog.h"

#include "blocking_queue.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace media::inference {
namespace {

constexpr std::uint64_t Bit(std::size_t index) noexcept {
    return std::uint64_t{1} << index;
}

std::uint64_t CapacityMask(std::size_t count) noexcept {
    return count >= 64 ? std::numeric_limits<std::uint64_t>::max() : Bit(count) - 1;
}

std::optional<std::size_t> SelectBit(std::uint64_t mask, std::size_t start) noexcept {
    if (mask == 0) {
        return std::nullopt;
    }
    if (start < 64) {
        const auto after_start = mask & (std::numeric_limits<std::uint64_t>::max() << start);
        if (after_start != 0) {
            return static_cast<std::size_t>(std::countr_zero(after_start));
        }
    }
    return static_cast<std::size_t>(std::countr_zero(mask));
}

core::Status ValidateOptions(const SegmentedFrameBacklogOptions& options) {
    if (options.max_sessions == 0) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "max_sessions must be positive");
    }
    if (options.segments_per_session == 0 || options.segments_per_session > 64) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "segments_per_session must be in [1, 64]");
    }
    if (options.slots_per_segment == 0 || options.slots_per_segment > 64) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "slots_per_segment must be in [1, 64]");
    }
    return core::Status::Ok();
}

} // namespace

OwnedInferenceFrame::OwnedInferenceFrame(InferenceFrameMetadata metadata,
                                         core::MemoryBlock payload,
                                         std::size_t payload_size) noexcept
    : metadata_(std::move(metadata)),
      payload_(std::move(payload)),
      payload_size_(payload_size) {}

std::span<const std::byte> OwnedInferenceFrame::bytes() const noexcept {
    return {payload_.data(), std::min(payload_size_, payload_.size())};
}

bool OwnedInferenceFrame::valid() const noexcept {
    return !metadata_.session_id.empty() &&
           !payload_.empty() &&
           payload_size_ > 0 &&
           payload_size_ <= payload_.size();
}

core::Result<OwnedInferenceFrame> CopyInferenceFrame(
    core::RawMemoryPool& memory_pool,
    InferenceFrameMetadata metadata,
    std::span<const std::byte> payload) {
    if (metadata.session_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "frame session_id is required");
    }
    if (payload.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "frame payload is empty");
    }

    auto block = memory_pool.allocate(payload.size(), alignof(std::max_align_t));
    if (!block.ok()) {
        return block.status();
    }
    std::memcpy(block.value().data(), payload.data(), payload.size());
    return OwnedInferenceFrame(
        std::move(metadata),
        std::move(block).value(),
        payload.size());
}

class SessionInferenceFrameResultTable::Impl {
public:
    struct SessionResults {
        mutable std::mutex mutex;
        std::map<InferenceFrameOrderKey, InferenceFrameResultRecord> ordered_results;
        std::unordered_set<std::uint64_t> selected_sequences;
        std::string session_id;
        std::string execution_id;
        bool closed = false;
    };

    Impl(InferenceFrameResultTableOptions options, core::LoggerAdapter logger)
        : options_(std::move(options)),
          logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("media-inference")) {}

    ~Impl() {
        Shutdown();
    }

    core::Status Publish(InferenceFrameResultRecord record) {
        if (record.frame.session_id.empty()) {
            return Reject(core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "inference result session_id is required"));
        }
        if (record.status.ok() && !record.result.has_value()) {
            return Reject(core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "successful inference result payload is missing"));
        }
        if (!record.frame.execution_id.empty() && record.frame.selected_sequence == 0) {
            return Reject(core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "execution result selected_sequence is required"));
        }
        if (shutdown_.load(std::memory_order_acquire)) {
            return Reject(core::Status::Error(
                core::ErrorCode::Cancelled,
                "inference result table is shut down"));
        }

        const auto partition_key = record.frame.execution_id.empty()
            ? "session\n" + record.frame.session_id
            : "execution\n" + record.frame.execution_id;
        auto session = GetOrCreateSession(
            partition_key,
            record.frame.session_id,
            record.frame.execution_id);
        if (!session.ok()) {
            return Reject(session.status());
        }

        std::lock_guard lock(session.value()->mutex);
        if (shutdown_.load(std::memory_order_acquire) || session.value()->closed) {
            return Reject(core::Status::Error(
                core::ErrorCode::Cancelled,
                "inference result session is finalized"));
        }
        if (session.value()->ordered_results.size() >= options_.max_results_per_session) {
            return Reject(core::Status::Error(
                core::ErrorCode::ResourceExhausted,
                "inference result session capacity reached"));
        }
        if (record.frame.selected_sequence != 0 &&
            session.value()->selected_sequences.contains(record.frame.selected_sequence)) {
            return Reject(core::Status::Error(
                core::ErrorCode::AlreadyExists,
                "inference result selected_sequence already exists"));
        }

        const InferenceFrameOrderKey key{
            .selected_sequence = record.frame.selected_sequence,
            .timestamp_us = record.frame.timestamp_us,
            .frame_id = record.frame.frame_id,
        };
        const auto [it, inserted] = session.value()->ordered_results.emplace(key, std::move(record));
        static_cast<void>(it);
        if (!inserted) {
            return Reject(core::Status::Error(
                core::ErrorCode::AlreadyExists,
                "inference result key already exists"));
        }
        if (it->second.frame.selected_sequence != 0) {
            session.value()->selected_sequences.insert(it->second.frame.selected_sequence);
        }
        published_results_.fetch_add(1, std::memory_order_relaxed);
        return core::Status::Ok();
    }

    core::Result<std::vector<InferenceFrameResultRecord>> FinalizeSession(
        std::string_view session_id) {
        if (session_id.empty()) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "session_id is required");
        }
        std::vector<std::shared_ptr<SessionResults>> matched;
        {
            std::lock_guard sessions_lock(sessions_mutex_);
            for (auto it = sessions_.begin(); it != sessions_.end();) {
                if (it->second->session_id != session_id) {
                    ++it;
                    continue;
                }
                std::lock_guard session_lock(it->second->mutex);
                it->second->closed = true;
                matched.push_back(it->second);
                it = sessions_.erase(it);
            }
        }
        if (matched.empty()) {
            return core::Status::Error(core::ErrorCode::NotFound, "inference result session not found");
        }
        auto results = Extract(std::move(matched));
        std::sort(results.begin(), results.end(), [](const auto& lhs, const auto& rhs) {
            return InferenceFrameOrderKey{
                       lhs.frame.selected_sequence,
                       lhs.frame.timestamp_us,
                       lhs.frame.frame_id} <
                   InferenceFrameOrderKey{
                       rhs.frame.selected_sequence,
                       rhs.frame.timestamp_us,
                       rhs.frame.frame_id};
        });
        return results;
    }

    core::Result<std::vector<InferenceFrameResultRecord>> FinalizeExecution(
        std::string_view execution_id) {
        if (execution_id.empty()) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "execution_id is required");
        }
        std::shared_ptr<SessionResults> session;
        {
            std::lock_guard sessions_lock(sessions_mutex_);
            const auto it = sessions_.find("execution\n" + std::string(execution_id));
            if (it == sessions_.end()) {
                return core::Status::Error(core::ErrorCode::NotFound, "inference result execution not found");
            }
            session = it->second;
            std::lock_guard session_lock(session->mutex);
            session->closed = true;
            sessions_.erase(it);
        }
        return Extract({std::move(session)});
    }

    void Shutdown() {
        if (shutdown_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        std::vector<std::shared_ptr<SessionResults>> sessions;
        {
            std::lock_guard lock(sessions_mutex_);
            sessions.reserve(sessions_.size());
            for (auto& [id, session] : sessions_) {
                static_cast<void>(id);
                sessions.push_back(std::move(session));
            }
            sessions_.clear();
        }
        for (auto& session : sessions) {
            std::lock_guard lock(session->mutex);
            session->closed = true;
            session->ordered_results.clear();
            session->selected_sequences.clear();
        }
    }

    InferenceFrameResultTableSnapshot Snapshot() const {
        InferenceFrameResultTableSnapshot snapshot;
        std::lock_guard sessions_lock(sessions_mutex_);
        snapshot.session_count = sessions_.size();
        for (const auto& [id, session] : sessions_) {
            static_cast<void>(id);
            std::lock_guard session_lock(session->mutex);
            snapshot.pending_results += session->ordered_results.size();
        }
        snapshot.published_results = published_results_.load(std::memory_order_relaxed);
        snapshot.rejected_results = rejected_results_.load(std::memory_order_relaxed);
        return snapshot;
    }

private:
    static std::vector<InferenceFrameResultRecord> Extract(
        std::vector<std::shared_ptr<SessionResults>> sessions) {
        std::vector<InferenceFrameResultRecord> results;
        for (auto& session : sessions) {
            std::lock_guard lock(session->mutex);
            results.reserve(results.size() + session->ordered_results.size());
            for (auto& [key, record] : session->ordered_results) {
                static_cast<void>(key);
                results.push_back(std::move(record));
            }
            session->ordered_results.clear();
            session->selected_sequences.clear();
        }
        return results;
    }

    core::Result<std::shared_ptr<SessionResults>> GetOrCreateSession(
        std::string_view partition_key,
        std::string_view session_id,
        std::string_view execution_id) {
        std::lock_guard lock(sessions_mutex_);
        if (shutdown_.load(std::memory_order_acquire)) {
            return core::Status::Error(
                core::ErrorCode::Cancelled,
                "inference result table is shut down");
        }
        if (auto it = sessions_.find(std::string(partition_key)); it != sessions_.end()) {
            if (it->second->session_id != session_id || it->second->execution_id != execution_id) {
                return core::Status::Error(
                    core::ErrorCode::FailedPrecondition,
                    "inference result partition identity does not match its first publisher");
            }
            return it->second;
        }
        if (sessions_.size() >= options_.max_sessions) {
            return core::Status::Error(
                core::ErrorCode::ResourceExhausted,
                "inference result session limit reached");
        }
        auto session = std::make_shared<SessionResults>();
        session->session_id = session_id;
        session->execution_id = execution_id;
        sessions_.emplace(std::string(partition_key), session);
        return session;
    }

    core::Status Reject(core::Status status) {
        rejected_results_.fetch_add(1, std::memory_order_relaxed);
        logger_.warn("[InferenceFrameResultRejected] code={} error={}",
                     static_cast<int>(status.code()),
                     status.message());
        return status;
    }

    InferenceFrameResultTableOptions options_;
    core::LoggerAdapter logger_;
    mutable std::mutex sessions_mutex_;
    std::unordered_map<std::string, std::shared_ptr<SessionResults>> sessions_;
    std::atomic<bool> shutdown_{false};
    std::atomic<std::size_t> published_results_{0};
    std::atomic<std::size_t> rejected_results_{0};
};

class SegmentedInferenceFrameBacklog::Impl {
public:
    struct Segment;
    struct SessionState;

    struct ReadySegmentToken {
        std::shared_ptr<SessionState> session;
        std::size_t segment_index = 0;
    };

    class SegmentLease {
    public:
        SegmentLease() = default;
        explicit SegmentLease(Segment& segment) noexcept;
        ~SegmentLease();

        SegmentLease(const SegmentLease&) = delete;
        SegmentLease& operator=(const SegmentLease&) = delete;
        SegmentLease(SegmentLease&& other) noexcept;
        SegmentLease& operator=(SegmentLease&& other) noexcept;

        explicit operator bool() const noexcept {
            return segment_ != nullptr;
        }

    private:
        void Release() noexcept;
        Segment* segment_ = nullptr;
    };

    struct alignas(64) Segment {
        explicit Segment(std::size_t slot_count)
            : slots(slot_count),
              free_slots(CapacityMask(slot_count)) {}

        std::atomic_flag lease = ATOMIC_FLAG_INIT;
        std::atomic<bool> token_queued{false};
        std::atomic<std::uint64_t> ready_slots{0};
        std::atomic<std::uint64_t> free_slots{0};
        std::vector<std::optional<OwnedInferenceFrame>> slots;
    };

    struct SessionState {
        SessionState(std::string id,
                     std::size_t segment_count,
                     std::size_t slots_per_segment)
            : session_id(std::move(id)) {
            segments.reserve(segment_count);
            for (std::size_t index = 0; index < segment_count; ++index) {
                segments.push_back(std::make_unique<Segment>(slots_per_segment));
            }
            writable_segments.store(CapacityMask(segment_count), std::memory_order_relaxed);
        }

        std::string session_id;
        std::vector<std::unique_ptr<Segment>> segments;
        std::atomic<std::uint64_t> ready_segments{0};
        std::atomic<std::uint64_t> writable_segments{0};
        std::atomic<std::size_t> write_cursor{0};
        std::atomic<std::size_t> take_cursor{0};
        std::atomic<std::size_t> queued_frames{0};
        std::atomic<bool> closed{false};
    };

    Impl(SegmentedFrameBacklogOptions options, core::LoggerAdapter logger)
        : options_(std::move(options)),
          logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("media-inference")),
          ready_segments_(0),
          options_status_(ValidateOptions(options_)) {}

    ~Impl() {
        Shutdown();
    }

    InferenceFrameSubmitOutcome TrySubmit(OwnedInferenceFrame frame) {
        auto reject = [&](core::Status status, bool expected_backpressure = false) -> InferenceFrameSubmitOutcome {
            return {
                .status = LogFailure(std::move(status), expected_backpressure),
                .rejected_frame = std::optional<OwnedInferenceFrame>(std::move(frame)),
            };
        };
        if (!options_status_.ok()) {
            return reject(options_status_);
        }
        if (!frame.valid()) {
            return reject(core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "owned inference frame is invalid"));
        }
        if (shutdown_.load(std::memory_order_acquire)) {
            return reject(core::Status::Error(
                core::ErrorCode::Cancelled,
                "inference frame backlog is shut down"));
        }

        auto session = GetOrCreateSession(frame.metadata().session_id);
        if (!session.ok()) {
            return reject(session.status());
        }
        if (session.value()->closed.load(std::memory_order_acquire)) {
            return reject(core::Status::Error(
                core::ErrorCode::Cancelled,
                "inference frame session is closed"));
        }

        auto& state = *session.value();
        auto candidate_mask = state.writable_segments.load(std::memory_order_acquire);
        const auto start = state.write_cursor.fetch_add(1, std::memory_order_relaxed) % options_.segments_per_session;

        while (candidate_mask != 0) {
            const auto selected = SelectBit(candidate_mask, start);
            if (!selected) {
                break;
            }
            const auto segment_index = *selected;
            candidate_mask &= ~Bit(segment_index);

            auto& segment = *state.segments[segment_index];
            bool submitted = false;
            {
                SegmentLease lease(segment);
                if (!lease) {
                    continue;
                }
                if (state.closed.load(std::memory_order_acquire)) {
                    return reject(core::Status::Error(
                        core::ErrorCode::Cancelled,
                        "inference frame session is closed"));
                }

                const auto free_mask = segment.free_slots.load(std::memory_order_relaxed);
                if (free_mask == 0) {
                    state.writable_segments.fetch_and(~Bit(segment_index), std::memory_order_release);
                    continue;
                }

                const auto slot_index = static_cast<std::size_t>(std::countr_zero(free_mask));
                segment.slots[slot_index].emplace(std::move(frame));
                const auto new_free_mask = free_mask & ~Bit(slot_index);
                const auto ready_mask = segment.ready_slots.load(std::memory_order_relaxed) | Bit(slot_index);
                segment.free_slots.store(new_free_mask, std::memory_order_release);
                segment.ready_slots.store(ready_mask, std::memory_order_release);
                state.ready_segments.fetch_or(Bit(segment_index), std::memory_order_release);
                if (new_free_mask == 0) {
                    state.writable_segments.fetch_and(~Bit(segment_index), std::memory_order_release);
                }
                state.queued_frames.fetch_add(1, std::memory_order_relaxed);
                submitted_frames_.fetch_add(1, std::memory_order_relaxed);
                submitted = true;
            }
            if (submitted) {
                ScheduleSegment(session.value(), segment_index);
                return {};
            }
        }

        rejected_frames_.fetch_add(1, std::memory_order_relaxed);
        return reject(
            core::Status::Error(
                core::ErrorCode::ResourceExhausted,
                "inference frame session backlog is full"),
            true);
    }

    core::Result<OwnedInferenceFrame> TryTake() {
        for (;;) {
            auto token = ready_segments_.TryPop();
            if (!token.ok()) {
                return token.status();
            }
            auto frame = TakeFromToken(std::move(token).value());
            if (frame.ok()) {
                return frame;
            }
            if (frame.status().code() != core::ErrorCode::NotFound &&
                frame.status().code() != core::ErrorCode::Cancelled) {
                return frame.status();
            }
        }
    }

    core::Result<OwnedInferenceFrame> WaitTake(std::chrono::milliseconds timeout) {
        if (timeout.count() < 0) {
            timeout = options_.default_wait_timeout;
        }
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        for (;;) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) {
                return core::Status::Error(core::ErrorCode::Timeout, "inference frame wait timed out");
            }
            auto token = ready_segments_.WaitPopFor(deadline - now);
            if (!token.ok()) {
                return token.status();
            }
            auto frame = TakeFromToken(std::move(token).value());
            if (frame.ok()) {
                return frame;
            }
            if (frame.status().code() != core::ErrorCode::NotFound &&
                frame.status().code() != core::ErrorCode::Cancelled) {
                return frame.status();
            }
        }
    }

    core::Status CloseSession(std::string_view session_id) {
        std::shared_ptr<SessionState> session;
        {
            std::lock_guard lock(sessions_mutex_);
            const auto it = sessions_.find(std::string(session_id));
            if (it == sessions_.end()) {
                return core::Status::Error(core::ErrorCode::NotFound, "inference frame session not found");
            }
            session = std::move(it->second);
            sessions_.erase(it);
        }

        session->closed.store(true, std::memory_order_release);
        const auto discarded = DiscardSession(*session);
        discarded_frames_.fetch_add(discarded, std::memory_order_relaxed);
        logger_.debug(
            "[InferenceFrameSessionClosed] session_id={} discarded_frames={}",
            session_id,
            discarded);
        return core::Status::Ok();
    }

    void Shutdown() {
        if (shutdown_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        ready_segments_.Close(true);

        std::vector<std::shared_ptr<SessionState>> sessions;
        {
            std::lock_guard lock(sessions_mutex_);
            sessions.reserve(sessions_.size());
            for (auto& [id, session] : sessions_) {
                static_cast<void>(id);
                session->closed.store(true, std::memory_order_release);
                sessions.push_back(std::move(session));
            }
            sessions_.clear();
        }
        std::size_t discarded = 0;
        for (auto& session : sessions) {
            discarded += DiscardSession(*session);
        }
        discarded_frames_.fetch_add(discarded, std::memory_order_relaxed);
    }

    SegmentedFrameBacklogSnapshot Snapshot() const {
        SegmentedFrameBacklogSnapshot snapshot;
        {
            std::lock_guard lock(sessions_mutex_);
            snapshot.session_count = sessions_.size();
            for (const auto& [id, session] : sessions_) {
                static_cast<void>(id);
                snapshot.queued_frames += session->queued_frames.load(std::memory_order_relaxed);
            }
        }
        snapshot.submitted_frames = submitted_frames_.load(std::memory_order_relaxed);
        snapshot.taken_frames = taken_frames_.load(std::memory_order_relaxed);
        snapshot.rejected_frames = rejected_frames_.load(std::memory_order_relaxed);
        snapshot.discarded_frames = discarded_frames_.load(std::memory_order_relaxed);
        snapshot.ready_segment_tokens = ready_segments_.size();
        snapshot.shutdown = shutdown_.load(std::memory_order_acquire);
        return snapshot;
    }

private:
    core::Result<std::shared_ptr<SessionState>> GetOrCreateSession(std::string_view session_id) {
        std::lock_guard lock(sessions_mutex_);
        if (shutdown_.load(std::memory_order_acquire)) {
            return core::Status::Error(
                core::ErrorCode::Cancelled,
                "inference frame backlog is shut down");
        }
        if (auto it = sessions_.find(std::string(session_id)); it != sessions_.end()) {
            return it->second;
        }
        if (sessions_.size() >= options_.max_sessions) {
            return core::Status::Error(
                core::ErrorCode::ResourceExhausted,
                "inference frame session limit reached");
        }
        auto session = std::make_shared<SessionState>(
            std::string(session_id),
            options_.segments_per_session,
            options_.slots_per_segment);
        sessions_.emplace(session->session_id, session);
        return session;
    }

    void ScheduleSegment(const std::shared_ptr<SessionState>& session, std::size_t segment_index) {
        auto& segment = *session->segments[segment_index];
        if (segment.ready_slots.load(std::memory_order_acquire) == 0 ||
            segment.token_queued.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        auto status = ready_segments_.TryPush(ReadySegmentToken{session, segment_index});
        if (!status.ok()) {
            segment.token_queued.store(false, std::memory_order_release);
        }
    }

    core::Result<OwnedInferenceFrame> TakeFromToken(ReadySegmentToken token) {
        if (!token.session || token.segment_index >= token.session->segments.size()) {
            return core::Status::Error(core::ErrorCode::NotFound, "ready segment token is stale");
        }

        auto& session = *token.session;
        auto& segment = *session.segments[token.segment_index];
        segment.token_queued.store(false, std::memory_order_release);
        if (session.closed.load(std::memory_order_acquire)) {
            return core::Status::Error(core::ErrorCode::Cancelled, "inference frame session is closed");
        }

        std::optional<OwnedInferenceFrame> frame;
        bool reschedule = false;
        {
            SegmentLease lease(segment);
            if (!lease) {
                ScheduleSegment(token.session, token.segment_index);
                return core::Status::Error(core::ErrorCode::NotFound, "ready segment is busy");
            }

            const auto ready_mask = segment.ready_slots.load(std::memory_order_relaxed);
            if (ready_mask == 0) {
                session.ready_segments.fetch_and(~Bit(token.segment_index), std::memory_order_release);
                return core::Status::Error(core::ErrorCode::NotFound, "ready segment is empty");
            }

            const auto start = session.take_cursor.fetch_add(1, std::memory_order_relaxed) % options_.slots_per_segment;
            const auto selected = SelectBit(ready_mask, start);
            if (!selected || !segment.slots[*selected].has_value()) {
                return core::Status::Error(core::ErrorCode::InternalError, "ready slot metadata is inconsistent");
            }

            frame.emplace(std::move(*segment.slots[*selected]));
            segment.slots[*selected].reset();
            const auto new_ready_mask = ready_mask & ~Bit(*selected);
            segment.ready_slots.store(new_ready_mask, std::memory_order_release);
            segment.free_slots.fetch_or(Bit(*selected), std::memory_order_release);
            session.writable_segments.fetch_or(Bit(token.segment_index), std::memory_order_release);
            if (new_ready_mask == 0) {
                session.ready_segments.fetch_and(~Bit(token.segment_index), std::memory_order_release);
            }
            session.queued_frames.fetch_sub(1, std::memory_order_relaxed);
            taken_frames_.fetch_add(1, std::memory_order_relaxed);
            reschedule = new_ready_mask != 0;
        }
        if (reschedule) {
            ScheduleSegment(token.session, token.segment_index);
        }
        return std::move(*frame);
    }

    std::size_t DiscardSession(SessionState& session) {
        std::size_t discarded = 0;
        for (std::size_t segment_index = 0; segment_index < session.segments.size(); ++segment_index) {
            auto& segment = *session.segments[segment_index];
            SegmentLease lease(segment);
            while (!lease) {
                std::this_thread::yield();
                lease = SegmentLease(segment);
            }
            for (auto& slot : segment.slots) {
                if (slot.has_value()) {
                    slot.reset();
                    ++discarded;
                }
            }
            segment.ready_slots.store(0, std::memory_order_release);
            segment.free_slots.store(CapacityMask(segment.slots.size()), std::memory_order_release);
            segment.token_queued.store(false, std::memory_order_release);
        }
        session.ready_segments.store(0, std::memory_order_release);
        session.writable_segments.store(CapacityMask(session.segments.size()), std::memory_order_release);
        session.queued_frames.store(0, std::memory_order_release);
        return discarded;
    }

    core::Status LogFailure(core::Status status, bool expected_backpressure) const {
        if (expected_backpressure) {
            logger_.debug("[InferenceFrameBacklogBackpressure] code={} error={}",
                          static_cast<int>(status.code()),
                          status.message());
        } else {
            logger_.warn("[InferenceFrameBacklogRejected] code={} error={}",
                         static_cast<int>(status.code()),
                         status.message());
        }
        return status;
    }

    SegmentedFrameBacklogOptions options_;
    core::LoggerAdapter logger_;
    core::BlockingQueue<ReadySegmentToken> ready_segments_;
    core::Status options_status_;
    mutable std::mutex sessions_mutex_;
    std::unordered_map<std::string, std::shared_ptr<SessionState>> sessions_;
    std::atomic<bool> shutdown_{false};
    std::atomic<std::size_t> submitted_frames_{0};
    std::atomic<std::size_t> taken_frames_{0};
    std::atomic<std::size_t> rejected_frames_{0};
    std::atomic<std::size_t> discarded_frames_{0};
};

SegmentedInferenceFrameBacklog::Impl::SegmentLease::SegmentLease(Segment& segment) noexcept {
    if (!segment.lease.test_and_set(std::memory_order_acquire)) {
        segment_ = &segment;
    }
}

SegmentedInferenceFrameBacklog::Impl::SegmentLease::~SegmentLease() {
    Release();
}

SegmentedInferenceFrameBacklog::Impl::SegmentLease::SegmentLease(SegmentLease&& other) noexcept
    : segment_(other.segment_) {
    other.segment_ = nullptr;
}

SegmentedInferenceFrameBacklog::Impl::SegmentLease&
SegmentedInferenceFrameBacklog::Impl::SegmentLease::operator=(SegmentLease&& other) noexcept {
    if (this != &other) {
        Release();
        segment_ = other.segment_;
        other.segment_ = nullptr;
    }
    return *this;
}

void SegmentedInferenceFrameBacklog::Impl::SegmentLease::Release() noexcept {
    if (segment_) {
        segment_->lease.clear(std::memory_order_release);
        segment_ = nullptr;
    }
}

SessionInferenceFrameResultTable::SessionInferenceFrameResultTable(
    InferenceFrameResultTableOptions options,
    core::LoggerAdapter logger)
    : impl_(std::make_unique<Impl>(std::move(options), std::move(logger))) {}

SessionInferenceFrameResultTable::~SessionInferenceFrameResultTable() = default;

core::Status SessionInferenceFrameResultTable::Publish(InferenceFrameResultRecord record) {
    return impl_->Publish(std::move(record));
}

core::Result<std::vector<InferenceFrameResultRecord>>
SessionInferenceFrameResultTable::FinalizeSession(std::string_view session_id) {
    return impl_->FinalizeSession(session_id);
}

core::Result<std::vector<InferenceFrameResultRecord>>
SessionInferenceFrameResultTable::FinalizeExecution(std::string_view execution_id) {
    return impl_->FinalizeExecution(execution_id);
}

void SessionInferenceFrameResultTable::Shutdown() {
    impl_->Shutdown();
}

InferenceFrameResultTableSnapshot SessionInferenceFrameResultTable::Snapshot() const {
    return impl_->Snapshot();
}

SegmentedInferenceFrameBacklog::SegmentedInferenceFrameBacklog(
    SegmentedFrameBacklogOptions options,
    core::LoggerAdapter logger)
    : impl_(std::make_unique<Impl>(std::move(options), std::move(logger))) {}

SegmentedInferenceFrameBacklog::~SegmentedInferenceFrameBacklog() = default;

InferenceFrameSubmitOutcome SegmentedInferenceFrameBacklog::TrySubmit(OwnedInferenceFrame frame) {
    return impl_->TrySubmit(std::move(frame));
}

core::Result<OwnedInferenceFrame> SegmentedInferenceFrameBacklog::TryTake() {
    return impl_->TryTake();
}

core::Result<OwnedInferenceFrame> SegmentedInferenceFrameBacklog::WaitTake(
    std::chrono::milliseconds timeout) {
    return impl_->WaitTake(timeout);
}

core::Status SegmentedInferenceFrameBacklog::CloseSession(std::string_view session_id) {
    return impl_->CloseSession(session_id);
}

void SegmentedInferenceFrameBacklog::Shutdown() {
    impl_->Shutdown();
}

SegmentedFrameBacklogSnapshot SegmentedInferenceFrameBacklog::Snapshot() const {
    return impl_->Snapshot();
}

} // namespace media::inference
