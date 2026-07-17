#pragma once

#include "logger_adapter.h"
#include "memory_pool.h"
#include "result.h"
#include "vision_inference_interfaces.h"

#include <chrono>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace media::inference {

struct InferenceFrameTiming {
    std::int64_t published_at_unix_us = 0;
    std::int64_t received_at_unix_us = 0;
    std::int64_t admitted_at_unix_us = 0;
    std::int64_t spooled_at_unix_us = 0;
    std::int64_t replayed_at_unix_us = 0;
    std::int64_t inference_started_at_unix_us = 0;
    std::int64_t terminal_at_unix_us = 0;
};

inline std::int64_t InferenceFrameNowUnixUs() noexcept {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

inline std::uint64_t InferenceFrameDurationUs(
    std::int64_t started_at_unix_us,
    std::int64_t ended_at_unix_us) noexcept {
    if (started_at_unix_us <= 0 || ended_at_unix_us < started_at_unix_us) {
        return 0;
    }
    return static_cast<std::uint64_t>(ended_at_unix_us - started_at_unix_us);
}

enum class InferenceFrameFormat {
    Unknown,
    Jpeg,
    Png,
    Rgb,
    Bgr,
    Nv12,
    I420
};

struct InferenceFrameMetadata {
    std::string execution_id;
    std::string session_id;
    std::string trace_id;
    std::uint64_t selected_sequence = 0;
    std::uint64_t transport_sequence = 0;
    std::uint64_t frame_id = 0;
    std::int64_t timestamp_us = 0;
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    InferenceFrameFormat format = InferenceFrameFormat::Unknown;
    double saliency = 0.0;
    InferenceFrameTiming timing;
};

class OwnedInferenceFrame {
public:
    OwnedInferenceFrame() = default;
    OwnedInferenceFrame(InferenceFrameMetadata metadata,
                        core::MemoryBlock payload,
                        std::size_t payload_size) noexcept;

    OwnedInferenceFrame(const OwnedInferenceFrame&) = delete;
    OwnedInferenceFrame& operator=(const OwnedInferenceFrame&) = delete;
    OwnedInferenceFrame(OwnedInferenceFrame&&) noexcept = default;
    OwnedInferenceFrame& operator=(OwnedInferenceFrame&&) noexcept = default;

    const InferenceFrameMetadata& metadata() const noexcept {
        return metadata_;
    }

    InferenceFrameMetadata& metadata() noexcept {
        return metadata_;
    }

    std::span<const std::byte> bytes() const noexcept;
    std::size_t payload_size() const noexcept {
        return payload_size_;
    }

    bool valid() const noexcept;

private:
    InferenceFrameMetadata metadata_;
    core::MemoryBlock payload_;
    std::size_t payload_size_ = 0;
};

struct InferenceFrameSubmitOutcome {
    core::Status status = core::Status::Ok();
    std::optional<OwnedInferenceFrame> rejected_frame;

    bool accepted() const noexcept {
        return status.ok();
    }
};

core::Result<OwnedInferenceFrame> CopyInferenceFrame(
    core::RawMemoryPool& memory_pool,
    InferenceFrameMetadata metadata,
    std::span<const std::byte> payload);

struct SegmentedFrameBacklogOptions {
    std::size_t max_sessions = 64;
    std::size_t segments_per_session = 8;
    std::size_t slots_per_segment = 8;
    std::chrono::milliseconds default_wait_timeout{100};
};

struct SegmentedFrameBacklogSnapshot {
    std::size_t session_count = 0;
    std::size_t queued_frames = 0;
    std::size_t submitted_frames = 0;
    std::size_t taken_frames = 0;
    std::size_t rejected_frames = 0;
    std::size_t discarded_frames = 0;
    std::size_t ready_segment_tokens = 0;
    bool shutdown = false;
};

struct InferenceFrameOrderKey {
    std::uint64_t selected_sequence = 0;
    std::int64_t timestamp_us = 0;
    std::uint64_t frame_id = 0;

    auto operator<=>(const InferenceFrameOrderKey&) const = default;
};

struct InferenceFrameResultRecord {
    InferenceFrameMetadata frame;
    core::Status status = core::Status::Ok();
    std::optional<VisionInferenceResult> result;
};

struct InferenceFrameResultTableOptions {
    std::size_t max_sessions = 64;
    std::size_t max_results_per_session = 1024;
};

struct InferenceFrameResultTableSnapshot {
    std::size_t session_count = 0;
    std::size_t pending_results = 0;
    std::size_t published_results = 0;
    std::size_t rejected_results = 0;
};

class IInferenceFrameResultTable {
public:
    virtual ~IInferenceFrameResultTable() = default;

    virtual core::Status Publish(InferenceFrameResultRecord record) = 0;
    virtual core::Result<std::vector<InferenceFrameResultRecord>> FinalizeSession(
        std::string_view session_id) = 0;
    virtual core::Result<std::vector<InferenceFrameResultRecord>> FinalizeExecution(
        std::string_view execution_id) {
        return core::Status::Error(
            core::ErrorCode::Unimplemented,
            "inference result table does not support execution finalization");
    }
    virtual void Shutdown() = 0;
    virtual InferenceFrameResultTableSnapshot Snapshot() const = 0;
};

class SessionInferenceFrameResultTable final : public IInferenceFrameResultTable {
public:
    explicit SessionInferenceFrameResultTable(
        InferenceFrameResultTableOptions options = {},
        core::LoggerAdapter logger = {});
    ~SessionInferenceFrameResultTable() override;

    SessionInferenceFrameResultTable(const SessionInferenceFrameResultTable&) = delete;
    SessionInferenceFrameResultTable& operator=(const SessionInferenceFrameResultTable&) = delete;

    core::Status Publish(InferenceFrameResultRecord record) override;
    core::Result<std::vector<InferenceFrameResultRecord>> FinalizeSession(
        std::string_view session_id) override;
    core::Result<std::vector<InferenceFrameResultRecord>> FinalizeExecution(
        std::string_view execution_id) override;
    void Shutdown() override;
    InferenceFrameResultTableSnapshot Snapshot() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class IInferenceFrameBacklog {
public:
    virtual ~IInferenceFrameBacklog() = default;

    virtual InferenceFrameSubmitOutcome TrySubmit(OwnedInferenceFrame frame) = 0;
    virtual core::Status Submit(OwnedInferenceFrame frame) {
        return TrySubmit(std::move(frame)).status;
    }
    virtual core::Result<OwnedInferenceFrame> TryTake() = 0;
    virtual core::Result<OwnedInferenceFrame> WaitTake(
        std::chrono::milliseconds timeout) = 0;
    virtual core::Status CloseSession(std::string_view session_id) = 0;
    virtual void Shutdown() = 0;
    virtual SegmentedFrameBacklogSnapshot Snapshot() const = 0;
};

class IInferenceFrameAdmissionSink {
public:
    virtual ~IInferenceFrameAdmissionSink() = default;
    virtual core::Status AdmitFrame(OwnedInferenceFrame frame) = 0;
};

class SegmentedInferenceFrameBacklog final : public IInferenceFrameBacklog {
public:
    explicit SegmentedInferenceFrameBacklog(
        SegmentedFrameBacklogOptions options = {},
        core::LoggerAdapter logger = {});
    ~SegmentedInferenceFrameBacklog() override;

    SegmentedInferenceFrameBacklog(const SegmentedInferenceFrameBacklog&) = delete;
    SegmentedInferenceFrameBacklog& operator=(const SegmentedInferenceFrameBacklog&) = delete;

    InferenceFrameSubmitOutcome TrySubmit(OwnedInferenceFrame frame) override;
    core::Result<OwnedInferenceFrame> TryTake() override;
    core::Result<OwnedInferenceFrame> WaitTake(
        std::chrono::milliseconds timeout) override;
    core::Status CloseSession(std::string_view session_id) override;
    void Shutdown() override;
    SegmentedFrameBacklogSnapshot Snapshot() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace media::inference
