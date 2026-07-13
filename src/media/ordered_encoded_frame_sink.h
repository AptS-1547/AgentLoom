#pragma once

#include "frame_encoding.h"
#include "logger_adapter.h"
#include "ordered_bitmap_window.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>

namespace media {

struct OrderedEncodedFrameSinkOptions {
    std::size_t max_executions = 256;
    std::size_t window_capacity = 1024;
    std::function<void(const EncodedVideoFrameMetadata&, const core::Status&)> publish_observer;
};

struct OrderedEncodedFrameSinkSnapshot {
    std::size_t execution_count = 0;
    std::size_t admitted_frames = 0;
    std::size_t skipped_frames = 0;
    std::size_t published_frames = 0;
    std::size_t rejected_frames = 0;
};

class IOrderedEncodedFrameSink : public IEncodedVideoFrameSink {
public:
    ~IOrderedEncodedFrameSink() override = default;

    virtual core::Status MarkSkipped(std::string_view session_id,
                                     std::string_view execution_id,
                                     std::uint64_t selected_sequence,
                                     core::Status reason) = 0;
    virtual core::Result<std::uint64_t> SealExecution(
        std::string_view session_id,
        std::string_view execution_id,
        std::uint64_t final_selected_sequence) = 0;
    virtual OrderedEncodedFrameSinkSnapshot Snapshot() const = 0;
};

class OrderedEncodedFrameSink final : public IOrderedEncodedFrameSink {
public:
    OrderedEncodedFrameSink(
        std::shared_ptr<IEncodedVideoFrameSink> downstream,
        OrderedEncodedFrameSinkOptions options = {},
        core::LoggerAdapter logger = {});
    ~OrderedEncodedFrameSink() override;

    core::Status Publish(EncodedVideoFrame frame) override;
    core::Status PublishBorrowed(const EncodedVideoFrame& frame) override;
    core::Status MarkSkipped(std::string_view session_id,
                             std::string_view execution_id,
                             std::uint64_t selected_sequence,
                             core::Status reason) override;
    core::Result<std::uint64_t> SealExecution(
        std::string_view session_id,
        std::string_view execution_id,
        std::uint64_t final_selected_sequence) override;
    OrderedEncodedFrameSinkSnapshot Snapshot() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace media
