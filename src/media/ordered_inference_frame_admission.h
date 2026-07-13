#pragma once

#include "inference_frame_backlog.h"
#include "ordered_bitmap_window.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace media::inference {

struct OrderedInferenceFrameAdmissionOptions {
    std::size_t max_executions = 256;
    std::size_t window_capacity = 1024;
};

struct OrderedInferenceFrameAdmissionSnapshot {
    std::size_t execution_count = 0;
    std::size_t admitted_frames = 0;
    std::size_t forwarded_frames = 0;
    std::size_t rejected_frames = 0;
};

class IOrderedInferenceFrameAdmission : public IInferenceFrameAdmissionSink {
public:
    ~IOrderedInferenceFrameAdmission() override = default;

    virtual core::Status SealExecution(std::string_view session_id,
                                       std::string_view execution_id,
                                       std::uint64_t final_transport_sequence) = 0;
    virtual OrderedInferenceFrameAdmissionSnapshot Snapshot() const = 0;
};

class OrderedInferenceFrameAdmission final : public IOrderedInferenceFrameAdmission {
public:
    OrderedInferenceFrameAdmission(
        std::shared_ptr<IInferenceFrameAdmissionSink> downstream,
        OrderedInferenceFrameAdmissionOptions options = {},
        core::LoggerAdapter logger = {});
    ~OrderedInferenceFrameAdmission() override;

    core::Status AdmitFrame(OwnedInferenceFrame frame) override;
    core::Status SealExecution(std::string_view session_id,
                               std::string_view execution_id,
                               std::uint64_t final_transport_sequence) override;
    OrderedInferenceFrameAdmissionSnapshot Snapshot() const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace media::inference
