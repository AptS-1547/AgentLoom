#include "inference_frame_coordinator.h"

#include <algorithm>
#include <atomic>
#include <exception>
#include <string_view>
#include <utility>

namespace media::inference {
namespace {

std::string_view MimeType(InferenceFrameFormat format) noexcept {
    switch (format) {
    case InferenceFrameFormat::Jpeg:
        return "image/jpeg";
    case InferenceFrameFormat::Png:
        return "image/png";
    case InferenceFrameFormat::Unknown:
    case InferenceFrameFormat::Rgb:
    case InferenceFrameFormat::Bgr:
    case InferenceFrameFormat::Nv12:
    case InferenceFrameFormat::I420:
        return {};
    }
    return {};
}

void UpdateMaximum(std::atomic<std::uint64_t>& target, std::uint64_t value) noexcept {
    auto current = target.load(std::memory_order_relaxed);
    while (current < value &&
           !target.compare_exchange_weak(current, value, std::memory_order_relaxed)) {
    }
}

} // namespace

class InferenceFrameCoordinator::Impl {
public:
    Impl(
        IInferenceFrameBacklog& backlog,
        IVlmVisionClient& vlm_client,
        IInferenceFrameResultTable& result_table,
        InferenceFrameCoordinatorOptions options,
        core::LoggerAdapter logger)
        : backlog_(backlog),
          vlm_client_(vlm_client),
          result_table_(result_table),
          options_(std::move(options)),
          logger_(logger.valid() ? std::move(logger) : core::LoggerAdapter::ForModule("media-inference")),
          workers_({
              .worker_count = options_.worker_count,
              .queue_capacity = options_.worker_count,
              .name = "inference-frame-workers",
          }) {}

    ~Impl() {
        Shutdown();
    }

    core::Status Start() {
        if (options_.worker_count == 0) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "inference worker_count must be positive");
        }
        if (running_.exchange(true, std::memory_order_acq_rel)) {
            return core::Status::Ok();
        }

        auto status = workers_.Start();
        if (!status.ok()) {
            running_.store(false, std::memory_order_release);
            return status;
        }
        for (std::size_t index = 0; index < options_.worker_count; ++index) {
            status = workers_.Submit(
                [this] { return WorkerLoop(); },
                {},
                "inference-frame-worker");
            if (!status.ok()) {
                running_.store(false, std::memory_order_release);
                if (options_.shutdown_backlog) {
                    backlog_.Shutdown();
                }
                workers_.Shutdown(false);
                return status;
            }
        }
        return core::Status::Ok();
    }

    void Shutdown() {
        if (!running_.exchange(false, std::memory_order_acq_rel)) {
            return;
        }
        if (options_.shutdown_backlog) {
            backlog_.Shutdown();
        }
        workers_.Shutdown(true);
    }

    InferenceFrameCoordinatorSnapshot Snapshot() const {
        return {
            .processed_frames = processed_frames_.load(std::memory_order_relaxed),
            .successful_frames = successful_frames_.load(std::memory_order_relaxed),
            .failed_frames = failed_frames_.load(std::memory_order_relaxed),
            .result_publish_failures = result_publish_failures_.load(std::memory_order_relaxed),
            .total_latency_us = total_latency_us_.load(std::memory_order_relaxed),
            .max_latency_us = max_latency_us_.load(std::memory_order_relaxed),
            .running = running_.load(std::memory_order_acquire),
        };
    }

private:
    core::Status WorkerLoop() {
        while (running_.load(std::memory_order_acquire)) {
            auto frame = backlog_.WaitTake(options_.wait_timeout);
            if (!frame.ok()) {
                if (frame.status().code() == core::ErrorCode::Timeout) {
                    continue;
                }
                if (frame.status().code() == core::ErrorCode::Cancelled) {
                    break;
                }
                logger_.warn(
                    "[frame-coordinator] backlog take failed code={} message={}",
                    static_cast<int>(frame.status().code()),
                    frame.status().message());
                continue;
            }

            auto owned_frame = std::move(frame).value();
            const auto started = std::chrono::steady_clock::now();
            InferenceFrameResultRecord record;
            record.frame = owned_frame.metadata();
            const auto mime_type = MimeType(record.frame.format);
            if (mime_type.empty()) {
                record.status = core::Status::Error(
                    core::ErrorCode::InvalidArgument,
                    "VLM coordinator requires JPEG or PNG input");
            } else {
                VisionInferenceRequest request;
                request.session_id = record.frame.session_id;
                request.frame_id = record.frame.frame_id;
                request.encoded_image = owned_frame.bytes();
                request.mime_type = std::string(mime_type);
                try {
                    auto result = vlm_client_.Analyze(request);
                    if (result.ok()) {
                        record.result = std::move(result).value();
                    } else {
                        record.status = result.status();
                    }
                } catch (const std::exception&) {
                    record.status = core::Status::Error(
                        core::ErrorCode::InternalError,
                        "VLM client failed unexpectedly");
                } catch (...) {
                    record.status = core::Status::Error(
                        core::ErrorCode::Unknown,
                        "VLM client failed unexpectedly");
                }
            }

            const auto latency_us = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - started).count());
            total_latency_us_.fetch_add(latency_us, std::memory_order_relaxed);
            UpdateMaximum(max_latency_us_, latency_us);
            processed_frames_.fetch_add(1, std::memory_order_relaxed);
            if (record.status.ok()) {
                successful_frames_.fetch_add(1, std::memory_order_relaxed);
            } else {
                failed_frames_.fetch_add(1, std::memory_order_relaxed);
            }

            InferenceFrameTerminalEvent terminal_event{
                .frame = record.frame,
                .inference_status = record.status,
            };
            const auto publish_status = result_table_.Publish(std::move(record));
            terminal_event.publish_status = publish_status;
            if (!publish_status.ok()) {
                result_publish_failures_.fetch_add(1, std::memory_order_relaxed);
                logger_.warn(
                    "[frame-coordinator] result publish failed code={} message={}",
                    static_cast<int>(publish_status.code()),
                    publish_status.message());
            }
            if (options_.terminal_observer) {
                try {
                    options_.terminal_observer(std::move(terminal_event));
                } catch (const std::exception&) {
                    logger_.warn("[frame-coordinator] terminal observer failed unexpectedly");
                } catch (...) {
                    logger_.warn("[frame-coordinator] terminal observer failed unexpectedly");
                }
            }
        }
        return core::Status::Ok();
    }

    IInferenceFrameBacklog& backlog_;
    IVlmVisionClient& vlm_client_;
    IInferenceFrameResultTable& result_table_;
    InferenceFrameCoordinatorOptions options_;
    core::LoggerAdapter logger_;
    core::ThreadPool workers_;
    std::atomic<bool> running_{false};
    std::atomic<std::size_t> processed_frames_{0};
    std::atomic<std::size_t> successful_frames_{0};
    std::atomic<std::size_t> failed_frames_{0};
    std::atomic<std::size_t> result_publish_failures_{0};
    std::atomic<std::uint64_t> total_latency_us_{0};
    std::atomic<std::uint64_t> max_latency_us_{0};
};

InferenceFrameCoordinator::InferenceFrameCoordinator(
    IInferenceFrameBacklog& backlog,
    IVlmVisionClient& vlm_client,
    IInferenceFrameResultTable& result_table,
    InferenceFrameCoordinatorOptions options,
    core::LoggerAdapter logger)
    : impl_(std::make_unique<Impl>(
          backlog,
          vlm_client,
          result_table,
          std::move(options),
          std::move(logger))) {}

InferenceFrameCoordinator::~InferenceFrameCoordinator() = default;

core::Status InferenceFrameCoordinator::Start() {
    return impl_->Start();
}

void InferenceFrameCoordinator::Shutdown() {
    impl_->Shutdown();
}

InferenceFrameCoordinatorSnapshot InferenceFrameCoordinator::Snapshot() const {
    return impl_->Snapshot();
}

} // namespace media::inference
