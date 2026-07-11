#include "frame_encoding.h"
#include "inference_frame_backlog.h"
#include "inference_frame_coordinator.h"
#include "inference_frame_gateway_producer.h"
#include "inference_frame_ipc_receiver.h"
#include "inference_frame_shared_memory.h"
#include "memory_pool.h"
#include "opencv_frame_sampler.h"
#include "thread_pool.h"

#include <gst/gst.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

constexpr std::int64_t kUnsetTime = 0;

std::int64_t NowNs() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
}

struct FrameRecord {
    std::atomic<std::int64_t> generated_at{kUnsetTime};
    std::atomic<std::int64_t> compute_started_at{kUnsetTime};
    std::atomic<std::int64_t> selected_at{kUnsetTime};
    std::atomic<std::int64_t> encoded_at{kUnsetTime};
    std::atomic<std::int64_t> ipc_published_at{kUnsetTime};
    std::atomic<std::int64_t> receiver_admitted_at{kUnsetTime};
    std::atomic<std::int64_t> vlm_started_at{kUnsetTime};
    std::atomic<std::int64_t> vlm_finished_at{kUnsetTime};
    std::atomic<double> saliency{0.0};
    std::atomic<bool> compute_rejected{false};
    std::atomic<bool> sampler_evaluated{false};
    std::atomic<bool> sampler_selected{false};
    std::atomic<bool> encode_failed{false};
    std::atomic<bool> io_rejected{false};
    std::atomic<bool> ipc_publish_rejected{false};
    std::atomic<bool> receiver_rejected{false};
    std::atomic<bool> vlm_succeeded{false};
    std::atomic<bool> vlm_failed{false};
    std::size_t session_index = 0;
    std::size_t event_index = std::numeric_limits<std::size_t>::max();
};

struct Scenario {
    std::string name;
    std::size_t sessions = 4;
    std::size_t fps = 15;
    std::chrono::milliseconds duration{3000};
    std::size_t compute_workers = 4;
    std::size_t compute_queue = 128;
    std::size_t io_workers = 2;
    std::size_t io_queue = 64;
    std::size_t ipc_slots = 32;
    std::size_t backlog_slots_per_session = 16;
    std::size_t vlm_workers = 4;
    std::chrono::milliseconds vlm_latency{20};
    bool synchronized_burst = false;
};

struct Percentiles {
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
};

std::size_t Count(const std::unique_ptr<FrameRecord[]>& records,
                  std::size_t count,
                  const auto& predicate) {
    std::size_t result = 0;
    for (std::size_t index = 0; index < count; ++index) {
        if (predicate(records[index])) {
            ++result;
        }
    }
    return result;
}

Percentiles LatencyPercentiles(
    const std::unique_ptr<FrameRecord[]>& records,
    std::size_t count,
    const auto& start_time,
    const auto& end_time) {
    std::vector<double> values;
    values.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto start = start_time(records[index]);
        const auto end = end_time(records[index]);
        if (start != kUnsetTime && end >= start) {
            values.push_back(static_cast<double>(end - start) / 1'000'000.0);
        }
    }
    if (values.empty()) {
        return {};
    }
    std::sort(values.begin(), values.end());
    const auto at = [&](double quantile) {
        const auto index = static_cast<std::size_t>(
            std::ceil(quantile * static_cast<double>(values.size())) - 1.0);
        return values[std::min(index, values.size() - 1)];
    };
    return {.p50_ms = at(0.50), .p95_ms = at(0.95), .p99_ms = at(0.99)};
}

double Rate(std::size_t numerator, std::size_t denominator) noexcept {
    return denominator == 0 ? 0.0 : static_cast<double>(numerator) / static_cast<double>(denominator);
}

std::optional<std::size_t> RecordIndex(std::uint64_t frame_id, std::size_t total_frames) noexcept {
    if (frame_id == 0 || frame_id > total_frames) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(frame_id - 1);
}

class FakeVlm final : public media::IVlmVisionClient {
public:
    FakeVlm(
        std::unique_ptr<FrameRecord[]>& records,
        std::size_t total_frames,
        std::chrono::milliseconds latency)
        : records_(records),
          total_frames_(total_frames),
          latency_(latency) {}

    core::Result<media::VisionInferenceResult> Analyze(
        const media::VisionInferenceRequest& request) override {
        const auto index = RecordIndex(request.frame_id, total_frames_);
        if (!index) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "FakeVLM received an unknown frame id");
        }
        auto& record = records_[*index];
        record.vlm_started_at.store(NowNs(), std::memory_order_release);
        std::this_thread::sleep_for(latency_);
        if (request.encoded_image.empty()) {
            record.vlm_failed.store(true, std::memory_order_release);
            record.vlm_finished_at.store(NowNs(), std::memory_order_release);
            return core::Status::Error(core::ErrorCode::InvalidArgument, "FakeVLM received an empty image");
        }

        media::VisionInferenceResult result;
        result.scene_hint = "synthetic-scene";
        result.confidence = 0.9;
        record.vlm_succeeded.store(true, std::memory_order_release);
        record.vlm_finished_at.store(NowNs(), std::memory_order_release);
        return result;
    }

private:
    std::unique_ptr<FrameRecord[]>& records_;
    std::size_t total_frames_ = 0;
    std::chrono::milliseconds latency_;
};

std::shared_ptr<std::vector<std::byte>> MakeSyntheticRgb(
    std::uint32_t width,
    std::uint32_t height,
    std::size_t session_index,
    std::size_t frame_in_session,
    bool event_frame) {
    auto bytes = std::make_shared<std::vector<std::byte>>(
        static_cast<std::size_t>(width) * height * 3,
        std::byte{0x08});
    const auto block_size = event_frame ? 72u : 24u;
    const auto max_x = width > block_size ? width - block_size : 1u;
    const auto max_y = height > block_size ? height - block_size : 1u;
    const auto origin_x = static_cast<std::uint32_t>(
        (frame_in_session * 17 + session_index * 29) % max_x);
    const auto origin_y = static_cast<std::uint32_t>(
        (frame_in_session * 11 + session_index * 13) % max_y);
    for (std::uint32_t y = origin_y; y < std::min(height, origin_y + block_size); ++y) {
        for (std::uint32_t x = origin_x; x < std::min(width, origin_x + block_size); ++x) {
            const auto offset = (static_cast<std::size_t>(y) * width + x) * 3;
            (*bytes)[offset] = event_frame ? std::byte{0xF0} : std::byte{0x30};
            (*bytes)[offset + 1] = static_cast<std::byte>((frame_in_session * 19) & 0xFF);
            (*bytes)[offset + 2] = event_frame ? std::byte{0x20} : std::byte{0x80};
        }
    }
    return bytes;
}

std::size_t EventIndex(
    const Scenario& scenario,
    std::size_t session_index,
    std::size_t frame_in_session,
    std::size_t frames_per_session) noexcept {
    const auto event_width = std::max<std::size_t>(2, scenario.fps / 2);
    const auto first_start = frames_per_session / 3;
    const auto second_start = (frames_per_session * 2) / 3;
    const auto session_offset = scenario.synchronized_burst ? 0 : session_index * std::max<std::size_t>(1, scenario.fps / 8);
    const auto shifted = frame_in_session + session_offset;
    if (shifted >= first_start && shifted < first_start + event_width) {
        return session_index * 2;
    }
    if (shifted >= second_start && shifted < second_start + event_width) {
        return session_index * 2 + 1;
    }
    return std::numeric_limits<std::size_t>::max();
}

void PrintLatency(std::string_view name, const Percentiles& value) {
    std::cout << "  " << name
              << " p50=" << value.p50_ms << "ms"
              << " p95=" << value.p95_ms << "ms"
              << " p99=" << value.p99_ms << "ms\n";
}

core::Status RunScenario(const Scenario& scenario) {
    constexpr std::uint32_t kWidth = 320;
    constexpr std::uint32_t kHeight = 180;
    const auto frames_per_session = static_cast<std::size_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(scenario.duration).count() * scenario.fps / 1000);
    const auto total_frames = scenario.sessions * frames_per_session;
    auto records = std::make_unique<FrameRecord[]>(total_frames);

    core::BucketMemoryPool gateway_pool;
    core::BucketMemoryPool inference_pool;
    auto encoder = media::GStreamerVideoFrameEncoder::Create(
        gateway_pool,
        {
            .format = media::EncodedVideoFrameFormat::Jpeg,
            .preference = media::VideoImageEncoderPreference::Auto,
            .jpeg_quality = 85,
            .max_encoded_bytes = 1024 * 1024,
        });
    if (!encoder.ok()) {
        return encoder.status();
    }

    media::OpenCvFrameSampler sampler({
        .resize_width = 320,
        .temporal_vote_window = 1,
        .temporal_vote_required = 1,
        .peak_threshold = 0.20,
        .cooldown_seconds = 0.0,
        .adaptive_enabled = false,
    });

    const auto ipc_name = "agent-media-stress-" + scenario.name + "-" + std::to_string(NowNs());
    const ipc::media::InferenceFrameSharedMemoryOptions ipc_options{
        .name = ipc_name,
        .slot_count = scenario.ipc_slots,
        .payload_capacity = 1024 * 1024,
        .remove_existing = true,
        .remove_on_destroy = true,
    };
    auto sink_channel = ipc::media::SharedMemoryInferenceFrameChannel::Create(ipc_options);
    if (!sink_channel.ok()) {
        return sink_channel.status();
    }
    auto source_options = ipc_options;
    source_options.remove_existing = false;
    source_options.remove_on_destroy = false;
    auto source_channel = ipc::media::SharedMemoryInferenceFrameChannel::Open(source_options);
    if (!source_channel.ok()) {
        return source_channel.status();
    }

    media::InferenceFrameGatewayProducer producer(*sink_channel.value());
    media::inference::SegmentedInferenceFrameBacklog backlog({
        .max_sessions = scenario.sessions,
        .segments_per_session = 1,
        .slots_per_segment = scenario.backlog_slots_per_session,
        .default_wait_timeout = 20ms,
    });
    media::inference::SessionInferenceFrameResultTable result_table({
        .max_sessions = scenario.sessions,
        .max_results_per_session = frames_per_session,
    });
    FakeVlm fake_vlm(records, total_frames, scenario.vlm_latency);
    media::inference::InferenceFrameCoordinator coordinator(
        backlog,
        fake_vlm,
        result_table,
        {
            .worker_count = scenario.vlm_workers,
            .wait_timeout = 20ms,
            .shutdown_backlog = true,
        });

    media::inference::InferenceFrameIpcReceiver receiver(
        *source_channel.value(),
        inference_pool,
        backlog,
        {},
        [&](const ipc::media::SharedFrameMetadata& metadata, const core::Status& status) {
            const auto index = RecordIndex(metadata.frame_id, total_frames);
            if (!index) {
                return;
            }
            if (status.ok()) {
                records[*index].receiver_admitted_at.store(NowNs(), std::memory_order_release);
            } else {
                records[*index].receiver_rejected.store(true, std::memory_order_release);
            }
        });

    core::ThreadPool compute_pool({
        .worker_count = scenario.compute_workers,
        .queue_capacity = scenario.compute_queue,
        .name = "media-stress-compute",
    });
    core::ThreadPool io_pool({
        .worker_count = scenario.io_workers,
        .queue_capacity = scenario.io_queue,
        .name = "media-stress-io",
    });
    auto status = compute_pool.Start();
    if (!status.ok()) {
        return status;
    }
    status = io_pool.Start();
    if (!status.ok()) {
        compute_pool.Shutdown(false);
        return status;
    }
    status = coordinator.Start();
    if (!status.ok()) {
        io_pool.Shutdown(false);
        compute_pool.Shutdown(false);
        return status;
    }

    std::atomic<bool> receiver_running{true};
    std::jthread receiver_thread([&] {
        while (receiver_running.load(std::memory_order_acquire)) {
            const auto poll_status = receiver.PollOnce();
            if (poll_status.ok()) {
                continue;
            }
            if (poll_status.code() == core::ErrorCode::NotFound) {
                std::this_thread::sleep_for(100us);
                continue;
            }
            if (poll_status.code() == core::ErrorCode::Cancelled) {
                break;
            }
        }
    });

    const auto generated_start = Clock::now();
    const auto frame_interval = std::chrono::duration<double>(1.0 / static_cast<double>(scenario.fps));
    for (std::size_t frame_in_session = 0; frame_in_session < frames_per_session; ++frame_in_session) {
        const auto due = generated_start + std::chrono::duration_cast<Clock::duration>(frame_interval * frame_in_session);
        std::this_thread::sleep_until(due);
        for (std::size_t session_index = 0; session_index < scenario.sessions; ++session_index) {
            const auto record_index = session_index * frames_per_session + frame_in_session;
            const auto frame_id = static_cast<std::uint64_t>(record_index + 1);
            auto& record = records[record_index];
            record.session_index = session_index;
            record.event_index = EventIndex(scenario, session_index, frame_in_session, frames_per_session);
            record.generated_at.store(NowNs(), std::memory_order_release);
            const auto event_frame = record.event_index != std::numeric_limits<std::size_t>::max();
            auto rgb = MakeSyntheticRgb(kWidth, kHeight, session_index, frame_in_session, event_frame);
            media::VideoFrameView frame{
                .session_id = "session-" + std::to_string(session_index),
                .frame_id = frame_id,
                .captured_at = due,
                .width = kWidth,
                .height = kHeight,
                .row_stride_bytes = static_cast<std::size_t>(kWidth) * 3,
                .format = media::VideoPixelFormat::Rgb,
                .bytes = std::string_view(
                    reinterpret_cast<const char*>(rgb->data()),
                    rgb->size()),
            };

            const auto submit_status = compute_pool.Submit(
                [&, record_index, rgb = std::move(rgb), frame = std::move(frame)]() mutable -> core::Status {
                    auto& task_record = records[record_index];
                    task_record.compute_started_at.store(NowNs(), std::memory_order_release);
                    const auto decision = sampler.Evaluate(frame);
                    task_record.sampler_evaluated.store(true, std::memory_order_release);
                    if (!decision.ok()) {
                        task_record.encode_failed.store(true, std::memory_order_release);
                        return decision.status();
                    }
                    task_record.saliency.store(decision.value().saliency_score, std::memory_order_release);
                    if (!decision.value().submit_to_vlm) {
                        return core::Status::Ok();
                    }
                    task_record.sampler_selected.store(true, std::memory_order_release);
                    task_record.selected_at.store(NowNs(), std::memory_order_release);
                    auto encoded = encoder.value()->Encode(
                        frame,
                        decision.value().saliency_score,
                        "stress-" + std::to_string(frame.frame_id));
                    if (!encoded.ok()) {
                        task_record.encode_failed.store(true, std::memory_order_release);
                        return encoded.status();
                    }
                    task_record.encoded_at.store(NowNs(), std::memory_order_release);
                    const auto io_status = io_pool.Submit(
                        [&, record_index, encoded = std::move(encoded).value()]() mutable -> core::Status {
                            records[record_index].ipc_published_at.store(NowNs(), std::memory_order_release);
                            const auto publish_status = producer.Publish(std::move(encoded));
                            if (!publish_status.ok()) {
                                records[record_index].ipc_published_at.store(kUnsetTime, std::memory_order_release);
                                records[record_index].ipc_publish_rejected.store(true, std::memory_order_release);
                                return publish_status;
                            }
                            return core::Status::Ok();
                        },
                        {},
                        "media-stress-ipc-publish");
                    if (!io_status.ok()) {
                        task_record.io_rejected.store(true, std::memory_order_release);
                    }
                    return core::Status::Ok();
                },
                {},
                "media-stress-sample-encode");
            if (!submit_status.ok()) {
                record.compute_rejected.store(true, std::memory_order_release);
            }
        }
    }

    compute_pool.Shutdown(true);
    io_pool.Shutdown(true);
    const auto drain_deadline = Clock::now() + scenario.duration + 10s;
    while (Clock::now() < drain_deadline) {
        const auto ipc_snapshot = sink_channel.value()->Snapshot();
        const auto receiver_snapshot = receiver.Snapshot();
        const auto backlog_snapshot = backlog.Snapshot();
        const auto coordinator_snapshot = coordinator.Snapshot();
        if (ipc_snapshot.acknowledged_frames >= ipc_snapshot.published_frames &&
            backlog_snapshot.queued_frames == 0 &&
            coordinator_snapshot.processed_frames >= receiver_snapshot.submitted_frames) {
            break;
        }
        std::this_thread::sleep_for(2ms);
    }
    receiver_running.store(false, std::memory_order_release);
    receiver.Shutdown();
    receiver_thread.join();
    coordinator.Shutdown();

    const auto generated = total_frames;
    const auto compute_rejected = Count(records, total_frames, [](const FrameRecord& record) {
        return record.compute_rejected.load(std::memory_order_acquire);
    });
    const auto sampler_evaluated = Count(records, total_frames, [](const FrameRecord& record) {
        return record.sampler_evaluated.load(std::memory_order_acquire);
    });
    const auto selected = Count(records, total_frames, [](const FrameRecord& record) {
        return record.sampler_selected.load(std::memory_order_acquire);
    });
    const auto encode_failed = Count(records, total_frames, [](const FrameRecord& record) {
        return record.encode_failed.load(std::memory_order_acquire);
    });
    const auto io_rejected = Count(records, total_frames, [](const FrameRecord& record) {
        return record.io_rejected.load(std::memory_order_acquire);
    });
    const auto ipc_rejected = Count(records, total_frames, [](const FrameRecord& record) {
        return record.ipc_publish_rejected.load(std::memory_order_acquire);
    });
    const auto ipc_published = Count(records, total_frames, [](const FrameRecord& record) {
        return record.ipc_published_at.load(std::memory_order_acquire) != kUnsetTime;
    });
    const auto receiver_rejected = Count(records, total_frames, [](const FrameRecord& record) {
        return record.receiver_rejected.load(std::memory_order_acquire);
    });
    const auto backlog_submitted = Count(records, total_frames, [](const FrameRecord& record) {
        return record.receiver_admitted_at.load(std::memory_order_acquire) != kUnsetTime;
    });
    const auto vlm_started = Count(records, total_frames, [](const FrameRecord& record) {
        return record.vlm_started_at.load(std::memory_order_acquire) != kUnsetTime;
    });
    const auto vlm_succeeded = Count(records, total_frames, [](const FrameRecord& record) {
        return record.vlm_succeeded.load(std::memory_order_acquire);
    });
    const auto vlm_failed = Count(records, total_frames, [](const FrameRecord& record) {
        return record.vlm_failed.load(std::memory_order_acquire);
    });
    const auto high_saliency_selected = Count(records, total_frames, [](const FrameRecord& record) {
        return record.sampler_selected.load(std::memory_order_acquire) &&
               record.saliency.load(std::memory_order_acquire) >= 0.7;
    });
    const auto high_saliency_dropped = Count(records, total_frames, [](const FrameRecord& record) {
        return record.sampler_selected.load(std::memory_order_acquire) &&
               record.saliency.load(std::memory_order_acquire) >= 0.7 &&
               record.vlm_started_at.load(std::memory_order_acquire) == kUnsetTime;
    });

    const auto event_count = scenario.sessions * 2;
    std::vector<bool> event_selected(event_count, false);
    std::vector<bool> event_covered(event_count, false);
    for (std::size_t index = 0; index < total_frames; ++index) {
        const auto event_index = records[index].event_index;
        if (event_index >= event_count || !records[index].sampler_selected.load(std::memory_order_acquire)) {
            continue;
        }
        event_selected[event_index] = true;
        if (records[index].vlm_started_at.load(std::memory_order_acquire) != kUnsetTime) {
            event_covered[event_index] = true;
        }
    }
    const auto selected_events = static_cast<std::size_t>(
        std::count(event_selected.begin(), event_selected.end(), true));
    const auto covered_events = static_cast<std::size_t>(
        std::count(event_covered.begin(), event_covered.end(), true));

    std::vector<double> session_completion;
    session_completion.reserve(scenario.sessions);
    for (std::size_t session_index = 0; session_index < scenario.sessions; ++session_index) {
        std::size_t session_selected = 0;
        std::size_t session_completed = 0;
        for (std::size_t frame_index = 0; frame_index < frames_per_session; ++frame_index) {
            const auto& record = records[session_index * frames_per_session + frame_index];
            if (record.sampler_selected.load(std::memory_order_acquire)) {
                ++session_selected;
                if (record.vlm_finished_at.load(std::memory_order_acquire) != kUnsetTime) {
                    ++session_completed;
                }
            }
        }
        session_completion.push_back(1.0 - Rate(session_selected - session_completed, session_selected));
    }
    const auto [min_completion, max_completion] = std::minmax_element(
        session_completion.begin(), session_completion.end());

    const auto sampler_queue = LatencyPercentiles(
        records,
        total_frames,
        [](const FrameRecord& record) { return record.generated_at.load(std::memory_order_acquire); },
        [](const FrameRecord& record) { return record.compute_started_at.load(std::memory_order_acquire); });
    const auto encode_latency = LatencyPercentiles(
        records,
        total_frames,
        [](const FrameRecord& record) { return record.selected_at.load(std::memory_order_acquire); },
        [](const FrameRecord& record) { return record.encoded_at.load(std::memory_order_acquire); });
    const auto ipc_admission = LatencyPercentiles(
        records,
        total_frames,
        [](const FrameRecord& record) { return record.ipc_published_at.load(std::memory_order_acquire); },
        [](const FrameRecord& record) { return record.receiver_admitted_at.load(std::memory_order_acquire); });
    const auto backlog_wait = LatencyPercentiles(
        records,
        total_frames,
        [](const FrameRecord& record) { return record.receiver_admitted_at.load(std::memory_order_acquire); },
        [](const FrameRecord& record) { return record.vlm_started_at.load(std::memory_order_acquire); });
    const auto vlm_latency = LatencyPercentiles(
        records,
        total_frames,
        [](const FrameRecord& record) { return record.vlm_started_at.load(std::memory_order_acquire); },
        [](const FrameRecord& record) { return record.vlm_finished_at.load(std::memory_order_acquire); });
    const auto selected_to_complete = LatencyPercentiles(
        records,
        total_frames,
        [](const FrameRecord& record) { return record.selected_at.load(std::memory_order_acquire); },
        [](const FrameRecord& record) { return record.vlm_finished_at.load(std::memory_order_acquire); });

    const auto coordinator_snapshot = coordinator.Snapshot();
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nscenario=" << scenario.name
              << " sessions=" << scenario.sessions
              << " fps/session=" << scenario.fps
              << " duration_ms=" << scenario.duration.count()
              << " vlm_workers=" << scenario.vlm_workers
              << " vlm_latency_ms=" << scenario.vlm_latency.count() << '\n';
    std::cout << "  rtc_generated=" << generated
              << " compute_queue_rejected=" << compute_rejected
              << " sampler_evaluated=" << sampler_evaluated
              << " sampler_not_selected=" << (sampler_evaluated - selected)
              << " sampler_selected=" << selected << '\n';
    std::cout << "  encode_failed=" << encode_failed
              << " io_queue_rejected=" << io_rejected
              << " ipc_publish_rejected=" << ipc_rejected
              << " ipc_published=" << ipc_published
              << " receiver_admission_rejected=" << receiver_rejected
              << " backlog_submitted=" << backlog_submitted << '\n';
    std::cout << "  vlm_started=" << vlm_started
              << " vlm_succeeded=" << vlm_succeeded
              << " vlm_failed=" << vlm_failed
              << " result_publish_failed=" << coordinator_snapshot.result_publish_failures << '\n';
    std::cout << "  raw_drop_rate=" << Rate(compute_rejected, generated)
              << " post_sampler_drop_rate=" << Rate(selected - vlm_started, selected)
              << " terminal_loss_rate=" << Rate(selected - vlm_succeeded - vlm_failed, selected) << '\n';
    std::cout << "  high_saliency_selected=" << high_saliency_selected
              << " high_saliency_dropped=" << high_saliency_dropped
              << " high_saliency_drop_rate=" << Rate(high_saliency_dropped, high_saliency_selected) << '\n';
    std::cout << "  selected_event_count=" << selected_events
              << " vlm_covered_event_count=" << covered_events
              << " event_coverage_rate=" << Rate(covered_events, selected_events)
              << " events_with_all_selected_frames_dropped=" << (selected_events - covered_events) << '\n';
    std::cout << "  session_completion_min=" << *min_completion
              << " session_completion_max=" << *max_completion
              << " completion_spread=" << (*max_completion - *min_completion) << '\n';
    PrintLatency("sampler_queue", sampler_queue);
    PrintLatency("encode", encode_latency);
    PrintLatency("ipc_admission", ipc_admission);
    PrintLatency("backlog_wait", backlog_wait);
    PrintLatency("fake_vlm", vlm_latency);
    PrintLatency("selected_to_vlm_complete", selected_to_complete);
    return core::Status::Ok();
}

std::vector<Scenario> Scenarios() {
    return {
        {
            .name = "balanced",
            .sessions = 4,
            .fps = 15,
            .compute_workers = 4,
            .compute_queue = 128,
            .io_workers = 2,
            .io_queue = 64,
            .ipc_slots = 32,
            .backlog_slots_per_session = 16,
            .vlm_workers = 4,
            .vlm_latency = 20ms,
        },
        {
            .name = "moderate",
            .sessions = 8,
            .fps = 30,
            .compute_workers = 4,
            .compute_queue = 128,
            .io_workers = 2,
            .io_queue = 64,
            .ipc_slots = 32,
            .backlog_slots_per_session = 12,
            .vlm_workers = 2,
            .vlm_latency = 100ms,
        },
        {
            .name = "vlm_overload",
            .sessions = 8,
            .fps = 30,
            .compute_workers = 4,
            .compute_queue = 128,
            .io_workers = 2,
            .io_queue = 32,
            .ipc_slots = 16,
            .backlog_slots_per_session = 6,
            .vlm_workers = 2,
            .vlm_latency = 300ms,
        },
        {
            .name = "synchronized_burst",
            .sessions = 8,
            .fps = 30,
            .compute_workers = 4,
            .compute_queue = 64,
            .io_workers = 1,
            .io_queue = 16,
            .ipc_slots = 8,
            .backlog_slots_per_session = 4,
            .vlm_workers = 2,
            .vlm_latency = 250ms,
            .synchronized_burst = true,
        },
    };
}

} // namespace

int main(int argc, char** argv) {
    gst_init(&argc, &argv);
    spdlog::set_level(spdlog::level::err);
    const std::string requested = argc > 1 ? argv[1] : "all";
    bool matched = false;
    for (const auto& scenario : Scenarios()) {
        if (requested != "all" && requested != scenario.name) {
            continue;
        }
        matched = true;
        const auto status = RunScenario(scenario);
        if (!status.ok()) {
            std::cerr << "scenario " << scenario.name << " failed: " << status.message() << '\n';
            return EXIT_FAILURE;
        }
    }
    if (!matched) {
        std::cerr << "unknown scenario: " << requested
                  << " (expected all, balanced, moderate, vlm_overload, or synchronized_burst)\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
