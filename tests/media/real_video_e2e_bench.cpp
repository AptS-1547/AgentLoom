#include "frame_encoding.h"
#include "inference_frame_coordinator.h"
#include "inference_frame_gateway_producer.h"
#include "inference_frame_ipc_receiver.h"
#include "inference_frame_shared_memory.h"
#include "inference_frame_spool.h"
#include "inference_frame_spool_replayer.h"
#include "keyed_serial_executor.h"
#include "media_inference_execution.h"
#include "memory_pool.h"
#include "opencv_frame_sampler.h"
#include "ordered_encoded_frame_sink.h"
#include "ordered_inference_frame_admission.h"
#include "skill_session_manager.h"
#include "thread_pool.h"

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
using namespace std::chrono_literals;

std::int64_t NowNs() noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
}

struct LatencySummary {
    std::size_t samples = 0;
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double max_ms = 0.0;
};

LatencySummary Summarize(std::vector<double> values) {
    if (values.empty()) return {};
    std::sort(values.begin(), values.end());
    const auto percentile = [&](double quantile) {
        const auto index = static_cast<std::size_t>(
            std::ceil(quantile * static_cast<double>(values.size())) - 1.0);
        return values[std::min(index, values.size() - 1)];
    };
    double sum = 0.0;
    for (const auto value : values) sum += value;
    return {
        .samples = values.size(),
        .mean_ms = sum / static_cast<double>(values.size()),
        .p50_ms = percentile(0.50),
        .p95_ms = percentile(0.95),
        .p99_ms = percentile(0.99),
        .max_ms = values.back(),
    };
}

Json LatencyJson(const LatencySummary& value) {
    return {
        {"samples", value.samples},
        {"meanMs", value.mean_ms},
        {"p50Ms", value.p50_ms},
        {"p95Ms", value.p95_ms},
        {"p99Ms", value.p99_ms},
        {"maxMs", value.max_ms},
    };
}

struct FrameTiming {
    std::string session_id;
    std::int64_t playback_due_at = 0;
    std::int64_t decoded_at = 0;
    std::int64_t compute_started_at = 0;
    std::int64_t sampled_at = 0;
    std::int64_t selected_at = 0;
    std::int64_t encoded_at = 0;
    std::int64_t gateway_task_started_at = 0;
    std::int64_t ipc_published_at = 0;
    std::int64_t execution_route_at = 0;
    std::int64_t execution_admit_returned_at = 0;
    std::int64_t vlm_started_at = 0;
    std::int64_t vlm_finished_at = 0;
    std::int64_t result_published_at = 0;
    std::uint64_t selected_sequence = 0;
    std::size_t encoded_bytes = 0;
    double saliency = 0.0;
};

struct FailureRecord {
    std::string scenario;
    std::string stage;
    std::string session_id;
    std::uint64_t frame_id = 0;
    core::ErrorCode code = core::ErrorCode::Ok;
    std::string message;
};

class Telemetry final {
public:
    std::shared_ptr<FrameTiming> Create(
        std::string key,
        std::string session_id,
        std::int64_t playback_due_at,
        std::int64_t decoded_at) {
        auto timing = std::make_shared<FrameTiming>();
        timing->session_id = std::move(session_id);
        timing->playback_due_at = playback_due_at;
        timing->decoded_at = decoded_at;
        std::lock_guard lock(mutex_);
        frames_.emplace(std::move(key), timing);
        return timing;
    }

    std::shared_ptr<FrameTiming> Find(std::string_view session_id, std::uint64_t frame_id) const {
        std::lock_guard lock(mutex_);
        const auto it = frames_.find(Key(session_id, frame_id));
        return it == frames_.end() ? nullptr : it->second;
    }

    std::vector<std::shared_ptr<FrameTiming>> Frames() const {
        std::lock_guard lock(mutex_);
        std::vector<std::shared_ptr<FrameTiming>> values;
        values.reserve(frames_.size());
        for (const auto& [_, timing] : frames_) values.push_back(timing);
        return values;
    }

    static std::string Key(std::string_view session_id, std::uint64_t frame_id) {
        return std::string(session_id) + "\n" + std::to_string(frame_id);
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<FrameTiming>> frames_;
};

class FailureLog final {
public:
    FailureLog(std::filesystem::path path, std::string scenario)
        : output_(std::move(path), std::ios::out | std::ios::trunc), scenario_(std::move(scenario)) {}

    void Add(std::string stage,
             core::Status status,
             std::string session_id = {},
             std::uint64_t frame_id = 0) {
        std::lock_guard lock(mutex_);
        failures_.push_back({scenario_, stage, session_id, frame_id, status.code(), status.message()});
        if (output_) {
            output_ << Json{
                {"scenario", scenario_},
                {"stage", stage},
                {"sessionId", session_id},
                {"frameId", frame_id},
                {"code", static_cast<int>(status.code())},
                {"message", status.message()},
            }.dump() << '\n';
            output_.flush();
        }
    }

    std::vector<FailureRecord> Snapshot() const {
        std::lock_guard lock(mutex_);
        return failures_;
    }

private:
    mutable std::mutex mutex_;
    std::ofstream output_;
    std::string scenario_;
    std::vector<FailureRecord> failures_;
};

struct PoolHighWater {
    std::size_t queued = 0;
    std::size_t active = 0;
    std::size_t submitted = 0;
    std::size_t rejected = 0;
};

void SamplePool(core::ThreadPool& pool, PoolHighWater& high_water) {
    const auto stats = pool.Stats();
    high_water.queued = std::max(high_water.queued, stats.queued_tasks);
    high_water.active = std::max(high_water.active, stats.active_workers);
    high_water.submitted = std::max(high_water.submitted, stats.submitted_tasks);
    high_water.rejected = std::max(high_water.rejected, stats.rejected_tasks);
}

struct Scenario {
    std::string name;
    bool worst_case = false;
    std::size_t gateway_workers = 1;
    std::size_t gateway_queue = 16;
    std::size_t receiver_workers = 1;
    std::size_t receiver_queue = 16;
    std::size_t execution_io_workers = 1;
    std::size_t execution_io_queue = 64;
    std::size_t ipc_slots = 8;
    std::size_t backlog_slots = 2;
    std::size_t vlm_workers = 1;
    std::chrono::milliseconds vlm_latency{20};
};

std::vector<Scenario> Scenarios() {
    struct IoLevel {
        std::string name;
        std::size_t gateway_workers;
        std::size_t gateway_queue;
        std::size_t receiver_workers;
        std::size_t receiver_queue;
        std::size_t execution_workers;
        std::size_t execution_queue;
        std::size_t slots;
    };
    const std::vector<IoLevel> levels{
        {"io_low", 1, 16, 1, 16, 1, 64, 8},
        {"io_medium", 2, 64, 2, 64, 2, 256, 32},
        {"io_high", 4, 256, 4, 256, 4, 1024, 64},
    };
    const std::vector<std::chrono::milliseconds> delays{20ms, 150ms, 500ms};
    std::vector<Scenario> scenarios;
    for (const auto& level : levels) {
        for (const auto delay : delays) {
            scenarios.push_back({
                .name = level.name + "_vlm_" + std::to_string(delay.count()) + "ms",
                .gateway_workers = level.gateway_workers,
                .gateway_queue = level.gateway_queue,
                .receiver_workers = level.receiver_workers,
                .receiver_queue = level.receiver_queue,
                .execution_io_workers = level.execution_workers,
                .execution_io_queue = level.execution_queue,
                .ipc_slots = level.slots,
                .backlog_slots = 2,
                .vlm_workers = 1,
                .vlm_latency = delay,
            });
        }
    }
    for (const auto& level : {levels.front(), levels.back()}) {
        for (const auto delay : {1000ms, 3000ms}) {
            scenarios.push_back({
                .name = level.name + "_vlm_" + std::to_string(delay.count()) + "ms",
                .worst_case = true,
                .gateway_workers = level.gateway_workers,
                .gateway_queue = level.gateway_queue,
                .receiver_workers = level.receiver_workers,
                .receiver_queue = level.receiver_queue,
                .execution_io_workers = level.execution_workers,
                .execution_io_queue = level.execution_queue,
                .ipc_slots = level.slots,
                .backlog_slots = 2,
                .vlm_workers = 1,
                .vlm_latency = delay,
            });
        }
    }
    const auto& low = levels.front();
    scenarios.push_back({
        .name = "io_low_vlm_5000ms",
        .worst_case = true,
        .gateway_workers = low.gateway_workers,
        .gateway_queue = low.gateway_queue,
        .receiver_workers = low.receiver_workers,
        .receiver_queue = low.receiver_queue,
        .execution_io_workers = low.execution_workers,
        .execution_io_queue = low.execution_queue,
        .ipc_slots = low.slots,
        .backlog_slots = 2,
        .vlm_workers = 1,
        .vlm_latency = 5000ms,
    });
    return scenarios;
}

class FakeVlm final : public media::IVlmVisionClient {
public:
    FakeVlm(Telemetry& telemetry, std::chrono::milliseconds delay)
        : telemetry_(telemetry), delay_(delay) {}

    core::Result<media::VisionInferenceResult> Analyze(
        const media::VisionInferenceRequest& request) override {
        if (auto timing = telemetry_.Find(request.session_id, request.frame_id)) {
            timing->vlm_started_at = NowNs();
        }
        std::this_thread::sleep_for(delay_);
        if (request.encoded_image.empty()) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "FakeVLM received empty encoded image");
        }
        if (auto timing = telemetry_.Find(request.session_id, request.frame_id)) {
            timing->vlm_finished_at = NowNs();
        }
        media::VisionInferenceResult result;
        result.scene_hint = "real-video-frame-" + std::to_string(request.frame_id);
        result.confidence = 0.9;
        return result;
    }

private:
    Telemetry& telemetry_;
    std::chrono::milliseconds delay_;
};

class ExecutionRouter final : public media::inference::IInferenceFrameAdmissionSink {
public:
    explicit ExecutionRouter(Telemetry& telemetry) : telemetry_(telemetry) {}

    void Register(
        std::string execution_id,
        std::shared_ptr<agent::service::persona::MediaInferenceExecution> execution) {
        std::lock_guard lock(mutex_);
        executions_.emplace(std::move(execution_id), std::move(execution));
    }

    core::Status AdmitFrame(media::inference::OwnedInferenceFrame frame) override {
        const auto session_id = frame.metadata().session_id;
        const auto frame_id = frame.metadata().frame_id;
        auto timing = telemetry_.Find(session_id, frame_id);
        if (timing) timing->execution_route_at = NowNs();
        std::shared_ptr<agent::service::persona::MediaInferenceExecution> execution;
        {
            std::lock_guard lock(mutex_);
            const auto it = executions_.find(frame.metadata().execution_id);
            if (it == executions_.end()) {
                return core::Status::Error(core::ErrorCode::NotFound, "media execution route not found");
            }
            execution = it->second;
        }
        auto status = execution->AdmitFrame(std::move(frame));
        if (timing) timing->execution_admit_returned_at = NowNs();
        return status;
    }

private:
    std::mutex mutex_;
    Telemetry& telemetry_;
    std::unordered_map<std::string, std::shared_ptr<agent::service::persona::MediaInferenceExecution>> executions_;
};

class FileDecoder final {
public:
    using Callback = std::function<core::Status(media::VideoFrameView)>;

    FileDecoder(std::filesystem::path path,
                std::string session_id,
                double playback_rate,
                std::chrono::milliseconds media_limit,
                Callback callback)
        : path_(std::move(path)),
          session_id_(std::move(session_id)),
          playback_rate_(playback_rate),
          media_limit_(media_limit),
          callback_(std::move(callback)) {}

    core::Status Run() {
        GError* uri_error = nullptr;
        gchar* uri = gst_filename_to_uri(path_.string().c_str(), &uri_error);
        if (!uri) {
            std::string message = uri_error && uri_error->message ? uri_error->message : "failed to create media URI";
            if (uri_error) g_error_free(uri_error);
            return core::Status::Error(core::ErrorCode::InvalidArgument, std::move(message));
        }
        const std::string pipeline_description =
            "uridecodebin uri=\"" + std::string(uri) +
            "\" ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink sync=false max-buffers=8 drop=false";
        g_free(uri);

        GError* parse_error = nullptr;
        GstElement* pipeline = gst_parse_launch(pipeline_description.c_str(), &parse_error);
        if (!pipeline) {
            std::string message = parse_error && parse_error->message ? parse_error->message : "failed to create decode pipeline";
            if (parse_error) g_error_free(parse_error);
            return core::Status::Error(core::ErrorCode::Unavailable, std::move(message));
        }
        std::unique_ptr<GstElement, decltype(&gst_object_unref)> pipeline_owner(pipeline, gst_object_unref);
        GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
        if (!sink) return core::Status::Error(core::ErrorCode::InternalError, "decode appsink is missing");
        std::unique_ptr<GstElement, decltype(&gst_object_unref)> sink_owner(sink, gst_object_unref);
        GstBus* bus = gst_element_get_bus(pipeline);
        if (!bus) return core::Status::Error(core::ErrorCode::InternalError, "decode bus is missing");
        std::unique_ptr<GstBus, decltype(&gst_object_unref)> bus_owner(bus, gst_object_unref);

        if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
            return core::Status::Error(core::ErrorCode::Unavailable, "failed to start decode pipeline");
        }
        const auto wall_start = Clock::now();
        std::uint64_t frame_id = 0;
        core::Status status = core::Status::Ok();
        bool done = false;
        while (!done) {
            GstSample* sample = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), 20 * GST_MSECOND);
            if (sample) {
                GstBuffer* buffer = gst_sample_get_buffer(sample);
                GstCaps* caps = gst_sample_get_caps(sample);
                if (!buffer || !caps) {
                    gst_sample_unref(sample);
                    status = core::Status::Error(core::ErrorCode::InvalidArgument, "decoded sample is incomplete");
                    break;
                }
                const auto pts = GST_BUFFER_PTS_IS_VALID(buffer) ? GST_BUFFER_PTS(buffer) : 0;
                const auto media_elapsed = std::chrono::nanoseconds(pts);
                if (media_limit_.count() > 0 && media_elapsed > media_limit_) {
                    gst_sample_unref(sample);
                    break;
                }
                const auto scaled = playback_rate_ > 0.0
                    ? std::chrono::duration_cast<Clock::duration>(
                          std::chrono::duration<double>(media_elapsed).count() / playback_rate_ * 1s)
                    : Clock::duration::zero();
                const auto playback_due = wall_start + scaled;
                if (playback_rate_ > 0.0) std::this_thread::sleep_until(playback_due);
                const GstStructure* structure = gst_caps_get_structure(caps, 0);
                gint width = 0;
                gint height = 0;
                gst_structure_get_int(structure, "width", &width);
                gst_structure_get_int(structure, "height", &height);
                GstVideoInfo info;
                gst_video_info_init(&info);
                gst_video_info_from_caps(&info, caps);
                auto mapped = media::GstMappedFrameBuffer::Create(buffer);
                if (!mapped.ok()) {
                    gst_sample_unref(sample);
                    status = mapped.status();
                    break;
                }
                media::VideoFrameView frame{
                    .session_id = session_id_,
                    .frame_id = ++frame_id,
                    .captured_at = playback_due,
                    .timestamp_us = static_cast<std::int64_t>(pts / GST_USECOND),
                    .width = static_cast<std::uint32_t>(width),
                    .height = static_cast<std::uint32_t>(height),
                    .row_stride_bytes = static_cast<std::size_t>(std::max(0, GST_VIDEO_INFO_PLANE_STRIDE(&info, 0))),
                    .format = media::VideoPixelFormat::Rgb,
                    .buffer = std::move(mapped).value(),
                };
                frame.bytes = frame.buffer->bytes();
                status = callback_(std::move(frame));
                gst_sample_unref(sample);
                if (!status.ok()) break;
                continue;
            }

            GstMessage* message = gst_bus_pop_filtered(
                bus,
                static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
            if (!message) continue;
            if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
                GError* error = nullptr;
                gchar* debug = nullptr;
                gst_message_parse_error(message, &error, &debug);
                status = core::Status::Error(
                    core::ErrorCode::Unavailable,
                    error && error->message ? error->message : "decode pipeline failed");
                if (error) g_error_free(error);
                if (debug) g_free(debug);
            }
            gst_message_unref(message);
            done = true;
        }
        gst_element_set_state(pipeline, GST_STATE_NULL);
        decoded_frames_ = frame_id;
        return status;
    }

    std::size_t decoded_frames() const noexcept { return decoded_frames_; }

private:
    std::filesystem::path path_;
    std::string session_id_;
    double playback_rate_ = 1.0;
    std::chrono::milliseconds media_limit_{0};
    Callback callback_;
    std::size_t decoded_frames_ = 0;
};

struct SessionRuntime {
    std::string session_id;
    std::string execution_id;
    std::shared_ptr<media::inference::IInferenceFrameSpool> spool;
    std::shared_ptr<media::inference::IInferenceFrameSpoolReplayer> replayer;
    std::shared_ptr<agent::service::persona::MediaInferenceExecution> execution;
    std::atomic<std::uint64_t> selected_sequence{0};
    std::atomic<std::uint64_t> final_transport_sequence{0};
    std::atomic<std::size_t> finalized_results{0};
};

std::vector<double> Durations(
    const std::vector<std::shared_ptr<FrameTiming>>& frames,
    const auto& start,
    const auto& finish) {
    std::vector<double> values;
    for (const auto& frame : frames) {
        const auto begin = start(*frame);
        const auto end = finish(*frame);
        if (begin > 0 && end >= begin) values.push_back(static_cast<double>(end - begin) / 1'000'000.0);
    }
    return values;
}

Json RunScenario(const Scenario& scenario,
                 const std::vector<std::filesystem::path>& videos,
                 const std::filesystem::path& report_root,
                 double playback_rate,
                 std::chrono::milliseconds media_limit) {
    const auto scenario_root = report_root / scenario.name;
    std::filesystem::create_directories(scenario_root);
    FailureLog failures(scenario_root / "failures.jsonl", scenario.name);
    Telemetry telemetry;
    core::BucketMemoryPool gateway_memory;
    core::BucketMemoryPool inference_memory;

    auto encoder_result = media::GStreamerVideoFrameEncoder::Create(gateway_memory, {
        .format = media::EncodedVideoFrameFormat::Jpeg,
        .preference = media::VideoImageEncoderPreference::Auto,
        .jpeg_quality = 85,
        .max_encoded_bytes = 4 * 1024 * 1024,
    });
    if (!encoder_result.ok()) {
        failures.Add("encoder_create", encoder_result.status());
        return {{"scenario", scenario.name}, {"fatal", encoder_result.status().message()}};
    }
    auto encoder = std::move(encoder_result).value();
    auto sampler = std::make_shared<media::OpenCvFrameSampler>(media::OpenCvFrameSamplerConfig{
        .resize_width = 320,
        .temporal_vote_window = 1,
        .temporal_vote_required = 1,
        .peak_threshold = 0.20,
        .cooldown_seconds = 0.0,
        .adaptive_enabled = true,
        .adaptive_fps_min = 2.0,
        .adaptive_fps_max = 12.0,
    });

    auto gateway_pool = std::make_shared<core::ThreadPool>(core::ThreadPoolOptions{
        scenario.gateway_workers, scenario.gateway_queue, "real-video-gateway-io"});
    auto receiver_pool = std::make_shared<core::ThreadPool>(core::ThreadPoolOptions{
        scenario.receiver_workers, scenario.receiver_queue, "real-video-receiver-io"});
    auto compute_pool = std::make_shared<core::ThreadPool>(core::ThreadPoolOptions{
        4, 2048, "real-video-sampler-encode"});
    auto frame_executor = std::make_shared<core::KeyedSerialExecutor>(
        compute_pool,
        core::KeyedSerialExecutorOptions{
            .max_keys = videos.size(),
            .queue_capacity_per_key = 1024,
            .task_name_prefix = "real-video-frame-stream",
        });
    if (!gateway_pool->Start().ok() || !receiver_pool->Start().ok() || !compute_pool->Start().ok()) {
        return {{"scenario", scenario.name}, {"fatal", "failed to start media pools"}};
    }
    auto execution_runtime_result = agent::service::persona::MediaInferenceExecutionRuntime::Create({
        .io_pool = {scenario.execution_io_workers, scenario.execution_io_queue, "real-video-execution-io"},
        .control_pool = {2, 64, "real-video-control"},
        .aggregation_pool = {2, 64, "real-video-aggregation"},
    });
    if (!execution_runtime_result.ok()) {
        failures.Add("execution_runtime", execution_runtime_result.status());
        return {{"scenario", scenario.name}, {"fatal", execution_runtime_result.status().message()}};
    }
    auto execution_runtime = std::move(execution_runtime_result).value();

    const auto channel_name = "agent-real-video-" + scenario.name + "-" + std::to_string(NowNs());
    auto producer_channel = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = channel_name,
        .slot_count = scenario.ipc_slots,
        .payload_capacity = 4 * 1024 * 1024,
        .remove_existing = true,
        .remove_on_destroy = true,
    });
    auto consumer_channel = ipc::media::SharedMemoryInferenceFrameChannel::Open({
        .name = channel_name,
        .slot_count = scenario.ipc_slots,
        .payload_capacity = 4 * 1024 * 1024,
    });
    if (!producer_channel.ok() || !consumer_channel.ok()) {
        const auto status = producer_channel.ok() ? consumer_channel.status() : producer_channel.status();
        failures.Add("ipc_channel", status);
        return {{"scenario", scenario.name}, {"fatal", status.message()}};
    }
    auto producer = std::make_shared<media::InferenceFrameGatewayProducer>(*producer_channel.value());
    auto ordered_producer = std::make_shared<media::OrderedEncodedFrameSink>(
        producer,
        media::OrderedEncodedFrameSinkOptions{
            .max_executions = videos.size(),
            .window_capacity = 1024,
            .publish_observer = [&](const auto& metadata, const auto& status) {
                if (!status.ok()) {
                    failures.Add("ipc_publish", status, metadata.session_id, metadata.frame_id);
                    return;
                }
                if (auto timing = telemetry.Find(metadata.session_id, metadata.frame_id)) {
                    timing->ipc_published_at = NowNs();
                }
            },
        });
    auto backlog = std::make_shared<media::inference::SegmentedInferenceFrameBacklog>(
        media::inference::SegmentedFrameBacklogOptions{
            videos.size(), 1, scenario.backlog_slots, 2ms});
    auto result_table = std::make_shared<media::inference::SessionInferenceFrameResultTable>(
        media::inference::InferenceFrameResultTableOptions{videos.size(), 8192});
    auto skill_sessions = std::make_shared<agent::service::persona::SkillSessionManager>();
    std::vector<std::shared_ptr<SessionRuntime>> sessions;
    auto execution_router = std::make_shared<ExecutionRouter>(telemetry);
    auto ordered_admission = std::make_shared<media::inference::OrderedInferenceFrameAdmission>(
        execution_router,
        media::inference::OrderedInferenceFrameAdmissionOptions{
            .max_executions = videos.size(),
            .window_capacity = 1024,
        });
    std::unordered_map<std::string, std::weak_ptr<agent::service::persona::MediaInferenceExecution>> execution_registry;
    std::mutex registry_mutex;

    for (std::size_t index = 0; index < videos.size(); ++index) {
        auto session = std::make_shared<SessionRuntime>();
        session->session_id = "real-video-session-" + std::to_string(index);
        session->execution_id = scenario.name + "-execution-" + std::to_string(index);
        auto spool_result = media::inference::MappedInferenceFrameSpool::Create({
            .root_directory = scenario_root / "spool",
            .execution_id = session->execution_id,
            .segment_bytes = 32 * 1024 * 1024,
            .max_spool_bytes = 2ull * 1024 * 1024 * 1024,
            .flush_on_append = false,
            .remove_on_destroy = true,
        });
        if (!spool_result.ok()) {
            failures.Add("spool_create", spool_result.status(), session->session_id);
            return {{"scenario", scenario.name}, {"fatal", spool_result.status().message()}};
        }
        session->spool = std::shared_ptr<media::inference::IInferenceFrameSpool>(std::move(spool_result).value());
        session->replayer = std::make_shared<media::inference::InferenceFrameSpoolReplayer>(
            *session->spool, inference_memory, *backlog,
            media::inference::InferenceFrameSpoolReplayOptions{16});
        agent::service::persona::SkillSessionStartRequest start;
        start.execution_id = session->execution_id;
        start.session_id = session->session_id;
        start.skill_id = "vision.observe";
        start.trace_id = scenario.name;
        auto started = skill_sessions->Start(start);
        if (!started.ok()) return {{"scenario", scenario.name}, {"fatal", started.status().message()}};
        skill_sessions->MarkReady(start.session_id, start.skill_id, "ready", start.trace_id);
        auto execution_result = agent::service::persona::MediaInferenceExecution::Create(
            {session->execution_id, session->session_id, start.skill_id, start.trace_id, 1ms},
            {execution_runtime, backlog, session->spool, session->replayer, result_table, skill_sessions},
            [session](const auto& completion) {
                session->finalized_results.store(completion.results.size(), std::memory_order_release);
                return core::Status::Ok();
            });
        if (!execution_result.ok()) return {{"scenario", scenario.name}, {"fatal", execution_result.status().message()}};
        session->execution = std::move(execution_result).value();
        execution_router->Register(session->execution_id, session->execution);
        execution_registry.emplace(session->execution_id, session->execution);
        sessions.push_back(std::move(session));
    }

    FakeVlm fake_vlm(telemetry, scenario.vlm_latency);
    media::inference::InferenceFrameCoordinator coordinator(
        *backlog,
        fake_vlm,
        *result_table,
        {
            .worker_count = scenario.vlm_workers,
            .wait_timeout = 2ms,
            .shutdown_backlog = false,
            .terminal_observer = [&](auto event) {
                if (auto timing = telemetry.Find(event.frame.session_id, event.frame.frame_id)) {
                    timing->result_published_at = NowNs();
                }
                std::shared_ptr<agent::service::persona::MediaInferenceExecution> execution;
                {
                    std::lock_guard lock(registry_mutex);
                    if (auto it = execution_registry.find(event.frame.execution_id); it != execution_registry.end()) {
                        execution = it->second.lock();
                    }
                }
                if (execution) {
                    const auto status = execution->ObserveTerminal(std::move(event));
                    if (!status.ok()) failures.Add("terminal_observer", status);
                }
            },
        });
    if (!coordinator.Start().ok()) return {{"scenario", scenario.name}, {"fatal", "coordinator start failed"}};

    std::atomic<bool> receiver_running{true};
    std::atomic<std::size_t> receiver_inflight{0};
    media::inference::InferenceFrameIpcReceiver receiver(
        *consumer_channel.value(),
        inference_memory,
        *backlog,
        {},
        [&](const auto& metadata, const auto& status) {
            if (!status.ok() && status.code() != core::ErrorCode::NotFound) {
                failures.Add("receiver", status, metadata.session_id, metadata.frame_id);
            }
        },
        {.admission_sink = ordered_admission});
    std::vector<std::jthread> receiver_dispatchers;
    for (std::size_t index = 0; index < scenario.receiver_workers; ++index) {
        receiver_dispatchers.emplace_back([&] {
            while (receiver_running.load(std::memory_order_acquire)) {
                auto status = receiver_pool->Submit([&] {
                    receiver_inflight.fetch_add(1, std::memory_order_relaxed);
                    const auto poll = receiver.PollOnce();
                    receiver_inflight.fetch_sub(1, std::memory_order_relaxed);
                    if (!poll.ok() && poll.code() != core::ErrorCode::NotFound && poll.code() != core::ErrorCode::Cancelled) {
                        failures.Add("receiver_poll", poll);
                    }
                    return core::Status::Ok();
                }, {}, "real-video-ipc-poll");
                if (!status.ok()) {
                    if (status.code() != core::ErrorCode::ResourceExhausted) failures.Add("receiver_queue", status);
                    std::this_thread::yield();
                } else {
                    std::this_thread::sleep_for(50us);
                }
            }
        });
    }

    PoolHighWater gateway_high_water;
    PoolHighWater receiver_high_water;
    PoolHighWater execution_high_water;
    std::atomic<bool> monitor_running{true};
    std::jthread monitor([&] {
        while (monitor_running.load(std::memory_order_acquire)) {
            SamplePool(*gateway_pool, gateway_high_water);
            SamplePool(*receiver_pool, receiver_high_water);
            SamplePool(execution_runtime->io_pool(), execution_high_water);
            std::this_thread::sleep_for(1ms);
        }
    });

    const auto scenario_started = Clock::now();
    std::vector<std::jthread> decoders;
    std::vector<std::shared_ptr<FileDecoder>> decoder_objects;
    for (std::size_t index = 0; index < videos.size(); ++index) {
        auto session = sessions[index];
        auto decoder = std::make_shared<FileDecoder>(
            videos[index], session->session_id, playback_rate, media_limit,
            [&, session](media::VideoFrameView frame) -> core::Status {
                const auto playback_due_at = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    frame.captured_at.time_since_epoch()).count();
                auto timing = telemetry.Create(
                    Telemetry::Key(frame.session_id, frame.frame_id),
                    frame.session_id,
                    playback_due_at,
                    NowNs());
                const auto status = frame_executor->Submit(
                    session->execution_id,
                    [&, session, frame = std::move(frame), timing]() mutable -> core::Status {
                        timing->compute_started_at = NowNs();
                        if (frame.buffer) frame.bytes = frame.buffer->bytes();
                        auto decision = sampler->Evaluate(frame);
                        timing->sampled_at = NowNs();
                        if (!decision.ok()) {
                            failures.Add("sampler", decision.status(), frame.session_id, frame.frame_id);
                            return decision.status();
                        }
                        timing->saliency = decision.value().saliency_score;
                        if (!decision.value().submit_to_vlm) return core::Status::Ok();
                        timing->selected_at = NowNs();
                        const auto sequence = session->selected_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
                        timing->selected_sequence = sequence;
                        auto encoded = encoder->Encode(frame, decision.value().saliency_score, scenario.name);
                        if (!encoded.ok()) {
                            session->execution->RecordBadFrame(session->execution_id, sequence, encoded.status());
                            const auto skipped = ordered_producer->MarkSkipped(
                                session->session_id,
                                session->execution_id,
                                sequence,
                                encoded.status());
                            if (!skipped.ok()) {
                                failures.Add("publish_order_skip", skipped, frame.session_id, frame.frame_id);
                            }
                            failures.Add("encoder", encoded.status(), frame.session_id, frame.frame_id);
                            return core::Status::Ok();
                        }
                        timing->encoded_at = NowNs();
                        timing->encoded_bytes = encoded.value().bytes().size();
                        encoded.value().metadata().execution_id = session->execution_id;
                        encoded.value().metadata().selected_sequence = sequence;
                        for (;;) {
                            auto submitted = gateway_pool->Submit(
                                [&, timing, encoded_frame = std::move(encoded).value()]() mutable -> core::Status {
                                    timing->gateway_task_started_at = NowNs();
                                    return ordered_producer->Publish(std::move(encoded_frame));
                                }, {}, "real-video-ipc-publish");
                            if (submitted.ok()) break;
                            if (submitted.code() != core::ErrorCode::ResourceExhausted) {
                                failures.Add("gateway_queue", submitted, frame.session_id, frame.frame_id);
                                return submitted;
                            }
                            std::this_thread::yield();
                        }
                        return core::Status::Ok();
                    }, "real-video-sample-encode");
                if (!status.ok()) failures.Add("compute_queue", status, frame.session_id, frame.frame_id);
                return status;
            });
        decoder_objects.push_back(decoder);
        decoders.emplace_back([decoder, &failures] {
            const auto status = decoder->Run();
            if (!status.ok()) failures.Add("decoder", status);
        });
    }
    decoders.clear();
    for (const auto& session : sessions) {
        const auto idle = frame_executor->WaitIdle(session->execution_id, 30s);
        if (!idle.ok()) failures.Add("frame_stream_drain", idle, session->session_id);
    }
    compute_pool->Shutdown(true);
    gateway_pool->Shutdown(true);
    for (const auto& session : sessions) {
        auto sealed = ordered_producer->SealExecution(
            session->session_id,
            session->execution_id,
            session->selected_sequence.load(std::memory_order_acquire));
        if (!sealed.ok()) {
            failures.Add("publish_order_seal", sealed.status(), session->session_id);
        } else {
            session->final_transport_sequence.store(sealed.value(), std::memory_order_release);
        }
    }

    const auto ipc_deadline = Clock::now() + 30s;
    while (Clock::now() < ipc_deadline) {
        const auto channel = producer_channel.value()->Snapshot();
        if (channel.acknowledged_frames >= channel.published_frames && receiver_inflight.load() == 0) break;
        std::this_thread::sleep_for(1ms);
    }
    for (const auto& session : sessions) {
        const auto status = ordered_admission->SealExecution(
            session->session_id,
            session->execution_id,
            session->final_transport_sequence.load(std::memory_order_acquire));
        if (!status.ok()) failures.Add("receive_order_seal", status, session->session_id);
    }
    for (const auto& session : sessions) {
        const auto status = session->execution->BeginClosing("real video input complete");
        if (!status.ok()) failures.Add("begin_closing", status, session->session_id);
    }
    std::size_t selected_for_timeout = 0;
    for (const auto& session : sessions) {
        selected_for_timeout += static_cast<std::size_t>(
            session->selected_sequence.load(std::memory_order_acquire));
    }
    const auto estimated_vlm_drain = scenario.vlm_latency *
        static_cast<std::int64_t>(selected_for_timeout + 8);
    const auto completion_timeout = std::max(
        std::chrono::duration_cast<std::chrono::milliseconds>(120s),
        estimated_vlm_drain + 30s);
    for (const auto& session : sessions) {
        auto completed = session->execution->WaitForCompletion(completion_timeout);
        if (!completed.ok()) failures.Add("execution_wait", completed.status(), session->session_id);
        else if (completed.value().state != agent::service::persona::MediaInferenceExecutionState::Closed) {
            failures.Add("execution_terminal", completed.value().terminal_status, session->session_id);
        }
    }
    const auto scenario_finished = Clock::now();
    receiver_running.store(false, std::memory_order_release);
    receiver_dispatchers.clear();
    receiver.Shutdown();
    receiver_pool->Shutdown(true);
    coordinator.Shutdown();
    monitor_running.store(false, std::memory_order_release);
    monitor.join();
    SamplePool(*gateway_pool, gateway_high_water);
    SamplePool(*receiver_pool, receiver_high_water);
    SamplePool(execution_runtime->io_pool(), execution_high_water);

    const auto frames = telemetry.Frames();
    const auto decoded = frames.size();
    const auto selected = static_cast<std::size_t>(std::count_if(frames.begin(), frames.end(), [](const auto& f) {
        return f->selected_at > 0;
    }));
    const auto encoded = static_cast<std::size_t>(std::count_if(frames.begin(), frames.end(), [](const auto& f) {
        return f->encoded_at > 0;
    }));
    const auto published = producer->Snapshot().published_frames;
    const auto terminal = coordinator.Snapshot().processed_frames;
    std::size_t hot = 0;
    std::size_t spooled = 0;
    std::size_t committed = 0;
    std::size_t execution_terminal = 0;
    std::size_t final_results = 0;
    std::size_t spool_payload_bytes = 0;
    std::size_t spool_allocated_bytes = 0;
    std::size_t spool_segments = 0;
    Json per_session = Json::array();
    for (const auto& session : sessions) {
        const auto snapshot = session->execution->Snapshot();
        hot += snapshot.hot_frames;
        spooled += snapshot.spooled_frames;
        committed += snapshot.committed_frames;
        execution_terminal += snapshot.terminal_frames;
        final_results += session->finalized_results.load(std::memory_order_acquire);
        const auto spool_snapshot = session->spool->Snapshot();
        spool_payload_bytes += spool_snapshot.payload_bytes;
        spool_allocated_bytes += spool_snapshot.allocated_bytes;
        spool_segments += spool_snapshot.segment_count;
        const auto decoded_for_session = static_cast<std::size_t>(std::count_if(
            frames.begin(), frames.end(), [&](const auto& frame) {
                return frame->session_id == session->session_id;
            }));
        const auto selected_for_session = static_cast<std::size_t>(std::count_if(
            frames.begin(), frames.end(), [&](const auto& frame) {
                return frame->session_id == session->session_id && frame->selected_at > 0;
            }));
        per_session.push_back({
            {"sessionId", session->session_id},
            {"executionId", session->execution_id},
            {"decoded", decoded_for_session},
            {"selected", selected_for_session},
            {"committed", snapshot.committed_frames},
            {"hot", snapshot.hot_frames},
            {"spooled", snapshot.spooled_frames},
            {"bad", snapshot.bad_frames},
            {"terminal", snapshot.terminal_frames},
            {"finalResults", session->finalized_results.load(std::memory_order_acquire)},
            {"spoolPayloadBytes", spool_snapshot.payload_bytes},
            {"spoolAllocatedBytes", spool_snapshot.allocated_bytes},
            {"spoolSegments", spool_snapshot.segment_count},
        });
    }

    auto pipeline_without_vlm = Durations(frames,
        [](const FrameTiming& f) { return f.selected_at; },
        [](const FrameTiming& f) {
            if (f.result_published_at <= 0 || f.vlm_started_at <= 0 || f.vlm_finished_at < f.vlm_started_at) return std::int64_t{0};
            return f.result_published_at - (f.vlm_finished_at - f.vlm_started_at);
        });
    for (std::size_t index = 0; index < pipeline_without_vlm.size(); ++index) {
        if (pipeline_without_vlm[index] < 0) pipeline_without_vlm[index] = 0;
    }

    const auto failure_records = failures.Snapshot();
    std::map<std::string, std::size_t> failures_by_stage;
    for (const auto& failure : failure_records) ++failures_by_stage[failure.stage];
    const auto elapsed_s = std::chrono::duration<double>(scenario_finished - scenario_started).count();
    Json report{
        {"scenario", scenario.name},
        {"configuration", {
            {"playbackRate", playback_rate},
            {"mediaLimitMs", media_limit.count()},
            {"vlmLatencyMs", scenario.vlm_latency.count()},
            {"worstCase", scenario.worst_case},
            {"completionTimeoutMs", completion_timeout.count()},
            {"vlmWorkers", scenario.vlm_workers},
            {"gatewayIoWorkers", scenario.gateway_workers},
            {"gatewayIoQueue", scenario.gateway_queue},
            {"receiverIoWorkers", scenario.receiver_workers},
            {"receiverIoQueue", scenario.receiver_queue},
            {"executionIoWorkers", scenario.execution_io_workers},
            {"executionIoQueue", scenario.execution_io_queue},
            {"ipcSlots", scenario.ipc_slots},
            {"backlogSlotsPerSession", scenario.backlog_slots},
            {"orderedSamplerWorkers", 4},
        }},
        {"counts", {
            {"decoded", decoded}, {"selected", selected}, {"encoded", encoded},
            {"ipcPublished", published}, {"committed", committed},
            {"hot", hot}, {"spooled", spooled}, {"coordinatorTerminal", terminal},
            {"spoolPayloadBytes", spool_payload_bytes},
            {"spoolAllocatedBytes", spool_allocated_bytes},
            {"spoolSegments", spool_segments},
            {"executionTerminal", execution_terminal}, {"finalResults", final_results},
            {"failures", failure_records.size()},
        }},
        {"invariants", {
            {"selectedEqualsEncodedPlusBad", selected == encoded},
            {"publishedEqualsCommitted", published == committed},
            {"committedEqualsTerminal", committed == execution_terminal},
            {"terminalEqualsFinalResults", execution_terminal == final_results},
            {"zeroSilentLoss", published == committed && committed == execution_terminal && execution_terminal == final_results},
        }},
        {"throughput", {
            {"elapsedSeconds", elapsed_s},
            {"decodedFramesPerSecond", elapsed_s > 0 ? decoded / elapsed_s : 0.0},
            {"selectedFramesPerSecond", elapsed_s > 0 ? selected / elapsed_s : 0.0},
        }},
        {"poolHighWater", {
            {"gateway", {{"queued", gateway_high_water.queued}, {"active", gateway_high_water.active}, {"submitted", gateway_high_water.submitted}, {"rejected", gateway_high_water.rejected}}},
            {"receiver", {{"queued", receiver_high_water.queued}, {"active", receiver_high_water.active}, {"submitted", receiver_high_water.submitted}, {"rejected", receiver_high_water.rejected}}},
            {"execution", {{"queued", execution_high_water.queued}, {"active", execution_high_water.active}, {"submitted", execution_high_water.submitted}, {"rejected", execution_high_water.rejected}}},
        }},
        {"perSession", std::move(per_session)},
        {"latency", {
            {"playbackDueToDecoded", LatencyJson(Summarize(Durations(frames, [](const auto& f) { return f.playback_due_at; }, [](const auto& f) { return f.decoded_at; })))},
            {"decodeToCompute", LatencyJson(Summarize(Durations(frames, [](const auto& f) { return f.decoded_at; }, [](const auto& f) { return f.compute_started_at; })))},
            {"sampler", LatencyJson(Summarize(Durations(frames, [](const auto& f) { return f.compute_started_at; }, [](const auto& f) { return f.sampled_at; })))},
            {"encode", LatencyJson(Summarize(Durations(frames, [](const auto& f) { return f.selected_at; }, [](const auto& f) { return f.encoded_at; })))},
            {"gatewayQueue", LatencyJson(Summarize(Durations(frames, [](const auto& f) { return f.encoded_at; }, [](const auto& f) { return f.gateway_task_started_at; })))},
            {"ipcPublishCall", LatencyJson(Summarize(Durations(frames, [](const auto& f) { return f.gateway_task_started_at; }, [](const auto& f) { return f.ipc_published_at; })))},
            {"ipcPublishToExecutionRoute", LatencyJson(Summarize(Durations(frames, [](const auto& f) { return f.ipc_published_at; }, [](const auto& f) { return f.execution_route_at; })))},
            {"executionAdmitCall", LatencyJson(Summarize(Durations(frames, [](const auto& f) { return f.execution_route_at; }, [](const auto& f) { return f.execution_admit_returned_at; })))},
            {"executionRouteToVlmStart", LatencyJson(Summarize(Durations(frames, [](const auto& f) { return f.execution_route_at; }, [](const auto& f) { return f.vlm_started_at; })))},
            {"fakeVlm", LatencyJson(Summarize(Durations(frames, [](const auto& f) { return f.vlm_started_at; }, [](const auto& f) { return f.vlm_finished_at; })))},
            {"vlmFinishToResultPublish", LatencyJson(Summarize(Durations(frames, [](const auto& f) { return f.vlm_finished_at; }, [](const auto& f) { return f.result_published_at; })))},
            {"selectedToResult", LatencyJson(Summarize(Durations(frames, [](const auto& f) { return f.selected_at; }, [](const auto& f) { return f.result_published_at; })))},
            {"pipelineExcludingVlm", LatencyJson(Summarize(std::move(pipeline_without_vlm)))},
        }},
        {"failuresByStage", failures_by_stage},
    };
    std::ofstream(scenario_root / "metrics.json") << std::setw(2) << report << '\n';
    return report;
}

void WriteMarkdown(const std::filesystem::path& path,
                   const std::vector<std::filesystem::path>& videos,
                   const std::vector<Json>& reports) {
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    out << "# Real Video RTC Post-Decode Media E2E Performance Report\n\n";
    out << "This benchmark uses GStreamer file demux/decode as an RTC post-decode equivalent input. "
           "It does not measure browser signaling, RTP jitter, packet loss, or network transport.\n\n";
    out << "## Inputs\n\n";
    for (const auto& video : videos) out << "- `" << video.generic_string() << "`\n";
    out << "\n## Scenario Matrix\n\n";
    out << "| Scenario | VLM ms | Gateway IO | Receiver IO | Execution IO | Selected | Spool | Zero loss | Non-VLM p95 ms | End-to-end p95 ms | Failures |\n";
    out << "|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|\n";
    for (const auto& report : reports) {
        if (report.contains("fatal")) {
            out << "| " << report.value("scenario", "unknown") << " | - | - | - | - | - | - | no | - | - | fatal |\n";
            continue;
        }
        const auto& config = report["configuration"];
        const auto& counts = report["counts"];
        out << "| " << report["scenario"].get<std::string>()
            << " | " << config["vlmLatencyMs"]
            << " | " << config["gatewayIoWorkers"] << "/" << config["gatewayIoQueue"]
            << " | " << config["receiverIoWorkers"] << "/" << config["receiverIoQueue"]
            << " | " << config["executionIoWorkers"] << "/" << config["executionIoQueue"]
            << " | " << counts["selected"]
            << " | " << counts["spooled"]
            << " | " << (report["invariants"]["zeroSilentLoss"].get<bool>() ? "yes" : "no")
            << " | " << report["latency"]["pipelineExcludingVlm"]["p95Ms"]
            << " | " << report["latency"]["selectedToResult"]["p95Ms"]
            << " | " << counts["failures"] << " |\n";
    }
    out << "\n## Interpretation\n\n";
    out << "- `pipelineExcludingVlm` subtracts the measured FakeVLM service interval from selected-to-result latency.\n";
    out << "- `ipcPublishToExecutionRoute` includes receiver queueing, claim, pooled copy and ack.\n";
    out << "- `executionAdmitCall` measures the asynchronous enqueue call, not frame processing completion.\n";
    out << "- `executionRouteToVlmStart` includes execution admission, hot backlog wait, spool wait and replay delay.\n";
    out << "- Every scenario requires `ipcPublished == committed == executionTerminal == finalResults`.\n";
    out << "- Per-scenario `metrics.json` and `failures.jsonl` contain the complete observable data.\n";
}

} // namespace

int main(int argc, char** argv) {
    gst_init(&argc, &argv);
    const std::filesystem::path repo = std::filesystem::current_path();
    const std::vector<std::filesystem::path> videos{
        repo / "test" / "media" / "drink_test.avi",
        repo / "test" / "media" / "test_adaptive.avi",
    };
    for (const auto& video : videos) {
        if (!std::filesystem::exists(video)) {
            std::cerr << "missing real video: " << video << '\n';
            return 2;
        }
    }
    const std::string requested = argc > 1 ? argv[1] : "all";
    const auto media_seconds = argc > 2 ? std::stod(argv[2]) : 12.0;
    const auto playback_rate = argc > 3 ? std::stod(argv[3]) : 4.0;
    const auto timestamp = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    const auto report_root = repo / "build" / "reports" / ("media-real-video-e2e-" + timestamp);
    std::filesystem::create_directories(report_root);

    std::vector<Json> reports;
    bool matched = false;
    for (const auto& scenario : Scenarios()) {
        if (requested == "all" && scenario.worst_case) continue;
        if (requested == "worst" && !scenario.worst_case) continue;
        if (requested != "all" && requested != "worst" && requested != scenario.name) continue;
        matched = true;
        std::cout << "running " << scenario.name << "\n";
        reports.push_back(RunScenario(
            scenario,
            videos,
            report_root,
            playback_rate,
            std::chrono::milliseconds(static_cast<std::int64_t>(media_seconds * 1000.0))));
    }
    if (!matched) {
        std::cerr << "unknown scenario: " << requested << '\n';
        return 2;
    }
    WriteMarkdown(report_root / "PERFORMANCE_REPORT.md", videos, reports);
    std::ofstream(report_root / "matrix.json") << std::setw(2) << reports << '\n';
    std::cout << "report=" << (report_root / "PERFORMANCE_REPORT.md") << '\n';
    const auto failed = std::any_of(reports.begin(), reports.end(), [](const Json& report) {
        return report.contains("fatal") ||
               !report.value("invariants", Json::object()).value("zeroSilentLoss", false);
    });
    return failed ? 1 : 0;
}
