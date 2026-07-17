#include "frame_encoding.h"
#include "inference_frame_gateway_producer.h"
#include "inference_frame_ipc_control.h"
#include "inference_frame_ipc_grpc_signal.h"
#include "inference_frame_ipc_lifecycle.h"
#include "opencv_frame_sampler.h"
#include "ordered_encoded_frame_sink.h"

#include "multimodal_inference.grpc.pb.h"

#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Json = nlohmann::json;
using namespace std::chrono_literals;

struct SessionRun {
    std::string session_id;
    std::string execution_id;
    std::filesystem::path video;
    double gaussian_blur_sigma = 0.0;
    std::uint64_t decoded = 0;
    std::uint64_t selected = 0;
    std::uint64_t encoded = 0;
    std::uint64_t final_transport_sequence = 0;
    core::Status status = core::Status::Ok();
    multimodal_inference::SkillMediaExecutionResponse final_response;
};

using FrameResult = multimodal_inference::SkillMediaFrameResult;

std::uint64_t NearestRankPercentile(
    const std::vector<std::uint64_t>& sorted_values,
    double percentile) {
    if (sorted_values.empty()) return 0;
    const auto rank = static_cast<std::size_t>(std::ceil(
        percentile * static_cast<double>(sorted_values.size())));
    return sorted_values[std::min(sorted_values.size() - 1, std::max<std::size_t>(1, rank) - 1)];
}

Json LatencySummary(std::vector<std::uint64_t> values) {
    values.erase(std::remove(values.begin(), values.end(), 0), values.end());
    if (values.empty()) {
        return {
            {"samples", 0},
            {"averageUs", 0.0},
            {"p50Us", 0},
            {"p95Us", 0},
            {"p99Us", 0},
            {"maxUs", 0},
            {"averageMs", 0.0},
            {"p50Ms", 0.0},
            {"p95Ms", 0.0},
            {"p99Ms", 0.0},
            {"maxMs", 0.0},
        };
    }
    std::sort(values.begin(), values.end());
    long double total = 0.0;
    for (const auto value : values) total += static_cast<long double>(value);
    const auto average_us = static_cast<double>(total / values.size());
    const auto p50 = NearestRankPercentile(values, 0.50);
    const auto p95 = NearestRankPercentile(values, 0.95);
    const auto p99 = NearestRankPercentile(values, 0.99);
    const auto maximum = values.back();
    return {
        {"samples", values.size()},
        {"averageUs", average_us},
        {"p50Us", p50},
        {"p95Us", p95},
        {"p99Us", p99},
        {"maxUs", maximum},
        {"averageMs", average_us / 1000.0},
        {"p50Ms", static_cast<double>(p50) / 1000.0},
        {"p95Ms", static_cast<double>(p95) / 1000.0},
        {"p99Ms", static_cast<double>(p99) / 1000.0},
        {"maxMs", static_cast<double>(maximum) / 1000.0},
    };
}

template <typename Selector>
Json StageSummary(
    const std::vector<const FrameResult*>& results,
    Selector selector) {
    std::vector<std::uint64_t> values;
    values.reserve(results.size());
    for (const auto* result : results) values.push_back(selector(*result));
    return LatencySummary(std::move(values));
}

Json TimingMetrics(const std::vector<const FrameResult*>& results) {
    return {
        {"frames", results.size()},
        {"publishToReceive", StageSummary(results, [](const auto& result) {
            return result.publish_to_receive_us();
        })},
        {"receiveToAdmit", StageSummary(results, [](const auto& result) {
            return result.receive_to_admit_us();
        })},
        {"queueWait", StageSummary(results, [](const auto& result) {
            return result.queue_wait_us();
        })},
        {"spoolResidence", StageSummary(results, [](const auto& result) {
            return result.spool_wait_us();
        })},
        {"streamWait", StageSummary(results, [](const auto& result) {
            return result.stream_wait_us();
        })},
        {"replayWait", StageSummary(results, [](const auto& result) {
            return result.replay_wait_us();
        })},
        {"inference", StageSummary(results, [](const auto& result) {
            return result.inference_us();
        })},
        {"endToEnd", StageSummary(results, [](const auto& result) {
            return result.publish_to_terminal_us();
        })},
        {"processingEndToEnd", StageSummary(results, [](const auto& result) {
            return result.publish_to_terminal_excluding_stream_us();
        })},
    };
}

template <typename Selector>
Json MillisecondSummary(
    const std::vector<const FrameResult*>& results,
    Selector selector) {
    std::vector<std::uint64_t> values;
    values.reserve(results.size());
    for (const auto* result : results) {
        values.push_back(static_cast<std::uint64_t>(
            std::max(0.0, selector(*result)) * 1000.0));
    }
    return LatencySummary(std::move(values));
}

Json InferenceMetrics(const std::vector<const FrameResult*>& results) {
    auto metrics = TimingMetrics(results);
    metrics["promptEval"] = MillisecondSummary(results, [](const auto& result) {
        return result.prompt_eval_ms();
    });
    metrics["eval"] = MillisecondSummary(results, [](const auto& result) {
        return result.eval_ms();
    });
    return metrics;
}

Json BuildMetrics(const std::vector<SessionRun>& sessions) {
    std::vector<const FrameResult*> all;
    std::vector<const FrameResult*> cache;
    std::vector<const FrameResult*> fresh;
    std::vector<const FrameResult*> prompt_kv_hit;
    std::vector<const FrameResult*> prompt_kv_miss;
    std::vector<const FrameResult*> spooled;
    std::vector<const FrameResult*> hot;
    std::map<std::string, std::vector<const FrameResult*>> result_sources;
    for (const auto& session : sessions) {
        for (const auto& result : session.final_response.results()) {
            all.push_back(&result);
            (result.cache_hit() ? cache : fresh).push_back(&result);
            if (!result.cache_hit()) {
                (result.prompt_kv_cache_hit() ? prompt_kv_hit : prompt_kv_miss).push_back(&result);
            }
            (result.spooled() ? spooled : hot).push_back(&result);
            const auto source = result.result_source().empty()
                ? std::string("unspecified")
                : result.result_source();
            result_sources[source].push_back(&result);
        }
    }
    const auto hit_rate = all.empty()
        ? 0.0
        : static_cast<double>(cache.size()) / static_cast<double>(all.size());
    const auto prompt_kv_hit_rate = fresh.empty()
        ? 0.0
        : static_cast<double>(prompt_kv_hit.size()) / static_cast<double>(fresh.size());
    Json source_counts = Json::object();
    Json source_metrics = Json::object();
    for (const auto& [source, results] : result_sources) {
        source_counts[source] = results.size();
        source_metrics[source] = InferenceMetrics(results);
    }
    return {
        {"frames", all.size()},
        {"cacheHits", cache.size()},
        {"cacheHitRate", hit_rate},
        {"promptKvEligibleFrames", fresh.size()},
        {"promptKvHits", prompt_kv_hit.size()},
        {"promptKvHitRate", prompt_kv_hit_rate},
        {"resultSourceCounts", std::move(source_counts)},
        {"resultSources", std::move(source_metrics)},
        {"all", InferenceMetrics(all)},
        {"cache", InferenceMetrics(cache)},
        {"fresh", InferenceMetrics(fresh)},
        {"promptKvHit", InferenceMetrics(prompt_kv_hit)},
        {"promptKvMiss", InferenceMetrics(prompt_kv_miss)},
        {"spooled", InferenceMetrics(spooled)},
        {"hot", InferenceMetrics(hot)},
    };
}

std::string UniqueChannelName() {
    return "agent-real-vlm-" + std::to_string(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count());
}

void PrepareContext(grpc::ClientContext& context, std::chrono::milliseconds timeout) {
    context.set_deadline(std::chrono::system_clock::now() + timeout);
}

core::Status RpcStatus(std::string_view operation, const grpc::Status& status) {
    if (status.ok()) return core::Status::Ok();
    return core::Status::Error(
        status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED
            ? core::ErrorCode::Timeout
            : core::ErrorCode::Unavailable,
        std::string(operation) + " failed: " + status.error_message());
}

core::Status OpenExecution(
    multimodal_inference::MultimodalInference::Stub& stub,
    const SessionRun& session) {
    grpc::ClientContext context;
    PrepareContext(context, 10s);
    multimodal_inference::OpenSkillMediaExecutionRequest request;
    request.set_execution_id(session.execution_id);
    request.set_session_id(session.session_id);
    request.set_skill_id("vision.observe");
    request.set_trace_id("real-video-shared-vlm");
    multimodal_inference::SkillMediaExecutionResponse response;
    const auto status = stub.OpenSkillMediaExecution(&context, request, &response);
    if (!status.ok()) return RpcStatus("open media execution", status);
    if (!response.error().empty()) {
        return core::Status::Error(core::ErrorCode::InternalError, response.error());
    }
    return core::Status::Ok();
}

core::Status SealExecution(
    multimodal_inference::MultimodalInference::Stub& stub,
    const SessionRun& session) {
    grpc::ClientContext context;
    PrepareContext(context, 10s);
    multimodal_inference::SealSkillMediaInputRequest request;
    request.set_execution_id(session.execution_id);
    request.set_session_id(session.session_id);
    request.set_expected_selected_frames(session.encoded);
    request.set_final_transport_sequence(session.final_transport_sequence);
    request.set_reason("real video input complete");
    multimodal_inference::SkillMediaExecutionResponse response;
    const auto status = stub.SealSkillMediaInput(&context, request, &response);
    if (!status.ok()) return RpcStatus("seal media execution", status);
    if (!response.error().empty()) {
        return core::Status::Error(core::ErrorCode::InternalError, response.error());
    }
    return core::Status::Ok();
}

core::Result<multimodal_inference::SkillMediaExecutionResponse> GetExecution(
    multimodal_inference::MultimodalInference::Stub& stub,
    const SessionRun& session,
    bool include_results) {
    grpc::ClientContext context;
    PrepareContext(context, 10s);
    multimodal_inference::GetSkillMediaExecutionStatusRequest request;
    request.set_execution_id(session.execution_id);
    request.set_session_id(session.session_id);
    request.set_include_results(include_results);
    multimodal_inference::SkillMediaExecutionResponse response;
    const auto status = stub.GetSkillMediaExecutionStatus(&context, request, &response);
    if (!status.ok()) return RpcStatus("get media execution", status);
    return response;
}

core::Status RunVideo(
    SessionRun& session,
    media::IOrderedEncodedFrameSink& sink,
    double media_seconds,
    double playback_rate,
    std::size_t exact_replay_copies) {
    cv::VideoCapture capture(session.video.string());
    if (!capture.isOpened()) {
        return core::Status::Error(core::ErrorCode::NotFound, "failed to open real video input");
    }
    const double fps = std::max(1.0, capture.get(cv::CAP_PROP_FPS));
    const auto started = Clock::now();
    core::BucketMemoryPool memory_pool;
    auto encoder = media::GStreamerVideoFrameEncoder::Create(memory_pool, {
        .format = media::EncodedVideoFrameFormat::Jpeg,
        .preference = media::VideoImageEncoderPreference::Auto,
        .jpeg_quality = 85,
        .max_encoded_bytes = 4 * 1024 * 1024,
    });
    if (!encoder.ok()) return encoder.status();
    media::OpenCvFrameSampler sampler({
        .resize_width = 320,
        .temporal_vote_window = 1,
        .temporal_vote_required = 1,
        .peak_threshold = 0.20,
        .cooldown_seconds = 0.0,
        .adaptive_enabled = true,
        .adaptive_fps_min = 2.0,
        .adaptive_fps_max = 12.0,
    });

    cv::Mat frame;
    cv::Mat rgb;
    cv::Mat blurred_rgb;
    while (capture.read(frame)) {
        const auto frame_index = session.decoded++;
        const double source_seconds = static_cast<double>(frame_index) / fps;
        if (source_seconds > media_seconds) break;
        const auto due = started + std::chrono::duration_cast<Clock::duration>(
            std::chrono::duration<double>(source_seconds / playback_rate));
        std::this_thread::sleep_until(due);

        cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
        media::VideoFrameView view;
        view.session_id = session.session_id;
        view.frame_id = frame_index + 1;
        view.captured_at = Clock::now();
        view.timestamp_us = static_cast<std::int64_t>(source_seconds * 1'000'000.0);
        view.width = static_cast<std::uint32_t>(rgb.cols);
        view.height = static_cast<std::uint32_t>(rgb.rows);
        view.row_stride_bytes = rgb.step;
        view.format = media::VideoPixelFormat::Rgb;
        view.bytes = std::string_view(
            reinterpret_cast<const char*>(rgb.data),
            rgb.step * static_cast<std::size_t>(rgb.rows));

        auto decision = sampler.Evaluate(view);
        if (!decision.ok()) return decision.status();
        if (!decision.value().submit_to_vlm) continue;

        auto encoded_view = view;
        if (session.gaussian_blur_sigma > 0.0) {
            cv::GaussianBlur(
                rgb,
                blurred_rgb,
                cv::Size(0, 0),
                session.gaussian_blur_sigma,
                session.gaussian_blur_sigma,
                cv::BORDER_REPLICATE);
            encoded_view.row_stride_bytes = blurred_rgb.step;
            encoded_view.bytes = std::string_view(
                reinterpret_cast<const char*>(blurred_rgb.data),
                blurred_rgb.step * static_cast<std::size_t>(blurred_rgb.rows));
        }
        for (std::size_t replay = 0; replay <= exact_replay_copies; ++replay) {
            auto replay_view = encoded_view;
            replay_view.frame_id = view.frame_id * (exact_replay_copies + 1) + replay;
            replay_view.timestamp_us = view.timestamp_us.value_or(0) +
                static_cast<std::int64_t>(replay);
            ++session.selected;
            auto encoded = encoder.value()->Encode(
                replay_view,
                decision.value().saliency_score,
                "real-video-shared-vlm");
            if (!encoded.ok()) return encoded.status();
            encoded.value().metadata().execution_id = session.execution_id;
            encoded.value().metadata().selected_sequence = session.selected;
            auto status = sink.Publish(std::move(encoded).value());
            if (!status.ok()) return status;
            ++session.encoded;
        }
    }
    auto sealed = sink.SealExecution(
        session.session_id,
        session.execution_id,
        session.selected);
    if (!sealed.ok()) return sealed.status();
    session.final_transport_sequence = sealed.value();
    return core::Status::Ok();
}

Json ResultJson(const multimodal_inference::SkillMediaFrameResult& result) {
    const std::vector<std::string> facts(result.facts().begin(), result.facts().end());
    const std::vector<std::string> weak(
        result.weak_interpretations().begin(),
        result.weak_interpretations().end());
    return {
        {"selectedSequence", result.selected_sequence()},
        {"transportSequence", result.transport_sequence()},
        {"frameId", result.frame_id()},
        {"statusCode", result.status_code()},
        {"error", result.error()},
        {"sceneHint", result.scene_hint()},
        {"actionHint", result.action_hint()},
        {"objectHint", result.object_hint()},
        {"facts", facts},
        {"weakInterpretations", weak},
        {"rawText", result.raw_text()},
        {"confidence", result.confidence()},
        {"promptEvalMs", result.prompt_eval_ms()},
        {"evalMs", result.eval_ms()},
        {"promptTokens", result.prompt_tokens()},
        {"generatedTokens", result.generated_tokens()},
        {"cacheHit", result.cache_hit()},
        {"promptKvCacheHit", result.prompt_kv_cache_hit()},
        {"resultSource", result.result_source()},
        {"promptKvNear", {
            {"candidate", result.prompt_kv_near_candidate()},
            {"accepted", result.prompt_kv_near_accepted()},
            {"sameSession", result.prompt_kv_near_same_session()},
            {"globalCosine", result.prompt_kv_global_cosine()},
            {"meanTokenCosine", result.prompt_kv_mean_token_cosine()},
            {"p05TokenCosine", result.prompt_kv_p05_token_cosine()},
            {"minTokenCosine", result.prompt_kv_min_token_cosine()},
            {"relativeL2", result.prompt_kv_relative_l2()},
            {"maxAbsError", result.prompt_kv_max_abs_error()},
        }},
        {"timing", {
            {"publishToReceiveUs", result.publish_to_receive_us()},
            {"receiveToAdmitUs", result.receive_to_admit_us()},
            {"queueWaitUs", result.queue_wait_us()},
            {"spoolResidenceUs", result.spool_wait_us()},
            {"streamWaitUs", result.stream_wait_us()},
            {"replayWaitUs", result.replay_wait_us()},
            {"inferenceUs", result.inference_us()},
            {"publishToTerminalUs", result.publish_to_terminal_us()},
            {"publishToTerminalExcludingStreamUs", result.publish_to_terminal_excluding_stream_us()},
            {"spooled", result.spooled()},
        }},
    };
}

Json SessionJson(const SessionRun& session) {
    Json results = Json::array();
    for (const auto& result : session.final_response.results()) {
        results.push_back(ResultJson(result));
    }
    return {
        {"sessionId", session.session_id},
        {"executionId", session.execution_id},
        {"video", session.video.generic_string()},
        {"gaussianBlurSigma", session.gaussian_blur_sigma},
        {"decoded", session.decoded},
        {"selected", session.selected},
        {"encoded", session.encoded},
        {"finalTransportSequence", session.final_transport_sequence},
        {"state", session.final_response.state()},
        {"committed", session.final_response.committed_frames()},
        {"hot", session.final_response.hot_frames()},
        {"spooled", session.final_response.spooled_frames()},
        {"terminal", session.final_response.terminal_frames()},
        {"failedInference", session.final_response.failed_inference_frames()},
        {"error", session.final_response.error()},
        {"results", std::move(results)},
    };
}

} // namespace

int main(int argc, char** argv) {
    const std::filesystem::path repo = std::filesystem::current_path();
    const std::string target = argc > 1 ? argv[1] : "127.0.0.1:50051";
    const double media_seconds = argc > 2 ? std::stod(argv[2]) : 6.0;
    const double playback_rate = argc > 3 ? std::stod(argv[3]) : 4.0;
    const std::size_t exact_replay_copies = argc > 4
        ? static_cast<std::size_t>(std::stoull(argv[4]))
        : 0;
    const bool same_video_for_all_sessions = argc > 5
        ? std::stoull(argv[5]) != 0
        : false;
    const double gaussian_blur_sigma = argc > 6 ? std::stod(argv[6]) : 0.0;
    if (media_seconds <= 0.0 || playback_rate <= 0.0 || gaussian_blur_sigma < 0.0) {
        std::cerr << "media_seconds/playback_rate must be positive and gaussian_blur_sigma must be non-negative\n";
        return 2;
    }

    const auto run_id = std::to_string(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now().time_since_epoch()).count());
    std::vector<SessionRun> sessions{
        {"real-video-session-0-" + run_id, "real-vlm-execution-0-" + run_id, repo / "test" / "media" / "drink_test.avi"},
        {"real-video-session-1-" + run_id,
         "real-vlm-execution-1-" + run_id,
         repo / "test" / "media" /
             (same_video_for_all_sessions ? "drink_test.avi" : "test_adaptive.avi")},
    };
    for (auto& session : sessions) {
        session.gaussian_blur_sigma = gaussian_blur_sigma;
    }
    for (const auto& session : sessions) {
        if (!std::filesystem::exists(session.video)) {
            std::cerr << "missing video: " << session.video << '\n';
            return 2;
        }
    }

    auto channel = grpc::CreateChannel(target, grpc::InsecureChannelCredentials());
    if (!channel->WaitForConnected(std::chrono::system_clock::now() + 30s)) {
        std::cerr << "inference server is unavailable: " << target << '\n';
        return 1;
    }
    auto stub = multimodal_inference::MultimodalInference::NewStub(channel);

    auto created_sink = ipc::media::RecoverableInferenceFrameIpcSink::Create({
        .name = UniqueChannelName(),
        .slot_count = 64,
        .payload_capacity = 4 * 1024 * 1024,
        .remove_existing = true,
        .remove_on_destroy = true,
    });
    if (!created_sink.ok()) {
        std::cerr << created_sink.status().message() << '\n';
        return 1;
    }
    std::shared_ptr<ipc::media::IRecoverableInferenceFrameIpcSink> ipc_sink(
        std::move(created_sink).value());
    auto signal = std::make_shared<ipc::media::GrpcInferenceFrameIpcSignal>(
        channel,
        ipc::media::GrpcInferenceFrameIpcSignalOptions{.deadline = 10s});
    ipc::media::InferenceFrameIpcLeaseCoordinator lease(ipc_sink, signal);
    auto status = lease.Start();
    if (!status.ok()) {
        std::cerr << status.message() << '\n';
        return 1;
    }
    for (const auto& session : sessions) {
        status = OpenExecution(*stub, session);
        if (!status.ok()) {
            std::cerr << status.message() << '\n';
            lease.Shutdown();
            return 1;
        }
    }

    auto producer = std::make_shared<media::InferenceFrameGatewayProducer>(*ipc_sink);
    media::OrderedEncodedFrameSink ordered_sink(
        producer,
        {.max_executions = sessions.size(), .window_capacity = 4096});
    std::vector<std::jthread> workers;
    for (auto& session : sessions) {
        workers.emplace_back([&session, &ordered_sink, media_seconds, playback_rate, exact_replay_copies] {
            session.status = RunVideo(
                session,
                ordered_sink,
                media_seconds,
                playback_rate,
                exact_replay_copies);
        });
    }
    workers.clear();
    for (const auto& session : sessions) {
        if (!session.status.ok()) {
            std::cerr << session.execution_id << ": " << session.status.message() << '\n';
            lease.Shutdown();
            return 1;
        }
        status = SealExecution(*stub, session);
        if (!status.ok()) {
            std::cerr << session.execution_id << ": " << status.message() << '\n';
            lease.Shutdown();
            return 1;
        }
    }

    const auto completion_deadline = Clock::now() + 20min;
    while (Clock::now() < completion_deadline) {
        bool complete = true;
        for (auto& session : sessions) {
            auto current = GetExecution(*stub, session, false);
            if (!current.ok()) {
                std::cerr << current.status().message() << '\n';
                lease.Shutdown();
                return 1;
            }
            if (!current.value().complete()) complete = false;
            if (!current.value().error().empty() && current.value().state() == "failed") {
                std::cerr << session.execution_id << ": " << current.value().error() << '\n';
                lease.Shutdown();
                return 1;
            }
        }
        if (complete) break;
        std::this_thread::sleep_for(200ms);
    }

    bool valid = true;
    for (auto& session : sessions) {
        auto final = GetExecution(*stub, session, true);
        if (!final.ok()) {
            std::cerr << final.status().message() << '\n';
            valid = false;
            continue;
        }
        session.final_response = std::move(final).value();
        valid = valid && session.final_response.complete() && session.final_response.state() == "closed" &&
                session.final_response.results_size() == static_cast<int>(session.encoded) &&
                session.final_response.committed_frames() == session.encoded &&
                session.final_response.terminal_frames() == session.encoded;
    }
    const auto ipc = ipc_sink->Snapshot();
    valid = valid && ipc.published_frames == ipc.acknowledged_frames;

    const auto timestamp = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    const auto report_root = repo / "build" / "reports" / ("media-real-vlm-shm-" + timestamp);
    std::filesystem::create_directories(report_root);
    const auto metrics = BuildMetrics(sessions);
    Json report{
        {"transport", "shared_memory"},
        {"target", target},
        {"mediaSeconds", media_seconds},
        {"playbackRate", playback_rate},
        {"exactReplayCopies", exact_replay_copies},
        {"sameVideoForAllSessions", same_video_for_all_sessions},
        {"gaussianBlurSigma", gaussian_blur_sigma},
        {"ipc", {
            {"epoch", ipc.epoch},
            {"published", ipc.published_frames},
            {"claimed", ipc.claimed_frames},
            {"acknowledged", ipc.acknowledged_frames},
            {"rejected", ipc.rejected_frames},
        }},
        {"zeroSilentLoss", valid},
        {"metrics", metrics},
        {"sessions", Json::array()},
    };
    for (const auto& session : sessions) report["sessions"].push_back(SessionJson(session));
    std::ofstream(report_root / "report.json", std::ios::out | std::ios::trunc)
        << std::setw(2) << report << '\n';
    std::ofstream markdown(report_root / "REPORT.md", std::ios::out | std::ios::trunc);
    const auto& end_to_end = metrics["all"]["endToEnd"];
    const auto& processing_end_to_end = metrics["all"]["processingEndToEnd"];
    markdown << "# Real Video Shared-Memory VLM E2E\n\n"
             << "- target: `" << target << "`\n"
             << "- IPC published/acknowledged: " << ipc.published_frames << "/" << ipc.acknowledged_frames << "\n"
             << "- zero silent loss: " << (valid ? "yes" : "no") << "\n"
             << "- exact replay copies per selected image: " << exact_replay_copies << "\n"
             << "- same video for all sessions: "
             << (same_video_for_all_sessions ? "yes" : "no") << "\n"
             << "- Gaussian blur sigma: " << gaussian_blur_sigma << "\n"
             << "- cache hit rate: " << std::fixed << std::setprecision(3)
             << metrics["cacheHitRate"].get<double>() * 100.0 << "%\n"
             << "- prompt KV hit rate among result-cache misses: "
             << metrics["promptKvHitRate"].get<double>() * 100.0 << "% ("
             << metrics["promptKvHits"].get<std::size_t>() << "/"
             << metrics["promptKvEligibleFrames"].get<std::size_t>() << ")\n"
             << "- result sources: `" << metrics["resultSourceCounts"].dump() << "`\n"
             << "- observable full-path average/P99: " << end_to_end["averageMs"].get<double>()
             << "/" << end_to_end["p99Ms"].get<double>() << " ms\n"
             << "- processing full-path excluding stream wait average/P99: "
             << processing_end_to_end["averageMs"].get<double>()
             << "/" << processing_end_to_end["p99Ms"].get<double>() << " ms\n\n";
    for (const auto& session : sessions) {
        markdown << "## " << session.session_id << "\n\n"
                 << "decoded=" << session.decoded
                 << ", selected=" << session.selected
                 << ", spooled=" << session.final_response.spooled_frames()
                 << ", terminal=" << session.final_response.terminal_frames() << "\n\n";
        for (const auto& result : session.final_response.results()) {
            markdown << "- frame " << result.frame_id() << ": " << result.raw_text() << "\n";
        }
        markdown << '\n';
    }
    std::ofstream metrics_markdown(report_root / "METRICS.md", std::ios::out | std::ios::trunc);
    metrics_markdown << "# Full-Path Timing Metrics\n\n"
                     << "All percentiles use nearest-rank. Missing stage timestamps are excluded from that stage.\n\n"
                     << "- frames: " << metrics["frames"] << "\n"
                     << "- cache hits: " << metrics["cacheHits"] << "\n"
                     << "- cache hit rate: " << std::fixed << std::setprecision(3)
                     << metrics["cacheHitRate"].get<double>() * 100.0 << "%\n\n"
                     << "- prompt KV eligible frames: " << metrics["promptKvEligibleFrames"] << "\n"
                      << "- prompt KV hits: " << metrics["promptKvHits"] << "\n"
                      << "- prompt KV hit rate: " << metrics["promptKvHitRate"].get<double>() * 100.0 << "%\n\n"
                      << "## Result source distribution\n\n"
                      << "| source | frames | inference average ms | inference P99 ms |\n"
                      << "|---|---:|---:|---:|\n";
    for (const auto& [source, count] : metrics["resultSourceCounts"].items()) {
        const auto& inference = metrics["resultSources"][source]["inference"];
        metrics_markdown << "| " << source
                         << " | " << count.get<std::size_t>()
                         << " | " << inference["averageMs"].get<double>()
                         << " | " << inference["p99Ms"].get<double>() << " |\n";
    }
    metrics_markdown << "\n"
                      << "## End-to-end by path\n\n"
                     << "| path | frames | average ms | P50 ms | P95 ms | P99 ms | max ms |\n"
                     << "|---|---:|---:|---:|---:|---:|---:|\n";
    for (const auto* group : {"all", "cache", "fresh", "promptKvHit", "promptKvMiss", "spooled", "hot"}) {
        const auto& group_metrics = metrics[group];
        const auto& summary = group_metrics["endToEnd"];
        metrics_markdown << "| " << group
                         << " | " << group_metrics["frames"].get<std::size_t>()
                         << " | " << summary["averageMs"].get<double>()
                         << " | " << summary["p50Ms"].get<double>()
                         << " | " << summary["p95Ms"].get<double>()
                         << " | " << summary["p99Ms"].get<double>()
                         << " | " << summary["maxMs"].get<double>() << " |\n";
    }
    metrics_markdown << "\n## Processing end-to-end excluding stream wait\n\n"
                     << "| path | frames | average ms | P50 ms | P95 ms | P99 ms | max ms |\n"
                     << "|---|---:|---:|---:|---:|---:|---:|\n";
    for (const auto* group : {"all", "cache", "fresh", "promptKvHit", "promptKvMiss", "spooled", "hot"}) {
        const auto& group_metrics = metrics[group];
        const auto& summary = group_metrics["processingEndToEnd"];
        metrics_markdown << "| " << group
                         << " | " << group_metrics["frames"].get<std::size_t>()
                         << " | " << summary["averageMs"].get<double>()
                         << " | " << summary["p50Ms"].get<double>()
                         << " | " << summary["p95Ms"].get<double>()
                         << " | " << summary["p99Ms"].get<double>()
                         << " | " << summary["maxMs"].get<double>() << " |\n";
    }
    metrics_markdown << "\n## All-frame stages\n\n"
                     << "| stage | samples | average ms | P50 ms | P95 ms | P99 ms | max ms |\n"
                     << "|---|---:|---:|---:|---:|---:|---:|\n";
    for (const auto* stage : {"publishToReceive", "receiveToAdmit", "queueWait", "spoolResidence", "streamWait", "replayWait", "inference", "endToEnd", "processingEndToEnd"}) {
        const auto& summary = metrics["all"][stage];
        metrics_markdown << "| " << stage
                         << " | " << summary["samples"].get<std::size_t>()
                         << " | " << summary["averageMs"].get<double>()
                         << " | " << summary["p50Ms"].get<double>()
                         << " | " << summary["p95Ms"].get<double>()
                         << " | " << summary["p99Ms"].get<double>()
                         << " | " << summary["maxMs"].get<double>() << " |\n";
    }
    metrics_markdown << "\n## Model timing by cache path\n\n"
                     << "| path | frames | prompt eval avg ms | prompt eval P99 ms | eval avg ms | eval P99 ms |\n"
                     << "|---|---:|---:|---:|---:|---:|\n";
    for (const auto* group : {"fresh", "promptKvHit", "promptKvMiss"}) {
        const auto& group_metrics = metrics[group];
        const auto& prompt_eval = group_metrics["promptEval"];
        const auto& eval = group_metrics["eval"];
        metrics_markdown << "| " << group
                         << " | " << group_metrics["frames"].get<std::size_t>()
                         << " | " << prompt_eval["averageMs"].get<double>()
                         << " | " << prompt_eval["p99Ms"].get<double>()
                         << " | " << eval["averageMs"].get<double>()
                         << " | " << eval["p99Ms"].get<double>() << " |\n";
    }
    lease.Shutdown();
    std::cout << "report=" << (report_root / "REPORT.md") << '\n';
    return valid ? 0 : 1;
}
