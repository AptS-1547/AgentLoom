#include "inference_frame_backlog.h"

#include "memory_pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kSessionCount = 8;
constexpr std::size_t kFrameCount = 8192;
constexpr std::size_t kPayloadBytes = 4096;

media::inference::InferenceFrameMetadata Metadata(std::size_t index) {
    media::inference::InferenceFrameMetadata metadata;
    metadata.session_id = "bench-session-" + std::to_string(index % kSessionCount);
    metadata.frame_id = static_cast<std::uint64_t>(index + 1);
    metadata.timestamp_us = static_cast<std::int64_t>((index + 1) * 1'000);
    metadata.width = 64;
    metadata.height = 64;
    metadata.format = media::inference::InferenceFrameFormat::Rgb;
    return metadata;
}

double Milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

int main() {
    const auto worker_count = std::max<std::size_t>(2, std::thread::hardware_concurrency());
    core::BucketMemoryPool memory_pool(kFrameCount * kPayloadBytes, 128);
    media::inference::SegmentedInferenceFrameBacklog backlog({
        .max_sessions = kSessionCount,
        .segments_per_session = 16,
        .slots_per_segment = 64,
        .default_wait_timeout = std::chrono::seconds(5),
    });
    media::inference::SessionInferenceFrameResultTable results({
        .max_sessions = kSessionCount,
        .max_results_per_session = kFrameCount / kSessionCount,
    });
    std::vector<std::byte> source(kPayloadBytes, std::byte{0x42});

    const auto copy_submit_started = Clock::now();
    for (std::size_t index = 0; index < kFrameCount; ++index) {
        auto frame = media::inference::CopyInferenceFrame(
            memory_pool,
            Metadata(index),
            source);
        if (!frame.ok()) {
            std::cerr << "copy failed: " << frame.status().message() << '\n';
            return 1;
        }
        auto status = backlog.Submit(std::move(frame).value());
        if (!status.ok()) {
            std::cerr << "submit failed: " << status.message() << '\n';
            return 1;
        }
    }
    const auto copy_submit_finished = Clock::now();

    std::atomic<std::size_t> next_frame{0};
    std::atomic<std::size_t> failures{0};
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    const auto processing_started = Clock::now();
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&] {
            for (;;) {
                const auto index = next_frame.fetch_add(1, std::memory_order_relaxed);
                if (index >= kFrameCount) {
                    return;
                }
                auto frame = backlog.WaitTake(std::chrono::seconds(5));
                if (!frame.ok()) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                media::inference::InferenceFrameResultRecord record;
                record.frame = frame.value().metadata();
                media::VisionInferenceResult result;
                result.scene_hint = "benchmark";
                result.confidence = 1.0;
                record.result = std::move(result);
                auto status = results.Publish(std::move(record));
                if (!status.ok()) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    const auto processing_finished = Clock::now();

    const auto finalize_started = Clock::now();
    std::size_t finalized_count = 0;
    for (std::size_t session = 0; session < kSessionCount; ++session) {
        auto finalized = results.FinalizeSession("bench-session-" + std::to_string(session));
        if (!finalized.ok()) {
            std::cerr << "finalize failed: " << finalized.status().message() << '\n';
            return 1;
        }
        finalized_count += finalized.value().size();
    }
    const auto finalize_finished = Clock::now();

    if (failures.load(std::memory_order_relaxed) != 0 || finalized_count != kFrameCount) {
        std::cerr << "benchmark consistency failure: failures="
                  << failures.load(std::memory_order_relaxed)
                  << " finalized=" << finalized_count << '\n';
        return 1;
    }

    const auto copy_submit_ms = Milliseconds(copy_submit_started, copy_submit_finished);
    const auto processing_ms = Milliseconds(processing_started, processing_finished);
    const auto finalize_ms = Milliseconds(finalize_started, finalize_finished);
    const auto payload_mib = static_cast<double>(kFrameCount * kPayloadBytes) / (1024.0 * 1024.0);
    std::cout << "{\n"
              << "  \"sessions\": " << kSessionCount << ",\n"
              << "  \"frames\": " << kFrameCount << ",\n"
              << "  \"payload_bytes\": " << kPayloadBytes << ",\n"
              << "  \"workers\": " << worker_count << ",\n"
              << "  \"copy_submit_ms\": " << copy_submit_ms << ",\n"
              << "  \"copy_mib_per_second\": " << payload_mib / (copy_submit_ms / 1000.0) << ",\n"
              << "  \"take_publish_ms\": " << processing_ms << ",\n"
              << "  \"frames_per_second\": " << kFrameCount / (processing_ms / 1000.0) << ",\n"
              << "  \"finalize_ms\": " << finalize_ms << "\n"
              << "}\n";
    return 0;
}
