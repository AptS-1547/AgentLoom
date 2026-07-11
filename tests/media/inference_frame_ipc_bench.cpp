#include "inference_frame_ipc_receiver.h"
#include "inference_frame_shared_memory.h"

#include "memory_pool.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main() {
    constexpr std::size_t kFrameCount = 8192;
    constexpr std::size_t kPayloadSize = 4096;
    const std::string channel_name = "agent_frame_ipc_bench_" + std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    ipc::media::SharedMemoryInferenceFrameChannel::Remove(channel_name);

    auto producer = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = channel_name,
        .slot_count = 256,
        .payload_capacity = kPayloadSize,
    });
    auto source = ipc::media::SharedMemoryInferenceFrameChannel::Open({.name = channel_name});
    if (!producer.ok() || !source.ok()) {
        std::cerr << "failed to create IPC benchmark channel\n";
        return 1;
    }

    core::BucketMemoryPool pool(0, 256);
    media::inference::SegmentedInferenceFrameBacklog backlog({
        .max_sessions = 8,
        .segments_per_session = 8,
        .slots_per_segment = 64,
    });
    media::inference::InferenceFrameIpcReceiver receiver(*source.value(), pool, backlog);
    std::vector<std::byte> payload(kPayloadSize, std::byte{0x6A});

    const auto started = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < kFrameCount; ++index) {
        const std::string session_id = "session-" + std::to_string(index % 8);
        ipc::media::SharedFramePublishRequest request{
            .session_id = session_id,
            .trace_id = "bench",
            .frame_id = index + 1,
            .timestamp_us = static_cast<std::int64_t>(index * 1'000),
            .width = 640,
            .height = 360,
            .format = static_cast<std::uint32_t>(media::inference::InferenceFrameFormat::Jpeg),
            .payload = payload,
        };
        if (!producer.value()->Publish(request).ok() || !receiver.PollOnce().ok()) {
            std::cerr << "IPC benchmark failed at frame " << index << '\n';
            return 1;
        }
        auto frame = backlog.TryTake();
        if (!frame.ok()) {
            std::cerr << "IPC benchmark backlog failed at frame " << index << '\n';
            return 1;
        }
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    const auto seconds = std::chrono::duration<double>(elapsed).count();
    const auto mib = static_cast<double>(kFrameCount * kPayloadSize) / (1024.0 * 1024.0);

    std::cout << "frames: " << kFrameCount << '\n'
              << "payload: " << kPayloadSize << " bytes\n"
              << "publish + claim + private copy + submit + take: " << seconds * 1000.0 << " ms\n"
              << "throughput: " << static_cast<double>(kFrameCount) / seconds << " frames/s\n"
              << "payload throughput: " << mib / seconds << " MiB/s\n";

    source.value().reset();
    producer.value().reset();
    ipc::media::SharedMemoryInferenceFrameChannel::Remove(channel_name);
    return 0;
}
