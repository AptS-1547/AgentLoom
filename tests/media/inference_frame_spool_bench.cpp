#include "inference_frame_spool.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double MegabytesPerSecond(std::size_t bytes, Clock::duration elapsed) {
    const auto seconds = std::chrono::duration<double>(elapsed).count();
    return seconds > 0.0
        ? static_cast<double>(bytes) / (1024.0 * 1024.0) / seconds
        : 0.0;
}

int Fail(std::string_view stage, const core::Status& status) {
    std::cerr << stage << " failed: " << status.message() << '\n';
    return 1;
}

} // namespace

int main(int argc, char** argv) {
    const std::size_t record_count = argc > 1
        ? static_cast<std::size_t>(std::stoull(argv[1]))
        : 1024;
    const std::size_t payload_bytes = argc > 2
        ? static_cast<std::size_t>(std::stoull(argv[2]))
        : 128 * 1024;
    const bool flush_on_append = argc > 3 && std::string_view(argv[3]) == "--flush";
    if (record_count == 0 || payload_bytes == 0) {
        std::cerr << "record_count and payload_bytes must be positive\n";
        return 2;
    }

    const auto execution_id = "spool-bench-" +
        std::to_string(Clock::now().time_since_epoch().count());
    const auto root = std::filesystem::current_path() / "build" / "media-spool-bench";
    const auto total_payload_bytes = record_count * payload_bytes;
    auto spool = media::inference::MappedInferenceFrameSpool::Create({
        .root_directory = root,
        .execution_id = execution_id,
        .segment_bytes = 16 * 1024 * 1024,
        .max_spool_bytes = total_payload_bytes + 32 * 1024 * 1024,
        .flush_on_append = flush_on_append,
        .remove_on_destroy = true,
    });
    if (!spool.ok()) {
        return Fail("create", spool.status());
    }

    std::vector<std::byte> payload(payload_bytes, std::byte{0x5A});
    const auto append_started = Clock::now();
    for (std::size_t index = 0; index < record_count; ++index) {
        media::inference::SpoolFrameMetadata metadata;
        metadata.execution_id = execution_id;
        metadata.selected_sequence = index + 1;
        metadata.frame.session_id = "bench-session";
        metadata.frame.trace_id = "bench-trace";
        metadata.frame.frame_id = index + 1;
        metadata.frame.timestamp_us = static_cast<std::int64_t>(index * 33'333);
        metadata.frame.width = 1280;
        metadata.frame.height = 720;
        metadata.frame.format = media::inference::InferenceFrameFormat::Jpeg;
        metadata.frame.saliency = 0.8;
        auto append = spool.value()->Append(metadata, payload);
        if (!append.ok()) {
            return Fail("append", append.status());
        }
    }
    const auto append_finished = Clock::now();
    auto seal = spool.value()->Seal();
    if (!seal.ok()) {
        return Fail("seal", seal);
    }

    std::size_t replayed_records = 0;
    std::uint64_t checksum = 0;
    const auto replay_started = Clock::now();
    for (;;) {
        auto replay = spool.value()->ReplayNext();
        if (!replay.ok()) {
            return Fail("replay", replay.status());
        }
        if (!replay.value().has_value()) {
            break;
        }
        const auto& lease = replay.value().value();
        if (!lease.valid() || lease.bytes().size() != payload_bytes) {
            std::cerr << "replay returned an invalid record\n";
            return 1;
        }
        checksum += lease.metadata().selected_sequence;
        checksum += static_cast<std::uint8_t>(lease.bytes().front());
        ++replayed_records;
    }
    const auto replay_finished = Clock::now();

    const auto snapshot = spool.value()->Snapshot();
    std::cout << "records=" << record_count
              << " payload_bytes=" << payload_bytes
              << " flush_on_append=" << (flush_on_append ? "true" : "false")
              << " segments=" << snapshot.segment_count
              << " allocated_bytes=" << snapshot.allocated_bytes << '\n';
    std::cout << "append_mib_s="
              << MegabytesPerSecond(total_payload_bytes, append_finished - append_started)
              << " replay_mib_s="
              << MegabytesPerSecond(total_payload_bytes, replay_finished - replay_started)
              << " replayed_records=" << replayed_records
              << " checksum=" << checksum << '\n';
    return replayed_records == record_count ? 0 : 1;
}
