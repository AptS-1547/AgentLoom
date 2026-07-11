#include "inference_frame_backlog.h"
#include "inference_frame_ipc_receiver.h"
#include "inference_frame_shared_memory.h"

#include "memory_pool.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>

namespace {

using namespace std::chrono_literals;

core::Result<std::unique_ptr<ipc::media::SharedMemoryInferenceFrameChannel>> OpenWithRetry(
    const std::string& name) {
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    for (;;) {
        auto channel = ipc::media::SharedMemoryInferenceFrameChannel::Open({.name = name});
        if (channel.ok()) {
            return channel;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return channel.status();
        }
        std::this_thread::sleep_for(5ms);
    }
}

int Consume(const std::string& name, std::size_t expected_count, const std::string& report_path) {
    auto channel = OpenWithRetry(name);
    if (!channel.ok()) {
        return 10;
    }

    core::BucketMemoryPool pool(0, 64);
    media::inference::SegmentedInferenceFrameBacklog backlog({
        .max_sessions = 8,
        .segments_per_session = 4,
        .slots_per_segment = 16,
    });
    media::inference::InferenceFrameIpcReceiver receiver(*channel.value(), pool, backlog);
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    std::size_t count = 0;
    std::uint64_t first_frame = 0;
    std::uint64_t last_frame = 0;
    std::size_t total_bytes = 0;

    while (count < expected_count && std::chrono::steady_clock::now() < deadline) {
        const auto poll_status = receiver.PollOnce();
        if (!poll_status.ok()) {
            if (poll_status.code() == core::ErrorCode::NotFound) {
                std::this_thread::sleep_for(1ms);
                continue;
            }
            return 11;
        }

        auto frame = backlog.TryTake();
        if (!frame.ok()) {
            return 12;
        }
        if (frame.value().metadata().format != media::inference::InferenceFrameFormat::Jpeg ||
            frame.value().bytes().size() < 4 ||
            frame.value().bytes()[0] != std::byte{0xFF} ||
            frame.value().bytes()[1] != std::byte{0xD8}) {
            return 13;
        }
        if (count == 0) {
            first_frame = frame.value().metadata().frame_id;
        }
        last_frame = frame.value().metadata().frame_id;
        total_bytes += frame.value().bytes().size();
        ++count;
    }

    if (count != expected_count) {
        return 14;
    }
    std::ofstream report(report_path, std::ios::binary | std::ios::trunc);
    if (!report) {
        return 15;
    }
    report << count << ' ' << first_frame << ' ' << last_frame << ' ' << total_bytes << '\n';
    return report.good() ? 0 : 16;
}

[[noreturn]] void ClaimAndCrash(const std::string& name) {
    auto channel = OpenWithRetry(name);
    if (!channel.ok()) {
        std::_Exit(20);
    }
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    for (;;) {
        auto claimed = channel.value()->TryClaim();
        if (claimed.ok()) {
            std::_Exit(23);
        }
        if (claimed.status().code() != core::ErrorCode::NotFound ||
            std::chrono::steady_clock::now() >= deadline) {
            std::_Exit(21);
        }
        std::this_thread::sleep_for(1ms);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        return 2;
    }
    const std::string mode = argv[1];
    const std::string name = argv[2];
    if (mode == "consume" && argc == 5) {
        return Consume(name, static_cast<std::size_t>(std::stoull(argv[3])), argv[4]);
    }
    if (mode == "claim-crash") {
        ClaimAndCrash(name);
    }
    return 3;
}
