#include "frame_encoding.h"
#include "inference_frame_gateway_producer.h"
#include "inference_frame_ipc_lifecycle.h"

#include "memory_pool.h"

#ifdef _WIN32
#include <io.h>
#endif

#include <boost/asio/io_context.hpp>
#include <boost/process/v2.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifndef MEDIA_IPC_E2E_PEER_PATH
#error MEDIA_IPC_E2E_PEER_PATH is required
#endif

namespace {

namespace bp = boost::process::v2;
using namespace std::chrono_literals;

std::string UniqueName(std::string_view suffix) {
    static std::atomic<std::uint64_t> sequence{0};
    return "agent_frame_process_e2e_" + std::string(suffix) + "_" +
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
           std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}

class ProcessE2eCleanup {
public:
    explicit ProcessE2eCleanup(std::string name)
        : name_(std::move(name)),
          report_path_(std::filesystem::current_path() / (name_ + ".txt")) {
        ipc::media::SharedMemoryInferenceFrameChannel::Remove(name_);
        std::error_code error;
        std::filesystem::remove(report_path_, error);
    }

    ~ProcessE2eCleanup() {
        ipc::media::SharedMemoryInferenceFrameChannel::Remove(name_);
        std::error_code error;
        std::filesystem::remove(report_path_, error);
    }

    const std::string& name() const noexcept {
        return name_;
    }

    const std::filesystem::path& report_path() const noexcept {
        return report_path_;
    }

private:
    std::string name_;
    std::filesystem::path report_path_;
};

media::VideoFrameView MakeRgbFrame(
    std::vector<std::byte>& payload,
    std::uint64_t frame_id) {
    constexpr std::uint32_t width = 160;
    constexpr std::uint32_t height = 90;
    payload.resize(static_cast<std::size_t>(width) * height * 3);
    for (std::size_t index = 0; index < payload.size(); ++index) {
        payload[index] = static_cast<std::byte>((index * 17 + frame_id) & 0xFF);
    }
    media::VideoFrameView frame;
    frame.session_id = "session-process-e2e";
    frame.frame_id = frame_id;
    frame.captured_at = std::chrono::steady_clock::now();
    frame.width = width;
    frame.height = height;
    frame.format = media::VideoPixelFormat::Rgb;
    frame.bytes = {reinterpret_cast<const char*>(payload.data()), payload.size()};
    return frame;
}

core::Result<media::EncodedVideoFrame> MakeMinimalJpeg(
    core::RawMemoryPool& pool,
    std::uint64_t frame_id) {
    constexpr std::byte bytes[] = {
        std::byte{0xFF}, std::byte{0xD8}, std::byte{0xFF}, std::byte{0xD9},
    };
    auto block = core::SharedMemoryBlock::allocate(pool, sizeof(bytes));
    if (!block.ok()) {
        return block.status();
    }
    std::copy(std::begin(bytes), std::end(bytes), block.value().data());
    media::EncodedVideoFrameMetadata metadata;
    metadata.session_id = "session-crash";
    metadata.trace_id = "trace-crash";
    metadata.frame_id = frame_id;
    metadata.timestamp_us = static_cast<std::int64_t>(frame_id * 1000);
    metadata.width = 1;
    metadata.height = 1;
    metadata.format = media::EncodedVideoFrameFormat::Jpeg;
    return media::EncodedVideoFrame(std::move(metadata), std::move(block).value(), sizeof(bytes));
}

bool WaitForExit(bp::process& child, std::chrono::seconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (child.running() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(5ms);
    }
    if (child.running()) {
        child.terminate();
        child.wait();
        return false;
    }
    child.wait();
    return true;
}

bool WaitForReport(
    const std::filesystem::path& path,
    std::string_view expected,
    std::chrono::seconds timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        std::ifstream report(path, std::ios::binary);
        std::string value;
        if (report >> value && value == expected) {
            return true;
        }
        std::this_thread::sleep_for(5ms);
    }
    return false;
}

TEST(InferenceFrameIpcProcessE2eTest, EncodesPublishesAndReceivesAcrossProcesses) {
    constexpr std::size_t frame_count = 12;
    ProcessE2eCleanup cleanup(UniqueName("flow"));
    auto sink = ipc::media::RecoverableInferenceFrameIpcSink::Create({
        .name = cleanup.name(),
        .slot_count = 4,
        .payload_capacity = 1024 * 1024,
    });
    ASSERT_TRUE(sink.ok()) << sink.status().message();

    boost::asio::io_context process_context;
    bp::process consumer(
        process_context,
        MEDIA_IPC_E2E_PEER_PATH,
        std::vector<std::string>{
            "consume",
            cleanup.name(),
            std::to_string(frame_count),
            cleanup.report_path().string(),
        });

    core::BucketMemoryPool pool(0, 32);
    auto encoder = media::GStreamerVideoFrameEncoder::Create(pool, {
        .preference = media::VideoImageEncoderPreference::Software,
        .jpeg_quality = 85,
    });
    ASSERT_TRUE(encoder.ok()) << encoder.status().message();
    media::InferenceFrameGatewayProducer producer(*sink.value());
    std::vector<std::byte> rgb;

    for (std::uint64_t frame_id = 1; frame_id <= frame_count; ++frame_id) {
        auto encoded = encoder.value()->Encode(MakeRgbFrame(rgb, frame_id), 0.7, "trace-process-e2e");
        ASSERT_TRUE(encoded.ok()) << encoded.status().message();
        for (;;) {
            const auto status = producer.Publish(encoded.value());
            if (status.ok()) {
                break;
            }
            ASSERT_EQ(status.code(), core::ErrorCode::ResourceExhausted) << status.message();
            std::this_thread::sleep_for(1ms);
        }
    }

    ASSERT_TRUE(WaitForExit(consumer, 15s));
    ASSERT_EQ(consumer.exit_code(), 0);
    std::ifstream report(cleanup.report_path(), std::ios::binary);
    ASSERT_TRUE(report.good());
    std::size_t consumed = 0;
    std::uint64_t first = 0;
    std::uint64_t last = 0;
    std::size_t bytes = 0;
    report >> consumed >> first >> last >> bytes;
    EXPECT_EQ(consumed, frame_count);
    EXPECT_EQ(first, 1u);
    EXPECT_EQ(last, frame_count);
    EXPECT_GT(bytes, 0u);
    EXPECT_EQ(producer.Snapshot().published_frames, frame_count);
}

TEST(InferenceFrameIpcProcessE2eTest, RecreateRecoversSlotAbandonedByCrashedPeer) {
    ProcessE2eCleanup cleanup(UniqueName("crash"));
    auto sink = ipc::media::RecoverableInferenceFrameIpcSink::Create({
        .name = cleanup.name(),
        .slot_count = 2,
        .payload_capacity = 1024,
    });
    ASSERT_TRUE(sink.ok()) << sink.status().message();
    media::InferenceFrameGatewayProducer producer(*sink.value());
    core::BucketMemoryPool pool;

    auto first = MakeMinimalJpeg(pool, 1);
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(producer.Publish(std::move(first).value()).ok());
    const auto old_epoch = sink.value()->Snapshot().epoch;

    boost::asio::io_context crash_context;
    bp::process crashing_consumer(
        crash_context,
        MEDIA_IPC_E2E_PEER_PATH,
        std::vector<std::string>{"claim-crash", cleanup.name()});
    ASSERT_TRUE(WaitForExit(crashing_consumer, 10s));
    ASSERT_EQ(crashing_consumer.exit_code(), 23);

    auto second = MakeMinimalJpeg(pool, 2);
    auto blocked = MakeMinimalJpeg(pool, 3);
    ASSERT_TRUE(second.ok() && blocked.ok());
    ASSERT_TRUE(producer.Publish(std::move(second).value()).ok());
    EXPECT_EQ(producer.Publish(std::move(blocked).value()).code(), core::ErrorCode::ResourceExhausted);

    ASSERT_TRUE(sink.value()->Recreate().ok());
    const auto lifecycle = sink.value()->LifecycleSnapshot();
    EXPECT_EQ(lifecycle.recoveries, 1u);
    EXPECT_TRUE(lifecycle.connected);
    EXPECT_NE(lifecycle.channel.epoch, old_epoch);

    auto recovered = MakeMinimalJpeg(pool, 4);
    ASSERT_TRUE(recovered.ok());
    ASSERT_TRUE(producer.Publish(std::move(recovered).value()).ok());
    boost::asio::io_context replacement_context;
    bp::process replacement_consumer(
        replacement_context,
        MEDIA_IPC_E2E_PEER_PATH,
        std::vector<std::string>{
            "consume",
            cleanup.name(),
            "1",
            cleanup.report_path().string(),
        });
    ASSERT_TRUE(WaitForExit(replacement_consumer, 10s));
    EXPECT_EQ(replacement_consumer.exit_code(), 0);
}

TEST(InferenceFrameIpcProcessE2eTest, ControlPlaneFenceStopsSurvivingStaleReader) {
    ProcessE2eCleanup cleanup(UniqueName("fence"));
    auto sink = ipc::media::RecoverableInferenceFrameIpcSink::Create({
        .name = cleanup.name(),
        .slot_count = 2,
        .payload_capacity = 1024,
    });
    ASSERT_TRUE(sink.ok()) << sink.status().message();
    const auto old_grant = sink.value()->CurrentGrant();

    boost::asio::io_context stale_context;
    bp::process stale_reader(
        stale_context,
        MEDIA_IPC_E2E_PEER_PATH,
        std::vector<std::string>{
            "wait-fence",
            cleanup.name(),
            std::to_string(old_grant.epoch),
            cleanup.report_path().string(),
        });
    ASSERT_TRUE(WaitForReport(cleanup.report_path(), "ready", 10s));

    ASSERT_TRUE(sink.value()->Fence().ok());
    ASSERT_TRUE(WaitForExit(stale_reader, 10s));
    ASSERT_EQ(stale_reader.exit_code(), 0);
    ASSERT_TRUE(WaitForReport(cleanup.report_path(), "fenced", 1s));

    ASSERT_TRUE(sink.value()->Recreate().ok());
    const auto new_grant = sink.value()->CurrentGrant();
    ASSERT_NE(new_grant.epoch, old_grant.epoch);
    auto old_epoch_reader = ipc::media::SharedMemoryInferenceFrameChannel::Open({
        .name = cleanup.name(),
        .expected_epoch = old_grant.epoch,
    });
    ASSERT_FALSE(old_epoch_reader.ok());
    EXPECT_EQ(old_epoch_reader.status().code(), core::ErrorCode::FailedPrecondition);
    auto current_reader = ipc::media::SharedMemoryInferenceFrameChannel::Open({
        .name = cleanup.name(),
        .expected_epoch = new_grant.epoch,
    });
    EXPECT_TRUE(current_reader.ok()) << current_reader.status().message();
}

} // namespace
