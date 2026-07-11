#include "frame_encoding.h"
#include "inference_frame_gateway_producer.h"
#include "inference_frame_shared_memory.h"
#include "webrtc_media_pipeline.h"

#include "memory_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

std::string UniqueChannelName(std::string_view suffix) {
    static std::atomic<std::uint64_t> sequence{0};
    return "agent_frame_encoding_" + std::string(suffix) + "_" +
           std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
           std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}

class ChannelCleanup {
public:
    explicit ChannelCleanup(std::string name)
        : name_(std::move(name)) {
        ipc::media::SharedMemoryInferenceFrameChannel::Remove(name_);
    }

    ~ChannelCleanup() {
        ipc::media::SharedMemoryInferenceFrameChannel::Remove(name_);
    }

    const std::string& name() const noexcept {
        return name_;
    }

private:
    std::string name_;
};

media::VideoFrameView MakeRgbFrame(
    std::vector<std::byte>& payload,
    std::uint32_t width,
    std::uint32_t height,
    std::uint64_t frame_id) {
    payload.resize(static_cast<std::size_t>(width) * height * 3);
    for (std::uint32_t row = 0; row < height; ++row) {
        for (std::uint32_t column = 0; column < width; ++column) {
            const auto offset = (static_cast<std::size_t>(row) * width + column) * 3;
            payload[offset] = static_cast<std::byte>((column * 255) / std::max(1u, width - 1));
            payload[offset + 1] = static_cast<std::byte>((row * 255) / std::max(1u, height - 1));
            payload[offset + 2] = std::byte{0x60};
        }
    }
    media::VideoFrameView frame;
    frame.session_id = "session-encode";
    frame.frame_id = frame_id;
    frame.captured_at = std::chrono::steady_clock::now();
    frame.width = width;
    frame.height = height;
    frame.format = media::VideoPixelFormat::Rgb;
    frame.bytes = {
        reinterpret_cast<const char*>(payload.data()),
        payload.size(),
    };
    return frame;
}

TEST(GStreamerVideoFrameEncoderTest, AutoEncodesRgbFrameAsJpeg) {
    core::BucketMemoryPool pool;
    auto encoder = media::GStreamerVideoFrameEncoder::Create(pool, {
        .format = media::EncodedVideoFrameFormat::Jpeg,
        .preference = media::VideoImageEncoderPreference::Auto,
        .jpeg_quality = 88,
    });
    ASSERT_TRUE(encoder.ok()) << encoder.status().message();
    EXPECT_FALSE(encoder.value()->Selection().factory_name.empty());

    std::vector<std::byte> payload;
    auto frame = MakeRgbFrame(payload, 160, 90, 7);
    auto encoded = encoder.value()->Encode(frame, 0.82, "trace-encode");
    ASSERT_TRUE(encoded.ok()) << encoded.status().message();
    ASSERT_TRUE(encoded.value().valid());
    ASSERT_GT(encoded.value().bytes().size(), 4u);
    EXPECT_EQ(encoded.value().bytes()[0], std::byte{0xFF});
    EXPECT_EQ(encoded.value().bytes()[1], std::byte{0xD8});
    EXPECT_EQ(encoded.value().bytes()[encoded.value().bytes().size() - 2], std::byte{0xFF});
    EXPECT_EQ(encoded.value().bytes().back(), std::byte{0xD9});
    EXPECT_EQ(encoded.value().metadata().session_id, "session-encode");
    EXPECT_EQ(encoded.value().metadata().trace_id, "trace-encode");
    EXPECT_EQ(encoded.value().metadata().frame_id, 7u);
    EXPECT_EQ(encoded.value().metadata().width, 160u);
    EXPECT_EQ(encoded.value().metadata().height, 90u);
    EXPECT_DOUBLE_EQ(encoded.value().metadata().saliency, 0.82);
}

TEST(GStreamerVideoFrameEncoderTest, ReNegotiatesForChangedDimensions) {
    core::BucketMemoryPool pool;
    auto encoder = media::GStreamerVideoFrameEncoder::Create(pool, {
        .preference = media::VideoImageEncoderPreference::Software,
    });
    ASSERT_TRUE(encoder.ok()) << encoder.status().message();

    std::vector<std::byte> first_payload;
    std::vector<std::byte> second_payload;
    auto first = encoder.value()->Encode(MakeRgbFrame(first_payload, 128, 72, 1), 0.1);
    auto second = encoder.value()->Encode(MakeRgbFrame(second_payload, 192, 108, 2), 0.2);
    ASSERT_TRUE(first.ok()) << first.status().message();
    ASSERT_TRUE(second.ok()) << second.status().message();
    EXPECT_NE(first.value().bytes().size(), 0u);
    EXPECT_NE(second.value().bytes().size(), 0u);
}

TEST(GStreamerVideoFrameEncoderTest, EncodesRgbRowsWithPadding) {
    core::BucketMemoryPool pool;
    auto encoder = media::GStreamerVideoFrameEncoder::Create(pool, {
        .preference = media::VideoImageEncoderPreference::Software,
    });
    ASSERT_TRUE(encoder.ok());

    constexpr std::uint32_t width = 5;
    constexpr std::uint32_t height = 4;
    constexpr std::size_t packed_stride = width * 3;
    constexpr std::size_t padded_stride = 20;
    std::vector<std::byte> payload(padded_stride * height, std::byte{0x7F});
    for (std::size_t row = 0; row < height; ++row) {
        for (std::size_t column = 0; column < packed_stride; ++column) {
            payload[row * padded_stride + column] = static_cast<std::byte>((row * 31 + column) & 0xFF);
        }
    }
    media::VideoFrameView frame;
    frame.session_id = "session-stride";
    frame.frame_id = 1;
    frame.width = width;
    frame.height = height;
    frame.row_stride_bytes = padded_stride;
    frame.format = media::VideoPixelFormat::Rgb;
    frame.bytes = {reinterpret_cast<const char*>(payload.data()), payload.size()};

    auto encoded = encoder.value()->Encode(frame, 0.4);
    ASSERT_TRUE(encoded.ok()) << encoded.status().message();
    EXPECT_TRUE(encoded.value().valid());
}

TEST(GStreamerVideoFrameEncoderTest, ForcedNvidiaRequiresJpegFactory) {
    core::BucketMemoryPool pool;
    auto encoder = media::GStreamerVideoFrameEncoder::Create(pool, {
        .preference = media::VideoImageEncoderPreference::Nvidia,
    });

    if (encoder.ok()) {
        EXPECT_TRUE(encoder.value()->Selection().hardware);
        EXPECT_EQ(encoder.value()->Selection().backend, media::VideoImageEncoderBackend::Nvidia);
        EXPECT_EQ(encoder.value()->Selection().factory_name, "nvjpegenc");
    } else {
        EXPECT_EQ(encoder.status().code(), core::ErrorCode::Unavailable);
    }
}

TEST(GStreamerVideoFrameEncoderTest, RejectsIncompleteRgbPayload) {
    core::BucketMemoryPool pool;
    auto encoder = media::GStreamerVideoFrameEncoder::Create(pool, {
        .preference = media::VideoImageEncoderPreference::Software,
    });
    ASSERT_TRUE(encoder.ok());

    std::vector<std::byte> payload(16);
    media::VideoFrameView frame;
    frame.session_id = "session-invalid";
    frame.frame_id = 1;
    frame.width = 32;
    frame.height = 32;
    frame.format = media::VideoPixelFormat::Rgb;
    frame.bytes = {reinterpret_cast<const char*>(payload.data()), payload.size()};
    auto encoded = encoder.value()->Encode(frame, 0.0);

    ASSERT_FALSE(encoded.ok());
    EXPECT_EQ(encoded.status().code(), core::ErrorCode::InvalidArgument);
}

TEST(InferenceFrameGatewayProducerTest, PublishesEncodedFrameIntoSharedChannel) {
    ChannelCleanup cleanup(UniqueChannelName("producer"));
    auto channel = ipc::media::SharedMemoryInferenceFrameChannel::Create({
        .name = cleanup.name(),
        .slot_count = 4,
        .payload_capacity = 1024 * 1024,
    });
    ASSERT_TRUE(channel.ok()) << channel.status().message();
    auto consumer = ipc::media::SharedMemoryInferenceFrameChannel::Open({.name = cleanup.name()});
    ASSERT_TRUE(consumer.ok()) << consumer.status().message();

    core::BucketMemoryPool pool;
    auto encoder = media::GStreamerVideoFrameEncoder::Create(pool, {
        .preference = media::VideoImageEncoderPreference::Software,
    });
    ASSERT_TRUE(encoder.ok()) << encoder.status().message();
    std::vector<std::byte> payload;
    auto encoded = encoder.value()->Encode(MakeRgbFrame(payload, 160, 90, 19), 0.91, "trace-ipc-encode");
    ASSERT_TRUE(encoded.ok()) << encoded.status().message();

    media::InferenceFrameGatewayProducer producer(*channel.value());
    ASSERT_TRUE(producer.Publish(std::move(encoded).value()).ok());
    auto claimed = consumer.value()->TryClaim();
    ASSERT_TRUE(claimed.ok()) << claimed.status().message();
    EXPECT_EQ(claimed.value().metadata().session_id, "session-encode");
    EXPECT_EQ(claimed.value().metadata().trace_id, "trace-ipc-encode");
    EXPECT_EQ(claimed.value().metadata().frame_id, 19u);
    EXPECT_EQ(
        claimed.value().metadata().format,
        static_cast<std::uint32_t>(ipc::media::SharedFrameFormat::Jpeg));
    ASSERT_GT(claimed.value().payload().size(), 4u);
    EXPECT_EQ(claimed.value().payload()[0], std::byte{0xFF});
    EXPECT_EQ(claimed.value().payload()[1], std::byte{0xD8});
    EXPECT_EQ(producer.Snapshot().published_frames, 1u);
}

TEST(WebRtcMediaPipelineTest, RejectsIncompleteEncodingPair) {
    class NullEncoder final : public media::IVideoFrameEncoder {
    public:
        core::Result<media::EncodedVideoFrame> Encode(
            const media::VideoFrameView&,
            double,
            std::string) override {
            return core::Status::Error(core::ErrorCode::Unimplemented, "not used");
        }

        const media::VideoFrameEncoderSelection& Selection() const noexcept override {
            return selection_;
        }

    private:
        media::VideoFrameEncoderSelection selection_;
    };

    auto pipeline = media::WebRtcMediaPipeline::Create({
        .session_id = "session-invalid-encoding",
        .frame_encoder = std::make_shared<NullEncoder>(),
    });

    ASSERT_FALSE(pipeline.ok());
    EXPECT_EQ(pipeline.status().code(), core::ErrorCode::InvalidArgument);
}

} // namespace
