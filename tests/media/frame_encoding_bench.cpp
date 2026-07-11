#include "frame_encoding.h"

#include "memory_pool.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

int main() {
    constexpr std::uint32_t kWidth = 640;
    constexpr std::uint32_t kHeight = 360;
    constexpr std::size_t kFrameCount = 256;
    core::BucketMemoryPool pool(0, 64);
    auto encoder = media::GStreamerVideoFrameEncoder::Create(pool, {
        .format = media::EncodedVideoFrameFormat::Jpeg,
        .preference = media::VideoImageEncoderPreference::Auto,
        .jpeg_quality = 85,
    });
    if (!encoder.ok()) {
        std::cerr << encoder.status().message() << '\n';
        return 1;
    }

    std::vector<std::byte> payload(static_cast<std::size_t>(kWidth) * kHeight * 3);
    for (std::size_t index = 0; index < payload.size(); ++index) {
        payload[index] = static_cast<std::byte>((index * 31) & 0xFF);
    }
    media::VideoFrameView frame;
    frame.session_id = "encoding-bench";
    frame.width = kWidth;
    frame.height = kHeight;
    frame.format = media::VideoPixelFormat::Rgb;
    frame.bytes = {reinterpret_cast<const char*>(payload.data()), payload.size()};

    std::size_t encoded_bytes = 0;
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < kFrameCount; ++index) {
        frame.frame_id = index + 1;
        frame.captured_at = std::chrono::steady_clock::now();
        auto encoded = encoder.value()->Encode(frame, 0.5);
        if (!encoded.ok()) {
            std::cerr << encoded.status().message() << '\n';
            return 1;
        }
        encoded_bytes += encoded.value().bytes().size();
    }
    const auto elapsed = std::chrono::steady_clock::now() - started;
    const auto seconds = std::chrono::duration<double>(elapsed).count();

    std::cout << "factory: " << encoder.value()->Selection().factory_name << '\n'
              << "hardware: " << encoder.value()->Selection().hardware << '\n'
              << "frames: " << kFrameCount << '\n'
              << "resolution: " << kWidth << 'x' << kHeight << '\n'
              << "elapsed: " << seconds * 1000.0 << " ms\n"
              << "throughput: " << static_cast<double>(kFrameCount) / seconds << " frames/s\n"
              << "average encoded bytes: " << encoded_bytes / kFrameCount << '\n';
    return 0;
}
