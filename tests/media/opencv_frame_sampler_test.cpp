#include "opencv_frame_sampler.h"
#include "vision_runtime_interfaces.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <vector>

namespace {

media::VideoFrameView MakeFrame(std::vector<std::byte>& bytes,
                                std::uint64_t frame_id,
                                std::chrono::steady_clock::time_point captured_at,
                                bool bright_square) {
    constexpr std::uint32_t width = 160;
    constexpr std::uint32_t height = 120;
    bytes.assign(width * height * 3, std::byte{0});
    if (bright_square) {
        for (std::uint32_t y = 35; y < 85; ++y) {
            for (std::uint32_t x = 55; x < 105; ++x) {
                const auto offset = static_cast<std::size_t>((y * width + x) * 3);
                bytes[offset] = std::byte{255};
                bytes[offset + 1] = std::byte{255};
                bytes[offset + 2] = std::byte{255};
            }
        }
    }

    media::VideoFrameView frame;
    frame.session_id = "opencv-test";
    frame.frame_id = frame_id;
    frame.captured_at = captured_at;
    frame.width = width;
    frame.height = height;
    frame.format = media::VideoPixelFormat::Rgb;
    frame.bytes = std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return frame;
}

TEST(OpenCvFrameSamplerTest, AcceptsRgbRowsWithPadding) {
    media::OpenCvFrameSampler sampler;
    constexpr std::uint32_t width = 5;
    constexpr std::uint32_t height = 8;
    constexpr std::size_t stride = 20;
    std::vector<std::byte> bytes(stride * height, std::byte{0x20});
    media::VideoFrameView frame;
    frame.session_id = "session-stride";
    frame.frame_id = 1;
    frame.width = width;
    frame.height = height;
    frame.row_stride_bytes = stride;
    frame.format = media::VideoPixelFormat::Rgb;
    frame.bytes = {reinterpret_cast<const char*>(bytes.data()), bytes.size()};

    auto decision = sampler.Evaluate(frame);
    ASSERT_TRUE(decision.ok()) << decision.status().message();
}

} // namespace

TEST(OpenCvFrameSamplerTest, EmitsDecisionForLargeSyntheticChange) {
    media::OpenCvFrameSamplerConfig config;
    config.peak_threshold = 0.25;
    config.cooldown_seconds = 0.1;
    config.temporal_vote_required = 1;
    config.min_component_area_ratio = 0.001;
    media::OpenCvFrameSampler sampler(config);

    const auto start = std::chrono::steady_clock::now();
    bool emitted = false;
    std::vector<std::byte> storage;
    for (int i = 0; i < 12; ++i) {
        auto frame = MakeFrame(storage, static_cast<std::uint64_t>(i), start + std::chrono::milliseconds(100 * i), i >= 4);
        auto decision = sampler.Evaluate(frame);
        ASSERT_TRUE(decision.ok()) << decision.status().message();
        emitted = emitted || decision.value().submit_to_vlm;
    }

    EXPECT_TRUE(emitted);
}

TEST(OpenCvFrameSamplerTest, AdaptiveSamplerSkipsStableHighRateFrames) {
    media::OpenCvFrameSamplerConfig config;
    config.adaptive_enabled = true;
    config.adaptive_fps_min = 1.0;
    config.adaptive_fps_max = 2.0;
    config.adaptive_precheck_diff_threshold = 250.0;
    config.peak_threshold = 0.25;
    config.temporal_vote_required = 1;
    media::OpenCvFrameSampler sampler(config);

    const auto start = std::chrono::steady_clock::now();
    int adaptive_skips = 0;
    std::vector<std::byte> storage;
    for (int i = 0; i < 12; ++i) {
        auto frame = MakeFrame(storage, static_cast<std::uint64_t>(i), start + std::chrono::milliseconds(50 * i), false);
        auto decision = sampler.Evaluate(frame);
        ASSERT_TRUE(decision.ok()) << decision.status().message();
        if (decision.value().reason == "adaptive-skip") {
            adaptive_skips += 1;
        }
    }

    EXPECT_GT(adaptive_skips, 0);
}

TEST(VisionEventSinkTest, DropsOnlyHighSimilarityDuplicateEvents) {
    std::atomic<int> published{0};
    auto downstream = std::make_shared<media::ForwardingVisionEventSink>(
        [&published](media::VisionEvent) {
            published.fetch_add(1, std::memory_order_relaxed);
            return core::Status::Ok();
        });
    auto sink = std::make_shared<media::VectorDedupVisionEventSink>(
        media::VisionEventDedupOptions{
            .enabled = true,
            .high_similarity_threshold = 0.985f,
            .max_entries_per_session = 8,
            .ttl = std::chrono::seconds(60),
            .model_fingerprint = "test-vision-tower",
        },
        downstream);

    media::VisionEvent first;
    first.event_id = "event-1";
    first.session_id = "session-1";
    first.peak_score = 0.2;
    first.image_embedding = {1.0f, 0.0f, 0.0f};

    media::VisionEvent duplicate;
    duplicate.event_id = "event-2";
    duplicate.session_id = "session-1";
    duplicate.peak_score = 0.2;
    duplicate.image_embedding = {0.999f, 0.001f, 0.0f};

    media::VisionEvent distinct;
    distinct.event_id = "event-3";
    distinct.session_id = "session-1";
    distinct.peak_score = 0.2;
    distinct.image_embedding = {0.0f, 1.0f, 0.0f};

    ASSERT_TRUE(sink->Publish(std::move(first)).ok());
    ASSERT_TRUE(sink->Publish(std::move(duplicate)).ok());
    ASSERT_TRUE(sink->Publish(std::move(distinct)).ok());

    EXPECT_EQ(published.load(std::memory_order_relaxed), 2);
}

TEST(VisionEventMonitorTest, ClustersAndSummarizesWindows) {
    std::vector<media::VisionEvent> events;
    for (int i = 0; i < 3; ++i) {
        media::VisionEvent event;
        event.event_id = "event-" + std::to_string(i);
        event.session_id = "session-1";
        event.timestamp_seconds = 0.3 * i;
        event.peak_frame_id = static_cast<std::uint64_t>(i);
        event.peak_score = 0.4 + 0.1 * i;
        events.push_back(std::move(event));
    }
    media::VisionEvent later;
    later.event_id = "event-later";
    later.session_id = "session-1";
    later.timestamp_seconds = 4.0;
    later.peak_frame_id = 10;
    later.peak_score = 0.8;
    events.push_back(std::move(later));

    auto clusters = media::ClusterVisionEvents(events, 1.0);
    auto summaries = media::SummarizeVisionEventWindows(events, 3.0, 2, 1.0);

    ASSERT_EQ(clusters.size(), 2u);
    EXPECT_EQ(clusters[0].EventCount(), 3u);
    EXPECT_NEAR(clusters[0].AccumulatedScore(), 1.5, 1e-6);
    ASSERT_EQ(summaries.size(), 2u);
    EXPECT_EQ(summaries[0].total_events, 3u);
    EXPECT_EQ(summaries[0].total_clusters, 1u);
    EXPECT_EQ(summaries[1].total_events, 1u);
}

TEST(VisionEventMonitorTest, PromotesTriggeredSegmentWithRefractory) {
    media::VisionMonitorConfig config;
    config.trigger_window_seconds = 10.0;
    config.trigger_peak_score_threshold = 0.35;
    config.trigger_accumulated_score_threshold = 1.0;
    config.trigger_min_strong_events = 2;
    config.trigger_refractory_seconds = 8.0;
    config.summary_enabled = false;
    media::VisionEventMonitor monitor(config);

    media::VisionEvent first;
    first.event_id = "first";
    first.session_id = "session-1";
    first.timestamp_seconds = 1.0;
    first.peak_score = 0.45;
    auto first_update = monitor.ConsumeCandidate(first);

    media::VisionEvent second;
    second.event_id = "second";
    second.session_id = "session-1";
    second.timestamp_seconds = 2.0;
    second.peak_score = 0.55;
    auto second_update = monitor.ConsumeCandidate(second);

    media::VisionEvent third;
    third.event_id = "third";
    third.session_id = "session-1";
    third.timestamp_seconds = 3.0;
    third.peak_score = 0.8;
    auto third_update = monitor.ConsumeCandidate(third);

    EXPECT_TRUE(first_update.promoted_events.empty());
    ASSERT_EQ(second_update.promoted_events.size(), 1u);
    EXPECT_TRUE(third_update.promoted_events.empty());
    EXPECT_EQ(monitor.PromotedEvents().size(), 1u);
}

TEST(VisionEventMonitorTest, SinkForwardsPromotedEvents) {
    std::atomic<int> forwarded{0};
    auto downstream = std::make_shared<media::ForwardingVisionEventSink>(
        [&forwarded](media::VisionEvent event) {
            EXPECT_NE(event.event_id.find("trigger-"), std::string::npos);
            forwarded.fetch_add(1, std::memory_order_relaxed);
            return core::Status::Ok();
        });

    media::VisionMonitorConfig config;
    config.trigger_peak_score_threshold = 0.35;
    config.trigger_accumulated_score_threshold = 1.0;
    config.trigger_min_strong_events = 2;
    config.trigger_refractory_seconds = 8.0;
    config.summary_enabled = false;
    media::MonitorVisionEventSink sink(config, downstream);

    media::VisionEvent first;
    first.event_id = "first";
    first.session_id = "session-1";
    first.timestamp_seconds = 1.0;
    first.peak_score = 0.45;
    media::VisionEvent second;
    second.event_id = "second";
    second.session_id = "session-1";
    second.timestamp_seconds = 2.0;
    second.peak_score = 0.55;

    ASSERT_TRUE(sink.Publish(std::move(first)).ok());
    ASSERT_TRUE(sink.Publish(std::move(second)).ok());

    EXPECT_EQ(forwarded.load(std::memory_order_relaxed), 1);
    EXPECT_EQ(sink.PromotedEvents().size(), 1u);
}
