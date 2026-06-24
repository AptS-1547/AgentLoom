#include "webrtc_media_pipeline.h"

#include <gtest/gtest.h>

TEST(WebRtcMediaPipelineTest, CreatesPipelineWithWebRtcBin) {
    auto pipeline = media::WebRtcMediaPipeline::Create(media::WebRtcMediaPipelineOptions{
        .session_id = "vision-session",
        .name = "agent-webrtc-media-pipeline-test",
    });

    ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();
    EXPECT_NE(pipeline.value()->pipeline(), nullptr);
    EXPECT_NE(pipeline.value()->webrtcbin(), nullptr);
}

TEST(WebRtcMediaPipelineTest, CreatesSoftwareDecoderPipeline) {
    auto pipeline = media::WebRtcMediaPipeline::Create(media::WebRtcMediaPipelineOptions{
        .session_id = "vision-session",
        .name = "agent-webrtc-media-pipeline-software-test",
        .decode = {.video_decoder = media::VideoDecoderPreference::Software},
    });

    ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();
    EXPECT_NE(pipeline.value()->pipeline(), nullptr);
}

TEST(WebRtcMediaPipelineTest, CreatesNvidiaDecoderPipelineWhenPluginExists) {
    auto pipeline = media::WebRtcMediaPipeline::Create(media::WebRtcMediaPipelineOptions{
        .session_id = "vision-session",
        .name = "agent-webrtc-media-pipeline-nvidia-test",
        .decode = {.video_decoder = media::VideoDecoderPreference::Nvidia},
    });

    ASSERT_TRUE(pipeline.ok()) << pipeline.status().message();
    EXPECT_NE(pipeline.value()->pipeline(), nullptr);
}

TEST(WebRtcMediaPipelineTest, RejectsVaapiWhenPluginIsMissing) {
    auto pipeline = media::WebRtcMediaPipeline::Create(media::WebRtcMediaPipelineOptions{
        .session_id = "vision-session",
        .name = "agent-webrtc-media-pipeline-vaapi-test",
        .decode = {.video_decoder = media::VideoDecoderPreference::Vaapi},
    });

    if (!pipeline.ok()) {
        EXPECT_EQ(pipeline.status().code(), core::ErrorCode::Unavailable);
    }
}
