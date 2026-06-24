#include "webrtc_bin.h"

#include <gtest/gtest.h>

TEST(WebRtcBinTest, CreatesAndOwnsGStreamerElement) {
    auto webrtc = media::WebRtcBin::Create();
    ASSERT_TRUE(webrtc.ok()) << webrtc.status().message();
    EXPECT_NE(webrtc.value()->element(), nullptr);
    EXPECT_FALSE(webrtc.value()->name().empty());
}

TEST(WebRtcBinTest, RejectsEmptyTypedSdp) {
    auto webrtc = media::WebRtcBin::Create();
    ASSERT_TRUE(webrtc.ok()) << webrtc.status().message();

    auto status = webrtc.value()->SetRemoteDescriptionFromSdp(media::WebRtcSdpType::Offer, "");
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), core::ErrorCode::InvalidArgument);
}
