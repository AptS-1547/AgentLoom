#pragma once

#include "result.h"
#include "vector_cache.h"
#include "vision_inference_interfaces.h"

#include <gst/gst.h>

#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace media {

class GstMappedFrameBuffer {
public:
    /// 映射 GstBuffer 并持有其引用；返回对象销毁前 bytes() 始终有效。
    /// @param buffer 借用的 GStreamer buffer，Create 成功后由返回对象延长生命周期。
    static core::Result<std::shared_ptr<GstMappedFrameBuffer>> Create(GstBuffer* buffer);

    ~GstMappedFrameBuffer();

    GstMappedFrameBuffer(const GstMappedFrameBuffer&) = delete;
    GstMappedFrameBuffer& operator=(const GstMappedFrameBuffer&) = delete;
    GstMappedFrameBuffer(GstMappedFrameBuffer&&) = delete;
    GstMappedFrameBuffer& operator=(GstMappedFrameBuffer&&) = delete;

    std::string_view bytes() const noexcept {
        return bytes_;
    }

    std::size_t size() const noexcept {
        return bytes_.size();
    }

private:
    explicit GstMappedFrameBuffer(GstBuffer* buffer) noexcept;

    GstBuffer* buffer_ = nullptr;
    GstMapInfo map_{};
    std::string_view bytes_;
};

enum class VideoPixelFormat {
    Unknown,
    Rgb,
    Bgr,
    Nv12,
    I420
};

struct VideoFrameView {
    std::string session_id;
    std::uint64_t frame_id = 0;
    std::chrono::steady_clock::time_point captured_at = std::chrono::steady_clock::now();
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::size_t row_stride_bytes = 0;
    VideoPixelFormat format = VideoPixelFormat::Unknown;
    std::shared_ptr<const GstMappedFrameBuffer> buffer;
    std::string_view bytes;
};

using FrameObserver = std::function<void(const VideoFrameView&)>;

struct FrameSamplingDecision {
    bool submit_to_vlm = false;
    double saliency_score = 0.0;
    std::string reason;
};

class IFrameSampler {
public:
    virtual ~IFrameSampler() = default;

    /// 在 decoded frame 离开 GStreamer/appsink 后由 compute pool 调用。
    /// @param frame 借用的帧视图，不得保存其中 string_view；需要异步使用时必须复制。
    virtual core::Result<FrameSamplingDecision> Evaluate(const VideoFrameView& frame) = 0;
};

struct VisionAnalysis {
    std::string scene;
    std::vector<std::string> facts;
    std::vector<std::string> weak_interpretations;
    std::string memory_candidate;
    std::string agent_hint;
    std::string raw_text;
    double confidence = 0.0;

    std::string SummaryText() const;
};

struct VisionEvent {
    std::string event_id;
    std::string session_id;
    std::string trace_id;
    std::uint64_t peak_frame_id = 0;
    std::uint64_t representative_frame_id = 0;
    double timestamp_seconds = 0.0;
    std::chrono::steady_clock::time_point captured_at = std::chrono::steady_clock::now();
    double peak_score = 0.0;
    std::unordered_map<std::string, double> metrics;
    std::optional<VisionAnalysis> analysis;
    std::vector<float> image_embedding;
    bool rate_limited = false;
    bool duplicate = false;
    float duplicate_similarity = 0.0f;
    std::string duplicate_of_event_id;

    std::string SummaryText() const;
};

struct VisionMonitorConfig {
    double weak_monitor_buffer_seconds = 30.0;
    double segment_merge_gap_seconds = 2.0;
    bool trigger_analysis_enabled = true;
    double trigger_window_seconds = 10.0;
    double trigger_accumulated_score_threshold = 1.2;
    double trigger_peak_score_threshold = 0.35;
    int trigger_min_strong_events = 2;
    double trigger_refractory_seconds = 8.0;
    int explicit_request_top_k = 2;
    bool summary_enabled = true;
    double summary_window_seconds = 30.0;
    int summary_top_k = 3;
    int max_keyframes_per_event = 3;
};

struct VisionEventCluster {
    std::string cluster_id;
    double start_timestamp = 0.0;
    double end_timestamp = 0.0;
    std::vector<VisionEvent> events;
    std::optional<VisionEvent> peak_event;
    std::optional<VisionAnalysis> analysis;
    bool rate_limited = false;

    std::size_t EventCount() const noexcept;
    double PeakScore() const noexcept;
    double AccumulatedScore() const noexcept;
    double DurationSeconds() const noexcept;
    double SalienceScore() const noexcept;
    std::string SummaryText() const;
    std::optional<VisionEvent> RepresentativeEvent() const;
    VisionEvent ToVisionEvent(std::string_view mode,
                              const VisionMonitorConfig& config,
                              std::optional<VisionAnalysis> analysis_override = std::nullopt,
                              bool rate_limited_override = false) const;
};

struct VisionWindowSummary {
    int window_index = 0;
    double start_timestamp = 0.0;
    double end_timestamp = 0.0;
    std::vector<VisionEventCluster> top_clusters;
    std::size_t total_events = 0;
    std::size_t total_clusters = 0;
    double total_accumulated_score = 0.0;

    std::string ToText() const;
};

struct VisionMonitorUpdate {
    std::vector<VisionEvent> promoted_events;
    std::vector<VisionWindowSummary> completed_summaries;
};

using VisionAnalyzeCallback = std::function<std::pair<std::optional<VisionAnalysis>, bool>(VisionEvent)>;

std::vector<VisionEventCluster> ClusterVisionEvents(const std::vector<VisionEvent>& events,
                                                    double merge_gap_seconds);
std::vector<VisionWindowSummary> SummarizeVisionEventWindows(const std::vector<VisionEvent>& events,
                                                             double window_seconds,
                                                             int top_k,
                                                             double merge_gap_seconds);

class VisionEventMonitor final {
public:
    explicit VisionEventMonitor(VisionMonitorConfig config = {});

    VisionMonitorUpdate ConsumeCandidate(VisionEvent event, VisionAnalyzeCallback analyze_callback = {});
    std::vector<VisionEventCluster> RecentClusters(std::optional<double> window_seconds = std::nullopt) const;
    std::vector<VisionEvent> AnalyzeRecentBuffer(std::optional<int> top_k = std::nullopt,
                                                 VisionAnalyzeCallback analyze_callback = {});
    VisionMonitorUpdate Finalize();

    std::vector<VisionEvent> RecentEvents() const;
    std::vector<VisionEvent> PromotedEvents() const;
    std::vector<VisionWindowSummary> CompletedSummaries() const;

private:
    void PruneRecentEvents(double now);
    std::vector<VisionWindowSummary> ConsumeSummaryWindow(const VisionEvent& event);
    std::optional<VisionWindowSummary> FinalizeCurrentSummaryWindow();
    std::optional<VisionEvent> MaybePromoteRecentSegment(double now,
                                                         std::string_view mode,
                                                         VisionAnalyzeCallback analyze_callback);
    VisionEvent ClusterToPromotedEvent(const VisionEventCluster& cluster,
                                       std::string_view mode,
                                       VisionAnalyzeCallback analyze_callback);

    VisionMonitorConfig config_;
    std::deque<VisionEvent> recent_events_;
    std::deque<std::string> promoted_peak_event_ids_;
    std::vector<VisionEvent> promoted_events_;
    std::vector<VisionWindowSummary> completed_summaries_;
    std::optional<int> current_summary_window_index_;
    std::vector<VisionEvent> current_summary_events_;
    double last_trigger_time_ = -1.0e9;
};

struct VisionContextSnapshot {
    std::string session_id;
    std::vector<VisionEvent> recent_events;
    std::optional<VisionEvent> latest_event;
    std::string compact_context_text;
};

class IVisionEventSink {
public:
    virtual ~IVisionEventSink() = default;
    /// 发布视觉事件；按值传递允许 sink 接管较大的 analysis/embedding 数据。
    virtual core::Status Publish(VisionEvent event) = 0;
};

class IVisionContextProvider {
public:
    virtual ~IVisionContextProvider() = default;
    /// @param session_id 目标 Persona session。
    /// @param max_age 允许返回的视觉事件最大年龄。
    virtual core::Result<VisionContextSnapshot> Snapshot(const std::string& session_id,
                                                         std::chrono::milliseconds max_age) = 0;
};

class ForwardingVisionEventSink final : public IVisionEventSink {
public:
    using Handler = std::function<core::Status(VisionEvent)>;

    explicit ForwardingVisionEventSink(Handler handler);
    core::Status Publish(VisionEvent event) override;

private:
    Handler handler_;
};

class MonitorVisionEventSink final : public IVisionEventSink {
public:
    MonitorVisionEventSink(VisionMonitorConfig config,
                           std::shared_ptr<IVisionEventSink> promoted_sink = nullptr,
                           VisionAnalyzeCallback analyze_callback = {});

    core::Status Publish(VisionEvent event) override;
    VisionMonitorUpdate Finalize();
    std::vector<VisionEvent> RecentEvents() const;
    std::vector<VisionEvent> PromotedEvents() const;
    std::vector<VisionWindowSummary> CompletedSummaries() const;

private:
    VisionEventMonitor monitor_;
    std::shared_ptr<IVisionEventSink> promoted_sink_;
    VisionAnalyzeCallback analyze_callback_;
};

struct VisionEventDedupOptions {
    bool enabled = false;
    float high_similarity_threshold = 0.985f;
    std::size_t max_entries_per_session = 128;
    std::chrono::seconds ttl{300};
    std::string model_fingerprint = "vision-tower";
};

class VectorDedupVisionEventSink final : public IVisionEventSink {
public:
    VectorDedupVisionEventSink(VisionEventDedupOptions options,
                               std::shared_ptr<IVisionEventSink> downstream);

    core::Status Publish(VisionEvent event) override;

private:
    static std::string BucketKey(const VisionEvent& event);

    VisionEventDedupOptions options_;
    std::shared_ptr<IVisionEventSink> downstream_;
    vlm_cache::VectorIndex index_;
};

std::string DefaultVisionEventText(const VisionEvent& event);
std::string VisionEventToAgentText(const VisionEvent& event);
std::string VisionEventMemoryText(const VisionEvent& event);

} // namespace media
