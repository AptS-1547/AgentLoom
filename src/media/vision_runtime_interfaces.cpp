#include "vision_runtime_interfaces.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <utility>

namespace media {
namespace {

std::int64_t ToTtlSeconds(std::chrono::seconds ttl) {
    return ttl.count() > 0 ? ttl.count() : 0;
}

void AppendMetricText(std::ostringstream& out, std::string_view label, double value) {
    out << label << "=";
    out.setf(std::ios::fixed);
    out.precision(2);
    out << value;
}

double MetricOr(const VisionEvent& event, std::string_view key, double fallback = 0.0) {
    auto it = event.metrics.find(std::string(key));
    return it == event.metrics.end() ? fallback : it->second;
}

std::string MakeClusterId(const std::vector<VisionEvent>& events, const VisionEvent& peak) {
    const auto ms = static_cast<long long>(std::floor(events.front().timestamp_seconds * 1000.0));
    return std::to_string(ms) + "-" + std::to_string(peak.peak_frame_id);
}

bool ContainsId(const std::deque<std::string>& ids, const std::string& id) {
    return std::find(ids.begin(), ids.end(), id) != ids.end();
}

} // namespace

core::Result<std::shared_ptr<GstMappedFrameBuffer>> GstMappedFrameBuffer::Create(GstBuffer* buffer) {
    if (!buffer) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "GStreamer buffer is null");
    }

    auto holder = std::shared_ptr<GstMappedFrameBuffer>(new GstMappedFrameBuffer(gst_buffer_ref(buffer)));
    if (!gst_buffer_map(holder->buffer_, &holder->map_, GST_MAP_READ)) {
        return core::Status::Error(core::ErrorCode::InternalError, "failed to map GStreamer frame buffer");
    }
    holder->bytes_ = std::string_view(reinterpret_cast<const char*>(holder->map_.data), holder->map_.size);
    return holder;
}

GstMappedFrameBuffer::GstMappedFrameBuffer(GstBuffer* buffer) noexcept
    : buffer_(buffer) {}

GstMappedFrameBuffer::~GstMappedFrameBuffer() {
    if (buffer_) {
        if (map_.data) {
            gst_buffer_unmap(buffer_, &map_);
        }
        gst_buffer_unref(buffer_);
        buffer_ = nullptr;
    }
}

std::string VisionAnalysis::SummaryText() const {
    if (!agent_hint.empty()) {
        return agent_hint;
    }
    if (!facts.empty()) {
        std::ostringstream out;
        for (std::size_t i = 0; i < facts.size() && i < 2; ++i) {
            if (i > 0) {
                out << "; ";
            }
            out << facts[i];
        }
        return out.str();
    }
    if (!raw_text.empty()) {
        return raw_text;
    }
    return "detected visual event";
}

std::string VisionEvent::SummaryText() const {
    if (analysis.has_value()) {
        return analysis->SummaryText();
    }
    return DefaultVisionEventText(*this);
}

ForwardingVisionEventSink::ForwardingVisionEventSink(Handler handler)
    : handler_(std::move(handler)) {}

core::Status ForwardingVisionEventSink::Publish(VisionEvent event) {
    if (!handler_) {
        return core::Status::Ok();
    }
    return handler_(std::move(event));
}

MonitorVisionEventSink::MonitorVisionEventSink(VisionMonitorConfig config,
                                               std::shared_ptr<IVisionEventSink> promoted_sink,
                                               VisionAnalyzeCallback analyze_callback)
    : monitor_(std::move(config)),
      promoted_sink_(std::move(promoted_sink)),
      analyze_callback_(std::move(analyze_callback)) {}

core::Status MonitorVisionEventSink::Publish(VisionEvent event) {
    auto update = monitor_.ConsumeCandidate(std::move(event), analyze_callback_);
    if (!promoted_sink_) {
        return core::Status::Ok();
    }
    for (auto& promoted : update.promoted_events) {
        auto status = promoted_sink_->Publish(std::move(promoted));
        if (!status.ok()) {
            return status;
        }
    }
    return core::Status::Ok();
}

VisionMonitorUpdate MonitorVisionEventSink::Finalize() {
    return monitor_.Finalize();
}

std::vector<VisionEvent> MonitorVisionEventSink::RecentEvents() const {
    return monitor_.RecentEvents();
}

std::vector<VisionEvent> MonitorVisionEventSink::PromotedEvents() const {
    return monitor_.PromotedEvents();
}

std::vector<VisionWindowSummary> MonitorVisionEventSink::CompletedSummaries() const {
    return monitor_.CompletedSummaries();
}

VectorDedupVisionEventSink::VectorDedupVisionEventSink(VisionEventDedupOptions options,
                                                       std::shared_ptr<IVisionEventSink> downstream)
    : options_(std::move(options)),
      downstream_(std::move(downstream)),
      index_(vlm_cache::VectorOptions{
          .enabled = options_.enabled,
          .sim_threshold_high = options_.high_similarity_threshold,
          .sim_threshold_mid = options_.high_similarity_threshold,
          .max_saliency_for_mid = 0.0f,
          .max_entries_per_bucket = options_.max_entries_per_session,
          .ttl_seconds = ToTtlSeconds(options_.ttl),
          .persist = false,
      }) {}

core::Status VectorDedupVisionEventSink::Publish(VisionEvent event) {
    if (!downstream_) {
        return core::Status::Ok();
    }
    if (!options_.enabled || event.image_embedding.empty()) {
        return downstream_->Publish(std::move(event));
    }

    const auto bucket = BucketKey(event);
    auto hit = index_.Query(bucket,
                            options_.model_fingerprint,
                            event.image_embedding,
                            static_cast<float>(event.peak_score));
    if (hit && !hit->tentative) {
        event.duplicate = true;
        event.duplicate_similarity = hit->similarity;
        event.duplicate_of_event_id = hit->cache_key;
        return core::Status::Ok();
    }

    vlm_cache::VectorEntry entry;
    entry.cache_key = event.event_id;
    entry.model_fingerprint = options_.model_fingerprint;
    entry.embedding = event.image_embedding;
    index_.Put(bucket, std::move(entry));
    return downstream_->Publish(std::move(event));
}

std::string VectorDedupVisionEventSink::BucketKey(const VisionEvent& event) {
    return event.session_id.empty() ? "vision:session:default" : "vision:session:" + event.session_id;
}

std::string DefaultVisionEventText(const VisionEvent& event) {
    const auto area_it = event.metrics.find("foreground_ratio");
    const auto hist_it = event.metrics.find("histogram_distance");
    const auto edge_it = event.metrics.find("edge_change");
    const double area = area_it == event.metrics.end() ? 0.0 : area_it->second;
    const double hist = hist_it == event.metrics.end() ? 0.0 : hist_it->second;
    const double edge = edge_it == event.metrics.end() ? 0.0 : edge_it->second;

    if (hist >= 0.35 && area >= 0.15) {
        return "visual scene changed significantly";
    }
    if (area >= 0.12 && edge >= 0.05) {
        return "detected obvious action or object change";
    }
    if (area >= 0.04) {
        return "detected local visual change";
    }
    if (event.peak_score >= 0.3) {
        return "detected visual event";
    }
    return "detected slight visual change";
}

std::string VisionEventToAgentText(const VisionEvent& event) {
    return "[视觉事件] " + event.SummaryText();
}

std::string VisionEventMemoryText(const VisionEvent& event) {
    if (event.analysis && !event.analysis->memory_candidate.empty()) {
        return event.analysis->memory_candidate;
    }
    if (event.analysis && !event.analysis->facts.empty()) {
        std::ostringstream out;
        for (std::size_t i = 0; i < event.analysis->facts.size() && i < 2; ++i) {
            if (i > 0) {
                out << "; ";
            }
            out << event.analysis->facts[i];
        }
        return out.str();
    }

    std::ostringstream out;
    out << DefaultVisionEventText(event) << " (";
    AppendMetricText(out, "score", event.peak_score);
    out << ")";
    return out.str();
}

std::size_t VisionEventCluster::EventCount() const noexcept {
    return events.size();
}

double VisionEventCluster::PeakScore() const noexcept {
    return peak_event ? peak_event->peak_score : 0.0;
}

double VisionEventCluster::AccumulatedScore() const noexcept {
    double sum = 0.0;
    for (const auto& event : events) {
        sum += event.peak_score;
    }
    return sum;
}

double VisionEventCluster::DurationSeconds() const noexcept {
    return std::max(0.0, end_timestamp - start_timestamp);
}

double VisionEventCluster::SalienceScore() const noexcept {
    const double accumulated = std::min(AccumulatedScore(), 3.0);
    const double event_bonus = std::min(static_cast<double>(EventCount()), 5.0) * 0.05;
    return PeakScore() + 0.35 * accumulated + event_bonus;
}

std::string VisionEventCluster::SummaryText() const {
    if (analysis.has_value()) {
        return analysis->SummaryText();
    }
    if (peak_event && peak_event->analysis) {
        return peak_event->analysis->SummaryText();
    }
    if (peak_event) {
        return DefaultVisionEventText(*peak_event);
    }
    return "detected visual segment";
}

std::optional<VisionEvent> VisionEventCluster::RepresentativeEvent() const {
    if (events.empty()) {
        return std::nullopt;
    }
    return *std::max_element(events.begin(), events.end(), [](const auto& lhs, const auto& rhs) {
        const auto lhs_sharpness = MetricOr(lhs, "sharpness");
        const auto rhs_sharpness = MetricOr(rhs, "sharpness");
        if (lhs_sharpness != rhs_sharpness) {
            return lhs_sharpness < rhs_sharpness;
        }
        return lhs.peak_score < rhs.peak_score;
    });
}

VisionEvent VisionEventCluster::ToVisionEvent(std::string_view mode,
                                              const VisionMonitorConfig&,
                                              std::optional<VisionAnalysis> analysis_override,
                                              bool rate_limited_override) const {
    VisionEvent promoted = peak_event.value_or(VisionEvent{});
    promoted.event_id = std::string(mode) + "-" + cluster_id;
    promoted.metrics["mode_trigger"] = mode == "trigger" ? 1.0 : 0.0;
    promoted.metrics["segment_start_timestamp"] = start_timestamp;
    promoted.metrics["segment_end_timestamp"] = end_timestamp;
    promoted.metrics["segment_duration_seconds"] = DurationSeconds();
    promoted.metrics["segment_event_count"] = static_cast<double>(EventCount());
    promoted.metrics["segment_accumulated_score"] = AccumulatedScore();
    promoted.metrics["segment_salience_score"] = SalienceScore();
    if (analysis_override.has_value()) {
        promoted.analysis = std::move(analysis_override);
    } else if (analysis.has_value()) {
        promoted.analysis = analysis;
    }
    promoted.rate_limited = rate_limited_override || rate_limited;
    return promoted;
}

std::string VisionWindowSummary::ToText() const {
    if (top_clusters.empty()) {
        std::ostringstream out;
        out << start_timestamp << "-" << end_timestamp << "s no significant visual segment";
        return out.str();
    }
    std::ostringstream out;
    out << start_timestamp << "-" << end_timestamp << "s total_clusters=" << total_clusters
        << " retained=" << top_clusters.size() << ": ";
    for (std::size_t i = 0; i < top_clusters.size(); ++i) {
        if (i > 0) {
            out << " ";
        }
        out << (i + 1) << ". " << top_clusters[i].SummaryText()
            << " peak=" << top_clusters[i].PeakScore()
            << " count=" << top_clusters[i].EventCount();
    }
    return out.str();
}

std::vector<VisionEventCluster> ClusterVisionEvents(const std::vector<VisionEvent>& events,
                                                    double merge_gap_seconds) {
    if (events.empty()) {
        return {};
    }
    auto sorted = events;
    std::sort(sorted.begin(), sorted.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.timestamp_seconds < rhs.timestamp_seconds;
    });

    std::vector<VisionEventCluster> clusters;
    std::vector<VisionEvent> current{sorted.front()};
    auto build_cluster = [](const std::vector<VisionEvent>& items) {
        const auto peak_it = std::max_element(items.begin(), items.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.peak_score != rhs.peak_score) {
                return lhs.peak_score < rhs.peak_score;
            }
            return MetricOr(lhs, "sharpness") < MetricOr(rhs, "sharpness");
        });
        VisionEventCluster cluster;
        cluster.cluster_id = MakeClusterId(items, *peak_it);
        cluster.start_timestamp = items.front().timestamp_seconds;
        cluster.end_timestamp = items.back().timestamp_seconds;
        cluster.events = items;
        cluster.peak_event = *peak_it;
        return cluster;
    };

    for (std::size_t i = 1; i < sorted.size(); ++i) {
        const double gap = sorted[i].timestamp_seconds - current.back().timestamp_seconds;
        if (gap <= merge_gap_seconds) {
            current.push_back(sorted[i]);
            continue;
        }
        clusters.push_back(build_cluster(current));
        current = {sorted[i]};
    }
    clusters.push_back(build_cluster(current));
    return clusters;
}

std::vector<VisionWindowSummary> SummarizeVisionEventWindows(const std::vector<VisionEvent>& events,
                                                             double window_seconds,
                                                             int top_k,
                                                             double merge_gap_seconds) {
    if (events.empty() || window_seconds <= 0.0 || top_k <= 0) {
        return {};
    }

    std::unordered_map<int, std::vector<VisionEvent>> windows;
    for (const auto& event : events) {
        const int index = static_cast<int>(std::floor(event.timestamp_seconds / window_seconds));
        windows[index].push_back(event);
    }

    std::vector<int> indices;
    indices.reserve(windows.size());
    for (const auto& [index, _] : windows) {
        indices.push_back(index);
    }
    std::sort(indices.begin(), indices.end());

    std::vector<VisionWindowSummary> summaries;
    for (int index : indices) {
        auto clusters = ClusterVisionEvents(windows[index], merge_gap_seconds);
        std::sort(clusters.begin(), clusters.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.SalienceScore() != rhs.SalienceScore()) {
                return lhs.SalienceScore() > rhs.SalienceScore();
            }
            if (lhs.PeakScore() != rhs.PeakScore()) {
                return lhs.PeakScore() > rhs.PeakScore();
            }
            return lhs.AccumulatedScore() > rhs.AccumulatedScore();
        });
        VisionWindowSummary summary;
        summary.window_index = index;
        summary.start_timestamp = index * window_seconds;
        summary.end_timestamp = summary.start_timestamp + window_seconds;
        summary.total_events = windows[index].size();
        summary.total_clusters = clusters.size();
        for (const auto& event : windows[index]) {
            summary.total_accumulated_score += event.peak_score;
        }
        const auto keep = std::min<std::size_t>(static_cast<std::size_t>(top_k), clusters.size());
        summary.top_clusters.assign(clusters.begin(), clusters.begin() + keep);
        summaries.push_back(std::move(summary));
    }
    return summaries;
}

VisionEventMonitor::VisionEventMonitor(VisionMonitorConfig config)
    : config_(std::move(config)) {}

VisionMonitorUpdate VisionEventMonitor::ConsumeCandidate(VisionEvent event,
                                                         VisionAnalyzeCallback analyze_callback) {
    VisionMonitorUpdate update;
    recent_events_.push_back(event);
    PruneRecentEvents(event.timestamp_seconds);

    if (config_.summary_enabled) {
        auto completed = ConsumeSummaryWindow(event);
        update.completed_summaries.insert(update.completed_summaries.end(),
                                          completed.begin(),
                                          completed.end());
    }

    if (config_.trigger_analysis_enabled) {
        auto promoted = MaybePromoteRecentSegment(event.timestamp_seconds, "trigger", std::move(analyze_callback));
        if (promoted.has_value()) {
            promoted_events_.push_back(*promoted);
            update.promoted_events.push_back(std::move(*promoted));
        }
    }
    return update;
}

std::vector<VisionEventCluster> VisionEventMonitor::RecentClusters(std::optional<double> window_seconds) const {
    auto events = RecentEvents();
    if (window_seconds.has_value() && !events.empty()) {
        const double cutoff = events.back().timestamp_seconds - *window_seconds;
        events.erase(std::remove_if(events.begin(), events.end(), [&](const auto& event) {
                         return event.timestamp_seconds < cutoff;
                     }),
                     events.end());
    }
    return ClusterVisionEvents(events, config_.segment_merge_gap_seconds);
}

std::vector<VisionEvent> VisionEventMonitor::AnalyzeRecentBuffer(std::optional<int> top_k,
                                                                 VisionAnalyzeCallback analyze_callback) {
    auto clusters = RecentClusters();
    std::sort(clusters.begin(), clusters.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.SalienceScore() != rhs.SalienceScore()) {
            return lhs.SalienceScore() > rhs.SalienceScore();
        }
        if (lhs.PeakScore() != rhs.PeakScore()) {
            return lhs.PeakScore() > rhs.PeakScore();
        }
        return lhs.AccumulatedScore() > rhs.AccumulatedScore();
    });

    const int limit = std::max(0, top_k.value_or(config_.explicit_request_top_k));
    std::vector<VisionEvent> promoted;
    const auto keep = std::min<std::size_t>(static_cast<std::size_t>(limit), clusters.size());
    for (std::size_t i = 0; i < keep; ++i) {
        promoted.push_back(ClusterToPromotedEvent(clusters[i], "manual", analyze_callback));
    }
    return promoted;
}

VisionMonitorUpdate VisionEventMonitor::Finalize() {
    VisionMonitorUpdate update;
    if (config_.summary_enabled) {
        auto summary = FinalizeCurrentSummaryWindow();
        if (summary.has_value()) {
            completed_summaries_.push_back(*summary);
            update.completed_summaries.push_back(std::move(*summary));
        }
    }
    return update;
}

std::vector<VisionEvent> VisionEventMonitor::RecentEvents() const {
    return {recent_events_.begin(), recent_events_.end()};
}

std::vector<VisionEvent> VisionEventMonitor::PromotedEvents() const {
    return promoted_events_;
}

std::vector<VisionWindowSummary> VisionEventMonitor::CompletedSummaries() const {
    return completed_summaries_;
}

void VisionEventMonitor::PruneRecentEvents(double now) {
    const double cutoff = now - std::max(0.0, config_.weak_monitor_buffer_seconds);
    while (!recent_events_.empty() && recent_events_.front().timestamp_seconds < cutoff) {
        recent_events_.pop_front();
    }
}

std::vector<VisionWindowSummary> VisionEventMonitor::ConsumeSummaryWindow(const VisionEvent& event) {
    if (config_.summary_window_seconds <= 0.0) {
        return {};
    }

    const int window_index = static_cast<int>(std::floor(event.timestamp_seconds / config_.summary_window_seconds));
    std::vector<VisionWindowSummary> completed;
    if (!current_summary_window_index_.has_value()) {
        current_summary_window_index_ = window_index;
    } else if (*current_summary_window_index_ != window_index) {
        auto summary = FinalizeCurrentSummaryWindow();
        if (summary.has_value()) {
            completed_summaries_.push_back(*summary);
            completed.push_back(std::move(*summary));
        }
        current_summary_window_index_ = window_index;
    }

    current_summary_events_.push_back(event);
    return completed;
}

std::optional<VisionWindowSummary> VisionEventMonitor::FinalizeCurrentSummaryWindow() {
    if (!current_summary_window_index_.has_value() ||
        current_summary_events_.empty() ||
        config_.summary_top_k <= 0) {
        current_summary_events_.clear();
        return std::nullopt;
    }

    auto summaries = SummarizeVisionEventWindows(current_summary_events_,
                                                 config_.summary_window_seconds,
                                                 config_.summary_top_k,
                                                 config_.segment_merge_gap_seconds);
    current_summary_events_.clear();
    if (summaries.empty()) {
        return std::nullopt;
    }
    return summaries.front();
}

std::optional<VisionEvent> VisionEventMonitor::MaybePromoteRecentSegment(double now,
                                                                         std::string_view mode,
                                                                         VisionAnalyzeCallback analyze_callback) {
    if ((now - last_trigger_time_) < config_.trigger_refractory_seconds) {
        return std::nullopt;
    }
    const double cutoff = now - config_.trigger_window_seconds;
    std::vector<VisionEvent> window_events;
    for (const auto& event : recent_events_) {
        if (event.timestamp_seconds >= cutoff) {
            window_events.push_back(event);
        }
    }
    if (window_events.empty()) {
        return std::nullopt;
    }

    auto clusters = ClusterVisionEvents(window_events, config_.segment_merge_gap_seconds);
    std::vector<VisionEventCluster> strong_clusters;
    for (const auto& cluster : clusters) {
        if (cluster.PeakScore() >= config_.trigger_peak_score_threshold) {
            strong_clusters.push_back(cluster);
        }
    }
    if (strong_clusters.empty()) {
        return std::nullopt;
    }

    double accumulated = 0.0;
    for (const auto& event : window_events) {
        accumulated += event.peak_score;
    }
    if (accumulated < config_.trigger_accumulated_score_threshold &&
        static_cast<int>(strong_clusters.size()) < config_.trigger_min_strong_events) {
        return std::nullopt;
    }

    const auto best_it = std::max_element(strong_clusters.begin(), strong_clusters.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.SalienceScore() != rhs.SalienceScore()) {
            return lhs.SalienceScore() < rhs.SalienceScore();
        }
        if (lhs.PeakScore() != rhs.PeakScore()) {
            return lhs.PeakScore() < rhs.PeakScore();
        }
        return lhs.AccumulatedScore() < rhs.AccumulatedScore();
    });
    const std::string peak_id = best_it->peak_event ? best_it->peak_event->event_id : "";
    if (!peak_id.empty() && ContainsId(promoted_peak_event_ids_, peak_id)) {
        return std::nullopt;
    }

    auto promoted = ClusterToPromotedEvent(*best_it, mode, std::move(analyze_callback));
    if (!peak_id.empty()) {
        promoted_peak_event_ids_.push_back(peak_id);
        while (promoted_peak_event_ids_.size() > 128) {
            promoted_peak_event_ids_.pop_front();
        }
    }
    last_trigger_time_ = now;
    return promoted;
}

VisionEvent VisionEventMonitor::ClusterToPromotedEvent(const VisionEventCluster& cluster,
                                                       std::string_view mode,
                                                       VisionAnalyzeCallback analyze_callback) {
    auto promoted = cluster.ToVisionEvent(mode, config_);
    if (analyze_callback) {
        auto [analysis, rate_limited] = analyze_callback(promoted);
        if (analysis.has_value()) {
            promoted.analysis = std::move(analysis);
        }
        promoted.rate_limited = rate_limited;
    }
    return promoted;
}

} // namespace media
