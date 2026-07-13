#include "opencv_frame_sampler.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace media {
namespace {

std::string DecisionReason(double score, bool motion_vote, bool peak, double foreground_ratio) {
    std::ostringstream out;
    out << (peak ? "peak" : "skip")
        << ";score=" << score
        << ";motion=" << (motion_vote ? "true" : "false")
        << ";foreground=" << foreground_ratio;
    return out.str();
}

} // namespace

OpenCvFrameSampler::OpenCvFrameSampler(OpenCvFrameSamplerConfig config)
    : config_(config) {}

core::Result<FrameSamplingDecision> OpenCvFrameSampler::Evaluate(const VideoFrameView& frame) {
    auto rgb = BuildRgbMat(frame);
    if (!rgb.ok()) {
        return rgb.status();
    }

    std::shared_ptr<SessionEntry> entry;
    {
        std::lock_guard lock(sessions_mutex_);
        auto [it, inserted] = sessions_.try_emplace(frame.session_id);
        if (inserted) {
            it->second = std::make_shared<SessionEntry>();
            it->second->state = MakeSessionState();
        }
        entry = it->second;
    }
    std::lock_guard entry_lock(entry->mutex);
    auto& state = entry->state;
    const double now_seconds = frame.timestamp_us.has_value()
        ? static_cast<double>(*frame.timestamp_us) / 1'000'000.0
        : std::chrono::duration<double>(frame.captured_at.time_since_epoch()).count();

    if (config_.adaptive_enabled && !ShouldRunHeavyAnalysis(state, rgb.value(), now_seconds)) {
        return FrameSamplingDecision{
            false,
            state.previous_smoothed_score,
            "adaptive-skip",
        };
    }

    cv::Mat resized;
    if (config_.resize_width > 0 && rgb.value().cols > config_.resize_width) {
        const double scale = static_cast<double>(config_.resize_width) / static_cast<double>(rgb.value().cols);
        cv::resize(rgb.value(), resized, cv::Size(config_.resize_width, std::max(1, static_cast<int>(rgb.value().rows * scale))));
    } else {
        resized = rgb.value();
    }

    cv::Mat gray;
    cv::cvtColor(resized, gray, cv::COLOR_RGB2GRAY);

    const int gaussian_kernel = OddKernel(config_.gaussian_kernel_size);
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(gaussian_kernel, gaussian_kernel), 0);

    cv::Mat foreground;
    state.subtractor->apply(blurred, foreground);
    if (config_.mog2_detect_shadows) {
        cv::threshold(foreground, foreground, 200, 255, cv::THRESH_BINARY);
    }
    cv::erode(foreground, foreground, state.morphology_kernel, cv::Point(-1, -1), 1);
    cv::dilate(foreground, foreground, state.morphology_kernel, cv::Point(-1, -1), 1);
    const double foreground_ratio = FilterSmallComponents(foreground, config_.min_component_area_ratio);

    const int hist_size = 32;
    const float range[] = {0.0f, 256.0f};
    const float* ranges[] = {range};
    cv::Mat histogram;
    cv::calcHist(&blurred, 1, nullptr, cv::Mat(), histogram, 1, &hist_size, ranges);
    cv::normalize(histogram, histogram);

    double histogram_distance = 0.0;
    if (state.has_previous_histogram) {
        histogram_distance = cv::compareHist(histogram, state.previous_histogram, cv::HISTCMP_BHATTACHARYYA);
    }

    cv::Mat edges;
    cv::Canny(blurred, edges, config_.canny_threshold1, config_.canny_threshold2);
    const double edge_density = static_cast<double>(cv::countNonZero(edges)) / static_cast<double>(edges.total());
    double edge_change = 0.0;
    if (state.has_previous_edge_density) {
        edge_change = std::abs(edge_density - state.previous_edge_density);
    }

    const double area_score = SigmoidNormalize(foreground_ratio, config_.area_sigmoid_center, config_.area_sigmoid_scale);
    const double histogram_score = SigmoidNormalize(
        histogram_distance,
        config_.histogram_sigmoid_center,
        config_.histogram_sigmoid_scale);
    const double edge_score = SigmoidNormalize(edge_change, config_.edge_sigmoid_center, config_.edge_sigmoid_scale);

    const double change_score = config_.area_weight * area_score +
                                config_.histogram_weight * histogram_score +
                                config_.edge_weight * edge_score;
    const double smoothed_score = config_.ema_alpha * change_score +
                                  (1.0 - config_.ema_alpha) * state.previous_smoothed_score;

    const bool motion_flag = foreground_ratio >= config_.min_component_area_ratio;
    state.temporal_votes.push_back(motion_flag);
    while (static_cast<int>(state.temporal_votes.size()) > std::max(1, config_.temporal_vote_window)) {
        state.temporal_votes.pop_front();
    }
    const int votes = static_cast<int>(std::count(state.temporal_votes.begin(), state.temporal_votes.end(), true));
    const bool motion_vote = votes >= std::max(1, config_.temporal_vote_required);

    const bool cooldown_ready = (now_seconds - state.last_peak_seconds) >= config_.cooldown_seconds;
    const bool peak = smoothed_score >= config_.peak_threshold && motion_vote && cooldown_ready;
    if (peak) {
        state.last_peak_seconds = now_seconds;
    }

    state.previous_histogram = histogram.clone();
    state.previous_edge_density = edge_density;
    state.has_previous_histogram = true;
    state.has_previous_edge_density = true;
    state.previous_smoothed_score = smoothed_score;
    if (config_.adaptive_enabled) {
        MarkAdaptiveSampled(state, rgb.value(), now_seconds);
        UpdateAdaptive(state, change_score, smoothed_score, now_seconds);
    }

    return FrameSamplingDecision{
        peak,
        smoothed_score,
        DecisionReason(smoothed_score, motion_vote, peak, foreground_ratio),
    };
}

core::Result<cv::Mat> OpenCvFrameSampler::BuildRgbMat(const VideoFrameView& frame) const {
    if (frame.format != VideoPixelFormat::Rgb) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "OpenCvFrameSampler expects RGB frames");
    }
    if (frame.width == 0 || frame.height == 0 || frame.bytes.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "frame is empty");
    }

    const auto packed_stride = static_cast<std::size_t>(frame.width) * 3;
    const auto row_stride = frame.row_stride_bytes == 0 ? packed_stride : frame.row_stride_bytes;
    if (row_stride < packed_stride) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "RGB frame stride is smaller than width*3");
    }
    const auto expected = row_stride * static_cast<std::size_t>(frame.height);
    if (frame.bytes.size() < expected) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "RGB frame buffer is smaller than width*height*3");
    }

    return cv::Mat(
        static_cast<int>(frame.height),
        static_cast<int>(frame.width),
        CV_8UC3,
        const_cast<char*>(frame.bytes.data()),
        row_stride);
}

OpenCvFrameSampler::SessionState OpenCvFrameSampler::MakeSessionState() const {
    SessionState state;
    state.subtractor = cv::createBackgroundSubtractorMOG2(
        config_.mog2_history,
        config_.mog2_var_threshold,
        config_.mog2_detect_shadows);
    state.morphology_kernel = cv::Mat::ones(
        OddKernel(config_.morphology_kernel_size),
        OddKernel(config_.morphology_kernel_size),
        CV_8U);
    state.current_target_fps = config_.adaptive_fps_max;
    return state;
}

int OpenCvFrameSampler::OddKernel(int value) noexcept {
    value = std::max(1, value);
    return value % 2 == 0 ? value + 1 : value;
}

double OpenCvFrameSampler::SigmoidNormalize(double value, double center, double scale) noexcept {
    const double safe_scale = std::max(1.0e-9, scale);
    const double x = std::clamp((value - center) / safe_scale, -60.0, 60.0);
    return 1.0 / (1.0 + std::exp(-x));
}

double OpenCvFrameSampler::FilterSmallComponents(const cv::Mat& mask, double min_area_ratio) {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    const double total_pixels = static_cast<double>(mask.rows) * static_cast<double>(mask.cols);
    if (total_pixels <= 0.0) {
        return 0.0;
    }

    const int min_area = std::max(1, static_cast<int>(total_pixels * min_area_ratio));
    int kept_pixels = 0;
    for (int label = 1; label < count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area >= min_area) {
            kept_pixels += area;
        }
    }
    return static_cast<double>(kept_pixels) / total_pixels;
}

bool OpenCvFrameSampler::ShouldRunHeavyAnalysis(SessionState& state, const cv::Mat& rgb, double now_seconds) const {
    const double fps = EffectiveFps(state, now_seconds);
    const double interval = 1.0 / std::max(fps, 0.1);
    if ((now_seconds - state.last_sampled_seconds) >= interval) {
        return true;
    }

    const int width = std::max(1, config_.adaptive_precheck_resize_width);
    const int height = std::max(1, static_cast<int>(static_cast<double>(rgb.rows) * width / std::max(1, rgb.cols)));
    cv::Mat small;
    cv::resize(rgb, small, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    cv::Mat gray;
    cv::cvtColor(small, gray, cv::COLOR_RGB2GRAY);
    if (state.precheck_ref_gray.empty()) {
        state.precheck_ref_gray = gray;
        return false;
    }

    cv::Mat diff;
    cv::absdiff(gray, state.precheck_ref_gray, diff);
    const double mean_diff = cv::mean(diff)[0];
    if (mean_diff >= config_.adaptive_precheck_diff_threshold) {
        state.precheck_ref_gray = gray;
        return true;
    }
    return false;
}

void OpenCvFrameSampler::MarkAdaptiveSampled(SessionState& state, const cv::Mat& rgb, double now_seconds) const {
    state.last_sampled_seconds = now_seconds;
    const int width = std::max(1, config_.adaptive_precheck_resize_width);
    const int height = std::max(1, static_cast<int>(static_cast<double>(rgb.rows) * width / std::max(1, rgb.cols)));
    cv::Mat small;
    cv::resize(rgb, small, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
    cv::cvtColor(small, state.precheck_ref_gray, cv::COLOR_RGB2GRAY);
}

void OpenCvFrameSampler::UpdateAdaptive(
    SessionState& state,
    double change_score,
    double smoothed_score,
    double now_seconds) const {
    state.score_buffer.push_back(smoothed_score);
    while (static_cast<int>(state.score_buffer.size()) > std::max(1, config_.adaptive_stft_window_size)) {
        state.score_buffer.pop_front();
    }
    if (change_score >= config_.adaptive_spike_threshold) {
        state.last_spike_seconds = now_seconds;
    }
    state.frames_since_last_fft += 1;
    if (state.frames_since_last_fft >= std::max(1, config_.adaptive_stft_hop_size)) {
        RecomputeFft(state);
        state.frames_since_last_fft = 0;
    }
}

void OpenCvFrameSampler::RecomputeFft(SessionState& state) const {
    const int n = std::max(1, config_.adaptive_stft_window_size);
    if (static_cast<int>(state.score_buffer.size()) < n) {
        return;
    }

    cv::Mat signal(n, 1, CV_32F);
    double mean = 0.0;
    for (double value : state.score_buffer) {
        mean += value;
    }
    mean /= static_cast<double>(n);

    for (int i = 0; i < n; ++i) {
        const double hann = 0.5 * (1.0 - std::cos((2.0 * CV_PI * i) / std::max(1, n - 1)));
        signal.at<float>(i, 0) = static_cast<float>((state.score_buffer[static_cast<std::size_t>(i)] - mean) * hann);
    }

    cv::Mat spectrum;
    cv::dft(signal, spectrum, cv::DFT_COMPLEX_OUTPUT);

    const int bins = n / 2 + 1;
    const int cutoff = std::max(1, static_cast<int>(bins * config_.adaptive_highfreq_cutoff_ratio));
    double total_energy = 0.0;
    double high_energy = 0.0;
    for (int i = 1; i < bins; ++i) {
        const auto value = spectrum.at<cv::Vec2f>(i, 0);
        const double power = static_cast<double>(value[0]) * value[0] + static_cast<double>(value[1]) * value[1];
        total_energy += power;
        if (i >= cutoff) {
            high_energy += power;
        }
    }

    const double ratio = high_energy / (total_energy + 1.0e-8);
    const double raw_fps = config_.adaptive_fps_min +
                           (config_.adaptive_fps_max - config_.adaptive_fps_min) *
                               std::pow(std::clamp(ratio, 0.0, 1.0), config_.adaptive_gamma);
    const double alpha = std::clamp(config_.adaptive_smoothing_alpha, 0.0, 1.0);
    state.current_target_fps = alpha * raw_fps + (1.0 - alpha) * state.current_target_fps;
    state.spectral_activity_ratio = ratio;
}

double OpenCvFrameSampler::EffectiveFps(const SessionState& state, double now_seconds) const noexcept {
    if ((now_seconds - state.last_spike_seconds) < config_.adaptive_spike_boost_seconds) {
        return config_.adaptive_fps_max;
    }
    return std::clamp(state.current_target_fps, config_.adaptive_fps_min, config_.adaptive_fps_max);
}

} // namespace media
