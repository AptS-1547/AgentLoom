#pragma once

#include "vision_runtime_interfaces.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace media {

struct OpenCvFrameSamplerConfig {
    int resize_width = 480;
    int gaussian_kernel_size = 5;
    int morphology_kernel_size = 5;
    int mog2_history = 120;
    double mog2_var_threshold = 24.0;
    bool mog2_detect_shadows = true;

    double min_component_area_ratio = 0.002;
    int temporal_vote_window = 5;
    int temporal_vote_required = 2;

    double area_weight = 0.5;
    double histogram_weight = 0.3;
    double edge_weight = 0.2;

    double area_sigmoid_center = 0.02;
    double area_sigmoid_scale = 0.01;
    double histogram_sigmoid_center = 0.2;
    double histogram_sigmoid_scale = 0.08;
    double edge_sigmoid_center = 0.02;
    double edge_sigmoid_scale = 0.01;

    double ema_alpha = 0.35;
    double peak_threshold = 0.55;
    double cooldown_seconds = 1.5;
    int canny_threshold1 = 50;
    int canny_threshold2 = 150;

    bool adaptive_enabled = false;
    int adaptive_stft_window_size = 32;
    int adaptive_stft_hop_size = 4;
    double adaptive_highfreq_cutoff_ratio = 0.3;
    double adaptive_fps_min = 4.0;
    double adaptive_fps_max = 15.0;
    double adaptive_gamma = 0.7;
    double adaptive_spike_threshold = 0.4;
    double adaptive_spike_boost_seconds = 2.0;
    double adaptive_smoothing_alpha = 0.3;
    double adaptive_precheck_diff_threshold = 15.0;
    int adaptive_precheck_resize_width = 160;
};

class OpenCvFrameSampler final : public IFrameSampler {
public:
    explicit OpenCvFrameSampler(OpenCvFrameSamplerConfig config = {});

    core::Result<FrameSamplingDecision> Evaluate(const VideoFrameView& frame) override;

private:
    struct SessionState {
        cv::Ptr<cv::BackgroundSubtractor> subtractor;
        cv::Mat morphology_kernel;
        std::deque<bool> temporal_votes;
        cv::Mat previous_histogram;
        double previous_edge_density = 0.0;
        bool has_previous_histogram = false;
        bool has_previous_edge_density = false;
        double previous_smoothed_score = 0.0;
        double last_peak_seconds = -1.0e9;
        std::deque<double> score_buffer;
        double current_target_fps = 15.0;
        double last_spike_seconds = -1.0e9;
        double last_sampled_seconds = -1.0e9;
        int frames_since_last_fft = 0;
        double spectral_activity_ratio = 0.0;
        cv::Mat precheck_ref_gray;
    };

    core::Result<cv::Mat> BuildRgbMat(const VideoFrameView& frame) const;
    SessionState MakeSessionState() const;
    static int OddKernel(int value) noexcept;
    static double SigmoidNormalize(double value, double center, double scale) noexcept;
    static double FilterSmallComponents(const cv::Mat& mask, double min_area_ratio);
    bool ShouldRunHeavyAnalysis(SessionState& state, const cv::Mat& rgb, double now_seconds) const;
    void MarkAdaptiveSampled(SessionState& state, const cv::Mat& rgb, double now_seconds) const;
    void UpdateAdaptive(SessionState& state, double change_score, double smoothed_score, double now_seconds) const;
    void RecomputeFft(SessionState& state) const;
    double EffectiveFps(const SessionState& state, double now_seconds) const noexcept;

    OpenCvFrameSamplerConfig config_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, SessionState> sessions_;
};

} // namespace media
