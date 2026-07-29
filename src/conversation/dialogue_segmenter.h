#pragma once

#include "optimizer.h"
#include "result.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace vector {
class EmbeddingPipeline;
}

namespace agent::conversation {

inline constexpr std::size_t kBoundaryFeatureCount = 5;

enum class BoundaryFeature : std::size_t {
    SemanticSimilarity = 0,
    TimeContinuity,
    SpeakerChanged,
    PreviousIsQuestion,
    CurrentIsShort,
};

struct BoundaryFeatures {
    std::array<double, kBoundaryFeatureCount> values{};

    double operator[](BoundaryFeature feature) const noexcept {
        return values[static_cast<std::size_t>(feature)];
    }
};

struct BoundaryTrainingExample {
    BoundaryFeatures features;
    double boundary_label = 0.0;
    double sample_weight = 1.0;
};

class IBoundaryPenaltyModel {
public:
    virtual ~IBoundaryPenaltyModel() = default;

    virtual double BoundaryProbability(
        const BoundaryFeatures& features) const noexcept = 0;
    virtual double PenaltyAdjustment(
        const BoundaryFeatures& features) const noexcept = 0;
};

struct LinearBoundaryLogitOptions {
    double penalty_adjustment_scale = 0.2;
    double probability_epsilon = 1e-6;
};

class LinearBoundaryLogitModel final : public IBoundaryPenaltyModel {
public:
    static core::Result<std::shared_ptr<LinearBoundaryLogitModel>> Create(
        LinearBoundaryLogitOptions options = {});

    double BoundaryProbability(
        const BoundaryFeatures& features) const noexcept override;
    double PenaltyAdjustment(
        const BoundaryFeatures& features) const noexcept override;

    core::Result<double> TrainBatch(
        std::span<const BoundaryTrainingExample> examples,
        core::optimization::IOptimizer& optimizer);
    core::Status LoadParameters(std::span<const double> parameters);

    std::span<const double> Parameters() const noexcept;

private:
    explicit LinearBoundaryLogitModel(LinearBoundaryLogitOptions options) noexcept;

    static constexpr std::size_t kParameterCount = kBoundaryFeatureCount + 1;
    LinearBoundaryLogitOptions options_;
    std::array<double, kParameterCount> parameters_{};
};

struct DialogueTurn {
    std::string turn_id;
    std::string speaker;
    std::string text;
    // Optional pre-rendered context for embedding. Empty means using text.
    std::string embedding_text;
    std::int64_t timestamp_us = 0;
    std::map<std::string, std::string> metadata;
};

// Embedding spans must be L2-normalized and have identical dimensions.
core::Result<BoundaryFeatures> ExtractBoundaryFeatures(
    const DialogueTurn& previous,
    const DialogueTurn& current,
    std::span<const float> previous_embedding,
    std::span<const float> current_embedding,
    double clock_half_life_seconds);

class ITextEmbeddingProvider {
public:
    virtual ~ITextEmbeddingProvider() = default;

    // text uses UTF-8 and remains borrowed for the duration of the call.
    virtual core::Result<std::vector<float>> EmbedText(std::string_view text) = 0;

    struct Batch {
        std::size_t batch_size = 0;
        std::size_t dimension = 0;
        std::vector<float> embeddings;

        std::span<const float> row(std::size_t index) const noexcept {
            if (index >= batch_size || dimension == 0) {
                return {};
            }
            return std::span<const float>(
                embeddings.data() + index * dimension,
                dimension);
        }
    };

    // Compatibility fallback for lightweight providers. Production embedding
    // adapters should override this with one tokenizer/model batch call.
    virtual core::Result<Batch> EmbedBatch(
        std::span<const std::string_view> texts);

    virtual bool Normalized() const noexcept { return false; }
};

class EmbeddingPipelineProvider final : public ITextEmbeddingProvider {
public:
    explicit EmbeddingPipelineProvider(
        std::shared_ptr<::vector::EmbeddingPipeline> pipeline);

    core::Result<std::vector<float>> EmbedText(std::string_view text) override;
    core::Result<Batch> EmbedBatch(
        std::span<const std::string_view> texts) override;
    bool Normalized() const noexcept override;

private:
    std::shared_ptr<::vector::EmbeddingPipeline> pipeline_;
};

struct DialogueSegmenterOptions {
    std::size_t min_block_turns = 1;
    std::size_t max_block_turns = 16;
    std::size_t embedding_batch_size = 32;
    double boundary_penalty = 0.9;
    double time_continuity_bonus = 0.1;
    double short_block_penalty = 0.25;
    double turn_half_life = 8.0;
    double clock_half_life_seconds = 300.0;
    std::string context_mode = "current_turn";
    std::string model_version = "bounded_dp_v1";
    std::string params_version = "bounded_dp_default_v1";
    std::shared_ptr<const IBoundaryPenaltyModel> boundary_penalty_model;
};

struct DialogueBlock {
    std::string block_id;
    std::string session_id;
    std::size_t owned_begin = 0;
    std::size_t owned_end = 0;
    std::vector<std::string> turn_ids;
    std::int64_t start_time_us = 0;
    std::int64_t end_time_us = 0;
    double segmentation_cost = 0.0;
    std::string segmentation_model_version;
    std::string segmentation_params_version;
    std::string context_mode;
};

struct DialogueSegmentationResult {
    std::string session_id;
    std::vector<DialogueBlock> blocks;
    double total_cost = 0.0;
    std::string model_version;
    std::string params_version;
};

class IDialogueSegmenter {
public:
    virtual ~IDialogueSegmenter() = default;

    virtual core::Result<DialogueSegmentationResult> Segment(
        std::string_view session_id,
        const std::vector<DialogueTurn>& turns) = 0;
};

class BoundedDpDialogueSegmenter final : public IDialogueSegmenter {
public:
    BoundedDpDialogueSegmenter(
        std::shared_ptr<ITextEmbeddingProvider> embedding_provider,
        DialogueSegmenterOptions options = {});

    core::Result<DialogueSegmentationResult> Segment(
        std::string_view session_id,
        const std::vector<DialogueTurn>& turns) override;

private:
    std::shared_ptr<ITextEmbeddingProvider> embedding_provider_;
    DialogueSegmenterOptions options_;
};

} // namespace agent::conversation
