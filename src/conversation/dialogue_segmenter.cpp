#include "dialogue_segmenter.h"

#include "async_embedding.h"
#include "embedding_pipeline.h"
#include "semantic_cache_pipeline.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <utility>

#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

namespace agent::conversation {
namespace {

float DotProduct(const float* lhs, const float* rhs, std::size_t dimension) noexcept {
    switch (dimension) {
    case 128:
        return dot_product_unrolled<128>(lhs, rhs);
    case 256:
        return dot_product_unrolled<256>(lhs, rhs);
    case agent::semantic_cache::kExpectedEmbeddingDim:
        return dot_product_unrolled<agent::semantic_cache::kExpectedEmbeddingDim>(lhs, rhs);
    case 512:
        return dot_product_unrolled<512>(lhs, rhs);
    case 768:
        return dot_product_unrolled<768>(lhs, rhs);
    case 1024:
        return dot_product_unrolled<1024>(lhs, rhs);
    default:
        return std::inner_product(lhs, lhs + dimension, rhs, 0.0f);
    }
}

core::Status ValidateOptions(const DialogueSegmenterOptions& options) {
    if (options.min_block_turns == 0 || options.max_block_turns == 0 ||
        options.min_block_turns > options.max_block_turns ||
        options.embedding_batch_size == 0) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "dialogue segmentation size options are invalid");
    }
    if (!std::isfinite(options.boundary_penalty) || options.boundary_penalty < 0.0 ||
        !std::isfinite(options.time_continuity_bonus) || options.time_continuity_bonus < 0.0 ||
        !std::isfinite(options.short_block_penalty) || options.short_block_penalty < 0.0 ||
        !std::isfinite(options.turn_half_life) || options.turn_half_life <= 0.0 ||
        !std::isfinite(options.clock_half_life_seconds) || options.clock_half_life_seconds <= 0.0) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "dialogue segmentation parameters are invalid");
    }
    return core::Status::Ok();
}

// 归一化探测容差：模型 normalize=true 的输出平方范数应≈1；
// |‖v‖²−1| 超过该阈值即视为未归一化输入（仅异步路径开启探测）。
constexpr float kUnitNormSquaredTolerance = 1e-3f;

core::Status ValidateBatchRows(const ITextEmbeddingProvider::Batch& batch,
                               bool require_unit_norm = false) {
    for (std::size_t row_index = 0; row_index < batch.batch_size; ++row_index) {
        const auto* row = batch.embeddings.data() + row_index * batch.dimension;
        for (std::size_t component = 0; component < batch.dimension; ++component) {
            if (!std::isfinite(row[component])) {
                return core::Status::Error(
                    core::ErrorCode::InvalidArgument,
                    "embedding contains a non-finite value");
            }
        }
        const auto squared_norm = DotProduct(row, row, batch.dimension);
        if (!std::isfinite(squared_norm) ||
            squared_norm <= std::numeric_limits<float>::epsilon()) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "embedding norm is zero");
        }
        if (require_unit_norm &&
            std::fabs(squared_norm - 1.0f) > kUnitNormSquaredTolerance) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "embedding is not L2-normalized");
        }
    }
    return core::Status::Ok();
}

std::size_t Utf8CodePointCount(std::string_view text) noexcept {
    return static_cast<std::size_t>(std::count_if(
        text.begin(),
        text.end(),
        [](unsigned char value) { return (value & 0xC0U) != 0x80U; }));
}

bool EndsWithQuestionMark(std::string_view text) noexcept {
    while (!text.empty() &&
           (text.back() == ' ' || text.back() == '\t' ||
            text.back() == '\r' || text.back() == '\n')) {
        text.remove_suffix(1);
    }
    return text.ends_with("?") || text.ends_with("？");
}

double TimeContinuity(
    const DialogueTurn& previous,
    const DialogueTurn& current,
    double half_life_seconds) noexcept {
    if (previous.timestamp_us <= 0 || current.timestamp_us < previous.timestamp_us) {
        return 1.0;
    }
    const auto gap_seconds = static_cast<double>(
        current.timestamp_us - previous.timestamp_us) / 1'000'000.0;
    return std::exp(-std::numbers::ln2 * gap_seconds / half_life_seconds);
}

BoundaryFeatures BuildBoundaryFeaturesUnchecked(
    const DialogueTurn& previous,
    const DialogueTurn& current,
    const float* previous_embedding,
    const float* current_embedding,
    std::size_t dimension,
    double clock_half_life_seconds) noexcept {
    BoundaryFeatures features;
    features.values[static_cast<std::size_t>(BoundaryFeature::SemanticSimilarity)] =
        std::clamp(
            static_cast<double>(DotProduct(previous_embedding, current_embedding, dimension)),
            -1.0,
            1.0);
    features.values[static_cast<std::size_t>(BoundaryFeature::TimeContinuity)] =
        TimeContinuity(previous, current, clock_half_life_seconds);
    features.values[static_cast<std::size_t>(BoundaryFeature::SpeakerChanged)] =
        previous.speaker != current.speaker ? 1.0 : 0.0;
    features.values[static_cast<std::size_t>(BoundaryFeature::PreviousIsQuestion)] =
        EndsWithQuestionMark(previous.text) ? 1.0 : 0.0;
    features.values[static_cast<std::size_t>(BoundaryFeature::CurrentIsShort)] =
        Utf8CodePointCount(current.text) <= 8 ? 1.0 : 0.0;
    return features;
}

} // namespace

core::Result<BoundaryFeatures> ExtractBoundaryFeatures(
    const DialogueTurn& previous,
    const DialogueTurn& current,
    std::span<const float> previous_embedding,
    std::span<const float> current_embedding,
    double clock_half_life_seconds) {
    if (previous_embedding.empty() ||
        previous_embedding.size() != current_embedding.size() ||
        !std::isfinite(clock_half_life_seconds) ||
        clock_half_life_seconds <= 0.0) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "boundary feature inputs are invalid");
    }
    for (std::size_t index = 0; index < previous_embedding.size(); ++index) {
        if (!std::isfinite(previous_embedding[index]) ||
            !std::isfinite(current_embedding[index])) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "boundary feature embeddings must be finite");
        }
    }
    return BuildBoundaryFeaturesUnchecked(
        previous,
        current,
        previous_embedding.data(),
        current_embedding.data(),
        previous_embedding.size(),
        clock_half_life_seconds);
}

LinearBoundaryLogitModel::LinearBoundaryLogitModel(
    LinearBoundaryLogitOptions options) noexcept
    : options_(std::move(options)) {}

core::Result<std::shared_ptr<LinearBoundaryLogitModel>> LinearBoundaryLogitModel::Create(
    LinearBoundaryLogitOptions options) {
    if (!std::isfinite(options.penalty_adjustment_scale) ||
        options.penalty_adjustment_scale < 0.0 ||
        !std::isfinite(options.probability_epsilon) ||
        options.probability_epsilon <= 0.0 ||
        options.probability_epsilon >= 0.5) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "linear boundary Logit options are invalid");
    }
    return std::shared_ptr<LinearBoundaryLogitModel>(
        new LinearBoundaryLogitModel(std::move(options)));
}

double LinearBoundaryLogitModel::BoundaryProbability(
    const BoundaryFeatures& features) const noexcept {
    double logit = parameters_[0];
    for (std::size_t index = 0; index < kBoundaryFeatureCount; ++index) {
        logit += parameters_[index + 1] * features.values[index];
    }
    if (logit >= 0.0) {
        return 1.0 / (1.0 + std::exp(-logit));
    }
    const auto exponential = std::exp(logit);
    return exponential / (1.0 + exponential);
}

double LinearBoundaryLogitModel::PenaltyAdjustment(
    const BoundaryFeatures& features) const noexcept {
    const auto probability = std::clamp(
        BoundaryProbability(features),
        options_.probability_epsilon,
        1.0 - options_.probability_epsilon);
    return options_.penalty_adjustment_scale * (1.0 - 2.0 * probability);
}

core::Result<double> LinearBoundaryLogitModel::TrainBatch(
    std::span<const BoundaryTrainingExample> examples,
    core::optimization::IOptimizer& optimizer) {
    if (examples.empty()) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "linear boundary Logit training batch is empty");
    }
    using Derivatives = Eigen::Matrix<double, kParameterCount, 1>;
    using AutoDiff = Eigen::AutoDiffScalar<Derivatives>;
    std::array<AutoDiff, kParameterCount> parameters;
    for (std::size_t index = 0; index < kParameterCount; ++index) {
        parameters[index].value() = parameters_[index];
        parameters[index].derivatives() = Derivatives::Unit(
            static_cast<Eigen::Index>(index));
    }

    AutoDiff total_loss = 0.0;
    double total_weight = 0.0;
    for (const auto& example : examples) {
        if (example.boundary_label < 0.0 || example.boundary_label > 1.0 ||
            !std::isfinite(example.boundary_label) ||
            !std::isfinite(example.sample_weight) || example.sample_weight <= 0.0 ||
            !std::all_of(
                example.features.values.begin(),
                example.features.values.end(),
                [](double value) { return std::isfinite(value); })) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "linear boundary Logit training example is invalid");
        }
        AutoDiff logit = parameters[0];
        for (std::size_t index = 0; index < kBoundaryFeatureCount; ++index) {
            logit += parameters[index + 1] * example.features.values[index];
        }
        using std::exp;
        using std::log;
        if (logit.value() >= 0.0) {
            const AutoDiff correction = log(1.0 + exp(-logit));
            total_loss += example.sample_weight *
                (logit + correction - example.boundary_label * logit);
        } else {
            const AutoDiff softplus = log(1.0 + exp(logit));
            total_loss += example.sample_weight *
                (softplus - example.boundary_label * logit);
        }
        total_weight += example.sample_weight;
    }
    total_loss /= total_weight;

    std::array<double, kParameterCount> gradients{};
    for (std::size_t index = 0; index < kParameterCount; ++index) {
        gradients[index] = total_loss.derivatives()(static_cast<Eigen::Index>(index));
    }
    auto status = optimizer.Step(parameters_, gradients);
    if (!status.ok()) {
        return status;
    }
    return total_loss.value();
}

core::Status LinearBoundaryLogitModel::LoadParameters(
    std::span<const double> parameters) {
    if (parameters.size() != parameters_.size() ||
        !std::all_of(
            parameters.begin(),
            parameters.end(),
            [](double value) { return std::isfinite(value); })) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "linear boundary Logit parameters are invalid");
    }
    std::copy(parameters.begin(), parameters.end(), parameters_.begin());
    return core::Status::Ok();
}

std::span<const double> LinearBoundaryLogitModel::Parameters() const noexcept {
    return parameters_;
}

core::Result<ITextEmbeddingProvider::Batch> ITextEmbeddingProvider::EmbedBatch(
    std::span<const std::string_view> texts) {
    if (texts.empty()) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "embedding batch is empty");
    }

    Batch batch;
    batch.batch_size = texts.size();
    for (const auto text : texts) {
        auto embedded = EmbedText(text);
        if (!embedded.ok()) {
            return embedded.status();
        }
        auto row = std::move(embedded).value();
        if (batch.dimension == 0) {
            batch.dimension = row.size();
            if (batch.dimension == 0) {
                return core::Status::Error(
                    core::ErrorCode::InvalidArgument,
                    "embedding is empty");
            }
            if (batch.batch_size >
                std::numeric_limits<std::size_t>::max() / batch.dimension) {
                return core::Status::Error(
                    core::ErrorCode::ResourceExhausted,
                    "embedding batch is too large");
            }
            batch.embeddings.reserve(batch.batch_size * batch.dimension);
        } else if (row.size() != batch.dimension) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "embedding dimensions do not match");
        }
        batch.embeddings.insert(batch.embeddings.end(), row.begin(), row.end());
    }
    return batch;
}

EmbeddingPipelineProvider::EmbeddingPipelineProvider(
    std::shared_ptr<::vector::EmbeddingPipeline> pipeline)
    : pipeline_(std::move(pipeline)) {}

core::Result<std::vector<float>> EmbeddingPipelineProvider::EmbedText(
    std::string_view text) {
    if (!pipeline_) {
        return core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "embedding pipeline is required");
    }
    return pipeline_->Encode(text);
}

core::Result<ITextEmbeddingProvider::Batch> EmbeddingPipelineProvider::EmbedBatch(
    std::span<const std::string_view> texts) {
    if (!pipeline_) {
        return core::Status::Error(
            core::ErrorCode::FailedPrecondition,
            "embedding pipeline is required");
    }
    auto encoded = pipeline_->EncodeBatch(texts);
    if (!encoded.ok()) {
        return encoded.status();
    }
    auto source = std::move(encoded).value();
    Batch batch;
    batch.batch_size = source.batch_size;
    batch.dimension = source.dimension;
    batch.embeddings = std::move(source.embeddings);
    return batch;
}

bool EmbeddingPipelineProvider::Normalized() const noexcept {
    return pipeline_ && pipeline_->Normalized();
}

BoundedDpDialogueSegmenter::BoundedDpDialogueSegmenter(
    std::shared_ptr<ITextEmbeddingProvider> embedding_provider,
    DialogueSegmenterOptions options)
    : embedding_provider_(std::move(embedding_provider)),
      options_(std::move(options)) {}

BoundedDpDialogueSegmenter::BoundedDpDialogueSegmenter(
    std::shared_ptr<vector::EmbeddingBatchCoordinator> coordinator,
    DialogueSegmenterOptions options)
    : coordinator_(std::move(coordinator)),
      options_(std::move(options)) {}

core::Result<DialogueSegmentationResult> BoundedDpDialogueSegmenter::Segment(
    std::string_view session_id,
    const std::vector<DialogueTurn>& turns) {
    auto options_status = ValidateOptions(options_);
    if (!options_status.ok()) {
        return options_status;
    }
    if (session_id.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "session_id is required");
    }
    if (turns.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "dialogue turns are empty");
    }
    if (!embedding_provider_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "embedding provider is required");
    }

    std::vector<float> embeddings;
    std::size_t dimension = 0;
    std::vector<std::string_view> batch_texts;
    batch_texts.reserve(std::min(options_.embedding_batch_size, turns.size()));
    const bool provider_normalized = embedding_provider_->Normalized();
    for (std::size_t batch_begin = 0; batch_begin < turns.size();
         batch_begin += options_.embedding_batch_size) {
        const auto batch_end = std::min(
            batch_begin + options_.embedding_batch_size,
            turns.size());
        batch_texts.clear();
        for (std::size_t index = batch_begin; index < batch_end; ++index) {
            const auto& turn = turns[index];
            if (turn.turn_id.empty()) {
                return core::Status::Error(core::ErrorCode::InvalidArgument, "dialogue turn_id is required");
            }
            if (turn.text.empty()) {
                return core::Status::Error(core::ErrorCode::InvalidArgument, "dialogue turn text is empty");
            }
            batch_texts.push_back(turn.embedding_text.empty() ? turn.text : turn.embedding_text);
        }

        auto embedded = embedding_provider_->EmbedBatch(batch_texts);
        if (!embedded.ok()) {
            return embedded.status();
        }
        auto batch = std::move(embedded).value();
        if (batch.batch_size != batch_texts.size() || batch.dimension == 0 ||
            batch.batch_size > std::numeric_limits<std::size_t>::max() / batch.dimension ||
            batch.embeddings.size() != batch.batch_size * batch.dimension) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "embedding batch shape does not match dialogue input");
        }
        if (dimension == 0) {
            dimension = batch.dimension;
            if (turns.size() > std::numeric_limits<std::size_t>::max() / dimension) {
                return core::Status::Error(
                    core::ErrorCode::ResourceExhausted,
                    "dialogue embedding matrix is too large");
            }
            embeddings.reserve(turns.size() * dimension);
        } else if (dimension != batch.dimension) {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "embedding dimensions do not match");
        }

        auto rows_status = ValidateBatchRows(batch);
        if (!rows_status.ok()) {
            return rows_status;
        }
        if (!provider_normalized) {
            ::vector::pooling::L2NormalizeRows(
                batch.embeddings.data(),
                batch.batch_size,
                batch.dimension);
        }
        embeddings.insert(
            embeddings.end(),
            batch.embeddings.begin(),
            batch.embeddings.end());
    }

    return RunDp(session_id, turns, embeddings, dimension);
}

core::Result<DialogueSegmentationResult> BoundedDpDialogueSegmenter::RunDp(
    std::string_view session_id,
    const std::vector<DialogueTurn>& turns,
    const std::vector<float>& embeddings,
    std::size_t dimension) const {
    const auto count = turns.size();
    if (embeddings.size() != count * dimension) {
        return core::Status::Error(
            core::ErrorCode::InternalError,
            "dialogue embedding matrix is incomplete");
    }

    const auto invalid = std::numeric_limits<double>::infinity();
    std::vector<double> dp(count + 1, invalid);
    std::vector<std::size_t> previous(count + 1, 0);
    std::vector<double> block_cost(count + 1, 0.0);
    dp[0] = 0.0;

    std::vector<double> time_prefix(count + 1, 0.0);
    std::vector<double> local_boundary_penalties(count, options_.boundary_penalty);
    for (std::size_t index = 1; index < count; ++index) {
        const auto turn_distance = 1.0;
        const auto clock_distance = turns[index - 1].timestamp_us > 0 && turns[index].timestamp_us >= turns[index - 1].timestamp_us
            ? static_cast<double>(turns[index].timestamp_us - turns[index - 1].timestamp_us) / 1'000'000.0
            : 0.0;
        time_prefix[index + 1] = time_prefix[index] +
            turn_distance / options_.turn_half_life +
            clock_distance / options_.clock_half_life_seconds;
        const auto features = BuildBoundaryFeaturesUnchecked(
            turns[index - 1],
            turns[index],
            embeddings.data() + (index - 1) * dimension,
            embeddings.data() + index * dimension,
            dimension,
            options_.clock_half_life_seconds);
        local_boundary_penalties[index] += options_.time_continuity_bonus *
            features[BoundaryFeature::TimeContinuity];
        if (options_.boundary_penalty_model) {
            local_boundary_penalties[index] +=
                options_.boundary_penalty_model->PenaltyAdjustment(features);
        }
        local_boundary_penalties[index] = std::max(0.0, local_boundary_penalties[index]);
    }

    auto interval_cost = [&](std::size_t begin, std::size_t end, double squared_sum_norm) {
        const auto length = end - begin;
        double cost = static_cast<double>(length) - squared_sum_norm / static_cast<double>(length);
        cost += time_prefix[end] - time_prefix[begin + 1];
        if (length < options_.min_block_turns) {
            cost += options_.short_block_penalty;
        }
        return cost;
    };

    std::vector<float> interval_sum(dimension, 0.0f);
    for (std::size_t end = 1; end <= count; ++end) {
        std::fill(interval_sum.begin(), interval_sum.end(), 0.0f);
        const auto first_begin = end > options_.max_block_turns ? end - options_.max_block_turns : 0;
        for (std::size_t begin = end; begin > first_begin;) {
            --begin;
            const auto* embedding_row = embeddings.data() + begin * dimension;
            for (std::size_t component = 0; component < dimension; ++component) {
                interval_sum[component] += embedding_row[component];
            }
            if (dp[begin] == invalid) {
                continue;
            }
            const auto squared_sum_norm = std::max(
                0.0,
                static_cast<double>(DotProduct(
                    interval_sum.data(),
                    interval_sum.data(),
                    dimension)));
            const auto cost = interval_cost(begin, end, squared_sum_norm);
            const auto candidate = dp[begin] + cost +
                (begin == 0 ? 0.0 : local_boundary_penalties[begin]);
            if (candidate < dp[end] ||
                (candidate == dp[end] && begin < previous[end])) {
                dp[end] = candidate;
                previous[end] = begin;
                block_cost[end] = cost;
            }
        }
    }
    if (!std::isfinite(dp[count]) || previous[count] == count) {
        return core::Status::Error(core::ErrorCode::InternalError, "dialogue DP could not produce a segmentation");
    }

    std::vector<std::pair<std::size_t, std::size_t>> ranges;
    for (std::size_t end = count; end > 0;) {
        const auto begin = previous[end];
        ranges.emplace_back(begin, end);
        end = begin;
    }
    std::reverse(ranges.begin(), ranges.end());

    DialogueSegmentationResult result;
    result.session_id = std::string(session_id);
    result.total_cost = dp[count];
    result.model_version = options_.model_version;
    result.params_version = options_.params_version;
    result.blocks.reserve(ranges.size());
    for (std::size_t block_index = 0; block_index < ranges.size(); ++block_index) {
        const auto [begin, end] = ranges[block_index];
        DialogueBlock block;
        block.block_id = std::string(session_id) + ":block-" + std::to_string(block_index);
        block.session_id = std::string(session_id);
        block.owned_begin = begin;
        block.owned_end = end;
        block.start_time_us = turns[begin].timestamp_us;
        block.end_time_us = turns[end - 1].timestamp_us;
        block.segmentation_cost = block_cost[end];
        block.segmentation_model_version = options_.model_version;
        block.segmentation_params_version = options_.params_version;
        block.context_mode = options_.context_mode;
        block.turn_ids.reserve(end - begin);
        for (std::size_t index = begin; index < end; ++index) {
            block.turn_ids.push_back(turns[index].turn_id);
        }
        result.blocks.push_back(std::move(block));
    }
    return result;
}

core::async::task<core::Result<DialogueSegmentationResult>>
BoundedDpDialogueSegmenter::SegmentAsync(
    std::string_view session_id,
    const std::vector<DialogueTurn>& turns) {
    auto options_status = ValidateOptions(options_);
    if (!options_status.ok()) {
        co_return options_status;
    }
    if (session_id.empty()) {
        co_return core::Status::Error(core::ErrorCode::InvalidArgument, "session_id is required");
    }
    if (turns.empty()) {
        co_return core::Status::Error(core::ErrorCode::InvalidArgument, "dialogue turns are empty");
    }
    if (!coordinator_) {
        co_return core::Status::Error(core::ErrorCode::FailedPrecondition, "embedding coordinator is required");
    }

    std::vector<float> embeddings;
    std::size_t dimension = 0;
    std::vector<std::string_view> batch_texts;
    batch_texts.reserve(std::min(options_.embedding_batch_size, turns.size()));
    for (std::size_t batch_begin = 0; batch_begin < turns.size();
         batch_begin += options_.embedding_batch_size) {
        const auto batch_end = std::min(
            batch_begin + options_.embedding_batch_size, turns.size());
        batch_texts.clear();
        for (std::size_t index = batch_begin; index < batch_end; ++index) {
            const auto& turn = turns[index];
            if (turn.turn_id.empty()) {
                co_return core::Status::Error(core::ErrorCode::InvalidArgument, "dialogue turn_id is required");
            }
            if (turn.text.empty()) {
                co_return core::Status::Error(core::ErrorCode::InvalidArgument, "dialogue turn text is empty");
            }
            batch_texts.push_back(turn.embedding_text.empty() ? turn.text : turn.embedding_text);
        }

        auto embedded = co_await EmbedBatchAsync(*coordinator_, batch_texts, session_id);
        if (!embedded.ok()) {
            co_return embedded.status();
        }
        auto batch = std::move(embedded).value();
        if (batch.batch_size != batch_texts.size() || batch.dimension == 0 ||
            batch.batch_size > std::numeric_limits<std::size_t>::max() / batch.dimension ||
            batch.embeddings.size() != batch.batch_size * batch.dimension) {
            co_return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "embedding batch shape does not match dialogue input");
        }
        if (dimension == 0) {
            dimension = batch.dimension;
            if (turns.size() > std::numeric_limits<std::size_t>::max() / dimension) {
                co_return core::Status::Error(
                    core::ErrorCode::ResourceExhausted,
                    "dialogue embedding matrix is too large");
            }
            embeddings.reserve(turns.size() * dimension);
        } else if (dimension != batch.dimension) {
            co_return core::Status::Error(core::ErrorCode::InvalidArgument, "embedding dimensions do not match");
        }

        // 异步路径不信任 provider 的归一化声明，改为运行时探测每行为单位范数；
        // 非归一化输入直接拒绝，而不是像同步 Segment 那样静默 L2 归一化。
        auto rows_status = ValidateBatchRows(batch, /*require_unit_norm=*/true);
        if (!rows_status.ok()) {
            co_return rows_status;
        }
        embeddings.insert(
            embeddings.end(),
            batch.embeddings.begin(),
            batch.embeddings.end());
    }

    co_return RunDp(session_id, turns, embeddings, dimension);
}

} // namespace agent::conversation
