#include "dialogue_segmenter.h"
#include "embedding_pipeline.h"
#include "hf_tokenizer.h"
#include "onnx_text_embedding_model.h"
#include "semantic_cache_pipeline.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
using Json = nlohmann::json;

namespace {

struct BoundaryCounts {
    std::size_t reference = 0;
    std::size_t predicted = 0;
    std::size_t exact_matches = 0;
    std::size_t relaxed_matches = 0;
    std::size_t pk_errors = 0;
    std::size_t pk_windows = 0;
    std::size_t window_diff_errors = 0;
    std::size_t window_diff_windows = 0;
};

double SafeRatio(std::size_t numerator, std::size_t denominator) {
    return denominator == 0
        ? 0.0
        : static_cast<double>(numerator) / static_cast<double>(denominator);
}

double F1(double precision, double recall) {
    return precision + recall == 0.0
        ? 0.0
        : 2.0 * precision * recall / (precision + recall);
}

std::vector<std::size_t> BoundaryPositions(std::span<const std::uint8_t> boundaries) {
    std::vector<std::size_t> positions;
    for (std::size_t index = 0; index < boundaries.size(); ++index) {
        if (boundaries[index] != 0) {
            positions.push_back(index);
        }
    }
    return positions;
}

std::size_t RelaxedMatches(
    std::span<const std::size_t> reference,
    std::span<const std::size_t> predicted,
    std::size_t tolerance) {
    std::size_t matched = 0;
    std::size_t reference_index = 0;
    for (const auto prediction : predicted) {
        while (reference_index < reference.size() &&
               reference[reference_index] + tolerance < prediction) {
            ++reference_index;
        }
        if (reference_index < reference.size() &&
            reference[reference_index] <= prediction + tolerance) {
            ++matched;
            ++reference_index;
        }
    }
    return matched;
}

std::vector<std::size_t> SegmentIds(std::span<const std::uint8_t> boundaries) {
    std::vector<std::size_t> ids(boundaries.size() + 1, 0);
    std::size_t segment = 0;
    for (std::size_t index = 0; index < ids.size(); ++index) {
        ids[index] = segment;
        if (index < boundaries.size() && boundaries[index] != 0) {
            ++segment;
        }
    }
    return ids;
}

BoundaryCounts Evaluate(
    std::span<const std::uint8_t> reference,
    std::span<const std::uint8_t> predicted) {
    if (reference.size() != predicted.size()) {
        throw std::runtime_error("reference/predicted boundary sizes differ");
    }

    BoundaryCounts result;
    const auto reference_positions = BoundaryPositions(reference);
    const auto predicted_positions = BoundaryPositions(predicted);
    result.reference = reference_positions.size();
    result.predicted = predicted_positions.size();
    result.exact_matches = std::count_if(
        predicted_positions.begin(),
        predicted_positions.end(),
        [&](const auto position) { return reference[position] != 0; });
    result.relaxed_matches = RelaxedMatches(reference_positions, predicted_positions, 1);

    const auto turn_count = reference.size() + 1;
    if (turn_count < 2) {
        return result;
    }
    const auto segment_count = reference_positions.size() + 1;
    const auto average_segment_length = static_cast<double>(turn_count) /
        static_cast<double>(segment_count);
    const auto window = std::clamp<std::size_t>(
        static_cast<std::size_t>(std::llround(average_segment_length / 2.0)),
        1,
        turn_count - 1);
    const auto reference_ids = SegmentIds(reference);
    const auto predicted_ids = SegmentIds(predicted);
    for (std::size_t begin = 0; begin + window < turn_count; ++begin) {
        const auto end = begin + window;
        const bool reference_same = reference_ids[begin] == reference_ids[end];
        const bool predicted_same = predicted_ids[begin] == predicted_ids[end];
        result.pk_errors += reference_same != predicted_same ? 1 : 0;
        ++result.pk_windows;

        std::size_t reference_in_window = 0;
        std::size_t predicted_in_window = 0;
        for (std::size_t boundary = begin; boundary < end; ++boundary) {
            reference_in_window += reference[boundary] != 0 ? 1 : 0;
            predicted_in_window += predicted[boundary] != 0 ? 1 : 0;
        }
        result.window_diff_errors += reference_in_window != predicted_in_window ? 1 : 0;
        ++result.window_diff_windows;
    }
    return result;
}

void Add(BoundaryCounts& total, const BoundaryCounts& value) {
    total.reference += value.reference;
    total.predicted += value.predicted;
    total.exact_matches += value.exact_matches;
    total.relaxed_matches += value.relaxed_matches;
    total.pk_errors += value.pk_errors;
    total.pk_windows += value.pk_windows;
    total.window_diff_errors += value.window_diff_errors;
    total.window_diff_windows += value.window_diff_windows;
}

Json ReadJson(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open dataset: " + path.string());
    }
    Json root;
    input >> root;
    return root;
}

std::string PathUtf8(const fs::path& path) {
    const auto value = path.generic_u8string();
    return std::string(
        reinterpret_cast<const char*>(value.data()),
        value.size());
}

std::vector<std::uint8_t> ReferenceBoundaries(const Json& turns) {
    if (turns.size() < 2) {
        return {};
    }
    std::vector<std::uint8_t> boundaries(turns.size() - 1, 0);
    for (std::size_t index = 0; index < boundaries.size(); ++index) {
        boundaries[index] = turns[index].value("segmentation_label", 0) == 1 ? 1 : 0;
    }
    return boundaries;
}

std::vector<agent::conversation::DialogueTurn> ParseTurns(
    std::string_view dataset,
    const Json& dialogue,
    std::size_t dialogue_index) {
    const auto& json_turns = dialogue.at("turns");
    std::vector<agent::conversation::DialogueTurn> turns;
    turns.reserve(json_turns.size());
    const auto dialogue_id = dialogue.value(
        "dial_id",
        std::string(dataset) + "-" + std::to_string(dialogue_index));
    for (std::size_t turn_index = 0; turn_index < json_turns.size(); ++turn_index) {
        const auto& source = json_turns[turn_index];
        turns.push_back({
            .turn_id = dialogue_id + ":" + std::to_string(turn_index),
            .speaker = source.value("role", "unknown"),
            .text = source.at("utterance").get<std::string>(),
        });
    }
    return turns;
}

struct EmbeddedTurns {
    std::size_t dimension = 0;
    std::vector<float> values;
};

core::Result<EmbeddedTurns> EmbedTurns(
    agent::conversation::ITextEmbeddingProvider& provider,
    const std::vector<agent::conversation::DialogueTurn>& turns,
    std::size_t batch_size) {
    EmbeddedTurns embedded_turns;
    std::vector<std::string_view> texts;
    texts.reserve(std::min(batch_size, turns.size()));
    for (std::size_t batch_begin = 0; batch_begin < turns.size(); batch_begin += batch_size) {
        const auto batch_end = std::min(batch_begin + batch_size, turns.size());
        texts.clear();
        for (std::size_t index = batch_begin; index < batch_end; ++index) {
            texts.push_back(turns[index].text);
        }
        auto batch_result = provider.EmbedBatch(texts);
        if (!batch_result.ok()) {
            return batch_result.status();
        }
        auto batch = std::move(batch_result).value();
        if (batch.batch_size != texts.size() || batch.dimension == 0 ||
            batch.embeddings.size() != batch.batch_size * batch.dimension) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "training embedding batch shape is invalid");
        }
        if (embedded_turns.dimension == 0) {
            embedded_turns.dimension = batch.dimension;
        } else if (embedded_turns.dimension != batch.dimension) {
            return core::Status::Error(
                core::ErrorCode::InvalidArgument,
                "training embedding dimensions changed");
        }
        if (!provider.Normalized()) {
            vector::pooling::L2NormalizeRows(
                batch.embeddings.data(),
                batch.batch_size,
                batch.dimension);
        }
        embedded_turns.values.insert(
            embedded_turns.values.end(),
            batch.embeddings.begin(),
            batch.embeddings.end());
    }
    return embedded_turns;
}

core::Result<std::vector<agent::conversation::BoundaryTrainingExample>>
CollectBoundaryTrainingExamples(
    const Json& root,
    agent::conversation::ITextEmbeddingProvider& provider,
    const agent::conversation::DialogueSegmenterOptions& options) {
    const auto& dial_data = root.at("dial_data");
    if (!dial_data.is_object() || dial_data.size() != 1) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "training dial_data must contain exactly one dataset");
    }
    const auto dataset = dial_data.begin().key();
    const auto& dialogues = dial_data.begin().value();
    std::vector<agent::conversation::BoundaryTrainingExample> examples;
    for (std::size_t dialogue_index = 0; dialogue_index < dialogues.size(); ++dialogue_index) {
        const auto& dialogue = dialogues[dialogue_index];
        auto turns = ParseTurns(dataset, dialogue, dialogue_index);
        if (turns.size() < 2) {
            continue;
        }
        auto embedded_result = EmbedTurns(
            provider,
            turns,
            options.embedding_batch_size);
        if (!embedded_result.ok()) {
            return embedded_result.status();
        }
        auto embedded = std::move(embedded_result).value();
        const auto reference = ReferenceBoundaries(dialogue.at("turns"));
        for (std::size_t index = 1; index < turns.size(); ++index) {
            auto features = agent::conversation::ExtractBoundaryFeatures(
                turns[index - 1],
                turns[index],
                std::span<const float>(
                    embedded.values.data() + (index - 1) * embedded.dimension,
                    embedded.dimension),
                std::span<const float>(
                    embedded.values.data() + index * embedded.dimension,
                    embedded.dimension),
                options.clock_half_life_seconds);
            if (!features.ok()) {
                return features.status();
            }
            examples.push_back({
                .features = std::move(features).value(),
                .boundary_label = reference[index - 1] != 0 ? 1.0 : 0.0,
            });
        }
    }
    const auto positive_count = static_cast<std::size_t>(std::count_if(
        examples.begin(),
        examples.end(),
        [](const auto& example) { return example.boundary_label > 0.5; }));
    const auto negative_count = examples.size() - positive_count;
    if (positive_count == 0 || negative_count == 0) {
        return core::Status::Error(
            core::ErrorCode::InvalidArgument,
            "training boundaries must contain both classes");
    }
    const auto half = static_cast<double>(examples.size()) / 2.0;
    const auto positive_weight = half / static_cast<double>(positive_count);
    const auto negative_weight = half / static_cast<double>(negative_count);
    for (auto& example : examples) {
        example.sample_weight = example.boundary_label > 0.5
            ? positive_weight
            : negative_weight;
    }
    return examples;
}

std::vector<std::uint8_t> PredictedBoundaries(
    std::size_t turn_count,
    const agent::conversation::DialogueSegmentationResult& segmentation) {
    if (turn_count < 2) {
        return {};
    }
    std::vector<std::uint8_t> boundaries(turn_count - 1, 0);
    for (std::size_t index = 0; index + 1 < segmentation.blocks.size(); ++index) {
        const auto owned_end = segmentation.blocks[index].owned_end;
        if (owned_end == 0 || owned_end >= turn_count) {
            throw std::runtime_error("predicted block boundary is outside dialogue");
        }
        boundaries[owned_end - 1] = 1;
    }
    return boundaries;
}

int Run(int argc, char** argv) {
    if (argc < 4 || argc > 12) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset.json> <tokenizer.json> <model.onnx>"
                     " [max_dialogues] [execution_provider]"
                     " [logit_training_dataset|-] [logit_epochs]"
                     " [boundary_penalty] [time_continuity_bonus] [turn_half_life]"
                     " [logit_penalty_scale]\n";
        return 2;
    }

    const fs::path dataset_path = fs::u8path(argv[1]);
    const fs::path tokenizer_path = fs::u8path(argv[2]);
    const fs::path model_path = fs::u8path(argv[3]);
    const std::size_t max_dialogues = argc >= 5
        ? static_cast<std::size_t>(std::stoull(argv[4]))
        : 0;
    const std::string execution_provider = argc >= 6 ? argv[5] : "auto";
    const fs::path logit_training_path = argc >= 7 && std::string_view(argv[6]) != "-"
        ? fs::u8path(argv[6])
        : fs::path{};
    const int logit_epochs = argc >= 8 ? std::stoi(argv[7]) : 150;
    const double boundary_penalty = argc >= 9 ? std::stod(argv[8]) : 0.9;
    const double time_continuity_bonus = argc >= 10 ? std::stod(argv[9]) : 0.1;
    const double turn_half_life = argc >= 11 ? std::stod(argv[10]) : 8.0;
    const double logit_penalty_scale = argc >= 12 ? std::stod(argv[11]) : 0.2;
    if (execution_provider != "auto" &&
        execution_provider != "cpu" &&
        execution_provider != "cuda") {
        std::cerr << "execution_provider must be auto, cpu, or cuda\n";
        return 2;
    }
    if (!logit_training_path.empty() && logit_epochs <= 0) {
        std::cerr << "logit_epochs must be positive when training is enabled\n";
        return 2;
    }

    const auto load_started = std::chrono::steady_clock::now();
    auto tokenizer = vector::HfTokenizer::LoadFromFile(tokenizer_path);
    if (!tokenizer.ok()) {
        std::cerr << "tokenizer: " << tokenizer.status().message() << '\n';
        return 1;
    }

    vector::EmbeddingModelOptions model_options;
    model_options.model_path = model_path;
    model_options.execution_provider = execution_provider;
    model_options.allow_cpu_fallback = execution_provider == "auto";
    model_options.pooling = vector::PoolingStrategy::Mean;
    model_options.normalize = true;
    model_options.expected_dimension = agent::semantic_cache::kExpectedEmbeddingDim;
    auto model = vector::OnnxTextEmbeddingModel::Load(std::move(model_options));
    if (!model.ok()) {
        std::cerr << "model: " << model.status().message() << '\n';
        return 1;
    }
    auto concrete_model = std::move(model).value();
    const auto provider_name = concrete_model->GetActiveExecutionProvider();
    if (execution_provider != "auto" && provider_name != execution_provider) {
        std::cerr << "requested provider " << execution_provider
                  << " but active provider is " << provider_name << '\n';
        return 1;
    }
    std::shared_ptr<vector::IEmbeddingModel> shared_model(std::move(concrete_model));

    vector::EmbeddingPipelineOptions pipeline_options;
    pipeline_options.tokenizer_options.max_length = 128;
    pipeline_options.tokenizer_options.truncation = true;
    pipeline_options.tokenizer_options.padding = true;
    pipeline_options.tokenizer_options.pad_to_longest_in_batch = true;
    auto pipeline = std::make_shared<vector::EmbeddingPipeline>(
        std::make_shared<vector::HfTokenizer>(std::move(tokenizer).value()),
        std::move(shared_model),
        pipeline_options);
    auto embedding_provider =
        std::make_shared<agent::conversation::EmbeddingPipelineProvider>(pipeline);
    const auto model_load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - load_started).count();

    constexpr std::string_view warmup_text =
        "Warm up the embedding execution provider before measuring dataset throughput.";
    std::vector<std::string_view> warmup_texts(32, warmup_text);
    const auto warmup_started = std::chrono::steady_clock::now();
    auto warmup = embedding_provider->EmbedBatch(warmup_texts);
    if (!warmup.ok()) {
        std::cerr << "embedding warmup: " << warmup.status().message() << '\n';
        return 1;
    }
    const auto warmup_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - warmup_started).count();

    agent::conversation::DialogueSegmenterOptions segmenter_options;
    segmenter_options.max_block_turns = 16;
    segmenter_options.embedding_batch_size = 32;
    segmenter_options.boundary_penalty = boundary_penalty;
    segmenter_options.time_continuity_bonus = time_continuity_bonus;
    segmenter_options.turn_half_life = turn_half_life;
    segmenter_options.context_mode = "current_turn";
    segmenter_options.params_version = "time_continuity_v1";

    std::size_t logit_training_examples = 0;
    double logit_initial_loss = 0.0;
    double logit_final_loss = 0.0;
    std::int64_t logit_feature_collection_ms = 0;
    std::int64_t logit_optimizer_ms = 0;
    if (!logit_training_path.empty()) {
        agent::conversation::LinearBoundaryLogitOptions logit_options;
        logit_options.penalty_adjustment_scale = logit_penalty_scale;
        auto created_logit = agent::conversation::LinearBoundaryLogitModel::Create(logit_options);
        if (!created_logit.ok()) {
            std::cerr << "linear Logit: " << created_logit.status().message() << '\n';
            return 1;
        }
        auto logit_model = std::move(created_logit).value();
        const auto feature_collection_started = std::chrono::steady_clock::now();
        auto examples_result = CollectBoundaryTrainingExamples(
            ReadJson(logit_training_path),
            *embedding_provider,
            segmenter_options);
        logit_feature_collection_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - feature_collection_started).count();
        if (!examples_result.ok()) {
            std::cerr << "Logit training data: "
                      << examples_result.status().message() << '\n';
            return 1;
        }
        auto examples = std::move(examples_result).value();
        logit_training_examples = examples.size();
        core::optimization::AdamOptions adam_options;
        adam_options.learning_rate = 0.03;
        adam_options.weight_decay = 1e-4;
        adam_options.max_gradient_norm = 5.0;
        auto created_optimizer = core::optimization::AdamOptimizer::Create(adam_options);
        if (!created_optimizer.ok()) {
            std::cerr << "Adam: " << created_optimizer.status().message() << '\n';
            return 1;
        }
        auto optimizer = std::move(created_optimizer).value();
        const auto optimizer_started = std::chrono::steady_clock::now();
        for (int epoch = 1; epoch <= logit_epochs; ++epoch) {
            auto loss = logit_model->TrainBatch(examples, *optimizer);
            if (!loss.ok()) {
                std::cerr << "Logit training: " << loss.status().message() << '\n';
                return 1;
            }
            if (epoch == 1) {
                logit_initial_loss = loss.value();
            }
            logit_final_loss = loss.value();
        }
        logit_optimizer_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - optimizer_started).count();
        std::cout << "logit_training_dataset=" << PathUtf8(logit_training_path) << '\n'
                  << "logit_training_examples=" << logit_training_examples << '\n'
                  << "logit_epochs=" << logit_epochs << '\n'
                  << "logit_penalty_scale=" << logit_penalty_scale << '\n'
                  << "logit_feature_collection_ms=" << logit_feature_collection_ms << '\n'
                  << "logit_optimizer_ms=" << logit_optimizer_ms << '\n'
                  << "logit_initial_loss=" << logit_initial_loss << '\n'
                  << "logit_final_loss=" << logit_final_loss << '\n'
                  << "logit_parameters=";
        for (const auto parameter : logit_model->Parameters()) {
            std::cout << parameter << ',';
        }
        std::cout << '\n';
        segmenter_options.boundary_penalty_model = std::move(logit_model);
        segmenter_options.params_version = "time_continuity_linear_logit_v1";
    }

    const auto root = ReadJson(dataset_path);
    const auto& dial_data = root.at("dial_data");
    if (!dial_data.is_object() || dial_data.size() != 1) {
        throw std::runtime_error("dial_data must contain exactly one dataset");
    }
    const auto dataset = dial_data.begin().key();
    const auto& dialogues = dial_data.begin().value();
    const auto dialogue_limit = max_dialogues == 0
        ? dialogues.size()
        : std::min(max_dialogues, dialogues.size());

    agent::conversation::BoundedDpDialogueSegmenter segmenter(
        embedding_provider,
        segmenter_options);

    BoundaryCounts metrics;
    std::size_t total_turns = 0;
    std::size_t total_blocks = 0;
    const auto benchmark_started = std::chrono::steady_clock::now();
    for (std::size_t dialogue_index = 0; dialogue_index < dialogue_limit; ++dialogue_index) {
        const auto& dialogue = dialogues[dialogue_index];
        const auto& json_turns = dialogue.at("turns");
        auto turns = ParseTurns(dataset, dialogue, dialogue_index);
        const auto dialogue_id = dialogue.value(
            "dial_id",
            dataset + "-" + std::to_string(dialogue_index));

        auto segmented = segmenter.Segment(dialogue_id, turns);
        if (!segmented.ok()) {
            std::cerr << "segment " << dialogue_id << ": "
                      << segmented.status().message() << '\n';
            return 1;
        }
        const auto reference = ReferenceBoundaries(json_turns);
        const auto predicted = PredictedBoundaries(
            turns.size(),
            segmented.value());
        Add(metrics, Evaluate(reference, predicted));
        total_turns += turns.size();
        total_blocks += segmented.value().blocks.size();
    }
    const auto benchmark_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - benchmark_started).count();

    const auto exact_precision = SafeRatio(metrics.exact_matches, metrics.predicted);
    const auto exact_recall = SafeRatio(metrics.exact_matches, metrics.reference);
    const auto relaxed_precision = SafeRatio(metrics.relaxed_matches, metrics.predicted);
    const auto relaxed_recall = SafeRatio(metrics.relaxed_matches, metrics.reference);
    const auto seconds = static_cast<double>(benchmark_ms) / 1000.0;

    std::cout << std::fixed << std::setprecision(6)
              << "dataset=" << dataset << '\n'
              << "dialogues=" << dialogue_limit << '\n'
              << "turns=" << total_turns << '\n'
              << "blocks=" << total_blocks << '\n'
              << "reference_boundaries=" << metrics.reference << '\n'
              << "predicted_boundaries=" << metrics.predicted << '\n'
              << "boundary_penalty=" << segmenter_options.boundary_penalty << '\n'
              << "time_continuity_bonus=" << segmenter_options.time_continuity_bonus << '\n'
              << "turn_half_life=" << segmenter_options.turn_half_life << '\n'
              << "requested_provider=" << execution_provider << '\n'
              << "model_provider=" << provider_name << '\n'
              << "model_load_ms=" << model_load_ms << '\n'
              << "warmup_ms=" << warmup_ms << '\n'
              << "benchmark_ms=" << benchmark_ms << '\n'
              << "turns_per_second=" << (seconds > 0.0 ? total_turns / seconds : 0.0) << '\n'
              << "exact_precision=" << exact_precision << '\n'
              << "exact_recall=" << exact_recall << '\n'
              << "exact_f1=" << F1(exact_precision, exact_recall) << '\n'
              << "relaxed_1_precision=" << relaxed_precision << '\n'
              << "relaxed_1_recall=" << relaxed_recall << '\n'
              << "relaxed_1_f1=" << F1(relaxed_precision, relaxed_recall) << '\n'
              << "pk=" << SafeRatio(metrics.pk_errors, metrics.pk_windows) << '\n'
              << "window_diff="
              << SafeRatio(metrics.window_diff_errors, metrics.window_diff_windows) << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    try {
        return Run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "dataset benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
