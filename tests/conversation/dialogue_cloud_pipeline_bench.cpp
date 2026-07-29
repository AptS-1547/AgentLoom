#include "cloud_task_coordinator.h"
#include "dialogue_segmenter.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

class BenchEmbeddingProvider final : public agent::conversation::ITextEmbeddingProvider {
public:
    core::Result<std::vector<float>> EmbedText(std::string_view text) override {
        const auto separator = text.find('-');
        const auto index = separator == std::string_view::npos
            ? 0u
            : static_cast<unsigned>(std::stoul(std::string(text.substr(separator + 1))));
        std::vector<float> embedding(128, 0.0f);
        embedding[(index / 20) % embedding.size()] = 1.0f;
        return embedding;
    }

    core::Result<Batch> EmbedBatch(
        std::span<const std::string_view> texts) override {
        Batch batch;
        batch.batch_size = texts.size();
        batch.dimension = 128;
        batch.embeddings.assign(batch.batch_size * batch.dimension, 0.0f);
        for (std::size_t row = 0; row < texts.size(); ++row) {
            const auto separator = texts[row].find('-');
            const auto index = separator == std::string_view::npos
                ? 0u
                : static_cast<unsigned>(std::stoul(std::string(texts[row].substr(separator + 1))));
            batch.embeddings[row * batch.dimension + (index / 20) % batch.dimension] = 1.0f;
        }
        return batch;
    }

    bool Normalized() const noexcept override { return true; }
};

class BenchLlmClient final : public agent::llm::ILlmClient {
public:
    core::Result<agent::llm::ChatCompletionResponse> Complete(
        const agent::llm::ChatCompletionRequest&) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        agent::llm::ChatCompletionResponse response;
        response.content = "ok";
        response.prompt_tokens = 8;
        response.completion_tokens = 1;
        response.total_tokens = 9;
        return response;
    }
};

} // namespace

int main() {
    constexpr std::size_t kTurnCount = 10'000;
    auto embedding = std::make_shared<BenchEmbeddingProvider>();
    agent::conversation::BoundedDpDialogueSegmenter segmenter(
        embedding,
        {.max_block_turns = 32, .boundary_penalty = 0.4});
    std::vector<agent::conversation::DialogueTurn> turns;
    turns.reserve(kTurnCount);
    for (std::size_t index = 0; index < kTurnCount; ++index) {
        turns.push_back({
            .turn_id = "turn-" + std::to_string(index),
            .speaker = index % 2 == 0 ? "customer" : "agent",
            .text = "topic-" + std::to_string(index),
            .timestamp_us = static_cast<std::int64_t>(index) * 1'000'000,
        });
    }

    const auto segment_started = std::chrono::steady_clock::now();
    auto segmented = segmenter.Segment("bench-session", turns);
    const auto segment_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - segment_started).count();
    if (!segmented.ok()) {
        std::cerr << "segmentation failed: " << segmented.status().message() << '\n';
        return 1;
    }

    const auto chunk_count = std::min<std::size_t>(segmented.value().blocks.size(), 256);
    std::vector<agent::llm::CloudTaskChunk> chunks;
    chunks.reserve(chunk_count);
    for (std::size_t index = 0; index < chunk_count; ++index) {
        agent::llm::CloudTaskChunk chunk;
        chunk.task.session_id = "bench-session";
        chunk.task.task_id = "bench-task";
        chunk.chunk_sequence = index + 1;
        chunk.final_chunk_sequence = chunk_count;
        chunk.request.messages.push_back({agent::llm::ChatRole::User, "annotate"});
        chunks.push_back(std::move(chunk));
    }
    agent::llm::CloudTaskCoordinator coordinator(
        std::make_shared<BenchLlmClient>(),
        {.worker_count = 8, .max_chunks = 256});
    const auto cloud_started = std::chrono::steady_clock::now();
    auto cloud_results = coordinator.Execute(std::move(chunks));
    const auto cloud_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - cloud_started).count();
    if (!cloud_results.ok()) {
        std::cerr << "cloud task failed: " << cloud_results.status().message() << '\n';
        return 1;
    }

    std::cout << "turns=" << kTurnCount
              << " blocks=" << segmented.value().blocks.size()
              << " segment_ms=" << segment_ms
              << " cloud_chunks=" << cloud_results.value().size()
              << " cloud_ms=" << cloud_ms << '\n';
    return 0;
}
