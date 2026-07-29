#include "cloud_task_coordinator.h"
#include "dialogue_segmenter.h"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>

using namespace std::chrono_literals;

namespace {

class PipelineEmbeddingProvider final : public agent::conversation::ITextEmbeddingProvider {
public:
    core::Result<std::vector<float>> EmbedText(std::string_view text) override {
        if (text.starts_with("price")) {
            return std::vector<float>{1.0f, 0.0f};
        }
        if (text.starts_with("delivery")) {
            return std::vector<float>{0.0f, 1.0f};
        }
        return core::Status::Error(core::ErrorCode::NotFound, "pipeline embedding is missing");
    }
};

class PipelineLlmClient final : public agent::llm::ILlmClient {
public:
    core::Result<agent::llm::ChatCompletionResponse> Complete(
        const agent::llm::ChatCompletionRequest& request) override {
        const auto input = request.messages.front().content;
        if (input.ends_with("0")) {
            std::this_thread::sleep_for(20ms);
        }
        agent::llm::ChatCompletionResponse response;
        response.content = "behavior:" + input;
        response.prompt_tokens = 4;
        response.completion_tokens = 2;
        response.total_tokens = 6;
        return response;
    }
};

} // namespace

TEST(DialogueCloudPipelineE2ETest, SegmentsThenAnnotatesBlocksInSequenceOrder) {
    auto embedding = std::make_shared<PipelineEmbeddingProvider>();
    agent::conversation::BoundedDpDialogueSegmenter segmenter(
        embedding,
        {
            .min_block_turns = 2,
            .max_block_turns = 4,
            .boundary_penalty = 0.2,
            .short_block_penalty = 1.0,
            .context_mode = "offline_bidirectional_v1",
        });
    std::vector<agent::conversation::DialogueTurn> turns{
        {.turn_id = "t1", .speaker = "customer", .text = "price concern"},
        {.turn_id = "t2", .speaker = "agent", .text = "price response"},
        {.turn_id = "t3", .speaker = "customer", .text = "delivery question"},
        {.turn_id = "t4", .speaker = "agent", .text = "delivery response"},
    };

    auto segmentation = segmenter.Segment("session-e2e", turns);
    ASSERT_TRUE(segmentation.ok()) << segmentation.status().message();
    ASSERT_EQ(segmentation.value().blocks.size(), 2u);

    std::vector<agent::llm::CloudTaskChunk> chunks;
    for (std::size_t index = 0; index < segmentation.value().blocks.size(); ++index) {
        agent::llm::CloudTaskChunk chunk;
        chunk.task.session_id = segmentation.value().session_id;
        chunk.task.task_id = "behavior-extraction-v1";
        chunk.session_sequence = 1;
        chunk.chunk_sequence = index + 1;
        chunk.final_chunk_sequence = segmentation.value().blocks.size();
        chunk.source_begin = segmentation.value().blocks[index].owned_begin;
        chunk.source_end = segmentation.value().blocks[index].owned_end;
        chunk.route_alias = "behavior.extract";
        chunk.request.messages.push_back({
            agent::llm::ChatRole::User,
            segmentation.value().blocks[index].block_id,
        });
        chunks.push_back(std::move(chunk));
    }

    agent::llm::CloudTaskCoordinator coordinator(
        std::make_shared<PipelineLlmClient>(),
        {.worker_count = 2, .max_chunks = 8});
    auto submission = coordinator.ExecuteAsync(std::move(chunks));
    ASSERT_TRUE(submission.ok()) << submission.status().message();
    auto results = std::move(submission).value().get();

    ASSERT_TRUE(results.ok()) << results.status().message();
    ASSERT_EQ(results.value().size(), 2u);
    EXPECT_EQ(results.value()[0].chunk_sequence, 1u);
    EXPECT_EQ(results.value()[1].chunk_sequence, 2u);
    EXPECT_EQ(results.value()[0].response->content, "behavior:session-e2e:block-0");
    EXPECT_EQ(results.value()[1].response->content, "behavior:session-e2e:block-1");
    EXPECT_EQ(segmentation.value().blocks[0].context_mode, "offline_bidirectional_v1");
}
