#include "cloud_task_coordinator.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

namespace {

class OutOfOrderLlmClient final : public agent::llm::ILlmClient {
public:
    core::Result<agent::llm::ChatCompletionResponse> Complete(
        const agent::llm::ChatCompletionRequest& request) override {
        const auto active = in_flight.fetch_add(1, std::memory_order_acq_rel) + 1;
        auto maximum = max_in_flight.load(std::memory_order_relaxed);
        while (maximum < active &&
               !max_in_flight.compare_exchange_weak(maximum, active, std::memory_order_relaxed)) {
        }
        const auto content = request.messages.front().content;
        const auto sequence = std::stoi(content);
        std::this_thread::sleep_for(std::chrono::milliseconds((5 - sequence) * 8));
        in_flight.fetch_sub(1, std::memory_order_acq_rel);
        if (content == "3" && fail_sequence_three) {
            return core::Status::Error(core::ErrorCode::Unavailable, "scripted failure");
        }
        agent::llm::ChatCompletionResponse response;
        response.content = content;
        response.prompt_tokens = sequence;
        response.completion_tokens = 1;
        response.total_tokens = sequence + 1;
        return response;
    }

    std::atomic<int> in_flight{0};
    std::atomic<int> max_in_flight{0};
    bool fail_sequence_three = false;
};

std::vector<agent::llm::CloudTaskChunk> MakeChunks(std::size_t count) {
    std::vector<agent::llm::CloudTaskChunk> chunks;
    chunks.reserve(count);
    for (std::size_t index = 1; index <= count; ++index) {
        agent::llm::CloudTaskChunk chunk;
        chunk.task.session_id = "session-1";
        chunk.task.task_id = "task-1";
        chunk.session_sequence = 7;
        chunk.chunk_sequence = index;
        chunk.final_chunk_sequence = count;
        chunk.request.messages.push_back({
            agent::llm::ChatRole::User,
            std::to_string(index),
        });
        chunks.push_back(std::move(chunk));
    }
    return chunks;
}

} // namespace

TEST(CloudTaskCoordinatorTest, ExecutesConcurrentlyAndReturnsChunkOrder) {
    auto llm = std::make_shared<OutOfOrderLlmClient>();
    agent::llm::CloudTaskCoordinator coordinator(
        llm,
        {
            .worker_count = 4,
            .max_chunks = 8,
            .queue_capacity = 4,
        });

    auto results = coordinator.Execute(MakeChunks(4));

    ASSERT_TRUE(results.ok()) << results.status().message();
    ASSERT_EQ(results.value().size(), 4u);
    EXPECT_GT(llm->max_in_flight.load(), 1);
    for (std::size_t index = 0; index < results.value().size(); ++index) {
        EXPECT_EQ(results.value()[index].chunk_sequence, index + 1);
        ASSERT_TRUE(results.value()[index].status.ok());
        ASSERT_TRUE(results.value()[index].response.has_value());
        EXPECT_EQ(results.value()[index].response->content, std::to_string(index + 1));
    }
    const auto snapshot = coordinator.Snapshot();
    EXPECT_EQ(snapshot.successful_chunks, 4u);
    EXPECT_EQ(snapshot.failed_chunks, 0u);
    EXPECT_FALSE(snapshot.running);
}

TEST(CloudTaskCoordinatorTest, FailureIsTerminalAndDoesNotBlockLaterChunks) {
    auto llm = std::make_shared<OutOfOrderLlmClient>();
    llm->fail_sequence_three = true;
    agent::llm::CloudTaskCoordinator coordinator(
        llm,
        {
            .worker_count = 4,
            .max_chunks = 8,
        });

    auto results = coordinator.Execute(MakeChunks(4));

    ASSERT_TRUE(results.ok()) << results.status().message();
    ASSERT_EQ(results.value().size(), 4u);
    EXPECT_EQ(results.value()[2].status.code(), core::ErrorCode::Unavailable);
    EXPECT_FALSE(results.value()[2].response.has_value());
    EXPECT_TRUE(results.value()[3].status.ok());
    EXPECT_EQ(results.value()[3].response->content, "4");
    const auto snapshot = coordinator.Snapshot();
    EXPECT_EQ(snapshot.successful_chunks, 3u);
    EXPECT_EQ(snapshot.failed_chunks, 1u);
}

TEST(CloudTaskCoordinatorTest, AsyncSubmissionReturnsBeforeOrderedResultIsReady) {
    auto llm = std::make_shared<OutOfOrderLlmClient>();
    agent::llm::CloudTaskCoordinator coordinator(
        llm,
        {
            .worker_count = 2,
            .max_chunks = 8,
            .max_pending_tasks = 2,
        });

    auto submission = coordinator.ExecuteAsync(MakeChunks(4));

    ASSERT_TRUE(submission.ok()) << submission.status().message();
    auto future = std::move(submission).value();
    EXPECT_EQ(future.wait_for(0ms), std::future_status::timeout);
    auto results = future.get();
    ASSERT_TRUE(results.ok()) << results.status().message();
    ASSERT_EQ(results.value().size(), 4u);
    for (std::size_t index = 0; index < results.value().size(); ++index) {
        EXPECT_EQ(results.value()[index].chunk_sequence, index + 1);
    }
}

TEST(CloudTaskCoordinatorTest, RejectsNonContiguousSequencesBeforeCallingProvider) {
    auto llm = std::make_shared<OutOfOrderLlmClient>();
    agent::llm::CloudTaskCoordinator coordinator(llm);
    auto chunks = MakeChunks(2);
    chunks[1].chunk_sequence = 3;

    auto results = coordinator.Execute(std::move(chunks));

    ASSERT_FALSE(results.ok());
    EXPECT_EQ(results.status().code(), core::ErrorCode::InvalidArgument);
    EXPECT_EQ(llm->max_in_flight.load(), 0);
}
