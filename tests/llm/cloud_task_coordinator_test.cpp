#include "cloud_task_coordinator.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace std::chrono_literals;

namespace {

class ManualAsyncOperation final : public agent::llm::IAsyncLlmOperation {
public:
    explicit ManualAsyncOperation(agent::llm::IAsyncLlmClient::Callback callback)
        : callback_(std::move(callback)) {}

    void Complete(core::Result<agent::llm::ChatCompletionResponse> result) {
        if (completed_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        callback_(std::move(result));
    }

    void Cancel() noexcept override {
        cancel_count_.fetch_add(1, std::memory_order_relaxed);
        Complete(core::Status::Error(core::ErrorCode::Cancelled, "manual operation cancelled"));
    }

    std::atomic<std::size_t> cancel_count_{0};

private:
    agent::llm::IAsyncLlmClient::Callback callback_;
    std::atomic<bool> completed_{false};
};

class ManualAsyncLlmClient final : public agent::llm::IAsyncLlmClient {
public:
    core::Result<std::shared_ptr<agent::llm::IAsyncLlmOperation>> CompleteAsync(
        agent::llm::ChatCompletionRequest request,
        Callback callback) override {
        const auto sequence = std::stoi(request.messages.front().content);
        auto operation = std::make_shared<ManualAsyncOperation>(
            [this, callback = std::move(callback)](auto result) mutable {
                active_.fetch_sub(1, std::memory_order_acq_rel);
                callback(std::move(result));
            });
        {
            std::lock_guard lock(mutex_);
            operations_[sequence] = operation;
        }
        const auto active = active_.fetch_add(1, std::memory_order_acq_rel) + 1;
        auto maximum = max_in_flight_.load(std::memory_order_relaxed);
        while (maximum < active &&
               !max_in_flight_.compare_exchange_weak(
                   maximum, active, std::memory_order_relaxed)) {
        }
        condition_.notify_all();
        return std::static_pointer_cast<agent::llm::IAsyncLlmOperation>(operation);
    }

    bool WaitForInFlight(std::size_t expected, std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex_);
        return condition_.wait_for(lock, timeout, [&] {
            return active_.load(std::memory_order_acquire) == expected;
        });
    }

    bool CompleteSequence(int sequence, bool success = true) {
        std::shared_ptr<ManualAsyncOperation> operation;
        {
            std::lock_guard lock(mutex_);
            auto found = operations_.find(sequence);
            if (found == operations_.end()) {
                return false;
            }
            operation = found->second;
            operations_.erase(found);
        }
        if (success) {
            agent::llm::ChatCompletionResponse response;
            response.content = std::to_string(sequence);
            operation->Complete(std::move(response));
        } else {
            operation->Complete(core::Status::Error(
                core::ErrorCode::Unavailable,
                "manual provider failure"));
        }
        return true;
    }

    std::size_t MaxInFlight() const noexcept {
        return max_in_flight_.load(std::memory_order_relaxed);
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::unordered_map<int, std::shared_ptr<ManualAsyncOperation>> operations_;
    std::atomic<std::size_t> active_{0};
    std::atomic<std::size_t> max_in_flight_{0};
};

class RecordingCloudTaskSink final : public agent::llm::ICloudTaskResultSink {
public:
    core::Status OnChunk(agent::llm::CloudTaskChunkResult result) override {
        std::lock_guard lock(mutex_);
        sequences_.push_back(result.chunk_sequence);
        condition_.notify_all();
        return core::Status::Ok();
    }

    core::Status OnCompleted(agent::llm::CloudTaskSummary summary) override {
        std::lock_guard lock(mutex_);
        summary_ = std::move(summary);
        condition_.notify_all();
        return core::Status::Ok();
    }

    bool WaitForCompletion(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex_);
        return condition_.wait_for(lock, timeout, [&] { return summary_.has_value(); });
    }

    std::vector<std::uint64_t> Sequences() const {
        std::lock_guard lock(mutex_);
        return sequences_;
    }

    agent::llm::CloudTaskSummary Summary() const {
        std::lock_guard lock(mutex_);
        return summary_.value();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::vector<std::uint64_t> sequences_;
    std::optional<agent::llm::CloudTaskSummary> summary_;
};

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

}

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

TEST(CloudTaskCoordinatorTest, AsyncSinkPreservesOrderAndBoundsProviderConcurrency) {
    auto llm = std::make_shared<ManualAsyncLlmClient>();
    auto sink = std::make_shared<RecordingCloudTaskSink>();
    agent::llm::CloudTaskCoordinator coordinator(
        std::static_pointer_cast<agent::llm::IAsyncLlmClient>(llm),
        {.worker_count = 2, .max_chunks = 8, .max_pending_tasks = 2});

    auto operation = coordinator.StartAsync(MakeChunks(4), sink);
    ASSERT_TRUE(operation.ok()) << operation.status().message();
    ASSERT_TRUE(llm->WaitForInFlight(2, 1s));
    EXPECT_LE(llm->MaxInFlight(), 2u);

    ASSERT_TRUE(llm->CompleteSequence(2));
    ASSERT_TRUE(llm->CompleteSequence(1));
    ASSERT_TRUE(llm->WaitForInFlight(2, 1s));
    ASSERT_TRUE(llm->CompleteSequence(4));
    ASSERT_TRUE(llm->CompleteSequence(3));
    ASSERT_TRUE(sink->WaitForCompletion(2s));

    EXPECT_EQ(sink->Sequences(), (std::vector<std::uint64_t>{1, 2, 3, 4}));
    EXPECT_TRUE(sink->Summary().status.ok());
    EXPECT_EQ(sink->Summary().completed_chunks, 4u);
}

TEST(CloudTaskCoordinatorTest, AsyncCancelPropagatesToActiveOperationsOnce) {
    auto llm = std::make_shared<ManualAsyncLlmClient>();
    auto sink = std::make_shared<RecordingCloudTaskSink>();
    agent::llm::CloudTaskCoordinator coordinator(
        std::static_pointer_cast<agent::llm::IAsyncLlmClient>(llm),
        {.worker_count = 2, .max_chunks = 8, .max_pending_tasks = 2});

    auto operation = coordinator.StartAsync(MakeChunks(4), sink);
    ASSERT_TRUE(operation.ok()) << operation.status().message();
    ASSERT_TRUE(llm->WaitForInFlight(2, 1s));
    operation.value()->Cancel();
    operation.value()->Cancel();
    ASSERT_TRUE(sink->WaitForCompletion(2s));
    EXPECT_EQ(sink->Summary().status.code(), core::ErrorCode::Cancelled);
    EXPECT_TRUE(sink->Sequences().empty());
}

TEST(CloudTaskCoordinatorTest, AsyncStartRejectsWhenActiveTaskCapacityIsReached) {
    auto llm = std::make_shared<ManualAsyncLlmClient>();
    auto first_sink = std::make_shared<RecordingCloudTaskSink>();
    auto second_sink = std::make_shared<RecordingCloudTaskSink>();
    agent::llm::CloudTaskCoordinator coordinator(
        std::static_pointer_cast<agent::llm::IAsyncLlmClient>(llm),
        {.worker_count = 1, .max_chunks = 8, .max_pending_tasks = 1});

    auto first = coordinator.StartAsync(MakeChunks(2), first_sink);
    ASSERT_TRUE(first.ok()) << first.status().message();
    ASSERT_TRUE(llm->WaitForInFlight(1, 1s));

    auto second = coordinator.StartAsync(MakeChunks(2), second_sink);
    ASSERT_FALSE(second.ok());
    EXPECT_EQ(second.status().code(), core::ErrorCode::ResourceExhausted);

    first.value()->Cancel();
    ASSERT_TRUE(first_sink->WaitForCompletion(2s));
}
