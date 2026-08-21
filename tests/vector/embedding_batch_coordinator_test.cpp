#include "embedding_batch_coordinator.h"

#include <gtest/gtest.h>

#include <chrono>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

class FakeEmbeddingProvider final : public vector::IEmbeddingBatchProvider {
public:
    core::Result<vector::EmbeddingBatch> EncodeBatch(
        std::span<const std::string_view> texts) const override {
        {
            std::lock_guard lock(mutex_);
            batch_sizes_.push_back(texts.size());
        }
        condition_.notify_all();
        vector::EmbeddingBatch result;
        result.batch_size = texts.size();
        result.dimension = 1;
        result.embeddings.reserve(texts.size());
        for (const auto text : texts) {
            result.embeddings.push_back(static_cast<float>(text.size()));
        }
        return result;
    }

    bool WaitForBatchCount(std::size_t count) const {
        std::unique_lock lock(mutex_);
        return condition_.wait_for(lock, std::chrono::seconds(2), [&] {
            return batch_sizes_.size() >= count;
        });
    }

    std::vector<std::size_t> BatchSizes() const {
        std::lock_guard lock(mutex_);
        return batch_sizes_;
    }

private:
    mutable std::mutex mutex_;
    mutable std::condition_variable condition_;
    mutable std::vector<std::size_t> batch_sizes_;
};

class OutOfOrderEmbeddingProvider final : public vector::IEmbeddingBatchProvider {
public:
    core::Result<vector::EmbeddingBatch> EncodeBatch(
        std::span<const std::string_view> texts) const override {
        const auto call = calls_.fetch_add(1, std::memory_order_relaxed);
        if (call == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        vector::EmbeddingBatch result;
        result.batch_size = texts.size();
        result.dimension = 1;
        for (const auto text : texts) {
            result.embeddings.push_back(static_cast<float>(text.size()));
        }
        return result;
    }

private:
    mutable std::atomic<std::size_t> calls_{0};
};

class BlockingEmbeddingProvider final : public vector::IEmbeddingBatchProvider {
public:
    core::Result<vector::EmbeddingBatch> EncodeBatch(
        std::span<const std::string_view> texts) const override {
        {
            std::lock_guard lock(mutex_);
            entered_ = true;
        }
        condition_.notify_all();
        std::unique_lock lock(mutex_);
        condition_.wait(lock, [&] { return released_; });
        vector::EmbeddingBatch result;
        result.batch_size = texts.size();
        result.dimension = 1;
        result.embeddings.assign(texts.size(), 1.0f);
        return result;
    }

    bool WaitUntilEntered() const {
        std::unique_lock lock(mutex_);
        return condition_.wait_for(lock, std::chrono::seconds(2), [&] {
            return entered_;
        });
    }

    void Release() {
        {
            std::lock_guard lock(mutex_);
            released_ = true;
        }
        condition_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    mutable std::condition_variable condition_;
    mutable bool entered_ = false;
    mutable bool released_ = false;
};

class EmbeddingBatchCoordinatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(compute_pool_.Start().ok());
        provider_ = std::make_shared<FakeEmbeddingProvider>();
    }

    void TearDown() override {
        if (coordinator_) {
            coordinator_->Shutdown();
        }
        compute_pool_.Shutdown(true);
    }

    core::ThreadPool compute_pool_{{.worker_count = 2,
                                    .queue_capacity = 32,
                                    .name = "embedding-batch-test"}};
    std::shared_ptr<FakeEmbeddingProvider> provider_;
    std::unique_ptr<vector::EmbeddingBatchCoordinator> coordinator_;
};

TEST_F(EmbeddingBatchCoordinatorTest, FlushesFullBatchWithoutBlockingWorkerAdmission) {
    coordinator_ = std::make_unique<vector::EmbeddingBatchCoordinator>(
        compute_pool_, provider_,
        vector::EmbeddingBatchCoordinatorOptions{
            .max_pending_requests = 16,
            .max_batch_size = 3,
            .max_batch_wait = std::chrono::milliseconds(100),
            .max_inflight_batches = 1,
        });
    ASSERT_TRUE(coordinator_->Start().ok());

    std::mutex mutex;
    std::condition_variable condition;
    std::vector<std::string> completed;
    for (const auto text : {"a", "bb", "ccc"}) {
        const std::string expected_text = text;
        ASSERT_TRUE(coordinator_->Submit({
            .session_id = expected_text,
            .text = expected_text,
            .completion = [&, expected_text](core::Result<std::vector<float>> result) {
                ASSERT_TRUE(result.ok());
                std::lock_guard lock(mutex);
                completed.emplace_back(expected_text);
                condition.notify_all();
            },
        }).ok());
    }

    std::unique_lock lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(2), [&] {
        return completed.size() == 3;
    }));
    ASSERT_TRUE(provider_->WaitForBatchCount(1));
    ASSERT_EQ(provider_->BatchSizes(), std::vector<std::size_t>({3}));
}

TEST_F(EmbeddingBatchCoordinatorTest, FlushesPartialBatchAfterTimeout) {
    coordinator_ = std::make_unique<vector::EmbeddingBatchCoordinator>(
        compute_pool_, provider_,
        vector::EmbeddingBatchCoordinatorOptions{
            .max_pending_requests = 16,
            .max_batch_size = 8,
            .max_batch_wait = std::chrono::milliseconds(10),
            .max_inflight_batches = 1,
        });
    ASSERT_TRUE(coordinator_->Start().ok());

    std::mutex mutex;
    std::condition_variable condition;
    bool completed = false;
    ASSERT_TRUE(coordinator_->Submit({
        .session_id = "partial",
        .text = "partial",
        .completion = [&](core::Result<std::vector<float>> result) {
            ASSERT_TRUE(result.ok());
            std::lock_guard lock(mutex);
            completed = true;
            condition.notify_all();
        },
    }).ok());

    std::unique_lock lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(2), [&] {
        return completed;
    }));
    ASSERT_TRUE(provider_->WaitForBatchCount(1));
    EXPECT_EQ(provider_->BatchSizes(), std::vector<std::size_t>({1}));
    EXPECT_EQ(coordinator_->Snapshot().partial_batches, 1u);
}

TEST_F(EmbeddingBatchCoordinatorTest, DeliversSameSessionResultsInAdmissionOrder) {
    auto out_of_order_provider = std::make_shared<OutOfOrderEmbeddingProvider>();
    coordinator_ = std::make_unique<vector::EmbeddingBatchCoordinator>(
        compute_pool_, out_of_order_provider,
        vector::EmbeddingBatchCoordinatorOptions{
            .max_pending_requests = 16,
            .max_batch_size = 1,
            .max_batch_wait = std::chrono::milliseconds(1),
            .max_inflight_batches = 2,
        });
    ASSERT_TRUE(coordinator_->Start().ok());

    std::mutex mutex;
    std::condition_variable condition;
    std::vector<float> observed;
    for (const auto text : {"first", "second"}) {
        ASSERT_TRUE(coordinator_->Submit({
            .session_id = "ordered-session",
            .text = text,
            .completion = [&](core::Result<std::vector<float>> result) {
                ASSERT_TRUE(result.ok());
                std::lock_guard lock(mutex);
                observed.push_back(result.value().front());
                condition.notify_all();
            },
        }).ok());
    }

    std::unique_lock lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(2), [&] {
        return observed.size() == 2;
    }));
    EXPECT_EQ(observed, (std::vector<float>{5.0f, 6.0f}));
}

TEST_F(EmbeddingBatchCoordinatorTest, KeepsOnlyOneSameSessionCompletionInflight) {
    auto out_of_order_provider = std::make_shared<OutOfOrderEmbeddingProvider>();
    coordinator_ = std::make_unique<vector::EmbeddingBatchCoordinator>(
        compute_pool_, out_of_order_provider,
        vector::EmbeddingBatchCoordinatorOptions{
            .max_pending_requests = 16,
            .max_batch_size = 1,
            .max_batch_wait = std::chrono::milliseconds(1),
            .max_inflight_batches = 2,
        });
    ASSERT_TRUE(coordinator_->Start().ok());

    std::mutex mutex;
    std::condition_variable condition;
    bool first_entered = false;
    bool release_first = false;
    bool second_entered = false;
    ASSERT_TRUE(coordinator_->Submit({
        .session_id = "serial-continuation",
        .text = "first",
        .completion = [&](core::Result<std::vector<float>> result) {
            ASSERT_TRUE(result.ok());
            std::unique_lock lock(mutex);
            first_entered = true;
            condition.notify_all();
            condition.wait(lock, [&] { return release_first; });
        },
    }).ok());
    ASSERT_TRUE(coordinator_->Submit({
        .session_id = "serial-continuation",
        .text = "second",
        .completion = [&](core::Result<std::vector<float>> result) {
            ASSERT_TRUE(result.ok());
            std::lock_guard lock(mutex);
            second_entered = true;
            condition.notify_all();
        },
    }).ok());

    {
        std::unique_lock lock(mutex);
        ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(2), [&] {
            return first_entered;
        }));
        EXPECT_FALSE(second_entered);
        release_first = true;
    }
    condition.notify_all();
    std::unique_lock lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, std::chrono::seconds(2), [&] {
        return second_entered;
    }));
}

TEST_F(EmbeddingBatchCoordinatorTest, EnforcesBoundedPendingQueue) {
    coordinator_ = std::make_unique<vector::EmbeddingBatchCoordinator>(
        compute_pool_, provider_,
        vector::EmbeddingBatchCoordinatorOptions{
            .max_pending_requests = 1,
            .max_batch_size = 8,
            .max_batch_wait = std::chrono::milliseconds(100),
            .max_inflight_batches = 1,
        });
    ASSERT_TRUE(coordinator_->Start().ok());

    ASSERT_TRUE(coordinator_->Submit({
        .session_id = "bounded-1",
        .text = "one",
        .completion = [](core::Result<std::vector<float>>) {},
    }).ok());
    const auto rejected = coordinator_->Submit({
        .session_id = "bounded-2",
        .text = "two",
        .completion = [](core::Result<std::vector<float>>) {},
    });
    ASSERT_FALSE(rejected.ok());
    EXPECT_EQ(rejected.code(), core::ErrorCode::ResourceExhausted);
    EXPECT_EQ(coordinator_->Snapshot().rejected_requests, 1u);
}

TEST_F(EmbeddingBatchCoordinatorTest, ShutdownWaitsForInflightRequestCompletion) {
    auto blocking_provider = std::make_shared<BlockingEmbeddingProvider>();
    coordinator_ = std::make_unique<vector::EmbeddingBatchCoordinator>(
        compute_pool_, blocking_provider,
        vector::EmbeddingBatchCoordinatorOptions{
            .max_pending_requests = 8,
            .max_batch_size = 1,
            .max_batch_wait = std::chrono::milliseconds(1),
            .max_inflight_batches = 1,
        });
    ASSERT_TRUE(coordinator_->Start().ok());

    std::atomic<int> callback_count{0};
    ASSERT_TRUE(coordinator_->Submit({
        .session_id = "shutdown-session",
        .text = "pending",
        .completion = [&](core::Result<std::vector<float>> result) {
            EXPECT_TRUE(result.ok()) << result.status().message();
            callback_count.fetch_add(1, std::memory_order_relaxed);
        },
    }).ok());
    ASSERT_TRUE(blocking_provider->WaitUntilEntered());

    auto shutdown = std::async(std::launch::async, [this] {
        coordinator_->Shutdown();
    });
    EXPECT_EQ(shutdown.wait_for(std::chrono::milliseconds(100)),
              std::future_status::timeout);
    blocking_provider->Release();
    EXPECT_EQ(shutdown.wait_for(std::chrono::seconds(2)),
              std::future_status::ready);
    EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 1);
    EXPECT_EQ(coordinator_->Snapshot().completed_requests, 1u);
}

} // namespace
