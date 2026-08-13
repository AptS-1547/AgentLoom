#include "blocking_queue.h"
#include "memory_pool.h"
#include "object_pool.h"
#include "shared_memory_block.h"
#include "thread_pool.h"
#include "unique_handle.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace test_resources {

struct FakeResource {};

inline std::atomic<int> fake_close_count{0};
inline std::atomic<int> fake_last_closed{0};

} // namespace test_resources

namespace core {

template <>
struct ResourceTraits<test_resources::FakeResource> {
    using handle_type = int;

    static handle_type invalid() noexcept {
        return 0;
    }

    static bool valid(handle_type handle) noexcept {
        return handle != 0;
    }

    static void close(handle_type handle) noexcept {
        test_resources::fake_last_closed.store(handle, std::memory_order_relaxed);
        test_resources::fake_close_count.fetch_add(1, std::memory_order_relaxed);
    }
};

} // namespace core

namespace {

using namespace std::chrono_literals;

struct PoolTrackedObject {
    static inline std::atomic<int> alive{0};

    explicit PoolTrackedObject(int value) : value(value) {
        alive.fetch_add(1, std::memory_order_relaxed);
    }

    ~PoolTrackedObject() {
        alive.fetch_sub(1, std::memory_order_relaxed);
    }

    int value = 0;
};

struct ThrowingPoolObject {
    ThrowingPoolObject() {
        throw std::runtime_error("construction failed");
    }
};

TEST(BucketMemoryPoolTest, AllocateHonorsSizeAlignmentAndStats) {
    core::BucketMemoryPool pool(0, 8);

    auto block_result = pool.allocate(128, 64);
    ASSERT_TRUE(block_result.ok()) << block_result.status().message();

    auto block = std::move(block_result).value();
    ASSERT_NE(block.data(), nullptr);
    EXPECT_GE(block.size(), 128u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(block.data()) % 64u, 0u);

    auto stats = pool.stats();
    EXPECT_EQ(stats.allocations, 1u);
    EXPECT_EQ(stats.deallocations, 0u);
    EXPECT_GE(stats.allocated_bytes, block.size());
    EXPECT_GE(stats.peak_allocated_bytes, block.size());

    const auto block_size = block.size();
    block.reset();

    stats = pool.stats();
    EXPECT_EQ(stats.allocations, 1u);
    EXPECT_EQ(stats.deallocations, 1u);
    EXPECT_EQ(stats.allocated_bytes, 0u);
    EXPECT_GE(stats.free_bytes, block_size);
}

TEST(BucketMemoryPoolTest, SupportsConcurrentAllocateRelease) {
    core::BucketMemoryPool pool(0, 32);

    constexpr int kThreadCount = 8;
    constexpr int kIterations = 256;
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreadCount);

    for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
        threads.emplace_back([&, thread_index] {
            for (int i = 0; i < kIterations; ++i) {
                const auto size = static_cast<std::size_t>(33 + ((thread_index + i) % 128));
                auto block_result = pool.allocate(size, 16);
                if (!block_result.ok()) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                auto block = std::move(block_result).value();
                if (!block.data() || block.size() < size) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                block.data()[0] = std::byte{0x2A};
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    auto stats = pool.stats();
    EXPECT_EQ(failures.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(stats.allocations, static_cast<std::size_t>(kThreadCount * kIterations));
    EXPECT_EQ(stats.deallocations, stats.allocations);
    EXPECT_EQ(stats.allocated_bytes, 0u);
}

TEST(ObjectPoolTest, MoveTransfersOwnershipAndResetDestroysObject) {
    PoolTrackedObject::alive.store(0, std::memory_order_relaxed);
    core::BucketMemoryPool memory_pool;
    core::ObjectPool<PoolTrackedObject> object_pool(memory_pool);

    auto object_result = object_pool.make(42);
    ASSERT_TRUE(object_result.ok()) << object_result.status().message();

    auto object = std::move(object_result).value();
    ASSERT_TRUE(object);
    EXPECT_EQ(object->value, 42);
    EXPECT_EQ(PoolTrackedObject::alive.load(std::memory_order_relaxed), 1);

    core::PooledObject<PoolTrackedObject> moved = std::move(object);
    EXPECT_FALSE(object);
    ASSERT_TRUE(moved);
    EXPECT_EQ(moved->value, 42);

    moved.reset();
    EXPECT_EQ(PoolTrackedObject::alive.load(std::memory_order_relaxed), 0);

    auto stats = memory_pool.stats();
    EXPECT_EQ(stats.allocations, 1u);
    EXPECT_EQ(stats.deallocations, 1u);
}

TEST(ObjectPoolTest, ConstructionFailureReleasesMemory) {
    core::BucketMemoryPool memory_pool;
    core::ObjectPool<ThrowingPoolObject> object_pool(memory_pool);

    auto object_result = object_pool.make();
    ASSERT_FALSE(object_result.ok());
    EXPECT_EQ(object_result.status().code(), core::ErrorCode::InternalError);

    auto stats = memory_pool.stats();
    EXPECT_EQ(stats.allocations, 1u);
    EXPECT_EQ(stats.deallocations, 1u);
    EXPECT_EQ(stats.allocated_bytes, 0u);
}

TEST(SharedMemoryBlockTest, CopyMoveAndResetManageReferenceCount) {
    core::BucketMemoryPool pool;

    auto block_result = core::SharedMemoryBlock::allocate(pool, 256, 32);
    ASSERT_TRUE(block_result.ok()) << block_result.status().message();

    auto block = std::move(block_result).value();
    ASSERT_TRUE(block);
    EXPECT_EQ(block.use_count(), 1u);

    {
        core::SharedMemoryBlock copy = block;
        EXPECT_EQ(block.use_count(), 2u);
        EXPECT_EQ(copy.use_count(), 2u);

        core::SharedMemoryBlock moved = std::move(copy);
        EXPECT_FALSE(copy);
        EXPECT_EQ(block.use_count(), 2u);
        EXPECT_EQ(moved.use_count(), 2u);
    }

    EXPECT_EQ(block.use_count(), 1u);
    block.reset();
    EXPECT_FALSE(block);

    auto stats = pool.stats();
    EXPECT_EQ(stats.allocations, 1u);
    EXPECT_EQ(stats.deallocations, 1u);
    EXPECT_EQ(stats.allocated_bytes, 0u);
}

TEST(BlockingQueueTest, TryPushReportsResourceExhaustedWhenFull) {
    core::BlockingQueue<int> queue(1);

    EXPECT_TRUE(queue.TryPush(1).ok());
    auto status = queue.TryPush(2);
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(status.code(), core::ErrorCode::ResourceExhausted);

    auto popped = queue.TryPop();
    ASSERT_TRUE(popped.ok()) << popped.status().message();
    EXPECT_EQ(std::move(popped).value(), 1);

    queue.Close();
    auto closed_pop = queue.WaitPop();
    ASSERT_FALSE(closed_pop.ok());
    EXPECT_EQ(closed_pop.status().code(), core::ErrorCode::Cancelled);
}

TEST(BlockingQueueTest, SupportsMultipleProducersAndConsumers) {
    core::BlockingQueue<int> queue(64);

    constexpr int kProducerCount = 4;
    constexpr int kConsumerCount = 4;
    constexpr int kItemsPerProducer = 500;
    constexpr int kStopValue = -1;

    std::atomic<int> consumed_count{0};
    std::atomic<long long> consumed_sum{0};
    std::atomic<int> failures{0};
    std::vector<std::thread> consumers;
    std::vector<std::thread> producers;

    for (int i = 0; i < kConsumerCount; ++i) {
        consumers.emplace_back([&] {
            for (;;) {
                auto value_result = queue.WaitPop();
                if (!value_result.ok()) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                    return;
                }

                const int value = std::move(value_result).value();
                if (value == kStopValue) {
                    return;
                }

                consumed_count.fetch_add(1, std::memory_order_relaxed);
                consumed_sum.fetch_add(value, std::memory_order_relaxed);
            }
        });
    }

    for (int producer = 0; producer < kProducerCount; ++producer) {
        producers.emplace_back([&] {
            for (int value = 1; value <= kItemsPerProducer; ++value) {
                auto status = queue.Push(value);
                if (!status.ok()) {
                    failures.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    for (auto& producer : producers) {
        producer.join();
    }

    for (int i = 0; i < kConsumerCount; ++i) {
        ASSERT_TRUE(queue.Push(kStopValue).ok());
    }

    for (auto& consumer : consumers) {
        consumer.join();
    }

    const auto expected_count = kProducerCount * kItemsPerProducer;
    const auto expected_sum = static_cast<long long>(kProducerCount) * kItemsPerProducer * (kItemsPerProducer + 1) / 2;
    EXPECT_EQ(failures.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(consumed_count.load(std::memory_order_relaxed), expected_count);
    EXPECT_EQ(consumed_sum.load(std::memory_order_relaxed), expected_sum);
}

TEST(ThreadPoolTest, ExecutesTaskAndReportsWorkerStatus) {
    core::ThreadPool pool({2, 16, "status-test-pool"});

    auto start_status = pool.Start();
    ASSERT_TRUE(start_status.ok()) << start_status.message();

    std::promise<void> entered;
    auto entered_future = entered.get_future();
    std::promise<void> release;
    auto release_future = release.get_future().share();
    std::atomic<int> ran{0};

    auto submit_status = pool.Submit(
        [&](core::ThreadPoolContext& context) -> core::Status {
            EXPECT_EQ(context.pool_name(), "status-test-pool");
            entered.set_value();
            release_future.wait();
            ran.fetch_add(1, std::memory_order_relaxed);
            return core::Status::Ok();
        },
        {},
        "blocked-task");
    ASSERT_TRUE(submit_status.ok()) << submit_status.message();

    if (entered_future.wait_for(2s) != std::future_status::ready) {
        release.set_value();
        FAIL() << "thread pool task did not start";
    }

    auto statuses = pool.WorkerStatuses();
    ASSERT_EQ(statuses.size(), 2u);
    const auto running = std::find_if(statuses.begin(), statuses.end(), [](const core::WorkerStatus& status) {
        return status.state == core::WorkerState::Running && status.current_task == "blocked-task";
    });
    ASSERT_NE(running, statuses.end());

    release.set_value();
    pool.Shutdown(true);

    EXPECT_EQ(ran.load(std::memory_order_relaxed), 1);

    auto stats = pool.Stats();
    EXPECT_EQ(stats.worker_count, 2u);
    EXPECT_EQ(stats.completed_tasks, 1u);
    EXPECT_EQ(stats.failed_tasks, 0u);

    statuses = pool.WorkerStatuses();
    ASSERT_EQ(statuses.size(), 2u);
    for (const auto& status : statuses) {
        EXPECT_EQ(status.state, core::WorkerState::Stopped);
        EXPECT_TRUE(status.current_task.empty());
    }
}

TEST(ThreadPoolTest, SupportsConcurrentSubmitFromMultipleProducers) {
    constexpr int kWorkerCount = 4;
    constexpr int kProducerCount = 8;
    constexpr int kTasksPerProducer = 250;
    constexpr int kTotalTasks = kProducerCount * kTasksPerProducer;

    core::ThreadPool pool({kWorkerCount, kTotalTasks, "multi-submit-pool"});
    core::BucketMemoryPool memory_pool;

    auto start_status = pool.Start();
    ASSERT_TRUE(start_status.ok()) << start_status.message();

    std::promise<void> start_gate;
    auto start_signal = start_gate.get_future().share();
    std::atomic<int> accepted_tasks{0};
    std::atomic<int> rejected_tasks{0};
    std::atomic<int> executed_tasks{0};
    std::atomic<long long> executed_sum{0};
    std::atomic<int> payload_failures{0};
    std::vector<std::thread> producers;
    producers.reserve(kProducerCount);

    for (int producer = 0; producer < kProducerCount; ++producer) {
        producers.emplace_back([&, producer] {
            start_signal.wait();
            for (int task_index = 0; task_index < kTasksPerProducer; ++task_index) {
                const int value = producer * kTasksPerProducer + task_index + 1;
                auto payload_result = core::SharedMemoryBlock::allocate(memory_pool, sizeof(int), alignof(int));
                if (!payload_result.ok()) {
                    rejected_tasks.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                auto payload = std::move(payload_result).value();
                *reinterpret_cast<int*>(payload.data()) = value;

                auto submit_status = pool.Submit(
                    [&, value](core::ThreadPoolContext& context) -> core::Status {
                        if (!context.payload() || context.payload().size() < sizeof(int)) {
                            payload_failures.fetch_add(1, std::memory_order_relaxed);
                            return core::Status::Error(core::ErrorCode::InvalidArgument, "payload missing");
                        }

                        const int observed = *reinterpret_cast<const int*>(context.payload().data());
                        if (observed != value) {
                            payload_failures.fetch_add(1, std::memory_order_relaxed);
                            return core::Status::Error(core::ErrorCode::InternalError, "payload mismatch");
                        }

                        executed_tasks.fetch_add(1, std::memory_order_relaxed);
                        executed_sum.fetch_add(value, std::memory_order_relaxed);
                        return core::Status::Ok();
                    },
                    std::move(payload),
                    "producer-task");

                if (submit_status.ok()) {
                    accepted_tasks.fetch_add(1, std::memory_order_relaxed);
                } else {
                    rejected_tasks.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    start_gate.set_value();
    for (auto& producer : producers) {
        producer.join();
    }

    ASSERT_EQ(rejected_tasks.load(std::memory_order_relaxed), 0);
    ASSERT_EQ(accepted_tasks.load(std::memory_order_relaxed), kTotalTasks);

    pool.Shutdown(true);

    const auto expected_sum = static_cast<long long>(kTotalTasks) * (kTotalTasks + 1) / 2;
    EXPECT_EQ(payload_failures.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(executed_tasks.load(std::memory_order_relaxed), kTotalTasks);
    EXPECT_EQ(executed_sum.load(std::memory_order_relaxed), expected_sum);

    auto stats = pool.Stats();
    EXPECT_EQ(stats.worker_count, static_cast<std::size_t>(kWorkerCount));
    EXPECT_EQ(stats.submitted_tasks, static_cast<std::size_t>(kTotalTasks));
    EXPECT_EQ(stats.completed_tasks, static_cast<std::size_t>(kTotalTasks));
    EXPECT_EQ(stats.failed_tasks, 0u);
    EXPECT_EQ(stats.rejected_tasks, 0u);

    auto memory_stats = memory_pool.stats();
    EXPECT_EQ(memory_stats.allocations, static_cast<std::size_t>(kTotalTasks));
    EXPECT_EQ(memory_stats.deallocations, static_cast<std::size_t>(kTotalTasks));
    EXPECT_EQ(memory_stats.allocated_bytes, 0u);
}

TEST(ThreadPoolTest, ShutdownDrainCompletesQueuedTasks) {
    constexpr int kTaskCount = 128;

    core::ThreadPool pool({1, kTaskCount, "drain-test-pool"});

    auto start_status = pool.Start();
    ASSERT_TRUE(start_status.ok()) << start_status.message();

    std::promise<void> first_task_entered;
    auto first_task_entered_future = first_task_entered.get_future();
    std::promise<void> release_first_task;
    auto release_first_task_future = release_first_task.get_future().share();
    std::atomic<int> executed_tasks{0};
    std::atomic<bool> first_task_notified{false};

    for (int i = 0; i < kTaskCount; ++i) {
        auto submit_status = pool.Submit(
            [&, i] {
                if (i == 0) {
                    if (!first_task_notified.exchange(true, std::memory_order_relaxed)) {
                        first_task_entered.set_value();
                    }
                    release_first_task_future.wait();
                }
                executed_tasks.fetch_add(1, std::memory_order_relaxed);
                return core::Status::Ok();
            },
            {},
            "drain-task");
        ASSERT_TRUE(submit_status.ok()) << submit_status.message();
    }

    ASSERT_EQ(first_task_entered_future.wait_for(2s), std::future_status::ready);

    std::thread shutdown_thread([&] {
        pool.Shutdown(true);
    });

    std::this_thread::sleep_for(20ms);
    EXPECT_LT(executed_tasks.load(std::memory_order_relaxed), kTaskCount);

    release_first_task.set_value();
    shutdown_thread.join();

    EXPECT_EQ(executed_tasks.load(std::memory_order_relaxed), kTaskCount);

    auto stats = pool.Stats();
    EXPECT_EQ(stats.submitted_tasks, static_cast<std::size_t>(kTaskCount));
    EXPECT_EQ(stats.completed_tasks, static_cast<std::size_t>(kTaskCount));
    EXPECT_EQ(stats.failed_tasks, 0u);
    EXPECT_EQ(stats.rejected_tasks, 0u);
}

TEST(ThreadPoolTest, RecordsFailedTaskStatus) {
    core::ThreadPool pool({1, 4, "failure-test-pool"});

    auto start_status = pool.Start();
    ASSERT_TRUE(start_status.ok()) << start_status.message();

    auto submit_status = pool.Submit(
        [] {
            return core::Status::Error(core::ErrorCode::InvalidArgument, "bad task");
        },
        {},
        "bad-task");
    ASSERT_TRUE(submit_status.ok()) << submit_status.message();

    bool observed_failure = false;
    for (int i = 0; i < 100; ++i) {
        auto statuses = pool.WorkerStatuses();
        if (!statuses.empty() && statuses[0].failed_tasks == 1) {
            observed_failure = true;
            break;
        }
        std::this_thread::sleep_for(10ms);
    }

    ASSERT_TRUE(observed_failure);

    auto statuses = pool.WorkerStatuses();
    ASSERT_EQ(statuses.size(), 1u);
    EXPECT_EQ(statuses[0].failed_tasks, 1u);
    EXPECT_EQ(statuses[0].completed_tasks, 0u);
    EXPECT_EQ(statuses[0].last_status.code(), core::ErrorCode::InvalidArgument);
    EXPECT_EQ(statuses[0].last_status.message(), "bad task");

    pool.Shutdown(true);

    auto stats = pool.Stats();
    EXPECT_EQ(stats.completed_tasks, 0u);
    EXPECT_EQ(stats.failed_tasks, 1u);
}

TEST(ThreadPoolTest, DeferredCompletionReleasesWorkerAndDrainWaitsForContinuation) {
    core::ThreadPool pool({1, 8, "deferred-completion-pool"});
    ASSERT_TRUE(pool.Start().ok());

    std::promise<core::DeferredTaskCompletion> deferred_ready;
    auto deferred_future = deferred_ready.get_future();
    std::promise<void> short_task_ran;
    auto short_task_future = short_task_ran.get_future();

    ASSERT_TRUE(pool.Submit(
        [&](core::ThreadPoolContext& context) -> core::Status {
            auto token = context.DeferCompletion();
            if (!token.ok()) {
                return token.status();
            }
            deferred_ready.set_value(std::move(token).value());
            return core::Status::Ok();
        },
        {},
        "start-async-operation").ok());
    ASSERT_EQ(deferred_future.wait_for(2s), std::future_status::ready);
    auto deferred = deferred_future.get();

    // 单 worker 已经可继续执行，证明异步等待没有占住工作线程。
    ASSERT_TRUE(pool.Submit([&] { short_task_ran.set_value(); }).ok());
    ASSERT_EQ(short_task_future.wait_for(2s), std::future_status::ready);
    EXPECT_EQ(pool.Stats().completed_tasks, 1u);

    std::atomic<bool> shutdown_returned{false};
    std::jthread shutdown([&] {
        pool.Shutdown(true);
        shutdown_returned.store(true, std::memory_order_release);
    });
    std::this_thread::sleep_for(20ms);
    EXPECT_FALSE(shutdown_returned.load(std::memory_order_acquire));

    deferred.Complete();
    shutdown.join();
    EXPECT_TRUE(shutdown_returned.load(std::memory_order_acquire));
    EXPECT_EQ(pool.Stats().completed_tasks, 2u);
    EXPECT_EQ(pool.Stats().failed_tasks, 0u);
}

TEST(TaskGroupTest, TracksNestedTasksAcrossPoolsAfterSeal) {
    core::ThreadPool compute({1, 8, "task-group-compute"});
    core::ThreadPool io({1, 8, "task-group-io"});
    ASSERT_TRUE(compute.Start().ok());
    ASSERT_TRUE(io.Start().ok());

    core::TaskGroup group;
    std::promise<void> root_entered;
    auto root_entered_future = root_entered.get_future();
    std::promise<void> release_root;
    auto release_root_future = release_root.get_future().share();
    std::atomic<int> completed{0};

    auto submit = compute.Submit(
        group,
        [&](core::ThreadPoolContext& context) -> core::Status {
            root_entered.set_value();
            release_root_future.wait();
            auto child = context.AcquireChildTask();
            if (!child.ok()) {
                return child.status();
            }
            auto child_submit = io.Submit(
                std::move(child).value(),
                [&] {
                    completed.fetch_add(1, std::memory_order_relaxed);
                    return core::Status::Ok();
                },
                {},
                "task-group-child");
            if (!child_submit.ok()) {
                return child_submit;
            }
            completed.fetch_add(1, std::memory_order_relaxed);
            return core::Status::Ok();
        },
        {},
        "task-group-root");
    ASSERT_TRUE(submit.ok()) << submit.message();
    ASSERT_EQ(root_entered_future.wait_for(2s), std::future_status::ready);

    ASSERT_TRUE(group.Seal().ok());
    EXPECT_EQ(compute.Submit(group, [] { return core::Status::Ok(); }).code(),
              core::ErrorCode::FailedPrecondition);
    release_root.set_value();

    auto drained = group.WaitFor(2s);
    ASSERT_TRUE(drained.ok()) << drained.status().message();
    EXPECT_TRUE(drained.value().drained);
    EXPECT_EQ(drained.value().outstanding_tasks, 0u);
    EXPECT_EQ(drained.value().completed_tasks, 2u);
    EXPECT_EQ(drained.value().failed_tasks, 0u);
    EXPECT_EQ(completed.load(std::memory_order_relaxed), 2);

    compute.Shutdown(true);
    io.Shutdown(true);
}

TEST(TaskGroupTest, ReportsRejectedTrackedTaskAsFailure) {
    core::ThreadPool pool({1, 1, "task-group-reject"});
    core::TaskGroup group;

    auto submit = pool.Submit(group, [] { return core::Status::Ok(); });
    EXPECT_EQ(submit.code(), core::ErrorCode::Unavailable);
    ASSERT_TRUE(group.Seal().ok());

    auto drained = group.WaitFor(100ms);
    ASSERT_TRUE(drained.ok()) << drained.status().message();
    EXPECT_EQ(drained.value().completed_tasks, 0u);
    EXPECT_EQ(drained.value().failed_tasks, 1u);
    EXPECT_EQ(drained.value().first_failure.code(), core::ErrorCode::Unavailable);
}

TEST(TaskGroupTest, InvokesDrainedCallbackOnceAfterLastTask) {
    core::ThreadPool pool({1, 4, "task-group-callback"});
    ASSERT_TRUE(pool.Start().ok());
    core::TaskGroup group;
    std::atomic<int> callback_count{0};
    std::promise<void> callback_called;
    auto callback_future = callback_called.get_future();

    ASSERT_TRUE(group.OnDrained([&] {
        EXPECT_TRUE(group.Snapshot().drained);
        callback_count.fetch_add(1, std::memory_order_relaxed);
        callback_called.set_value();
    }).ok());
    ASSERT_TRUE(pool.Submit(group, [] { return core::Status::Ok(); }).ok());
    ASSERT_TRUE(group.Seal().ok());
    ASSERT_EQ(callback_future.wait_for(2s), std::future_status::ready);
    EXPECT_EQ(callback_count.load(std::memory_order_relaxed), 1);

    auto drained = group.WaitFor(2s);
    ASSERT_TRUE(drained.ok()) << drained.status().message();
    pool.Shutdown(true);
}

TEST(UniqueHandleTest, MoveReleaseAndResetFollowResourceTraits) {
    test_resources::fake_close_count.store(0, std::memory_order_relaxed);
    test_resources::fake_last_closed.store(0, std::memory_order_relaxed);

    {
        auto handle = core::make_unique_handle<test_resources::FakeResource>(7);
        ASSERT_TRUE(handle);

        auto moved = std::move(handle);
        EXPECT_FALSE(handle);
        ASSERT_TRUE(moved);
        EXPECT_EQ(moved.get(), 7);
    }

    EXPECT_EQ(test_resources::fake_close_count.load(std::memory_order_relaxed), 1);
    EXPECT_EQ(test_resources::fake_last_closed.load(std::memory_order_relaxed), 7);

    {
        auto handle = core::make_unique_handle<test_resources::FakeResource>(8);
        const auto raw = handle.release();
        EXPECT_EQ(raw, 8);
        EXPECT_FALSE(handle);
    }

    EXPECT_EQ(test_resources::fake_close_count.load(std::memory_order_relaxed), 1);

    {
        core::UniqueHandle<test_resources::FakeResource> handle;
        handle.reset(9);
        handle.reset(10);
        EXPECT_EQ(test_resources::fake_close_count.load(std::memory_order_relaxed), 2);
        EXPECT_EQ(test_resources::fake_last_closed.load(std::memory_order_relaxed), 9);
    }

    EXPECT_EQ(test_resources::fake_close_count.load(std::memory_order_relaxed), 3);
    EXPECT_EQ(test_resources::fake_last_closed.load(std::memory_order_relaxed), 10);
}

} // namespace
