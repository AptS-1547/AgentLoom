#include "keyed_serial_executor.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

TEST(KeyedSerialExecutorTest, PreservesPerKeyOrderWhileDifferentKeysRunConcurrently) {
    auto pool = std::make_shared<core::ThreadPool>(core::ThreadPoolOptions{
        .worker_count = 4,
        .queue_capacity = 256,
        .name = "keyed-serial-test",
    });
    ASSERT_TRUE(pool->Start().ok());
    auto executor = std::make_shared<core::KeyedSerialExecutor>(pool);

    std::mutex records_mutex;
    std::vector<int> first_records;
    std::vector<int> second_records;
    std::atomic<int> first_active{0};
    std::atomic<int> second_active{0};
    std::atomic<int> all_active{0};
    std::atomic<int> max_all_active{0};
    std::atomic<bool> overlap_within_key{false};

    auto submit = [&](std::string key,
                      int value,
                      std::atomic<int>& key_active,
                      std::vector<int>& records) {
        return executor->Submit(key, [&, value]() -> core::Status {
            if (key_active.fetch_add(1, std::memory_order_acq_rel) != 0) {
                overlap_within_key.store(true, std::memory_order_release);
            }
            const int active = all_active.fetch_add(1, std::memory_order_acq_rel) + 1;
            int observed = max_all_active.load(std::memory_order_relaxed);
            while (active > observed &&
                   !max_all_active.compare_exchange_weak(observed, active, std::memory_order_relaxed)) {}
            std::this_thread::sleep_for(200us);
            {
                std::lock_guard lock(records_mutex);
                records.push_back(value);
            }
            all_active.fetch_sub(1, std::memory_order_acq_rel);
            key_active.fetch_sub(1, std::memory_order_acq_rel);
            return core::Status::Ok();
        });
    };

    for (int index = 0; index < 64; ++index) {
        ASSERT_TRUE(submit("stream-a", index, first_active, first_records).ok());
        ASSERT_TRUE(submit("stream-b", index, second_active, second_records).ok());
    }

    ASSERT_TRUE(executor->WaitIdle("stream-a", 5s).ok());
    ASSERT_TRUE(executor->WaitIdle("stream-b", 5s).ok());
    pool->Shutdown(true);

    EXPECT_FALSE(overlap_within_key.load(std::memory_order_acquire));
    EXPECT_GE(max_all_active.load(std::memory_order_relaxed), 2);
    ASSERT_EQ(first_records.size(), 64u);
    ASSERT_EQ(second_records.size(), 64u);
    for (int index = 0; index < 64; ++index) {
        EXPECT_EQ(first_records[static_cast<std::size_t>(index)], index);
        EXPECT_EQ(second_records[static_cast<std::size_t>(index)], index);
    }
}
