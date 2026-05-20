#include "hf_tokenizer.h"
#include "tokenizer_pool.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using vector::HfTokenizer;
using vector::TokenizerLease;
using vector::TokenizerPool;
using vector::TokenizerPoolOptions;

const char* kFixtureEnv = "HF_TOKENIZER_FIXTURE_JSON";

std::filesystem::path FixturePath() {
    const char* env = std::getenv(kFixtureEnv);
    if (env == nullptr || *env == '\0') {
        return {};
    }
    return std::filesystem::path(env);
}

bool FixtureAvailable() {
    auto p = FixturePath();
    if (p.empty()) {
        return false;
    }
    std::error_code ec;
    return std::filesystem::exists(p, ec) && !ec;
}

#define REQUIRE_FIXTURE()                                                                 \
    do {                                                                                  \
        if (!FixtureAvailable()) {                                                        \
            GTEST_SKIP() << "Set " << kFixtureEnv << " to a tokenizer.json file to run "  \
                         << "tokenizer-pool tests.";                                      \
        }                                                                                 \
    } while (0)

HfTokenizer LoadFixture() {
    auto result = HfTokenizer::LoadFromFile(FixturePath());
    EXPECT_TRUE(result.ok()) << result.status().message();
    return std::move(result).value();
}

std::shared_ptr<TokenizerPool> BuildPool(const HfTokenizer& source, std::size_t n,
                                         std::chrono::milliseconds timeout =
                                             std::chrono::milliseconds{1000}) {
    TokenizerPoolOptions opts;
    opts.size = n;
    opts.default_acquire_timeout = timeout;
    auto result = TokenizerPool::Create(source, opts);
    EXPECT_TRUE(result.ok()) << result.status().message();
    return std::move(result).value();
}

} // namespace

TEST(TokenizerPoolUnitTest, CreateRejectsZeroSize) {
    HfTokenizer dummy;
    TokenizerPoolOptions opts;
    opts.size = 0;
    auto result = TokenizerPool::Create(dummy, opts);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), core::ErrorCode::InvalidArgument);
}

TEST(TokenizerPoolUnitTest, CreateRejectsInvalidSourceTokenizer) {
    HfTokenizer invalid;
    TokenizerPoolOptions opts;
    opts.size = 2;
    auto result = TokenizerPool::Create(invalid, opts);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), core::ErrorCode::FailedPrecondition);
}

TEST(TokenizerPoolE2ETest, BuildAndAcquireRelease) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    auto pool = BuildPool(tk, 2);
    ASSERT_NE(pool, nullptr);
    EXPECT_EQ(pool->size(), 2u);
    EXPECT_EQ(pool->available(), 2u);

    {
        auto lease = pool->Acquire();
        ASSERT_TRUE(lease.ok()) << lease.status().message();
        EXPECT_EQ(pool->available(), 1u);

        auto enc = lease.value()->Encode("hello pool");
        ASSERT_TRUE(enc.ok()) << enc.status().message();
        EXPECT_EQ(enc.value().batch_size, 1u);
    }
    EXPECT_EQ(pool->available(), 2u);
}

TEST(TokenizerPoolE2ETest, TryAcquireExhaustionReturnsNotFound) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    auto pool = BuildPool(tk, 1);

    auto first = pool->TryAcquire();
    ASSERT_TRUE(first.ok());

    auto second = pool->TryAcquire();
    ASSERT_FALSE(second.ok());
    EXPECT_EQ(second.status().code(), core::ErrorCode::NotFound);
}

TEST(TokenizerPoolE2ETest, AcquireBlocksUntilReleased) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    auto pool = BuildPool(tk, 1, std::chrono::milliseconds{2000});

    auto held = pool->Acquire();
    ASSERT_TRUE(held.ok());

    std::atomic<bool> woke{false};
    auto fut = std::async(std::launch::async, [&]() {
        auto lease = pool->Acquire(std::chrono::milliseconds{2000});
        woke = true;
        return lease.ok();
    });

    std::this_thread::sleep_for(std::chrono::milliseconds{200});
    EXPECT_FALSE(woke.load());

    held.value().Release();

    EXPECT_TRUE(fut.get());
    EXPECT_TRUE(woke.load());
}

TEST(TokenizerPoolE2ETest, AcquireTimesOutWhenExhausted) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    auto pool = BuildPool(tk, 1);

    auto held = pool->Acquire();
    ASSERT_TRUE(held.ok());

    auto start = std::chrono::steady_clock::now();
    auto blocked = pool->Acquire(std::chrono::milliseconds{120});
    auto elapsed = std::chrono::steady_clock::now() - start;

    ASSERT_FALSE(blocked.ok());
    EXPECT_EQ(blocked.status().code(), core::ErrorCode::Timeout);
    EXPECT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 100);
}

TEST(TokenizerPoolE2ETest, ConcurrentCallersAchieveParallelism) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    constexpr std::size_t kPoolSize = 4;
    constexpr int kThreads = 8;
    constexpr int kIters = 64;
    auto pool = BuildPool(tk, kPoolSize, std::chrono::milliseconds{5000});

    std::vector<std::future<bool>> futures;
    for (int i = 0; i < kThreads; ++i) {
        futures.emplace_back(std::async(std::launch::async, [&]() {
            for (int j = 0; j < kIters; ++j) {
                auto lease = pool->Acquire();
                if (!lease.ok()) {
                    return false;
                }
                auto enc = lease.value()->Encode("一个并发压力测试用例");
                if (!enc.ok()) {
                    return false;
                }
                if (enc.value().batch_size != 1) {
                    return false;
                }
            }
            return true;
        }));
    }
    for (auto& f : futures) {
        EXPECT_TRUE(f.get());
    }
    EXPECT_EQ(pool->available(), kPoolSize);
}

TEST(TokenizerPoolE2ETest, CloseCancelsWaiters) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    auto pool = BuildPool(tk, 1, std::chrono::milliseconds{5000});

    auto held = pool->Acquire();
    ASSERT_TRUE(held.ok());

    auto fut = std::async(std::launch::async, [&]() {
        return pool->Acquire(std::chrono::milliseconds{3000}).status().code();
    });

    std::this_thread::sleep_for(std::chrono::milliseconds{100});
    pool->Close();

    EXPECT_EQ(fut.get(), core::ErrorCode::Cancelled);
    EXPECT_TRUE(pool->closed());
}

TEST(TokenizerPoolE2ETest, ReleaseAfterCloseDoesNotCrash) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    auto pool = BuildPool(tk, 2);

    auto lease = pool->Acquire();
    ASSERT_TRUE(lease.ok());

    pool->Close();
    // Destroying the lease here should drop the tokenizer locally.
    lease.value().Release();
    SUCCEED();
}

TEST(TokenizerPoolE2ETest, LeaseOutlivingPoolIsSafe) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    TokenizerLease lease;
    {
        auto pool = BuildPool(tk, 1);
        auto acquired = pool->Acquire();
        ASSERT_TRUE(acquired.ok());
        lease = std::move(acquired).value();
    }
    // pool destroyed; lease still holds a live tokenizer.
    ASSERT_TRUE(lease.valid());
    auto enc = lease->Encode("post-pool destruction");
    ASSERT_TRUE(enc.ok()) << enc.status().message();
    lease.Release();
    SUCCEED();
}
