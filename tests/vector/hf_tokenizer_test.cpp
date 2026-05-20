#include "hf_tokenizer.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <future>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using vector::EncodeOptions;
using vector::HfTokenizer;
using vector::TokenizedBatch;

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
                         << "end-to-end tokenizer tests.";                                \
        }                                                                                 \
    } while (0)

HfTokenizer LoadFixture() {
    auto result = HfTokenizer::LoadFromFile(FixturePath());
    EXPECT_TRUE(result.ok()) << result.status().message();
    return std::move(result).value();
}

} // namespace

TEST(HfTokenizerAbiTest, AbiVersionIsExpected) {
    EXPECT_EQ(HfTokenizer::AbiVersion(), 1);
}

TEST(HfTokenizerAbiTest, DefaultConstructedIsInvalid) {
    HfTokenizer tk;
    EXPECT_FALSE(tk.valid());
    EXPECT_FALSE(static_cast<bool>(tk));
}

TEST(HfTokenizerAbiTest, LoadMissingFileFails) {
    auto result = HfTokenizer::LoadFromFile("__definitely_not_a_real_path__/tokenizer.json");
    ASSERT_FALSE(result.ok());
    EXPECT_FALSE(result.status().message().empty());
}

TEST(HfTokenizerAbiTest, EncodeOnInvalidHandleFailsCleanly) {
    HfTokenizer tk;
    auto enc = tk.Encode("hello");
    ASSERT_FALSE(enc.ok());
    EXPECT_EQ(enc.status().code(), core::ErrorCode::FailedPrecondition);
}

TEST(HfTokenizerAbiTest, EncodeBatchOnInvalidHandleFailsCleanly) {
    HfTokenizer tk;
    std::array<std::string_view, 1> texts{"hello"};
    auto enc = tk.EncodeBatch(std::span<const std::string_view>(texts));
    ASSERT_FALSE(enc.ok());
    EXPECT_EQ(enc.status().code(), core::ErrorCode::FailedPrecondition);
}

TEST(HfTokenizerE2ETest, LoadAndEncodeAscii) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    ASSERT_TRUE(tk.valid());

    auto enc = tk.Encode("hello world");
    ASSERT_TRUE(enc.ok()) << enc.status().message();
    const auto& batch = enc.value();
    EXPECT_EQ(batch.batch_size, 1u);
    EXPECT_GT(batch.sequence_length, 0u);
    EXPECT_EQ(batch.input_ids.size(), batch.batch_size * batch.sequence_length);
    EXPECT_EQ(batch.attention_mask.size(), batch.input_ids.size());
    EXPECT_EQ(batch.token_type_ids.size(), batch.input_ids.size());

    bool any_attended = false;
    for (auto m : batch.attention_mask) {
        if (m != 0) {
            any_attended = true;
            break;
        }
    }
    EXPECT_TRUE(any_attended);
}

TEST(HfTokenizerE2ETest, EncodeChineseUtf8) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    ASSERT_TRUE(tk.valid());

    auto enc = tk.Encode("你好，世界。今天天气怎么样？");
    ASSERT_TRUE(enc.ok()) << enc.status().message();
    EXPECT_EQ(enc.value().batch_size, 1u);
    EXPECT_GT(enc.value().sequence_length, 0u);
}

TEST(HfTokenizerE2ETest, EncodeEmptyStringDoesNotCrash) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    ASSERT_TRUE(tk.valid());

    auto enc = tk.Encode("");
    ASSERT_TRUE(enc.ok()) << enc.status().message();
    EXPECT_EQ(enc.value().batch_size, 1u);
}

TEST(HfTokenizerE2ETest, InvalidUtf8IsRejected) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    ASSERT_TRUE(tk.valid());

    const char raw[] = {static_cast<char>(0xC3), static_cast<char>(0x28), 0};
    std::string_view bad(raw, 2);
    auto enc = tk.Encode(bad);
    ASSERT_FALSE(enc.ok());
    EXPECT_EQ(enc.status().code(), core::ErrorCode::InvalidArgument);
}

TEST(HfTokenizerE2ETest, EncodeBatchPaddingShape) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    ASSERT_TRUE(tk.valid());

    std::array<std::string_view, 3> texts{
        "short",
        "a slightly longer english sentence to force padding",
        "中文一句话",
    };

    auto enc = tk.EncodeBatch(std::span<const std::string_view>(texts));
    ASSERT_TRUE(enc.ok()) << enc.status().message();
    const auto& batch = enc.value();
    EXPECT_EQ(batch.batch_size, texts.size());
    EXPECT_GT(batch.sequence_length, 0u);
    EXPECT_EQ(batch.input_ids.size(), batch.batch_size * batch.sequence_length);
    EXPECT_EQ(batch.attention_mask.size(), batch.input_ids.size());
    EXPECT_EQ(batch.token_type_ids.size(), batch.input_ids.size());

    for (std::size_t row = 0; row < batch.batch_size; ++row) {
        bool saw_attended = false;
        for (std::size_t col = 0; col < batch.sequence_length; ++col) {
            if (batch.attention_mask[row * batch.sequence_length + col] != 0) {
                saw_attended = true;
                break;
            }
        }
        EXPECT_TRUE(saw_attended) << "row " << row << " has zero attention everywhere";
    }
}

TEST(HfTokenizerE2ETest, FixedLengthPaddingFillsToMaxLength) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    ASSERT_TRUE(tk.valid());

    EncodeOptions opts;
    opts.max_length = 32;
    opts.truncation = true;
    opts.padding = true;
    opts.pad_to_longest_in_batch = false;

    auto enc = tk.Encode("ping", opts);
    ASSERT_TRUE(enc.ok()) << enc.status().message();
    EXPECT_EQ(enc.value().sequence_length, opts.max_length);
}

TEST(HfTokenizerE2ETest, RepeatedSerialEncodeIsStable) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    ASSERT_TRUE(tk.valid());

    auto first = tk.Encode("一个稳定可重复的测试用例");
    ASSERT_TRUE(first.ok()) << first.status().message();
    for (int i = 0; i < 32; ++i) {
        auto again = tk.Encode("一个稳定可重复的测试用例");
        ASSERT_TRUE(again.ok()) << again.status().message();
        EXPECT_EQ(again.value().input_ids, first.value().input_ids);
        EXPECT_EQ(again.value().attention_mask, first.value().attention_mask);
        EXPECT_EQ(again.value().token_type_ids, first.value().token_type_ids);
    }
}

TEST(HfTokenizerE2ETest, ConcurrentCallersAreSerializedSafely) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    ASSERT_TRUE(tk.valid());

    constexpr int kThreads = 8;
    constexpr int kIters = 50;

    auto baseline = tk.Encode("baseline reference text");
    ASSERT_TRUE(baseline.ok());

    std::vector<std::future<bool>> futures;
    for (int i = 0; i < kThreads; ++i) {
        futures.emplace_back(std::async(std::launch::async, [&tk, &baseline]() {
            for (int j = 0; j < kIters; ++j) {
                auto enc = tk.Encode("baseline reference text");
                if (!enc.ok()) {
                    return false;
                }
                if (enc.value().input_ids != baseline.value().input_ids) {
                    return false;
                }
            }
            return true;
        }));
    }
    for (auto& f : futures) {
        EXPECT_TRUE(f.get());
    }
}

TEST(HfTokenizerE2ETest, MoveSemanticsPreserveHandle) {
    REQUIRE_FIXTURE();
    auto tk = LoadFixture();
    ASSERT_TRUE(tk.valid());

    HfTokenizer moved = std::move(tk);
    EXPECT_FALSE(tk.valid());
    ASSERT_TRUE(moved.valid());

    auto enc = moved.Encode("post-move encode");
    ASSERT_TRUE(enc.ok()) << enc.status().message();
}
