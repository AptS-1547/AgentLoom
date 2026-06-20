#include "teaching_evaluator.h"

#include <gtest/gtest.h>

namespace agent::service::evaluation {

TEST(TeachingEvaluatorTest, SplitsUtf8ChinesePunctuationWithoutCorruptingText) {
    const auto sentences = TeachingEvaluator::SplitSentences("所谓惯性。比如小车刹车，人会向前倾？注意观察");

    ASSERT_EQ(sentences.size(), 4u);
    EXPECT_EQ(sentences[0], "所谓惯性");
    EXPECT_EQ(sentences[1], "比如小车刹车");
    EXPECT_EQ(sentences[2], "人会向前倾");
    EXPECT_EQ(sentences[3], "注意观察");
}

TEST(TeachingEvaluatorTest, KeepsUtf8ChineseTextWhenOnlyAsciiPunctuationAppears) {
    const auto sentences = TeachingEvaluator::SplitSentences("比如苹果落下,说明重力存在.注意方向");

    ASSERT_EQ(sentences.size(), 3u);
    EXPECT_EQ(sentences[0], "比如苹果落下");
    EXPECT_EQ(sentences[1], "说明重力存在");
    EXPECT_EQ(sentences[2], "注意方向");
}

} // namespace agent::service::evaluation
