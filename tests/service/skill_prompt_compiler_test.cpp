#include "../../src/skill/skill_prompt_compiler.h"
#include <gtest/gtest.h>

TEST(SkillPromptCompilerTest, EmitsStaticProtocolAndManifest) {
    auto registry = std::make_shared<agent::skill::InMemorySkillRegistry>();
    agent::skill::SkillManifest m;
    m.skill_id = "vision.observe";
    m.version = "1.0.0";
    m.description = "observe";
    m.prompt_instruction = "call when observation is required";
    m.executor.reference = m.skill_id;
    ASSERT_TRUE(registry->Register(m).ok());
    agent::skill::SkillPromptCompiler compiler(registry);
    auto prompt = compiler.Compile({{"vision.observe"}, {}, {}});
    ASSERT_TRUE(prompt.ok()) << prompt.status().message();
    EXPECT_NE(prompt.value().find("skill_tool_protocol"), std::string::npos);
    EXPECT_NE(prompt.value().find("vision.observe"), std::string::npos);
    EXPECT_NE(prompt.value().find("input_schema"), std::string::npos);
}

TEST(SkillCallParserTest, ParsesStandardToolCallAndValidatesRequiredArgument) {
    auto registry = std::make_shared<agent::skill::InMemorySkillRegistry>();
    agent::skill::SkillManifest manifest;
    manifest.skill_id = "vision.observe";
    manifest.version = "1.0.0";
    manifest.input_schema_json = R"({"type":"object","required":["reason"]})";
    manifest.executor.reference = manifest.skill_id;
    ASSERT_TRUE(registry->Register(manifest).ok());
    agent::llm::ChatCompletionResponse response;
    response.tool_calls.push_back({"call-1", "vision.observe", R"({"reason":"look"})"});
    auto calls = agent::skill::ParseToolCalls(response, *registry);
    ASSERT_TRUE(calls.ok()) << calls.status().message();
    ASSERT_EQ(calls.value().size(), 1u);
    EXPECT_EQ(calls.value()[0].skill_version, "1.0.0");
}

TEST(SkillCallParserTest, RejectsMissingRequiredArgument) {
    auto registry = std::make_shared<agent::skill::InMemorySkillRegistry>();
    auto manifest = agent::skill::SkillManifest{};
    manifest.skill_id = "vision.observe";
    manifest.version = "1.0.0";
    manifest.input_schema_json = R"({"type":"object","required":["reason"]})";
    manifest.executor.reference = manifest.skill_id;
    ASSERT_TRUE(registry->Register(manifest).ok());
    agent::llm::ChatCompletionResponse response;
    response.tool_calls.push_back({"call-1", "vision.observe", "{}"});
    EXPECT_EQ(agent::skill::ParseToolCalls(response, *registry).status().code(),
              core::ErrorCode::InvalidArgument);
}

TEST(SkillPromptCompilerTest, BuildsToolResultMessageForFollowUpTurn) {
    agent::skill::SkillResult result;
    result.call_id = "call-1";
    result.skill_id = "vision.observe";
    result.result_json = R"({"observed":true})";
    auto message = agent::skill::MakeToolResultMessage(result);
    EXPECT_EQ(message.role, agent::llm::ChatRole::Tool);
    EXPECT_EQ(message.tool_call_id, "call-1");
    EXPECT_EQ(message.content, R"({"observed":true})");
}
