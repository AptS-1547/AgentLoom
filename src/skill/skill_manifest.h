#pragma once

#include "../core/result.h"

#include <cstddef>
#include <string>
#include <vector>

namespace agent::skill {

struct SkillExecutorSpec {
    std::string type = "native";
    std::string reference;
    std::string configuration_json = "{}";
};

enum class ExecutionMode {
    Exclusive,
    SerialQueue,
    BoundedParallel,
    StreamWindowed,
};

struct SkillManifest {
    std::string skill_id;
    // OpenAI-compatible function name；与可含点号的 skill_id 分离。
    std::string tool_name;
    std::string version;
    std::string kind = "native";
    std::string description;
    std::string input_schema_json = R"({"type":"object"})";
    std::string output_schema_json = R"({"type":"object"})";
    std::string prompt_instruction;
    // 触发该工具的关键词；由 provider 编译为确定性正则兜底通道，与向量召回互补。
    std::vector<std::string> keywords;
    ExecutionMode execution_mode = ExecutionMode::Exclusive;
    std::size_t max_parallel_per_session = 1;
    bool requires_confirmation = false;
    SkillExecutorSpec executor;
};

core::Status ValidateManifest(const SkillManifest& manifest);

} // namespace agent::skill
