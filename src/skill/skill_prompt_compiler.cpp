#include "skill_prompt_compiler.h"

#include <sstream>
#include <unordered_set>
#include <nlohmann/json.hpp>

namespace agent::skill {

core::Result<std::vector<SkillCall>> ParseToolCalls(
    const llm::ChatCompletionResponse& response) {
    std::vector<SkillCall> calls;
    for (const auto& tool_call : response.tool_calls) {
        if (tool_call.name.empty() || tool_call.arguments_json.empty()) {
            return core::Status::Error(core::ErrorCode::InvalidArgument,
                                       "tool call is missing name or arguments");
        }
        try {
            const auto arguments = nlohmann::json::parse(tool_call.arguments_json);
            if (!arguments.is_object()) {
                return core::Status::Error(core::ErrorCode::InvalidArgument,
                                           "tool call arguments must be a JSON object");
            }
        } catch (const nlohmann::json::exception& error) {
            return core::Status::Error(core::ErrorCode::InvalidArgument,
                                       std::string("invalid tool call arguments: ") + error.what());
        }
        calls.push_back(SkillCall{tool_call.id, tool_call.name, {}, {}, tool_call.arguments_json});
    }
    if (!calls.empty()) return calls;
    return core::Status::Error(core::ErrorCode::NotFound, "LLM response contains no tool calls");
}

llm::ChatMessage MakeToolResultMessage(const SkillResult& result) {
    llm::ChatMessage message;
    message.role = llm::ChatRole::Tool;
    message.tool_call_id = result.call_id;
    if (result.status.ok()) {
        message.content = result.result_json.empty() ? "{}" : result.result_json;
    } else {
        nlohmann::json error = {
            {"error", {
                {"code", static_cast<int>(result.status.code())},
                {"message", result.status.message()}
            }}
        };
        message.content = error.dump();
    }
    return message;
}

core::Result<std::vector<SkillCall>> ParseToolCalls(
    const llm::ChatCompletionResponse& response,
    const ISkillRegistry& registry) {
    auto parsed = ParseToolCalls(response);
    if (!parsed.ok()) return parsed.status();
    for (auto& call : parsed.value()) {
        auto manifest = registry.RequireByToolName(call.tool_name);
        if (!manifest.ok() && manifest.status().code() == core::ErrorCode::NotFound) {
            // 兼容早期 provider：旧协议把 skill_id 直接作为 function name。
            manifest = registry.Require(call.tool_name);
        }
        if (!manifest.ok()) return manifest.status();
        try {
            const auto schema = nlohmann::json::parse(manifest.value().input_schema_json);
            const auto arguments = nlohmann::json::parse(call.arguments_json);
            if (schema.contains("required") && schema["required"].is_array()) {
                for (const auto& required : schema["required"]) {
                    if (required.is_string() && !arguments.contains(required.get<std::string>())) {
                        return core::Status::Error(core::ErrorCode::InvalidArgument,
                                                   "tool call is missing required argument");
                    }
                }
            }
        } catch (const nlohmann::json::exception& error) {
            return core::Status::Error(core::ErrorCode::InvalidArgument,
                                       std::string("invalid tool manifest schema: ") + error.what());
        }
        call.skill_id = manifest.value().skill_id;
        call.skill_version = manifest.value().version;
    }
    return parsed;
}

SkillPromptCompiler::SkillPromptCompiler(std::shared_ptr<const ISkillRegistry> registry)
    : registry_(std::move(registry)) {}

core::Result<std::string> SkillPromptCompiler::Compile(const SkillPromptRequest& request) const {
    if (!registry_) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition,
                                   "skill registry is not configured");
    }
    std::ostringstream out;
    out << "<skill_tool_protocol>\n"
        << "Use only the declared tools and valid JSON arguments.\n"
        << "Do not claim an external result before the tool returns.\n"
        << "</skill_tool_protocol>\n";
    std::unordered_set<std::string> seen;
    for (const auto& id : request.skill_ids) {
        if (!seen.insert(id).second) continue;
        auto manifest = registry_->Require(id);
        if (!manifest.ok()) return manifest.status();
        out << "<skill_tool id=\"" << manifest.value().skill_id << "\" name=\""
            << manifest.value().tool_name << "\" version=\""
            << manifest.value().version << "\">\n"
            << "description: " << manifest.value().description << "\n"
            << "instruction: " << manifest.value().prompt_instruction << "\n"
            << "input_schema: " << manifest.value().input_schema_json << "\n"
            << "</skill_tool>\n";
    }
    if (!request.session_state.empty()) out << "<skill_session_state>\n" << request.session_state << "\n</skill_session_state>\n";
    if (!request.external_context.empty()) out << "<skill_external_context>\n" << request.external_context << "\n</skill_external_context>\n";
    return out.str();
}

} // namespace agent::skill
