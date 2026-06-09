#include "document_types.h"

#include <algorithm>
#include <cmath>
#include <regex>
#include <sstream>

namespace agent::document {
namespace {

bool MatchAny(std::string_view text, const char* pattern) {
    return std::regex_search(std::string(text), std::regex(pattern, std::regex::icase));
}

std::size_t CountHeadings(const std::vector<DocumentBlock>& blocks) {
    return static_cast<std::size_t>(std::count_if(blocks.begin(), blocks.end(), [](const DocumentBlock& block) {
        return block.kind == "heading" || block.heading_level.has_value();
    }));
}

std::string JoinText(const std::vector<DocumentBlock>& blocks) {
    std::string text;
    for (const auto& block : blocks) {
        if (!block.text.empty()) {
            if (!text.empty()) {
                text.push_back('\n');
            }
            text.append(block.text);
        }
    }
    return text;
}

double Clamp01(double value) {
    return std::max(0.0, std::min(1.0, value));
}

double JsonCoverageScore(const nlohmann::json& knowledge_coverage) {
    if (knowledge_coverage.is_object() && knowledge_coverage.contains("coverage_score") && knowledge_coverage["coverage_score"].is_number()) {
        return Clamp01(knowledge_coverage["coverage_score"].get<double>());
    }
    return -1.0;
}

nlohmann::json NullKnowledgeCoverage() {
    return {
        {"enabled", false},
        {"expected_points", nlohmann::json::array()},
        {"covered_points", nlohmann::json::array()},
        {"missing_points", nlohmann::json::array()},
        {"coverage_score", nullptr},
        {"evidence", nlohmann::json::array()},
    };
}

} // namespace

nlohmann::json BuildDiagnosis(const std::vector<DocumentBlock>& blocks,
                              const nlohmann::json& mindmap,
                              const std::vector<ChunkTrunk>& chunks,
                              const nlohmann::json& knowledge_coverage) {
    (void)mindmap;
    const auto total = blocks.size();
    const auto headings = CountHeadings(blocks);
    const auto text = JoinText(blocks);

    const bool has_examples = MatchAny(text, R"((例题|示例|案例|example|case))");
    const bool has_practice = MatchAny(text, R"((练习|习题|作业|检测|思考|practice|exercise|quiz))");
    const bool has_objective = MatchAny(text, R"((目标|重点|难点|objective|goal|aim))");

    const double total_d = static_cast<double>(total);
    const double heading_d = static_cast<double>(headings);
    const double chunk_d = static_cast<double>(chunks.size());
    const double structure_score = total == 0 ? 0.0 : std::min(1.0, heading_d / std::max(3.0, total_d * 0.18));
    const double example_score = has_examples ? 1.0 : 0.35;
    const double practice_score = has_practice ? 1.0 : 0.30;
    const double objective_score = has_objective ? 1.0 : 0.45;
    double content_score = std::min(1.0, total_d / 12.0);
    const double coverage_score = JsonCoverageScore(knowledge_coverage);
    if (coverage_score >= 0.0) {
        content_score = std::max(content_score, coverage_score);
    }
    const double chunk_score = total == 0 ? 0.0 : std::min(1.0, chunk_d / std::max(1.0, total_d * 0.15));

    const double score_01 =
        0.27 * structure_score
        + 0.18 * example_score
        + 0.18 * practice_score
        + 0.14 * objective_score
        + 0.16 * content_score
        + 0.07 * chunk_score;
    const double score = std::round(std::max(1.0, std::min(10.0, score_01 * 10.0)) * 10.0) / 10.0;

    std::ostringstream coverage;
    coverage << "已提取 " << total << " 个可分析文本块、" << headings << " 个结构标题、"
             << chunks.size() << " 个语义主干。材料"
             << (has_objective ? "包含" : "未清晰呈现") << "学习目标，"
             << (has_examples ? "包含" : "未清晰包含") << "例题或案例，"
             << (has_practice ? "包含" : "未清晰包含") << "练习或检测活动。";

    std::vector<std::string> highlights;
    if (headings > 0) {
        highlights.push_back("材料包含可识别的结构标题。");
    }
    if (has_examples) {
        highlights.push_back("材料包含例题、示例或案例。");
    }
    if (has_practice) {
        highlights.push_back("材料包含练习、作业或检测活动。");
    }
    if (!chunks.empty()) {
        highlights.push_back("弱结构片段已归并为可复用语义主干。");
    }
    if (highlights.empty()) {
        highlights.push_back("材料具备可提取内容，可用于基础结构分析。");
    }

    std::string highlight;
    for (const auto& item : highlights) {
        if (!highlight.empty()) {
            highlight.push_back(' ');
        }
        highlight.append(item);
    }

    nlohmann::json suggestions = nlohmann::json::array();
    if (!has_objective) {
        suggestions.push_back("建议在开头补充明确的学习目标或教学重点。");
    }
    if (!has_examples) {
        suggestions.push_back("建议在核心概念后增加至少一个递进式例题。");
    }
    if (!has_practice) {
        suggestions.push_back("建议在关键知识点后增加短练习或课堂检测问题。");
    }
    if (headings < 3 && total >= 8) {
        suggestions.push_back("建议增加更清晰的小节标题，帮助学生理解知识结构。");
    }
    suggestions.push_back("可进入课堂模拟环节进行讲解演练。");

    nlohmann::json summary = {
        {"knowlege_point", nlohmann::json::array()},
        {"ability", nlohmann::json::array()},
        {"emotion", nlohmann::json::array()},
    };

    nlohmann::json coverage_payload = knowledge_coverage.is_object() && !knowledge_coverage.empty()
        ? knowledge_coverage
        : NullKnowledgeCoverage();

    return {
        {"score", score},
        {"coverage", coverage.str()},
        {"highlight", highlight},
        {"structure", "生成的大纲基于 Office 显式结构、编号、幻灯片顺序和段落邻近关系；弱标题信号片段会挂接到最近的结构父节点。"},
        {"suggestions", suggestions},
        {"summary", summary},
        {"knowledge_coverage", coverage_payload},
        {"rule_features", {
            {"block_count", total},
            {"heading_count", headings},
            {"chunk_count", chunks.size()},
            {"has_examples", has_examples},
            {"has_practice", has_practice},
            {"has_objective", has_objective},
            {"structure_score", structure_score},
            {"content_score", content_score},
            {"chunk_score", chunk_score},
        }},
    };
}

} // namespace agent::document
