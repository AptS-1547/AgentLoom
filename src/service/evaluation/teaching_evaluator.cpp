#include "teaching_evaluator.h"

#include "result.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <set>
#include <sstream>
#include <utility>

namespace agent::service::evaluation {

namespace {

constexpr const char* kConversationPayloadType = "conversation_turn";

std::optional<double> ParseDouble(const std::unordered_map<std::string, std::string>& values,
                                  const std::string& key) {
    auto it = values.find(key);
    if (it == values.end() || it->second.empty()) {
        return std::nullopt;
    }
    try {
        return std::stod(it->second);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::string JsonString(const nlohmann::json& value, const char* key, std::string fallback = {}) {
    auto it = value.find(key);
    if (it == value.end() || !it->is_string()) {
        return fallback;
    }
    return it->get<std::string>();
}

double JsonDouble(const nlohmann::json& value, const char* key, double fallback) {
    auto it = value.find(key);
    if (it == value.end() || !it->is_number()) {
        return fallback;
    }
    return it->get<double>();
}

int JsonInt(const nlohmann::json& value, const char* key, int fallback) {
    auto it = value.find(key);
    if (it == value.end() || !it->is_number_integer()) {
        return fallback;
    }
    return it->get<int>();
}

bool JsonBool(const nlohmann::json& value, const char* key, bool fallback) {
    auto it = value.find(key);
    if (it == value.end() || !it->is_boolean()) {
        return fallback;
    }
    return it->get<bool>();
}

std::vector<std::string> JsonStringArray(const nlohmann::json& value, const char* key) {
    std::vector<std::string> out;
    auto it = value.find(key);
    if (it == value.end() || !it->is_array()) {
        return out;
    }
    for (const auto& item : *it) {
        if (item.is_string()) {
            out.push_back(item.get<std::string>());
        }
    }
    return out;
}

std::string JoinRegexAlternation(const std::vector<std::string>& keywords) {
    std::ostringstream out;
    for (std::size_t i = 0; i < keywords.size(); ++i) {
        if (i != 0) {
            out << '|';
        }
        out << keywords[i];
    }
    return out.str();
}

bool IsPositiveEmotion(const std::string& emotion) {
    static const std::set<std::string> positive = {
        "joy",
        "excitement",
        "tenderness",
        "curiosity",
    };
    return positive.contains(emotion);
}

} // namespace

core::Result<nlohmann::json> TeachingEvaluator::Evaluate(const TeachingEvaluationRequest& request) const {
    auto turns = LoadTurns(request);
    if (!turns.ok()) {
        return turns.status();
    }
    if (turns.value().empty()) {
        return core::Status::Error(core::ErrorCode::NotFound, "no evaluation turns found");
    }

    auto config = LoadConfig(request.config_path);
    if (!config.ok()) {
        return config.status();
    }

    nlohmann::json categories = nlohmann::json::object();
    const auto total_turns = static_cast<double>(turns.value().size());
    const auto& categories_cfg = config.value().at("categories");

    for (auto cat_it = categories_cfg.begin(); cat_it != categories_cfg.end(); ++cat_it) {
        const auto& cat_key = cat_it.key();
        const auto& cat_cfg = cat_it.value();
        nlohmann::json indicators = nlohmann::json::object();
        std::vector<double> weights;
        std::vector<double> scores;

        const auto& indicators_cfg = cat_cfg.at("indicators");
        for (auto ind_it = indicators_cfg.begin(); ind_it != indicators_cfg.end(); ++ind_it) {
            const auto& ind_key = ind_it.key();
            const auto& ind_cfg = ind_it.value();

            IndicatorConfig indicator;
            indicator.key = ind_key;
            indicator.name = JsonString(ind_cfg, "name", ind_key);
            indicator.apply_to = JsonString(ind_cfg, "apply_to", "teacher");
            indicator.keywords = JsonStringArray(ind_cfg, "keywords");
            indicator.weight = JsonDouble(ind_cfg, "weight", 1.0);
            indicator.negative = JsonBool(ind_cfg, "negative", false);
            if (!indicator.keywords.empty()) {
                try {
                    indicator.keyword_regex = std::regex(JoinRegexAlternation(indicator.keywords));
                } catch (const std::regex_error&) {
                    indicator.keyword_regex = std::nullopt;
                }
            }

            std::vector<nlohmann::json> hits;
            for (std::size_t turn_index = 0; turn_index < turns.value().size(); ++turn_index) {
                const auto& turn = turns.value()[turn_index];
                std::string text;
                if (indicator.apply_to == "student") {
                    text = turn.student_response;
                } else if (indicator.apply_to == "both") {
                    text = turn.teacher_input + " " + turn.student_response;
                } else {
                    text = turn.teacher_input;
                }
                bool matched_turn = false;
                for (const auto& sentence : SplitSentences(text)) {
                    if (sentence.size() < 2 || matched_turn || !indicator.keyword_regex) {
                        continue;
                    }
                    std::smatch match;
                    if (std::regex_search(sentence, match, *indicator.keyword_regex)) {
                        hits.push_back(nlohmann::json{
                            {"turn_index", turn_index},
                            {"sentence", sentence},
                            {"method", "keyword"},
                            {"matched", match.str()},
                            {"score", 1.0},
                        });
                        matched_turn = true;
                    }
                }
            }

            const auto count = static_cast<int>(hits.size());
            const auto normalized = total_turns > 0.0
                ? std::round((static_cast<double>(count) / total_turns) * 10000.0) / 10000.0
                : 0.0;
            indicators[ind_key] = nlohmann::json{
                {"name", indicator.name},
                {"count", count},
                {"normalized_score", normalized},
            };

            double score = normalized;
            if (indicator.negative) {
                score = std::max(0.0, 1.0 - score);
            }
            weights.push_back(indicator.weight);
            scores.push_back(score);
        }

        categories[cat_key] = nlohmann::json{
            {"name", JsonString(cat_cfg, "name", cat_key)},
            {"score", 0.0},
            {"indicators", indicators},
        };
        double weight_sum = 0.0;
        double score_sum = 0.0;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            weight_sum += weights[i];
            score_sum += weights[i] * scores[i];
        }
        categories[cat_key]["_weight_sum"] = weight_sum;
        categories[cat_key]["_score_sum"] = score_sum;
    }

    const auto medium = config.value().value("medium_indicators", nlohmann::json::object());
    const auto medium_cfg = medium.value("student_emotion_improvement", nlohmann::json::object());
    auto bcei = EvaluateBcei(
        turns.value(),
        request.ou,
        JsonInt(medium_cfg, "min_window_size", 3),
        JsonInt(medium_cfg, "max_gap", 1));
    if (categories.contains("psychological_communication")) {
        categories["psychological_communication"]["indicators"]["student_emotion_improvement"] = bcei;
        if (bcei.contains("score") && bcei["score"].is_number()) {
            const double weight = JsonDouble(medium_cfg, "weight", 1.5);
            categories["psychological_communication"]["_weight_sum"] =
                categories["psychological_communication"].value("_weight_sum", 0.0) + weight;
            categories["psychological_communication"]["_score_sum"] =
                categories["psychological_communication"].value("_score_sum", 0.0) + weight * bcei["score"].get<double>();
        }
    }

    for (auto cat_it = categories.begin(); cat_it != categories.end(); ++cat_it) {
        const double weight_sum = cat_it.value().value("_weight_sum", 0.0);
        const double score_sum = cat_it.value().value("_score_sum", 0.0);
        cat_it.value()["score"] = weight_sum > 0.0
            ? std::round((score_sum / weight_sum) * 10000.0) / 10000.0
            : 0.0;
        cat_it.value().erase("_weight_sum");
        cat_it.value().erase("_score_sum");
    }

    std::string persona;
    for (const auto& turn : turns.value()) {
        if (!turn.persona.empty()) {
            persona = turn.persona;
            break;
        }
    }

    return nlohmann::json{
        {"session_id", request.session_id},
        {"trace_id", request.trace_id},
        {"total_turns", turns.value().size()},
        {"persona", persona},
        {"ou_params", {
            {"baseline_valence", request.ou.baseline_valence},
            {"baseline_arousal", request.ou.baseline_arousal},
            {"alpha", request.ou.alpha},
            {"delta", request.ou.delta},
            {"negativity_bias", request.ou.negativity_bias},
        }},
        {"categories", categories},
        {"summary", nlohmann::json::object()},
    };
}

core::Result<std::vector<EvalTurn>> TeachingEvaluator::LoadTurns(const TeachingEvaluationRequest& request) const {
    if (!request.redis_pool) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "redis pool is required for evaluation");
    }
    std::vector<EvalTurn> turns;
    auto keys = request.redis_pool->Scan("cache:batch:" + request.user_uuid + ":*");
    if (!keys.ok()) {
        return keys.status();
    }

    for (const auto& key : keys.value()) {
        auto values = request.redis_pool->HGetAll(key);
        if (!values.ok()) {
            return values.status();
        }
        for (const auto& [field, serialized] : values.value()) {
            static_cast<void>(field);
            auto record = semantic_cache::DeserializeCacheRecord(serialized);
            if (!record.ok()) {
                continue;
            }
            if (record.value().metadata.session_id != request.session_id) {
                continue;
            }
            if (record.value().payload_type != kConversationPayloadType &&
                record.value().extra_metadata.find("emotion") == record.value().extra_metadata.end()) {
                continue;
            }
            turns.push_back(TurnFromRecord(record.value()));
            if (static_cast<int>(turns.size()) >= request.max_records) {
                break;
            }
        }
    }

    std::sort(turns.begin(), turns.end(), [](const EvalTurn& lhs, const EvalTurn& rhs) {
        return lhs.created_at_ms < rhs.created_at_ms;
    });
    return turns;
}

core::Result<nlohmann::json> TeachingEvaluator::LoadConfig(const std::filesystem::path& path) const {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return core::Status::Error(core::ErrorCode::NotFound, "evaluation config not found: " + path.string());
    }
    try {
        return nlohmann::json::parse(file);
    } catch (const std::exception& e) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, std::string("evaluation config parse failed: ") + e.what());
    }
}

EvalTurn TeachingEvaluator::TurnFromRecord(const storage::CacheRecord& record) {
    EvalTurn turn;
    turn.teacher_input = record.input;
    turn.student_response = record.response;
    turn.persona = record.metadata.persona_id;
    turn.created_at_ms = record.metadata.created_at_ms;
    turn.timestamp = std::to_string(record.metadata.created_at_ms);
    const auto& extra = record.extra_metadata;
    if (auto it = extra.find("emotion"); it != extra.end()) turn.emotion = it->second;
    if (auto intensity = ParseDouble(extra, "intensity")) turn.intensity = *intensity;
    if (auto it = extra.find("behavior"); it != extra.end()) turn.behavior = it->second;
    if (auto it = extra.find("tone"); it != extra.end()) turn.tone = it->second;
    turn.valence = ParseDouble(extra, "valence");
    turn.arousal = ParseDouble(extra, "arousal");
    return turn;
}

std::vector<std::string> TeachingEvaluator::SplitSentences(const std::string& text) {
    static const std::vector<std::string> separators = {
        "\n", ";", ",", ".", "!", "?",
        "。", "，", "；", "！", "？",
    };

    std::vector<std::string> out;
    std::size_t start = 0;
    while (start < text.size()) {
        std::size_t next_pos = std::string::npos;
        std::size_t next_len = 0;
        for (const auto& separator : separators) {
            const auto pos = text.find(separator, start);
            if (pos != std::string::npos && (next_pos == std::string::npos || pos < next_pos)) {
                next_pos = pos;
                next_len = separator.size();
            }
        }

        if (next_pos == std::string::npos) {
            out.push_back(text.substr(start));
            break;
        }
        if (next_pos > start) {
            out.push_back(text.substr(start, next_pos - start));
        }
        start = next_pos + next_len;
    }
    return out;
}

double TeachingEvaluator::ComputePhiV(double valence, const OUParams& ou) {
    if (valence < ou.baseline_valence && ou.negativity_bias > 1.0) {
        return ou.alpha - ou.delta / ou.negativity_bias;
    }
    return ou.alpha - ou.delta;
}

nlohmann::json TeachingEvaluator::EvaluateBcei(const std::vector<EvalTurn>& turns,
                                               const OUParams& ou,
                                               int min_window,
                                               int max_gap) const {
    int va_count = 0;
    for (const auto& turn : turns) {
        if (turn.valence) {
            ++va_count;
        }
    }
    if (va_count < min_window) {
        return {
            {"name", "学生情绪改善"},
            {"type", "medium"},
            {"score", nullptr},
            {"mean_bcei", nullptr},
            {"window_count", 0},
            {"note", "V-A data is insufficient"},
        };
    }

    std::vector<std::pair<int, int>> windows;
    for (int i = 0; i < static_cast<int>(turns.size());) {
        if (!IsPositiveEmotion(turns[static_cast<std::size_t>(i)].emotion)) {
            ++i;
            continue;
        }
        const int start = i;
        int gap_count = 0;
        int j = i + 1;
        for (; j < static_cast<int>(turns.size()); ++j) {
            if (IsPositiveEmotion(turns[static_cast<std::size_t>(j)].emotion)) {
                gap_count = 0;
            } else if (++gap_count > max_gap) {
                break;
            }
        }
        int end = j - 1;
        while (end > start && !IsPositiveEmotion(turns[static_cast<std::size_t>(end)].emotion)) {
            --end;
        }
        int positives = 0;
        for (int k = start; k <= end; ++k) {
            if (IsPositiveEmotion(turns[static_cast<std::size_t>(k)].emotion)) {
                ++positives;
            }
        }
        if (positives >= min_window) {
            windows.emplace_back(start, end);
        }
        i = end + 1;
    }

    std::vector<double> bceis;
    for (const auto& [start, end] : windows) {
        int entry = start;
        while (entry <= end && !turns[static_cast<std::size_t>(entry)].valence) ++entry;
        int exit = end;
        while (exit >= entry && !turns[static_cast<std::size_t>(exit)].valence) --exit;
        if (entry >= exit) {
            continue;
        }
        const double v_entry = *turns[static_cast<std::size_t>(entry)].valence;
        const double v_exit = *turns[static_cast<std::size_t>(exit)].valence;
        const double expected = std::pow(ComputePhiV(v_entry, ou), exit - entry) * (v_entry - ou.baseline_valence);
        const double actual = v_exit - ou.baseline_valence;
        bceis.push_back(actual - expected);
    }
    if (bceis.empty()) {
        return {
            {"name", "学生情绪改善"},
            {"type", "medium"},
            {"score", nullptr},
            {"mean_bcei", nullptr},
            {"window_count", 0},
            {"note", "window V-A data is insufficient"},
        };
    }
    const double mean = std::accumulate(bceis.begin(), bceis.end(), 0.0) / static_cast<double>(bceis.size());
    const double score = 0.5 + 0.5 * std::tanh(2.0 * mean);
    return {
        {"name", "学生情绪改善"},
        {"type", "medium"},
        {"score", std::round(score * 10000.0) / 10000.0},
        {"mean_bcei", std::round(mean * 10000.0) / 10000.0},
        {"window_count", bceis.size()},
    };
}

} // namespace agent::service::evaluation
