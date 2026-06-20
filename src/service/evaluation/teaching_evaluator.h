#pragma once

#include "redis_connection_pool.h"
#include "semantic_cache_pipeline.h"

#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace agent::service::evaluation {

struct EvalTurn {
    std::string teacher_input;
    std::string student_response;
    std::string emotion = "neutral";
    double intensity = 0.0;
    std::string behavior;
    std::string tone;
    std::string timestamp;
    std::int64_t created_at_ms = 0;
    std::string persona;
    std::optional<double> valence;
    std::optional<double> arousal;
};

struct OUParams {
    double baseline_valence = 0.15;
    double baseline_arousal = 0.28;
    double alpha = 0.75;
    double delta = 0.15;
    double negativity_bias = 1.3;
};

struct TeachingEvaluationRequest {
    std::string user_uuid;
    std::string session_id;
    std::string trace_id;
    std::filesystem::path config_path;
    OUParams ou;
    std::shared_ptr<semantic_cache::RedisConnectionPool> redis_pool;
    int max_records = 10000;
};

class TeachingEvaluator final {
public:
    core::Result<nlohmann::json> Evaluate(const TeachingEvaluationRequest& request) const;
    static std::vector<std::string> SplitSentences(const std::string& text);

private:
    struct IndicatorConfig {
        std::string key;
        std::string name;
        std::string apply_to = "teacher";
        std::vector<std::string> keywords;
        double weight = 1.0;
        bool negative = false;
        std::optional<std::regex> keyword_regex;
    };

    core::Result<std::vector<EvalTurn>> LoadTurns(const TeachingEvaluationRequest& request) const;
    core::Result<nlohmann::json> LoadConfig(const std::filesystem::path& path) const;
    nlohmann::json EvaluateBcei(const std::vector<EvalTurn>& turns,
                                const OUParams& ou,
                                int min_window,
                                int max_gap) const;

    static EvalTurn TurnFromRecord(const storage::CacheRecord& record);
    static double ComputePhiV(double valence, const OUParams& ou);
};

} // namespace agent::service::evaluation
