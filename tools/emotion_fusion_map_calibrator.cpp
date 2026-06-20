#include "beast_http_client.h"
#include "embedding_pipeline.h"
#include "emotion_fusion_analyzer.h"
#include "grpc_emotion_analyzer.h"
#include "hf_tokenizer.h"
#include "onnx_text_embedding_model.h"
#include "openai_llm_client.h"
#include "thread_pool.h"
#include "tls_context.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Json = nlohmann::json;
using agent::service::persona::EmotionAnalysis;
using agent::service::persona::EmotionEvidence;
using agent::service::persona::EmotionFusionAnalyzerOptions;
using agent::service::persona::EmotionFusionFeature;
using agent::service::persona::FusedEmotionAnalyzer;

struct ToolOptions {
    fs::path dataset_path;
    fs::path config_path;
    fs::path output_path = "emotion_fusion_calibrated.json";
    int epochs = 200;
    double learning_rate = 0.03;
    double prior_lambda = 0.02;
    std::size_t max_samples = 0;
    std::size_t io_workers = 0;
    std::size_t batch_size = 1;
    std::set<std::string> labels;
    bool enable_llm = true;
    bool enable_vector = true;
    bool bert_only = false;
};

struct TrainingSample {
    std::string text;
    std::string label;
};

struct PreparedSample {
    std::string gold_label;
    std::vector<EmotionFusionFeature> features;
    bool llm_called = false;
};

struct FeatureGradient {
    double head_bias = 0.0;
    double bert_signal_weight = 0.0;
    double keyword_signal_weight = 0.0;
    double vector_signal_weight = 0.0;
    double llm_signal_weight = 0.0;
    double margin_signal_weight = 0.0;
};

struct EvalStats {
    double loss = 0.0;
    double accuracy = 0.0;
    double macro_f1 = 0.0;
    std::map<std::string, std::map<std::string, int>> confusion;
};

struct LlmGateCandidate {
    double confidence = 0.0;
    double min_delta = 0.0;
    double call_rate = 0.0;
    EvalStats stats;
};

struct PrepareContext {
    std::shared_ptr<agent::service::persona::IEmotionAnalyzer> bert;
    std::vector<std::shared_ptr<agent::service::persona::IEmotionEvidenceProvider>> providers;
    std::shared_ptr<agent::service::persona::IEmotionAnalyzer> llm_analyzer;
    EmotionFusionAnalyzerOptions options;
    std::set<std::string> labels;
    bool bert_only = false;
};

struct PrepareResult {
    std::optional<PreparedSample> sample;
    bool failed = false;
    bool llm_called = false;
    std::string error;
};

double Clamp(double value, double lo, double hi) {
    return std::max(lo, std::min(value, hi));
}

std::string GetString(const Json& json, std::string_view key, std::string fallback = {}) {
    auto it = json.find(std::string(key));
    return it == json.end() || !it->is_string() ? fallback : it->get<std::string>();
}

double GetDouble(const Json& json, std::string_view key, double fallback) {
    auto it = json.find(std::string(key));
    return it == json.end() || !it->is_number() ? fallback : it->get<double>();
}

int GetInt(const Json& json, std::string_view key, int fallback) {
    auto it = json.find(std::string(key));
    return it == json.end() || !it->is_number_integer() ? fallback : it->get<int>();
}

bool GetBool(const Json& json, std::string_view key, bool fallback) {
    auto it = json.find(std::string(key));
    return it == json.end() || !it->is_boolean() ? fallback : it->get<bool>();
}

std::set<std::string> SplitLabels(std::string_view value) {
    std::set<std::string> labels;
    std::string current;
    for (const char ch : value) {
        if (ch == ',') {
            if (!current.empty()) {
                labels.insert(current);
                current.clear();
            }
            continue;
        }
        if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        labels.insert(current);
    }
    return labels;
}

fs::path ResolvePath(const fs::path& config_path, const std::string& raw) {
    if (raw.empty()) {
        return {};
    }
    fs::path path(raw);
    if (path.is_absolute()) {
        return path;
    }
    return fs::absolute(config_path.parent_path() / path).lexically_normal();
}

std::string ReadTextFile(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return {};
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string ResolveApiKey(const fs::path& config_path, const Json& llm) {
    const auto env_name = GetString(llm, "api_key_env", "AGENT_LLM_API_KEY");
    if (!env_name.empty()) {
        if (const char* value = std::getenv(env_name.c_str()); value && *value) {
            return value;
        }
    }
    const auto key_file = GetString(llm, "api_key_file");
    if (!key_file.empty()) {
        auto key = ReadTextFile(ResolvePath(config_path, key_file));
        while (!key.empty() && (key.back() == '\n' || key.back() == '\r' || key.back() == ' ')) {
            key.pop_back();
        }
        return key;
    }
    return {};
}

std::optional<TrainingSample> ParseDatasetItem(const Json& item) {
    if (!item.is_object()) {
        return std::nullopt;
    }
    std::string text = item.value("text", item.value("content", item.value("sentence", std::string{})));
    std::string label = item.value("label", item.value("gold_label", item.value("emotion", std::string{})));
    if (text.empty() || label.empty()) {
        return std::nullopt;
    }
    return TrainingSample{std::move(text), std::move(label)};
}

std::vector<TrainingSample> LoadDataset(const fs::path& path, std::size_t max_samples) {
    std::vector<TrainingSample> samples;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open dataset: " + path.string());
    }
    const auto content = std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    if (content.empty()) {
        return samples;
    }
    if (content.front() == '[') {
        auto array = Json::parse(content);
        for (const auto& item : array) {
            auto parsed = ParseDatasetItem(item);
            if (parsed) {
                samples.push_back(std::move(*parsed));
            }
            if (max_samples > 0 && samples.size() >= max_samples) {
                break;
            }
        }
        return samples;
    }

    std::istringstream lines(content);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.empty()) {
            continue;
        }
        auto parsed = ParseDatasetItem(Json::parse(line));
        if (parsed) {
            samples.push_back(std::move(*parsed));
        }
        if (max_samples > 0 && samples.size() >= max_samples) {
            break;
        }
    }
    return samples;
}

void FilterFeaturesToLabels(std::vector<EmotionFusionFeature>& features,
                            const std::set<std::string>& labels) {
    if (labels.empty()) {
        return;
    }
    features.erase(std::remove_if(features.begin(), features.end(),
                       [&](const EmotionFusionFeature& feature) {
                           return !labels.contains(feature.label);
                       }),
                   features.end());
}

std::vector<agent::service::persona::EmotionKeywordRule> ParseKeywordRules(const Json& fusion) {
    auto rules = agent::service::persona::DefaultEmotionKeywordRules();
    auto it = fusion.find("keyword_rules");
    if (it == fusion.end() || !it->is_object()) {
        return rules;
    }
    for (const auto& [label, value] : it->items()) {
        if (value.is_string()) {
            rules.push_back({label, value.get<std::string>(), 0.95});
        } else if (value.is_array()) {
            for (const auto& item : value) {
                if (item.is_string()) {
                    rules.push_back({label, item.get<std::string>(), 0.95});
                } else if (item.is_object()) {
                    rules.push_back({label,
                                     item.value("pattern", std::string{}),
                                     item.value("score", 0.95)});
                }
            }
        } else if (value.is_object()) {
            rules.push_back({label,
                             value.value("pattern", std::string{}),
                             value.value("score", 0.95)});
        }
    }
    return rules;
}

EmotionFusionAnalyzerOptions ParseFusionOptions(const Json& fusion) {
    EmotionFusionAnalyzerOptions options;
    options.enabled = GetBool(fusion, "enabled", true);
    options.bert_weight = GetDouble(fusion, "bert_weight", options.bert_weight);
    options.default_reliability = GetDouble(fusion, "default_reliability", options.default_reliability);
    options.accept_confidence = GetDouble(fusion, "accept_confidence", options.accept_confidence);
    options.ambiguity_margin = GetDouble(fusion, "ambiguity_margin", options.ambiguity_margin);
    options.head_bias = GetDouble(fusion, "head_bias", options.head_bias);
    options.bert_signal_weight = GetDouble(fusion, "bert_signal_weight", options.bert_signal_weight);
    const double evidence_weight = GetDouble(fusion, "evidence_signal_weight", options.keyword_signal_weight);
    options.keyword_signal_weight = GetDouble(fusion, "keyword_signal_weight", evidence_weight);
    options.vector_signal_weight = GetDouble(fusion, "vector_signal_weight", evidence_weight);
    options.llm_signal_weight = GetDouble(fusion, "llm_signal_weight", 0.8);
    options.margin_signal_weight = GetDouble(fusion, "margin_signal_weight", options.margin_signal_weight);
    options.llm_gate_confidence = GetDouble(fusion, "llm_gate_confidence", options.llm_gate_confidence);
    options.llm_gate_min_delta = GetDouble(fusion, "llm_gate_min_delta", options.llm_gate_min_delta);
    if (auto it = fusion.find("label_reliability"); it != fusion.end() && it->is_object()) {
        for (const auto& [label, value] : it->items()) {
            if (value.is_number()) {
                options.label_reliability[label] = value.get<double>();
            }
        }
    }
    if (auto it = fusion.find("source_weights"); it != fusion.end() && it->is_object()) {
        for (const auto& [source, value] : it->items()) {
            if (value.is_number()) {
                options.source_weights[source] = value.get<double>();
            }
        }
    }
    return options;
}

std::shared_ptr<agent::llm::ILlmClient> CreateLlmClient(const fs::path& config_path, const Json& root) {
    const auto llm = root.value("llm", Json::object());
    if (!GetBool(llm, "enabled", true)) {
        return nullptr;
    }
    agent::llm::OpenAiLlmClientOptions options;
    options.base_url = GetString(llm, "base_url");
    options.default_model = GetString(llm, "model", "deepseek-chat");
    options.timeout_ms = GetInt(llm, "timeout_ms", 30000);
    options.retry_policy.max_retries = GetInt(llm, "max_retries", 1);
    options.require_api_key = GetBool(llm, "require_api_key", true);
    options.api_key = ResolveApiKey(config_path, llm);
    if (options.base_url.empty() || (options.require_api_key && options.api_key.empty())) {
        return nullptr;
    }
    agent::net::TlsClientOptions tls_options;
#ifdef _WIN32
    if (GetBool(llm, "disable_tls_verify_on_windows", false)) {
        tls_options.verify_mode = agent::net::TlsVerifyMode::None;
    }
#endif
    auto tls = agent::net::TlsContext::CreateClient(tls_options);
    if (!tls.ok()) {
        return nullptr;
    }
    agent::net::BeastHttpClientOptions http_options;
    http_options.tls_context = std::move(tls).value();
    auto transport_result = agent::net::BeastHttpClient::Create(std::move(http_options));
    if (!transport_result.ok()) {
        return nullptr;
    }
    auto transport = std::shared_ptr<agent::net::IHttpClient>(std::move(transport_result).value());
    auto client = agent::llm::OpenAiLlmClient::Create(options, *transport);
    if (!client.ok()) {
        return nullptr;
    }
    struct Holder final : agent::llm::ILlmClient {
        std::shared_ptr<agent::net::IHttpClient> transport;
        std::unique_ptr<agent::llm::OpenAiLlmClient> client;
        core::Result<agent::llm::ChatCompletionResponse> Complete(
            const agent::llm::ChatCompletionRequest& request) override {
            return client->Complete(request);
        }
    };
    auto holder = std::make_shared<Holder>();
    holder->transport = std::move(transport);
    holder->client = std::move(client).value();
    return holder;
}

std::shared_ptr<agent::service::persona::IEmotionAnalyzer> CreateBertAnalyzer(
    const fs::path& config_path,
    const Json& root) {
    const auto grpc_json = root.value("emotion_analyzer", root.value("grpc_emotion", Json::object()));
    const auto tokenizer_path = ResolvePath(config_path, GetString(grpc_json, "tokenizer_path"));
    auto loaded = ::vector::HfTokenizer::LoadFromFile(tokenizer_path);
    if (!loaded.ok()) {
        throw std::runtime_error("failed to load emotion tokenizer: " + loaded.status().message());
    }
    auto tokenizer = std::make_shared<::vector::HfTokenizer>(std::move(loaded).value());

    agent::service::persona::GrpcEmotionAnalyzerOptions options;
    options.target = GetString(grpc_json, "target", "127.0.0.1:50052");
    options.deadline = std::chrono::milliseconds(GetInt(grpc_json, "deadline_ms", 3000));
    options.auth_token = GetString(grpc_json, "auth_token");
    options.auth_metadata_key = GetString(grpc_json, "auth_metadata_key", options.auth_metadata_key);
    options.tokenizer_options.max_length = static_cast<std::size_t>(GetInt(grpc_json, "max_length", 128));
    return std::make_shared<agent::service::persona::GrpcEmotionAnalyzer>(options, std::move(tokenizer));
}

std::vector<std::shared_ptr<agent::service::persona::IEmotionEvidenceProvider>> CreateEvidenceProviders(
    const fs::path& config_path,
    const Json& root,
    bool enable_vector) {
    std::vector<std::shared_ptr<agent::service::persona::IEmotionEvidenceProvider>> providers;
    const auto fusion = root.value("emotion_fusion", Json::object());
    providers.push_back(std::make_shared<agent::service::persona::KeywordEmotionEvidenceProvider>(
        ParseKeywordRules(fusion)));

    if (!enable_vector || !GetBool(fusion, "vector_enabled", false)) {
        return providers;
    }
    const auto embedding_json = root.value("embedding", Json::object());
    const auto tokenizer_path = ResolvePath(config_path, GetString(embedding_json, "tokenizer_path"));
    const auto model_path = ResolvePath(config_path, GetString(embedding_json, "model_path"));
    if (tokenizer_path.empty() || model_path.empty()) {
        std::cerr << "[calibrator] vector evidence skipped: embedding tokenizer/model path missing\n";
        return providers;
    }
    auto loaded = ::vector::HfTokenizer::LoadFromFile(tokenizer_path);
    if (!loaded.ok()) {
        std::cerr << "[calibrator] vector evidence skipped: " << loaded.status().message() << "\n";
        return providers;
    }
    auto tokenizer = std::make_shared<::vector::HfTokenizer>(std::move(loaded).value());
    ::vector::EmbeddingModelOptions model_options;
    model_options.model_path = model_path;
    model_options.expected_dimension = static_cast<std::size_t>(
        GetInt(embedding_json, "expected_dimension", GetInt(embedding_json, "dimension", 0)));
    model_options.execution_provider = GetString(embedding_json, "execution_provider", "auto");
    model_options.require_token_type_ids = GetBool(embedding_json, "require_token_type_ids", false);
    auto model = ::vector::OnnxTextEmbeddingModel::Load(model_options);
    if (!model.ok()) {
        std::cerr << "[calibrator] vector evidence skipped: " << model.status().message() << "\n";
        return providers;
    }
    auto shared_model = std::shared_ptr<::vector::IEmbeddingModel>(std::move(model).value().release());
    auto pipeline = std::make_shared<::vector::EmbeddingPipeline>(std::move(tokenizer), std::move(shared_model));
    auto vector_provider = agent::service::persona::VectorEmotionEvidenceProvider::Create(
        pipeline,
        agent::service::persona::DefaultEmotionVectorPrototypes());
    if (!vector_provider.ok()) {
        std::cerr << "[calibrator] vector evidence skipped: " << vector_provider.status().message() << "\n";
        return providers;
    }
    providers.push_back(std::move(vector_provider).value());
    return providers;
}

std::string ArgMaxLabel(const std::map<std::string, double>& probabilities) {
    auto it = std::max_element(probabilities.begin(), probabilities.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
    return it == probabilities.end() ? std::string{} : it->first;
}

bool ShouldUseLlmFallback(const std::map<std::string, double>& probabilities,
                          const EmotionFusionAnalyzerOptions& options) {
    double first = 0.0;
    double second = 0.0;
    for (const auto& [_, value] : probabilities) {
        if (value > first) {
            second = first;
            first = value;
        } else if (value > second) {
            second = value;
        }
    }
    return first < options.accept_confidence || first - second < options.ambiguity_margin;
}

FeatureGradient ComputeGradient(const std::vector<EmotionFusionFeature>& features,
                                const std::map<std::string, double>& probabilities,
                                const std::string& gold_label,
                                const EmotionFusionAnalyzerOptions& options) {
    FeatureGradient grad;
    for (const auto& feature : features) {
        const double p = probabilities.contains(feature.label) ? probabilities.at(feature.label) : 0.0;
        const double y = feature.label == gold_label ? 1.0 : 0.0;
        const double delta = p - y;
        grad.head_bias += delta;
        grad.bert_signal_weight += delta * options.bert_weight * feature.bert_prob * feature.reliability;
        grad.keyword_signal_weight += delta * feature.keyword_score;
        grad.vector_signal_weight += delta * feature.vector_score;
        grad.llm_signal_weight += delta * feature.llm_score;
        grad.margin_signal_weight += delta * feature.margin_bonus;
    }
    return grad;
}

PrepareResult PrepareOneSample(const PrepareContext& context,
                               const TrainingSample& sample,
                               const EmotionAnalysis& primary,
                               std::vector<EmotionEvidence> precomputed_evidence,
                               bool skip_providers,
                               std::size_t index) {
    PrepareResult result;
    FusedEmotionAnalyzer fusion(nullptr, context.options);
    const std::string trace = "calib-" + std::to_string(index);

    std::vector<EmotionEvidence> evidence = std::move(precomputed_evidence);
    if (!context.bert_only && !skip_providers) {
        for (const auto& provider : context.providers) {
            auto collected = provider->Collect(sample.text, trace);
            if (collected.ok()) {
                evidence.insert(evidence.end(),
                                std::make_move_iterator(collected.value().begin()),
                                std::make_move_iterator(collected.value().end()));
            }
        }
    }

    auto features = fusion.BuildFeaturesForCalibration(primary, evidence);
    FilterFeaturesToLabels(features, context.labels);
    if (features.empty()) {
        result.failed = true;
        result.error = "no calibration labels after filtering";
        return result;
    }
    auto logits = fusion.BuildLogitsForCalibration(features, context.options);
    auto probabilities = fusion.SoftmaxForCalibration(logits);
    if (!context.bert_only && context.llm_analyzer && ShouldUseLlmFallback(probabilities, context.options)) {
        result.llm_called = true;
        auto llm = context.llm_analyzer->Analyze(sample.text, "calib-llm-" + std::to_string(index), nullptr);
        if (llm.ok()) {
            evidence.push_back(EmotionEvidence{
                llm.value().emotion.primary,
                llm.value().emotion.primary_prob > 0.0 ? llm.value().emotion.primary_prob : 0.6,
                "llm",
                {},
            });
            features = fusion.BuildFeaturesForCalibration(primary, evidence);
            FilterFeaturesToLabels(features, context.labels);
        } else {
            result.error = llm.status().message();
        }
    }

    result.sample = PreparedSample{sample.label, std::move(features), result.llm_called};
    return result;
}

PrepareResult PrepareOneSample(const PrepareContext& context,
                               const TrainingSample& sample,
                               const EmotionAnalysis& primary,
                               std::size_t index) {
    return PrepareOneSample(context, sample, primary, {}, false, index);
}

PrepareResult PrepareOneSample(const PrepareContext& context,
                               const TrainingSample& sample,
                               std::size_t index) {
    const std::string trace = "calib-" + std::to_string(index);
    auto primary = context.bert->Analyze(sample.text, trace, nullptr);
    if (!primary.ok()) {
        PrepareResult result;
        result.failed = true;
        result.error = primary.status().message();
        return result;
    }
    return PrepareOneSample(context, sample, primary.value(), {}, false, index);
}

EvalStats Evaluate(const std::vector<PreparedSample>& samples,
                   const EmotionFusionAnalyzerOptions& options,
                   const FusedEmotionAnalyzer& fusion) {
    EvalStats stats;
    if (samples.empty()) {
        return stats;
    }
    std::set<std::string> labels;
    int correct = 0;
    for (const auto& sample : samples) {
        labels.insert(sample.gold_label);
        auto logits = fusion.BuildLogitsForCalibration(sample.features, options);
        auto probs = fusion.SoftmaxForCalibration(logits);
        const double gold_prob = std::max(1e-12, probs.contains(sample.gold_label) ? probs.at(sample.gold_label) : 1e-12);
        stats.loss += -std::log(gold_prob);
        const auto pred = ArgMaxLabel(probs);
        labels.insert(pred);
        stats.confusion[sample.gold_label][pred] += 1;
        if (pred == sample.gold_label) {
            ++correct;
        }
    }
    stats.loss /= static_cast<double>(samples.size());
    stats.accuracy = static_cast<double>(correct) / static_cast<double>(samples.size());

    double f1_sum = 0.0;
    int f1_count = 0;
    for (const auto& label : labels) {
        int tp = stats.confusion[label][label];
        int fp = 0;
        int fn = 0;
        for (const auto& [gold, row] : stats.confusion) {
            for (const auto& [pred, count] : row) {
                if (pred == label && gold != label) {
                    fp += count;
                }
                if (gold == label && pred != label) {
                    fn += count;
                }
            }
        }
        const double precision = tp + fp == 0 ? 0.0 : static_cast<double>(tp) / static_cast<double>(tp + fp);
        const double recall = tp + fn == 0 ? 0.0 : static_cast<double>(tp) / static_cast<double>(tp + fn);
        const double f1 = precision + recall == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);
        f1_sum += f1;
        ++f1_count;
    }
    stats.macro_f1 = f1_count == 0 ? 0.0 : f1_sum / f1_count;
    return stats;
}

std::optional<EmotionFusionFeature> FindLlmFeature(const PreparedSample& sample) {
    const auto it = std::max_element(sample.features.begin(), sample.features.end(),
        [](const EmotionFusionFeature& lhs, const EmotionFusionFeature& rhs) {
            return lhs.llm_score < rhs.llm_score;
        });
    if (it == sample.features.end() || it->llm_score <= 0.0) {
        return std::nullopt;
    }
    return *it;
}

EvalStats EvaluateWithLlmGate(const std::vector<PreparedSample>& samples,
                              EmotionFusionAnalyzerOptions options,
                              const FusedEmotionAnalyzer& fusion,
                              double gate_confidence,
                              double gate_min_delta) {
    EvalStats stats;
    if (samples.empty()) {
        return stats;
    }

    options.llm_signal_weight = 0.0;
    std::set<std::string> labels;
    int correct = 0;
    for (const auto& sample : samples) {
        labels.insert(sample.gold_label);
        auto logits = fusion.BuildLogitsForCalibration(sample.features, options);
        auto probs = fusion.SoftmaxForCalibration(logits);
        std::string pred = ArgMaxLabel(probs);
        double pred_prob = probs.contains(pred) ? probs[pred] : 0.0;

        const auto llm = FindLlmFeature(sample);
        if (llm) {
            const double fused_same_prob = probs.contains(llm->label) ? probs[llm->label] : 0.0;
            if (llm->llm_score >= gate_confidence && llm->llm_score - fused_same_prob >= gate_min_delta) {
                pred = llm->label;
                pred_prob = llm->llm_score;
                probs[pred] = std::max(probs[pred], pred_prob);
                double sum = 0.0;
                for (const auto& [_, value] : probs) {
                    sum += std::max(0.0, value);
                }
                if (sum > 0.0) {
                    for (auto& [_, value] : probs) {
                        value = std::max(0.0, value) / sum;
                    }
                }
            }
        }

        const double gold_prob = std::max(1e-12, probs.contains(sample.gold_label) ? probs.at(sample.gold_label) : 1e-12);
        stats.loss += -std::log(gold_prob);
        labels.insert(pred);
        stats.confusion[sample.gold_label][pred] += 1;
        if (pred == sample.gold_label) {
            ++correct;
        }
    }
    stats.loss /= static_cast<double>(samples.size());
    stats.accuracy = static_cast<double>(correct) / static_cast<double>(samples.size());

    double f1_sum = 0.0;
    int f1_count = 0;
    for (const auto& label : labels) {
        int tp = stats.confusion[label][label];
        int fp = 0;
        int fn = 0;
        for (const auto& [gold, row] : stats.confusion) {
            for (const auto& [pred, count] : row) {
                if (pred == label && gold != label) {
                    fp += count;
                }
                if (gold == label && pred != label) {
                    fn += count;
                }
            }
        }
        const double precision = tp + fp == 0 ? 0.0 : static_cast<double>(tp) / static_cast<double>(tp + fp);
        const double recall = tp + fn == 0 ? 0.0 : static_cast<double>(tp) / static_cast<double>(tp + fn);
        const double f1 = precision + recall == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);
        f1_sum += f1;
        ++f1_count;
    }
    stats.macro_f1 = f1_count == 0 ? 0.0 : f1_sum / static_cast<double>(f1_count);
    return stats;
}

std::vector<LlmGateCandidate> ScanLlmGate(const std::vector<PreparedSample>& samples,
                                          const EmotionFusionAnalyzerOptions& options,
                                          const FusedEmotionAnalyzer& fusion) {
    std::vector<LlmGateCandidate> candidates;
    const std::vector<double> confidence_grid{0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90};
    const std::vector<double> delta_grid{-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20};
    for (const double confidence : confidence_grid) {
        for (const double min_delta : delta_grid) {
            std::size_t accepted = 0;
            EmotionFusionAnalyzerOptions no_llm = options;
            no_llm.llm_signal_weight = 0.0;
            for (const auto& sample : samples) {
                auto llm = FindLlmFeature(sample);
                if (!llm) {
                    continue;
                }
                auto logits = fusion.BuildLogitsForCalibration(sample.features, no_llm);
                auto probs = fusion.SoftmaxForCalibration(logits);
                const double fused_same_prob = probs.contains(llm->label) ? probs[llm->label] : 0.0;
                if (llm->llm_score >= confidence && llm->llm_score - fused_same_prob >= min_delta) {
                    ++accepted;
                }
            }
            candidates.push_back(LlmGateCandidate{
                confidence,
                min_delta,
                samples.empty() ? 0.0 : static_cast<double>(accepted) / static_cast<double>(samples.size()),
                EvaluateWithLlmGate(samples, options, fusion, confidence, min_delta),
            });
        }
    }
    std::sort(candidates.begin(), candidates.end(),
        [](const LlmGateCandidate& lhs, const LlmGateCandidate& rhs) {
            if (std::abs(lhs.stats.macro_f1 - rhs.stats.macro_f1) > 1e-12) {
                return lhs.stats.macro_f1 > rhs.stats.macro_f1;
            }
            if (std::abs(lhs.stats.accuracy - rhs.stats.accuracy) > 1e-12) {
                return lhs.stats.accuracy > rhs.stats.accuracy;
            }
            return lhs.call_rate < rhs.call_rate;
        });
    return candidates;
}

void AdamUpdate(double& value, double grad, double lr, int step, double& m, double& v) {
    constexpr double beta1 = 0.9;
    constexpr double beta2 = 0.999;
    constexpr double eps = 1e-8;
    m = beta1 * m + (1.0 - beta1) * grad;
    v = beta2 * v + (1.0 - beta2) * grad * grad;
    const double m_hat = m / (1.0 - std::pow(beta1, step));
    const double v_hat = v / (1.0 - std::pow(beta2, step));
    value -= lr * m_hat / (std::sqrt(v_hat) + eps);
}

void PrintUsage() {
    std::cerr
        << "Usage: emotion_fusion_map_calibrator --config <gateway.json> --dataset <smp.json|jsonl> --output <out.json> [options]\n"
        << "Options:\n"
        << "  --epochs N              default 200\n"
        << "  --lr VALUE              default 0.03\n"
        << "  --prior-lambda VALUE    default 0.02\n"
        << "  --max-samples N         default all\n"
        << "  --io-workers N          default from persona_gateway.io_pool.worker_count\n"
        << "  --batch-size N          batch BERT gRPC requests, default 1\n"
        << "  --labels a,b,c          restrict fusion/evaluation label space\n"
        << "  --no-llm                disable LLM evidence collection\n"
        << "  --no-vector             disable vector evidence collection\n"
        << "  --bert-only             evaluate BERT without keyword/vector/LLM evidence\n";
}

ToolOptions ParseArgs(int argc, char** argv) {
    ToolOptions options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](std::string_view name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + std::string(name));
            }
            return argv[++i];
        };
        if (arg == "--config") {
            options.config_path = require_value(arg);
        } else if (arg == "--dataset") {
            options.dataset_path = require_value(arg);
        } else if (arg == "--output") {
            options.output_path = require_value(arg);
        } else if (arg == "--epochs") {
            options.epochs = std::stoi(require_value(arg));
        } else if (arg == "--lr") {
            options.learning_rate = std::stod(require_value(arg));
        } else if (arg == "--prior-lambda") {
            options.prior_lambda = std::stod(require_value(arg));
        } else if (arg == "--max-samples") {
            options.max_samples = static_cast<std::size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--io-workers") {
            options.io_workers = static_cast<std::size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--batch-size") {
            options.batch_size = static_cast<std::size_t>(std::stoull(require_value(arg)));
        } else if (arg == "--labels") {
            options.labels = SplitLabels(require_value(arg));
        } else if (arg == "--no-llm") {
            options.enable_llm = false;
        } else if (arg == "--no-vector") {
            options.enable_vector = false;
        } else if (arg == "--bert-only") {
            options.bert_only = true;
            options.enable_llm = false;
            options.enable_vector = false;
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.config_path.empty() || options.dataset_path.empty()) {
        throw std::runtime_error("--config and --dataset are required");
    }
    options.config_path = fs::absolute(options.config_path).lexically_normal();
    options.dataset_path = fs::absolute(options.dataset_path).lexically_normal();
    options.output_path = fs::absolute(options.output_path).lexically_normal();
    options.batch_size = std::max<std::size_t>(1, options.batch_size);
    return options;
}

Json ToJson(const EmotionFusionAnalyzerOptions& options,
            const EvalStats& before,
            const EvalStats& after,
            std::size_t samples,
            std::size_t llm_calls,
            std::size_t batch_size,
            const std::set<std::string>& labels,
            std::string_view mode) {
    Json out;
    out["emotion_fusion"] = {
        {"head_bias", options.head_bias},
        {"bert_signal_weight", options.bert_signal_weight},
        {"keyword_signal_weight", options.keyword_signal_weight},
        {"vector_signal_weight", options.vector_signal_weight},
        {"llm_signal_weight", options.llm_signal_weight},
        {"margin_signal_weight", options.margin_signal_weight},
        {"llm_gate_confidence", options.llm_gate_confidence},
        {"llm_gate_min_delta", options.llm_gate_min_delta},
    };
    out["calibration"] = {
        {"mode", std::string(mode)},
        {"batch_size", batch_size},
        {"labels", labels.empty() ? Json(nullptr) : Json(labels)},
        {"samples", samples},
        {"llm_calls", llm_calls},
        {"llm_call_rate", samples == 0 ? 0.0 : static_cast<double>(llm_calls) / static_cast<double>(samples)},
        {"before_loss", before.loss},
        {"before_accuracy", before.accuracy},
        {"before_macro_f1", before.macro_f1},
        {"after_loss", after.loss},
        {"after_accuracy", after.accuracy},
        {"after_macro_f1", after.macro_f1},
    };
    return out;
}

Json LlmGateScanToJson(const std::vector<LlmGateCandidate>& candidates) {
    Json out = Json::array();
    const std::size_t count = std::min<std::size_t>(10, candidates.size());
    for (std::size_t i = 0; i < count; ++i) {
        const auto& candidate = candidates[i];
        out.push_back({
            {"llm_gate_confidence", candidate.confidence},
            {"llm_gate_min_delta", candidate.min_delta},
            {"llm_accept_rate", candidate.call_rate},
            {"loss", candidate.stats.loss},
            {"accuracy", candidate.stats.accuracy},
            {"macro_f1", candidate.stats.macro_f1},
        });
    }
    return out;
}

Json BuildLlmFallbackStatsByLabel(const std::vector<PreparedSample>& samples) {
    struct LabelStats {
        std::size_t samples = 0;
        std::size_t llm_calls = 0;
    };
    std::map<std::string, LabelStats> by_label;
    for (const auto& sample : samples) {
        auto& stats = by_label[sample.gold_label];
        ++stats.samples;
        if (sample.llm_called) {
            ++stats.llm_calls;
        }
    }
    Json out = Json::object();
    for (const auto& [label, stats] : by_label) {
        out[label] = {
            {"samples", stats.samples},
            {"llm_calls", stats.llm_calls},
            {"llm_call_rate", stats.samples == 0
                                  ? 0.0
                                  : static_cast<double>(stats.llm_calls) / static_cast<double>(stats.samples)},
        };
    }
    return out;
}

void WarmupBertAnalyzer(const std::shared_ptr<agent::service::persona::IEmotionAnalyzer>& bert) {
    if (!bert) {
        throw std::runtime_error("BERT analyzer is not configured");
    }
    constexpr int max_attempts = 3;
    for (int attempt = 1; attempt <= max_attempts; ++attempt) {
        std::cout << "[calibrator] warming up BERT analyzer attempt=" << attempt << "/" << max_attempts << "\n";
        const auto warmup = bert->Analyze("我有点紧张，但也很好奇这一步为什么这样做。",
                                          "calib-bert-warmup-" + std::to_string(attempt),
                                          nullptr);
        if (warmup.ok()) {
            std::cout << "[calibrator] BERT warmup primary=" << warmup.value().emotion.primary
                      << " prob=" << std::fixed << std::setprecision(4) << warmup.value().emotion.primary_prob << "\n";
            return;
        }
        if (attempt == max_attempts) {
            throw std::runtime_error("BERT warmup failed: " + warmup.status().message());
        }
        std::cout << "[calibrator] BERT warmup retry after error: " << warmup.status().message() << "\n";
        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto args = ParseArgs(argc, argv);
        const auto root = Json::parse(ReadTextFile(args.config_path));
        const auto fusion_json = root.value("emotion_fusion", Json::object());
        auto options = ParseFusionOptions(fusion_json);
        const auto prior = options;

        auto samples = LoadDataset(args.dataset_path, args.max_samples);
        if (!args.labels.empty()) {
            samples.erase(std::remove_if(samples.begin(), samples.end(),
                              [&](const TrainingSample& sample) {
                                  return !args.labels.contains(sample.label);
                              }),
                          samples.end());
        }
        if (samples.empty()) {
            throw std::runtime_error("dataset has no usable samples");
        }

        auto bert = CreateBertAnalyzer(args.config_path, root);
        WarmupBertAnalyzer(bert);
        auto providers = args.bert_only
                             ? std::vector<std::shared_ptr<agent::service::persona::IEmotionEvidenceProvider>>{}
                             : CreateEvidenceProviders(args.config_path, root, args.enable_vector);
        auto llm_client = args.enable_llm && !args.bert_only ? CreateLlmClient(args.config_path, root) : nullptr;
        std::shared_ptr<agent::service::persona::IEmotionAnalyzer> llm_analyzer;
        if (llm_client) {
            agent::service::persona::LlmEmotionFallbackOptions llm_options;
            llm_options.model = root.value("llm", Json::object()).value("model", std::string{});
            llm_analyzer = std::make_shared<agent::service::persona::LlmEmotionFallbackAnalyzer>(
                std::move(llm_client),
                llm_options);
        }

        std::size_t io_workers = args.io_workers;
        if (io_workers == 0) {
            const auto gateway = root.value("persona_gateway", Json::object());
            const auto io_pool_json = gateway.value("io_pool", Json::object());
            io_workers = static_cast<std::size_t>(GetInt(io_pool_json, "worker_count", 2));
        }
        io_workers = std::max<std::size_t>(1, io_workers);
        core::ThreadPool io_pool({io_workers, std::max<std::size_t>(samples.size(), 64), "emotion-calibrator-io"});
        auto pool_status = io_pool.Start();
        if (!pool_status.ok()) {
            throw std::runtime_error("failed to start calibrator io pool: " + pool_status.message());
        }

        PrepareContext prepare_context{
            bert,
            providers,
            llm_analyzer,
            options,
            args.labels,
            args.bert_only,
        };
        std::vector<PreparedSample> prepared;
        prepared.reserve(samples.size());
        std::atomic_size_t failed{0};
        std::atomic_size_t llm_calls{0};
        std::atomic_size_t completed{0};
        std::mutex futures_mutex;
        std::vector<std::future<PrepareResult>> futures;
        futures.reserve(samples.size());
        auto submit_prepare = [&](std::size_t i,
                                  std::optional<EmotionAnalysis> primary,
                                  std::optional<std::vector<EmotionEvidence>> evidence) {
            auto promise = std::make_shared<std::promise<PrepareResult>>();
            futures.push_back(promise->get_future());
            auto status = io_pool.Submit(
                [&, i, primary = std::move(primary), evidence = std::move(evidence), promise](
                    core::ThreadPoolContext&) mutable -> core::Status {
                    PrepareResult result;
                    try {
                        if (primary && evidence) {
                            result = PrepareOneSample(
                                prepare_context, samples[i], *primary, std::move(*evidence), true, i);
                        } else if (primary) {
                            result = PrepareOneSample(prepare_context, samples[i], *primary, i);
                        } else {
                            result = PrepareOneSample(prepare_context, samples[i], i);
                        }
                    } catch (const std::exception& e) {
                        result.failed = true;
                        result.error = e.what();
                    } catch (...) {
                        result.failed = true;
                        result.error = "unknown prepare task exception";
                    }
                    if (result.failed) {
                        ++failed;
                    }
                    if (result.llm_called) {
                        ++llm_calls;
                    }
                    const auto done = ++completed;
                    if (done % 50 == 0 || done == samples.size()) {
                        std::cout << "[calibrator] prepared " << done << "/" << samples.size()
                                  << " failed=" << failed.load()
                                  << " llm_calls=" << llm_calls.load() << "\n";
                    }
                    promise->set_value(std::move(result));
                    return core::Status::Ok();
                },
                {},
                "prepare-emotion-sample");
            if (!status.ok()) {
                throw std::runtime_error("failed to submit prepare task: " + status.message());
            }
        };

        if (args.batch_size <= 1) {
            for (std::size_t i = 0; i < samples.size(); ++i) {
                submit_prepare(i, std::nullopt, std::nullopt);
            }
        } else {
            auto grpc_bert = std::dynamic_pointer_cast<agent::service::persona::GrpcEmotionAnalyzer>(bert);
            if (!grpc_bert) {
                throw std::runtime_error("--batch-size requires GrpcEmotionAnalyzer");
            }
            for (std::size_t offset = 0; offset < samples.size(); offset += args.batch_size) {
                const auto count = std::min(args.batch_size, samples.size() - offset);
                std::vector<std::string_view> texts;
                texts.reserve(count);
                for (std::size_t j = 0; j < count; ++j) {
                    texts.push_back(samples[offset + j].text);
                }
                auto batch = grpc_bert->AnalyzeBatch(texts, "calib-batch-" + std::to_string(offset), nullptr);
                if (!batch.ok()) {
                    std::cerr << "[calibrator] BERT batch failed offset=" << offset
                              << " count=" << count
                              << " error=" << batch.status().message() << "\n";
                    for (std::size_t j = 0; j < count; ++j) {
                        auto promise = std::make_shared<std::promise<PrepareResult>>();
                        futures.push_back(promise->get_future());
                        PrepareResult result;
                        result.failed = true;
                        result.error = batch.status().message();
                        ++failed;
                        const auto done = ++completed;
                        if (done % 50 == 0 || done == samples.size()) {
                            std::cout << "[calibrator] prepared " << done << "/" << samples.size()
                                      << " failed=" << failed.load()
                                      << " llm_calls=" << llm_calls.load() << "\n";
                        }
                        promise->set_value(std::move(result));
                    }
                    continue;
                }
                if (batch.value().size() != count) {
                    throw std::runtime_error("BERT batch response size mismatch");
                }
                std::vector<std::vector<EmotionEvidence>> batch_evidence(count);
                if (!args.bert_only) {
                    for (const auto& provider : providers) {
                        auto collected = provider->CollectBatch(texts, "calib-evidence-batch-" + std::to_string(offset));
                        if (!collected.ok()) {
                            std::cerr << "[calibrator] evidence batch skipped offset=" << offset
                                      << " count=" << count
                                      << " error=" << collected.status().message() << "\n";
                            continue;
                        }
                        if (collected.value().size() != count) {
                            std::cerr << "[calibrator] evidence batch skipped offset=" << offset
                                      << " count=" << count
                                      << " error=response size mismatch\n";
                            continue;
                        }
                        for (std::size_t j = 0; j < count; ++j) {
                            auto& target = batch_evidence[j];
                            auto& source = collected.value()[j];
                            target.insert(target.end(),
                                          std::make_move_iterator(source.begin()),
                                          std::make_move_iterator(source.end()));
                        }
                    }
                }
                for (std::size_t j = 0; j < count; ++j) {
                    submit_prepare(offset + j, std::move(batch.value()[j]), std::move(batch_evidence[j]));
                }
            }
        }

        for (auto& future : futures) {
            auto result = future.get();
            if (result.sample) {
                prepared.push_back(std::move(*result.sample));
            }
        }
        io_pool.Shutdown(true);
        if (prepared.empty()) {
            throw std::runtime_error("no samples prepared");
        }

        FusedEmotionAnalyzer fusion(nullptr, options);
        const auto before = Evaluate(prepared, options, fusion);
        double m_bias = 0.0, v_bias = 0.0;
        double m_bert = 0.0, v_bert = 0.0;
        double m_keyword = 0.0, v_keyword = 0.0;
        double m_vector = 0.0, v_vector = 0.0;
        double m_llm = 0.0, v_llm = 0.0;
        double m_margin = 0.0, v_margin = 0.0;
        for (int epoch = 1; epoch <= args.epochs; ++epoch) {
            FeatureGradient grad;
            double loss = 0.0;
            for (const auto& sample : prepared) {
                auto logits = fusion.BuildLogitsForCalibration(sample.features, options);
                auto probs = fusion.SoftmaxForCalibration(logits);
                const double gold_prob = std::max(1e-12, probs.contains(sample.gold_label) ? probs.at(sample.gold_label) : 1e-12);
                loss += -std::log(gold_prob);
                const auto sample_grad = ComputeGradient(sample.features, probs, sample.gold_label, options);
                grad.head_bias += sample_grad.head_bias;
                grad.bert_signal_weight += sample_grad.bert_signal_weight;
                grad.keyword_signal_weight += sample_grad.keyword_signal_weight;
                grad.vector_signal_weight += sample_grad.vector_signal_weight;
                grad.llm_signal_weight += sample_grad.llm_signal_weight;
                grad.margin_signal_weight += sample_grad.margin_signal_weight;
            }
            const double inv_n = 1.0 / static_cast<double>(prepared.size());
            grad.head_bias = grad.head_bias * inv_n + 2.0 * args.prior_lambda * (options.head_bias - prior.head_bias);
            grad.bert_signal_weight = grad.bert_signal_weight * inv_n + 2.0 * args.prior_lambda * (options.bert_signal_weight - prior.bert_signal_weight);
            grad.keyword_signal_weight = grad.keyword_signal_weight * inv_n + 2.0 * args.prior_lambda * (options.keyword_signal_weight - prior.keyword_signal_weight);
            grad.vector_signal_weight = grad.vector_signal_weight * inv_n + 2.0 * args.prior_lambda * (options.vector_signal_weight - prior.vector_signal_weight);
            grad.margin_signal_weight = grad.margin_signal_weight * inv_n + 2.0 * args.prior_lambda * (options.margin_signal_weight - prior.margin_signal_weight);

            AdamUpdate(options.head_bias, grad.head_bias, args.learning_rate, epoch, m_bias, v_bias);
            AdamUpdate(options.bert_signal_weight, grad.bert_signal_weight, args.learning_rate, epoch, m_bert, v_bert);
            AdamUpdate(options.keyword_signal_weight, grad.keyword_signal_weight, args.learning_rate, epoch, m_keyword, v_keyword);
            AdamUpdate(options.vector_signal_weight, grad.vector_signal_weight, args.learning_rate, epoch, m_vector, v_vector);
            AdamUpdate(options.margin_signal_weight, grad.margin_signal_weight, args.learning_rate, epoch, m_margin, v_margin);

            options.bert_signal_weight = Clamp(options.bert_signal_weight, -10.0, 10.0);
            options.keyword_signal_weight = Clamp(options.keyword_signal_weight, -10.0, 10.0);
            options.vector_signal_weight = Clamp(options.vector_signal_weight, -10.0, 10.0);
            options.margin_signal_weight = Clamp(options.margin_signal_weight, -10.0, 10.0);
            if (epoch % 25 == 0 || epoch == 1 || epoch == args.epochs) {
                const auto stats = Evaluate(prepared, options, fusion);
                std::cout << "[calibrator] epoch=" << epoch
                          << " loss=" << std::fixed << std::setprecision(4) << stats.loss
                          << " acc=" << std::setprecision(2) << stats.accuracy * 100.0 << "%"
                          << " macro_f1=" << stats.macro_f1 * 100.0 << "%\n";
            }
            static_cast<void>(loss);
        }
        const auto after = Evaluate(prepared, options, fusion);
        fs::create_directories(args.output_path.parent_path());
        std::ofstream out(args.output_path, std::ios::binary);
        const std::string mode = args.bert_only ? "bert_only" : (args.enable_llm ? "full_fusion" : "no_llm_fusion");
        auto report = ToJson(options, before, after, prepared.size(), llm_calls.load(), args.batch_size, args.labels, mode);
        report["llm_fallback_by_label"] = BuildLlmFallbackStatsByLabel(prepared);
        if (args.enable_llm && !args.bert_only) {
            auto gate_candidates = ScanLlmGate(prepared, options, fusion);
            if (!gate_candidates.empty()) {
                const auto& best = gate_candidates.front();
                report["llm_gate_best"] = {
                    {"llm_gate_confidence", best.confidence},
                    {"llm_gate_min_delta", best.min_delta},
                    {"llm_accept_rate", best.call_rate},
                    {"loss", best.stats.loss},
                    {"accuracy", best.stats.accuracy},
                    {"macro_f1", best.stats.macro_f1},
                };
                report["llm_gate_top_candidates"] = LlmGateScanToJson(gate_candidates);
            }
        }
        out << report.dump(2) << "\n";
        std::cout << "[calibrator] wrote " << args.output_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[calibrator] failed: " << e.what() << "\n";
        PrintUsage();
        return 1;
    }
}
