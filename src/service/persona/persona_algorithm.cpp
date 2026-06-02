#include "persona_algorithm.h"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>

namespace agent::service::persona {
namespace {

double Clamp(double value, double lo, double hi) {
    return std::max(lo, std::min(value, hi));
}

std::string Join(const std::vector<std::string>& values, std::string_view sep) {
    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out.append(sep);
        }
        out.append(values[i]);
    }
    return out;
}

double MapGet(const std::map<std::string, double>& map, const std::string& key, double fallback) {
    auto it = map.find(key);
    return it == map.end() ? fallback : it->second;
}

std::string StripLeadingUser(std::string value) {
    constexpr std::string_view prefix = "用户";
    if (value.rfind(prefix, 0) == 0) {
        value.erase(0, prefix.size());
    }
    return value;
}

} // namespace

EmotionStateTracker::EmotionStateTracker(EmotionStateConfig config,
                                         std::optional<EmotionState> initial_state,
                                         std::uint32_t random_seed)
    : config_(std::move(config)),
      state_(std::move(initial_state).value_or(EmotionState{})),
      rng_(random_seed) {}

const EmotionState& EmotionStateTracker::state() const noexcept {
    return state_;
}

const EmotionStateConfig& EmotionStateTracker::config() const noexcept {
    return config_;
}

const std::unordered_map<std::string, EmotionVa>& EmotionStateTracker::EmotionMap() {
    static const std::unordered_map<std::string, EmotionVa> map{
        {"neutral", {0.0, 0.1}},
        {"joy", {0.8, 0.6}},
        {"sadness", {-0.7, 0.2}},
        {"anger", {-0.4, 0.8}},
        {"fear", {-0.6, 0.7}},
        {"surprise", {0.1, 0.8}},
        {"disgust", {-0.5, 0.4}},
        {"excitement", {0.7, 0.9}},
        {"tenderness", {0.6, 0.3}},
        {"curiosity", {0.5, 0.6}},
    };
    return map;
}

const std::unordered_map<std::string, std::string>& EmotionStateTracker::StateHintMap() {
    static const std::unordered_map<std::string, std::string> map{
        {"joy", "最近聊得不错，可以自然地带一点轻松愉快"},
        {"sadness", "聊了一些沉重的话题，用陈述句回应，给予陪伴和理解就好"},
        {"anger", "对话里有些让你不太舒服的东西，直接表达感受，不展开追问"},
        {"fear", "话题让你有点不安，给予安全感，用肯定句而不是疑问句"},
        {"excitement", "对话让你很有活力，可以热情一些"},
        {"tenderness", "聊到了让你心软的内容，可以用更温柔的方式回应"},
        {"curiosity", "被这个话题激发了好奇，可以表现出想深入了解的兴趣"},
        {"disgust", "对话让你有点不舒服，直接说出感受，不追问原因"},
    };
    return map;
}

std::string EmotionStateTracker::StateToLabel(double valence, double arousal) {
    std::string best = "neutral";
    double best_dist = std::numeric_limits<double>::infinity();
    for (const auto& [label, va] : EmotionMap()) {
        const double dv = valence - va.valence;
        const double da = arousal - va.arousal;
        const double dist = dv * dv + da * da;
        if (dist < best_dist) {
            best_dist = dist;
            best = label;
        }
    }
    return best;
}

core::Result<EmotionVa> EmotionStateTracker::Project(std::string_view emotion, double intensity) const {
    if (!std::isfinite(intensity)) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "emotion intensity must be finite");
    }
    const auto key = std::string(emotion);
    auto it = EmotionMap().find(key);
    const EmotionVa base = it == EmotionMap().end() ? EmotionVa{} : it->second;
    return EmotionVa{base.valence * intensity, base.arousal * intensity};
}

core::Result<EmotionState> EmotionStateTracker::Update(std::string_view user_emotion,
                                                       double user_intensity,
                                                       std::string_view ai_emotion,
                                                       double ai_intensity) {
    auto user = Project(user_emotion, user_intensity);
    if (!user.ok()) {
        return user.status();
    }
    auto ai = Project(ai_emotion, ai_intensity);
    if (!ai.ok()) {
        return ai.status();
    }

    const auto user_va = std::move(user).value();
    const auto ai_va = std::move(ai).value();
    const double v = state_.valence;
    const double a = state_.arousal;

    std::normal_distribution<double> noise(0.0, config_.noise_sigma);
    const double eps_v = config_.noise_sigma == 0.0 ? 0.0 : noise(rng_);
    const double eps_a = config_.noise_sigma == 0.0 ? 0.0 : noise(rng_);

    const double phi = config_.alpha - config_.delta;
    const double theta = 1.0 - phi;
    double theta_v = theta;
    if (v < config_.baseline_valence && config_.negativity_bias > 1.0) {
        theta_v = theta / config_.negativity_bias;
    }
    const double phi_v = 1.0 - theta_v;

    state_.prev_valence = v;
    state_.valence = std::tanh(phi_v * v
        + theta_v * config_.baseline_valence
        + config_.kappa * (a - config_.baseline_arousal)
        + config_.beta * ai_va.valence
        + config_.gamma * user_va.valence
        + eps_v);
    state_.arousal = std::tanh(phi * a
        + theta * config_.baseline_arousal
        + config_.kappa * (v - config_.baseline_valence)
        + config_.beta * ai_va.arousal
        + config_.gamma * user_va.arousal
        + eps_a);
    ++state_.turn_count;
    UpdateLabelTrace();
    return state_;
}

core::Result<EmotionState> EmotionStateTracker::ApplyStimulus(double valence_delta,
                                                              double arousal_delta) {
    if (!std::isfinite(valence_delta) || !std::isfinite(arousal_delta)) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "stimulus deltas must be finite");
    }

    const double v = state_.valence;
    const double a = state_.arousal;
    const double phi = config_.alpha - config_.delta;
    const double theta = 1.0 - phi;
    double theta_v = theta;
    if (v < config_.baseline_valence && config_.negativity_bias > 1.0) {
        theta_v = theta / config_.negativity_bias;
    }
    const double phi_v = 1.0 - theta_v;

    state_.prev_valence = v;
    state_.valence = std::tanh(phi_v * v
        + theta_v * config_.baseline_valence
        + config_.kappa * (a - config_.baseline_arousal)
        + valence_delta);
    state_.arousal = std::tanh(phi * a
        + theta * config_.baseline_arousal
        + config_.kappa * (v - config_.baseline_valence)
        + arousal_delta);
    UpdateLabelTrace();
    return state_;
}

GenerationParams EmotionStateTracker::GetParamAdjustments(const GenerationParams& base) const {
    const double v = state_.valence;
    const double a = state_.arousal;

    double temp = base.temperature + 0.20 * a - 0.08 * std::max(0.0, -v);
    temp = Clamp(temp, base.temperature * 0.75, base.temperature * 1.35);

    int tokens = static_cast<int>(base.max_tokens * (1.0 + 0.25 * std::max(0.0, -v)));
    tokens = std::max(base.max_tokens, std::min(tokens, static_cast<int>(base.max_tokens * 1.25)));

    const double arousal_dev = std::abs(a - 0.3);
    const double top_p = Clamp(base.top_p + 0.08 * arousal_dev, 0.80, 0.99);

    return GenerationParams{std::round(temp * 10000.0) / 10000.0, tokens,
                            std::round(top_p * 10000.0) / 10000.0};
}

std::optional<std::string> EmotionStateTracker::GetPromptHint() const {
    const double dev_v = state_.valence - config_.baseline_valence;
    const double dev_a = state_.arousal - config_.baseline_arousal;
    const double deviation = std::sqrt(dev_v * dev_v + dev_a * dev_a);
    if (deviation < config_.injection_threshold) {
        return std::nullopt;
    }

    auto hint_it = StateHintMap().find(state_.last_emotion);
    if (hint_it == StateHintMap().end()) {
        return std::nullopt;
    }

    std::string hint = hint_it->second;
    if (deviation > 0.5) {
        hint += "（情绪比较强烈）";
    } else if (deviation <= 0.3) {
        hint += "（只是淡淡的）";
    }

    const double delta_v = state_.valence - state_.prev_valence;
    if (state_.turn_count > 1 && std::abs(delta_v) > 0.08) {
        if (delta_v > 0 && state_.valence < config_.baseline_valence) {
            hint += "，不过情绪正在慢慢好转";
        } else if (delta_v < 0 && state_.valence > config_.baseline_valence) {
            hint += "，但情绪有些往下走";
        } else if (delta_v > 0 && state_.valence >= config_.baseline_valence) {
            hint += "，而且越来越高兴";
        } else {
            hint += "，情绪还在继续低落";
        }
    }

    if (state_.sustained_turns >= 5) {
        hint += "。这种状态已经持续了一段时间";
    } else if (state_.sustained_turns == 1 && state_.turn_count > 1) {
        hint += "。刚刚情绪发生了转变";
    }
    return hint;
}

EmotionStateSnapshot EmotionStateTracker::Snapshot() const {
    return EmotionStateSnapshot{state_, config_};
}

void EmotionStateTracker::UpdateLabelTrace() {
    const std::string label = StateToLabel(state_.valence, state_.arousal);
    if (label == state_.sustained_label) {
        ++state_.sustained_turns;
    } else {
        state_.sustained_label = label;
        state_.sustained_turns = 1;
    }
    state_.last_emotion = label;
}

PromptBuilder::PromptBuilder(PersonalityConfig personality,
                             std::optional<EmotionPromptConfig> emotion_prompt_config,
                             bool time_awareness)
    : personality_(std::move(personality)),
      emotion_prompt_config_(std::move(emotion_prompt_config)),
      time_awareness_(time_awareness) {}

std::string PromptBuilder::PersonalitySection() const {
    if (!personality_.description.empty()) {
        return personality_.description;
    }
    const std::string formality = personality_.formality < 0.4
        ? "口语"
        : (personality_.formality < 0.7 ? "适中" : "正式");
    std::ostringstream oss;
    oss << "[标签]" << Join(personality_.traits, ",")
        << " [开放]" << std::fixed << std::setprecision(1) << personality_.openness
        << " [外向]" << personality_.extraversion
        << " [幽默]" << personality_.humor_tendency
        << " [共情]" << personality_.empathy_level
        << " [好奇]" << personality_.curiosity_level
        << " [风格]" << formality;
    return oss.str();
}

core::Result<std::string> PromptBuilder::BuildSystemPrompt(
    std::string_view recalled_context,
    const std::optional<EmotionAnalysis>& emotion_analysis,
    const std::optional<std::string>& emotion_state_hint) const {
    if (personality_.name.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "personality name is required");
    }

    std::string prompt = "你是" + personality_.name + "。\n<persona>" + PersonalitySection() + "</persona>";
    if (time_awareness_) {
        prompt += "\n<current_time>" + CurrentTimeText() + "</current_time>";
    }
    if (emotion_analysis && emotion_prompt_config_) {
        auto directives = BuildEmotionDirectives(*emotion_analysis);
        if (!directives.ok()) {
            return directives.status();
        }
        if (!directives.value().empty()) {
            prompt += "\n<mood>\n" + directives.value() + "\n</mood>";
        }
    }
    if (!recalled_context.empty()) {
        prompt += "\n";
        prompt.append(recalled_context);
    }
    if (emotion_state_hint && !emotion_state_hint->empty()) {
        prompt += "\n<feeling>" + *emotion_state_hint + "</feeling>";
    }
    prompt += "\n自然回复就好。";
    return prompt;
}

core::Result<std::vector<PromptBlock>> PromptBuilder::BuildSystemPromptBlocks(
    std::string_view recalled_context,
    const std::optional<EmotionAnalysis>& emotion_analysis,
    const std::optional<std::string>& emotion_state_hint) const {
    if (personality_.name.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "personality name is required");
    }

    std::vector<PromptBlock> blocks;
    std::string static_prompt = "你是" + personality_.name + "。\n<persona>" + PersonalitySection() + "</persona>";
    if (time_awareness_) {
        static_prompt += "\n<current_time>" + CurrentTimeText() + "</current_time>";
    }
    static_prompt += "\n自然回复就好。";
    blocks.push_back(PromptBlock{"text", std::move(static_prompt), true});

    if (!recalled_context.empty()) {
        blocks.push_back(PromptBlock{"text", std::string(recalled_context), true});
    }
    if (emotion_analysis && emotion_prompt_config_) {
        auto directives = BuildEmotionDirectives(*emotion_analysis);
        if (!directives.ok()) {
            return directives.status();
        }
        if (!directives.value().empty()) {
            blocks.push_back(PromptBlock{"text", "<mood>\n" + directives.value() + "\n</mood>", false});
        }
    }
    if (emotion_state_hint && !emotion_state_hint->empty()) {
        blocks.push_back(PromptBlock{"text", "<feeling>" + *emotion_state_hint + "</feeling>", false});
    }
    return blocks;
}

core::Result<std::string> PromptBuilder::BuildEmotionDirectives(const EmotionAnalysis& emotion_analysis) const {
    if (!emotion_prompt_config_) {
        return std::string{};
    }
    if (emotion_analysis.emotion.primary.empty()) {
        return core::Status::Error(core::ErrorCode::InvalidArgument, "primary emotion is required");
    }

    const auto& cfg = *emotion_prompt_config_;
    auto directive_it = cfg.emotion_map.find(emotion_analysis.emotion.primary);
    if (directive_it == cfg.emotion_map.end() || directive_it->second.empty()) {
        return std::string{};
    }

    const double reliability = MapGet(cfg.emotion_reliability, emotion_analysis.emotion.primary, 0.7);
    const double effective_confidence = emotion_analysis.emotion.primary_prob * reliability;
    const double strong = MapGet(cfg.confidence_thresholds, "strong", 0.5);
    const double weak = MapGet(cfg.confidence_thresholds, "weak", 0.3);
    const double high_min = MapGet(cfg.intensity_levels, "high_min", 0.7);

    if (effective_confidence >= strong) {
        if (emotion_analysis.emotion.intensity >= high_min) {
            return "（强烈）" + directive_it->second;
        }
        return directive_it->second;
    }
    if (effective_confidence >= weak) {
        return "好像" + StripLeadingUser(directive_it->second) + "，但也说不准";
    }
    return std::string{};
}

std::string PromptBuilder::CurrentTimeText() const {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#if defined(_WIN32)
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    std::ostringstream oss;
    oss << std::put_time(&local, "%Y年%m月%d日 %H:%M");
    return oss.str();
}

EmotionNeuronFusion::EmotionNeuronFusion(EmotionFusionConfig config,
                                         std::map<std::string, double> emotion_reliability)
    : config_(config), emotion_reliability_(std::move(emotion_reliability)) {
    emotion_labels_.reserve(emotion_reliability_.size());
    for (const auto& [label, _] : emotion_reliability_) {
        emotion_labels_.push_back(label);
    }
}

core::Result<EmotionAnalysis> EmotionNeuronFusion::Fuse(
    const EmotionAnalysis& bert_result,
    const std::optional<LlmEmotionResult>& llm_result) const {
    if (!llm_result || llm_result->emotion.empty()) {
        return bert_result;
    }
    if (emotion_labels_.empty()) {
        return core::Status::Error(core::ErrorCode::FailedPrecondition, "emotion labels are empty");
    }

    std::map<std::string, double> z_scores;
    for (const auto& label : emotion_labels_) {
        const double bert_prob = MapGet(bert_result.emotion.probabilities, label,
                                        label == bert_result.emotion.primary ? bert_result.emotion.primary_prob : 0.0);
        const double x_bert = bert_prob * MapGet(emotion_reliability_, label, 0.7);
        const double x_llm = label == llm_result->emotion
            ? (llm_result->confidence > 0.0 ? llm_result->confidence : config_.llm_confidence_default)
            : 0.0;
        z_scores[label] = config_.w_bert * x_bert + config_.w_llm * x_llm + config_.bias;
    }

    auto probs = Softmax(z_scores);
    auto best = std::max_element(probs.begin(), probs.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
    if (best == probs.end()) {
        return core::Status::Error(core::ErrorCode::InternalError, "fusion produced no scores");
    }

    EmotionAnalysis result = bert_result;
    result.emotion.primary = best->first;
    result.emotion.primary_prob = best->second;
    result.emotion.probabilities = std::move(probs);
    result.emotion.intensity = Clamp(best->second, 0.0, 1.0);
    return result;
}

std::map<std::string, double> EmotionNeuronFusion::Softmax(const std::map<std::string, double>& scores) const {
    std::map<std::string, double> out;
    if (scores.empty()) {
        return out;
    }
    const auto max_it = std::max_element(scores.begin(), scores.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
    const double max_score = max_it->second;
    double sum = 0.0;
    for (const auto& [label, score] : scores) {
        const double value = std::exp(score - max_score);
        out[label] = value;
        sum += value;
    }
    if (sum <= 0.0) {
        return out;
    }
    for (auto& [_, value] : out) {
        value /= sum;
    }
    return out;
}

} // namespace agent::service::persona
