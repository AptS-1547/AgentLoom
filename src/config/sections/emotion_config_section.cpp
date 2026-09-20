#include "config_section.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace server_config {
namespace {

DECLARE_CONFIG_SECTION(EmotionConfigSection, "emotion")
    void Validate(MultimodalServerOptions& options) const override;
};

void SetFiniteDouble(const Json& section,
                     std::string_view section_name,
                     std::string_view field_name,
                     double& target,
                     double minimum,
                     double maximum) {
    const Json* value = FindField(section, section_name, field_name);
    if (!value) {
        return;
    }
    if (!value->is_number()) {
        throw std::runtime_error(std::string(section_name) + "." +
                                 std::string(field_name) + " must be a number");
    }
    const double parsed = value->get<double>();
    if (!std::isfinite(parsed) || parsed < minimum || parsed > maximum) {
        throw std::runtime_error(std::string(section_name) + "." +
                                 std::string(field_name) + " is out of range");
    }
    target = parsed;
}

void EmotionConfigSection::LoadJson(const Json& root, MultimodalServerOptions& options) const {
    const Json* emotion = FindSection(root, Name());
    if (!emotion) {
        return;
    }
    const Json* generation = FindField(*emotion, Name(), "generation");
    if (!generation) {
        return;
    }
    if (!generation->is_object()) {
        throw std::runtime_error("emotion.generation must be an object");
    }

    auto& target = options.emotion_generation;
    SetInt(*generation, "emotion.generation", "max_tokens", target.max_tokens, 1, 1048576);
    SetInt(*generation, "emotion.generation", "min_tokens", target.min_tokens, 1, 1048576);
    SetFiniteDouble(*generation, "emotion.generation", "max_token_ratio",
                    target.max_token_ratio, 1.0, 4.0);
    SetFiniteDouble(*generation, "emotion.generation", "default_token_weight",
                    target.default_token_weight, 0.01, 4.0);
    SetFiniteDouble(*generation, "emotion.generation", "high_intensity_threshold",
                    target.high_intensity_threshold, 0.0, 1.0);
    SetFiniteDouble(*generation, "emotion.generation", "high_intensity_multiplier",
                    target.high_intensity_multiplier, 0.01, 4.0);

    if (const Json* weights = FindField(*generation, "emotion.generation", "token_weights")) {
        if (!weights->is_object()) {
            throw std::runtime_error("emotion.generation.token_weights must be an object");
        }
        target.token_weights.clear();
        for (const auto& [label, value] : weights->items()) {
            if (label.empty() || !value.is_number()) {
                throw std::runtime_error(
                    "emotion.generation.token_weights requires non-empty numeric entries");
            }
            const double weight = value.get<double>();
            if (!std::isfinite(weight) || weight < 0.01 || weight > 4.0) {
                throw std::runtime_error(
                    "emotion.generation.token_weights values must be in [0.01, 4.0]");
            }
            target.token_weights.emplace(label, weight);
        }
    }
}

bool EmotionConfigSection::LoadCli(CliCursor&, MultimodalServerOptions&) const {
    return false;
}

void EmotionConfigSection::Validate(MultimodalServerOptions& options) const {
    const auto& generation = options.emotion_generation;
    const auto maximum = static_cast<double>(generation.max_tokens) * generation.max_token_ratio;
    if (static_cast<double>(generation.min_tokens) > maximum) {
        throw std::runtime_error(
            "emotion.generation.min_tokens must not exceed max_tokens * max_token_ratio");
    }
}

}

REGISTER_CONFIG_SECTION(EmotionConfigSection)

}
