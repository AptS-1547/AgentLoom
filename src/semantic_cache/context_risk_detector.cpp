#include "context_risk_detector.h"

#include <algorithm>
#include <cctype>
#include <utility>

namespace agent::semantic_cache {

namespace {

std::string LowerAscii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

bool ContainsMarker(const std::string& lowered_text, const std::vector<std::string>& markers) {
    for (const auto& marker : markers) {
        if (!marker.empty() && lowered_text.find(LowerAscii(marker)) != std::string::npos) {
            return true;
        }
    }
    return false;
}

} // namespace

KeywordContextRiskDetector::KeywordContextRiskDetector(KeywordContextRiskDetectorOptions options)
    : options_(std::move(options)) {}

core::Result<ContextRiskAssessment> KeywordContextRiskDetector::Assess(const CacheLookupRequest& req) {
    ContextRiskAssessment assessment;
    if (req.scope != CacheScope::Global) {
        return assessment;
    }
    if (req.answer_type == AnswerType::Personalized) {
        assessment.blocks_global_cache = true;
        assessment.reason = "personalized answer";
        return assessment;
    }
    if (req.has_image_reference) {
        assessment.blocks_global_cache = true;
        assessment.reason = "image-local reference";
        return assessment;
    }

    const auto lowered = LowerAscii(req.text);
    if (!req.recent_turns.empty() && ContainsMarker(lowered, options_.markers)) {
        assessment.blocks_global_cache = true;
        assessment.reason = "recent-turn context marker";
        return assessment;
    }
    return assessment;
}

}  // namespace agent::semantic_cache
