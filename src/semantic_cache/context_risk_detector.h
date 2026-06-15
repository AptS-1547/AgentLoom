#pragma once

#include "semantic_cache_types.h"
#include "result.h"

#include <string>
#include <vector>

namespace agent::semantic_cache {

/// Result of a context-risk pre-check.
struct ContextRiskAssessment {
    /// If true, the request is too context-dependent for global semantic reuse.
    bool blocks_global_cache = false;
    /// Optional human-readable reason for logging.
    std::string reason;
};

/// Decides whether a request is safe to look up in a shared semantic cache.
///
/// Strong context markers (this/that/above/previous step, image-local
/// references, recent-turn dependencies) should block the global path.
/// Implementations may evolve from keyword-based heuristics to small
/// classifier models.
class IContextRiskDetector {
public:
    virtual ~IContextRiskDetector() = default;

    virtual core::Result<ContextRiskAssessment> Assess(const CacheLookupRequest& req) = 0;
};

struct KeywordContextRiskDetectorOptions {
    std::vector<std::string> markers = {
        "这个", "那个", "这些", "那些", "上面", "下面", "前面", "后面", "刚才", "刚刚",
        "上一", "上一步", "前一步", "这张图", "图片里", "图中",
        "this", "that", "these", "those", "above", "previous", "last step", "the image"
    };
};

class KeywordContextRiskDetector final : public IContextRiskDetector {
public:
    explicit KeywordContextRiskDetector(KeywordContextRiskDetectorOptions options = {});

    core::Result<ContextRiskAssessment> Assess(const CacheLookupRequest& req) override;

private:
    KeywordContextRiskDetectorOptions options_;
};

}  // namespace agent::semantic_cache
