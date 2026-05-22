#include "context_risk_detector.h"

namespace agent::semantic_cache {

// TODO(orange): implement KeywordContextRiskDetector.
//
// First-pass design ideas:
//   - configurable list of Chinese / English markers
//     ("这个/那个/上面/刚才/前面/上一步/this/that/above/previous step")
//   - any non-empty recent_turns + a marker → block
//   - has_image_reference + image-deictic markers → block
//   - answer_type == Personalized → always block global
//
// Constructor should accept a config struct so tests can override the
// marker list without recompiling.

}  // namespace agent::semantic_cache
