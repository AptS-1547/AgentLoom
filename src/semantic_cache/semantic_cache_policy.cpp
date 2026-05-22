#include "semantic_cache_policy.h"

namespace agent::semantic_cache {

// TODO(orange): implement DefaultPolicyMatcher.
//
// Suggested check order (fail closed at any step):
//   1. fingerprint exact match (tokenizer / embedding / corpus / policy / dim)
//   2. scope compatibility — global request never reads user/session entries
//   3. tenant equality when scope is Tenant/User/Session
//   4. subject / grade / topic / persona equality
//   5. answer_type compatibility
//   6. quality_score >= configurable floor
//   7. expires_at_ms in the future
//
// Constructor should take a config struct with the quality floor and any
// answer_type → answer_type compatibility table.

}  // namespace agent::semantic_cache
