#include "config_section.h"

namespace server_config {

void RegisterConfigPathSection();
void RegisterModelsConfigSection();
void RegisterGrpcConfigSection();
void RegisterAuthConfigSection();
void RegisterLimitsConfigSection();
void RegisterVramGuardConfigSection();
void RegisterVlmCacheConfigSection();

void RegisterBuiltinConfigSections() {
    RegisterConfigPathSection();
    RegisterModelsConfigSection();
    RegisterGrpcConfigSection();
    RegisterAuthConfigSection();
    RegisterLimitsConfigSection();
    RegisterVramGuardConfigSection();
    RegisterVlmCacheConfigSection();
}

} // namespace server_config
