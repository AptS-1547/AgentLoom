#include <AgentLoom/config/config_section.h>

namespace server_config {
namespace {

DECLARE_CONFIG_SECTION(SdkConsumerConfigSection, "sdk_consumer")
};

void SdkConsumerConfigSection::LoadJson(const Json& root, MultimodalServerOptions& options) const {
    const Json* section = FindSection(root, Name());
    if (section) {
        SetString(*section, Name(), "value", options.auth.token);
    }
}

bool SdkConsumerConfigSection::LoadCli(CliCursor&, MultimodalServerOptions&) const {
    return false;
}

REGISTER_CONFIG_SECTION(SdkConsumerConfigSection)

} // namespace
} // namespace server_config
