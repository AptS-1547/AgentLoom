#include "config_section.h"

namespace server_config {
namespace {

DECLARE_CONFIG_SECTION(ConfigPathSection, "config")
};

void ConfigPathSection::LoadJson(const Json&, MultimodalServerOptions&) const {}

bool ConfigPathSection::LoadCli(CliCursor& cursor, MultimodalServerOptions&) const {
    CliArgumentParser parser(cursor);
    CONFIG_VALUE_ARG(kConfigPathFlag, value, (void)value;)
    return false;
}

} // namespace

REGISTER_CONFIG_SECTION(ConfigPathSection)

} // namespace server_config
