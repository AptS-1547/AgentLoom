#pragma once

#include "config_section.h"

#include <filesystem>
#include <optional>

namespace server_config {

std::optional<std::filesystem::path> FindConfigPath(int argc, char** argv);
void LoadConfigFile(const std::filesystem::path& path, MultimodalServerOptions& options);
void LoadConfigFile(const std::filesystem::path& path,
                    MultimodalServerOptions& options,
                    const ConfigSectionSelection& selection);
void ApplyCliFallbackOptions(int argc, char** argv, MultimodalServerOptions& options);
void ApplyCliFallbackOptions(int argc,
                             char** argv,
                             MultimodalServerOptions& options,
                             const ConfigSectionSelection& selection);

}
